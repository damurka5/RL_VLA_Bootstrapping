#!/usr/bin/env python3
"""Record continuous three-stage ``put_into`` chains. No optimizer, ever.

This is the collection half of ``CDPR_THREE_STAGE_PUT_INTO_SFT_DESIGN.md``. It
runs the move-to, pickup and placement teachers against ONE live scene per
world, hands off between them without a reset or a pose write, and writes
durable per-round shards plus the pictures every accepted chain needs.

It trains nothing. A tool that could collect and train in one invocation is a
tool that will one day train on a bank whose acceptance nobody looked at.

What comes out
--------------

``staged_<tag>_r<N>.npz``   every executed action, every pre-decision
                            observation, the three predicates' verdicts per env
                            step, and the stage machine's events per world.
``frames_<tag>_r<N>.npz``   overview and wrist pictures for accepted worlds and
                            successful pickup prefixes, keyed by explicit
                            ``episode_uid``, plus the terminal post-action frame.
``collection.json``         teacher manifest with hashes, scene manifest hash,
                            budgets, yaw calibration, per-round and pooled
                            yields, and the rejection census.

Frames are the memory bound, and the arithmetic belongs in the plan rather than
in an OOM: two cameras at 240x320x3 cost 460.8 kB per world per decision, so a
64-world round over the design's 128-decision budget holds 3.8 GB before it is
written. ``--worlds`` is the knob.

Usage (remote, one GPU per shard)::

    RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run -n cdpr-mjlab python \\
      tools/audit/record_cdpr_staged_put_into.py \\
        --config configs/examples/cdpr_smolvla_three_stage_put_into.yaml \\
        --scene-manifest runs/three_stage/scenes.json --split collection \\
        --teacher move_to=runs/.../step_3416645/smolvla_grpo_adapter.pt \\
        --teacher pick_up=runs/.../step_3540208/smolvla_grpo_adapter.pt \\
        --teacher placement=runs/.../step_2117145/smolvla_grpo_adapter.pt \\
        --yaw-calibration runs/three_stage/yaw_calibration.json \\
        --worlds 64 --rounds 8 --output runs/three_stage/bank_shard0
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Before anything can reach huggingface_hub: the probe sets both offline
# switches at import time and they are read into module constants on the hub's
# first import, so setting them afterwards is silently too late.
from tools.audit.xy_approach_probe import _build_world  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
from dataclasses import replace  # noqa: E402
import time  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (  # noqa: E402
    FAILURE_NAMES,
    TEACHER_ROLES,
    PickupReadiness,
    PickupYawCalibration,
    StageBudgets,
    StagedRolloutConfig,
    TeacherBank,
    load_teacher_entries,
    run_staged_chains,
    write_staged_frames,
)
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (  # noqa: E402
    FullTaskSceneResetter,
)
from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import (  # noqa: E402
    SPLIT_NAMES,
    read_manifest,
    select_split,
)


def parse_teachers(specs: Sequence[str]) -> dict[str, Path]:
    roles: dict[str, Path] = {}
    for spec in specs:
        role, _, path = str(spec).partition("=")
        if role not in TEACHER_ROLES or not path:
            raise SystemExit(
                f"--teacher expects ROLE=PATH with ROLE in {list(TEACHER_ROLES)}, "
                f"got {spec!r}"
            )
        roles[role] = Path(path)
    missing = [role for role in TEACHER_ROLES if role not in roles]
    if missing:
        raise SystemExit(f"--teacher is missing {missing}.")
    return roles


def plan_batches(
    scenes: Sequence[Any],
    *,
    worlds: int,
    rounds: int,
    repeats_per_scene: int,
    shard: int,
    num_shards: int,
) -> list[list[tuple[Any, int]]]:
    """Which scenes each round runs, and at which rollout index.

    Scenes are sharded BEFORE they are batched, so two GPUs never collect the
    same scene: a shared scene between shards would produce two episodes with
    the same ``scene_uid``, which the split treats as one unit and the
    uncertainty analysis treats as one cluster -- so they would look like two
    independent samples and count twice.

    A repeat is an explicit rollout index, not an accident. Repeating a scene is
    legitimate here because the SmolVLA prior draws fresh noise on every
    forward, so the same start under the same teacher is genuinely a different
    rollout -- but the repeats share a scene and therefore a split, and they
    must never be counted as distinct data.
    """

    if num_shards < 1 or not 0 <= shard < num_shards:
        raise SystemExit(f"Invalid shard {shard}/{num_shards}.")
    mine = [scene for index, scene in enumerate(scenes) if index % num_shards == shard]
    pool: list[tuple[Any, int]] = [
        (scene, repeat)
        for repeat in range(max(1, int(repeats_per_scene)))
        for scene in mine
    ]
    needed = int(worlds) * int(rounds)
    if len(pool) < needed:
        raise SystemExit(
            f"Shard {shard} holds {len(mine)} scenes x {repeats_per_scene} "
            f"repeats = {len(pool)} chains, but --worlds {worlds} x --rounds "
            f"{rounds} asks for {needed}. Generate more scenes, lower the "
            "budget, or raise --repeats-per-scene deliberately -- repeats are "
            "not new data and are reported separately."
        )
    return [
        pool[index * worlds : (index + 1) * worlds] for index in range(int(rounds))
    ]


def pooled_summary(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum the per-round censuses without averaging away a bad round."""

    total: dict[str, Any] = {
        "rounds": len(rounds),
        "worlds": sum(int(row["worlds"]) for row in rounds),
        "reached": sum(int(row["reached"]) for row in rounds),
        "aligned": sum(int(row["aligned"]) for row in rounds),
        "picked_up": sum(int(row["picked_up"]) for row in rounds),
        "native_placement_success": sum(
            int(row["native_placement_success"]) for row in rounds
        ),
        "accepted_chains": sum(int(row["accepted_chains"]) for row in rounds),
        "diverged_worlds": sum(int(row["diverged_worlds"]) for row in rounds),
    }
    for key in ("rejection_reasons", "failure_counts"):
        merged: dict[str, int] = {}
        for row in rounds:
            for name, count in dict(row.get(key) or {}).items():
                merged[name] = merged.get(name, 0) + int(count)
        total[key] = dict(sorted(merged.items()))
    for key in ("by_destination", "by_object"):
        merged_table: dict[str, dict[str, int]] = {}
        for row in rounds:
            for name, values in dict(row.get(key) or {}).items():
                target = merged_table.setdefault(name, {})
                for field_name, count in dict(values).items():
                    target[field_name] = target.get(field_name, 0) + int(count)
        total[key] = dict(sorted(merged_table.items()))
    # Conditional yields, which is what the next collection budget is planned
    # against. An unconditional acceptance rate cannot say whether the loss is
    # in the approach, the grasp or the release.
    total["align_given_reach"] = _ratio(total["aligned"], total["reached"])
    total["pickup_given_align"] = _ratio(total["picked_up"], total["aligned"])
    total["placement_given_pickup"] = _ratio(
        total["native_placement_success"], total["picked_up"]
    )
    total["accepted_given_native_placement"] = _ratio(
        total["accepted_chains"], total["native_placement_success"]
    )
    total["accepted_per_world"] = _ratio(
        total["accepted_chains"], total["worlds"]
    )
    return total


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def endpoint_yaw_report(round_result: Any, calibration: PickupYawCalibration) -> dict[str, Any]:
    """The reach teacher's actual endpoint yaw, and what the tail had to undo.

    The design asks for the endpoint yaw to be MEASURED rather than assumed, and
    this is that measurement taken from the recording instead of from a second
    rollout: the wrist's yaw at the last env step of the decision the reach
    event fired on, which is the state the alignment tail inherited.
    """

    per = int(round_result.actions_per_decision)
    events = np.asarray(round_result.reach_event)
    live = np.flatnonzero(events >= 0)
    if live.size == 0:
        return {"endpoints": 0}
    steps = np.minimum(
        (events[live] + 1) * per - 1, round_result.ee_yaw.shape[0] - 1
    )
    yaws = round_result.ee_yaw[steps, live]
    error = np.abs(
        np.angle(np.exp(1j * (yaws - float(calibration.target_yaw))))
    )
    return {
        "endpoints": int(live.size),
        "yaw_mean_rad": round(float(yaws.mean()), 5),
        "yaw_std_rad": round(float(yaws.std()), 5),
        "abs_error_mean_deg": round(float(np.degrees(error.mean())), 3),
        "abs_error_p90_deg": round(float(np.degrees(np.percentile(error, 90))), 3),
        "abs_error_max_deg": round(float(np.degrees(error.max())), 3),
        "already_within_tolerance": int(
            (error <= float(calibration.tolerance_rad)).sum()
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--split", default="collection", choices=SPLIT_NAMES)
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="One per role: move_to, pick_up, placement.",
    )
    parser.add_argument(
        "--runtime-checkpoint",
        type=Path,
        default=None,
        help=(
            "The checkpoint whose saved args rebuild the SmolVLA runtime and "
            "the residual architecture. Defaults to the move_to teacher; every "
            "teacher must agree on the contract anyway, which is checked."
        ),
    )
    parser.add_argument("--yaw-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repeats-per-scene", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--move-decisions", type=int, default=32)
    parser.add_argument(
        "--align-decisions",
        type=int,
        default=0,
        help=(
            "Cap on the yaw-alignment tail, in decisions. 0 means the same as "
            "--move-decisions. It is counted separately in the global loop "
            "budget: the tail gets its own counter, and a loop shorter than "
            "the sum of the stage caps cuts chains off with no failure code."
        ),
    )
    parser.add_argument("--pickup-decisions", type=int, default=32)
    parser.add_argument("--placement-decisions", type=int, default=64)
    parser.add_argument(
        "--settle-decisions",
        type=int,
        default=0,
        help=(
            "Recorded hold after the placement predicate fires, in decisions. "
            "0 keeps acceptance identical to the production verdict; a "
            "positive value makes it strictly stricter and must then be "
            "applied in student evaluation too."
        ),
    )
    parser.add_argument(
        "--no-yaw-hold-during-pickup",
        action="store_true",
        help=(
            "Let the pickup teacher control yaw after the tail. The controlled "
            "variant holds it; this is the ablation arm."
        ),
    )
    parser.add_argument(
        "--align-xy-centring",
        action="store_true",
        help=(
            "Close the lateral error between the reach endpoint and the object "
            "during the alignment tail, so the descent starts from over the "
            "object rather than beside it. OFF by default: it is the one part "
            "of the tail that reads privileged geometry (the object's live XY), "
            "and it hands the controller a materially larger share of the "
            "demonstration than the yaw tail does. With it on, the lateral "
            "readiness bound stops gating the REACH transition -- otherwise the "
            "bridge never runs on the chains that need it -- and keeps gating "
            "the handoff, which is where it matters."
        ),
    )
    parser.add_argument(
        "--align-xy-deadband",
        type=float,
        default=0.005,
        help=(
            "Inside this the bridge commands exactly zero. 5 mm sits well "
            "inside the apple's 13 mm lateral slack."
        ),
    )
    parser.add_argument(
        "--align-xy-abort",
        type=float,
        default=0.009,
        help=(
            "Hysteresis on the centring bridge: the drift tolerated once the "
            "descent has begun. Entering needs --align-xy-deadband; without "
            "the split, ordinary drift repeatedly pauses the vertical bridge."
        ),
    )
    parser.add_argument(
        "--align-handoff-at-clearance",
        action="store_true",
        help=(
            "Stop the alignment tail at the rotation clearance and hand off "
            "from there, instead of descending to the pickup teacher's trained "
            "height. Every descent abort lives in that descent -- at grasp "
            "point + 0.01 m the finger tips straddle the object, so neither "
            "rotating nor translating is free -- and the teacher is trained to "
            "approach from within a 0.20 m cap anyway. An arm to compare, not "
            "a replacement."
        ),
    )
    parser.add_argument(
        "--align-yaw-servo-gain",
        type=float,
        default=0.35,
        help=(
            "Damping on the yaw servo. The command is recomputed every action "
            "and the plant integrates it, so at unity gain the setpoint "
            "absorbs the full measured error four times per decision while the "
            "wrist is still travelling -- integrator windup, and it rings at "
            "0.0824 rad against an 0.0873 rad acceptance band. 1.0 reproduces "
            "the undamped behaviour."
        ),
    )
    parser.add_argument(
        "--align-descent-gain",
        type=float,
        default=1.0,
        help=(
            "Gain on the recorded alignment descent only. Unity reproduces "
            "the original saturated descent; a smaller value gives the "
            "lateral centring loop time to settle before finger contact."
        ),
    )
    parser.add_argument(
        "--grasp-xy-margin",
        type=float,
        default=0.003,
        help=(
            "Safety margin on the lateral readiness gate, in metres. The gate "
            "is (0.0475 open half-aperture - object hull radius - margin), so "
            "0.0130 m of slack for an apple and 0.0185 m for the others. The "
            "production move_to window is 0.02 m -- WIDER than the grasp "
            "tolerance -- which is why a chain can pass the reach predicate "
            "and still hand the pickup teacher a pose it cannot grasp from."
        ),
    )
    parser.add_argument(
        "--pickup-prompt",
        choices=("pick_up", "destination"),
        default="pick_up",
        help=(
            "Which prompt drives the pickup stage. 'destination' uses the "
            "episode's final put_into prompt instead of 'pick up X'. It is a "
            "screening variable because the same adapter commands +0.40 mean "
            "a_z while holding under a put_into prompt and +0.02 under a "
            "pick_up one, and the lift is the pickup stage's known bottleneck."
        ),
    )
    parser.add_argument(
        "--no-gripper-hold-before-pickup",
        action="store_true",
        help=(
            "Let the move-to teacher control the gripper through the approach. "
            "This is the ablation arm. The controlled variant holds the hand "
            "open, because the move-to reward has no gripper term under "
            # Escaped: argparse %-expands help strings, and a bare "% o" is
            # read as the %o conversion, which crashed --help outright.
            "sparse_binary_reward and the shared policy arrives closed on ~100%% "
            "of reaches -- a hand the pickup teacher cannot grasp with."
        ),
    )
    parser.add_argument(
        "--yaw-hold-during-placement",
        action="store_true",
        help=(
            "Pin the wrist through the carry and release as well. Off by "
            "default: the carry is the teacher's, and a controller writing "
            "part of it is provenance nobody asked for."
        ),
    )
    parser.add_argument(
        "--min-height-above-grasp",
        type=float,
        default=-0.005,
        help=(
            "Pickup readiness, RELATIVE to the grasp point (object centre plus "
            "the pad offset). An absolute floor is what broke the first "
            "teacher screen: the pickup teacher's own aligned start is 0.195-"
            "0.202 m for these objects, so a 0.20 m absolute floor sits on top "
            "of the correct answer."
        ),
    )
    parser.add_argument(
        "--align-consecutive-decisions",
        type=int,
        default=None,
        help=(
            "Override how many CONSECUTIVE decision boundaries the full "
            "alignment conjunction must hold before the pickup teacher takes "
            "over. Defaults to the calibration file's value (2). Collect with "
            "whatever the teacher screen selected: this changes the recorded "
            "handoff distribution, so a bank collected at 1 and a bank "
            "collected at 2 are not the same dataset."
        ),
    )
    parser.add_argument("--max-height-above-grasp", type=float, default=0.12)
    parser.add_argument(
        "--min-ee-z",
        type=float,
        default=0.18,
        help="Absolute rail: the configured controller floor, not a reach gate.",
    )
    parser.add_argument("--max-ee-z", type=float, default=0.40)
    parser.add_argument("--min-gripper-opening", type=float, default=0.90)
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help=(
            "Skip the pictures. Only for a timing or yield probe -- a bank "
            "without frames cannot be refreshed onto a new student and cannot "
            "train the vision path."
        ),
    )
    parser.add_argument("--tag", default="staged")
    args = parser.parse_args(argv)

    import torch

    roles = parse_teachers(args.teacher)
    runtime_checkpoint = (
        args.runtime_checkpoint or roles["move_to"]
    ).expanduser().resolve()

    calibration = PickupYawCalibration.from_json(
        json.loads(args.yaw_calibration.expanduser().resolve().read_text("utf-8"))
    )
    if args.align_consecutive_decisions is not None:
        calibration = replace(
            calibration,
            consecutive_decisions=int(args.align_consecutive_decisions),
        )
    calibration.validate()

    scenes, manifest = read_manifest(args.scene_manifest.expanduser().resolve())
    selected = select_split(scenes, args.split)
    if not selected:
        raise SystemExit(
            f"The manifest has no {args.split!r} scenes. Splits present: "
            f"{sorted({scene.split for scene in scenes})}"
        )
    batches = plan_batches(
        selected,
        worlds=int(args.worlds),
        rounds=int(args.rounds),
        repeats_per_scene=int(args.repeats_per_scene),
        shard=int(args.shard),
        num_shards=int(args.num_shards),
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # group_size 2 is the smallest layout the GRPO plumbing accepts. Nothing
    # here uses groups: the full-task route places one scene per world and never
    # calls broadcast_group_state. It is a shape the shared builder demands, not
    # a statement about the data.
    if int(args.worlds) % 2:
        raise SystemExit("--worlds must be even (the shared layout needs pairs).")
    print(
        f"[staged] building the training stack from {runtime_checkpoint}",
        flush=True,
    )
    world = _build_world(
        controller_workspace_from_config=True,
        checkpoint=runtime_checkpoint,
        config_path=args.config.expanduser().resolve(),
        device_str=str(args.device),
        worlds=int(args.worlds),
        group_size=2,
        microbatch=int(args.microbatch),
        load_policy=True,
        run_dir=output,
    )

    entries = load_teacher_entries(torch, roles)
    bank = TeacherBank(
        torch=torch,
        runtime=world.runtime,
        trainer=world.trainer,
        entries=entries,
    )
    for entry in entries:
        print(
            f"[staged] teacher {entry.role}: {entry.checkpoint.name} "
            f"sha256={entry.sha256[:16]} residual_scale={entry.residual_scale}",
            flush=True,
        )

    resetter = FullTaskSceneResetter(
        backend=world.backend,
        worlds_per_rank=int(args.worlds),
        support_surface_z=float(
            world.task_metadata.get("support_surface_z", 0.15)
        ),
        task_metadata=world.task_metadata,
    )

    vision_dim = (
        int(getattr(world.args, "residual_vision_dim", 0))
        if bool(getattr(world.args, "residual_vision_features", False))
        else 0
    )
    config = StagedRolloutConfig(
        actions_per_decision=int(world.args.replan_every),
        state_dim=int(world.payload["state_dim"]),
        chunk_size=int(world.payload["chunk_size"]),
        budgets=StageBudgets(
            move_decisions=int(args.move_decisions),
            align_decisions=int(args.align_decisions) or None,
            pickup_decisions=int(args.pickup_decisions),
            placement_decisions=int(args.placement_decisions),
            settle_decisions=int(args.settle_decisions),
        ),
        calibration=calibration,
        readiness=PickupReadiness(
            min_height_above_grasp=float(args.min_height_above_grasp),
            max_height_above_grasp=float(args.max_height_above_grasp),
            min_ee_z=float(args.min_ee_z),
            max_ee_z=float(args.max_ee_z),
            min_gripper_opening=float(args.min_gripper_opening),
            grasp_xy_margin=float(args.grasp_xy_margin),
            # The bridge exists to fix an off-centre reach, so gating the reach
            # on being centred would stop it ever running.
            require_centred_at_reach=not bool(args.align_xy_centring),
        ),
        pick_grasp_height_offset=float(
            world.task_metadata.get("pick_grasp_height_offset", 0.0075)
        ),
        include_relative_target=bool(
            getattr(world.args, "residual_relative_target", False)
        ),
        vision_feature_dim=vision_dim,
        microbatch_size=int(args.microbatch),
        action_step_xyz=float(world.args.action_step_xyz),
        action_step_yaw=float(world.args.action_step_yaw),
        action_step_gripper=float(world.args.action_step_gripper),
        gripper_hold_open_before_pickup=not bool(
            args.no_gripper_hold_before_pickup
        ),
        align_xy_centring=bool(args.align_xy_centring),
        align_xy_deadband=float(args.align_xy_deadband),
        align_xy_abort=float(args.align_xy_abort),
        align_handoff_at_clearance=bool(args.align_handoff_at_clearance),
        align_yaw_servo_gain=float(args.align_yaw_servo_gain),
        align_descent_gain=float(args.align_descent_gain),
        pickup_prompt=str(args.pickup_prompt),
        yaw_hold_during_pickup=not bool(args.no_yaw_hold_during_pickup),
        yaw_hold_during_placement=bool(args.yaw_hold_during_placement),
        record_frames=not bool(args.no_frames),
    )
    config.validate()
    print(
        f"[staged] budget {config.budgets.total_decisions} decisions x "
        f"{config.actions_per_decision} actions = "
        f"{config.budgets.total_decisions * config.actions_per_decision} env "
        "steps per chain",
        flush=True,
    )
    if config.record_frames:
        held = (
            int(args.worlds)
            * config.budgets.total_decisions
            * 2
            * 240
            * 320
            * 3
            / 1e9
        )
        print(f"[staged] frame buffers will hold ~{held:.2f} GB", flush=True)

    round_reports: list[dict[str, Any]] = []
    shard_files: list[dict[str, Any]] = []
    started = time.perf_counter()
    for round_index, batch in enumerate(batches):
        stem = f"{args.tag}_s{int(args.shard)}_r{round_index}"
        batch_scenes = [scene for scene, _ in batch]
        repeats = [repeat for _, repeat in batch]
        episode_uids = [
            f"{stem}/r{round_index}w{world_index}"
            for world_index in range(len(batch_scenes))
        ]
        round_started = time.perf_counter()
        result = run_staged_chains(
            backend=world.backend,
            collector=world.collector,
            resetter=resetter,
            bank=bank,
            scenes=batch_scenes,
            config=config,
            round_index=round_index,
            episode_uids=episode_uids,
            rollout_index=repeats,
        )
        accepted, reasons, _ = result.acceptance()
        summary = result.summary()
        summary["wall_seconds"] = round(time.perf_counter() - round_started, 1)
        summary["endpoint_yaw"] = endpoint_yaw_report(result, calibration)
        summary["scene_uids"] = [scene.scene_uid for scene in batch_scenes]
        round_reports.append(summary)

        record_path = output / f"staged_{stem}.npz"
        result.to_npz(record_path)
        files = {"record": str(record_path)}
        if config.record_frames:
            # Keep every world that completed the move/alignment transition.
            # Its successful move slice is reusable even when pickup later
            # fails; pickup-success worlds additionally contribute their pick
            # slice, and accepted worlds contribute placement. This is the
            # earliest verified transition, so it is the broadest frame mask
            # needed by ``stage_transitions.npz`` without retaining arbitrary
            # failures. Older accepted/pickup-only masks made those successful
            # move slices impossible to refresh from images.
            reusable = accepted | (np.asarray(result.align_event) >= 0)
            files["frames"] = write_staged_frames(
                output / f"frames_{stem}.npz",
                buffers=result.frames,
                episode_uids=episode_uids,
                world_index=list(range(len(batch_scenes))),
                keep=reusable,
            )
        # Release the pictures before the next round allocates its own. Holding
        # eight rounds of a 64-world budget would be 30 GB of uint8 for data
        # already on disk.
        result.frames = None
        shard_files.append(files)
        print(
            f"[staged] round {round_index}: reached {summary['reached']}, "
            f"aligned {summary['aligned']}, picked {summary['picked_up']}, "
            f"native placement {summary['native_placement_success']}, "
            f"ACCEPTED {summary['accepted_chains']}/{summary['worlds']} "
            f"in {summary['wall_seconds']}s",
            flush=True,
        )
        print(f"[staged]   rejections: {summary['rejection_reasons']}", flush=True)
        print(f"[staged]   reach: {summary['reach_diagnostics']}", flush=True)
        print(f"[staged]   align: {summary['align_diagnostics']}", flush=True)
        print(f"[staged]   pickup: {summary['pickup_diagnostics']}", flush=True)
        print(
            f"[staged]   placement: {summary['placement_diagnostics']}",
            flush=True,
        )
        print(f"[staged]   endpoint yaw: {summary['endpoint_yaw']}", flush=True)

    report = {
        "tool": "record_cdpr_staged_put_into.py",
        "config": str(args.config),
        "scene_manifest": str(args.scene_manifest),
        "scene_manifest_sha256": manifest.get("manifest_sha256"),
        "split": str(args.split),
        "shard": int(args.shard),
        "num_shards": int(args.num_shards),
        "worlds": int(args.worlds),
        "rounds": int(args.rounds),
        "repeats_per_scene": int(args.repeats_per_scene),
        "runtime_checkpoint": str(runtime_checkpoint),
        "teachers": bank.manifest(),
        "yaw_calibration": calibration.to_json(),
        "budgets": {
            "move_decisions": int(args.move_decisions),
            "align_decisions": int(args.align_decisions) or int(args.move_decisions),
            "pickup_decisions": int(args.pickup_decisions),
            "placement_decisions": int(args.placement_decisions),
            "settle_decisions": int(args.settle_decisions),
        },
        "align_xy_centring": bool(args.align_xy_centring),
        "align_xy_deadband": float(args.align_xy_deadband),
        "align_xy_abort": float(args.align_xy_abort),
        "align_handoff_at_clearance": bool(args.align_handoff_at_clearance),
        "align_yaw_servo_gain": float(args.align_yaw_servo_gain),
        "align_descent_gain": float(args.align_descent_gain),
        "pickup_prompt": str(args.pickup_prompt),
        "gripper_hold_open_before_pickup": not bool(
            args.no_gripper_hold_before_pickup
        ),
        "yaw_hold_during_pickup": not bool(args.no_yaw_hold_during_pickup),
        "yaw_hold_during_placement": bool(args.yaw_hold_during_placement),
        "record_frames": bool(config.record_frames),
        "files": shard_files,
        "rounds_detail": round_reports,
        "pooled": pooled_summary(round_reports),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "failure_vocabulary": list(FAILURE_NAMES),
    }
    (output / "collection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    pooled = report["pooled"]
    print(
        f"[staged] POOLED accepted {pooled['accepted_chains']}/"
        f"{pooled['worlds']} chains "
        f"({pooled['accepted_per_world']}); align|reach "
        f"{pooled['align_given_reach']}, pickup|align "
        f"{pooled['pickup_given_align']}, placement|pickup "
        f"{pooled['placement_given_pickup']}",
        flush=True,
    )
    print(f"[staged] wrote {output / 'collection.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
