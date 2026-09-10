#!/usr/bin/env python3
"""Choose the three stage teachers on the NEW scene and handoff distribution.

There is no supported global winner among these checkpoints. Every historical
score was measured at a different start distance, a different yaw distribution,
a different caught fraction or a different success predicate, so a shortlist is
a list of things to test and not a ranking. This tool does the testing.

The procedure, and why it is sequential rather than a grid
---------------------------------------------------------

A grid over three candidate sets is the product of their sizes, and most of
those combinations are not worth a GPU-hour. The design's procedure is greedy
and each step is scored on the ACTUAL upstream state rather than on a synthetic
stand-in:

1. **Reach.** Every move-to candidate runs the same full-task starts. The score
   is not native XY success: it is how many chains reach a pose the pickup can
   begin from -- open gripper, no grasp, inside the height band -- and then
   hold the calibrated yaw. XY success alone does not establish a usable pickup
   pose, which is the whole reason the alignment tail exists.

2. **Pickup**, with the chosen reach teacher fixed. The pickup candidates are
   therefore compared on real reach endpoints produced by the teacher that will
   actually produce them, from the same scenes. No cloned state, no "similar"
   reset: the endpoints are the ones the chain arrives at.

3. **Placement**, with reach and pickup fixed, scored separately for plate and
   bowl on real post-lift handoffs. A placement candidate's historical
   caught-start score says nothing about this distribution -- the caught reset
   hands the episode a grasp at a hover height, and a real handoff arrives from
   a lift with whatever pose the grasp left behind.

4. **Confirmation** of the winning triple on full chains, because the quantity
   that matters is complete-chain yield and not the sum of three stage maxima.

Stages 1-3 stop early by setting the DOWNSTREAM budgets to one decision, so a
reach comparison does not pay for 96 decisions of placement it will discard.

Scoring and uncertainty
-----------------------

Every rate is reported with a scene-clustered bootstrap interval. One chain is
one scene, and repeats of a scene are one cluster, so treating chains as
independent samples would overstate the precision by the repeat factor. A
screening budget of 64 scenes per destination separates large differences and
not small ones; the tool says so rather than printing a winner.

The teacher-selection SPLIT is used and the final test split is never touched.

Usage::

    RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run -n cdpr-mjlab python \\
      tools/audit/select_cdpr_stage_teachers.py \\
        --config configs/examples/cdpr_smolvla_three_stage_put_into.yaml \\
        --scene-manifest runs/three_stage/scenes.json \\
        --yaw-calibration runs/three_stage/yaw_calibration.json \\
        --candidate move_to=runs/A/.../smolvla_grpo_adapter.pt \\
        --candidate move_to=runs/B/.../smolvla_grpo_adapter.pt \\
        --candidate pick_up=runs/C/.../smolvla_grpo_adapter.pt \\
        --candidate placement=runs/D/.../smolvla_grpo_adapter.pt \\
        --worlds 64 --output runs/three_stage/teacher_selection
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.xy_approach_probe import _build_world  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (  # noqa: E402
    TEACHER_ROLES,
    PickupReadiness,
    PickupYawCalibration,
    StageBudgets,
    StagedRolloutConfig,
    TeacherBank,
    load_teacher_entries,
    run_staged_chains,
)
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (  # noqa: E402
    FullTaskSceneResetter,
)
from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import (  # noqa: E402
    read_manifest,
    select_split,
)


def bootstrap_interval(
    successes: np.ndarray,
    clusters: np.ndarray,
    *,
    draws: int = 2000,
    seed: int = 20260910,
) -> dict[str, Any]:
    """A 90% interval that resamples SCENES, not chains.

    Two rollouts of one scene share the start, the object and the geometry;
    they differ by the policy's sampling noise. Resampling chains would treat
    them as two independent observations and report an interval narrower than
    the experiment supports -- by roughly the square root of the repeat factor.
    """

    successes = np.asarray(successes, dtype=float)
    clusters = np.asarray(clusters)
    if successes.size == 0:
        return {"rate": None, "low": None, "high": None, "chains": 0, "clusters": 0}
    unique = np.unique(clusters)
    groups = [successes[clusters == name] for name in unique]
    generator = np.random.default_rng(int(seed))
    samples = np.empty((int(draws),), dtype=float)
    for index in range(int(draws)):
        picked = generator.integers(len(groups), size=len(groups))
        pooled = np.concatenate([groups[position] for position in picked])
        samples[index] = float(pooled.mean())
    return {
        "rate": round(float(successes.mean()), 4),
        "low": round(float(np.percentile(samples, 5)), 4),
        "high": round(float(np.percentile(samples, 95)), 4),
        "chains": int(successes.size),
        "clusters": int(unique.size),
    }


def _stem(path: Path) -> str:
    """A filename-safe identity for a checkpoint: its run and step."""

    parts = Path(path).parts
    return "_".join(part for part in parts[-3:-1]) or Path(path).stem


def parse_candidates(specs: Sequence[str]) -> dict[str, list[Path]]:
    table: dict[str, list[Path]] = {role: [] for role in TEACHER_ROLES}
    for spec in specs:
        role, _, path = str(spec).partition("=")
        if role not in TEACHER_ROLES or not path:
            raise SystemExit(
                f"--candidate expects ROLE=PATH with ROLE in "
                f"{list(TEACHER_ROLES)}, got {spec!r}"
            )
        table[role].append(Path(path))
    empty = [role for role, paths in table.items() if not paths]
    if empty:
        raise SystemExit(f"No candidates supplied for {empty}.")
    return table


def score_rounds(results: Sequence[Any], *, phase: str) -> dict[str, Any]:
    """Turn a candidate's screening rounds into the number it is chosen on.

    Rounds are POOLED, not averaged: they are extra rollouts of the same scenes,
    so the cluster is the scene and the pooling has to happen before the
    interval is computed. Averaging per-round rates would also silently weight a
    round that lost worlds to divergence the same as a full one.
    """

    def cat(extract: Any) -> np.ndarray:
        return np.concatenate([np.asarray(extract(row)) for row in results])

    scenes = cat(lambda row: row.scene_uid)
    destination = cat(lambda row: row.destination)
    reached = cat(lambda row: row.reach_event) >= 0
    aligned = cat(lambda row: row.align_event) >= 0
    picked = cat(lambda row: row.pickup_event) >= 0
    placed = cat(lambda row: row.placement_event) >= 0
    accepted = cat(lambda row: row.acceptance()[0])
    reasons: dict[str, int] = {}
    failures: dict[str, int] = {}
    diagnostics: list[dict[str, Any]] = []
    pickup: list[dict[str, Any]] = []
    placement: list[dict[str, Any]] = []
    for row in results:
        summary = row.summary()
        for name, count in summary["rejection_reasons"].items():
            reasons[name] = reasons.get(name, 0) + int(count)
        for name, count in summary["failure_counts"].items():
            failures[name] = failures.get(name, 0) + int(count)
        diagnostics.append(summary["reach_diagnostics"])
        pickup.append(summary["pickup_diagnostics"])
        placement.append(summary["placement_diagnostics"])

    if phase == "move_to":
        # READINESS, not XY success: the chain has to arrive somewhere a pickup
        # can start, and then hold the calibrated yaw for two decisions.
        primary = aligned
    elif phase == "pick_up":
        primary = picked
    else:
        primary = accepted

    report: dict[str, Any] = {
        "phase": phase,
        "primary": bootstrap_interval(primary, scenes),
        "reached": bootstrap_interval(reached, scenes),
        "aligned": bootstrap_interval(aligned, scenes),
        "picked_up": bootstrap_interval(picked, scenes),
        "native_placement": bootstrap_interval(placed, scenes),
        "accepted": bootstrap_interval(accepted, scenes),
        "rejection_reasons": dict(sorted(reasons.items())),
        "failure_counts": dict(sorted(failures.items())),
        # Per round, because a zero has to be readable without a second run.
        "reach_diagnostics": diagnostics,
        "pickup_diagnostics": pickup,
        "placement_diagnostics": placement,
    }
    # Conditional yields, which is where a comparison becomes actionable: a
    # pickup candidate that converts 40% of the reaches it is given is better
    # than one that converts 20%, whatever the unconditional numbers say.
    denominators = {"aligned": reached, "picked_up": aligned, "native_placement": picked}
    for name, given in denominators.items():
        selected = np.flatnonzero(given)
        values = {"aligned": aligned, "picked_up": picked, "native_placement": placed}[
            name
        ]
        report[f"{name}_given_upstream"] = bootstrap_interval(
            values[selected], scenes[selected]
        )
    conditional_key = {
        "move_to": "aligned_given_upstream",
        "pick_up": "picked_up_given_upstream",
        "placement": "native_placement_given_upstream",
    }[phase]
    report["conditional_metric"] = conditional_key
    report["conditional"] = report[conditional_key]
    for name in np.unique(destination):
        mask = destination == name
        report[f"primary_{name}"] = bootstrap_interval(
            primary[mask], scenes[mask]
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--yaw-calibration", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", default=[], metavar="ROLE=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=1)
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
            "Ablation: let the move-to teacher control the gripper through the "
            "approach. The screen must use the SAME setting the collection "
            "will, or it ranks teachers on a protocol the bank will not use."
        ),
    )
    parser.add_argument(
        "--dump-rounds",
        action="store_true",
        help=(
            "Write each screening round's npz beside the report. Off by "
            "default because a screen is throwaway, but a screen that scores "
            "zero is not throwaway -- it is the thing to look at."
        ),
    )
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--move-decisions", type=int, default=32)
    parser.add_argument("--pickup-decisions", type=int, default=32)
    parser.add_argument("--placement-decisions", type=int, default=64)
    parser.add_argument(
        "--split",
        default="teacher_selection",
        help=(
            "Deliberately its own split. Selecting a teacher on the final test "
            "scenes makes the final number a training score."
        ),
    )
    parser.add_argument(
        "--runtime-checkpoint",
        type=Path,
        default=None,
        help="Architecture reference; defaults to the first move_to candidate.",
    )
    args = parser.parse_args(argv)

    import torch

    candidates = parse_candidates(args.candidate)
    if str(args.split) == "final_test":
        raise SystemExit(
            "Refusing to select teachers on the final test split. That is the "
            "one split whose number has to stay uncontaminated."
        )
    calibration = PickupYawCalibration.from_json(
        json.loads(args.yaw_calibration.expanduser().resolve().read_text("utf-8"))
    )
    calibration.validate()
    scenes, manifest = read_manifest(args.scene_manifest.expanduser().resolve())
    selection = select_split(scenes, str(args.split))
    if len(selection) < int(args.worlds):
        raise SystemExit(
            f"The {args.split} split holds {len(selection)} scenes against "
            f"--worlds {args.worlds}. Screening on fewer scenes than worlds "
            "would repeat scenes inside one round and make the clustered "
            "interval meaningless."
        )
    batch = selection[: int(args.worlds)]

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime_checkpoint = (
        args.runtime_checkpoint or candidates["move_to"][0]
    ).expanduser().resolve()

    if int(args.worlds) % 2:
        raise SystemExit("--worlds must be even (the shared layout needs pairs).")
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

    def make_config(phase: str) -> StagedRolloutConfig:
        # Downstream budgets collapse to one decision so a reach screen does
        # not pay for placement it will throw away. The UPSTREAM budgets are
        # never reduced: the phase being scored has to be given the same
        # opportunity every candidate gets.
        move = int(args.move_decisions)
        pick = 1 if phase == "move_to" else int(args.pickup_decisions)
        place = 1 if phase in {"move_to", "pick_up"} else int(
            args.placement_decisions
        )
        return StagedRolloutConfig(
            actions_per_decision=int(world.args.replan_every),
            state_dim=int(world.payload["state_dim"]),
            chunk_size=int(world.payload["chunk_size"]),
            budgets=StageBudgets(move, pick, place),
            calibration=calibration,
            readiness=PickupReadiness(
                grasp_xy_margin=float(args.grasp_xy_margin)
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
            pickup_prompt=str(args.pickup_prompt),
            record_frames=False,
        )

    def evaluate(roles: Mapping[str, Path], phase: str, *, artifact_tag: str = "screen") -> dict[str, Any]:
        entries = load_teacher_entries(torch, roles)
        bank = TeacherBank(
            torch=torch,
            runtime=world.runtime,
            trainer=world.trainer,
            entries=entries,
        )
        collected: list[Any] = []
        for round_index in range(int(args.rounds)):
            result = run_staged_chains(
                backend=world.backend,
                collector=world.collector,
                resetter=resetter,
                bank=bank,
                scenes=batch,
                config=make_config(phase),
                round_index=round_index,
                episode_uids=[
                    f"select_{phase}_r{round_index}/r{round_index}w{index}"
                    for index in range(len(batch))
                ],
                rollout_index=[round_index] * len(batch),
            )
            result.frames = None
            if bool(args.dump_rounds):
                result.to_npz(
                    output / f"{artifact_tag}_{phase}_{_stem(roles[phase])}_r{round_index}.npz"
                )
            collected.append(result)
        merged = score_rounds(collected, phase=phase)
        merged["rounds"] = int(args.rounds)
        merged["teachers"] = {
            role: {
                "path": str(entry.checkpoint),
                "sha256": entry.sha256,
            }
            for role, entry in bank.entries.items()
        }
        return merged

    report: dict[str, Any] = {
        "tool": "select_cdpr_stage_teachers.py",
        "config": str(args.config),
        "scene_manifest": str(args.scene_manifest),
        "scene_manifest_sha256": manifest.get("manifest_sha256"),
        "split": str(args.split),
        "scenes_scored": len(batch),
        "yaw_calibration": calibration.to_json(),
        "controller_workspace_z_bounds": list(world.args.controller_workspace_z_bounds),
        "teacher_sampling": "role_decision_seeded_v1",
        "phases": {},
    }
    started = time.perf_counter()
    chosen: dict[str, Path] = {
        role: paths[0] for role, paths in candidates.items()
    }

    for phase in ("move_to", "pick_up", "placement"):
        rows: list[dict[str, Any]] = []
        for candidate in candidates[phase]:
            roles = dict(chosen)
            roles[phase] = candidate
            print(f"[select] {phase}: {candidate}", flush=True)
            scored = evaluate(roles, phase)
            scored["candidate"] = str(candidate)
            rows.append(scored)
            print(
                f"[select]   primary {scored['primary']}; "
                f"{scored['conditional_metric']} {scored['conditional']}",
                flush=True,
            )
            head = scored["reach_diagnostics"][0]
            print(
                "[select]   reach: predicate fired on "
                f"{head['predicate_fired_worlds']}/{head['worlds']} worlds, "
                f"ready {head['ready_worlds']}, closest XY "
                f"{head['closest_xy_distance_m']}, height above grasp "
                f"{head['height_above_grasp_m']}",
                flush=True,
            )
            if head.get("among_predicate_steps"):
                print(
                    f"[select]   gates: {head['among_predicate_steps']}",
                    flush=True,
                )
            print(f"[select]   pickup: {scored['pickup_diagnostics'][0]}", flush=True)
            print(
                f"[select]   placement: {scored['placement_diagnostics'][0]}",
                flush=True,
            )
        best = max(rows, key=lambda row: (row["primary"]["rate"] or 0.0))
        if not best["primary"]["rate"]:
            report["phases"][phase] = {"candidates": rows, "chosen": None}
            report["status"] = "blocked_zero_stage_success"
            report["blocked_phase"] = phase
            report["wall_seconds"] = round(time.perf_counter() - started, 1)
            (output / "teacher_selection.json").write_text(json.dumps(report, indent=2))
            (output / "selected_teachers.json").unlink(missing_ok=True)
            print(f"[select] STOP: no successful {phase} candidate. Saved diagnostics; "
                  "no selected_teachers.json. Downstream roles cannot be selected.", flush=True)
            return 2
        chosen[phase] = Path(best["candidate"])
        # Overlapping intervals mean the screen did not separate these two, and
        # saying so is the point of computing them.
        contested = [
            row["candidate"]
            for row in rows
            if row is not best
            and (row["primary"]["high"] or 0.0)
            >= (best["primary"]["low"] or 0.0)
        ]
        report["phases"][phase] = {
            "candidates": rows,
            "chosen": str(chosen[phase]),
            "not_separated_from_chosen": contested,
        }
        print(
            f"[select] {phase} -> {chosen[phase]}"
            + (
                f"  (NOT separated from {len(contested)} other candidate(s) at "
                "this budget; expand the selection scenes rather than reading "
                "the point estimate)"
                if contested
                else ""
            ),
            flush=True,
        )

    print("[select] confirming the chosen triple on full chains", flush=True)
    confirmation = evaluate(chosen, "placement", artifact_tag="confirmation")
    report["confirmation"] = confirmation
    report["chosen"] = {role: str(path) for role, path in chosen.items()}
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    report["status"] = ("selected" if confirmation["accepted"]["rate"]
                        else "blocked_zero_full_chain_success")
    (output / "teacher_selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not confirmation["accepted"]["rate"]:
        (output / "selected_teachers.json").unlink(missing_ok=True)
        print(f"[select] STOP: chosen triple produced no accepted full chains; "
              f"diagnostics saved to {output / 'teacher_selection.json'}", flush=True)
        return 2
    (output / "selected_teachers.json").write_text(
        json.dumps(
            {
                "teachers": confirmation["teachers"],
                "scene_manifest_sha256": manifest.get("manifest_sha256"),
                "split": str(args.split),
                "accepted": confirmation["accepted"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"[select] chosen {report['chosen']}; full-chain acceptance "
        f"{confirmation['accepted']}",
        flush=True,
    )
    print(f"[select] wrote {output / 'teacher_selection.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
