#!/usr/bin/env python3
"""Measure the student on the complete task: one prompt, no help, from step 0.

This is the only number that answers the user's request. The teacher chain's
yield, the per-stage assisted scores and the native placement predicate all
measure something narrower, and this tool reports them beside each other so
they cannot be confused.

What is deliberately absent
---------------------------

**The stage state machine.** There are no handoffs to make: one policy runs the
whole episode. Its phases are OBSERVED (did it get near the object, did it
grasp, did it lift, did it release) but nothing acts on the observation.

**The yaw servo.** The alignment tail is part of what the demonstrations teach.
Re-applying the same controller at evaluation would measure a policy plus a
controller, and the design requires that to be a separately reported diagnostic
arm rather than the headline. ``--assisted-yaw`` runs that arm and labels it.

**The teachers.** One checkpoint, one instruction, from the first action.

What is deliberately identical to collection
--------------------------------------------

The scenes, the reset route, the persistent placement observer, the action
budget and -- when the bank was collected with one -- the settle window. A
student given a shorter budget than the teacher chain would be measured on a
different task; a student judged by a laxer contract would be measured by a
different predicate.

Two verdicts are reported and never merged:

``native``      the production placement predicate, comparable with every
                historical score in the campaign report.
``strict``      the full-chain data contract: began empty and outside the goal,
                grasped and lifted, carried without a slip, released
                intentionally, ended in the receptacle.

Usage::

    RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run -n cdpr-mjlab python \\
      tools/audit/evaluate_cdpr_full_put_into.py \\
        --config configs/examples/cdpr_smolvla_three_stage_put_into.yaml \\
        --checkpoint runs/three_stage/sft/sil_sft_adapter.pt \\
        --scene-manifest runs/three_stage/scenes.json \\
        --split student_validation --worlds 64 --rounds 2 \\
        --output runs/three_stage/eval_student
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
    PickupYawCalibration,
    YawTailController,
    clone_task_state,
    destination_instruction_id,
    object_label,
    student_instruction_text,
)
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (  # noqa: E402
    FullTaskSceneResetter,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (  # noqa: E402
    INSTRUCTION_TO_ID,
    build_smolvla_state_tensor,
    evaluate_active_sparse_tasks,
)
from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import (  # noqa: E402
    read_manifest,
    select_split,
)
from tools.audit.select_cdpr_stage_teachers import bootstrap_interval  # noqa: E402


def run_unassisted(
    *,
    world: Any,
    resetter: FullTaskSceneResetter,
    scenes: Sequence[Any],
    decisions: int,
    settle_decisions: int,
    assisted_yaw: PickupYawCalibration | None,
    vision_dim: int,
) -> dict[str, Any]:
    """One rollout of the student over a batch of full-task scenes."""

    import torch

    worlds = len(scenes)
    labels = [object_label(scene.target_catalog) for scene in scenes]
    texts = [
        student_instruction_text(label, scene.destination)
        for label, scene in zip(labels, scenes)
    ]
    reset = resetter.reset(
        scenes,
        instruction_ids=[
            destination_instruction_id(scene.destination) for scene in scenes
        ],
        instruction_texts=texts,
        horizons=[int(decisions) + int(settle_decisions)] * worlds,
    )
    collector = world.collector
    torch_device = world.device
    per = int(world.args.replan_every)
    thresholds = collector._task_thresholds()

    # The pickup observer exists ONLY to record the phase diagnostics. It is a
    # separate allocation, never the placement observer, because the predicate
    # writes grasp history in place and the placement verdict depends on it.
    pick_state = clone_task_state(
        torch, reset.task_state, instruction_id=INSTRUCTION_TO_ID["pick_up"]
    )
    place_state = reset.task_state

    servo = (
        None
        if assisted_yaw is None
        else YawTailController(
            torch=torch,
            calibration=assisted_yaw,
            action_step_yaw=float(world.args.action_step_yaw),
            action_step_xyz=float(world.args.action_step_xyz),
        )
    )

    active = torch.ones((worlds,), dtype=torch.bool, device=torch_device)
    native = torch.zeros_like(active)
    ever_grasped = torch.zeros_like(active)
    ever_lifted = torch.zeros_like(active)
    ever_released = torch.zeros_like(active)
    wrong_place = torch.zeros_like(active)
    carry_slip = torch.zeros_like(active)
    approached = torch.zeros_like(active)
    completion_steps = torch.zeros(
        (worlds,), dtype=torch.int64, device=torch_device
    )
    min_target_distance = torch.full(
        (worlds,), float("inf"), dtype=torch.float32, device=torch_device
    )
    peak_lift = torch.zeros(
        (worlds,), dtype=torch.float32, device=torch_device
    )
    proprio_dim = int(world.payload["state_dim"]) - int(vision_dim)

    with torch.inference_mode():
        for _ in range(int(decisions) + int(settle_decisions)):
            if not bool(active.any().item()):
                break
            cameras = world.backend.render_policy_cameras()
            low_dim = world.backend.low_dim_observations()
            proprio = build_smolvla_state_tensor(
                ee_position=low_dim.ee_position,
                ee_yaw=low_dim.ee_yaw,
                gripper_opening=low_dim.gripper_opening,
                object_positions=low_dim.object_positions,
                target_slots=place_state.target_slots,
                state_dim=proprio_dim,
                include_relative_target=bool(
                    getattr(world.args, "residual_relative_target", False)
                ),
                goal_slots=place_state.reference_slots,
            )
            if int(vision_dim) > 0:
                prior, vision = (
                    world.runtime.sample_cdpr_chunks_and_vision_from_tensors(
                        primary_images=cameras.overview,
                        wrist_images=cameras.wrist,
                        states=proprio,
                        instructions=tuple(texts),
                        vision_dim=int(vision_dim),
                        microbatch_size=int(
                            world.args.smolvla_inference_microbatch_size
                        ),
                    )
                )
                state_tensor = torch.cat(
                    [proprio, vision.to(dtype=proprio.dtype)], dim=-1
                )
            else:
                prior = world.runtime.sample_cdpr_chunks_from_tensors(
                    primary_images=cameras.overview,
                    wrist_images=cameras.wrist,
                    states=proprio,
                    instructions=tuple(texts),
                    microbatch_size=int(
                        world.args.smolvla_inference_microbatch_size
                    ),
                )
                state_tensor = proprio
            chunk = world.trainer.deterministic_action_chunks_tensor(
                states=state_tensor, priors=prior, action_count=per
            )
            for action_index in range(per):
                step_active = active.clone()
                action = chunk[:, action_index].clone()
                if servo is not None:
                    # The assisted DIAGNOSTIC arm, and it is labelled as such
                    # in the report. It measures a policy plus a controller.
                    action[:, 3] = servo.yaw_command(low_dim.ee_yaw)
                low_dim = world.backend.step(action, step_active)
                low_dim, caught, grasp_diagnostics = (
                    collector._update_physical_grasp(reset, low_dim, step_active)
                )
                pick_result = evaluate_active_sparse_tasks(
                    state=pick_state,
                    ee_position=low_dim.ee_position,
                    object_positions=low_dim.object_positions,
                    gripper_opening=low_dim.gripper_opening,
                    caught_target=caught,
                    active_mask=step_active,
                    max_steps=1_000_000_000,
                    thresholds=thresholds,
                    move_to_distance_reward=collector.move_to_distance_reward,
                    catch_release_dense_reward=(
                        collector.catch_release_dense_reward
                    ),
                    bilateral_contact=grasp_diagnostics["bilateral_contact"],
                )
                place_result = evaluate_active_sparse_tasks(
                    state=place_state,
                    ee_position=low_dim.ee_position,
                    object_positions=low_dim.object_positions,
                    gripper_opening=low_dim.gripper_opening,
                    caught_target=caught,
                    active_mask=step_active,
                    max_steps=1_000_000_000,
                    thresholds=thresholds,
                    move_to_distance_reward=collector.move_to_distance_reward,
                    catch_release_dense_reward=(
                        collector.catch_release_dense_reward
                    ),
                    bilateral_contact=grasp_diagnostics["bilateral_contact"],
                )
                released = place_result.diagnostics["released"]
                lift = pick_result.diagnostics["target_lift"]

                native |= place_result.success
                ever_grasped |= caught & step_active
                ever_lifted |= pick_result.success
                ever_released |= released & step_active & ever_grasped
                wrong_place |= place_result.diagnostics["wrong_place_drop"]
                # A slip is the object leaving a hand that is still CLOSED, the
                # same distinction the collector makes -- otherwise the
                # intentional release at the end of every success would be
                # counted as a drop.
                carry_slip |= (
                    ever_lifted & ~caught & ~released & step_active & ~native
                )
                approached |= (
                    pick_result.diagnostics["pick_grasp_distance"] <= 0.03
                ) & step_active
                min_target_distance = torch.minimum(
                    min_target_distance,
                    torch.where(
                        step_active,
                        pick_result.diagnostics["pick_grasp_distance"],
                        min_target_distance,
                    ),
                )
                peak_lift = torch.maximum(
                    peak_lift,
                    torch.where(step_active, lift, torch.zeros_like(lift)),
                )
                completion_steps += step_active.to(dtype=completion_steps.dtype)
                # A world stops when the production predicate ends it, exactly
                # as it does in training. `terminated` is success, a settled
                # wrong place, or a timeout that cannot fire at this max_steps.
                active &= ~place_result.terminated

    geometry = (
        place_result.diagnostics["container_xy_error"]
        <= place_result.diagnostics["container_xy_radius"]
    )
    strict = (
        native
        & ever_grasped
        & ever_lifted
        & ever_released
        & ~carry_slip
        & ~wrong_place
    )
    host = lambda tensor: tensor.detach().cpu().numpy()  # noqa: E731
    return {
        "native": host(native),
        "strict": host(strict),
        "approached": host(approached),
        "grasped": host(ever_grasped),
        "lifted": host(ever_lifted),
        "released": host(ever_released),
        "carry_slip": host(carry_slip),
        "wrong_place": host(wrong_place),
        "final_geometry_ok": host(geometry),
        "completion_steps": host(completion_steps),
        "min_grasp_distance": host(min_target_distance),
        "peak_lift": host(peak_lift),
        "scene_uid": np.asarray([scene.scene_uid for scene in scenes]),
        "destination": np.asarray([scene.destination for scene in scenes]),
        "target_catalog": np.asarray(
            [scene.target_catalog for scene in scenes]
        ),
    }


def summarize(rollouts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    pooled = {
        key: np.concatenate([row[key] for row in rollouts])
        for key in rollouts[0]
    }
    scenes = pooled["scene_uid"]
    report: dict[str, Any] = {
        "chains": int(scenes.size),
        "scenes": int(np.unique(scenes).size),
    }
    for name in ("native", "strict"):
        report[name] = bootstrap_interval(pooled[name], scenes)
        for destination in np.unique(pooled["destination"]):
            mask = pooled["destination"] == destination
            report[f"{name}_{destination}"] = bootstrap_interval(
                pooled[name][mask], scenes[mask]
            )
    # The phase ladder. Where a full-task policy loses is the question a single
    # success rate cannot answer, and the campaign has spent whole phases on
    # the wrong half of it: composed put_into was assumed to fail at placement
    # and measured to fail at the grasp.
    report["phases"] = {
        name: round(float(pooled[name].mean()), 4)
        for name in (
            "approached",
            "grasped",
            "lifted",
            "released",
            "carry_slip",
            "wrong_place",
            "final_geometry_ok",
        )
    }
    ladder = {}
    for upper, lower in (
        ("grasped", "approached"),
        ("lifted", "grasped"),
        ("released", "lifted"),
        ("native", "released"),
    ):
        given = pooled[lower].astype(bool)
        ladder[f"{upper}_given_{lower}"] = (
            round(float(pooled[upper][given].mean()), 4)
            if bool(given.any())
            else None
        )
    report["conditional_ladder"] = ladder
    report["by_object"] = {
        str(name): bootstrap_interval(
            pooled["strict"][pooled["target_catalog"] == name],
            scenes[pooled["target_catalog"] == name],
        )
        for name in np.unique(pooled["target_catalog"])
    }
    succeeded = pooled["native"].astype(bool)
    report["completion_env_steps"] = {
        "mean_all": round(float(pooled["completion_steps"].mean()), 2),
        "mean_successful": (
            round(float(pooled["completion_steps"][succeeded].mean()), 2)
            if bool(succeeded.any())
            else None
        ),
    }
    report["min_grasp_distance_m"] = {
        "median": round(float(np.median(pooled["min_grasp_distance"])), 4),
        "p10": round(float(np.percentile(pooled["min_grasp_distance"], 10)), 4),
    }
    report["peak_lift_m"] = {
        "median": round(float(np.median(pooled["peak_lift"])), 4),
        "p90": round(float(np.percentile(pooled["peak_lift"], 90)), 4),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--split", default="student_validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument(
        "--decisions",
        type=int,
        default=128,
        help=(
            "The SAME total budget the teacher chain had (32 move + 32 pickup "
            "+ 64 placement). A student given less is measured on a different "
            "task; report shorter budgets as their own arm rather than "
            "attributing a horizon change to learning."
        ),
    )
    parser.add_argument("--settle-decisions", type=int, default=0)
    parser.add_argument(
        "--assisted-yaw",
        type=Path,
        default=None,
        help=(
            "Run the DIAGNOSTIC arm: re-apply the collection yaw servo during "
            "evaluation. This measures a policy plus a controller and is "
            "reported under its own key, never as the headline."
        ),
    )
    args = parser.parse_args(argv)

    scenes, manifest = read_manifest(args.scene_manifest.expanduser().resolve())
    selected = select_split(scenes, str(args.split))
    if len(selected) < int(args.worlds):
        raise SystemExit(
            f"The {args.split} split holds {len(selected)} scenes against "
            f"--worlds {args.worlds}."
        )
    batch = selected[: int(args.worlds)]
    if int(args.worlds) % 2:
        raise SystemExit("--worlds must be even (the shared layout needs pairs).")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    calibration = None
    if args.assisted_yaw is not None:
        calibration = PickupYawCalibration.from_json(
            json.loads(
                args.assisted_yaw.expanduser().resolve().read_text("utf-8")
            )
        )
        calibration.validate()

    world = _build_world(
        checkpoint=args.checkpoint.expanduser().resolve(),
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

    started = time.perf_counter()
    rollouts = [
        run_unassisted(
            world=world,
            resetter=resetter,
            scenes=batch,
            decisions=int(args.decisions),
            settle_decisions=int(args.settle_decisions),
            assisted_yaw=calibration,
            vision_dim=vision_dim,
        )
        for _ in range(int(args.rounds))
    ]
    summary = summarize(rollouts)

    report = {
        "tool": "evaluate_cdpr_full_put_into.py",
        "arm": "assisted_yaw_diagnostic" if calibration else "unassisted",
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "scene_manifest": str(args.scene_manifest),
        "scene_manifest_sha256": manifest.get("manifest_sha256"),
        "split": str(args.split),
        "worlds": int(args.worlds),
        "rounds": int(args.rounds),
        "decisions": int(args.decisions),
        "settle_decisions": int(args.settle_decisions),
        "results": summary,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "notes": [
            "One student prompt from the first action; no stage machine and "
            "no teacher.",
            "native = the production placement predicate, comparable with the "
            "campaign report's historical scores.",
            "strict = the full-chain data contract used to accept "
            "demonstrations: began empty and outside the goal, grasped, "
            "lifted, carried without a slip, released intentionally.",
        ],
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[eval] {report['arm']}: native {summary['native']}, strict "
        f"{summary['strict']}",
        flush=True,
    )
    for destination in ("plate", "bowl"):
        key = f"strict_{destination}"
        if key in summary:
            print(f"[eval]   {key}: {summary[key]}", flush=True)
    print(f"[eval] phases {summary['phases']}", flush=True)
    print(f"[eval] ladder {summary['conditional_ladder']}", flush=True)
    print(f"[eval] wrote {output / 'evaluation.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
