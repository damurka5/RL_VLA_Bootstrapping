#!/usr/bin/env python3
"""Preflight the WidowX-200 pipeline on the plant itself, before any training.

Builds the production MJWarp backend for the arm and drives it through the
pipeline's own stage configs, checking the things that would otherwise surface
hours into a run as a policy that "cannot learn":

1. **resets** -- the collector's real resetter, for every skill and at the
   bottom, middle and top of its approach ladder. Every start the arm is asked
   for must be reachable (``reachable`` from the IK, not a tautology on the
   clamped target), every caught start must still be holding its object after
   a few still steps, nothing may be non-finite, and active objects must not
   interpenetrate.
2. **the task is physically solvable** -- a scripted top-down grasp and 5 cm
   lift on every world of a pick_up reset, through the same normalized
   five-channel action the policy emits. This is the WidowX analogue of the
   CDPR oracle ladder: if the script cannot lift at a cell, neither can RL.
3. **cameras** -- one render, checked for finite non-blank images, saved as a
   PNG grid (overview | wrist) for a human to look at.

Runs on CUDA (production) or, with ``RLVLA_MJWARP_ALLOW_CPU=1 --device cpu``,
on Warp's CPU device for a laptop check. Writes ``preflight.json`` and exits
non-zero when a check fails.

Usage::

    RLVLA_EMBODIMENT=widowx200 python scripts/preflight_widowx200_pipeline.py \\
        --output runs/widowx200_preflight
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("RLVLA_EMBODIMENT", "widowx200")

from tools.train.run_widowx200_pipeline import (  # noqa: E402
    PIPELINE_KEY,
    deep_merge,
    load_yaml,
)


def merged_skill(raw: dict[str, Any], skill: str) -> dict[str, Any]:
    spec = raw[PIPELINE_KEY]
    base = {key: value for key, value in raw.items() if key != PIPELINE_KEY}
    return deep_merge(base, spec["skills"][skill].get("overrides") or {})


def ladder_points(metadata: dict[str, Any], instructions: list[str]) -> list[dict[str, float]]:
    """Bottom, middle and top of every instruction's ladder, as cap tables."""

    by_instruction = metadata.get("random_workspace_start_distance_ladder_by_instruction") or {}
    shared = metadata.get("random_workspace_start_distance_ladder") or [
        float(metadata.get("random_workspace_start_distance_initial", 0.03)),
        float(metadata.get("random_workspace_start_distance_final", 0.30)),
    ]
    points = []
    for position in ("bottom", "middle", "top"):
        table = {}
        for name in instructions:
            ladder = [float(v) for v in (by_instruction.get(name) or shared)]
            index = {"bottom": 0, "middle": len(ladder) // 2, "top": len(ladder) - 1}[position]
            table[name] = ladder[index]
        points.append(table)
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "examples" / "widowx200_smolvla_pipeline.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-reachable", type=float, default=0.98)
    parser.add_argument("--min-caught-held", type=float, default=0.90)
    parser.add_argument("--min-scripted-lift", type=float, default=0.80)
    args = parser.parse_args()

    import numpy as np
    import torch

    from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
        BatchedReverseFrontierResetter,
        RankLocalCurriculum,
    )
    from rl_vla_bootstrapping.policy.rank_local_grpo import RankLocalGroupLayout
    from rl_vla_bootstrapping.simulation.cdpr_backend import CDPRBackendConfig, create_cdpr_backend
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID
    from rl_vla_bootstrapping.simulation.cdpr_object_catalog import ACTIVE_EMBODIMENT

    if ACTIVE_EMBODIMENT != "widowx200":
        raise SystemExit("Set RLVLA_EMBODIMENT=widowx200.")
    raw = load_yaml(Path(args.config))
    base_args = raw["training"]["rl"]["args"]
    xml = (Path(args.config).resolve().parent / raw["simulator"]["fixed_scene_xml"]).resolve()
    worlds = int(args.worlds)
    group_size = int(base_args.get("grpo_group_size", 8))
    layout = RankLocalGroupLayout(worlds_per_rank=worlds, groups_per_rank=worlds // group_size, group_size=group_size)
    layout.validate()
    device = torch.device(args.device)
    backend = create_cdpr_backend(
        CDPRBackendConfig(
            backend="mjlab_mjwarp",
            worlds_per_rank=worlds,
            groups_per_rank=worlds // group_size,
            grpo_group_size=group_size,
            hold_steps=int(base_args["hold_steps"]),
            action_step_xyz=float(base_args["action_step_xyz"]),
            action_step_yaw=float(base_args["action_step_yaw"]),
            action_step_gripper=float(base_args["action_step_gripper"]),
            render_width=int(raw["simulator"]["render_width"]),
            render_height=int(raw["simulator"]["render_height"]),
            nconmax=int(raw["simulator"]["nconmax"]),
            njmax=int(raw["simulator"]["njmax"]),
            device=str(device),
            xml_path=xml,
            workspace_z=(0.168, 0.27),
        )
    )
    report: dict[str, Any] = {"device": str(device), "worlds": worlds, "skills": {}, "failures": []}
    active_all = torch.ones((worlds,), dtype=torch.bool, device=device)
    zeros = torch.zeros((worlds, 5), dtype=torch.float32, device=device)

    def check(condition: bool, message: str) -> None:
        if not condition:
            report["failures"].append(message)
            print(f"[preflight] FAIL {message}", flush=True)

    for skill in ("move_to", "pick_up", "put_into", "joint"):
        config = merged_skill(raw, skill)
        metadata = dict(config["task"]["metadata"])
        instructions = list(config["task"]["instruction_types"])
        allowed = list(metadata.get("scene_object_pool") or config["task"]["target_objects"])
        resetter = BatchedReverseFrontierResetter(
            backend=backend,
            layout=layout,
            curriculum=RankLocalCurriculum(device=device),
            rank=0,
            base_seed=12345,
            instruction_types=instructions,
            allowed_objects=allowed,
            task_metadata=metadata,
        )
        entries = []
        for caps in ladder_points(metadata, instructions):
            resetter.set_random_start_max_goal_distance({INSTRUCTION_TO_ID[name]: cap for name, cap in caps.items()})
            for round_index in range(int(args.rounds)):
                reset = resetter.reset(update_index=0, round_index=round_index, allow_prelifted=True)
                state = backend.controller_state()
                reachable = float(np.mean(state["reachable"]))
                low = backend.low_dim_observations()
                finite = bool(torch.isfinite(low.ee_position).all() and torch.isfinite(low.object_positions).all())
                caught = reset.task_state.ever_grasped.to(torch.bool)
                target = low.object_positions[torch.arange(worlds, device=device), reset.task_state.target_slots]
                for _ in range(4):
                    low = backend.step(zeros, active_all)
                after = low.object_positions[torch.arange(worlds, device=device), reset.task_state.target_slots]
                held = (after - low.ee_position).norm(dim=-1) < 0.03
                held_rate = float(held[caught].float().mean()) if bool(caught.any()) else float("nan")
                positions = low.object_positions
                active = positions[..., 2] < 1.0
                # 3-D: the release rung holds the object straight above its
                # receptacle on purpose, which an XY gap would call a collision.
                gaps = (positions[:, :, None, :] - positions[:, None, :, :]).norm(dim=-1)
                pair = active[:, :, None] & active[:, None, :] & ~torch.eye(4, dtype=torch.bool, device=device)[None]
                min_gap = float(gaps[pair].min()) if bool(pair.any()) else float("nan")
                horizons = reset.horizons.float()
                entry = {
                    "caps": caps,
                    "round": round_index,
                    "reachable_fraction": round(reachable, 4),
                    "finite": finite,
                    "caught_worlds": int(caught.sum()),
                    "caught_held_fraction": None if math.isnan(held_rate) else round(held_rate, 4),
                    "min_object_gap_m": round(min_gap, 4),
                    "target_z_mean": round(float(target[:, 2].mean()), 4),
                    "horizon_decisions": [int(horizons.min()), int(horizons.max())],
                }
                entries.append(entry)
                print(f"[preflight] {skill} {json.dumps(entry)}", flush=True)
                check(finite, f"{skill} {caps}: non-finite state after reset")
                check(reachable >= args.min_reachable, f"{skill} {caps}: only {reachable:.3f} of reset poses reachable")
                if not math.isnan(held_rate):
                    check(held_rate >= args.min_caught_held, f"{skill} {caps}: caught starts holding {held_rate:.3f}")
                check(not (min_gap < 0.03), f"{skill} {caps}: objects {min_gap:.3f} m apart")
        report["skills"][skill] = entries

    # Scripted grasp-and-lift at every lattice cell and every graspable.
    config = merged_skill(raw, "pick_up")
    metadata = dict(config["task"]["metadata"])
    metadata.update(pick_up_prelifted_group_fraction=0.0, pick_up_aligned_group_fraction=1.0)
    resetter = BatchedReverseFrontierResetter(
        backend=backend,
        layout=layout,
        curriculum=RankLocalCurriculum(device=device),
        rank=0,
        base_seed=777,
        instruction_types=["pick_up"],
        allowed_objects=list(metadata["scene_object_pool"]),
        task_metadata=metadata,
    )
    resetter.set_random_start_max_goal_distance({INSTRUCTION_TO_ID["pick_up"]: 0.03})
    lifted_rates = []
    for round_index in range(int(args.rounds)):
        reset = resetter.reset(update_index=1, round_index=round_index, allow_prelifted=True)
        rows = torch.arange(worlds, device=device)
        slots = reset.task_state.target_slots
        low = backend.low_dim_observations()
        start = low.object_positions[rows, slots].clone()
        step = float(base_args["action_step_xyz"])
        for phase, count in (("align", 8), ("descend", 8), ("close", 18), ("lift", 12), ("hold", 6)):
            for _ in range(count):
                low = backend.low_dim_observations()
                obj = low.object_positions[rows, slots]
                action = torch.zeros((worlds, 5), dtype=torch.float32, device=device)
                if phase in ("align", "descend"):
                    action[:, :2] = ((obj[:, :2] - low.ee_position[:, :2]) / step).clamp(-1, 1)
                if phase == "descend":
                    action[:, 2] = ((obj[:, 2] - low.ee_position[:, 2]) / step).clamp(-1, 1)
                if phase in ("close", "lift", "hold"):
                    action[:, 4] = -1.0
                if phase == "lift":
                    action[:, 2] = 1.0
                backend.step(action, active_all)
        low = backend.low_dim_observations()
        lift = low.object_positions[rows, slots][:, 2] - start[:, 2]
        rate = float((lift >= 0.04).float().mean())
        lifted_rates.append(rate)
        print(f"[preflight] scripted grasp+lift round {round_index}: {rate:.3f} lifted >= 4 cm", flush=True)
    report["scripted_lift_rate"] = round(float(np.mean(lifted_rates)), 4)
    check(report["scripted_lift_rate"] >= args.min_scripted_lift, f"scripted grasp+lift only {report['scripted_lift_rate']:.3f}")

    cameras = backend.render_policy_cameras()
    overview = cameras.overview[:4].detach().float().cpu().numpy()
    wrist = cameras.wrist[:4].detach().float().cpu().numpy()
    check(bool(np.isfinite(overview).all() and np.isfinite(wrist).all()), "non-finite camera image")
    check(float(overview.std()) > 0.02 and float(wrist.std()) > 0.02, "blank camera image")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        rows_img = [
            np.concatenate([overview[i].transpose(1, 2, 0), wrist[i].transpose(1, 2, 0)], axis=1)
            for i in range(overview.shape[0])
        ]
        grid = (np.clip(np.concatenate(rows_img, axis=0), 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(grid).save(args.output / "cameras.png")
        report["cameras_png"] = str(args.output / "cameras.png")
    except Exception as error:  # pragma: no cover - PIL missing is not a failure
        report["cameras_png_error"] = str(error)
    report["backend_metadata"] = {
        key: value
        for key, value in backend.metadata().items()
        if key in ("controller_implementation", "physics_substeps_per_action", "simulated_seconds_per_action", "versions", "embodiment")
    }
    report["capacity"] = backend.capacity_status()
    check(report["capacity"]["contact_overflow"] == 0 and report["capacity"]["constraint_overflow"] == 0, "MJWarp contact/constraint capacity overflow")
    report["passed"] = not report["failures"]
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[preflight] {'PASSED' if report['passed'] else 'FAILED'}; report {args.output / 'preflight.json'}", flush=True)
    backend.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
