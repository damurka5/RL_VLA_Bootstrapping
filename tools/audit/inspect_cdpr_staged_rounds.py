#!/usr/bin/env python3
"""Explain rejection of saved staged demonstrations without a GPU rollout.

Pass a selection directory or one NPZ. Native placement episodes are expanded
by default; --all-pickups also expands episodes that entered pickup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
    FAILURE_NAMES, STAGE_NAMES, StagedRound,
)


def inspect_round(record, *, all_pickups=False):
    accepted, reasons, counts = record.acceptance()
    per = int(record.actions_per_decision)
    rows = []
    interesting = record.align_event >= 0 if all_pickups else record.placement_event >= 0
    for world in np.flatnonzero(interesting):
        start = (int(record.pickup_event[world]) + 1) * per
        release = np.flatnonzero(record.released[start:, world] & record.active[start:, world])
        stop = start + int(release[0]) if release.size else record.actions.shape[0]
        losses = np.flatnonzero(
            record.active[start:stop, world] & ~record.physical_grasp[start:stop, world]
        ) + start if record.pickup_event[world] >= 0 else np.array([], dtype=int)
        # Show the action causing the first loss alongside the surrounding
        # post-action observations. No inference that a loss is intentional.
        anchor = int(losses[0]) if losses.size else max(0, start - 1)
        trace = []
        for step in range(max(0, anchor - 2), min(record.actions.shape[0], anchor + 4)):
            if record.active[step, world]:
                trace.append({
                    "step": step,
                    "stage": STAGE_NAMES[int(record.step_stage[step, world])],
                    "grasped": bool(record.physical_grasp[step, world]),
                    "released": bool(record.released[step, world]),
                    "opening": round(float(record.gripper_opening[step, world]), 5),
                    "lift_m": round(float(record.target_lift[step, world]), 5),
                    "target_receptacle_xy_m": round(float(np.linalg.norm(
                        record.object_xyz[step, world, 0, :2]
                        - record.object_xyz[step, world, 1, :2])), 5),
                    "placement_geometry_ok": bool(record.placement_geometry_ok[step, world]),
                    "target_xyz": np.round(record.object_xyz[step, world, 0], 5).tolist(),
                    "ee_xyz": np.round(record.ee_xyz[step, world], 5).tolist(),
                    "action": np.round(record.actions[step, world], 5).tolist(),
                })
        rows.append({
            "world": int(world), "scene_uid": str(record.scene_uid[world]),
            "object": str(record.target_catalog[world]),
            "destination": str(record.destination[world]),
            "accepted": bool(accepted[world]), "rejection": str(reasons[world]),
            "final_stage": STAGE_NAMES[int(record.final_stage[world])],
            "failure": FAILURE_NAMES[int(record.failure_code[world])],
            "handoff_lift_m": round(float(record.handoff_lift[world]), 6),
            "events": {name: int(getattr(record, name)[world]) for name in
                       ("reach_event", "align_event", "pickup_event", "placement_event")},
            "first_release_step_after_handoff": int(stop) if release.size else None,
            "carry_loss_steps_before_release": losses[:20].tolist(),
            "carry_loss_step_count": int(losses.size),
            "post_action_trace": trace,
        })
    return {"worlds": record.worlds, "accepted": int(accepted.sum()),
            "rejections": counts, "episodes": rows}


def initial_difference(reference, other):
    """Separate reset differences from prior differences at the first decision."""
    if not np.array_equal(reference.scene_uid, other.scene_uid):
        return {"comparable": False, "reason": "different scene order"}
    arrays = {"reset_ee_m": (reference.reset_ee_xyz, other.reset_ee_xyz),
              "reset_object_m": (reference.reset_object_xyz, other.reset_object_xyz),
              "first_state": (reference.states[0], other.states[0]),
              "first_prior": (reference.priors[0], other.priors[0]),
              "first_teacher_action": (reference.teacher_actions[0], other.teacher_actions[0])}
    return {"comparable": True, "max_absolute_difference": {
        name: float(np.max(np.abs(a - b))) for name, (a, b) in arrays.items()
    }}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--all-pickups", action="store_true")
    args = parser.parse_args(argv)
    paths = sorted(args.path.glob("*.npz")) if args.path.is_dir() else [args.path]
    if not paths:
        parser.error("No saved NPZ rounds found; use the directory from --dump-rounds.")
    reference = reference_path = reference_key = None
    for path in paths:
        record = StagedRound.from_npz(path)
        result = {"file": str(path), **inspect_round(record, all_pickups=args.all_pickups)}
        teacher = json.loads(record.teacher_manifest_json).get("move_to", {})
        key = (teacher.get("sha256", teacher.get("checkpoint")), int(record.round_index))
        if reference is not None and key == reference_key:
            result["initial_comparison"] = {
                "reference": str(reference_path), **initial_difference(reference, record)}
        else:
            reference, reference_path, reference_key = record, path, key
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
