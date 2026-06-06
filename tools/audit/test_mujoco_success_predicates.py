#!/usr/bin/env python3
"""Scripted success-predicate smoke tests through the simulator-agnostic API."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tools" / "audit" / "out"
for path in (REPO_ROOT, REPO_ROOT / "robots" / "cdpr"):
    raw = path.as_posix()
    if raw not in sys.path:
        sys.path.insert(0, raw)

from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (  # noqa: E402
    InstructionSpec,
    RewardState,
    compute_instruction_validation_success,
)


class _ScriptedAPI:
    def __init__(self, poses: dict[str, np.ndarray]):
        self.poses = {str(k): np.asarray(v, dtype=np.float32).reshape(3) for k, v in poses.items()}

    def get_body_pose(self, name: str) -> dict[str, np.ndarray]:
        return {
            "position": self.poses[str(name)].copy(),
            "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }


class _ScriptedEnv:
    def __init__(self, poses: dict[str, np.ndarray]):
        self._api = _ScriptedAPI(poses)

    def state_api(self) -> _ScriptedAPI:
        return self._api


def _reward_state(initial_obj=(0.0, 0.0, 0.04), grasped=False) -> RewardState:
    return RewardState(
        initial_ee_pos=np.array([0.0, 0.0, 0.30], dtype=np.float32),
        initial_obj_pos=np.asarray(initial_obj, dtype=np.float32),
        prev_ee_pos=np.array([0.0, 0.0, 0.30], dtype=np.float32),
        prev_obj_pos=np.asarray(initial_obj, dtype=np.float32),
        prev_distance=0.0,
        prev_camera_align=None,
        gripper_closed=False,
        grasped=bool(grasped),
        step_count=0,
    )


def _spec(instruction_type: str) -> InstructionSpec:
    return InstructionSpec(
        instruction_type=instruction_type,
        text=instruction_type,
        target_object="ycb_apple",
        direction=np.zeros(3, dtype=np.float32),
        target_displacement=0.10,
        lift_target=0.05,
        reference_object="plate",
        second_reference_object="bowl",
    )


def _run_case(name: str, instruction_type: str, poses: dict[str, Any], metadata: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    env = _ScriptedEnv({key: np.asarray(value, dtype=np.float32) for key, value in poses.items()})
    success, info = compute_instruction_validation_success(
        spec=_spec(instruction_type),
        ee_pos=np.asarray(kwargs.pop("ee_pos", [0.0, 0.0, 0.30]), dtype=np.float32),
        reward_state=_reward_state(
            kwargs.pop("initial_obj", [0.0, 0.0, 0.04]),
            grasped=bool(kwargs.pop("grasped", False)),
        ),
        task_metadata=metadata,
        current_success=False,
        obj_pos=np.asarray(kwargs.pop("obj_pos", poses.get("target", [0.0, 0.0, 0.04])), dtype=np.float32),
        goal_pos=np.asarray(kwargs.pop("goal_pos", poses.get("target", [0.0, 0.0, 0.04])), dtype=np.float32),
        env=env,
        target_body_name="target",
        reference_body_name="reference",
        second_reference_body_name="second_reference",
        gripper_opening=kwargs.pop("gripper_opening", 1.0),
        caught_object_is_target=bool(kwargs.pop("caught_object_is_target", False)),
        caught_object_score=float(kwargs.pop("caught_object_score", 0.0)),
    )
    return {
        "case": name,
        "instruction_type": instruction_type,
        "success": bool(success),
        "info": info,
    }


def main() -> int:
    metadata = {
        "reward_mode": "sparse_binary",
        "move_to_object_validation_distance_threshold": 0.03,
        "push_success_displacement": 0.08,
        "relation_left_right_offset": 0.08,
        "relation_front_behind_offset": 0.08,
        "move_relation_success_zone_size": 0.05,
        "move_relation_require_target_grasp": False,
        "between_xy_tolerance": 0.04,
        "relation_require_target_grasp": False,
        "relation_min_target_motion": 0.0,
        "put_plate_xy_tolerance": 0.08,
        "put_plate_z_tolerance": 0.10,
        "put_require_release": True,
        "put_release_opening_threshold": 0.55,
        "put_min_target_motion": 0.0,
    }
    cases = [
        _run_case(
            "move_to_object",
            "move_to_object",
            {"target": [0.02, 0.01, 0.04], "reference": [0.2, 0.0, 0.04], "second_reference": [-0.2, 0.0, 0.04]},
            metadata,
            ee_pos=[0.021, 0.011, 0.30],
        ),
        _run_case(
            "push_left",
            "push_left",
            {"target": [-0.09, 0.0, 0.04], "reference": [0.2, 0.0, 0.04], "second_reference": [-0.2, 0.0, 0.04]},
            metadata,
            initial_obj=[0.0, 0.0, 0.04],
        ),
        _run_case(
            "move_left_of_object",
            "move_left_of_object",
            {"target": [0.02, 0.0, 0.04], "reference": [0.10, 0.0, 0.04], "second_reference": [-0.2, 0.0, 0.04]},
            metadata,
            initial_obj=[0.02, 0.0, 0.04],
        ),
        _run_case(
            "move_between_objects",
            "move_between_objects",
            {"target": [0.0, 0.0, 0.04], "reference": [-0.10, 0.0, 0.04], "second_reference": [0.10, 0.0, 0.04]},
            metadata,
            initial_obj=[0.0, 0.0, 0.04],
        ),
        _run_case(
            "put_into_plate",
            "put_into_plate",
            {"target": [0.10, 0.0, 0.04], "reference": [0.10, 0.0, 0.02], "second_reference": [-0.2, 0.0, 0.04]},
            metadata,
            initial_obj=[0.0, 0.0, 0.04],
            grasped=True,
            gripper_opening=0.80,
        ),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "mujoco_success_predicate_smoke.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "instruction_type", "success", "info_json"])
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case": case["case"],
                    "instruction_type": case["instruction_type"],
                    "success": int(case["success"]),
                    "info_json": json.dumps(case["info"], sort_keys=True),
                }
            )
    summary = {
        "csv": str(csv_path),
        "cases": len(cases),
        "successes": sum(1 for case in cases if case["success"]),
        "all_passed": all(case["success"] for case in cases),
    }
    (OUT_DIR / "mujoco_success_predicate_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
