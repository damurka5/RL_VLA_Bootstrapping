from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


INSTRUCTION_TYPES: tuple[str, ...] = (
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "move_top",
    "move_bottom",
    "move_center",
)

MOVE_DIRECTIONS: dict[str, np.ndarray] = {
    "move_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "move_down": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    "move_left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "move_right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "move_top": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "move_bottom": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "move_center": np.zeros((3,), dtype=np.float32),
}

INSTRUCTION_TEXT: dict[str, str] = {
    "move_up": "move up",
    "move_down": "move down",
    "move_left": "move left",
    "move_right": "move right",
    "move_top": "move top",
    "move_bottom": "move bottom",
    "move_center": "move center",
}


@dataclass(frozen=True)
class InstructionSpec:
    instruction_type: str
    text: str
    target_object: str
    direction: np.ndarray
    target_displacement: float
    lift_target: float


@dataclass
class RewardState:
    initial_ee_pos: np.ndarray
    initial_obj_pos: np.ndarray
    prev_ee_pos: np.ndarray
    prev_obj_pos: np.ndarray
    prev_distance: Optional[float] = None
    prev_camera_align: Optional[float] = None
    gripper_closed: bool = False
    grasped: bool = False
    step_count: int = 0


def canonical_object_name(name: str) -> str:
    return str(name).replace("_", " ").strip()


def instruction_type_to_index(instruction_type: str) -> int:
    try:
        return INSTRUCTION_TYPES.index(instruction_type)
    except ValueError as exc:
        raise KeyError(f"Unknown instruction type: {instruction_type}") from exc


def instruction_to_onehot(spec: InstructionSpec) -> np.ndarray:
    out = np.zeros((len(INSTRUCTION_TYPES),), dtype=np.float32)
    out[instruction_type_to_index(spec.instruction_type)] = 1.0
    return out


def sample_instruction(
    target_object: str | None,
    rng: np.random.Generator,
    allowed_instruction_types: Optional[Sequence[str]] = None,
    move_distance: float = 0.40,
    lift_distance: float = 0.10,
) -> InstructionSpec:
    if allowed_instruction_types is None:
        candidates = list(INSTRUCTION_TYPES)
    else:
        allowed_set = {str(item) for item in allowed_instruction_types}
        candidates = [instruction for instruction in INSTRUCTION_TYPES if instruction in allowed_set]

    if not candidates:
        raise ValueError("allowed_instruction_types removed all instruction types.")

    instruction_type = candidates[int(rng.integers(0, len(candidates)))]
    return InstructionSpec(
        instruction_type=instruction_type,
        text=INSTRUCTION_TEXT[instruction_type],
        target_object=str(target_object or ""),
        direction=MOVE_DIRECTIONS[instruction_type].astype(np.float32),
        target_displacement=float(move_distance),
        lift_target=float(lift_distance),
    )


def init_reward_state(initial_ee_pos: np.ndarray, initial_obj_pos: np.ndarray) -> RewardState:
    initial_ee_pos = np.asarray(initial_ee_pos, dtype=np.float32).copy()
    initial_obj_pos = np.asarray(initial_obj_pos, dtype=np.float32).copy()
    return RewardState(
        initial_ee_pos=initial_ee_pos,
        initial_obj_pos=initial_obj_pos,
        prev_ee_pos=initial_ee_pos.copy(),
        prev_obj_pos=initial_obj_pos.copy(),
    )


def _safe_unit(vector: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        padded = np.zeros((3,), dtype=np.float32)
        padded[: arr.size] = arr
        arr = padded
    else:
        arr = arr[:3]
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.zeros((3,), dtype=np.float32), 0.0
    return arr / norm, norm


def compute_instruction_reward(
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    obj_pos: np.ndarray,
    reward_state: RewardState,
    camera_alignment: Optional[float] = None,
    goal_direction: Optional[np.ndarray] = None,
    distance_reward_gain: float = 8.0,
    camera_alignment_weight: float = 0.30,
    success_distance: float = 0.03,
    success_camera_alignment: float = 0.60,
) -> tuple[float, bool, dict[str, float]]:
    del spec

    ee_pos = np.asarray(ee_pos, dtype=np.float32)
    goal_pos = np.asarray(obj_pos, dtype=np.float32)
    prev_goal_pos = np.asarray(reward_state.prev_obj_pos, dtype=np.float32)

    distance_vec = goal_pos - ee_pos
    prev_distance_vec = prev_goal_pos - reward_state.prev_ee_pos
    distance = float(np.linalg.norm(distance_vec))
    prev_distance = float(np.linalg.norm(prev_distance_vec))
    distance_delta = float(prev_distance - distance)

    xy_distance = float(np.linalg.norm(distance_vec[:2]))
    prev_xy_distance = float(np.linalg.norm(prev_distance_vec[:2]))

    goal_dir_unit, goal_dir_norm = _safe_unit(
        goal_direction if goal_direction is not None else (goal_pos - reward_state.initial_ee_pos)
    )
    camera_align = float(np.clip(0.0 if camera_alignment is None else camera_alignment, 0.0, 1.0))
    prev_camera_align = 0.0 if reward_state.prev_camera_align is None else float(reward_state.prev_camera_align)
    camera_alignment_delta = float(camera_align - prev_camera_align)

    distance_reward = float(np.exp(-float(distance_reward_gain) * distance))
    camera_reward = float(camera_alignment_weight * camera_align)
    reward = float(distance_reward + camera_reward)

    camera_required = bool(goal_dir_norm > 1e-8)
    success = bool(
        distance <= float(success_distance)
        and (
            (not camera_required)
            or camera_align >= float(success_camera_alignment)
        )
    )

    reward_state.prev_ee_pos = ee_pos.copy()
    reward_state.prev_obj_pos = goal_pos.copy()
    reward_state.prev_distance = distance
    reward_state.prev_camera_align = camera_align
    reward_state.step_count += 1

    info = {
        "distance_to_goal": distance,
        "distance_to_goal_xy": xy_distance,
        "distance_to_goal_prev": prev_distance,
        "distance_to_goal_prev_xy": prev_xy_distance,
        "distance_delta": distance_delta,
        "distance_reward": distance_reward,
        "camera_alignment": camera_align,
        "camera_alignment_delta": camera_alignment_delta,
        "camera_reward": camera_reward,
        "goal_direction_x": float(goal_dir_unit[0]),
        "goal_direction_y": float(goal_dir_unit[1]),
        "goal_direction_z": float(goal_dir_unit[2]),
        "goal_direction_norm": goal_dir_norm,
        "camera_required": float(camera_required),
        "success_distance_threshold": float(success_distance),
        "success_camera_alignment_threshold": float(success_camera_alignment),
        # Backward-compatible aliases used by some diagnostics/tests.
        "distance_ee_to_object": distance,
        "distance_ee_to_object_xyz": distance,
        "distance_ee_to_object_xy": xy_distance,
        "distance_ee_to_object_prev": prev_distance,
        "distance_ee_to_object_prev_xyz": prev_distance,
        "distance_ee_to_object_prev_xy": prev_xy_distance,
        "orientation_reward": camera_reward,
        "success_bonus": 0.0,
    }
    return reward, success, info
