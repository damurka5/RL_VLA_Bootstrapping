from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


INSTRUCTION_TYPES: tuple[str, ...] = (
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "move_top",
    "move_bottom",
    "move_center",
    "pick_up",
)

MOVE_DIRECTIONS: dict[str, np.ndarray] = {
    "move_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "move_down": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    "move_left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "move_right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "move_top": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "move_bottom": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "move_center": np.zeros((3,), dtype=np.float32),
    "pick_up": np.zeros((3,), dtype=np.float32),
}

_DIRECTIONAL_SUCCESS_AXES: dict[str, tuple[int, float]] = {
    "move_right": (0, 1.0),
    "move_left": (0, -1.0),
    "move_top": (1, 1.0),
    "move_bottom": (1, -1.0),
    "move_up": (2, 1.0),
    "move_down": (2, -1.0),
}

INSTRUCTION_TEXT: dict[str, str] = {
    "move_up": "move up",
    "move_down": "move down",
    "move_left": "move left",
    "move_right": "move right",
    "move_top": "move forward",
    "move_bottom": "move backward",
    "move_center": "move center",
    "pick_up": "pick up object",
}

OBJECT_CENTRIC_INSTRUCTION_TYPES: tuple[str, ...] = ("pick_up",)

_OBJECT_LANGUAGE_ALIASES: dict[str, str] = {
    "ycb_apple": "apple",
    "ycb_pear": "pear",
    "ycb_peach": "peach",
    "ycb_b_cups": "cups",
    "ycb_mug": "mug",
    "ycb_baseball": "baseball",
    "ycb_foam_brick": "foam brick",
    "ycb_ruiks_cube": "rubik's cube",
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
    raw = str(name).strip()
    if not raw:
        return "object"
    if raw in _OBJECT_LANGUAGE_ALIASES:
        return _OBJECT_LANGUAGE_ALIASES[raw]

    cleaned = raw
    for prefix in ("ycb_", "libero_", "obj_"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    cleaned = cleaned.replace("ruiks", "rubiks")
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned or "object"


def instruction_uses_target_object(instruction_type: str) -> bool:
    return str(instruction_type) in OBJECT_CENTRIC_INSTRUCTION_TYPES


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
    instruction_type: str | None = None,
) -> InstructionSpec:
    if allowed_instruction_types is None:
        candidates = list(INSTRUCTION_TYPES)
    else:
        allowed_set = {str(item) for item in allowed_instruction_types}
        candidates = [instruction for instruction in INSTRUCTION_TYPES if instruction in allowed_set]

    if not candidates:
        raise ValueError("allowed_instruction_types removed all instruction types.")

    if instruction_type is None:
        selected_instruction_type = str(candidates[int(rng.integers(0, len(candidates)))])
    else:
        selected_instruction_type = str(instruction_type)
        if selected_instruction_type not in candidates:
            raise ValueError(
                f"Requested instruction_type {selected_instruction_type!r} is not allowed. "
                f"Allowed candidates: {candidates}"
            )

    instruction_text = INSTRUCTION_TEXT[selected_instruction_type]
    target_name = str(target_object or "").strip()
    if instruction_uses_target_object(selected_instruction_type):
        instruction_text = f"pick up {canonical_object_name(target_name)}"
    return InstructionSpec(
        instruction_type=selected_instruction_type,
        text=instruction_text,
        target_object=target_name,
        direction=MOVE_DIRECTIONS[selected_instruction_type].astype(np.float32),
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


def _metadata_float(task_metadata: dict[str, Any] | None, key: str, default: float) -> float:
    if not isinstance(task_metadata, dict) or key not in task_metadata:
        return float(default)
    raw = task_metadata.get(key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Task metadata `{key}` must be numeric, got {raw!r}") from exc


def _action_saturation_stats(
    action: np.ndarray | None,
    *,
    threshold: float,
    exponent: float,
) -> tuple[float, float, float]:
    if action is None:
        return 0.0, 0.0, 0.0

    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.size > 1:
        # Exclude the gripper dimension so deliberate open/close commands are not punished.
        action_arr = action_arr[:-1]
    if action_arr.size == 0:
        return 0.0, 0.0, 0.0

    abs_action = np.abs(action_arr)
    normalized_threshold = float(np.clip(threshold, 0.0, 0.999999))
    denom = max(1e-6, 1.0 - normalized_threshold)
    excess = np.clip((abs_action - normalized_threshold) / denom, 0.0, 1.0)
    penalty_raw = float(np.mean(np.power(excess, float(max(exponent, 1e-6)))))
    saturation_rate = float(np.mean(abs_action >= normalized_threshold))
    saturation_max_abs = float(np.max(abs_action))
    return penalty_raw, saturation_rate, saturation_max_abs


def compute_instruction_reward(
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    obj_pos: np.ndarray,
    reward_state: RewardState,
    action: Optional[np.ndarray] = None,
    camera_alignment: Optional[float] = None,
    goal_direction: Optional[np.ndarray] = None,
    distance_reward_gain: float = 8.0,
    camera_alignment_weight: float = 0.0,
    success_distance: float = 0.03,
    success_camera_alignment: float = 0.60,
    task_metadata: Optional[dict[str, Any]] = None,
    gripper_opening: Optional[float] = None,
    support_surface_z: Optional[float] = None,
    caught_object_is_target: bool = False,
    caught_object_score: float = 0.0,
    caught_object_catalog: str | None = None,
) -> tuple[float, bool, dict[str, float]]:
    ee_pos = np.asarray(ee_pos, dtype=np.float32)
    goal_pos = np.asarray(obj_pos, dtype=np.float32)
    prev_goal_pos = np.asarray(reward_state.prev_obj_pos, dtype=np.float32)

    if spec.instruction_type == "pick_up":
        return _compute_pick_up_reward(
            spec=spec,
            ee_pos=ee_pos,
            obj_pos=goal_pos,
            reward_state=reward_state,
            action=action,
            task_metadata=task_metadata,
            gripper_opening=gripper_opening,
            support_surface_z=support_surface_z,
            caught_object_is_target=caught_object_is_target,
            caught_object_score=caught_object_score,
            caught_object_catalog=caught_object_catalog,
        )

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

    reward_scale_default = max(float(spec.target_displacement), float(spec.lift_target), 1.0 / max(distance_reward_gain, 1e-6))
    distance_reward_scale = max(
        _metadata_float(task_metadata, "distance_reward_scale", reward_scale_default),
        1e-6,
    )
    distance_reward_weight = _metadata_float(task_metadata, "distance_reward_weight", 1.0)
    distance_reward_exponent = _metadata_float(task_metadata, "distance_reward_exponent", 2.0)
    camera_alignment_weight = _metadata_float(task_metadata, "camera_alignment_weight", camera_alignment_weight)
    success_distance = _metadata_float(task_metadata, "success_distance", success_distance)
    success_camera_alignment = _metadata_float(
        task_metadata,
        "success_camera_alignment",
        success_camera_alignment,
    )
    success_bonus = _metadata_float(task_metadata, "success_bonus", 1.0)
    action_saturation_threshold = _metadata_float(task_metadata, "action_saturation_threshold", 0.95)
    action_saturation_penalty_weight = _metadata_float(
        task_metadata,
        "action_saturation_penalty_weight",
        1.0,
    )
    action_saturation_exponent = _metadata_float(task_metadata, "action_saturation_exponent", 2.0)

    normalized_distance = float(distance / distance_reward_scale)
    quadratic_term = float(np.power(normalized_distance, max(distance_reward_exponent, 1e-6)))
    distance_reward = float(distance_reward_weight / (1.0 + quadratic_term))
    camera_reward = float(camera_alignment_weight * camera_align)
    action_saturation_penalty_raw, action_saturation_rate, action_saturation_max_abs = _action_saturation_stats(
        action,
        threshold=action_saturation_threshold,
        exponent=action_saturation_exponent,
    )
    action_saturation_penalty = float(action_saturation_penalty_weight * action_saturation_penalty_raw)

    camera_required = bool(
        camera_alignment_weight > 0.0
        and goal_dir_norm > 1e-8
        and success_camera_alignment > 0.0
    )
    success = bool(
        distance <= float(success_distance)
        and (
            (not camera_required)
            or camera_align >= float(success_camera_alignment)
        )
    )
    success_reward = float(success_bonus if success else 0.0)
    reward = float(distance_reward + camera_reward + success_reward - action_saturation_penalty)

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
        "distance_to_goal_normalized": normalized_distance,
        "distance_reward": distance_reward,
        "distance_reward_scale": float(distance_reward_scale),
        "distance_reward_weight": float(distance_reward_weight),
        "distance_reward_exponent": float(distance_reward_exponent),
        "camera_alignment": camera_align,
        "camera_alignment_delta": camera_alignment_delta,
        "camera_reward": camera_reward,
        "action_saturation_penalty": action_saturation_penalty,
        "action_saturation_penalty_raw": action_saturation_penalty_raw,
        "action_saturation_rate": action_saturation_rate,
        "action_saturation_max_abs": action_saturation_max_abs,
        "action_saturation_threshold": float(action_saturation_threshold),
        "action_saturation_exponent": float(action_saturation_exponent),
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
        "success_bonus": success_reward,
    }
    return reward, success, info


def _compute_pick_up_reward(
    *,
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    obj_pos: np.ndarray,
    reward_state: RewardState,
    action: Optional[np.ndarray] = None,
    task_metadata: Optional[dict[str, Any]] = None,
    gripper_opening: Optional[float] = None,
    support_surface_z: Optional[float] = None,
    caught_object_is_target: bool = False,
    caught_object_score: float = 0.0,
    caught_object_catalog: str | None = None,
) -> tuple[float, bool, dict[str, float]]:
    task_metadata = dict(task_metadata or {})
    prev_ee = np.asarray(reward_state.prev_ee_pos, dtype=np.float32)
    prev_obj = np.asarray(reward_state.prev_obj_pos, dtype=np.float32)

    xy_distance = float(np.linalg.norm(obj_pos[:2] - ee_pos[:2]))
    prev_xy_distance = float(np.linalg.norm(prev_obj[:2] - prev_ee[:2]))
    distance = float(np.linalg.norm(obj_pos - ee_pos))
    prev_distance = float(np.linalg.norm(prev_obj - prev_ee))
    distance_delta = float(prev_distance - distance)

    hover_height = _metadata_float(task_metadata, "pick_hover_height", max(float(spec.lift_target) * 0.8, 0.06))
    grasp_height_offset = _metadata_float(task_metadata, "pick_grasp_height_offset", 0.015)
    xy_reward_scale = max(_metadata_float(task_metadata, "pick_xy_reward_scale", 0.08), 1e-6)
    z_reward_scale = max(_metadata_float(task_metadata, "pick_z_reward_scale", 0.05), 1e-6)
    lift_success_height = max(
        _metadata_float(task_metadata, "pick_lift_success_height", max(float(spec.lift_target), 0.05)),
        1e-6,
    )
    lift_reward_scale = max(_metadata_float(task_metadata, "pick_lift_reward_scale", lift_success_height), 1e-6)
    grasp_xy_threshold = max(_metadata_float(task_metadata, "pick_grasp_xy_threshold", 0.04), 1e-6)
    closed_opening_threshold = _metadata_float(task_metadata, "pick_gripper_closed_opening_threshold", 0.010)
    open_opening_reference = max(
        _metadata_float(task_metadata, "pick_gripper_opening_reference", 0.028),
        closed_opening_threshold + 1e-6,
    )

    approach_weight = _metadata_float(task_metadata, "pick_approach_weight", 0.55)
    open_weight = _metadata_float(task_metadata, "pick_open_weight", 0.20)
    descend_weight = _metadata_float(task_metadata, "pick_descend_weight", 0.35)
    grasp_weight = _metadata_float(task_metadata, "pick_grasp_weight", 0.80)
    lift_weight = _metadata_float(task_metadata, "pick_lift_weight", 1.20)
    caught_target_bonus = _metadata_float(task_metadata, "pick_caught_target_bonus", 0.60)
    wrong_object_penalty_weight = _metadata_float(task_metadata, "pick_wrong_object_penalty_weight", 0.35)
    success_bonus = _metadata_float(task_metadata, "success_bonus", 2.0)
    action_saturation_threshold = _metadata_float(task_metadata, "action_saturation_threshold", 0.95)
    action_saturation_penalty_weight = _metadata_float(
        task_metadata,
        "action_saturation_penalty_weight",
        1.0,
    )
    action_saturation_exponent = _metadata_float(task_metadata, "action_saturation_exponent", 2.0)

    hover_target_z = float(obj_pos[2] + hover_height)
    grasp_target_z = float(obj_pos[2] + grasp_height_offset)
    hover_z_error = float(abs(float(ee_pos[2]) - hover_target_z))
    grasp_z_error = float(abs(float(ee_pos[2]) - grasp_target_z))

    near_xy = float(np.exp(-np.power(xy_distance / xy_reward_scale, 2.0)))
    near_hover = float(np.exp(-np.power(hover_z_error / z_reward_scale, 2.0)))
    near_grasp = float(np.exp(-np.power(grasp_z_error / z_reward_scale, 2.0)))
    grasp_xy_gate = float(np.exp(-np.power(xy_distance / grasp_xy_threshold, 2.0)))
    pregrasp_gate = float(np.exp(-np.power(xy_distance / max(xy_reward_scale * 1.25, 1e-6), 2.0)))

    if gripper_opening is None or not np.isfinite(gripper_opening):
        gripper_is_closed = bool(reward_state.gripper_closed)
        open_fraction = 0.0 if gripper_is_closed else 1.0
    else:
        opening_value = float(gripper_opening)
        gripper_is_closed = bool(opening_value <= closed_opening_threshold)
        open_fraction = float(
            np.clip(
                (opening_value - closed_opening_threshold)
                / max(open_opening_reference - closed_opening_threshold, 1e-6),
                0.0,
                1.0,
            )
        )
    reward_state.gripper_closed = gripper_is_closed
    if not gripper_is_closed:
        reward_state.grasped = False

    approach_reward = float(approach_weight * near_xy * near_hover)
    open_reward = float(open_weight * pregrasp_gate * open_fraction * (1.0 - float(reward_state.grasped)))
    descend_reward = float(
        descend_weight
        * grasp_xy_gate
        * near_grasp
        * open_fraction
        * (1.0 - float(reward_state.grasped))
    )

    contact_score = float(np.exp(-np.power(distance / max(xy_reward_scale, 1e-6), 2.0)))
    effective_caught_score = float(max(float(caught_object_score), contact_score))
    target_caught = bool(caught_object_is_target and gripper_is_closed)
    if target_caught:
        reward_state.grasped = True
    grasped_flag = bool(reward_state.grasped)

    grasp_reward = float(
        grasp_weight
        * effective_caught_score
        * (1.0 if gripper_is_closed else 0.0)
        * max(grasp_xy_gate, near_grasp)
    )
    caught_reward = float(caught_target_bonus if target_caught else 0.0)
    wrong_object_penalty = float(
        wrong_object_penalty_weight * float(caught_object_score)
        if (gripper_is_closed and float(caught_object_score) > 0.0 and not bool(caught_object_is_target))
        else 0.0
    )

    initial_obj_z = float(np.asarray(reward_state.initial_obj_pos, dtype=np.float32)[2])
    target_lift = max(float(obj_pos[2]) - initial_obj_z, 0.0)
    if support_surface_z is not None and np.isfinite(support_surface_z):
        support_height = max(float(support_surface_z), initial_obj_z)
        target_lift = max(target_lift, float(obj_pos[2]) - support_height)
    normalized_lift = float(np.clip(target_lift / lift_reward_scale, 0.0, 1.0))
    lift_reward = float(lift_weight * normalized_lift * (1.0 if grasped_flag else 0.0))

    action_saturation_penalty_raw, action_saturation_rate, action_saturation_max_abs = _action_saturation_stats(
        action,
        threshold=action_saturation_threshold,
        exponent=action_saturation_exponent,
    )
    action_saturation_penalty = float(action_saturation_penalty_weight * action_saturation_penalty_raw)

    success = bool(grasped_flag and target_lift >= lift_success_height)
    success_reward = float(success_bonus if success else 0.0)
    reward = float(
        approach_reward
        + open_reward
        + descend_reward
        + grasp_reward
        + caught_reward
        + lift_reward
        + success_reward
        - wrong_object_penalty
        - action_saturation_penalty
    )

    reward_state.prev_ee_pos = ee_pos.copy()
    reward_state.prev_obj_pos = obj_pos.copy()
    reward_state.prev_distance = distance
    reward_state.prev_camera_align = None
    reward_state.step_count += 1

    info = {
        "distance_to_goal": distance,
        "distance_to_goal_xy": xy_distance,
        "distance_to_goal_prev": prev_distance,
        "distance_to_goal_prev_xy": prev_xy_distance,
        "distance_delta": distance_delta,
        "distance_reward": approach_reward + descend_reward,
        "camera_alignment": 0.0,
        "camera_alignment_delta": 0.0,
        "camera_reward": 0.0,
        "action_saturation_penalty": action_saturation_penalty,
        "action_saturation_penalty_raw": action_saturation_penalty_raw,
        "action_saturation_rate": action_saturation_rate,
        "action_saturation_max_abs": action_saturation_max_abs,
        "action_saturation_threshold": float(action_saturation_threshold),
        "action_saturation_exponent": float(action_saturation_exponent),
        "goal_direction_x": 0.0,
        "goal_direction_y": 0.0,
        "goal_direction_z": 0.0,
        "goal_direction_norm": 0.0,
        "camera_required": 0.0,
        "success_distance_threshold": 0.0,
        "success_camera_alignment_threshold": 0.0,
        "distance_ee_to_object": distance,
        "distance_ee_to_object_xyz": distance,
        "distance_ee_to_object_xy": xy_distance,
        "distance_ee_to_object_prev": prev_distance,
        "distance_ee_to_object_prev_xyz": prev_distance,
        "distance_ee_to_object_prev_xy": prev_xy_distance,
        "orientation_reward": 0.0,
        "success_bonus": success_reward,
        "gripper_closed": float(gripper_is_closed),
        "grasped": float(grasped_flag),
        "pick_hover_height": float(hover_height),
        "pick_hover_z_error": hover_z_error,
        "pick_grasp_height_offset": float(grasp_height_offset),
        "pick_grasp_z_error": grasp_z_error,
        "pick_pregrasp_gate": pregrasp_gate,
        "pick_grasp_xy_gate": grasp_xy_gate,
        "pick_open_fraction": open_fraction,
        "pick_approach_reward": approach_reward,
        "pick_open_reward": open_reward,
        "pick_descend_reward": descend_reward,
        "pick_grasp_reward": grasp_reward,
        "pick_caught_target_reward": caught_reward,
        "pick_wrong_object_penalty": wrong_object_penalty,
        "pick_lift_reward": lift_reward,
        "pick_target_lift": target_lift,
        "pick_target_lift_normalized": normalized_lift,
        "pick_lift_success_height": float(lift_success_height),
        "pick_contact_score": contact_score,
        "caught_object_score": float(caught_object_score),
        "caught_object_is_target": float(bool(caught_object_is_target)),
        "caught_object_catalog_matches_target": float(bool(caught_object_is_target)),
        "caught_object_catalog": 1.0 if (caught_object_catalog or "") else 0.0,
    }
    return reward, success, info


def compute_instruction_validation_success(
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    reward_state: RewardState,
    task_metadata: Optional[dict[str, Any]] = None,
    current_success: bool = False,
) -> tuple[bool, dict[str, float]]:
    ee_arr = np.asarray(ee_pos, dtype=np.float32).reshape(-1)
    start_arr = np.asarray(reward_state.initial_ee_pos, dtype=np.float32).reshape(-1)
    if ee_arr.size < 3 or start_arr.size < 3:
        return bool(current_success), {}

    axis_spec = _DIRECTIONAL_SUCCESS_AXES.get(spec.instruction_type)
    if axis_spec is None:
        return bool(current_success), {
            "validation_success_mode": 0.0,
        }

    axis_idx, axis_sign = axis_spec
    displacement_threshold = _metadata_float(
        task_metadata,
        "directional_success_displacement_threshold",
        0.20,
    )
    raw_displacement = float(ee_arr[axis_idx] - start_arr[axis_idx])
    signed_displacement = float(axis_sign * raw_displacement)
    success = bool(signed_displacement >= float(displacement_threshold))

    return success, {
        "validation_success_mode": 1.0,
        "directional_success_axis": float(axis_idx),
        "directional_success_sign": float(axis_sign),
        "directional_success_raw_displacement": raw_displacement,
        "directional_success_signed_displacement": signed_displacement,
        "directional_success_threshold": float(displacement_threshold),
    }
