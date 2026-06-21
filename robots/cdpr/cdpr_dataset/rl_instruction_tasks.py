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
    "move_to_object",
    "pick_up",
    "grab_object",
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_in_front_of_object",
    "move_behind_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
    "push_left",
    "push_right",
    "push_forward",
    "push_backward",
    "catch_object",
    "grip_object",
    "release_object",
    "free_object",
    "open_gripper",
    "close_gripper",
    "rotate_gripper_clockwise",
    "rotate_gripper_counterclockwise",
    "rotate_clockwise",
    "rotate_counterclockwise",
)

MOVE_DIRECTIONS: dict[str, np.ndarray] = {
    "move_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "move_down": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    "move_left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "move_right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "move_top": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "move_bottom": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "move_center": np.zeros((3,), dtype=np.float32),
    "move_to_object": np.zeros((3,), dtype=np.float32),
    "pick_up": np.zeros((3,), dtype=np.float32),
    "grab_object": np.zeros((3,), dtype=np.float32),
    "put_into_plate": np.zeros((3,), dtype=np.float32),
    "move_left_of_object": np.zeros((3,), dtype=np.float32),
    "move_right_of_object": np.zeros((3,), dtype=np.float32),
    "move_in_front_of_object": np.zeros((3,), dtype=np.float32),
    "move_behind_object": np.zeros((3,), dtype=np.float32),
    "put_in_front_of_object": np.zeros((3,), dtype=np.float32),
    "put_behind_object": np.zeros((3,), dtype=np.float32),
    "move_between_objects": np.zeros((3,), dtype=np.float32),
    "push_left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "push_right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "push_forward": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "push_backward": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "catch_object": np.zeros((3,), dtype=np.float32),
    "grip_object": np.zeros((3,), dtype=np.float32),
    "release_object": np.zeros((3,), dtype=np.float32),
    "free_object": np.zeros((3,), dtype=np.float32),
    "open_gripper": np.zeros((3,), dtype=np.float32),
    "close_gripper": np.zeros((3,), dtype=np.float32),
    "rotate_gripper_clockwise": np.zeros((3,), dtype=np.float32),
    "rotate_gripper_counterclockwise": np.zeros((3,), dtype=np.float32),
    "rotate_clockwise": np.zeros((3,), dtype=np.float32),
    "rotate_counterclockwise": np.zeros((3,), dtype=np.float32),
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
    "move_to_object": "move to object",
    "pick_up": "pick up object",
    "grab_object": "grab object",
    "put_into_plate": "put object into plate",
    "move_left_of_object": "move object to the left of object",
    "move_right_of_object": "move object to the right of object",
    "move_in_front_of_object": "move object in front of object",
    "move_behind_object": "move object behind object",
    "put_in_front_of_object": "put object in front of object",
    "put_behind_object": "put object behind object",
    "move_between_objects": "move object between two objects",
    "push_left": "push object left",
    "push_right": "push object right",
    "push_forward": "push object forward",
    "push_backward": "push object backward",
    "catch_object": "catch object",
    "grip_object": "grip object",
    "release_object": "release object",
    "free_object": "free object",
    "open_gripper": "open the gripper",
    "close_gripper": "close the gripper",
    "rotate_gripper_clockwise": "rotate the gripper clockwise",
    "rotate_gripper_counterclockwise": "rotate the gripper counterclockwise",
    "rotate_clockwise": "rotate object clockwise",
    "rotate_counterclockwise": "rotate object counterclockwise",
}

INSTRUCTION_SUCCESS_CRITERIA: dict[str, str] = {
    "move_up": "end effector moves upward by the configured directional threshold",
    "move_down": "end effector moves downward by the configured directional threshold",
    "move_left": "end effector crosses the left workspace-center threshold",
    "move_right": "end effector crosses the right workspace-center threshold",
    "move_top": "end effector crosses the forward workspace-center threshold",
    "move_bottom": "end effector crosses the backward workspace-center threshold",
    "move_center": "falls back to the task point-success predicate",
    "move_to_object": "end-effector XY distance to the target object is within the configured tolerance",
    "pick_up": "target object is grasped and lifted by the configured height",
    "grab_object": "gripper is closed while the target object is detected as caught",
    "put_into_plate": "catchable target object is within the bowl/plate XY/Z tolerance, with release required only when configured",
    "move_left_of_object": "catchable target object is inside the configured left-of-reference success zone",
    "move_right_of_object": "catchable target object is inside the configured right-of-reference success zone",
    "move_in_front_of_object": "catchable target object is inside the configured in-front-of-reference success zone",
    "move_behind_object": "catchable target object is inside the configured behind-reference success zone",
    "put_in_front_of_object": "catchable target object is in front of the reference by the configured offset, aligned in X, moved enough, and grasped when configured",
    "put_behind_object": "catchable target object is behind the reference by the configured offset, aligned in X, moved enough, and grasped when configured",
    "move_between_objects": "catchable target object is near the midpoint between two references, projects onto their segment, moved enough, and grasped when configured",
    "push_left": "target object has moved left by the configured push displacement",
    "push_right": "target object has moved right by the configured push displacement",
    "push_forward": "target object has moved forward by the configured push displacement",
    "push_backward": "target object has moved backward by the configured push displacement",
    "catch_object": "sum of finger-to-object-edge distances is below the configured dense catch threshold",
    "grip_object": "sum of finger-to-object-edge distances is below the configured dense catch threshold",
    "release_object": "sum of finger-to-object-edge clearances is above the configured dense release threshold",
    "free_object": "sum of finger-to-object-edge clearances is above the configured dense release threshold",
    "open_gripper": "normalized gripper opening reaches the configured open threshold",
    "close_gripper": "normalized gripper opening reaches the configured closed threshold",
    "rotate_gripper_clockwise": "end-effector yaw rotates clockwise by the configured angle",
    "rotate_gripper_counterclockwise": "end-effector yaw rotates counterclockwise by the configured angle",
    "rotate_clockwise": "target object yaw has changed clockwise by the configured target angle",
    "rotate_counterclockwise": "target object yaw has changed counterclockwise by the configured target angle",
}

OBJECT_CENTRIC_INSTRUCTION_TYPES: tuple[str, ...] = (
    "move_to_object",
    "pick_up",
    "grab_object",
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_in_front_of_object",
    "move_behind_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
    "push_left",
    "push_right",
    "push_forward",
    "push_backward",
    "catch_object",
    "grip_object",
    "release_object",
    "free_object",
    "rotate_clockwise",
    "rotate_counterclockwise",
)

REFERENCE_OBJECT_INSTRUCTION_TYPES: tuple[str, ...] = (
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_in_front_of_object",
    "move_behind_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
)

MANIPULATION_SPARSE_INSTRUCTION_TYPES: tuple[str, ...] = (
    "grab_object",
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_in_front_of_object",
    "move_behind_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
    "push_left",
    "push_right",
    "push_forward",
    "push_backward",
)

DENSE_GRIPPER_CATCH_INSTRUCTION_TYPES: tuple[str, ...] = (
    "catch_object",
    "grip_object",
)

DENSE_GRIPPER_RELEASE_INSTRUCTION_TYPES: tuple[str, ...] = (
    "release_object",
    "free_object",
)

DENSE_GRIPPER_EDGE_INSTRUCTION_TYPES: tuple[str, ...] = (
    *DENSE_GRIPPER_CATCH_INSTRUCTION_TYPES,
    *DENSE_GRIPPER_RELEASE_INSTRUCTION_TYPES,
)

DIRECT_GRIPPER_INSTRUCTION_TYPES: tuple[str, ...] = (
    "open_gripper",
    "close_gripper",
)

DIRECT_TRANSLATION_INSTRUCTION_TYPES: tuple[str, ...] = (
    "move_left",
    "move_right",
    "move_top",
    "move_bottom",
    "move_up",
    "move_down",
)

DIRECT_GRIPPER_YAW_INSTRUCTION_TYPES: tuple[str, ...] = (
    "rotate_gripper_clockwise",
    "rotate_gripper_counterclockwise",
)

DIRECT_ACTUATOR_INSTRUCTION_TYPES: tuple[str, ...] = (
    *DIRECT_GRIPPER_INSTRUCTION_TYPES,
    *DIRECT_GRIPPER_YAW_INSTRUCTION_TYPES,
)

ROTATE_OBJECT_INSTRUCTION_TYPES: tuple[str, ...] = (
    "rotate_clockwise",
    "rotate_counterclockwise",
)

CATCHABLE_TARGET_INSTRUCTION_TYPES: tuple[str, ...] = (
    "grab_object",
    "pick_up",
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_in_front_of_object",
    "move_behind_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
    "catch_object",
    "grip_object",
    "release_object",
    "free_object",
    "rotate_clockwise",
    "rotate_counterclockwise",
)

PLANAR_RELATION_INSTRUCTION_TYPES: tuple[str, ...] = (
    "move_left_of_object",
    "move_right_of_object",
    "move_in_front_of_object",
    "move_behind_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
)

DEFAULT_CATCHABLE_OBJECTS: tuple[str, ...] = (
    "ycb_apple",
    "ycb_pear",
    "ycb_peach",
    "ycb_baseball",
    "ycb_b_cups",
)

DEFAULT_CONTAINER_OBJECTS: tuple[str, ...] = (
    "plate",
    "bowl",
    "ycb_plate",
    "ycb_bowl",
)

_OBJECT_LANGUAGE_ALIASES: dict[str, str] = {
    "ycb_apple": "apple",
    "ycb_pear": "pear",
    "ycb_peach": "peach",
    "ycb_b_cups": "cup",
    "ycb_mug": "mug",
    "ycb_baseball": "baseball",
    "ycb_plate": "plate",
    "ycb_bowl": "bowl",
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
    reference_object: str = ""
    second_reference_object: str = ""


@dataclass
class RewardState:
    initial_ee_pos: np.ndarray
    initial_obj_pos: np.ndarray
    prev_ee_pos: np.ndarray
    prev_obj_pos: np.ndarray
    prev_distance: Optional[float] = None
    prev_camera_align: Optional[float] = None
    initial_obj_yaw: Optional[float] = None
    prev_obj_yaw: Optional[float] = None
    initial_ee_yaw: Optional[float] = None
    prev_ee_yaw: Optional[float] = None
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


def instruction_uses_reference_object(instruction_type: str) -> bool:
    return str(instruction_type) in REFERENCE_OBJECT_INSTRUCTION_TYPES


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
    reference_object: str | None = None,
    second_reference_object: str | None = None,
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
    reference_name = str(reference_object or "").strip()
    second_reference_name = str(second_reference_object or "").strip()
    target_text = canonical_object_name(target_name)
    reference_text = canonical_object_name(reference_name)
    second_reference_text = canonical_object_name(second_reference_name)
    if selected_instruction_type == "move_to_object":
        instruction_text = f"move to {target_text}"
    elif selected_instruction_type == "pick_up":
        instruction_text = f"pick up {target_text}"
    elif selected_instruction_type == "grab_object":
        instruction_text = f"grab {target_text}"
    elif selected_instruction_type == "catch_object":
        instruction_text = f"catch {target_text}"
    elif selected_instruction_type == "grip_object":
        instruction_text = f"grip {target_text}"
    elif selected_instruction_type == "release_object":
        instruction_text = f"release {target_text}"
    elif selected_instruction_type == "free_object":
        instruction_text = f"free {target_text}"
    elif selected_instruction_type == "rotate_clockwise":
        instruction_text = f"rotate {target_text} clockwise"
    elif selected_instruction_type == "rotate_counterclockwise":
        instruction_text = f"rotate {target_text} counterclockwise"
    elif selected_instruction_type == "put_into_plate":
        plate_text = reference_text if reference_name else "plate"
        instruction_text = f"put {target_text} into {plate_text}"
    elif selected_instruction_type == "move_left_of_object":
        instruction_text = f"move {target_text} to the left of {reference_text}"
    elif selected_instruction_type == "move_right_of_object":
        instruction_text = f"move {target_text} to the right of {reference_text}"
    elif selected_instruction_type == "move_in_front_of_object":
        instruction_text = f"move {target_text} in front of {reference_text}"
    elif selected_instruction_type == "move_behind_object":
        instruction_text = f"move {target_text} behind {reference_text}"
    elif selected_instruction_type == "put_in_front_of_object":
        instruction_text = f"put {target_text} in front of {reference_text}"
    elif selected_instruction_type == "put_behind_object":
        instruction_text = f"put {target_text} behind {reference_text}"
    elif selected_instruction_type == "move_between_objects":
        instruction_text = f"move {target_text} between {reference_text} and {second_reference_text}"
    elif selected_instruction_type == "push_left":
        instruction_text = f"push {target_text} left"
    elif selected_instruction_type == "push_right":
        instruction_text = f"push {target_text} right"
    elif selected_instruction_type == "push_forward":
        instruction_text = f"push {target_text} forward"
    elif selected_instruction_type == "push_backward":
        instruction_text = f"push {target_text} backward"
    return InstructionSpec(
        instruction_type=selected_instruction_type,
        text=instruction_text,
        target_object=target_name,
        direction=MOVE_DIRECTIONS[selected_instruction_type].astype(np.float32),
        target_displacement=float(move_distance),
        lift_target=float(lift_distance),
        reference_object=reference_name,
        second_reference_object=second_reference_name,
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


def _metadata_bool(task_metadata: dict[str, Any] | None, key: str, default: bool) -> bool:
    if not isinstance(task_metadata, dict) or key not in task_metadata:
        return bool(default)
    raw = task_metadata.get(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    if isinstance(raw, (int, float)):
        return bool(raw)
    raise ValueError(f"Task metadata `{key}` must be boolean-like, got {raw!r}")


def _metadata_reward_mode(task_metadata: dict[str, Any] | None) -> str:
    if not isinstance(task_metadata, dict):
        return ""
    for key in ("reward_mode", "reward_type", "sparse_reward_mode"):
        raw = task_metadata.get(key)
        if raw is not None:
            return str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return ""


def _use_sparse_binary_reward(task_metadata: dict[str, Any] | None) -> bool:
    mode = _metadata_reward_mode(task_metadata)
    if mode in {"sparse", "binary", "binary_sparse", "sparse_binary", "sparse_binary_reward"}:
        return True
    try:
        return _metadata_bool(task_metadata, "binary_sparse_reward", False)
    except ValueError:
        return False


def _action_saturation_stats(
    action: np.ndarray | None,
    *,
    threshold: float,
    exponent: float,
    include_gripper: bool = False,
) -> tuple[float, float, float]:
    if action is None:
        return 0.0, 0.0, 0.0

    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.size > 1 and not include_gripper:
        # Exclude the gripper dimension so deliberate open/close commands are not punished.
        action_arr = action_arr[:-1]
    if action_arr.size == 0:
        return 0.0, 0.0, 0.0

    abs_action = np.abs(action_arr)
    normalized_threshold = float(np.clip(threshold, 0.0, 0.999999))
    denom = max(1e-6, 1.0 - normalized_threshold)
    excess = np.clip((abs_action - normalized_threshold) / denom, 0.0, 1.0)
    penalty_raw = float(np.mean(np.power(excess, float(max(exponent, 1e-6)))))
    saturation_rate = float(np.mean(abs_action > normalized_threshold))
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
    env: Any | None = None,
    target_body_name: str | None = None,
    reference_body_name: str | None = None,
    second_reference_body_name: str | None = None,
) -> tuple[float, bool, dict[str, float]]:
    ee_pos = np.asarray(ee_pos, dtype=np.float32)
    goal_pos = np.asarray(obj_pos, dtype=np.float32)
    prev_goal_pos = np.asarray(reward_state.prev_obj_pos, dtype=np.float32)

    if (
        spec.instruction_type in DIRECT_TRANSLATION_INSTRUCTION_TYPES
        and _metadata_bool(task_metadata, "direct_translation_reward_enabled", False)
    ):
        return _compute_direct_translation_reward(
            spec=spec,
            ee_pos=ee_pos,
            goal_pos=goal_pos,
            reward_state=reward_state,
            action=action,
            task_metadata=task_metadata,
        )

    if spec.instruction_type in DIRECT_ACTUATOR_INSTRUCTION_TYPES:
        return _compute_direct_actuator_reward(
            spec=spec,
            ee_pos=ee_pos,
            goal_pos=goal_pos,
            reward_state=reward_state,
            action=action,
            task_metadata=task_metadata,
            gripper_opening=gripper_opening,
            env=env,
        )

    if _use_sparse_binary_reward(task_metadata):
        return _compute_sparse_binary_reward(
            spec=spec,
            ee_pos=ee_pos,
            goal_pos=goal_pos,
            reward_state=reward_state,
            action=action,
            task_metadata=task_metadata,
            gripper_opening=gripper_opening,
            support_surface_z=support_surface_z,
            caught_object_is_target=caught_object_is_target,
            caught_object_score=caught_object_score,
            caught_object_catalog=caught_object_catalog,
            env=env,
            target_body_name=target_body_name,
            reference_body_name=reference_body_name,
            second_reference_body_name=second_reference_body_name,
        )

    if spec.instruction_type in DENSE_GRIPPER_EDGE_INSTRUCTION_TYPES:
        return _compute_dense_gripper_edge_reward(
            spec=spec,
            ee_pos=ee_pos,
            goal_pos=goal_pos,
            reward_state=reward_state,
            task_metadata=task_metadata,
            gripper_opening=gripper_opening,
            env=env,
            target_body_name=target_body_name,
        )

    if spec.instruction_type in ROTATE_OBJECT_INSTRUCTION_TYPES:
        return _compute_dense_rotate_reward(
            spec=spec,
            ee_pos=ee_pos,
            goal_pos=goal_pos,
            reward_state=reward_state,
            action=action,
            task_metadata=task_metadata,
            env=env,
            target_body_name=target_body_name,
        )

    if spec.instruction_type in MANIPULATION_SPARSE_INSTRUCTION_TYPES:
        return _compute_sparse_manipulation_reward(
            spec=spec,
            ee_pos=ee_pos,
            goal_pos=goal_pos,
            reward_state=reward_state,
            action=action,
            task_metadata=task_metadata,
            gripper_opening=gripper_opening,
            support_surface_z=support_surface_z,
            caught_object_is_target=caught_object_is_target,
            caught_object_score=caught_object_score,
            caught_object_catalog=caught_object_catalog,
            env=env,
            target_body_name=target_body_name,
            reference_body_name=reference_body_name,
            second_reference_body_name=second_reference_body_name,
        )
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
    if spec.instruction_type == "move_to_object":
        return _compute_move_to_object_reward(
            spec=spec,
            ee_pos=ee_pos,
            obj_pos=goal_pos,
            reward_state=reward_state,
            action=action,
            task_metadata=task_metadata,
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


def _read_env_body_position(env: Any | None, body_name: str | None) -> np.ndarray | None:
    if env is None or not body_name:
        return None
    api_getter = getattr(env, "state_api", None)
    api = None
    if callable(api_getter):
        try:
            api = api_getter()
        except Exception:
            api = None
    if api is None:
        api = env
    pose_getter = getattr(api, "get_body_pose", None)
    if callable(pose_getter):
        try:
            pose = pose_getter(str(body_name))
            pos = np.asarray(pose.get("position"), dtype=np.float32).reshape(-1)
            if pos.size >= 3 and np.all(np.isfinite(pos[:3])):
                return pos[:3].astype(np.float32)
        except Exception:
            pass
    getter = getattr(env, "_get_task_body_position", None)
    if not callable(getter):
        getter = getattr(env, "_get_body_position", None)
    if not callable(getter):
        return None
    try:
        pos = np.asarray(getter(str(body_name)), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
        return None
    return pos[:3].astype(np.float32)


def _sparse_reward_value(
    *,
    success: bool,
    task_metadata: Optional[dict[str, Any]],
) -> float:
    success_reward = _metadata_float(task_metadata, "sparse_success_reward", 1.0)
    failure_reward = _metadata_float(task_metadata, "sparse_failure_reward", 0.0)
    return float(success_reward if success else failure_reward)


def _force_binary_sparse_metadata(task_metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    metadata = dict(task_metadata or {})
    metadata["action_saturation_penalty_weight"] = 0.0
    return metadata


def _safe_quat_wxyz_to_yaw(quat_wxyz: Sequence[float] | np.ndarray) -> float | None:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if quat.size < 4 or not np.all(np.isfinite(quat[:4])):
        return None
    norm = float(np.linalg.norm(quat[:4]))
    if norm < 1e-9:
        return None
    w, x, y, z = (quat[:4] / norm).tolist()
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _angle_delta(to_angle: float, from_angle: float) -> float:
    return float((float(to_angle) - float(from_angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _read_env_body_yaw(env: Any | None, body_name: str | None) -> float | None:
    if env is not None and body_name:
        api_getter = getattr(env, "state_api", None)
        api = None
        if callable(api_getter):
            try:
                api = api_getter()
            except Exception:
                api = None
        if api is None:
            api = env
        pose_getter = getattr(api, "get_body_pose", None)
        if callable(pose_getter):
            try:
                pose = pose_getter(str(body_name))
            except Exception:
                pose = None
            if isinstance(pose, dict):
                for key in ("quat_wxyz", "quat"):
                    if key in pose:
                        yaw = _safe_quat_wxyz_to_yaw(pose[key])
                        if yaw is not None:
                            return yaw

    if env is not None:
        for attr_name in ("_read_current_yaw", "get_yaw"):
            getter = getattr(env, attr_name, None)
            if callable(getter):
                try:
                    yaw = float(getter())
                except Exception:
                    continue
                if np.isfinite(yaw):
                    return yaw
        raw_yaw = getattr(env, "_yaw", None)
        if raw_yaw is not None:
            try:
                yaw = float(raw_yaw)
            except (TypeError, ValueError):
                yaw = float("nan")
            if np.isfinite(yaw):
                return yaw
    return None


def _finger_edge_metrics(
    *,
    env: Any | None,
    target_body_name: str | None,
    goal_pos: np.ndarray,
    gripper_opening: Optional[float],
    task_metadata: Optional[dict[str, Any]],
) -> dict[str, float]:
    metadata = dict(task_metadata or {})
    if env is not None and target_body_name:
        geometry_getter = getattr(env, "_finger_pair_geometry", None)
        width_getter = getattr(env, "_body_width_along_axis", None)
        if callable(geometry_getter) and callable(width_getter):
            try:
                geometry = geometry_getter()
            except Exception:
                geometry = None
            if isinstance(geometry, dict):
                try:
                    axis = np.asarray(geometry["axis"], dtype=np.float64).reshape(3)
                    center = np.asarray(geometry["center"], dtype=np.float64).reshape(3)
                    inner_gap = float(geometry["inner_gap"])
                except Exception:
                    axis = np.zeros((3,), dtype=np.float64)
                    center = np.zeros((3,), dtype=np.float64)
                    inner_gap = float("nan")
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm > 1e-9 and np.isfinite(inner_gap):
                    axis = axis / axis_norm
                    try:
                        width_raw = width_getter(str(target_body_name), axis.astype(np.float32))
                    except Exception:
                        width_raw = None
                    try:
                        object_width = float(width_raw)
                    except (TypeError, ValueError):
                        object_width = float("nan")
                    target_pos = _read_env_body_position(env, target_body_name)
                    if target_pos is None:
                        target_pos = np.asarray(goal_pos, dtype=np.float32).reshape(-1)[:3]
                    target_pos64 = np.asarray(target_pos, dtype=np.float64).reshape(3)
                    if np.isfinite(object_width) and object_width > 0.0 and np.all(np.isfinite(target_pos64)):
                        rel_center = float(np.dot(target_pos64 - center, axis))
                        half_gap = 0.5 * max(inner_gap, 0.0)
                        half_width = 0.5 * max(object_width, 0.0)
                        positive_clearance = float(half_gap - (rel_center + half_width))
                        negative_clearance = float((rel_center - half_width) + half_gap)
                        edge_distance_sum = float(abs(positive_clearance) + abs(negative_clearance))
                        clearance_sum = float(max(0.0, positive_clearance) + max(0.0, negative_clearance))
                        return {
                            "edge_distance_sum": edge_distance_sum,
                            "clearance_sum": clearance_sum,
                            "min_clearance": float(min(positive_clearance, negative_clearance)),
                            "positive_clearance": positive_clearance,
                            "negative_clearance": negative_clearance,
                            "inner_gap": float(inner_gap),
                            "object_width": float(object_width),
                            "object_center_offset": rel_center,
                            "geometry_available": 1.0,
                        }

    opening = float("nan") if gripper_opening is None else float(gripper_opening)
    if not np.isfinite(opening):
        opening = _metadata_float(metadata, "dense_gripper_fallback_opening", 1.0)
    target_opening = _metadata_float(metadata, "dense_gripper_target_opening", 0.0)
    clearance = max(0.0, float(opening) - float(target_opening))
    return {
        "edge_distance_sum": float(abs(float(opening) - float(target_opening))),
        "clearance_sum": float(clearance),
        "min_clearance": float(clearance),
        "positive_clearance": float(0.5 * clearance),
        "negative_clearance": float(0.5 * clearance),
        "inner_gap": float(opening),
        "object_width": float(target_opening),
        "object_center_offset": 0.0,
        "geometry_available": 0.0,
    }


def _compute_dense_gripper_edge_reward(
    *,
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    reward_state: RewardState,
    task_metadata: Optional[dict[str, Any]] = None,
    gripper_opening: Optional[float] = None,
    env: Any | None = None,
    target_body_name: str | None = None,
) -> tuple[float, bool, dict[str, float]]:
    metadata = dict(task_metadata or {})
    metrics = _finger_edge_metrics(
        env=env,
        target_body_name=target_body_name,
        goal_pos=goal_pos,
        gripper_opening=gripper_opening,
        task_metadata=metadata,
    )
    geometry_available = bool(metrics.get("geometry_available", 0.0) >= 0.5)
    scale_default = 0.015 if geometry_available else 0.10
    reward_scale = max(_metadata_float(metadata, "gripper_edge_reward_scale", scale_default), 1e-6)
    reward_weight = _metadata_float(metadata, "gripper_edge_reward_weight", 1.0)
    exponent = max(_metadata_float(metadata, "gripper_edge_reward_exponent", 2.0), 1e-6)
    success_bonus = _metadata_float(metadata, "gripper_edge_success_bonus", 0.0)

    is_release = spec.instruction_type in DENSE_GRIPPER_RELEASE_INSTRUCTION_TYPES
    if is_release:
        target_clearance = max(
            _metadata_float(
                metadata,
                "release_success_edge_clearance",
                _metadata_float(metadata, "gripper_release_success_clearance", 0.035 if geometry_available else 0.50),
            ),
            1e-6,
        )
        clearance_sum = float(metrics["clearance_sum"])
        remaining = float(max(0.0, target_clearance - clearance_sum))
        shaped = float(reward_weight / (1.0 + np.power(remaining / reward_scale, exponent)))
        success = bool(clearance_sum >= target_clearance)
        dense_error = remaining
        dense_progress = float(np.clip(clearance_sum / target_clearance, 0.0, 1.0))
    else:
        edge_distance_sum = float(metrics["edge_distance_sum"])
        success_threshold = max(
            _metadata_float(
                metadata,
                "gripper_edge_success_threshold",
                _metadata_float(metadata, "catch_success_edge_distance", 0.010 if geometry_available else 0.08),
            ),
            0.0,
        )
        shaped = float(reward_weight / (1.0 + np.power(edge_distance_sum / reward_scale, exponent)))
        success = bool(edge_distance_sum <= success_threshold)
        dense_error = edge_distance_sum
        dense_progress = float(1.0 / (1.0 + np.power(edge_distance_sum / reward_scale, exponent)))

    reward = float(shaped + (success_bonus if success else 0.0))
    reward_state.prev_ee_pos = np.asarray(ee_pos, dtype=np.float32).copy()
    reward_state.prev_obj_pos = np.asarray(goal_pos, dtype=np.float32).copy()
    reward_state.prev_distance = float(dense_error)
    reward_state.prev_camera_align = None
    reward_state.step_count += 1

    info = {
        "dense_gripper_reward_mode": 1.0,
        "dense_gripper_success": float(success),
        "gripper_edge_distance_sum": float(metrics["edge_distance_sum"]),
        "gripper_edge_clearance_sum": float(metrics["clearance_sum"]),
        "gripper_edge_min_clearance": float(metrics["min_clearance"]),
        "gripper_edge_positive_clearance": float(metrics["positive_clearance"]),
        "gripper_edge_negative_clearance": float(metrics["negative_clearance"]),
        "gripper_inner_gap": float(metrics["inner_gap"]),
        "gripper_object_width": float(metrics["object_width"]),
        "gripper_object_center_offset": float(metrics["object_center_offset"]),
        "gripper_edge_geometry_available": float(metrics["geometry_available"]),
        "distance_to_goal": float(dense_error),
        "distance_to_goal_xy": 0.0,
        "distance_delta": 0.0,
        "distance_reward": float(shaped),
        "success_bonus": float(success_bonus if success else 0.0),
        "gripper_edge_reward_scale": float(reward_scale),
        "gripper_edge_reward_weight": float(reward_weight),
        "gripper_edge_reward_exponent": float(exponent),
        "gripper_opening": float("nan") if gripper_opening is None else float(gripper_opening),
        "release_gripper_task": float(is_release),
        "grip_gripper_task": float(not is_release),
        "dense_gripper_progress": float(dense_progress),
        "camera_alignment": 0.0,
        "camera_alignment_delta": 0.0,
        "camera_reward": 0.0,
        "orientation_reward": 0.0,
        "action_saturation_penalty": 0.0,
        "action_saturation_rate": 0.0,
        "action_saturation_max_abs": 0.0,
    }
    if is_release:
        info["release_success_edge_clearance"] = float(target_clearance)
        info["release_remaining_edge_clearance"] = float(dense_error)
    else:
        info["gripper_edge_success_threshold"] = float(success_threshold)
    return reward, success, info


def _compute_dense_rotate_reward(
    *,
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    reward_state: RewardState,
    action: Optional[np.ndarray] = None,
    task_metadata: Optional[dict[str, Any]] = None,
    env: Any | None = None,
    target_body_name: str | None = None,
) -> tuple[float, bool, dict[str, float]]:
    metadata = dict(task_metadata or {})
    current_yaw = _read_env_body_yaw(env, target_body_name)
    if current_yaw is None:
        current_yaw = reward_state.prev_obj_yaw
    if current_yaw is None:
        current_yaw = reward_state.initial_obj_yaw
    if current_yaw is None:
        current_yaw = 0.0
    current_yaw = float(current_yaw)

    if reward_state.initial_obj_yaw is None:
        reward_state.initial_obj_yaw = float(current_yaw)
    if reward_state.prev_obj_yaw is None:
        reward_state.prev_obj_yaw = float(reward_state.initial_obj_yaw)
    prev_yaw = float(reward_state.prev_obj_yaw)

    direction_sign = -1.0 if spec.instruction_type == "rotate_clockwise" else 1.0
    total_signed_rotation = float(direction_sign * _angle_delta(current_yaw, float(reward_state.initial_obj_yaw)))
    step_signed_rotation = float(direction_sign * _angle_delta(current_yaw, prev_yaw))

    target_angle = max(
        _metadata_float(metadata, "rotate_success_angle", _metadata_float(metadata, "rotate_target_angle", 0.30)),
        1e-6,
    )
    step_scale = max(_metadata_float(metadata, "rotate_step_reward_scale", 0.08), 1e-6)
    progress_weight = _metadata_float(metadata, "rotate_progress_weight", 1.0)
    step_weight = _metadata_float(metadata, "rotate_step_weight", 0.25)
    wrong_direction_penalty_weight = _metadata_float(metadata, "rotate_wrong_direction_penalty_weight", 0.35)
    success_bonus = _metadata_float(metadata, "rotate_success_bonus", 0.0)

    progress = float(np.clip(total_signed_rotation / target_angle, 0.0, 1.0))
    correct_step = float(np.tanh(max(0.0, step_signed_rotation) / step_scale))
    wrong_step = float(np.tanh(max(0.0, -step_signed_rotation) / step_scale))
    success = bool(total_signed_rotation >= target_angle)
    reward = float(
        progress_weight * progress
        + step_weight * correct_step
        - wrong_direction_penalty_weight * wrong_step
        + (success_bonus if success else 0.0)
    )

    action_yaw = 0.0
    if action is not None:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size >= 4:
            action_yaw = float(action_arr[3])

    reward_state.prev_ee_pos = np.asarray(ee_pos, dtype=np.float32).copy()
    reward_state.prev_obj_pos = np.asarray(goal_pos, dtype=np.float32).copy()
    reward_state.prev_obj_yaw = float(current_yaw)
    reward_state.prev_distance = float(max(0.0, target_angle - total_signed_rotation))
    reward_state.prev_camera_align = None
    reward_state.step_count += 1

    info = {
        "dense_rotate_reward_mode": 1.0,
        "rotate_success": float(success),
        "rotate_direction_sign": float(direction_sign),
        "rotate_current_yaw": float(current_yaw),
        "rotate_initial_yaw": float(reward_state.initial_obj_yaw),
        "rotate_prev_yaw": float(prev_yaw),
        "rotate_total_signed_angle": float(total_signed_rotation),
        "rotate_step_signed_angle": float(step_signed_rotation),
        "rotate_target_angle": float(target_angle),
        "rotate_progress": float(progress),
        "rotate_correct_step_score": float(correct_step),
        "rotate_wrong_step_score": float(wrong_step),
        "rotate_action_yaw": float(action_yaw),
        "distance_to_goal": float(max(0.0, target_angle - total_signed_rotation)),
        "distance_to_goal_xy": 0.0,
        "distance_delta": float(step_signed_rotation),
        "distance_reward": float(progress_weight * progress + step_weight * correct_step),
        "success_bonus": float(success_bonus if success else 0.0),
        "orientation_reward": float(reward),
        "camera_alignment": 0.0,
        "camera_alignment_delta": 0.0,
        "camera_reward": 0.0,
        "action_saturation_penalty": 0.0,
        "action_saturation_rate": 0.0,
        "action_saturation_max_abs": 0.0,
    }
    return reward, success, info


def _compute_direct_translation_reward(
    *,
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    reward_state: RewardState,
    action: Optional[np.ndarray] = None,
    task_metadata: Optional[dict[str, Any]] = None,
) -> tuple[float, bool, dict[str, float]]:
    metadata = dict(task_metadata or {})
    axis_idx, axis_sign = _DIRECTIONAL_SUCCESS_AXES[spec.instruction_type]
    current = np.asarray(ee_pos, dtype=np.float32).reshape(3)
    initial = np.asarray(reward_state.initial_ee_pos, dtype=np.float32).reshape(3)
    previous = np.asarray(reward_state.prev_ee_pos, dtype=np.float32).reshape(3)
    action_arr = np.asarray(action if action is not None else np.zeros((5,)), dtype=np.float32).reshape(-1)

    target_displacement = max(
        _metadata_float(metadata, "direct_translation_success_displacement", 0.08),
        1e-6,
    )
    orthogonal_tolerance = max(
        _metadata_float(metadata, "direct_translation_orthogonal_tolerance", 0.05),
        0.0,
    )
    total_signed_displacement = float(axis_sign * (current[axis_idx] - initial[axis_idx]))
    step_signed_displacement = float(axis_sign * (current[axis_idx] - previous[axis_idx]))
    displacement = current - initial
    orthogonal = np.delete(displacement, axis_idx)
    orthogonal_drift = float(np.linalg.norm(orthogonal))
    progress = float(np.clip(total_signed_displacement / target_displacement, 0.0, 1.0))

    action_value = float(action_arr[axis_idx]) if action_arr.size > axis_idx else 0.0
    correct_action = float(np.clip(axis_sign * action_value, 0.0, 1.0))
    wrong_action = float(np.clip(-axis_sign * action_value, 0.0, 1.0))
    translation_actions = action_arr[:3] if action_arr.size >= 3 else np.pad(action_arr, (0, 3 - action_arr.size))
    off_axis_action = float(np.mean(np.abs(np.delete(translation_actions, axis_idx))))
    normalized_orthogonal_drift = float(orthogonal_drift / target_displacement)
    success = bool(
        total_signed_displacement >= target_displacement
        and orthogonal_drift <= orthogonal_tolerance
    )
    reward = float(
        _metadata_float(metadata, "direct_translation_progress_weight", 1.0) * progress
        + _metadata_float(metadata, "direct_translation_step_weight", 0.20)
        * np.tanh(
            max(0.0, step_signed_displacement)
            / max(target_displacement * 0.25, 1e-6)
        )
        + _metadata_float(metadata, "direct_translation_action_weight", 0.05) * correct_action
        - _metadata_float(metadata, "direct_translation_wrong_action_penalty", 0.15) * wrong_action
        - _metadata_float(metadata, "direct_translation_off_axis_action_penalty", 0.05)
        * off_axis_action
        - _metadata_float(metadata, "direct_translation_orthogonal_drift_penalty", 0.20)
        * normalized_orthogonal_drift
        + (_metadata_float(metadata, "direct_control_success_bonus", 0.0) if success else 0.0)
    )

    reward_state.prev_ee_pos = current.copy()
    reward_state.prev_obj_pos = np.asarray(goal_pos, dtype=np.float32).copy()
    reward_state.prev_distance = float(max(0.0, target_displacement - total_signed_displacement))
    reward_state.prev_camera_align = None
    reward_state.step_count += 1
    return reward, success, {
        "direct_translation_reward_mode": 1.0,
        "direct_translation_success": float(success),
        "direct_translation_axis": float(axis_idx),
        "direct_translation_sign": float(axis_sign),
        "direct_translation_total_signed_displacement": total_signed_displacement,
        "direct_translation_step_signed_displacement": step_signed_displacement,
        "direct_translation_target_displacement": float(target_displacement),
        "direct_translation_orthogonal_drift": orthogonal_drift,
        "direct_translation_orthogonal_tolerance": float(orthogonal_tolerance),
        "direct_translation_progress": progress,
        "direct_translation_action": action_value,
        "direct_translation_off_axis_action": off_axis_action,
        "distance_to_goal": float(max(0.0, target_displacement - total_signed_displacement)),
        "distance_reward": progress,
        "orientation_reward": 0.0,
        "action_saturation_penalty": 0.0,
        "action_saturation_rate": 0.0,
        "action_saturation_max_abs": float(np.max(np.abs(action_arr))) if action_arr.size else 0.0,
    }


def _compute_direct_actuator_reward(
    *,
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    reward_state: RewardState,
    action: Optional[np.ndarray] = None,
    task_metadata: Optional[dict[str, Any]] = None,
    gripper_opening: Optional[float] = None,
    env: Any | None = None,
) -> tuple[float, bool, dict[str, float]]:
    metadata = dict(task_metadata or {})
    action_arr = np.asarray(action if action is not None else np.zeros((5,)), dtype=np.float32).reshape(-1)
    reward_state.prev_ee_pos = np.asarray(ee_pos, dtype=np.float32).copy()
    reward_state.prev_obj_pos = np.asarray(goal_pos, dtype=np.float32).copy()
    reward_state.prev_camera_align = None
    reward_state.step_count += 1

    if spec.instruction_type in DIRECT_GRIPPER_INSTRUCTION_TYPES:
        opening = float(gripper_opening) if gripper_opening is not None else float("nan")
        if not np.isfinite(opening):
            opening = 0.0
        is_open = spec.instruction_type == "open_gripper"
        open_threshold = float(np.clip(_metadata_float(metadata, "direct_gripper_open_threshold", 0.80), 0.0, 1.0))
        close_threshold = float(
            np.clip(_metadata_float(metadata, "direct_gripper_closed_threshold", 0.20), 0.0, 1.0)
        )
        target = 1.0 if is_open else 0.0
        tolerance = max(1e-6, (1.0 - open_threshold) if is_open else close_threshold)
        error = float(abs(target - opening))
        progress = float(np.clip(1.0 - error, 0.0, 1.0))
        success = bool(opening >= open_threshold) if is_open else bool(opening <= close_threshold)
        action_value = float(action_arr[4]) if action_arr.size >= 5 else 0.0
        direction_sign = 1.0 if is_open else -1.0
        correct_action = float(np.clip(direction_sign * action_value, 0.0, 1.0))
        wrong_action = float(np.clip(-direction_sign * action_value, 0.0, 1.0))
        reward = float(
            _metadata_float(metadata, "direct_gripper_progress_weight", 1.0) * progress
            + _metadata_float(metadata, "direct_gripper_action_weight", 0.10) * correct_action
            - _metadata_float(metadata, "direct_gripper_wrong_action_penalty", 0.10) * wrong_action
            + (_metadata_float(metadata, "direct_actuator_success_bonus", 0.0) if success else 0.0)
        )
        reward_state.prev_distance = error
        reward_state.gripper_closed = bool(opening <= close_threshold)
        return reward, success, {
            "direct_actuator_reward_mode": 1.0,
            "direct_actuator_success": float(success),
            "direct_gripper_opening": float(opening),
            "direct_gripper_target": float(target),
            "direct_gripper_error": float(error),
            "direct_gripper_tolerance": float(tolerance),
            "direct_gripper_progress": float(progress),
            "direct_gripper_action": float(action_value),
            "distance_to_goal": float(error),
            "distance_reward": float(progress),
            "orientation_reward": 0.0,
            "action_saturation_penalty": 0.0,
            "action_saturation_rate": 0.0,
            "action_saturation_max_abs": float(abs(action_value)),
        }

    current_yaw = _read_env_body_yaw(env, None)
    if current_yaw is None:
        current_yaw = reward_state.prev_ee_yaw
    if current_yaw is None:
        current_yaw = reward_state.initial_ee_yaw
    if current_yaw is None:
        current_yaw = 0.0
    current_yaw = float(current_yaw)
    if reward_state.initial_ee_yaw is None:
        reward_state.initial_ee_yaw = current_yaw
    if reward_state.prev_ee_yaw is None:
        reward_state.prev_ee_yaw = float(reward_state.initial_ee_yaw)

    direction_sign = -1.0 if spec.instruction_type == "rotate_gripper_clockwise" else 1.0
    total_signed_rotation = float(
        direction_sign * _angle_delta(current_yaw, float(reward_state.initial_ee_yaw))
    )
    step_signed_rotation = float(
        direction_sign * _angle_delta(current_yaw, float(reward_state.prev_ee_yaw))
    )
    target_angle = max(_metadata_float(metadata, "direct_yaw_success_angle", 0.50), 1e-6)
    progress = float(np.clip(total_signed_rotation / target_angle, 0.0, 1.0))
    success = bool(total_signed_rotation >= target_angle)
    action_value = float(action_arr[3]) if action_arr.size >= 4 else 0.0
    correct_action = float(np.clip(direction_sign * action_value, 0.0, 1.0))
    wrong_action = float(np.clip(-direction_sign * action_value, 0.0, 1.0))
    reward = float(
        _metadata_float(metadata, "direct_yaw_progress_weight", 1.0) * progress
        + _metadata_float(metadata, "direct_yaw_step_weight", 0.20)
        * np.tanh(max(0.0, step_signed_rotation) / max(target_angle * 0.25, 1e-6))
        + _metadata_float(metadata, "direct_yaw_action_weight", 0.05) * correct_action
        - _metadata_float(metadata, "direct_yaw_wrong_action_penalty", 0.15) * wrong_action
        + (_metadata_float(metadata, "direct_actuator_success_bonus", 0.0) if success else 0.0)
    )
    reward_state.prev_ee_yaw = current_yaw
    reward_state.prev_distance = float(max(0.0, target_angle - total_signed_rotation))
    return reward, success, {
        "direct_actuator_reward_mode": 2.0,
        "direct_actuator_success": float(success),
        "direct_yaw_current": float(current_yaw),
        "direct_yaw_initial": float(reward_state.initial_ee_yaw),
        "direct_yaw_total_signed_angle": float(total_signed_rotation),
        "direct_yaw_step_signed_angle": float(step_signed_rotation),
        "direct_yaw_target_angle": float(target_angle),
        "direct_yaw_progress": float(progress),
        "direct_yaw_action": float(action_value),
        "distance_to_goal": float(max(0.0, target_angle - total_signed_rotation)),
        "distance_reward": float(progress),
        "orientation_reward": float(reward),
        "action_saturation_penalty": 0.0,
        "action_saturation_rate": 0.0,
        "action_saturation_max_abs": float(abs(action_value)),
    }


def _compute_sparse_binary_reward(
    *,
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    reward_state: RewardState,
    action: Optional[np.ndarray] = None,
    task_metadata: Optional[dict[str, Any]] = None,
    gripper_opening: Optional[float] = None,
    support_surface_z: Optional[float] = None,
    caught_object_is_target: bool = False,
    caught_object_score: float = 0.0,
    caught_object_catalog: str | None = None,
    env: Any | None = None,
    target_body_name: str | None = None,
    reference_body_name: str | None = None,
    second_reference_body_name: str | None = None,
) -> tuple[float, bool, dict[str, float]]:
    metadata = _force_binary_sparse_metadata(task_metadata)
    if spec.instruction_type in MANIPULATION_SPARSE_INSTRUCTION_TYPES:
        reward, success, info = _compute_sparse_manipulation_reward(
            spec=spec,
            ee_pos=ee_pos,
            goal_pos=goal_pos,
            reward_state=reward_state,
            action=action,
            task_metadata=metadata,
            gripper_opening=gripper_opening,
            support_surface_z=support_surface_z,
            caught_object_is_target=caught_object_is_target,
            caught_object_score=caught_object_score,
            caught_object_catalog=caught_object_catalog,
            env=env,
            target_body_name=target_body_name,
            reference_body_name=reference_body_name,
            second_reference_body_name=second_reference_body_name,
        )
        info["sparse_binary_reward"] = 1.0
        return float(reward), bool(success), info

    if spec.instruction_type == "pick_up":
        return _compute_sparse_pick_up_reward(
            spec=spec,
            ee_pos=ee_pos,
            goal_pos=goal_pos,
            reward_state=reward_state,
            action=action,
            task_metadata=metadata,
            gripper_opening=gripper_opening,
            support_surface_z=support_surface_z,
            caught_object_is_target=caught_object_is_target,
            caught_object_score=caught_object_score,
            caught_object_catalog=caught_object_catalog,
            env=env,
            target_body_name=target_body_name,
        )

    success, validation_info = compute_instruction_validation_success(
        spec=spec,
        ee_pos=ee_pos,
        reward_state=reward_state,
        task_metadata=metadata,
        current_success=False,
        obj_pos=goal_pos,
        goal_pos=goal_pos,
        reward_info=None,
        env=env,
        target_body_name=target_body_name,
        reference_body_name=reference_body_name,
        second_reference_body_name=second_reference_body_name,
        gripper_opening=gripper_opening,
        support_surface_z=support_surface_z,
        caught_object_is_target=caught_object_is_target,
        caught_object_score=caught_object_score,
    )

    prev_goal_pos = np.asarray(reward_state.prev_obj_pos, dtype=np.float32)
    prev_ee_pos = np.asarray(reward_state.prev_ee_pos, dtype=np.float32)
    distance_vec = goal_pos - ee_pos
    prev_distance_vec = prev_goal_pos - prev_ee_pos
    distance = float(np.linalg.norm(distance_vec))
    prev_distance = float(np.linalg.norm(prev_distance_vec))
    xy_distance = float(np.linalg.norm(distance_vec[:2]))
    prev_xy_distance = float(np.linalg.norm(prev_distance_vec[:2]))
    action_saturation_threshold = _metadata_float(metadata, "action_saturation_threshold", 0.95)
    action_saturation_exponent = _metadata_float(metadata, "action_saturation_exponent", 2.0)
    action_saturation_penalty_raw, action_saturation_rate, action_saturation_max_abs = _action_saturation_stats(
        action,
        threshold=action_saturation_threshold,
        exponent=action_saturation_exponent,
    )

    reward = _sparse_reward_value(success=bool(success), task_metadata=metadata)
    reward_state.prev_ee_pos = ee_pos.copy()
    reward_state.prev_obj_pos = goal_pos.copy()
    reward_state.prev_distance = xy_distance if spec.instruction_type == "move_to_object" else distance
    reward_state.prev_camera_align = None
    reward_state.step_count += 1

    info = {
        "sparse_success": float(bool(success)),
        "sparse_reward_mode": 1.0,
        "sparse_binary_reward": 1.0,
        "distance_to_goal": xy_distance if spec.instruction_type == "move_to_object" else distance,
        "distance_to_goal_xy": xy_distance,
        "distance_to_goal_prev": prev_xy_distance if spec.instruction_type == "move_to_object" else prev_distance,
        "distance_to_goal_prev_xy": prev_xy_distance,
        "distance_delta": float(prev_distance - distance),
        "distance_reward": 0.0,
        "camera_alignment": 0.0,
        "camera_alignment_delta": 0.0,
        "camera_reward": 0.0,
        "action_saturation_penalty": 0.0,
        "action_saturation_penalty_raw": action_saturation_penalty_raw,
        "action_saturation_rate": action_saturation_rate,
        "action_saturation_max_abs": action_saturation_max_abs,
        "action_saturation_threshold": float(action_saturation_threshold),
        "action_saturation_exponent": float(action_saturation_exponent),
        "distance_ee_to_object": distance,
        "distance_ee_to_object_xyz": distance,
        "distance_ee_to_object_xy": xy_distance,
        "distance_ee_to_object_prev": prev_distance,
        "distance_ee_to_object_prev_xyz": prev_distance,
        "distance_ee_to_object_prev_xy": prev_xy_distance,
        "orientation_reward": 0.0,
        "success_bonus": float(_metadata_float(metadata, "sparse_success_reward", 1.0) if success else 0.0),
    }
    info.update({str(key): float(value) for key, value in validation_info.items()})
    return float(reward), bool(success), info


def _compute_sparse_pick_up_reward(
    *,
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    reward_state: RewardState,
    action: Optional[np.ndarray] = None,
    task_metadata: Optional[dict[str, Any]] = None,
    gripper_opening: Optional[float] = None,
    support_surface_z: Optional[float] = None,
    caught_object_is_target: bool = False,
    caught_object_score: float = 0.0,
    caught_object_catalog: str | None = None,
    env: Any | None = None,
    target_body_name: str | None = None,
) -> tuple[float, bool, dict[str, float]]:
    metadata = _force_binary_sparse_metadata(task_metadata)
    target_pos = _read_env_body_position(env, target_body_name)
    if target_pos is None:
        target_pos = np.asarray(goal_pos, dtype=np.float32).reshape(-1)[:3]
    ee_distance = float(np.linalg.norm(np.asarray(target_pos[:3] - ee_pos[:3], dtype=np.float32)))
    ee_xy_distance = float(np.linalg.norm(np.asarray(target_pos[:2] - ee_pos[:2], dtype=np.float32)))
    prev_distance = float(np.linalg.norm(np.asarray(reward_state.prev_obj_pos[:3] - reward_state.prev_ee_pos[:3])))
    prev_xy_distance = float(np.linalg.norm(np.asarray(reward_state.prev_obj_pos[:2] - reward_state.prev_ee_pos[:2])))

    closed_threshold = _metadata_float(metadata, "pick_gripper_closed_opening_threshold", 0.010)
    if gripper_opening is None or not np.isfinite(gripper_opening):
        gripper_closed = bool(reward_state.gripper_closed)
    else:
        gripper_closed = bool(float(gripper_opening) <= closed_threshold)
    if not gripper_closed:
        reward_state.grasped = False
    if bool(caught_object_is_target) and gripper_closed:
        reward_state.grasped = True

    initial_obj_z = float(np.asarray(reward_state.initial_obj_pos, dtype=np.float32).reshape(-1)[2])
    support_height = initial_obj_z
    if support_surface_z is not None and np.isfinite(support_surface_z):
        support_height = max(float(support_surface_z), support_height)
    target_lift = max(float(target_pos[2]) - support_height, 0.0)
    lift_success_height = max(
        _metadata_float(metadata, "pick_lift_success_height", max(float(spec.lift_target), 0.05)),
        1e-6,
    )
    normalized_lift = float(np.clip(target_lift / lift_success_height, 0.0, 1.0))
    success = bool(reward_state.grasped and target_lift >= lift_success_height)

    target_delta, target_motion_xy = _target_motion_from_initial(target_pos, reward_state)
    action_saturation_threshold = _metadata_float(metadata, "action_saturation_threshold", 0.95)
    action_saturation_exponent = _metadata_float(metadata, "action_saturation_exponent", 2.0)
    action_saturation_penalty_raw, action_saturation_rate, action_saturation_max_abs = _action_saturation_stats(
        action,
        threshold=action_saturation_threshold,
        exponent=action_saturation_exponent,
    )
    reward = _sparse_reward_value(success=success, task_metadata=metadata)

    reward_state.prev_ee_pos = ee_pos.copy()
    reward_state.prev_obj_pos = target_pos.copy()
    reward_state.prev_distance = ee_distance
    reward_state.prev_camera_align = None
    reward_state.gripper_closed = bool(gripper_closed)
    reward_state.step_count += 1

    info = {
        "sparse_success": float(success),
        "sparse_reward_mode": 8.0,
        "sparse_binary_reward": 1.0,
        "distance_to_goal": ee_distance,
        "distance_to_goal_xy": ee_xy_distance,
        "distance_to_goal_prev": prev_distance,
        "distance_to_goal_prev_xy": prev_xy_distance,
        "distance_delta": float(prev_distance - ee_distance),
        "distance_reward": 0.0,
        "camera_alignment": 0.0,
        "camera_alignment_delta": 0.0,
        "camera_reward": 0.0,
        "action_saturation_penalty": 0.0,
        "action_saturation_penalty_raw": action_saturation_penalty_raw,
        "action_saturation_rate": action_saturation_rate,
        "action_saturation_max_abs": action_saturation_max_abs,
        "action_saturation_threshold": float(action_saturation_threshold),
        "action_saturation_exponent": float(action_saturation_exponent),
        "distance_ee_to_object": ee_distance,
        "distance_ee_to_object_xyz": ee_distance,
        "distance_ee_to_object_xy": ee_xy_distance,
        "distance_ee_to_object_prev": prev_distance,
        "distance_ee_to_object_prev_xyz": prev_distance,
        "distance_ee_to_object_prev_xy": prev_xy_distance,
        "target_motion_x": float(target_delta[0]),
        "target_motion_y": float(target_delta[1]),
        "target_motion_z": float(target_delta[2]),
        "target_motion_xy": float(target_motion_xy),
        "orientation_reward": 0.0,
        "success_bonus": float(_metadata_float(metadata, "sparse_success_reward", 1.0) if success else 0.0),
        "gripper_closed": float(gripper_closed),
        "grasped": float(reward_state.grasped),
        "pick_target_lift": float(target_lift),
        "pick_target_lift_normalized": normalized_lift,
        "pick_lift_success_height": float(lift_success_height),
        "caught_object_score": float(caught_object_score),
        "caught_object_is_target": float(bool(caught_object_is_target)),
        "caught_object_catalog_matches_target": float(bool(caught_object_is_target)),
        "caught_object_catalog": 1.0 if (caught_object_catalog or "") else 0.0,
    }
    return reward, success, info


def _target_motion_from_initial(target_pos: np.ndarray, reward_state: RewardState) -> tuple[np.ndarray, float]:
    initial = np.asarray(reward_state.initial_obj_pos, dtype=np.float32).reshape(-1)
    if initial.size < 3:
        padded = np.zeros((3,), dtype=np.float32)
        padded[: initial.size] = initial
        initial = padded
    delta = np.asarray(target_pos[:3] - initial[:3], dtype=np.float32)
    return delta, float(np.linalg.norm(delta[:2]))


def _compute_sparse_manipulation_reward(
    *,
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    reward_state: RewardState,
    action: Optional[np.ndarray] = None,
    task_metadata: Optional[dict[str, Any]] = None,
    gripper_opening: Optional[float] = None,
    support_surface_z: Optional[float] = None,
    caught_object_is_target: bool = False,
    caught_object_score: float = 0.0,
    caught_object_catalog: str | None = None,
    env: Any | None = None,
    target_body_name: str | None = None,
    reference_body_name: str | None = None,
    second_reference_body_name: str | None = None,
) -> tuple[float, bool, dict[str, float]]:
    task_metadata = dict(task_metadata or {})
    target_pos = _read_env_body_position(env, target_body_name)
    if target_pos is None:
        target_pos = np.asarray(goal_pos, dtype=np.float32).reshape(-1)[:3]
    reference_pos = _read_env_body_position(env, reference_body_name)
    second_reference_pos = _read_env_body_position(env, second_reference_body_name)

    ee_distance = float(np.linalg.norm(np.asarray(target_pos[:3] - ee_pos[:3], dtype=np.float32)))
    ee_xy_distance = float(np.linalg.norm(np.asarray(target_pos[:2] - ee_pos[:2], dtype=np.float32)))
    target_delta, target_motion_xy = _target_motion_from_initial(target_pos, reward_state)
    gripper_value = float("nan") if gripper_opening is None else float(gripper_opening)
    gripper_closed = bool(reward_state.gripper_closed)
    if np.isfinite(gripper_value):
        gripper_closed = bool(gripper_value <= _metadata_float(task_metadata, "grab_closed_opening_threshold", 0.35))
    if not gripper_closed:
        reward_state.grasped = False
    elif bool(caught_object_is_target):
        reward_state.grasped = True

    action_saturation_threshold = _metadata_float(task_metadata, "action_saturation_threshold", 0.95)
    action_saturation_penalty_weight = _metadata_float(
        task_metadata,
        "action_saturation_penalty_weight",
        0.0,
    )
    action_saturation_exponent = _metadata_float(task_metadata, "action_saturation_exponent", 2.0)
    action_saturation_penalty_raw, action_saturation_rate, action_saturation_max_abs = _action_saturation_stats(
        action,
        threshold=action_saturation_threshold,
        exponent=action_saturation_exponent,
    )
    action_saturation_penalty = float(action_saturation_penalty_weight * action_saturation_penalty_raw)

    success = False
    mode = 0.0
    relation_error = float("inf")
    signed_relation_offset = 0.0
    relation_motion_required = 0.0
    relation_motion_ok = True
    relation_grasp_required = False
    relation_grasp_ok = True
    relation_axis = -1
    relation_axis_sign = 0.0
    relation_axis_error = float("inf")
    relation_orthogonal_error = float("inf")
    relation_zone_size = 0.0
    relation_zone_half_extent = 0.0
    relation_left_right_offset = 0.0
    relation_front_behind_offset = 0.0
    push_success_displacement = 0.0

    if spec.instruction_type == "grab_object":
        mode = 3.0
        grab_xy_tolerance = _metadata_float(task_metadata, "grab_xy_tolerance", 0.025)
        require_caught = _metadata_bool(task_metadata, "grab_require_caught", True)
        proximity_success = bool(
            (not require_caught)
            and gripper_closed
            and ee_xy_distance <= float(grab_xy_tolerance)
        )
        success = bool(
            gripper_closed
            and (
                bool(caught_object_is_target)
                or proximity_success
            )
        )
        reward_state.grasped = bool(success or reward_state.grasped)

    elif spec.instruction_type in {"push_left", "push_right", "push_forward", "push_backward"}:
        mode = 4.0
        if spec.instruction_type in {"push_left", "push_right"}:
            push_axis = 0
            sign = -1.0 if spec.instruction_type == "push_left" else 1.0
        else:
            push_axis = 1
            sign = 1.0 if spec.instruction_type == "push_forward" else -1.0
        push_success_displacement = _metadata_float(task_metadata, "push_success_displacement", 0.08)
        signed_motion = float(sign * target_delta[push_axis])
        signed_relation_offset = signed_motion
        relation_axis = int(push_axis)
        relation_axis_sign = float(sign)
        relation_error = float(max(0.0, push_success_displacement - signed_motion))
        success = bool(signed_motion >= float(push_success_displacement))

    elif spec.instruction_type == "put_into_plate":
        mode = 5.0
        plate_pos = reference_pos if reference_pos is not None else goal_pos
        plate_xy_tolerance = _metadata_float(
            task_metadata,
            "put_container_xy_tolerance",
            _metadata_float(task_metadata, "put_plate_xy_tolerance", 0.08),
        )
        plate_z_tolerance = _metadata_float(
            task_metadata,
            "put_container_z_tolerance",
            _metadata_float(task_metadata, "put_plate_z_tolerance", 0.10),
        )
        release_threshold = _metadata_float(task_metadata, "put_release_opening_threshold", 0.55)
        require_release = _metadata_bool(task_metadata, "put_require_release", False)
        relation_motion_required = _metadata_float(task_metadata, "put_min_target_motion", 0.0)
        relation_grasp_required = _metadata_bool(task_metadata, "put_require_target_grasp", False)
        xy_error = float(np.linalg.norm(target_pos[:2] - plate_pos[:2]))
        z_error = float(abs(float(target_pos[2]) - float(plate_pos[2])))
        relation_error = xy_error
        release_ok = bool((not require_release) or (np.isfinite(gripper_value) and gripper_value >= release_threshold))
        relation_motion_ok = bool(target_motion_xy >= relation_motion_required)
        relation_grasp_ok = bool((not relation_grasp_required) or reward_state.grasped)
        success = bool(
            xy_error <= plate_xy_tolerance
            and z_error <= plate_z_tolerance
            and release_ok
            and relation_motion_ok
            and relation_grasp_ok
        )

    elif spec.instruction_type in {
        "move_left_of_object",
        "move_right_of_object",
        "move_in_front_of_object",
        "move_behind_object",
    }:
        mode = 6.0
        ref_pos = reference_pos if reference_pos is not None else goal_pos
        relation_left_right_offset = _metadata_float(task_metadata, "relation_left_right_offset", 0.08)
        relation_front_behind_offset = _metadata_float(
            task_metadata,
            "relation_front_behind_offset",
            relation_left_right_offset,
        )
        if spec.instruction_type in {"move_left_of_object", "move_right_of_object"}:
            axis = 0
            sign = -1.0 if spec.instruction_type == "move_left_of_object" else 1.0
            offset = relation_left_right_offset
        else:
            axis = 1
            sign = -1.0 if spec.instruction_type == "move_in_front_of_object" else 1.0
            offset = relation_front_behind_offset
        orthogonal_axis = 1 - axis
        relation_axis = int(axis)
        relation_axis_sign = float(sign)
        relation_zone_size = max(
            _metadata_float(
                task_metadata,
                "move_relation_success_zone_size",
                _metadata_float(task_metadata, "relation_success_zone_size", 0.05),
            ),
            1e-6,
        )
        relation_zone_half_extent = 0.5 * float(relation_zone_size)
        desired_axis_value = float(ref_pos[axis] + sign * offset)
        desired_orthogonal_value = float(ref_pos[orthogonal_axis])
        signed_relation_offset = float(sign * (target_pos[axis] - ref_pos[axis]))
        relation_axis_error = float(abs(float(target_pos[axis]) - desired_axis_value))
        relation_orthogonal_error = float(abs(float(target_pos[orthogonal_axis]) - desired_orthogonal_value))
        relation_error = float(
            max(0.0, relation_axis_error - relation_zone_half_extent)
            + max(0.0, relation_orthogonal_error - relation_zone_half_extent)
        )
        relation_motion_required = _metadata_float(task_metadata, "move_relation_min_target_motion", 0.0)
        relation_motion_ok = bool(target_motion_xy >= relation_motion_required)
        relation_grasp_required = _metadata_bool(task_metadata, "move_relation_require_target_grasp", False)
        relation_grasp_ok = bool((not relation_grasp_required) or reward_state.grasped)
        success = bool(
            relation_axis_error <= relation_zone_half_extent
            and relation_orthogonal_error <= relation_zone_half_extent
            and relation_motion_ok
            and relation_grasp_ok
        )

    elif spec.instruction_type in {"put_in_front_of_object", "put_behind_object"}:
        mode = 8.0
        ref_pos = reference_pos if reference_pos is not None else goal_pos
        sign = -1.0 if spec.instruction_type == "put_in_front_of_object" else 1.0
        offset = _metadata_float(
            task_metadata,
            "relation_front_behind_offset",
            _metadata_float(task_metadata, "relation_left_right_offset", 0.08),
        )
        relation_front_behind_offset = float(offset)
        x_tolerance = _metadata_float(
            task_metadata,
            "relation_x_tolerance",
            _metadata_float(task_metadata, "relation_y_tolerance", 0.12),
        )
        relation_motion_required = _metadata_float(task_metadata, "relation_min_target_motion", 0.02)
        signed_relation_offset = float(sign * (target_pos[1] - ref_pos[1]))
        x_error = float(abs(float(target_pos[0]) - float(ref_pos[0])))
        relation_axis = 1
        relation_axis_sign = float(sign)
        relation_axis_error = float(max(0.0, offset - signed_relation_offset))
        relation_orthogonal_error = float(x_error)
        relation_error = float(max(0.0, offset - signed_relation_offset) + max(0.0, x_error - x_tolerance))
        relation_motion_ok = bool(target_motion_xy >= relation_motion_required)
        relation_grasp_required = _metadata_bool(task_metadata, "relation_require_target_grasp", True)
        relation_grasp_ok = bool((not relation_grasp_required) or reward_state.grasped)
        success = bool(
            signed_relation_offset >= float(offset)
            and x_error <= float(x_tolerance)
            and relation_motion_ok
            and relation_grasp_ok
        )

    elif spec.instruction_type == "move_between_objects":
        mode = 7.0
        if reference_pos is not None and second_reference_pos is not None:
            midpoint = 0.5 * (reference_pos[:2] + second_reference_pos[:2])
            segment = second_reference_pos[:2] - reference_pos[:2]
            seg_len_sq = float(np.dot(segment, segment))
            if seg_len_sq <= 1e-8:
                projection = 0.5
            else:
                projection = float(np.dot(target_pos[:2] - reference_pos[:2], segment) / seg_len_sq)
            between_tolerance = _metadata_float(task_metadata, "between_xy_tolerance", 0.07)
            relation_motion_required = _metadata_float(task_metadata, "relation_min_target_motion", 0.02)
            relation_error = float(np.linalg.norm(target_pos[:2] - midpoint))
            relation_motion_ok = bool(target_motion_xy >= relation_motion_required)
            relation_grasp_required = _metadata_bool(task_metadata, "relation_require_target_grasp", True)
            relation_grasp_ok = bool((not relation_grasp_required) or reward_state.grasped)
            success = bool(
                relation_error <= float(between_tolerance)
                and 0.0 <= projection <= 1.0
                and relation_motion_ok
                and relation_grasp_ok
            )
            signed_relation_offset = projection
        else:
            success = False

    reward = _sparse_reward_value(success=success, task_metadata=task_metadata) - action_saturation_penalty
    reward_state.prev_ee_pos = ee_pos.copy()
    reward_state.prev_obj_pos = target_pos.copy()
    reward_state.prev_distance = relation_error if np.isfinite(relation_error) else ee_distance
    reward_state.prev_camera_align = None
    reward_state.gripper_closed = bool(gripper_closed)
    reward_state.step_count += 1

    info = {
        "sparse_success": float(success),
        "sparse_reward_mode": float(mode),
        "distance_ee_to_object": ee_distance,
        "distance_ee_to_object_xyz": ee_distance,
        "distance_ee_to_object_xy": ee_xy_distance,
        "target_motion_x": float(target_delta[0]),
        "target_motion_y": float(target_delta[1]),
        "target_motion_z": float(target_delta[2]),
        "target_motion_xy": float(target_motion_xy),
        "relation_error": float(relation_error) if np.isfinite(relation_error) else -1.0,
        "signed_relation_offset": float(signed_relation_offset),
        "relation_axis": float(relation_axis),
        "relation_axis_sign": float(relation_axis_sign),
        "relation_axis_error": float(relation_axis_error) if np.isfinite(relation_axis_error) else -1.0,
        "relation_orthogonal_error": (
            float(relation_orthogonal_error) if np.isfinite(relation_orthogonal_error) else -1.0
        ),
        "relation_zone_size": float(relation_zone_size),
        "relation_zone_half_extent": float(relation_zone_half_extent),
        "relation_left_right_offset": float(relation_left_right_offset),
        "relation_front_behind_offset": float(relation_front_behind_offset),
        "push_success_displacement": float(push_success_displacement),
        "relation_motion_required": float(relation_motion_required),
        "relation_motion_ok": float(relation_motion_ok),
        "relation_grasp_required": float(relation_grasp_required),
        "relation_grasp_ok": float(relation_grasp_ok),
        "gripper_closed": float(gripper_closed),
        "grasped": float(reward_state.grasped),
        "caught_object_score": float(caught_object_score),
        "caught_object_is_target": float(bool(caught_object_is_target)),
        "success_bonus": float(_metadata_float(task_metadata, "sparse_success_reward", 1.0) if success else 0.0),
        "action_saturation_penalty": action_saturation_penalty,
        "action_saturation_penalty_raw": action_saturation_penalty_raw,
        "action_saturation_rate": action_saturation_rate,
        "action_saturation_max_abs": action_saturation_max_abs,
        "action_saturation_threshold": float(action_saturation_threshold),
        "action_saturation_exponent": float(action_saturation_exponent),
    }
    return reward, success, info


def _compute_move_to_object_reward(
    *,
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    obj_pos: np.ndarray,
    reward_state: RewardState,
    action: Optional[np.ndarray] = None,
    task_metadata: Optional[dict[str, Any]] = None,
) -> tuple[float, bool, dict[str, float]]:
    task_metadata = dict(task_metadata or {})
    prev_ee = np.asarray(reward_state.prev_ee_pos, dtype=np.float32)
    prev_obj = np.asarray(reward_state.prev_obj_pos, dtype=np.float32)

    xy_offset = np.asarray(obj_pos[:2] - ee_pos[:2], dtype=np.float32)
    prev_xy_offset = np.asarray(prev_obj[:2] - prev_ee[:2], dtype=np.float32)
    xy_distance = float(np.linalg.norm(xy_offset))
    prev_xy_distance = float(np.linalg.norm(prev_xy_offset))
    xy_distance_delta = float(prev_xy_distance - xy_distance)
    xyz_distance = float(np.linalg.norm(obj_pos - ee_pos))
    prev_xyz_distance = float(np.linalg.norm(prev_obj - prev_ee))

    default_xy_tolerance = _metadata_float(task_metadata, "success_distance", 0.02)
    xy_tolerance = max(
        _metadata_float(task_metadata, "move_to_object_xy_tolerance", default_xy_tolerance),
        1e-6,
    )
    xy_reward_scale = max(
        _metadata_float(task_metadata, "move_to_object_xy_reward_scale", max(4.0 * xy_tolerance, 0.08)),
        1e-6,
    )
    distance_reward_weight = _metadata_float(
        task_metadata,
        "move_to_object_distance_reward_weight",
        _metadata_float(task_metadata, "move_to_object_proximity_weight", 1.0),
    )
    distance_reward_exponent = _metadata_float(task_metadata, "distance_reward_exponent", 2.0)
    z_window_low = _metadata_float(task_metadata, "move_to_object_z_window_low", 0.10)
    z_window_high = _metadata_float(task_metadata, "move_to_object_z_window_high", 0.20)
    z_window_low, z_window_high = sorted((float(z_window_low), float(z_window_high)))
    z_penalty_scale = max(
        _metadata_float(task_metadata, "move_to_object_z_penalty_scale", 0.05),
        1e-6,
    )
    z_penalty_weight = _metadata_float(task_metadata, "move_to_object_z_penalty_weight", 0.20)
    action_saturation_threshold = _metadata_float(task_metadata, "action_saturation_threshold", 0.70)
    action_saturation_penalty_weight = _metadata_float(
        task_metadata,
        "action_saturation_penalty_weight",
        0.20,
    )
    action_saturation_exponent = _metadata_float(task_metadata, "action_saturation_exponent", 1.0)
    action_saturation_include_gripper = _metadata_bool(
        task_metadata,
        "action_saturation_include_gripper",
        True,
    )

    normalized_xy_distance = float(xy_distance / xy_reward_scale)
    distance_reward = float(
        distance_reward_weight
        / (1.0 + np.power(normalized_xy_distance, max(distance_reward_exponent, 1e-6)))
    )
    above_target = bool(xy_distance <= xy_tolerance)
    ee_z = float(ee_pos[2])
    if ee_z < z_window_low:
        z_outside_distance = float(z_window_low - ee_z)
    elif ee_z > z_window_high:
        z_outside_distance = float(ee_z - z_window_high)
    else:
        z_outside_distance = 0.0
    z_in_window = bool(z_outside_distance <= 1e-9)
    z_penalty_raw = float(z_outside_distance / z_penalty_scale)
    z_penalty = float(z_penalty_weight * z_penalty_raw)

    action_saturation_penalty_raw, action_saturation_rate, action_saturation_max_abs = _action_saturation_stats(
        action,
        threshold=action_saturation_threshold,
        exponent=action_saturation_exponent,
        include_gripper=action_saturation_include_gripper,
    )
    action_saturation_penalty = float(action_saturation_penalty_weight * action_saturation_penalty_raw)

    success = bool(above_target and z_in_window)
    reward = float(distance_reward - z_penalty - action_saturation_penalty)

    reward_state.prev_ee_pos = ee_pos.copy()
    reward_state.prev_obj_pos = obj_pos.copy()
    reward_state.prev_distance = xy_distance
    reward_state.prev_camera_align = None
    reward_state.step_count += 1

    goal_dir_unit, goal_dir_norm = _safe_unit(np.array([xy_offset[0], xy_offset[1], 0.0], dtype=np.float32))
    info = {
        "distance_to_goal": xy_distance,
        "distance_to_goal_xy": xy_distance,
        "distance_to_goal_prev": prev_xy_distance,
        "distance_to_goal_prev_xy": prev_xy_distance,
        "distance_delta": xy_distance_delta,
        "distance_to_goal_normalized": normalized_xy_distance,
        "distance_reward": distance_reward,
        "distance_reward_scale": float(xy_reward_scale),
        "distance_reward_weight": float(distance_reward_weight),
        "distance_reward_exponent": float(distance_reward_exponent),
        "camera_alignment": 0.0,
        "camera_alignment_delta": 0.0,
        "camera_reward": 0.0,
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
        "camera_required": 0.0,
        "success_distance_threshold": float(xy_tolerance),
        "success_camera_alignment_threshold": 0.0,
        "distance_ee_to_object": xy_distance,
        "distance_ee_to_object_xyz": xyz_distance,
        "distance_ee_to_object_xy": xy_distance,
        "distance_ee_to_object_prev": prev_xy_distance,
        "distance_ee_to_object_prev_xyz": prev_xyz_distance,
        "distance_ee_to_object_prev_xy": prev_xy_distance,
        "orientation_reward": 0.0,
        "success_bonus": 0.0,
        "move_to_object_progress_reward": 0.0,
        "move_to_object_progress_clip": 0.0,
        "move_to_object_proximity_reward": distance_reward,
        "move_to_object_distance_reward": distance_reward,
        "move_to_object_distance_reward_weight": float(distance_reward_weight),
        "move_to_object_distance_reward_max": float(distance_reward_weight),
        "move_to_object_xy_distance": xy_distance,
        "move_to_object_xy_distance_prev": prev_xy_distance,
        "move_to_object_above_target": float(above_target),
        "move_to_object_above_bonus": 0.0,
        "move_to_object_xy_tolerance": float(xy_tolerance),
        "move_to_object_z": ee_z,
        "move_to_object_z_in_window": float(z_in_window),
        "move_to_object_z_window_low": float(z_window_low),
        "move_to_object_z_window_high": float(z_window_high),
        "move_to_object_z_outside_distance": z_outside_distance,
        "move_to_object_z_penalty_raw": z_penalty_raw,
        "move_to_object_z_penalty": z_penalty,
        "move_to_object_z_penalty_scale": float(z_penalty_scale),
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
    obj_pos: Optional[np.ndarray] = None,
    goal_pos: Optional[np.ndarray] = None,
    reward_info: Optional[dict[str, Any]] = None,
    env: Any | None = None,
    target_body_name: str | None = None,
    reference_body_name: str | None = None,
    second_reference_body_name: str | None = None,
    gripper_opening: Optional[float] = None,
    support_surface_z: Optional[float] = None,
    caught_object_is_target: bool = False,
    caught_object_score: float = 0.0,
) -> tuple[bool, dict[str, float]]:
    ee_arr = np.asarray(ee_pos, dtype=np.float32).reshape(-1)
    start_arr = np.asarray(reward_state.initial_ee_pos, dtype=np.float32).reshape(-1)
    if ee_arr.size < 3 or start_arr.size < 3:
        return bool(current_success), {}

    if (
        spec.instruction_type in DIRECT_TRANSLATION_INSTRUCTION_TYPES
        and _metadata_bool(task_metadata, "direct_translation_reward_enabled", False)
    ):
        axis_idx, axis_sign = _DIRECTIONAL_SUCCESS_AXES[spec.instruction_type]
        threshold = max(
            _metadata_float(
                task_metadata,
                "direct_translation_success_displacement",
                _metadata_float(task_metadata, "directional_success_displacement_threshold", 0.05),
            ),
            1e-6,
        )
        orthogonal_tolerance = max(
            _metadata_float(task_metadata, "direct_translation_orthogonal_tolerance", 0.05),
            0.0,
        )
        raw_displacement = float(ee_arr[axis_idx] - start_arr[axis_idx])
        signed_displacement = float(axis_sign * raw_displacement)
        orthogonal_drift = float(np.linalg.norm(np.delete(ee_arr[:3] - start_arr[:3], axis_idx)))
        direct_success = bool(
            signed_displacement >= threshold
            and orthogonal_drift <= orthogonal_tolerance
        )
        if isinstance(reward_info, dict) and "direct_translation_success" in reward_info:
            direct_success = bool(float(reward_info.get("direct_translation_success", 0.0)) >= 0.5)
        return direct_success, {
            "validation_success_mode": 8.0,
            "direct_translation_validation_success": float(direct_success),
            "direct_translation_axis": float(axis_idx),
            "direct_translation_sign": float(axis_sign),
            "direct_translation_signed_displacement": signed_displacement,
            "direct_translation_threshold": float(threshold),
            "direct_translation_orthogonal_drift": orthogonal_drift,
            "direct_translation_orthogonal_tolerance": float(orthogonal_tolerance),
            "directional_success_raw_displacement": raw_displacement,
            "directional_success_signed_displacement": signed_displacement,
        }

    if spec.instruction_type in DIRECT_ACTUATOR_INSTRUCTION_TYPES:
        direct_success = bool(current_success)
        if isinstance(reward_info, dict) and "direct_actuator_success" in reward_info:
            direct_success = bool(float(reward_info.get("direct_actuator_success", 0.0)) >= 0.5)
        return direct_success, {
            "validation_success_mode": 7.0,
            "direct_actuator_validation_success": float(direct_success),
        }

    if spec.instruction_type in DENSE_GRIPPER_EDGE_INSTRUCTION_TYPES:
        dense_success = bool(current_success)
        if isinstance(reward_info, dict) and "dense_gripper_success" in reward_info:
            dense_success = bool(float(reward_info.get("dense_gripper_success", 0.0)) >= 0.5)
        return dense_success, {
            "validation_success_mode": 5.0,
            "dense_gripper_validation_success": float(dense_success),
        }

    if spec.instruction_type in ROTATE_OBJECT_INSTRUCTION_TYPES:
        rotate_success = bool(current_success)
        if isinstance(reward_info, dict) and "rotate_success" in reward_info:
            rotate_success = bool(float(reward_info.get("rotate_success", 0.0)) >= 0.5)
        return rotate_success, {
            "validation_success_mode": 6.0,
            "rotate_validation_success": float(rotate_success),
        }

    if spec.instruction_type == "pick_up" and _use_sparse_binary_reward(task_metadata):
        reward_state_copy = RewardState(
            initial_ee_pos=np.asarray(reward_state.initial_ee_pos, dtype=np.float32).copy(),
            initial_obj_pos=np.asarray(reward_state.initial_obj_pos, dtype=np.float32).copy(),
            prev_ee_pos=np.asarray(reward_state.prev_ee_pos, dtype=np.float32).copy(),
            prev_obj_pos=np.asarray(reward_state.prev_obj_pos, dtype=np.float32).copy(),
            prev_distance=reward_state.prev_distance,
            prev_camera_align=reward_state.prev_camera_align,
            initial_obj_yaw=reward_state.initial_obj_yaw,
            prev_obj_yaw=reward_state.prev_obj_yaw,
            initial_ee_yaw=reward_state.initial_ee_yaw,
            prev_ee_yaw=reward_state.prev_ee_yaw,
            gripper_closed=bool(reward_state.gripper_closed),
            grasped=bool(reward_state.grasped),
            step_count=int(reward_state.step_count),
        )
        target_source = obj_pos if obj_pos is not None else goal_pos
        if target_source is None:
            target_source = np.zeros((3,), dtype=np.float32)
        _, sparse_success, sparse_info = _compute_sparse_pick_up_reward(
            spec=spec,
            ee_pos=ee_arr[:3],
            goal_pos=np.asarray(target_source, dtype=np.float32).reshape(-1)[:3],
            reward_state=reward_state_copy,
            task_metadata=task_metadata,
            gripper_opening=gripper_opening,
            support_surface_z=support_surface_z,
            caught_object_is_target=caught_object_is_target,
            caught_object_score=caught_object_score,
            env=env,
            target_body_name=target_body_name,
        )
        return sparse_success, {
            "validation_success_mode": 4.0,
            "pick_up_validation_success": float(sparse_success),
            "pick_target_lift": float(sparse_info.get("pick_target_lift", 0.0)),
            "pick_lift_success_height": float(sparse_info.get("pick_lift_success_height", 0.0)),
        }

    if spec.instruction_type in MANIPULATION_SPARSE_INSTRUCTION_TYPES:
        sparse_success = bool(current_success)
        if isinstance(reward_info, dict) and "sparse_success" in reward_info:
            sparse_success = bool(float(reward_info.get("sparse_success", 0.0)) >= 0.5)
        elif not sparse_success:
            reward_state_copy = RewardState(
                initial_ee_pos=np.asarray(reward_state.initial_ee_pos, dtype=np.float32).copy(),
                initial_obj_pos=np.asarray(reward_state.initial_obj_pos, dtype=np.float32).copy(),
                prev_ee_pos=np.asarray(reward_state.prev_ee_pos, dtype=np.float32).copy(),
                prev_obj_pos=np.asarray(reward_state.prev_obj_pos, dtype=np.float32).copy(),
                prev_distance=reward_state.prev_distance,
                prev_camera_align=reward_state.prev_camera_align,
                initial_obj_yaw=reward_state.initial_obj_yaw,
                prev_obj_yaw=reward_state.prev_obj_yaw,
                initial_ee_yaw=reward_state.initial_ee_yaw,
                prev_ee_yaw=reward_state.prev_ee_yaw,
                gripper_closed=bool(reward_state.gripper_closed),
                grasped=bool(reward_state.grasped),
                step_count=int(reward_state.step_count),
            )
            target_source = obj_pos if obj_pos is not None else goal_pos
            if target_source is None:
                target_source = np.zeros((3,), dtype=np.float32)
            _, sparse_success, _ = _compute_sparse_manipulation_reward(
                spec=spec,
                ee_pos=ee_arr[:3],
                goal_pos=np.asarray(target_source, dtype=np.float32).reshape(-1)[:3],
                reward_state=reward_state_copy,
                task_metadata=task_metadata,
                gripper_opening=gripper_opening,
                caught_object_is_target=caught_object_is_target,
                caught_object_score=caught_object_score,
                env=env,
                target_body_name=target_body_name,
                reference_body_name=reference_body_name,
                second_reference_body_name=second_reference_body_name,
            )
        return sparse_success, {
            "validation_success_mode": 3.0,
            "manipulation_validation_success": float(sparse_success),
        }

    if spec.instruction_type == "move_to_object":
        target_source = obj_pos if obj_pos is not None else goal_pos
        if target_source is None:
            success = bool(current_success)
            return success, {
                "validation_success_mode": 2.0,
                "move_to_object_validation_success": float(success),
            }
        target_arr = np.asarray(target_source, dtype=np.float32).reshape(-1)
        if target_arr.size < 3:
            return bool(current_success), {}

        distance_threshold = max(
            _metadata_float(
                task_metadata,
                "move_to_object_validation_distance_threshold",
                _metadata_float(task_metadata, "success_distance", 0.05),
            ),
            1e-6,
        )
        distance_xyz = float(np.linalg.norm(target_arr[:3] - ee_arr[:3]))
        distance_xy = float(np.linalg.norm(target_arr[:2] - ee_arr[:2]))
        success = bool(distance_xy <= float(distance_threshold))
        return success, {
            "validation_success_mode": 2.0,
            "move_to_object_validation_success": float(success),
            "move_to_object_validation_distance_xyz": distance_xyz,
            "move_to_object_validation_distance_xy": distance_xy,
            "move_to_object_validation_distance_threshold": float(distance_threshold),
        }

    axis_spec = _DIRECTIONAL_SUCCESS_AXES.get(spec.instruction_type)
    if axis_spec is None:
        return bool(current_success), {
            "validation_success_mode": 0.0,
        }

    axis_idx, axis_sign = axis_spec
    threshold = _metadata_float(
        task_metadata,
        "directional_success_center_threshold",
        _metadata_float(
            task_metadata,
            "directional_success_displacement_threshold",
            0.05,
        ),
    )
    use_workspace_center = bool(
        axis_idx in (0, 1)
        and isinstance(task_metadata, dict)
        and (
            "goal_center_xy" in task_metadata
            or "directional_success_center_threshold" in task_metadata
        )
    )
    if use_workspace_center:
        default_goal_center_xy = (0.0, 0.0)
        raw_center_xy = (
            task_metadata.get("goal_center_xy", default_goal_center_xy)
            if isinstance(task_metadata, dict)
            else default_goal_center_xy
        )
        center_xy = np.asarray(raw_center_xy, dtype=np.float32).reshape(-1)
        if center_xy.size < 2:
            padded = np.zeros((2,), dtype=np.float32)
            padded[: center_xy.size] = center_xy
            center_xy = padded
        reference_value = float(center_xy[axis_idx])
    else:
        reference_value = float(start_arr[axis_idx])

    raw_displacement = float(ee_arr[axis_idx] - reference_value)
    signed_displacement = float(axis_sign * raw_displacement)
    success = bool(signed_displacement >= float(threshold))

    info: dict[str, float] = {
        "validation_success_mode": 1.0,
        "directional_success_axis": float(axis_idx),
        "directional_success_sign": float(axis_sign),
        "directional_success_reference_value": reference_value,
        "directional_success_raw_displacement": raw_displacement,
        "directional_success_signed_displacement": signed_displacement,
        "directional_success_threshold": float(threshold),
    }
    if axis_idx in (0, 1):
        info["directional_success_reference_is_workspace_center"] = float(use_workspace_center)
    else:
        info["directional_success_reference_is_workspace_center"] = 0.0
    return success, info
