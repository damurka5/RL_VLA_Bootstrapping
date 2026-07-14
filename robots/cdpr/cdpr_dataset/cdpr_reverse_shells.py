from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rl_vla_bootstrapping.lchol.reverse_shells import ReverseShellReset, clamp_shell_id

from .rl_instruction_tasks import compute_instruction_validation_success
from .synthetic_tasks import clamp_xyz


_SHELL_COUNTS: dict[str, int] = {
    "move_to_object": 4,
    "grab_object": 5,
    "pick_up": 5,
    "put_into_plate": 6,
    "push_left": 5,
    "push_right": 5,
    "push_forward": 5,
    "push_backward": 5,
    "move_left_of_object": 5,
    "move_right_of_object": 5,
    "move_in_front_of_object": 5,
    "move_behind_object": 5,
    "put_in_front_of_object": 5,
    "put_behind_object": 5,
    "move_between_objects": 5,
}

SMOLVLA_COMPLEX_PROFILE = "smolvla_complex_v1"
_SMOLVLA_COMPLEX_SHELL_COUNTS: dict[str, int] = {
    "move_to_object": 4,
    "push_left": 4,
    "push_right": 4,
    "put_into_plate": 4,
    "put_into_bowl": 4,
    "move_left_of_object": 4,
    "move_right_of_object": 4,
    "move_between_objects": 4,
}
SMOLVLA_COMPLEX_POLICY_DECISION_BOUNDS: tuple[tuple[int, int], ...] = (
    (2, 3),
    (3, 4),
    (4, 6),
    (6, 10),
)

_TEMPLATES: dict[str, str] = {
    "move_to_object": "move to <object>",
    "grab_object": "grab <object>",
    "pick_up": "pick up <object>",
    "put_into_plate": "put <object> into <receptacle>",
    "put_into_bowl": "put <object> into <receptacle>",
    "push_left": "push <object> left",
    "push_right": "push <object> right",
    "push_forward": "push <object> forward",
    "push_backward": "push <object> backward",
    "move_left_of_object": "move <object> left of <reference>",
    "move_right_of_object": "move <object> right of <reference>",
    "move_in_front_of_object": "move <object> in front of <reference>",
    "move_behind_object": "move <object> behind <reference>",
    "put_in_front_of_object": "put <object> in front of <reference>",
    "put_behind_object": "put <object> behind <reference>",
    "move_between_objects": "move <object> between <ref1> and <ref2>",
}

_DEFAULT_HELD_OBJECT_OFFSET = np.array([0.0, 0.0, 0.005], dtype=np.float32)
_YCB_CAUGHT_OBJECT_MEASUREMENTS: dict[str, dict[str, float]] = {
    "ycb_apple": {"width_m": 0.0751, "opening": 0.8852, "finger_qpos_m": 0.0266},
    "apple": {"width_m": 0.0751, "opening": 0.8852, "finger_qpos_m": 0.0266},
    "ycb_pear": {"width_m": 0.0662, "opening": 0.7369, "finger_qpos_m": 0.0221},
    "pear": {"width_m": 0.0662, "opening": 0.7369, "finger_qpos_m": 0.0221},
    "ycb_peach": {"width_m": 0.0591, "opening": 0.6186, "finger_qpos_m": 0.0186},
    "peach": {"width_m": 0.0591, "opening": 0.6186, "finger_qpos_m": 0.0186},
    "ycb_baseball": {"width_m": 0.0720, "opening": 0.8337, "finger_qpos_m": 0.0250},
    "baseball": {"width_m": 0.0720, "opening": 0.8337, "finger_qpos_m": 0.0250},
}


@dataclass(frozen=True)
class CDPRReverseShellSpec:
    instruction_id: str
    instruction_template: str
    shell_count: int
    profile: str = "legacy"

    def sample_scene(self, rng: Any) -> Mapping[str, Any]:
        del rng
        return {}

    def sample_reset(self, shell_id: int, scene: Any, rng: Any, **kwargs: Any) -> ReverseShellReset:
        env = kwargs.get("env")
        if env is None:
            return ReverseShellReset(
                instruction_id=self.instruction_id,
                shell_id=clamp_shell_id(shell_id, self.shell_count),
                metadata={"scene": scene},
            )
        metadata = apply_cdpr_reverse_shell(env, shell_id=shell_id, rng=rng, profile=self.profile)
        return ReverseShellReset(
            instruction_id=self.instruction_id,
            shell_id=int(metadata["curriculum_shell"]),
            metadata=metadata,
        )

    def success(self, state: Any, instruction_binding: Mapping[str, Any]) -> bool:
        if isinstance(state, Mapping) and "success" in state:
            return bool(state["success"])
        if isinstance(state, Mapping) and "sparse_success" in state:
            return bool(float(state["sparse_success"]) >= 0.5)
        env = instruction_binding.get("env") if isinstance(instruction_binding, Mapping) else None
        if env is None:
            return False
        try:
            success, _ = compute_instruction_validation_success(
                spec=env._instruction_spec,
                ee_pos=env._get_ee_position(),
                reward_state=env._reward_state,
                task_metadata=env._task_metadata,
                current_success=False,
                obj_pos=env._current_target_reference_position(),
                goal_pos=env._goal_position,
                env=env,
                target_body_name=env._target_body_name,
                reference_body_name=env._reference_body_name,
                second_reference_body_name=env._second_reference_body_name,
                gripper_opening=env._get_gripper_opening(),
                support_surface_z=env._support_surface_z,
            )
        except Exception:
            return False
        return bool(success)


def get_cdpr_reverse_shell_specs(
    instruction_types: Sequence[str] | None = None,
    *,
    profile: str = "legacy",
) -> tuple[CDPRReverseShellSpec, ...]:
    profile = str(profile or "legacy")
    shell_counts = _SMOLVLA_COMPLEX_SHELL_COUNTS if profile == SMOLVLA_COMPLEX_PROFILE else _SHELL_COUNTS
    if instruction_types is None:
        names = tuple(shell_counts)
    else:
        names = tuple(str(item) for item in instruction_types if str(item) in shell_counts)
    return tuple(
        CDPRReverseShellSpec(
            instruction_id=name,
            instruction_template=_TEMPLATES[name],
            shell_count=int(shell_counts[name]),
            profile=profile,
        )
        for name in names
    )


def apply_cdpr_reverse_shell(
    env: Any,
    *,
    shell_id: int,
    rng: np.random.Generator | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    rng = rng or getattr(env, "np_random", np.random.default_rng())
    spec = getattr(env, "_instruction_spec", None)
    instruction_type = str(getattr(spec, "instruction_type", ""))
    metadata = getattr(env, "_task_metadata", {}) or {}
    profile = str(profile or metadata.get("reverse_frontier_profile") or "legacy")
    shell_counts = _SMOLVLA_COMPLEX_SHELL_COUNTS if profile == SMOLVLA_COMPLEX_PROFILE else _SHELL_COUNTS
    shell_count = int(shell_counts.get(instruction_type, 1))
    shell_id = clamp_shell_id(int(shell_id), shell_count)
    info: dict[str, Any] = {
        "curriculum_mode": "reverse_frontier",
        "curriculum_shell": int(shell_id),
        "curriculum_shell_count": int(shell_count),
        "curriculum_instruction_id": instruction_type,
        "curriculum_shell_source": "reverse_frontier",
        "curriculum_shell_normal_reset": bool(
            profile != SMOLVLA_COMPLEX_PROFILE and shell_id >= shell_count - 1
        ),
        "curriculum_shell_profile": profile,
        "curriculum_target_grasped": False,
    }
    if profile == SMOLVLA_COMPLEX_PROFILE:
        horizon = _shell_horizon_metadata(env, shell_id=shell_id, rng=rng)
        info.update(horizon)
        info.update(
            _apply_smolvla_complex_shell(
                env,
                shell_id=shell_id,
                target_policy_steps=int(horizon["curriculum_shell_target_policy_steps"]),
                rng=rng,
            )
        )
        return info
    if not instruction_type or shell_id >= shell_count - 1:
        return info

    if instruction_type == "move_to_object":
        info.update(_apply_move_to_object_shell(env, shell_id=shell_id, rng=rng))
    elif instruction_type in {"grab_object", "pick_up"}:
        info.update(_apply_grab_shell(env, shell_id=shell_id, rng=rng))
    elif instruction_type == "put_into_plate":
        info.update(_apply_put_shell(env, shell_id=shell_id, rng=rng))
    elif instruction_type in {"push_left", "push_right", "push_forward", "push_backward"}:
        info.update(_apply_push_shell(env, shell_id=shell_id, rng=rng))
    elif instruction_type in {
        "move_left_of_object",
        "move_right_of_object",
        "move_in_front_of_object",
        "move_behind_object",
        "put_in_front_of_object",
        "put_behind_object",
    }:
        info.update(_apply_binary_relation_shell(env, shell_id=shell_id, rng=rng))
    elif instruction_type == "move_between_objects":
        info.update(_apply_between_shell(env, shell_id=shell_id, rng=rng))
    return info


def _shell_horizon_metadata(
    env: Any,
    *,
    shell_id: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    bounds = _policy_decision_bounds(env)
    low, high = bounds[clamp_shell_id(shell_id, len(bounds))]
    target_policy_steps = int(rng.integers(int(low), int(high) + 1))
    simulator_steps_per_action = max(1, 1 + int(getattr(env, "hold_steps", 0)))
    target_sim_steps = int(target_policy_steps * simulator_steps_per_action)
    return {
        "curriculum_shell_policy_steps_low": int(low),
        "curriculum_shell_policy_steps_high": int(high),
        "curriculum_shell_sim_steps_low": int(low * simulator_steps_per_action),
        "curriculum_shell_sim_steps_high": int(high * simulator_steps_per_action),
        "curriculum_shell_target_sim_steps": int(target_sim_steps),
        "curriculum_shell_target_policy_steps": int(target_policy_steps),
        "curriculum_sim_steps_per_policy_action": int(simulator_steps_per_action),
        "curriculum_shell_nominal_progress_fraction": float(
            np.clip(
                _metadata_float(env, "reverse_frontier_nominal_progress_fraction", 0.44),
                0.05,
                1.0,
            )
        ),
    }


def _apply_smolvla_complex_shell(
    env: Any,
    *,
    shell_id: int,
    target_policy_steps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    instruction_type = str(getattr(getattr(env, "_instruction_spec", None), "instruction_type", ""))
    if instruction_type == "move_to_object":
        return _apply_complex_move_to_shell(
            env, shell_id=shell_id, target_policy_steps=target_policy_steps, rng=rng
        )
    if instruction_type in {"push_left", "push_right"}:
        return _apply_complex_push_shell(
            env, shell_id=shell_id, target_policy_steps=target_policy_steps, rng=rng
        )
    if instruction_type in {"put_into_plate", "put_into_bowl"}:
        return _apply_complex_put_shell(
            env, shell_id=shell_id, target_policy_steps=target_policy_steps, rng=rng
        )
    if instruction_type in {"move_left_of_object", "move_right_of_object"}:
        return _apply_complex_binary_relation_shell(
            env, shell_id=shell_id, target_policy_steps=target_policy_steps, rng=rng
        )
    if instruction_type == "move_between_objects":
        return _apply_complex_between_shell(
            env, shell_id=shell_id, target_policy_steps=target_policy_steps, rng=rng
        )
    return {}


def _policy_decision_bounds(env: Any) -> tuple[tuple[int, int], ...]:
    raw = (getattr(env, "_task_metadata", {}) or {}).get(
        "reverse_frontier_policy_decision_bounds"
    )
    if isinstance(raw, (list, tuple)) and len(raw) == len(
        SMOLVLA_COMPLEX_POLICY_DECISION_BOUNDS
    ):
        parsed: list[tuple[int, int]] = []
        try:
            for item in raw:
                lo, hi = int(item[0]), int(item[1])
                parsed.append((max(1, min(lo, hi)), max(1, max(lo, hi))))
        except (TypeError, ValueError, IndexError):
            parsed = []
        if len(parsed) == len(SMOLVLA_COMPLEX_POLICY_DECISION_BOUNDS):
            return tuple(parsed)
    return SMOLVLA_COMPLEX_POLICY_DECISION_BOUNDS


def _policy_travel_distance(env: Any, target_policy_steps: int) -> float:
    step = max(0.005, float(getattr(env, "action_step_xyz", 0.015)))
    progress_fraction = float(
        np.clip(
            _metadata_float(env, "reverse_frontier_nominal_progress_fraction", 0.44),
            0.05,
            1.0,
        )
    )
    # Half an action of margin makes an ideal max-magnitude controller enter
    # the success boundary on the requested decision rather than one earlier.
    return float(max(0.5, float(target_policy_steps) - 0.5) * step * progress_fraction)


def _direction_with_room(
    env: Any,
    *,
    center: np.ndarray,
    distance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    metadata = getattr(env, "_task_metadata", {}) or {}
    x_bounds = metadata.get("ee_workspace_x_bounds", (-0.28, 0.28))
    y_bounds = metadata.get("ee_workspace_y_bounds", (-0.28, 0.28))
    lo = np.array([min(float(x_bounds[0]), float(x_bounds[1])), min(float(y_bounds[0]), float(y_bounds[1]))])
    hi = np.array([max(float(x_bounds[0]), float(x_bounds[1])), max(float(y_bounds[0]), float(y_bounds[1]))])
    center_xy = np.asarray(center, dtype=np.float32).reshape(3)[:2]
    fallback = np.array([1.0, 0.0], dtype=np.float32)
    best_margin = float("-inf")
    for _ in range(32):
        direction = _unit_xy(rng)
        endpoint = center_xy + direction * float(distance)
        margin = float(np.min(np.minimum(endpoint - lo, hi - endpoint)))
        if margin >= 0.005:
            return direction
        if margin > best_margin:
            best_margin = margin
            fallback = direction
    inward = -center_xy
    if float(np.linalg.norm(inward)) > 1e-8:
        return (inward / float(np.linalg.norm(inward))).astype(np.float32)
    return fallback


def _apply_complex_move_to_shell(
    env: Any, *, shell_id: int, target_policy_steps: int, rng: np.random.Generator
) -> dict[str, Any]:
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if target is None:
        return {}
    success_boundary = max(
        0.0,
        _metadata_float(
            env,
            "move_to_object_validation_distance_threshold",
            _metadata_float(env, "move_to_object_xy_window_high", 0.03),
        ),
    )
    travel = _policy_travel_distance(env, target_policy_steps)
    distance = float(success_boundary + travel)
    direction = _direction_with_room(env, center=target, distance=distance, rng=rng)
    ee = np.asarray(target, dtype=np.float32).copy()
    ee[:2] += direction * distance
    # The strict move-to task is planar. Every shell begins at a valid fixed
    # working height, and success rejects vertical excursions from this reset.
    ee[2] = float(getattr(env, "_get_ee_position")()[2])
    _move_ee(env, ee)
    return {
        "curriculum_shell_distance_m": float(distance),
        "curriculum_shell_success_boundary_m": float(success_boundary),
        "curriculum_shell_required_travel_m": float(travel),
        "curriculum_shell_relation": "strict_planar_ee_to_target",
        "curriculum_put_start_stage": -1,
    }


def _apply_complex_push_shell(
    env: Any, *, shell_id: int, target_policy_steps: int, rng: np.random.Generator
) -> dict[str, Any]:
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if target is None:
        return {}
    instruction_type = str(getattr(getattr(env, "_instruction_spec", None), "instruction_type", "push_right"))
    sign = -1.0 if instruction_type == "push_left" else 1.0
    success_distance = max(_metadata_float(env, "push_success_displacement", 0.08), 0.01)
    effective_push_step = max(0.005, 0.5 * float(getattr(env, "action_step_xyz", 0.015)))
    remaining = min(success_distance, float(target_policy_steps) * effective_push_step)
    current_progress = max(0.0, success_distance - remaining)
    reward_initial = np.asarray(target, dtype=np.float32).copy()
    reward_initial[0] -= sign * current_progress
    contact_gap = 0.018 + min(0.04, 0.4 * remaining)
    ee = np.asarray(target, dtype=np.float32).copy()
    ee[0] -= sign * contact_gap
    ee[1] += float(rng.uniform(-0.008, 0.008))
    ee[2] = _ee_task_height(env, target, clearance=0.045)
    _force_gripper(env, 0.0)
    _move_ee(env, ee)
    return {
        "curriculum_shell_distance_m": float(remaining),
        "curriculum_shell_relation": "push_remaining_distance",
        "curriculum_reward_initial_obj_pos": reward_initial.tolist(),
        "curriculum_push_remaining_distance_m": float(remaining),
        "curriculum_shell_required_travel_m": float(remaining),
        "curriculum_put_start_stage": -1,
    }


def _full_put_start(env: Any, *, target: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    ee = _near_object_ee(env, target, rng=rng, distance_range=(0.0, 0.020), clearance=0.10)
    distance = float(np.linalg.norm(ee[:2] - np.asarray(target, dtype=np.float32)[:2]))
    _force_gripper(env, 1.0)
    _move_ee(env, ee)
    return {
        "curriculum_shell_distance_m": float(distance),
        "curriculum_shell_height_m": 0.10,
        "curriculum_shell_relation": "open_gripper_above_target",
        "curriculum_target_grasped": False,
        "curriculum_put_start_stage": 1,
    }


def _apply_complex_put_shell(
    env: Any, *, shell_id: int, target_policy_steps: int, rng: np.random.Generator
) -> dict[str, Any]:
    reference = _reference_position(env)
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if reference is None or target is None:
        return {}
    success_boundary = max(0.0, _metadata_float(env, "put_container_xy_tolerance", 0.08))
    travel = _policy_travel_distance(env, target_policy_steps)
    distance = float(success_boundary + travel)
    direction = _direction_with_room(env, center=reference, distance=distance, rng=rng)
    object_pos = np.asarray(reference, dtype=np.float32).copy()
    object_pos[:2] += direction * distance
    instruction_type = str(getattr(env._instruction_spec, "instruction_type", ""))
    clearance = 0.10 if instruction_type == "put_into_bowl" else 0.045
    object_pos[2] = float(reference[2] + clearance)
    held = _set_target_held_at(env, object_pos=object_pos)
    return {
        "curriculum_shell_distance_m": float(distance),
        "curriculum_shell_success_boundary_m": float(success_boundary),
        "curriculum_shell_required_travel_m": float(travel),
        "curriculum_shell_height_m": float(object_pos[2] - reference[2]),
        "curriculum_shell_relation": "held_object_near_receptacle",
        "curriculum_target_grasped": True,
        "curriculum_put_start_stage": 0,
        "curriculum_reward_initial_obj_pos": _reward_motion_baseline(env, object_pos, direction).tolist(),
        **held,
    }


def _apply_complex_binary_relation_shell(
    env: Any, *, shell_id: int, target_policy_steps: int, rng: np.random.Generator
) -> dict[str, Any]:
    reference = _reference_position(env)
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if reference is None or target is None:
        return {}
    instruction_type = str(getattr(getattr(env, "_instruction_spec", None), "instruction_type", ""))
    axis, sign, offset, _ = _relation_axis(env, instruction_type)
    desired = np.asarray(reference, dtype=np.float32).copy()
    desired[axis] += sign * offset
    desired[2] = max(float(target[2]), float(reference[2] + 0.035))
    travel = _policy_travel_distance(env, target_policy_steps)
    zone_half = 0.5 * max(
        1e-6, _metadata_float(env, "move_relation_success_zone_size", 0.06)
    )
    direction = _direction_with_room(
        env,
        center=desired,
        distance=float(np.sqrt(2.0) * zone_half + travel),
        rng=rng,
    )
    success_boundary = float(zone_half / max(float(np.max(np.abs(direction))), 1e-6))
    distance = float(success_boundary + travel)
    object_pos = desired.copy()
    object_pos[:2] += direction * distance
    held = _set_target_held_at(env, object_pos=object_pos)
    return {
        "curriculum_shell_distance_m": float(distance),
        "curriculum_shell_success_boundary_m": float(success_boundary),
        "curriculum_shell_required_travel_m": float(travel),
        "curriculum_shell_relation": "held_object_near_binary_relation",
        "curriculum_target_grasped": True,
        "curriculum_put_start_stage": 0,
        "curriculum_reward_initial_obj_pos": _reward_motion_baseline(env, object_pos, direction).tolist(),
        **held,
    }


def _apply_complex_between_shell(
    env: Any, *, shell_id: int, target_policy_steps: int, rng: np.random.Generator
) -> dict[str, Any]:
    ref_a = _reference_position(env, second=False)
    ref_b = _reference_position(env, second=True)
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if ref_a is None or ref_b is None or target is None:
        return {}
    midpoint = 0.5 * (np.asarray(ref_a, dtype=np.float32) + np.asarray(ref_b, dtype=np.float32))
    midpoint[2] = max(float(target[2]), float(midpoint[2] + 0.035))
    success_boundary = max(0.0, _metadata_float(env, "between_xy_tolerance", 0.06))
    travel = _policy_travel_distance(env, target_policy_steps)
    distance = float(success_boundary + travel)
    direction = _direction_with_room(env, center=midpoint, distance=distance, rng=rng)
    object_pos = midpoint.copy()
    object_pos[:2] += direction * distance
    held = _set_target_held_at(env, object_pos=object_pos)
    return {
        "curriculum_shell_distance_m": float(distance),
        "curriculum_shell_success_boundary_m": float(success_boundary),
        "curriculum_shell_required_travel_m": float(travel),
        "curriculum_shell_relation": "held_object_near_between_region",
        "curriculum_target_grasped": True,
        "curriculum_put_start_stage": 0,
        "curriculum_reward_initial_obj_pos": _reward_motion_baseline(env, object_pos, direction).tolist(),
        **held,
    }


def apply_cdpr_put_start_stage(
    env: Any,
    *,
    stage_id: int,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Apply the two-stage placement start: held object, then open gripper above it."""
    rng = rng or getattr(env, "np_random", np.random.default_rng())
    stage_id = int(np.clip(int(stage_id), 0, 1))
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if target is None:
        return {"curriculum_put_start_stage": int(stage_id)}
    if stage_id == 0:
        # reset(options={start_with_caught_object: true}) already created the
        # grasp; this branch records the abstraction without moving it again.
        return {
            "curriculum_mode": "lchol_put_stage",
            "curriculum_put_start_stage": 0,
            "curriculum_target_grasped": bool(getattr(env, "_caught_object_start_active", False)),
            "curriculum_shell_relation": "held_object_start",
        }
    return {
        "curriculum_mode": "lchol_put_stage",
        **_full_put_start(env, target=target, rng=rng),
    }


def _apply_move_to_object_shell(env: Any, *, shell_id: int, rng: np.random.Generator) -> dict[str, Any]:
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if target is None:
        return {}
    ranges = ((0.010, 0.020), (0.020, 0.050), (0.050, 0.100))
    distance = _uniform_range(rng, ranges[min(shell_id, len(ranges) - 1)])
    direction = _unit_xy(rng)
    ee = np.asarray(target, dtype=np.float32).copy()
    ee[:2] += direction * distance
    ee[2] = _ee_task_height(env, target, clearance=0.08)
    _move_ee(env, ee)
    return {
        "curriculum_shell_distance_m": float(distance),
        "curriculum_shell_relation": "ee_xy_near_target",
    }


def _apply_grab_shell(env: Any, *, shell_id: int, rng: np.random.Generator) -> dict[str, Any]:
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if target is None:
        return {}
    if shell_id == 0:
        xy_distance = _uniform_range(rng, (0.000, 0.010))
        z_clearance = _uniform_range(rng, (0.105, 0.120))
        gripper = 1.0
    elif shell_id == 1:
        xy_distance = _uniform_range(rng, (0.010, 0.025))
        z_clearance = _uniform_range(rng, (0.110, 0.125))
        gripper = 1.0
    elif shell_id == 2:
        xy_distance = _uniform_range(rng, (0.025, 0.050))
        z_clearance = _uniform_range(rng, (0.120, 0.140))
        gripper = 1.0
    else:
        xy_distance = _uniform_range(rng, (0.050, 0.100))
        z_clearance = _uniform_range(rng, (0.130, 0.160))
        gripper = 1.0
    direction = _unit_xy(rng)
    ee = np.asarray(target, dtype=np.float32).copy()
    ee[:2] += direction * xy_distance
    ee[2] = _ee_task_height(env, target, clearance=z_clearance)
    _force_gripper(env, gripper)
    _move_ee(env, ee)
    return {
        "curriculum_shell_distance_m": float(xy_distance),
        "curriculum_shell_height_m": float(z_clearance),
        "curriculum_shell_relation": "gripper_near_object",
    }


def _apply_put_shell(env: Any, *, shell_id: int, rng: np.random.Generator) -> dict[str, Any]:
    plate = _reference_position(env)
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if plate is None or target is None:
        return {}

    if shell_id <= 3:
        lateral_ranges = ((0.000, 0.010), (0.015, 0.040), (0.040, 0.080), (0.080, 0.140))
        height_ranges = ((0.010, 0.020), (0.020, 0.050), (0.040, 0.080), (0.060, 0.100))
        lateral = _uniform_range(rng, lateral_ranges[shell_id])
        height = _uniform_range(rng, height_ranges[shell_id])
        direction = _unit_xy(rng)
        object_pos = np.asarray(plate, dtype=np.float32).copy()
        object_pos[:2] += direction * lateral
        object_pos[2] = float(plate[2] + height)
        held = _set_target_held_at(env, object_pos=object_pos)
        return {
            "curriculum_shell_distance_m": float(lateral),
            "curriculum_shell_height_m": float(height),
            "curriculum_shell_relation": "held_object_near_receptacle",
            "curriculum_target_grasped": True,
            "curriculum_reward_initial_obj_pos": _reward_motion_baseline(env, object_pos, direction).tolist(),
            **held,
        }

    ee = _near_object_ee(env, target, rng=rng, distance_range=(0.010, 0.020), clearance=0.06)
    _force_gripper(env, 1.0)
    _move_ee(env, ee)
    return {
        "curriculum_shell_distance_m": float(np.linalg.norm(ee[:2] - target[:2])),
        "curriculum_shell_relation": "full_task_gripper_near_object",
    }


def _apply_push_shell(env: Any, *, shell_id: int, rng: np.random.Generator) -> dict[str, Any]:
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if target is None:
        return {}
    instruction_type = str(getattr(getattr(env, "_instruction_spec", None), "instruction_type", "push_right"))
    if instruction_type in {"push_left", "push_right"}:
        axis = 0
        sign = -1.0 if instruction_type == "push_left" else 1.0
        tolerance_axis = 1
    else:
        axis = 1
        sign = 1.0 if instruction_type == "push_forward" else -1.0
        tolerance_axis = 0
    push_distance = _metadata_float(env, "push_success_displacement", 0.08)
    one_step = max(0.005, min(0.020, 0.5 * float(getattr(env, "action_step_xyz", 0.02))))
    current_progress = max(0.0, float(push_distance) - one_step) if shell_id == 0 else 0.0
    reward_initial = np.asarray(target, dtype=np.float32).copy()
    reward_initial[axis] -= sign * current_progress

    behind_ranges = ((0.000, 0.005), (0.005, 0.020), (0.020, 0.050), (0.050, 0.100))
    behind = _uniform_range(rng, behind_ranges[min(shell_id, len(behind_ranges) - 1)])
    ee = np.asarray(target, dtype=np.float32).copy()
    ee[axis] -= sign * (behind + 0.020)
    ee[tolerance_axis] += float(rng.uniform(-0.010, 0.010))
    ee[2] = _ee_task_height(env, target, clearance=0.045)
    _force_gripper(env, 0.0)
    _move_ee(env, ee)
    return {
        "curriculum_shell_distance_m": float(behind),
        "curriculum_shell_relation": "ee_behind_push_contact",
        "curriculum_reward_initial_obj_pos": reward_initial.tolist(),
    }


def _apply_binary_relation_shell(env: Any, *, shell_id: int, rng: np.random.Generator) -> dict[str, Any]:
    ref = _reference_position(env)
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if ref is None or target is None:
        return {}
    instruction_type = str(getattr(getattr(env, "_instruction_spec", None), "instruction_type", ""))
    axis, sign, offset, tolerance_axis = _relation_axis(env, instruction_type)
    desired = np.asarray(ref, dtype=np.float32).copy()
    desired[axis] += sign * offset
    desired[2] = max(float(target[2]), float(ref[2] + 0.035))

    if shell_id <= 2:
        ranges = ((0.010, 0.020), (0.020, 0.050), (0.050, 0.100))
        distance = _uniform_range(rng, ranges[shell_id])
        object_pos = desired.copy()
        if shell_id == 0:
            object_pos[axis] = float(ref[axis] + sign * max(0.0, offset - distance))
        else:
            direction = _unit_xy(rng)
            object_pos[:2] += direction * distance
        object_pos[tolerance_axis] += float(rng.uniform(-0.015, 0.015))
        held = _set_target_held_at(env, object_pos=object_pos)
        return {
            "curriculum_shell_distance_m": float(distance),
            "curriculum_shell_relation": "held_object_near_binary_relation",
            "curriculum_target_grasped": True,
            "curriculum_reward_initial_obj_pos": _reward_motion_baseline(env, object_pos, _axis_xy(axis, sign)).tolist(),
            **held,
        }

    ee = _near_object_ee(env, target, rng=rng, distance_range=(0.010, 0.020), clearance=0.06)
    _force_gripper(env, 1.0)
    _move_ee(env, ee)
    return {
        "curriculum_shell_distance_m": float(np.linalg.norm(ee[:2] - target[:2])),
        "curriculum_shell_relation": "full_task_gripper_near_object",
    }


def _apply_between_shell(env: Any, *, shell_id: int, rng: np.random.Generator) -> dict[str, Any]:
    ref_a = _reference_position(env, second=False)
    ref_b = _reference_position(env, second=True)
    target = _body_position(env, getattr(env, "_target_body_name", ""))
    if ref_a is None or ref_b is None or target is None:
        return {}
    midpoint = 0.5 * (np.asarray(ref_a, dtype=np.float32) + np.asarray(ref_b, dtype=np.float32))
    midpoint[2] = max(float(target[2]), float(midpoint[2] + 0.035))

    if shell_id <= 2:
        ranges = ((0.010, 0.020), (0.020, 0.050), (0.050, 0.100))
        distance = _uniform_range(rng, ranges[shell_id])
        direction = _unit_xy(rng)
        object_pos = midpoint.copy()
        object_pos[:2] += direction * distance
        held = _set_target_held_at(env, object_pos=object_pos)
        return {
            "curriculum_shell_distance_m": float(distance),
            "curriculum_shell_relation": "held_object_near_between_region",
            "curriculum_target_grasped": True,
            "curriculum_reward_initial_obj_pos": _reward_motion_baseline(env, object_pos, direction).tolist(),
            **held,
        }

    ee = _near_object_ee(env, target, rng=rng, distance_range=(0.010, 0.020), clearance=0.06)
    _force_gripper(env, 1.0)
    _move_ee(env, ee)
    return {
        "curriculum_shell_distance_m": float(np.linalg.norm(ee[:2] - target[:2])),
        "curriculum_shell_relation": "full_task_gripper_near_object",
    }


def _metadata_float(env: Any, key: str, default: float) -> float:
    metadata = getattr(env, "_task_metadata", {}) or {}
    try:
        return float(metadata.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _uniform_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(rng.uniform(float(lo), float(hi)))


def _unit_xy(rng: np.random.Generator) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)


def _axis_xy(axis: int, sign: float) -> np.ndarray:
    out = np.zeros((2,), dtype=np.float32)
    out[int(axis)] = float(sign)
    return out


def _body_position(env: Any, body_name: str) -> np.ndarray | None:
    if not body_name:
        return None
    try:
        getter = getattr(env, "_get_task_body_position", None)
        if not callable(getter):
            getter = env._get_body_position
        return np.asarray(getter(str(body_name)), dtype=np.float32).reshape(3).copy()
    except Exception:
        return None


def _reference_position(env: Any, *, second: bool = False) -> np.ndarray | None:
    body_name = getattr(env, "_second_reference_body_name" if second else "_reference_body_name", "")
    pos = _body_position(env, str(body_name))
    if pos is not None:
        return pos
    try:
        return np.asarray(env._reference_object_position(second=second), dtype=np.float32).reshape(3).copy()
    except Exception:
        return None


def _ee_task_height(env: Any, base: np.ndarray, *, clearance: float) -> float:
    z = float(np.asarray(base, dtype=np.float32).reshape(3)[2] + float(clearance))
    support = float(getattr(env, "_support_surface_z", 0.0))
    ee_min = float(getattr(env, "_ee_min_z", float("-inf")))
    if np.isfinite(support):
        z = max(z, support + 0.045)
    if np.isfinite(ee_min):
        z = max(z, ee_min)
    return float(z)


def _move_ee(env: Any, xyz: np.ndarray) -> None:
    target_raw = np.asarray(xyz, dtype=np.float32).reshape(3)
    clamp_target = getattr(env, "_clamp_ee_target", None)
    if callable(clamp_target):
        target = np.asarray(clamp_target(target_raw), dtype=np.float32).reshape(3)
    else:
        target = np.asarray(clamp_xyz(target_raw), dtype=np.float32)
        ee_min = float(getattr(env, "_ee_min_z", float("-inf")))
        if np.isfinite(ee_min):
            target[2] = max(float(target[2]), ee_min)
    if hasattr(env, "_set_ee_target"):
        env._set_ee_target(target)
    sim = getattr(env, "sim", None)
    moved = _teleport_ee_free_joint(env, target)
    if sim is not None and hasattr(sim, "goto"):
        if not moved:
            try:
                sim.goto(target, max_steps=90, tol=0.008)
                moved = True
            except Exception:
                moved = False
    if not moved and sim is not None and hasattr(sim, "run_simulation_step"):
        for _ in range(4):
            try:
                sim.run_simulation_step(capture_frame=False)
            except Exception:
                break
    hold_current_pose = getattr(sim, "hold_current_pose", None)
    if callable(hold_current_pose):
        try:
            hold_current_pose(warm_steps=0)
        except Exception:
            pass
    if hasattr(env, "_locked_target_xyz"):
        env._locked_target_xyz = target.astype(np.float32)
    if hasattr(env, "_episode_ee_start"):
        env._episode_ee_start = target.astype(np.float32)


def _teleport_ee_free_joint(env: Any, target: np.ndarray) -> bool:
    sim = getattr(env, "sim", None)
    if sim is None or not hasattr(sim, "model") or not hasattr(sim, "data"):
        return False
    try:
        import mujoco as mj  # type: ignore
    except Exception:
        return False

    try:
        joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, "ee_free")
        if int(joint_id) == -1:
            return False
        qadr = int(sim.model.jnt_qposadr[int(joint_id)])
        current_quat = np.asarray(sim.data.qpos[qadr + 3 : qadr + 7], dtype=float).copy()
        if float(np.linalg.norm(current_quat)) < 1e-9:
            current_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        sim.data.qpos[qadr : qadr + 3] = np.asarray(target, dtype=float).reshape(3)
        sim.data.qpos[qadr + 3 : qadr + 7] = current_quat / max(float(np.linalg.norm(current_quat)), 1e-9)
        if hasattr(sim.data, "qvel") and hasattr(sim.model, "jnt_dofadr"):
            sim.data.qvel[:] = 0.0
        if hasattr(sim.data, "qacc_warmstart"):
            sim.data.qacc_warmstart[:] = 0.0
        mj.mj_forward(sim.model, sim.data)
        if hasattr(sim, "set_target_position"):
            sim.set_target_position(np.asarray(target, dtype=float).reshape(3))
        elif hasattr(sim, "target_pos"):
            sim.target_pos = np.asarray(target, dtype=float).reshape(3).copy()
        sync = getattr(sim, "_sync_controller_geometry_from_state", None)
        if callable(sync):
            sync()
        hold_current_pose = getattr(sim, "hold_current_pose", None)
        if callable(hold_current_pose):
            hold_current_pose(warm_steps=0)
        return True
    except Exception:
        return False


def _force_gripper(env: Any, opening: float) -> None:
    if hasattr(env, "_force_gripper_opening"):
        env._force_gripper_opening(float(np.clip(opening, 0.0, 1.0)))


def _target_measurement(env: Any, body_name: str) -> dict[str, float] | None:
    getter = getattr(env, "_caught_object_start_measurement_for_body", None)
    if callable(getter):
        try:
            measurement = getter(str(body_name))
        except Exception:
            measurement = None
        if measurement is not None:
            return dict(measurement)

    candidates = [
        str(getattr(env, "_target_catalog_name", "")),
        str(body_name or ""),
    ]
    inverse = getattr(env, "_inverse_catalog_to_body", {}) or {}
    if str(body_name) in inverse:
        candidates.append(str(inverse[str(body_name)]))
    for raw in candidates:
        name = str(raw or "").strip()
        if not name:
            continue
        for key in (name, name.replace("ycb_", ""), name.replace("_", " ")):
            measurement = _YCB_CAUGHT_OBJECT_MEASUREMENTS.get(str(key))
            if measurement is not None:
                return dict(measurement)
    return None


def _held_object_offset(env: Any) -> np.ndarray:
    metadata = getattr(env, "_task_metadata", {}) or {}
    raw = metadata.get("caught_object_start_object_offset", _DEFAULT_HELD_OBJECT_OFFSET)
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    except Exception:
        return _DEFAULT_HELD_OBJECT_OFFSET.copy()
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return _DEFAULT_HELD_OBJECT_OFFSET.copy()
    return arr[:3].astype(np.float32).copy()


def _held_gripper_opening(env: Any, body_name: str) -> float:
    measurement = _target_measurement(env, body_name)
    if measurement is not None and "opening" in measurement:
        joint_span = float(
            max(
                float(getattr(getattr(env, "sim", None), "gripper_joint_max", 0.03))
                - float(getattr(getattr(env, "sim", None), "gripper_joint_min", 0.0)),
                1e-6,
            )
        )
        clearance = max(0.0, _metadata_float(env, "caught_object_start_gripper_clearance", 0.0))
        compression = max(0.0, _metadata_float(env, "caught_object_start_grip_compression", 0.0))
        opening = float(measurement["opening"]) + (clearance - compression) / joint_span
        return float(np.clip(opening, 0.0, 1.0))
    if hasattr(env, "_caught_object_start_gripper_opening_for_body"):
        try:
            return float(np.clip(float(env._caught_object_start_gripper_opening_for_body(body_name)), 0.0, 1.0))
        except Exception:
            return 0.0
    return 0.0


def _hold_center(env: Any) -> np.ndarray | None:
    if not hasattr(env, "_caught_object_start_hold_center"):
        return None
    try:
        hold_center = env._caught_object_start_hold_center()
    except Exception:
        hold_center = None
    if hold_center is None:
        return None
    arr = np.asarray(hold_center, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(arr)):
        return None
    return arr.copy()


def _set_body(env: Any, body_name: str, xyz: np.ndarray) -> bool:
    if not body_name or not hasattr(env, "_set_body_position"):
        return False
    return bool(env._set_body_position(str(body_name), np.asarray(xyz, dtype=np.float32).reshape(3)))


def _set_target_held_at(env: Any, *, object_pos: np.ndarray) -> dict[str, Any]:
    target_body = str(getattr(env, "_target_body_name", ""))
    target_catalog = str(getattr(env, "_target_catalog_name", ""))
    requested_object_pos = _clamp_object_position(env, object_pos)
    measurement = _target_measurement(env, target_body)
    gripper_opening = _held_gripper_opening(env, target_body)
    _force_gripper(env, gripper_opening)

    try:
        ee_pos = np.asarray(env._get_ee_position(), dtype=np.float32).reshape(3)
    except Exception:
        ee_pos = requested_object_pos + np.array([0.0, 0.0, 0.080], dtype=np.float32)
    hold_offset = _held_object_offset(env)
    hold_center = _hold_center(env)
    if hold_center is not None:
        desired_hold_center = requested_object_pos - hold_offset
        desired_ee = ee_pos + (desired_hold_center - hold_center)
        _move_ee(env, desired_ee)
        _force_gripper(env, gripper_opening)
        try:
            ee_pos = np.asarray(env._get_ee_position(), dtype=np.float32).reshape(3)
        except Exception:
            pass
        hold_center = _hold_center(env)
        if hold_center is not None:
            correction = (requested_object_pos - hold_offset) - hold_center
            if float(np.linalg.norm(correction)) > 0.005:
                _move_ee(env, ee_pos + correction)
                _force_gripper(env, gripper_opening)
                try:
                    ee_pos = np.asarray(env._get_ee_position(), dtype=np.float32).reshape(3)
                except Exception:
                    pass
                hold_center = _hold_center(env)

    if hold_center is not None:
        object_pos = _clamp_object_position(env, hold_center + hold_offset)
    else:
        object_pos = requested_object_pos

    if not _set_body(env, target_body, object_pos):
        return {"curriculum_held_object_set": False}

    env._caught_object_start_active = True
    env._caught_object_start_body = target_body
    env._caught_object_start_catalog = target_catalog
    env._caught_object_start_position = object_pos.astype(np.float32)
    env._caught_object_start_ee_offset = (object_pos - ee_pos).astype(np.float32)
    env._caught_object_start_gripper_opening = float(gripper_opening)
    hold_center = _hold_center(env)
    if hold_center is not None:
        env._caught_object_start_hold_offset = (
            object_pos - np.asarray(hold_center, dtype=np.float32).reshape(3)
        ).astype(np.float32)
    else:
        env._caught_object_start_hold_offset = np.zeros((3,), dtype=np.float32)
    return {
        "curriculum_held_object_set": True,
        "curriculum_held_object_position": object_pos.tolist(),
        "curriculum_held_requested_object_position": requested_object_pos.tolist(),
        "curriculum_held_object_offset": hold_offset.tolist(),
        "curriculum_held_gripper_opening": float(gripper_opening),
        "curriculum_held_object_width_m": (
            float(measurement["width_m"]) if measurement is not None and "width_m" in measurement else float("nan")
        ),
        "curriculum_held_finger_qpos_m": (
            float(measurement["finger_qpos_m"])
            if measurement is not None and "finger_qpos_m" in measurement
            else float("nan")
        ),
    }


def _clamp_object_position(env: Any, xyz: np.ndarray) -> np.ndarray:
    target = np.asarray(clamp_xyz(np.asarray(xyz, dtype=np.float32).reshape(3)), dtype=np.float32)
    metadata = getattr(env, "_task_metadata", {}) or {}
    x_bounds = metadata.get("object_state_x_bounds", metadata.get("object_spawn_x_bounds", (-0.30, 0.30)))
    y_bounds = metadata.get("object_state_y_bounds", metadata.get("object_spawn_y_bounds", (-0.30, 0.30)))
    try:
        target[0] = float(np.clip(target[0], min(float(x_bounds[0]), float(x_bounds[1])), max(float(x_bounds[0]), float(x_bounds[1]))))
        target[1] = float(np.clip(target[1], min(float(y_bounds[0]), float(y_bounds[1])), max(float(y_bounds[0]), float(y_bounds[1]))))
    except Exception:
        target[0] = float(np.clip(target[0], -0.30, 0.30))
        target[1] = float(np.clip(target[1], -0.30, 0.30))
    support = float(getattr(env, "_support_surface_z", 0.0))
    target[2] = max(float(target[2]), support + 0.01)
    return target


def _near_object_ee(
    env: Any,
    target: np.ndarray,
    *,
    rng: np.random.Generator,
    distance_range: tuple[float, float],
    clearance: float,
) -> np.ndarray:
    direction = _unit_xy(rng)
    ee = np.asarray(target, dtype=np.float32).copy()
    ee[:2] += direction * _uniform_range(rng, distance_range)
    ee[2] = _ee_task_height(env, target, clearance=clearance)
    return ee


def _reward_motion_baseline(env: Any, object_pos: np.ndarray, direction_xy: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction_xy, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    required = max(
        _metadata_float(env, "put_min_target_motion", 0.0),
        _metadata_float(env, "relation_min_target_motion", 0.0),
    )
    baseline = np.asarray(object_pos, dtype=np.float32).reshape(3).copy()
    # Put the artificial pre-curriculum baseline farther away from the goal
    # than the held reset. Approaching the goal then preserves/increases the
    # credited motion instead of cancelling it and requiring an overshoot.
    baseline[:2] += direction * float(required + 0.020)
    return baseline.astype(np.float32)


def _relation_axis(env: Any, instruction_type: str) -> tuple[int, float, float, int]:
    if instruction_type == "move_left_of_object":
        return 0, -1.0, _metadata_float(env, "relation_left_right_offset", 0.08), 1
    if instruction_type == "move_right_of_object":
        return 0, 1.0, _metadata_float(env, "relation_left_right_offset", 0.08), 1
    if instruction_type in {"move_in_front_of_object", "put_in_front_of_object"}:
        return 1, -1.0, _metadata_float(env, "relation_front_behind_offset", 0.08), 0
    return 1, 1.0, _metadata_float(env, "relation_front_behind_offset", 0.08), 0
