from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTIVE_INSTRUCTION_TYPES: tuple[str, ...] = (
    "move_to_object",
    "push_left",
    "push_right",
    "put_into_bowl",
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_between_objects",
)
INSTRUCTION_TO_ID = {
    name: index for index, name in enumerate(ACTIVE_INSTRUCTION_TYPES)
}


@dataclass(frozen=True)
class BatchedTaskThresholds:
    move_to_xy_low: float = 0.0
    move_to_xy: float = 0.02
    push_displacement: float = 0.08
    push_orthogonal_tolerance: float = 0.02
    push_max_overshoot: float = 0.025
    push_support_min_clearance: float = 0.005
    push_support_vertical_tolerance: float = 0.04
    container_xy: float = 0.03
    container_z: float = 0.12
    minimum_target_motion: float = 0.04
    relation_offset: float = 0.10
    relation_zone_size: float = 0.03
    between_xy: float = 0.03
    release_opening: float = 0.55
    sparse_success_reward: float = 1.0
    sparse_failure_reward: float = 0.0


@dataclass(frozen=True)
class BatchedMoveToDistanceReward:
    """GPU form of the established CPU ``move to`` distance reward.

    The CPU hook in ``rl_instruction_tasks._compute_move_to_object_reward``
    computes an inverse-polynomial reward from the distance to an XY success
    window.  MJWarp cannot call that NumPy/Python hook per world without
    transferring simulator state back to the host, so this dataclass carries
    the same distance-term parameters for the tensorized hot path.
    """

    xy_window_low: float = 0.0
    xy_window_high: float = 0.02
    xy_reward_scale: float = 0.08
    distance_reward_weight: float = 1.0
    distance_reward_exponent: float = 2.0
    success_bonus: float = 0.0
    excess_distance_penalty_weight: float = 0.0
    too_close_penalty_weight: float = 0.0
    z_window_low: float = 0.10
    z_window_high: float = 0.20
    z_penalty_scale: float = 0.05
    z_penalty_weight: float = 0.20
    require_z_window: bool = False

    @classmethod
    def from_metadata(
        cls, metadata: dict[str, Any] | None
    ) -> "BatchedMoveToDistanceReward":
        values = dict(metadata or {})

        def number(key: str, default: float) -> float:
            raw = values.get(key, default)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return float(default)

        def flag(key: str, default: bool) -> bool:
            raw = values.get(key, default)
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "yes", "on"}
            return bool(raw)

        default_tolerance = number("success_distance", 0.02)
        high = max(
            number(
                "move_to_object_xy_window_high",
                number("move_to_object_xy_tolerance", default_tolerance),
            ),
            1.0e-6,
        )
        low = max(number("move_to_object_xy_window_low", 0.0), 0.0)
        if low > high:
            low, high = high, low
        scale = max(
            number(
                "move_to_object_xy_reward_scale",
                max(4.0 * high, 0.08),
            ),
            1.0e-6,
        )
        z_low, z_high = sorted(
            (
                number("move_to_object_z_window_low", 0.10),
                number("move_to_object_z_window_high", 0.20),
            )
        )
        return cls(
            xy_window_low=low,
            xy_window_high=high,
            xy_reward_scale=scale,
            distance_reward_weight=number(
                "move_to_object_distance_reward_weight",
                number("move_to_object_proximity_weight", 1.0),
            ),
            distance_reward_exponent=max(
                number("distance_reward_exponent", 2.0), 1.0e-6
            ),
            success_bonus=number(
                "move_to_object_success_bonus",
                number("success_bonus", 0.0),
            ),
            excess_distance_penalty_weight=number(
                "move_to_object_excess_distance_penalty_weight", 0.0
            ),
            too_close_penalty_weight=number(
                "move_to_object_too_close_penalty_weight", 0.0
            ),
            z_window_low=z_low,
            z_window_high=z_high,
            z_penalty_scale=max(
                number("move_to_object_z_penalty_scale", 0.05), 1.0e-6
            ),
            z_penalty_weight=number(
                "move_to_object_z_penalty_weight", 0.20
            ),
            require_z_window=flag(
                "move_to_object_require_z_window", False
            ),
        )


@dataclass
class BatchedTaskState:
    instruction_ids: Any
    target_slots: Any
    reference_slots: Any
    second_reference_slots: Any
    initial_target_positions: Any
    ever_grasped: Any
    grasped: Any
    step_count: Any
    release_threshold: Any
    support_surface_z: Any

    def validate(self, batch_size: int, device: Any) -> None:
        import torch

        one_dimensional = (
            "instruction_ids",
            "target_slots",
            "reference_slots",
            "second_reference_slots",
            "ever_grasped",
            "grasped",
            "step_count",
            "release_threshold",
            "support_surface_z",
        )
        for name in one_dimensional:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if tuple(value.shape) != (batch_size,) or value.device != device:
                raise ValueError(
                    f"{name} must have shape ({batch_size},) on {device}."
                )
        if (
            tuple(self.initial_target_positions.shape) != (batch_size, 3)
            or self.initial_target_positions.device != device
        ):
            raise ValueError(
                f"initial_target_positions must have shape ({batch_size}, 3) on {device}."
            )


@dataclass(frozen=True)
class BatchedTaskResult:
    rewards: Any
    success: Any
    terminated: Any
    diagnostics: dict[str, Any]


def move_to_distance_rewards(
    xy_distance: Any,
    success: Any,
    *,
    config: BatchedMoveToDistanceReward,
    ee_z: Any | None = None,
) -> Any:
    """Evaluate the existing inverse-polynomial move-to reward on GPU."""

    import torch

    low_error = (
        torch.full_like(xy_distance, float(config.xy_window_low)) - xy_distance
    ).clamp_min(0.0)
    high_error = (
        xy_distance
        - torch.full_like(xy_distance, float(config.xy_window_high))
    ).clamp_min(0.0)
    window_error = torch.maximum(low_error, high_error)
    normalized = window_error / max(float(config.xy_reward_scale), 1.0e-6)
    distance_reward = float(config.distance_reward_weight) / (
        1.0
        + normalized.pow(max(float(config.distance_reward_exponent), 1.0e-6))
    )
    reward = distance_reward + success.to(dtype=xy_distance.dtype) * float(
        config.success_bonus
    )
    reward = reward - float(
        config.excess_distance_penalty_weight
    ) * torch.tanh(high_error / max(float(config.xy_reward_scale), 1.0e-6))
    reward = reward - float(config.too_close_penalty_weight) * torch.tanh(
        low_error / max(float(config.xy_reward_scale), 1.0e-6)
    )
    if ee_z is not None:
        z_low_error = (
            torch.full_like(ee_z, float(config.z_window_low)) - ee_z
        ).clamp_min(0.0)
        z_high_error = (
            ee_z - torch.full_like(ee_z, float(config.z_window_high))
        ).clamp_min(0.0)
        z_outside_distance = torch.maximum(z_low_error, z_high_error)
        reward = reward - float(config.z_penalty_weight) * (
            z_outside_distance / max(float(config.z_penalty_scale), 1.0e-6)
        )
    return reward


def gather_world_slots(values: Any, slots: Any) -> Any:
    import torch

    safe_slots = slots.to(dtype=torch.int64).clamp(0, int(values.shape[1]) - 1)
    rows = torch.arange(
        int(values.shape[0]), dtype=torch.int64, device=values.device
    )
    return values[rows, safe_slots]


def build_smolvla_state_tensor(
    *,
    ee_position: Any,
    ee_yaw: Any,
    gripper_opening: Any,
    object_positions: Any,
    target_slots: Any,
    state_dim: int = 6,
) -> Any:
    """Match the first six elements of the established CPU state adapter."""

    import torch
    import torch.nn.functional as functional

    target = gather_world_slots(object_positions, target_slots)
    xy_distance = torch.linalg.vector_norm(
        target[:, :2] - ee_position[:, :2], dim=-1, keepdim=True
    )
    state = torch.cat(
        (
            ee_position,
            ee_yaw.reshape(-1, 1),
            gripper_opening.reshape(-1, 1),
            xy_distance,
        ),
        dim=-1,
    ).to(dtype=torch.float32)
    width = max(1, int(state_dim))
    if int(state.shape[1]) > width:
        return state[:, :width]
    if int(state.shape[1]) < width:
        return functional.pad(state, (0, width - int(state.shape[1])))
    return state


def evaluate_active_sparse_tasks(
    *,
    state: BatchedTaskState,
    ee_position: Any,
    object_positions: Any,
    gripper_opening: Any,
    caught_target: Any,
    active_mask: Any,
    max_steps: int,
    thresholds: BatchedTaskThresholds | None = None,
    move_to_distance_reward: BatchedMoveToDistanceReward | None = None,
) -> BatchedTaskResult:
    """Tensorized task predicates plus optional dense move-to rewards."""

    import torch

    cfg = thresholds or BatchedTaskThresholds()
    batch_size = int(ee_position.shape[0])
    state.validate(batch_size, ee_position.device)
    active = active_mask.to(dtype=torch.bool)
    target = gather_world_slots(object_positions, state.target_slots)
    reference = gather_world_slots(object_positions, state.reference_slots)
    second_reference = gather_world_slots(
        object_positions, state.second_reference_slots
    )
    delta = target - state.initial_target_positions
    target_motion_xy = torch.linalg.vector_norm(delta[:, :2], dim=-1)
    ee_xy_distance = torch.linalg.vector_norm(
        target[:, :2] - ee_position[:, :2], dim=-1
    )

    state.ever_grasped.logical_or_(caught_target.to(dtype=torch.bool))
    state.grasped.copy_(
        caught_target.to(dtype=torch.bool) & (gripper_opening <= 0.94)
    )
    released = gripper_opening >= torch.maximum(
        state.release_threshold,
        torch.full_like(state.release_threshold, float(cfg.release_opening)),
    )

    success = torch.zeros(
        (batch_size,), dtype=torch.bool, device=ee_position.device
    )
    instruction = state.instruction_ids.to(dtype=torch.int64)

    is_move_to = instruction == INSTRUCTION_TO_ID["move_to_object"]
    move_to_z_in_window = (
        (ee_position[:, 2] >= float(move_to_distance_reward.z_window_low))
        & (ee_position[:, 2] <= float(move_to_distance_reward.z_window_high))
        if move_to_distance_reward is not None
        else torch.ones_like(is_move_to)
    )
    move_to = (ee_xy_distance >= float(cfg.move_to_xy_low)) & (
        ee_xy_distance <= float(cfg.move_to_xy)
    )
    if (
        move_to_distance_reward is not None
        and move_to_distance_reward.require_z_window
    ):
        move_to &= move_to_z_in_window
    success |= is_move_to & move_to

    push_sign = torch.where(
        instruction == INSTRUCTION_TO_ID["push_left"],
        torch.full_like(delta[:, 0], -1.0),
        torch.ones_like(delta[:, 0]),
    )
    push_motion = push_sign * delta[:, 0]
    push_orthogonal = delta[:, 1].abs()
    push_overshoot = (push_motion - float(cfg.push_displacement)).clamp_min(0.0)
    support_clearance = target[:, 2] - state.support_surface_z
    push_ok = (
        (push_motion >= float(cfg.push_displacement))
        & (push_orthogonal <= float(cfg.push_orthogonal_tolerance))
        & (push_overshoot <= float(cfg.push_max_overshoot))
        & (support_clearance >= float(cfg.push_support_min_clearance))
        & (delta[:, 2].abs() <= float(cfg.push_support_vertical_tolerance))
    )
    is_push = (instruction == INSTRUCTION_TO_ID["push_left"]) | (
        instruction == INSTRUCTION_TO_ID["push_right"]
    )
    success |= is_push & push_ok

    container_xy = torch.linalg.vector_norm(
        target[:, :2] - reference[:, :2], dim=-1
    )
    container_z = (target[:, 2] - reference[:, 2]).abs()
    container_ok = (
        (container_xy <= float(cfg.container_xy))
        & (container_z <= float(cfg.container_z))
        & (target_motion_xy >= float(cfg.minimum_target_motion))
        & state.ever_grasped
        & released
    )
    is_container = (instruction == INSTRUCTION_TO_ID["put_into_bowl"]) | (
        instruction == INSTRUCTION_TO_ID["put_into_plate"]
    )
    success |= is_container & container_ok

    relation_sign = torch.where(
        instruction == INSTRUCTION_TO_ID["move_left_of_object"],
        torch.full_like(delta[:, 0], -1.0),
        torch.ones_like(delta[:, 0]),
    )
    desired_x = reference[:, 0] + relation_sign * float(cfg.relation_offset)
    half_zone = 0.5 * float(cfg.relation_zone_size)
    relation_ok = (
        ((target[:, 0] - desired_x).abs() <= half_zone)
        & ((target[:, 1] - reference[:, 1]).abs() <= half_zone)
        & (target_motion_xy >= float(cfg.minimum_target_motion))
        & state.ever_grasped
        & released
    )
    is_relation = (
        instruction == INSTRUCTION_TO_ID["move_left_of_object"]
    ) | (instruction == INSTRUCTION_TO_ID["move_right_of_object"])
    success |= is_relation & relation_ok

    midpoint = 0.5 * (reference[:, :2] + second_reference[:, :2])
    segment = second_reference[:, :2] - reference[:, :2]
    segment_norm_sq = (segment * segment).sum(dim=-1)
    projection = (
        ((target[:, :2] - reference[:, :2]) * segment).sum(dim=-1)
        / segment_norm_sq.clamp_min(1.0e-8)
    )
    between_ok = (
        (
            torch.linalg.vector_norm(target[:, :2] - midpoint, dim=-1)
            <= float(cfg.between_xy)
        )
        & (projection >= 0.0)
        & (projection <= 1.0)
        & (target_motion_xy >= float(cfg.minimum_target_motion))
        & state.ever_grasped
        & released
        & (state.second_reference_slots >= 0)
    )
    success |= (
        instruction == INSTRUCTION_TO_ID["move_between_objects"]
    ) & between_ok

    success &= active
    state.step_count.add_(active.to(dtype=state.step_count.dtype))
    timeout = state.step_count >= max(1, int(max_steps))
    terminated = success | timeout
    rewards = torch.where(
        success,
        torch.full_like(gripper_opening, float(cfg.sparse_success_reward)),
        torch.full_like(gripper_opening, float(cfg.sparse_failure_reward)),
    )
    if move_to_distance_reward is not None:
        dense_move_to = move_to_distance_rewards(
            ee_xy_distance,
            success,
            config=move_to_distance_reward,
            ee_z=ee_position[:, 2],
        )
        rewards = torch.where(is_move_to, dense_move_to, rewards)
    return BatchedTaskResult(
        rewards=rewards,
        success=success,
        terminated=terminated,
        diagnostics={
            "ee_xy_distance": ee_xy_distance,
            "move_to_distance_reward": rewards,
            "move_to_z": ee_position[:, 2],
            "move_to_z_in_window": move_to_z_in_window,
            "target_motion_xy": target_motion_xy,
            "push_motion": push_motion,
            "container_xy_error": container_xy,
            "container_z_error": container_z,
            "projection_between": projection,
            "released": released,
            "ever_grasped": state.ever_grasped,
        },
    )
