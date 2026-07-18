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
) -> BatchedTaskResult:
    """Tensorized sparse predicates for all instruction families in production."""

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

    move_to = ee_xy_distance <= float(cfg.move_to_xy)
    success |= (instruction == INSTRUCTION_TO_ID["move_to_object"]) & move_to

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
    return BatchedTaskResult(
        rewards=rewards,
        success=success,
        terminated=terminated,
        diagnostics={
            "ee_xy_distance": ee_xy_distance,
            "target_motion_xy": target_motion_xy,
            "push_motion": push_motion,
            "container_xy_error": container_xy,
            "container_z_error": container_z,
            "projection_between": projection,
            "released": released,
            "ever_grasped": state.ever_grasped,
        },
    )
