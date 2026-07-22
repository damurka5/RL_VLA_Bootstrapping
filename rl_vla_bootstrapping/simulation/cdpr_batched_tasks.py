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
    "pick_up",
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
    # When set, the dense shaping is a single bounded inverse-polynomial on the
    # 3-D distance to a hover point (target XY, ``approach_z``) instead of the
    # XY term plus an unbounded linear Z penalty. The bounded form keeps the
    # reward magnitude in [0, weight], which conditions the GRPO group-relative
    # advantage far better than the old penalty whose scale dwarfed the XY term.
    distance_include_z: bool = False
    approach_z: float = 0.27

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
            distance_include_z=flag(
                "move_to_object_distance_include_z", False
            ),
            approach_z=number(
                "move_to_object_approach_z", 0.5 * (z_low + z_high)
            ),
        )


@dataclass(frozen=True)
class BatchedCatchReleaseDenseReward:
    """Dense GPU reward parameters for caught-object placement and pick-up."""

    distance_reward_scale: float = 0.08
    distance_reward_weight: float = 1.0
    distance_reward_exponent: float = 2.0
    placement_success_bonus: float = 2.0
    placement_failure_penalty: float = 1.0
    plate_radius: float = 0.091
    bowl_radius: float = 0.057
    container_z_tolerance: float = 0.12
    wrong_place_settle_margin: float = 0.025
    # Placement shaping was XY-only, so nothing held the gripper at release
    # height while the frozen prior pushes +Z; the policy drifts up and drops
    # from height. With distance_include_z the term becomes a bounded 3-D
    # distance to a hover point above the receptacle.
    distance_include_z: bool = False
    plate_release_height: float = 0.045
    bowl_release_height: float = 0.10
    pick_grasp_height_offset: float = 0.08
    pick_distance_window: float = 0.02
    pick_grasp_bonus: float = 1.0
    # Partial credit for bilateral finger contact before the grasp latches.
    # Without it there is no gradient between "hovering at the grasp point with
    # the gripper open" and a full persistent grasp, so pick-up plateaus.
    pick_contact_bonus: float = 0.25
    pick_lift_reward_weight: float = 1.0
    pick_lift_reward_scale: float = 0.05
    pick_lift_success_height: float = 0.05
    pick_success_bonus: float = 2.0

    @classmethod
    def from_metadata(
        cls, metadata: dict[str, Any] | None
    ) -> "BatchedCatchReleaseDenseReward":
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

        default_scale = number("move_to_object_xy_reward_scale", 0.08)
        default_weight = number(
            "move_to_object_distance_reward_weight", 1.0
        )
        default_success_bonus = number("success_bonus", 2.0)
        return cls(
            distance_reward_scale=max(
                number("catch_release_distance_reward_scale", default_scale),
                1.0e-6,
            ),
            distance_reward_weight=number(
                "catch_release_distance_reward_weight", default_weight
            ),
            distance_reward_exponent=max(
                number("distance_reward_exponent", 2.0), 1.0e-6
            ),
            placement_success_bonus=number(
                "placement_release_success_bonus", default_success_bonus
            ),
            placement_failure_penalty=max(
                number("placement_wrong_drop_penalty", 1.0), 0.0
            ),
            plate_radius=max(
                number("put_plate_xy_tolerance", 0.091), 1.0e-6
            ),
            bowl_radius=max(
                number("put_bowl_xy_tolerance", 0.057), 1.0e-6
            ),
            container_z_tolerance=max(
                number("put_container_z_tolerance", 0.12), 0.0
            ),
            wrong_place_settle_margin=max(
                number("placement_wrong_drop_settle_margin", 0.025), 0.0
            ),
            distance_include_z=flag(
                "catch_release_distance_include_z", False
            ),
            plate_release_height=number("put_plate_release_height", 0.045),
            bowl_release_height=number("put_bowl_release_height", 0.10),
            pick_contact_bonus=number("pick_contact_bonus", 0.25),
            pick_grasp_height_offset=number(
                "pick_grasp_height_offset", 0.08
            ),
            pick_distance_window=max(
                number("pick_distance_window", 0.02), 0.0
            ),
            pick_grasp_bonus=number("pick_grasp_bonus", 1.0),
            pick_lift_reward_weight=number(
                "pick_lift_reward_weight", 1.0
            ),
            pick_lift_reward_scale=max(
                number("pick_lift_reward_scale", 0.05), 1.0e-6
            ),
            pick_lift_success_height=max(
                number("pick_lift_success_height", 0.05), 1.0e-6
            ),
            pick_success_bonus=number(
                "pick_success_bonus", default_success_bonus
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
    target_rest_height: Any | None = None

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
        if self.target_rest_height is not None:
            if (
                not isinstance(self.target_rest_height, torch.Tensor)
                or tuple(self.target_rest_height.shape) != (batch_size,)
                or self.target_rest_height.device != device
            ):
                raise ValueError(
                    f"target_rest_height must have shape ({batch_size},) on {device}."
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
    ee_position: Any | None = None,
    target_xy: Any | None = None,
) -> Any:
    """Evaluate the inverse-polynomial move-to reward on GPU.

    With ``config.distance_include_z`` and the geometry supplied, the shaping
    is one bounded inverse-polynomial term on the 3-D distance to a hover point
    ``(target XY, approach_z)`` -- a single well-conditioned potential. Without
    it, the legacy XY term plus an unbounded linear Z-band penalty is used.
    """

    import torch

    if (
        bool(config.distance_include_z)
        and ee_position is not None
        and target_xy is not None
    ):
        hover_target = torch.stack(
            (
                target_xy[:, 0],
                target_xy[:, 1],
                torch.full_like(target_xy[:, 0], float(config.approach_z)),
            ),
            dim=-1,
        )
        distance_3d = torch.linalg.vector_norm(
            ee_position - hover_target, dim=-1
        )
        window_error = (
            distance_3d
            - torch.full_like(distance_3d, float(config.xy_window_high))
        ).clamp_min(0.0)
        normalized = window_error / max(float(config.xy_reward_scale), 1.0e-6)
        distance_reward = float(config.distance_reward_weight) / (
            1.0
            + normalized.pow(
                max(float(config.distance_reward_exponent), 1.0e-6)
            )
        )
        return distance_reward + success.to(dtype=distance_3d.dtype) * float(
            config.success_bonus
        )

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


def inverse_polynomial_distance_reward(
    distance: Any,
    *,
    window_high: Any,
    scale: float,
    weight: float,
    exponent: float,
) -> Any:
    """Dense reward used by move-to, placement, and pick-up GPU tasks."""

    import torch

    high = torch.as_tensor(
        window_high, dtype=distance.dtype, device=distance.device
    )
    error = (distance - high).clamp_min(0.0)
    normalized = error / max(float(scale), 1.0e-6)
    return float(weight) / (
        1.0 + normalized.pow(max(float(exponent), 1.0e-6))
    )


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
    include_relative_target: bool = False,
    goal_slots: Any | None = None,
) -> Any:
    """Build the SmolVLA/residual state as pure proprioception.

    ``include_relative_target=False`` (the deployment-faithful default) emits
    ``[ee_xyz, ee_yaw, gripper, 0]`` -- pure proprioception with a neutral pad
    in column 5. The historical layout put ``xy_distance`` to the target in that
    column, but that is privileged task geometry the real robot never observes,
    so it is deliberately removed: the policy must localize from the cameras and
    the language instruction instead.

    ``include_relative_target=True`` additionally appends the ground-truth 3-D
    end-effector->goal vector ``(dx, dy, dz)`` as columns 6..8 for the trainable
    residual only. THIS IS AN ORACLE and must stay off for any run whose policy
    has to transfer to a real robot -- it lets the residual servo on an exact
    target direction that will not exist at deployment, which trains a policy
    that collapses the moment the vector is withheld (e.g. at evaluation).

    ``goal_slots`` is the slot the reward actually shapes toward and defaults to
    ``target_slots``. Placement tasks must pass the receptacle (reference) slot:
    their reward is the gripper->receptacle distance while the target object is
    already in the gripper, so a target-relative vector would be a near-constant
    ``(0, 0, -0.08)`` and carry no information about where to go.
    """

    import torch
    import torch.nn.functional as functional

    target = gather_world_slots(object_positions, target_slots)
    goal = (
        target
        if goal_slots is None
        else gather_world_slots(object_positions, goal_slots)
    )
    relative_target = goal - ee_position
    if include_relative_target:
        # Column 5 is a neutral pad, not the target distance, so the frozen
        # SmolVLA's truncated six-dim view is exactly proprioception.
        columns = [
            ee_position,
            ee_yaw.reshape(-1, 1),
            gripper_opening.reshape(-1, 1),
            torch.zeros_like(ee_yaw.reshape(-1, 1)),
            relative_target,
        ]
    else:
        # Pure proprioception: [ee_xyz, ee_yaw, gripper, 0]. The historical
        # layout reported xy_distance to the target in column 5, but that is
        # privileged geometry unavailable on a real robot, so it is dropped in
        # favour of a neutral pad. The policy localizes from the cameras.
        columns = [
            ee_position,
            ee_yaw.reshape(-1, 1),
            gripper_opening.reshape(-1, 1),
            torch.zeros_like(ee_yaw.reshape(-1, 1)),
        ]
    state = torch.cat(columns, dim=-1).to(dtype=torch.float32)
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
    catch_release_dense_reward: BatchedCatchReleaseDenseReward | None = None,
    bilateral_contact: Any | None = None,
) -> BatchedTaskResult:
    """Tensorized task predicates plus optional dense manipulation rewards."""

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
    ee_reference_xy_distance = torch.linalg.vector_norm(
        reference[:, :2] - ee_position[:, :2], dim=-1
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
    is_bowl = instruction == INSTRUCTION_TO_ID["put_into_bowl"]
    is_plate = instruction == INSTRUCTION_TO_ID["put_into_plate"]
    is_container = is_bowl | is_plate
    target_rest_height = (
        state.target_rest_height
        if state.target_rest_height is not None
        else torch.zeros_like(state.support_surface_z)
    )
    target_has_settled = torch.zeros_like(is_container)
    if catch_release_dense_reward is None:
        container_radius = torch.full_like(
            container_xy, float(cfg.container_xy)
        )
        container_z_tolerance = float(cfg.container_z)
        minimum_target_motion = float(cfg.minimum_target_motion)
    else:
        container_radius = torch.where(
            is_plate,
            torch.full_like(
                container_xy,
                float(catch_release_dense_reward.plate_radius),
            ),
            torch.full_like(
                container_xy,
                float(catch_release_dense_reward.bowl_radius),
            ),
        )
        container_z_tolerance = float(
            catch_release_dense_reward.container_z_tolerance
        )
        minimum_target_motion = 0.0
        target_has_settled = target[:, 2] <= (
            state.support_surface_z
            + target_rest_height
            + float(catch_release_dense_reward.wrong_place_settle_margin)
        )
    container_ok = (
        (container_xy <= container_radius)
        & (container_z <= float(cfg.container_z))
        & (container_z <= container_z_tolerance)
        & (target_motion_xy >= minimum_target_motion)
        & state.ever_grasped
        & released
        & (
            target_has_settled
            if catch_release_dense_reward is not None
            else torch.ones_like(is_container)
        )
    )
    success |= is_container & container_ok

    target_lift = (
        target[:, 2] - state.initial_target_positions[:, 2]
    ).clamp_min(0.0)
    is_pick_up = instruction == INSTRUCTION_TO_ID["pick_up"]
    pick_success = (
        state.grasped
        & (
            target_lift
            >= float(
                catch_release_dense_reward.pick_lift_success_height
                if catch_release_dense_reward is not None
                else 0.05
            )
        )
    )
    success |= is_pick_up & pick_success

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
    wrong_place_settled = torch.zeros_like(success)
    if catch_release_dense_reward is not None:
        wrong_place_settled = (
            is_container
            & state.ever_grasped
            & ~state.grasped
            & target_has_settled
            & ~container_ok
        )
    terminated = success | wrong_place_settled | timeout
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
            ee_position=ee_position,
            target_xy=target[:, :2],
        )
        rewards = torch.where(is_move_to, dense_move_to, rewards)
    dense_target_distance = ee_xy_distance
    pick_grasp_point = target.clone()
    if catch_release_dense_reward is not None:
        pick_grasp_point[:, 2] += float(
            catch_release_dense_reward.pick_grasp_height_offset
        )
        pick_grasp_distance = torch.linalg.vector_norm(
            pick_grasp_point - ee_position, dim=-1
        )
        # Bounded 3-D shaping toward a hover point above the receptacle keeps
        # the gripper at release height; the XY-only term let it drift up (the
        # frozen prior carries a +Z bias) and drop from height.
        placement_distance = ee_reference_xy_distance
        if bool(catch_release_dense_reward.distance_include_z):
            release_height = torch.where(
                is_bowl,
                torch.full_like(
                    ee_reference_xy_distance,
                    float(catch_release_dense_reward.bowl_release_height),
                ),
                torch.full_like(
                    ee_reference_xy_distance,
                    float(catch_release_dense_reward.plate_release_height),
                ),
            )
            # release_height is expressed in the HELD-OBJECT frame: the reset
            # places the carried object at reference_z + release_height. This
            # distance is against ee_position, which tracks the end-effector
            # body riding pick_grasp_height_offset above the grasp point, so the
            # offset must be added or the target sits a whole gripper-length too
            # low -- for the plate that landed at 0.205 m, below the 0.25 m
            # controller floor, i.e. a point the policy could never reach.
            # pick_up already applies the same offset to its grasp point.
            placement_hover = torch.stack(
                (
                    reference[:, 0],
                    reference[:, 1],
                    reference[:, 2]
                    + release_height
                    + float(catch_release_dense_reward.pick_grasp_height_offset),
                ),
                dim=-1,
            )
            placement_distance = torch.linalg.vector_norm(
                ee_position - placement_hover, dim=-1
            )
        placement_distance_reward = inverse_polynomial_distance_reward(
            placement_distance,
            window_high=container_radius,
            scale=catch_release_dense_reward.distance_reward_scale,
            weight=catch_release_dense_reward.distance_reward_weight,
            exponent=catch_release_dense_reward.distance_reward_exponent,
        )
        placement_reward = (
            placement_distance_reward
            + success.to(dtype=ee_position.dtype)
            * float(catch_release_dense_reward.placement_success_bonus)
            - wrong_place_settled.to(dtype=ee_position.dtype)
            * float(catch_release_dense_reward.placement_failure_penalty)
        )
        rewards = torch.where(is_container, placement_reward, rewards)

        pick_distance_reward = inverse_polynomial_distance_reward(
            pick_grasp_distance,
            window_high=float(
                catch_release_dense_reward.pick_distance_window
            ),
            scale=catch_release_dense_reward.distance_reward_scale,
            weight=catch_release_dense_reward.distance_reward_weight,
            exponent=catch_release_dense_reward.distance_reward_exponent,
        )
        normalized_lift = (
            target_lift
            / float(catch_release_dense_reward.pick_lift_reward_scale)
        ).clamp(0.0, 1.0)
        # Partial credit for touching the object with both pads bridges the
        # cliff between approaching and a latched persistent grasp.
        contact_credit = (
            torch.zeros_like(pick_distance_reward)
            if bilateral_contact is None
            else bilateral_contact.to(dtype=ee_position.dtype)
            * float(catch_release_dense_reward.pick_contact_bonus)
        )
        pick_reward = (
            pick_distance_reward
            + contact_credit
            + state.grasped.to(dtype=ee_position.dtype)
            * float(catch_release_dense_reward.pick_grasp_bonus)
            + normalized_lift
            * state.grasped.to(dtype=ee_position.dtype)
            * float(catch_release_dense_reward.pick_lift_reward_weight)
            + success.to(dtype=ee_position.dtype)
            * float(catch_release_dense_reward.pick_success_bonus)
        )
        rewards = torch.where(is_pick_up, pick_reward, rewards)
        dense_target_distance = torch.where(
            is_container,
            ee_reference_xy_distance,
            torch.where(is_pick_up, pick_grasp_distance, ee_xy_distance),
        )
    else:
        pick_grasp_distance = torch.linalg.vector_norm(
            target - ee_position, dim=-1
        )
    return BatchedTaskResult(
        rewards=rewards,
        success=success,
        terminated=terminated,
        diagnostics={
            "ee_xy_distance": ee_xy_distance,
            "ee_reference_xy_distance": ee_reference_xy_distance,
            "dense_target_distance": dense_target_distance,
            "move_to_distance_reward": rewards,
            "move_to_z": ee_position[:, 2],
            "move_to_z_in_window": move_to_z_in_window,
            "target_motion_xy": target_motion_xy,
            "target_lift": target_lift,
            "pick_grasp_distance": pick_grasp_distance,
            "push_motion": push_motion,
            "container_xy_error": container_xy,
            "container_xy_radius": container_radius,
            "container_z_error": container_z,
            "target_has_settled": target_has_settled,
            "projection_between": projection,
            "released": released,
            "ever_grasped": state.ever_grasped,
            "wrong_place_drop": wrong_place_settled,
        },
    )
