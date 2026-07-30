from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rl_vla_bootstrapping.policy.rank_local_grpo import (
    RankLocalGroupLayout,
    torch_group_advantages,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    ACTIVE_INSTRUCTION_TYPES,
    INSTRUCTION_TO_ID,
    BatchedCatchReleaseDenseReward,
    BatchedMoveToDistanceReward,
    BatchedTaskState,
    BatchedTaskThresholds,
    build_smolvla_state_tensor,
    evaluate_active_sparse_tasks,
    gather_world_slots,
)
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
    ACTIVE_CDPR_CATALOGS,
    BOWL_CATALOG,
    CATALOG_TO_ID,
    GRASPABLE_CDPR_CATALOGS,
    INACTIVE_CATALOG_ID,
    OBJECT_VARIANTS,
    PLATE_CATALOG,
    catalog_id,
)


_SHELL_COUNTS = (5, 5, 5, 5, 8, 8, 8, 8, 5)
_SHELL_HORIZON_LOW = (1, 1, 2, 3, 5, 7, 14, 21)
_SHELL_HORIZON_HIGH = (1, 2, 3, 4, 6, 13, 20, 32)
_SHELL_ACTION_LOW = (1, 4, 7, 11, 17, 25, 53, 81)
_SHELL_ACTION_HIGH = (2, 6, 10, 16, 24, 52, 80, 128)
_REST_HEIGHT = tuple(
    OBJECT_VARIANTS[catalog].rest_height for catalog in ACTIVE_CDPR_CATALOGS
)
_FITTED_GRIPPER = tuple(
    OBJECT_VARIANTS[catalog].fitted_gripper_opening
    for catalog in ACTIVE_CDPR_CATALOGS
)
_PLATE_CATALOG_ID = CATALOG_TO_ID[PLATE_CATALOG]
_BOWL_CATALOG_ID = CATALOG_TO_ID[BOWL_CATALOG]
_GRASP_MIN_PAD_FORCE_N = 0.05
_GRASP_PERSISTENCE_STEPS = 2
_GRASP_MAX_RELATIVE_POSITION_SLIP_M = 0.008
_GRASP_MAX_RELATIVE_ORIENTATION_SLIP_RAD = 0.15
_GRASP_MIN_LIFT_M = 0.015


def resolve_mjwarp_instruction_ids(
    instruction_types: Sequence[str] | None,
) -> tuple[int, ...]:
    names = tuple(instruction_types or ACTIVE_INSTRUCTION_TYPES)
    if not names:
        raise ValueError("At least one MJWarp instruction type is required.")
    resolved: list[int] = []
    seen: set[int] = set()
    for raw_name in names:
        name = str(raw_name).strip().lower().replace("-", "_")
        if name not in INSTRUCTION_TO_ID:
            raise ValueError(
                f"Unsupported MJWarp instruction type {raw_name!r}; supported "
                f"values are {', '.join(ACTIVE_INSTRUCTION_TYPES)}."
            )
        task_id = int(INSTRUCTION_TO_ID[name])
        if task_id not in seen:
            seen.add(task_id)
            resolved.append(task_id)
    return tuple(resolved)


def resolve_mjwarp_catalog_ids(
    allowed_objects: Sequence[str] | None,
) -> tuple[int, ...]:
    names = tuple(allowed_objects or ACTIVE_CDPR_CATALOGS)
    if not names:
        raise ValueError("At least one MJWarp object catalog is required.")
    resolved: list[int] = []
    seen: set[int] = set()
    for name in names:
        object_id = int(catalog_id(name))
        if object_id not in seen:
            seen.add(object_id)
            resolved.append(object_id)
    return tuple(resolved)


def _quaternion_multiply(left: Any, right: Any) -> Any:
    """Hamilton product for batched MuJoCo wxyz quaternions."""

    import torch

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _relative_quaternion(parent: Any, child: Any) -> Any:
    conjugate = parent.clone()
    conjugate[..., 1:] *= -1.0
    result = _quaternion_multiply(conjugate, child)
    return result / result.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _cosine_2d(vector: Any, reference: Any) -> Any:
    """Per-row cosine between two 2-D vectors; 0 when either is ~zero."""

    numerator = (vector * reference).sum(dim=-1)
    denominator = (
        vector.norm(dim=-1).clamp_min(1.0e-8)
        * reference.norm(dim=-1).clamp_min(1.0e-8)
    )
    return numerator / denominator


@dataclass
class RankLocalCurriculum:
    device: Any
    promotion_success: float = 0.80
    demotion_success: float = -1.0
    validation_rollouts_per_shell: int = 50
    min_updates: int = 1
    saturation_abort_threshold: float = 1.01
    current_shell: Any | None = None
    updates: int = 0

    def __post_init__(self) -> None:
        import torch

        if self.current_shell is None:
            self.current_shell = torch.zeros(
                (len(ACTIVE_INSTRUCTION_TYPES),),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.current_shell = torch.as_tensor(
                self.current_shell, dtype=torch.int64, device=self.device
            ).reshape(len(ACTIVE_INSTRUCTION_TYPES))
        self._shell_max = torch.tensor(
            [count - 1 for count in _SHELL_COUNTS],
            dtype=torch.int64,
            device=self.device,
        )
        count = len(ACTIVE_INSTRUCTION_TYPES)
        self.train_updates = torch.zeros(
            (count,), dtype=torch.int64, device=self.device
        )
        self.last_promoted_update = torch.zeros_like(self.train_updates)
        self.pending_success_sum = torch.zeros(
            (count,), dtype=torch.float32, device=self.device
        )
        self.pending_rollouts = torch.zeros(
            (count,), dtype=torch.int64, device=self.device
        )
        self.validation_success = torch.zeros_like(self.pending_success_sum)
        self.validation_rollouts = torch.zeros_like(self.pending_rollouts)
        self.action_saturation = torch.zeros_like(self.pending_success_sum)

    def update_once_per_optimizer_update(
        self,
        *,
        group_instruction_ids: Any,
        group_shell_ids: Any,
        candidate_success: Any,
    ) -> dict[str, float]:
        """Aggregate curriculum evidence once, then broadcast rank-0 state."""

        import torch
        import torch.distributed as dist

        task_ids = group_instruction_ids.to(dtype=torch.int64).reshape(-1)
        shell_ids = group_shell_ids.to(dtype=torch.int64).reshape(-1)
        outcomes = candidate_success.to(dtype=torch.float32)
        if outcomes.ndim != 2:
            raise ValueError(
                "candidate_success must have shape [groups, candidates]."
            )
        candidates_per_group = int(outcomes.shape[1])
        group_successes = outcomes.sum(dim=1)
        sums = torch.zeros(
            (len(ACTIVE_INSTRUCTION_TYPES),),
            dtype=torch.float32,
            device=self.device,
        )
        counts = torch.zeros(
            sums.shape, dtype=torch.int64, device=self.device
        )
        train_presence = torch.zeros_like(counts)
        frontier = self.current_shell.index_select(0, task_ids)
        frontier_sample = shell_ids == frontier
        sums.scatter_add_(
            0,
            task_ids,
            group_successes
            * frontier_sample.to(dtype=group_successes.dtype),
        )
        counts.scatter_add_(
            0,
            task_ids,
            frontier_sample.to(dtype=torch.int64) * candidates_per_group,
        )
        train_presence.scatter_add_(
            0, task_ids, frontier_sample.to(dtype=torch.int64)
        )
        packed = torch.cat(
            (
                sums,
                counts.to(dtype=torch.float32),
                train_presence.to(dtype=torch.float32),
            ),
            dim=0,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        sums, count_values, presence_values = packed.chunk(3)
        counts = count_values.to(dtype=torch.int64)
        train_presence = presence_values.to(dtype=torch.int64)
        self.pending_success_sum.add_(sums)
        self.pending_rollouts.add_(counts)
        self.train_updates.add_((train_presence > 0).to(dtype=torch.int64))
        self.updates += 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            required_rollouts = max(
                1, int(self.validation_rollouts_per_shell)
            )
            for instruction_index in range(len(ACTIVE_INSTRUCTION_TYPES)):
                rollouts = int(
                    self.pending_rollouts[instruction_index].item()
                )
                if rollouts < required_rollouts:
                    continue
                success_rate = float(
                    (
                        self.pending_success_sum[instruction_index]
                        / max(1, rollouts)
                    ).item()
                )
                self.validation_success[instruction_index] = success_rate
                self.validation_rollouts[instruction_index] = rollouts
                self.pending_success_sum[instruction_index] = 0.0
                self.pending_rollouts[instruction_index] = 0

                shell = int(self.current_shell[instruction_index].item())
                if (
                    success_rate <= float(self.demotion_success)
                    and shell > 0
                ):
                    self.current_shell[instruction_index] = shell - 1
                    continue
                can_validate = (
                    int(self.train_updates[instruction_index].item())
                    - int(self.last_promoted_update[instruction_index].item())
                    >= max(1, int(self.min_updates))
                )
                saturation_ok = (
                    float(self.action_saturation[instruction_index].item())
                    < float(self.saturation_abort_threshold)
                )
                if (
                    can_validate
                    and saturation_ok
                    and success_rate >= float(self.promotion_success)
                    and shell < int(self._shell_max[instruction_index].item())
                ):
                    self.current_shell[instruction_index] = shell + 1
                    self.last_promoted_update[instruction_index] = (
                        self.train_updates[instruction_index]
                    )
        if dist.is_available() and dist.is_initialized():
            canonical_tensors = (
                self.current_shell,
                self.train_updates,
                self.last_promoted_update,
                self.pending_success_sum,
                self.pending_rollouts,
                self.validation_success,
                self.validation_rollouts,
                self.action_saturation,
            )
            canonical = torch.cat(
                [
                    tensor.to(dtype=torch.float64).reshape(-1)
                    for tensor in canonical_tensors
                ],
                dim=0,
            )
            dist.broadcast(canonical, src=0)
            offset = 0
            for tensor in canonical_tensors:
                size = int(tensor.numel())
                tensor.copy_(
                    canonical[offset : offset + size]
                    .reshape(tensor.shape)
                    .to(dtype=tensor.dtype)
                )
                offset += size
        output: dict[str, float] = {}
        for index, name in enumerate(ACTIVE_INSTRUCTION_TYPES):
            pending_rate = float(
                (
                    self.pending_success_sum[index]
                    / self.pending_rollouts[index].clamp_min(1)
                ).item()
            )
            output[f"curriculum/{name}/success_rate"] = float(
                self.validation_success[index].item()
            )
            output[f"curriculum/{name}/rollouts"] = float(
                self.validation_rollouts[index].item()
            )
            output[f"curriculum/{name}/pending_success_rate"] = pending_rate
            output[f"curriculum/{name}/pending_rollouts"] = float(
                self.pending_rollouts[index].item()
            )
            output[f"curriculum/{name}/train_updates"] = float(
                self.train_updates[index].item()
            )
            output[f"curriculum/{name}/shell"] = float(
                self.current_shell[index].item()
            )
        return output

    def snapshot(self) -> dict[str, Any]:
        state = {}
        for index, name in enumerate(ACTIVE_INSTRUCTION_TYPES):
            state[name] = {
                "active_shell": int(self.current_shell[index].item()),
                "validation_success": float(
                    self.validation_success[index].item()
                ),
                "train_updates": int(self.train_updates[index].item()),
                "last_promoted_update": int(
                    self.last_promoted_update[index].item()
                ),
                "validation_rollouts": int(
                    self.validation_rollouts[index].item()
                ),
                "action_saturation": float(
                    self.action_saturation[index].item()
                ),
                "pending_success_sum": float(
                    self.pending_success_sum[index].item()
                ),
                "pending_rollouts": int(
                    self.pending_rollouts[index].item()
                ),
            }
        return {
            "profile": "smolvla_complex_v1_mjwarp",
            "updates": int(self.updates),
            "current_shell": {
                name: int(value)
                for name, value in zip(
                    ACTIVE_INSTRUCTION_TYPES,
                    self.current_shell.detach().cpu().tolist(),
                )
            },
            "frontier": {
                "config": {
                    "promotion_success": float(self.promotion_success),
                    "demotion_success": float(self.demotion_success),
                    "validation_rollouts_per_shell": int(
                        self.validation_rollouts_per_shell
                    ),
                    "min_train_updates_before_validation": int(
                        self.min_updates
                    ),
                    "max_shell_jump": 1,
                    "saturation_abort_threshold": float(
                        self.saturation_abort_threshold
                    ),
                },
                "state": state,
            },
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        raw_snapshot = dict(snapshot or {})
        frontier = raw_snapshot.get("frontier")
        frontier_state = (
            dict(frontier.get("state") or {})
            if isinstance(frontier, Mapping)
            else {}
        )
        values = dict(raw_snapshot.get("current_shell") or {})
        for index, name in enumerate(ACTIVE_INSTRUCTION_TYPES):
            raw_state = dict(frontier_state.get(name) or {})
            shell_value = raw_state.get(
                "active_shell", values.get(name, None)
            )
            if shell_value is not None:
                self.current_shell[index] = max(
                    0, min(int(shell_value), _SHELL_COUNTS[index] - 1)
                )
            for key, target, dtype in (
                ("train_updates", self.train_updates, int),
                (
                    "last_promoted_update",
                    self.last_promoted_update,
                    int,
                ),
                (
                    "pending_success_sum",
                    self.pending_success_sum,
                    float,
                ),
                ("pending_rollouts", self.pending_rollouts, int),
                (
                    "validation_success",
                    self.validation_success,
                    float,
                ),
                (
                    "validation_rollouts",
                    self.validation_rollouts,
                    int,
                ),
                (
                    "action_saturation",
                    self.action_saturation,
                    float,
                ),
            ):
                if key in raw_state:
                    target[index] = dtype(raw_state[key])
        self.updates = int(
            raw_snapshot.get(
                "updates",
                int(self.train_updates.max().item()),
            )
        )


@dataclass(frozen=True)
class BatchedReset:
    instructions: tuple[str, ...]
    task_state: BatchedTaskState
    group_instruction_ids: Any
    group_shell_ids: Any
    horizons: Any
    physical_grasp: Any
    grasp_eligible: Any
    bilateral_contact_steps: Any
    previous_relative_position: Any
    previous_relative_quaternion: Any
    target_rest_height: Any
    group_ids: Any
    group_target_catalog_ids: Any | None = None


class BatchedReverseFrontierResetter:
    def __init__(
        self,
        *,
        backend: Any,
        layout: RankLocalGroupLayout,
        curriculum: RankLocalCurriculum,
        rank: int,
        base_seed: int,
        instruction_types: Sequence[str] | None = None,
        allowed_objects: Sequence[str] | None = None,
        frontier_probability: float = 0.80,
        rehearsal_probability: float = 0.20,
        support_surface_z: float = 0.15,
        balanced_target_catalogs: bool = False,
        task_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        layout.validate()
        self.backend = backend
        self.layout = layout
        self.curriculum = curriculum
        self.rank = int(rank)
        self.base_seed = int(base_seed)
        self.frontier_probability = min(
            1.0, max(0.0, float(frontier_probability))
        )
        self.rehearsal_probability = float(rehearsal_probability)
        self.support_surface_z = float(support_surface_z)
        self.balanced_target_catalogs = bool(balanced_target_catalogs)
        metadata = dict(task_metadata or {})

        def flag(key: str, default: bool = False) -> bool:
            value = metadata.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        def number(key: str, default: float) -> float:
            try:
                return float(metadata.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        def bounds(
            key: str, default: tuple[float, float]
        ) -> tuple[float, float]:
            raw = metadata.get(key, default)
            try:
                low, high = tuple(float(value) for value in raw)
            except (TypeError, ValueError):
                low, high = default
            return (min(low, high), max(low, high))

        self.random_workspace_gripper_start = flag(
            "random_workspace_gripper_start", False
        )
        self.force_caught_container_start = flag(
            "placement_start_with_caught_object", False
        )
        self.workspace_x_bounds = bounds(
            "ee_workspace_x_bounds", (-0.28, 0.28)
        )
        self.workspace_y_bounds = bounds(
            "ee_workspace_y_bounds", (-0.28, 0.28)
        )
        self.workspace_z_bounds = bounds(
            "ee_workspace_z_bounds", (0.30, 0.48)
        )
        self.random_start_min_goal_distance = max(
            number("random_workspace_min_goal_xy_distance", 0.10), 0.0
        )
        # Approach curriculum: an upper bound on the EE start's XY distance from
        # its goal, set per-update by the trainer. inf disables the cap and
        # reproduces the historical full-workspace start distribution.
        self.random_start_max_goal_distance = float("inf")
        self.random_start_horizon_low = max(
            1, int(round(number("random_workspace_horizon_low", 21)))
        )
        self.random_start_horizon_high = max(
            self.random_start_horizon_low,
            int(round(number("random_workspace_horizon_high", 32))),
        )
        # Curriculum-coupled horizon: scale the episode length (in policy
        # decisions) with the current start-distance cap, since the batched
        # SmolVLA forward runs every decision -- a short reach needs far fewer
        # decisions than the fixed 32, so a short horizon cuts inference cost and
        # yields more group-starts per update. Interpolates linearly from
        # horizon_min at the initial cap to horizon_max at the final cap; falls
        # back to the fixed [low, high] sampling when disabled or the cap is inf.
        self.curriculum_horizon_coupling = flag(
            "curriculum_horizon_coupling_enabled", False
        )
        self.curriculum_horizon_min = max(
            1, int(round(number("curriculum_horizon_min", 8)))
        )
        self.curriculum_horizon_max = max(
            self.curriculum_horizon_min,
            int(round(number("curriculum_horizon_max", 32))),
        )
        self._horizon_cap_initial = max(
            number("random_workspace_start_distance_initial", 0.06), 1.0e-6
        )
        self._horizon_cap_final = max(
            number("random_workspace_start_distance_final", 0.34),
            self._horizon_cap_initial,
        )
        # Make the approach-curriculum cap bound the 3-D start distance, not just
        # XY. The rewards are 3-D (a bounded potential on the distance to a hover
        # point) but the cap only ever pulled the start in XY, so with a wide
        # ee_workspace_z_bounds a "3 cm" start was up to 0.25 m away once Z was
        # counted -- the easy regime the curriculum exists to create did not
        # exist, and the policy had to solve a near-full-reach descent to score
        # at the initial cap. With this on, the start Z is additionally confined
        # to the goal's shaping height +/- sqrt(cap^2 - dxy^2), so the cap means
        # what it says. The Z spread returns as the cap widens, so the policy
        # still has to learn descent -- just not before it can do anything else.
        self.curriculum_cap_includes_z = flag(
            "curriculum_cap_includes_z", False
        )
        self.move_to_approach_z = number("move_to_object_approach_z", 0.27)
    # 0.0075, the value MEASURED from the MJCF by
    # cdpr_gripper_geometry.load_cdpr_gripper_geometry: ee_platform sits 0.08 m
    # ABOVE ee_base and left_finger_pad sits 0.0875 m below that, so the pads are
    # 0.0075 m below the body ee_position tracks. The old 0.08 default was the
    # ee_platform offset, not the pad offset, and it put the reward's grasp point
    # 7.25 cm above the object -- pick_up then solved that reward accurately for
    # 10M steps while its grasp rate DECAYED and successes went 8/1024 -> 0/1024.
        self.pick_grasp_height_offset = number(
            "pick_grasp_height_offset", 0.0075
        )
        self.plate_release_height = number("put_plate_release_height", 0.045)
        self.bowl_release_height = number("put_bowl_release_height", 0.10)
        # Reset-on-drift: end a move-to trajectory once the EE has moved AWAY
        # from its goal for this many consecutive env steps, and freeze that
        # trajectory's GRPO return to the penalty. The return is the last active
        # step's reward, so a bare truncation would score the drifter at its
        # closer pre-drift point and reward drifting -- the penalty makes drift
        # explicitly bad while stopping the wasted steps (the freed record budget
        # is refilled with fresh starts). Disabled by default.
        self.reset_on_drift = flag("reset_on_drift_enabled", False)
        self.reset_on_drift_patience = max(
            1, int(round(number("reset_on_drift_patience", 4)))
        )
        self.reset_on_drift_penalty = float(
            number("reset_on_drift_penalty", 0.0)
        )
        # Deadband so sub-millimetre numerical jitter is not counted as drift.
        self.reset_on_drift_min_increase = max(
            0.0, number("reset_on_drift_min_increase_m", 0.001)
        )
        scene_min = min(4, max(1, int(number("min_scene_objects", 4))))
        scene_max = min(4, max(scene_min, int(number("max_scene_objects", 4))))
        self.scene_object_bounds = (scene_min, scene_max)
        self.scene_object_range = (scene_min, scene_max)
        self.torch = backend.torch
        self.device = backend.device
        instruction_ids = resolve_mjwarp_instruction_ids(instruction_types)
        catalog_ids = resolve_mjwarp_catalog_ids(allowed_objects)
        graspable_ids = tuple(
            value
            for value in catalog_ids
            if ACTIVE_CDPR_CATALOGS[value] in GRASPABLE_CDPR_CATALOGS
        )
        if not graspable_ids:
            move_to_id = INSTRUCTION_TO_ID["move_to_object"]
            if any(task_id != move_to_id for task_id in instruction_ids):
                raise ValueError(
                    "Non-move MJWarp tasks require at least one graspable "
                    "catalog in --allowed-objects."
                )
            # The tensor is still sampled before the task-specific torch.where;
            # for a move-only run its value is never selected.
            graspable_ids = catalog_ids
        # Per-instruction approach-curriculum caps; inf everywhere means the cap
        # is disabled. set_random_start_max_goal_distance rewrites this.
        self._start_cap_table = self.torch.full(
            (len(ACTIVE_INSTRUCTION_TYPES),),
            float("inf"),
            dtype=self.torch.float32,
            device=self.device,
        )
        self.instruction_ids = self.torch.tensor(
            instruction_ids, dtype=self.torch.int64, device=self.device
        )
        self.target_catalog_ids = self.torch.tensor(
            catalog_ids, dtype=self.torch.int64, device=self.device
        )
        self.graspable_target_catalog_ids = self.torch.tensor(
            graspable_ids, dtype=self.torch.int64, device=self.device
        )

    def set_scene_object_range(self, low: int, high: int) -> None:
        """Clamp the per-group active object count for curriculum schedules."""

        scene_min, scene_max = self.scene_object_bounds
        low = min(scene_max, max(scene_min, int(low)))
        high = min(scene_max, max(low, int(high)))
        self.scene_object_range = (low, high)

    def set_random_start_max_goal_distance(
        self, value: float | Mapping[int, float]
    ) -> None:
        """Cap the EE start's XY distance from its goal (approach curriculum).

        Accepts a scalar (one cap for every instruction) or a mapping from
        instruction id to cap. The per-instruction form exists because a shared
        cap driven by the mixed pass rate advances on the easiest task in the
        run: with put_into_plate passing and pick_up at zero, one global gate
        widens the starts for pick_up too and starves it of the close starts it
        still needs. Each instruction now climbs on its own success.

        A non-positive or non-finite value disables the cap for that
        instruction, restoring the full-workspace start distribution.
        """

        def sanitize(raw: Any) -> float:
            try:
                distance = float(raw)
            except (TypeError, ValueError):
                return float("inf")
            if not distance > 0.0 or distance == float("inf"):
                return float("inf")
            return distance

        if isinstance(value, Mapping):
            caps = {int(key): sanitize(raw) for key, raw in value.items()}
            self._start_cap_table = self.torch.tensor(
                [
                    caps.get(index, float("inf"))
                    for index in range(len(ACTIVE_INSTRUCTION_TYPES))
                ],
                dtype=self.torch.float32,
                device=self.device,
            )
            finite = [cap for cap in caps.values() if cap != float("inf")]
            # Scalar mirror kept for diagnostics and for callers that still read
            # the attribute; the tensor is what reset() actually applies.
            self.random_start_max_goal_distance = (
                max(finite) if finite else float("inf")
            )
            return
        distance = sanitize(value)
        self.random_start_max_goal_distance = distance
        self._start_cap_table = self.torch.full(
            (len(ACTIVE_INSTRUCTION_TYPES),),
            distance,
            dtype=self.torch.float32,
            device=self.device,
        )

    def _generator(self, update_index: int, round_index: int) -> Any:
        generator = self.torch.Generator(device=self.device)
        seed = (
            self.base_seed
            + self.rank * 1_000_003
            + int(update_index) * 10_000_019
            + int(round_index) * 100_003
        )
        generator.manual_seed(int(seed))
        return generator

    def reset(self, *, update_index: int, round_index: int) -> BatchedReset:
        torch = self.torch
        generator = self._generator(update_index, round_index)
        groups = int(self.layout.groups_per_rank)
        group_size = int(self.layout.group_size)
        worlds = int(self.layout.worlds_per_rank)
        group_ids = torch.arange(
            groups, dtype=torch.int64, device=self.device
        )
        configured_tasks = self.instruction_ids
        eligible_tasks = (
            configured_tasks
            if self.random_workspace_gripper_start
            else configured_tasks[
                self.curriculum.current_shell.index_select(
                    0, configured_tasks
                )
                < self.curriculum._shell_max.index_select(
                    0, configured_tasks
                )
            ]
        )
        if int(eligible_tasks.numel()) == 0:
            eligible_tasks = configured_tasks
        sampled_task_index = torch.randint(
            0,
            int(eligible_tasks.numel()),
            (groups,),
            generator=generator,
            device=self.device,
        )
        task_group = eligible_tasks.index_select(0, sampled_task_index)
        frontier_shell = (
            torch.zeros_like(task_group)
            if self.random_workspace_gripper_start
            else self.curriculum.current_shell.index_select(0, task_group)
        )
        rehearsal = (
            torch.rand((groups,), generator=generator, device=self.device)
            >= self.frontier_probability
        ) & (self.rehearsal_probability > 0.0) & (frontier_shell > 0)
        rehearsal_shell = torch.floor(
            torch.rand((groups,), generator=generator, device=self.device)
            * (frontier_shell.to(dtype=torch.float32) + 1.0)
        ).to(dtype=torch.int64)
        shell_group = torch.where(rehearsal, rehearsal_shell, frontier_shell)
        horizon_low = torch.tensor(
            _SHELL_HORIZON_LOW, dtype=torch.int64, device=self.device
        ).index_select(0, shell_group)
        horizon_high = torch.tensor(
            _SHELL_HORIZON_HIGH, dtype=torch.int64, device=self.device
        ).index_select(0, shell_group)
        horizon_group = horizon_low + torch.floor(
            torch.rand((groups,), generator=generator, device=self.device)
            * (horizon_high - horizon_low + 1).to(dtype=torch.float32)
        ).to(dtype=torch.int64)
        action_low = torch.tensor(
            _SHELL_ACTION_LOW, dtype=torch.int64, device=self.device
        ).index_select(0, shell_group)
        action_high = torch.tensor(
            _SHELL_ACTION_HIGH, dtype=torch.int64, device=self.device
        ).index_select(0, shell_group)
        target_action_low = torch.maximum(
            action_low, (horizon_group - 1) * 4 + 1
        )
        target_action_high = torch.minimum(action_high, horizon_group * 4)
        target_action_steps = target_action_low + torch.floor(
            torch.rand((groups,), generator=generator, device=self.device)
            * (target_action_high - target_action_low + 1).to(
                dtype=torch.float32
            )
        ).to(dtype=torch.int64)
        if self.random_workspace_gripper_start:
            # Per-group cap: each group runs one instruction, and each
            # instruction carries its own approach-curriculum cap.
            cap_group = self._start_cap_table.index_select(0, task_group)
            cap_active = torch.isfinite(cap_group) & (cap_group > 0.0)
            if self.curriculum_horizon_coupling and bool(cap_active.any().item()):
                span = max(
                    self._horizon_cap_final - self._horizon_cap_initial, 1.0e-6
                )
                frac = (
                    (cap_group - self._horizon_cap_initial) / span
                ).clamp(0.0, 1.0)
                coupled = (
                    self.curriculum_horizon_min
                    + frac
                    * float(
                        self.curriculum_horizon_max
                        - self.curriculum_horizon_min
                    )
                ).round().to(dtype=torch.int64)
                # Groups whose instruction has no cap keep the sampled horizon.
                uncapped_horizon = torch.randint(
                    self.random_start_horizon_low,
                    self.random_start_horizon_high + 1,
                    (groups,),
                    generator=generator,
                    device=self.device,
                    dtype=torch.int64,
                )
                horizon_group = torch.where(
                    cap_active, coupled, uncapped_horizon
                )
            else:
                horizon_group = torch.randint(
                    self.random_start_horizon_low,
                    self.random_start_horizon_high + 1,
                    (groups,),
                    generator=generator,
                    device=self.device,
                    dtype=torch.int64,
                )
            target_action_steps = (
                horizon_group * 4
            )
        travel_group = (
            (target_action_steps.to(dtype=torch.float32) - 0.5)
            .clamp_min(0.5)
            * 0.015
            * 0.44
        )

        target_pool = self.target_catalog_ids
        if self.balanced_target_catalogs:
            target_positions = (
                torch.arange(
                    groups, dtype=torch.int64, device=self.device
                )
                + self.rank * groups
                + int(round_index) * groups
            ) % int(target_pool.numel())
        else:
            target_positions = torch.randint(
                0,
                int(target_pool.numel()),
                (groups,),
                generator=generator,
                device=self.device,
            )
        move_target_catalog = target_pool.index_select(0, target_positions)
        graspable_pool = self.graspable_target_catalog_ids
        if self.balanced_target_catalogs:
            graspable_positions = (
                torch.arange(
                    groups, dtype=torch.int64, device=self.device
                )
                + self.rank * groups
                + int(round_index) * groups
            ) % int(graspable_pool.numel())
        else:
            graspable_positions = torch.randint(
                0,
                int(graspable_pool.numel()),
                (groups,),
                generator=generator,
                device=self.device,
            )
        graspable_target_catalog = graspable_pool.index_select(
            0,
            graspable_positions,
        )
        move_to_task = task_group == INSTRUCTION_TO_ID["move_to_object"]
        pick_up_task = task_group == INSTRUCTION_TO_ID["pick_up"]
        target_catalog = torch.where(
            move_to_task, move_target_catalog, graspable_target_catalog
        )
        target_pool_positions = (
            target_catalog[:, None] == target_pool[None, :]
        ).to(dtype=torch.int64).argmax(dim=1)
        distractor_offsets = torch.arange(
            4, dtype=torch.int64, device=self.device
        )
        catalog_positions = (
            target_pool_positions[:, None] + distractor_offsets[None, :]
        ) % int(target_pool.numel())
        catalogs_group = target_pool.index_select(
            0,
            catalog_positions.reshape(-1),
        ).reshape(groups, 4)
        catalogs_group[:, 0] = target_catalog
        is_bowl = task_group == ACTIVE_INSTRUCTION_TYPES.index("put_into_bowl")
        is_plate = task_group == ACTIVE_INSTRUCTION_TYPES.index("put_into_plate")
        is_container = is_bowl | is_plate
        catalogs_group[:, 1] = torch.where(
            is_bowl,
            torch.full_like(target_catalog, _BOWL_CATALOG_ID),
            torch.where(
                is_plate,
                torch.full_like(target_catalog, _PLATE_CATALOG_ID),
                catalogs_group[:, 1],
            ),
        )
        catalogs_group[:, 2] = torch.where(
            is_bowl,
            torch.full_like(target_catalog, _PLATE_CATALOG_ID),
            torch.where(
                is_plate,
                torch.full_like(target_catalog, _BOWL_CATALOG_ID),
                catalogs_group[:, 2],
            ),
        )

        is_relation_task = (
            task_group
            == ACTIVE_INSTRUCTION_TYPES.index("move_left_of_object")
        ) | (
            task_group
            == ACTIVE_INSTRUCTION_TYPES.index("move_right_of_object")
        )
        is_between_task = task_group == ACTIVE_INSTRUCTION_TYPES.index(
            "move_between_objects"
        )
        required_slots = torch.ones_like(task_group)
        required_slots = torch.where(
            is_relation_task, torch.full_like(task_group, 2), required_slots
        )
        required_slots = torch.where(
            is_container | is_between_task,
            torch.full_like(task_group, 3),
            required_slots,
        )
        scene_low, scene_high = self.scene_object_range
        active_object_counts = torch.randint(
            int(scene_low),
            int(scene_high) + 1,
            (groups,),
            generator=generator,
            dtype=torch.int64,
            device=self.device,
        )
        active_object_counts = torch.maximum(
            active_object_counts, required_slots
        )
        active_slots = (
            torch.arange(4, dtype=torch.int64, device=self.device)[None, :]
            < active_object_counts[:, None]
        )
        catalogs_group = torch.where(
            active_slots,
            catalogs_group,
            torch.full_like(catalogs_group, INACTIVE_CATALOG_ID),
        )

        # Move-to previously always named the slot-0 object, and slot 0 always
        # sat on the same lattice point, so every instruction meant the same
        # place and neither language nor vision could matter. Put the named
        # catalog in a random ACTIVE slot for move-to groups (swap with slot 0
        # so all four catalogs stay distinct). Manipulation tasks keep slot 0 as
        # the handled object because their container logic assumes slots 0/1/2.
        group_rows = torch.arange(
            groups, dtype=torch.int64, device=self.device
        )
        is_move_to_group = task_group == INSTRUCTION_TO_ID["move_to_object"]
        sampled_slot = (
            torch.rand((groups,), generator=generator, device=self.device)
            * active_object_counts.to(dtype=torch.float32)
        ).to(dtype=torch.int64).clamp(0, 3)
        target_slot_group = torch.where(
            is_move_to_group, sampled_slot, torch.zeros_like(sampled_slot)
        )
        slot_zero_catalog = catalogs_group[:, 0].clone()
        selected_catalog = catalogs_group[
            group_rows, target_slot_group
        ].clone()
        catalogs_group[group_rows, target_slot_group] = slot_zero_catalog
        catalogs_group[:, 0] = selected_catalog

        catalogs = catalogs_group.repeat_interleave(group_size, dim=0)

        all_worlds = torch.arange(
            worlds, dtype=torch.int64, device=self.device
        )
        self.backend.reset_worlds(all_worlds)
        self.backend.set_object_catalogs(catalogs)

        # Objects are drawn from a 3x3 candidate grid (9 cells) instead of the
        # old 4 fixed points: each group takes a random 4-cell subset, so the
        # target lands anywhere on the desk instead of one 7.6 cm box.
        # Separation: spacing 0.18 - 2*jitter 0.01 = 0.16 m, above the widest
        # realistic pair (plate 0.091 + bowl 0.057 = 0.148); the four slots
        # always hold distinct catalogs, so two plates can never coincide.
        # Framing: max |coord| = 0.18 + 0.01 + 0.015 = 0.205 m, inside the
        # ~0.23 m half-width the dollied-in overview camera still covers at the
        # near edge of the desk.
        grid_coordinates = torch.tensor(
            (-0.18, 0.0, 0.18), dtype=torch.float32, device=self.device
        )
        cell_choice = torch.rand(
            (groups, 9), generator=generator, device=self.device
        ).argsort(dim=1)[:, :4]
        cell_xy = torch.stack(
            (
                grid_coordinates.index_select(
                    0, (cell_choice % 3).reshape(-1)
                ).reshape(groups, 4),
                grid_coordinates.index_select(
                    0, (cell_choice // 3).reshape(-1)
                ).reshape(groups, 4),
            ),
            dim=-1,
        )
        # Common shift moves the whole scene without changing separation.
        scene_shift = (
            torch.rand(
                (groups, 1, 2), generator=generator, device=self.device
            )
            - 0.5
        ) * 0.03
        jitter = (
            torch.rand(
                (groups, 4, 2), generator=generator, device=self.device
            )
            - 0.5
        ) * 0.02
        object_xy_group = cell_xy + scene_shift + jitter
        rest_height = torch.tensor(
            _REST_HEIGHT, dtype=torch.float32, device=self.device
        ).index_select(
            0, catalogs_group.clamp_min(0).reshape(-1)
        ).reshape(groups, 4)
        rest_height = torch.where(
            active_slots, rest_height, torch.zeros_like(rest_height)
        )
        object_z_group = self.support_surface_z + rest_height
        object_positions_group = torch.cat(
            (object_xy_group, object_z_group[..., None]), dim=-1
        )
        # Inactive slots return to the XML park poses far outside the desk and
        # every camera frustum; their collision geoms are already disabled.
        parked_positions = torch.tensor(
            (
                (-4.0, -4.0, 4.0),
                (-4.0, 4.0, 4.0),
                (4.0, -4.0, 4.0),
                (4.0, 4.0, 4.0),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        object_positions_group = torch.where(
            active_slots[:, :, None],
            object_positions_group,
            parked_positions[None, :, :],
        )

        # target_slot_group was chosen above (random active slot for move-to,
        # slot 0 for every manipulation task).
        reference_slot_group = torch.where(
            is_container
            | (
                (task_group >= ACTIVE_INSTRUCTION_TYPES.index("move_left_of_object"))
                & ~pick_up_task
            ),
            torch.ones_like(target_slot_group),
            torch.full_like(target_slot_group, -1),
        )
        second_reference_slot_group = torch.where(
            task_group == ACTIVE_INSTRUCTION_TYPES.index("move_between_objects"),
            torch.full_like(target_slot_group, 2),
            torch.full_like(target_slot_group, -1),
        )
        rows = torch.arange(groups, dtype=torch.int64, device=self.device)
        reference = object_positions_group[rows, reference_slot_group.clamp_min(0)]
        ee_group = (
            torch.rand(
                (groups, 3), generator=generator, device=self.device
            )
            - 0.5
        )
        ee_group[:, 0:2] *= 0.40
        ee_group[:, 2] = 0.40
        if self.random_workspace_gripper_start:
            random_unit = torch.rand(
                (groups, 3), generator=generator, device=self.device
            )
            workspace_low = torch.tensor(
                (
                    self.workspace_x_bounds[0],
                    self.workspace_y_bounds[0],
                    self.workspace_z_bounds[0],
                ),
                dtype=torch.float32,
                device=self.device,
            )
            workspace_high = torch.tensor(
                (
                    self.workspace_x_bounds[1],
                    self.workspace_y_bounds[1],
                    self.workspace_z_bounds[1],
                ),
                dtype=torch.float32,
                device=self.device,
            )
            ee_group = workspace_low + random_unit * (
                workspace_high - workspace_low
            )
        random_angle = (
            torch.rand((groups,), generator=generator, device=self.device)
            * (2.0 * torch.pi)
        )
        random_direction = torch.stack(
            (torch.cos(random_angle), torch.sin(random_angle)), dim=-1
        )
        target_position = object_positions_group[:, 0].clone()
        initial_target_group = target_position.clone()
        baseline_direction = random_direction.clone()

        move_to_mask = (
            task_group == ACTIVE_INSTRUCTION_TYPES.index("move_to_object")
        )
        move_to_distance = 0.02 + travel_group
        move_to_ee = target_position.clone()
        move_to_ee[:, :2] = (
            target_position[:, :2]
            + random_direction * move_to_distance[:, None]
        )
        move_to_ee[:, 2] = 0.40
        if not self.random_workspace_gripper_start:
            ee_group = torch.where(
                move_to_mask[:, None], move_to_ee, ee_group
            )

        push_left = task_group == ACTIVE_INSTRUCTION_TYPES.index("push_left")
        push_right = task_group == ACTIVE_INSTRUCTION_TYPES.index("push_right")
        is_push = push_left | push_right
        push_sign = torch.where(
            push_left,
            torch.full((groups,), -1.0, device=self.device),
            torch.ones((groups,), device=self.device),
        )
        push_remaining = torch.minimum(
            torch.full((groups,), 0.08, device=self.device),
            target_action_steps.to(dtype=torch.float32) * 0.0075,
        )
        push_progress = (0.08 - push_remaining).clamp_min(0.0)
        push_initial = target_position.clone()
        push_initial[:, 0] -= push_sign * push_progress
        initial_target_group = torch.where(
            is_push[:, None], push_initial, initial_target_group
        )
        push_gap = torch.where(
            shell_group == 0,
            torch.full((groups,), 0.008, device=self.device),
            0.018 + (0.4 * push_remaining).clamp(max=0.04),
        )
        push_ee = target_position.clone()
        push_ee[:, 0] -= push_sign * push_gap
        push_ee[:, 1] += (
            torch.rand((groups,), generator=generator, device=self.device)
            - 0.5
        ) * 0.016
        push_ee[:, 2] = target_position[:, 2] + 0.045
        ee_group = torch.where(is_push[:, None], push_ee, ee_group)

        relation_left = (
            task_group == ACTIVE_INSTRUCTION_TYPES.index("move_left_of_object")
        )
        relation_right = (
            task_group == ACTIVE_INSTRUCTION_TYPES.index("move_right_of_object")
        )
        is_relation = relation_left | relation_right
        relation_sign = torch.where(
            relation_left,
            torch.full((groups,), -1.0, device=self.device),
            torch.ones((groups,), device=self.device),
        )
        between = (
            task_group == ACTIVE_INSTRUCTION_TYPES.index("move_between_objects")
        )
        second_reference = object_positions_group[:, 2]
        placement_task = is_container | is_relation | between
        grasp_learning = (
            (
                task_group
                == ACTIVE_INSTRUCTION_TYPES.index("put_into_plate")
            )
            | is_relation
            | between
        ) & (shell_group >= 5)
        if self.force_caught_container_start:
            grasp_learning &= ~is_container
        held_group = placement_task & ~grasp_learning
        shell_zero = shell_group == 0

        placement_goal = target_position.clone()
        container_goal = reference.clone()
        container_goal[:, 2] = reference[:, 2] + torch.where(
            is_bowl,
            torch.full((groups,), 0.035, device=self.device),
            torch.full((groups,), 0.035, device=self.device),
        )
        placement_goal = torch.where(
            is_container[:, None], container_goal, placement_goal
        )
        relation_goal = reference.clone()
        relation_goal[:, 0] += relation_sign * 0.10
        relation_goal[:, 2] = torch.maximum(
            target_position[:, 2], reference[:, 2] + 0.035
        )
        placement_goal = torch.where(
            is_relation[:, None], relation_goal, placement_goal
        )
        between_goal = 0.5 * (reference + second_reference)
        between_goal[:, 2] = torch.maximum(
            target_position[:, 2], between_goal[:, 2] + 0.035
        )
        placement_goal = torch.where(
            between[:, None], between_goal, placement_goal
        )
        if self.random_workspace_gripper_start:
            # The move-to target is a RANDOM active slot (the named catalog is
            # swapped into target_slot_group), so the curriculum must measure
            # against that slot, not slot 0. With slot 0 the start was pulled
            # within the cap of the wrong object whenever the scene held more
            # than one: at cap 0.03 the mean XY distance to the real target was
            # 0.204 m, i.e. the approach curriculum did nothing at all. Latent
            # while scenes hold one object (the sampled slot is then always 0)
            # and live from the first object-count unlock.
            goal_target = object_positions_group[group_rows, target_slot_group]
            random_goal = torch.where(
                is_container[:, None], reference, goal_target
            )
            # Per-group cap (each group runs a single instruction, and each
            # instruction owns its own approach-curriculum cap).
            max_goal_distance = self._start_cap_table.index_select(0, task_group)
            curriculum_active = torch.isfinite(max_goal_distance) & (
                max_goal_distance > 0.0
            )
            any_curriculum = bool(curriculum_active.any().item())
            # Keep the floor strictly below the cap so the early close starts are
            # not rejected back out to the workspace edge, which would defeat the
            # approach curriculum.
            effective_min_goal_distance = torch.where(
                curriculum_active,
                torch.minimum(
                    torch.full_like(
                        max_goal_distance, self.random_start_min_goal_distance
                    ),
                    0.5 * max_goal_distance,
                ),
                torch.full_like(
                    max_goal_distance, self.random_start_min_goal_distance
                ),
            )
            for _ in range(6):
                too_close = (
                    torch.linalg.vector_norm(
                        ee_group[:, :2] - random_goal[:, :2], dim=-1
                    )
                    < effective_min_goal_distance
                )
                replacement = torch.rand(
                    (groups, 2), generator=generator, device=self.device
                )
                replacement[:, 0] = (
                    self.workspace_x_bounds[0]
                    + replacement[:, 0]
                    * (
                        self.workspace_x_bounds[1]
                        - self.workspace_x_bounds[0]
                    )
                )
                replacement[:, 1] = (
                    self.workspace_y_bounds[0]
                    + replacement[:, 1]
                    * (
                        self.workspace_y_bounds[1]
                        - self.workspace_y_bounds[0]
                    )
                )
                ee_group[:, :2] = torch.where(
                    too_close[:, None], replacement, ee_group[:, :2]
                )
            too_close = (
                torch.linalg.vector_norm(
                    ee_group[:, :2] - random_goal[:, :2], dim=-1
                )
                < effective_min_goal_distance
            )
            if bool(too_close.any().item()):
                corners = torch.tensor(
                    (
                        (
                            self.workspace_x_bounds[0],
                            self.workspace_y_bounds[0],
                        ),
                        (
                            self.workspace_x_bounds[0],
                            self.workspace_y_bounds[1],
                        ),
                        (
                            self.workspace_x_bounds[1],
                            self.workspace_y_bounds[0],
                        ),
                        (
                            self.workspace_x_bounds[1],
                            self.workspace_y_bounds[1],
                        ),
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                corner_distance = torch.linalg.vector_norm(
                    corners[None, :, :] - random_goal[:, None, :2],
                    dim=-1,
                )
                farthest_corner = corners.index_select(
                    0, corner_distance.argmax(dim=1)
                )
                ee_group[:, :2] = torch.where(
                    too_close[:, None], farthest_corner, ee_group[:, :2]
                )
                remaining_too_close = (
                    torch.linalg.vector_norm(
                        ee_group[:, :2] - random_goal[:, :2], dim=-1
                    )
                    < effective_min_goal_distance
                )
                if bool(remaining_too_close.any().item()):
                    raise ValueError(
                        "The configured EE workspace cannot satisfy "
                        "random_workspace_min_goal_xy_distance."
                    )

            if any_curriculum:
                # Pull any start that is beyond its instruction's curriculum cap
                # in toward the goal, to a random distance in the feasible
                # annulus [effective_min, cap], preserving its sampled direction.
                # Clamp back into the workspace box so the start stays reachable.
                offset = ee_group[:, :2] - random_goal[:, :2]
                distance = torch.linalg.vector_norm(
                    offset, dim=-1, keepdim=True
                )
                too_far = curriculum_active & (
                    distance.squeeze(-1) > max_goal_distance
                )
                if bool(too_far.any().item()):
                    direction = offset / distance.clamp_min(1.0e-6)
                    # Uncapped instructions carry an inf cap; zero their span so
                    # no inf/NaN is produced in the discarded branch.
                    span = torch.where(
                        curriculum_active,
                        (
                            max_goal_distance - effective_min_goal_distance
                        ).clamp_min(0.0),
                        torch.zeros_like(effective_min_goal_distance),
                    )
                    sampled = effective_min_goal_distance[:, None] + torch.rand(
                        (groups, 1), generator=generator, device=self.device
                    ) * span[:, None]
                    pulled = random_goal[:, :2] + direction * sampled
                    pulled[:, 0] = pulled[:, 0].clamp(
                        self.workspace_x_bounds[0], self.workspace_x_bounds[1]
                    )
                    pulled[:, 1] = pulled[:, 1].clamp(
                        self.workspace_y_bounds[0], self.workspace_y_bounds[1]
                    )
                    ee_group[:, :2] = torch.where(
                        too_far[:, None], pulled, ee_group[:, :2]
                    )

            if self.curriculum_cap_includes_z and any_curriculum:
                # Confine the start height so the cap bounds the 3-D distance the
                # reward actually measures, not just its XY projection. Without
                # this the Z spread alone can exceed the cap several times over.
                # goal_z is the height each instruction's dense term pulls toward:
                # the move-to hover height, the grasp point (object + the gripper
                # hang), or the receptacle release height (+ the same hang).
                grasp_z = (
                    random_goal[:, 2] + float(self.pick_grasp_height_offset)
                )
                container_z = (
                    reference[:, 2]
                    + torch.where(
                        is_bowl,
                        torch.full_like(
                            reference[:, 2], float(self.bowl_release_height)
                        ),
                        torch.full_like(
                            reference[:, 2], float(self.plate_release_height)
                        ),
                    )
                    + float(self.pick_grasp_height_offset)
                )
                goal_z = torch.where(
                    is_container,
                    container_z,
                    torch.where(
                        move_to_mask,
                        torch.full_like(grasp_z, float(self.move_to_approach_z)),
                        grasp_z,
                    ),
                )
                planar = torch.linalg.vector_norm(
                    ee_group[:, :2] - random_goal[:, :2], dim=-1
                )
                # Whatever of the cap the XY offset has not already consumed.
                z_allowance = (
                    (max_goal_distance.square() - planar.square())
                    .clamp_min(0.0)
                    .sqrt()
                )
                low = torch.maximum(
                    goal_z - z_allowance,
                    torch.full_like(goal_z, self.workspace_z_bounds[0]),
                )
                high = torch.minimum(
                    goal_z + z_allowance,
                    torch.full_like(goal_z, self.workspace_z_bounds[1]),
                )
                # If the allowed band misses the workspace entirely, sit at the
                # reachable height closest to the goal rather than inverting the
                # clamp.
                fallback = goal_z.clamp(
                    self.workspace_z_bounds[0], self.workspace_z_bounds[1]
                )
                confined = torch.where(
                    low <= high,
                    torch.minimum(torch.maximum(ee_group[:, 2], low), high),
                    fallback,
                )
                ee_group[:, 2] = torch.where(
                    curriculum_active, confined, ee_group[:, 2]
                )

        zone_half = torch.full((groups,), 0.015, device=self.device)
        relation_boundary = zone_half / random_direction.abs().amax(
            dim=-1
        ).clamp_min(1.0e-6)
        success_boundary = torch.where(
            is_relation,
            relation_boundary,
            torch.full((groups,), 0.03, device=self.device),
        )
        held_position = placement_goal.clone()
        held_position[:, :2] += random_direction * (
            success_boundary + travel_group
        )[:, None]
        held_container_height = reference[:, 2] + torch.where(
            is_bowl,
            torch.full((groups,), 0.10, device=self.device),
            torch.full((groups,), 0.045, device=self.device),
        )
        held_position[:, 2] = torch.where(
            is_container, held_container_height, held_position[:, 2]
        )
        zero_radius = torch.where(
            is_relation,
            torch.full((groups,), 0.00525, device=self.device),
            torch.full((groups,), 0.0105, device=self.device),
        )
        zero_position = placement_goal.clone()
        zero_position[:, :2] += random_direction * (
            torch.rand((groups,), generator=generator, device=self.device)
            * zero_radius
        )[:, None]
        held_position = torch.where(
            shell_zero[:, None], zero_position, held_position
        )

        grasp_distance = torch.where(
            shell_group == 5,
            0.120
            + torch.rand((groups,), generator=generator, device=self.device)
            * 0.025,
            0.160
            + torch.rand((groups,), generator=generator, device=self.device)
            * 0.040,
        )
        grasp_position = placement_goal.clone()
        grasp_position[:, :2] += random_direction * grasp_distance[:, None]
        grasp_position = torch.where(
            (shell_group >= 7)[:, None], target_position, grasp_position
        )
        placement_position = torch.where(
            grasp_learning[:, None], grasp_position, held_position
        )
        object_positions_group[:, 0] = torch.where(
            placement_task[:, None],
            placement_position,
            object_positions_group[:, 0],
        )

        caught_group = held_group
        if self.random_workspace_gripper_start:
            random_caught_position = ee_group.clone()
            random_caught_position[:, 2] -= 0.08
            object_positions_group[:, 0] = torch.where(
                caught_group[:, None],
                random_caught_position,
                object_positions_group[:, 0],
            )
        caught_object_position = object_positions_group[:, 0].clone()
        caught_ee = caught_object_position.clone()
        caught_ee[:, 2] += 0.08
        ee_group = torch.where(caught_group[:, None], caught_ee, ee_group)
        grasp_pose = caught_object_position.clone()
        grasp_pose[:, 2] += 0.09
        ee_group = torch.where(
            (grasp_learning & (shell_group == 5))[:, None],
            grasp_pose,
            ee_group,
        )
        approach_direction_angle = (
            torch.rand((groups,), generator=generator, device=self.device)
            * (2.0 * torch.pi)
        )
        approach_direction = torch.stack(
            (
                torch.cos(approach_direction_angle),
                torch.sin(approach_direction_angle),
            ),
            dim=-1,
        )
        approach_pose = grasp_pose.clone()
        approach_pose[:, :2] += approach_direction * (
            0.035
            + torch.rand((groups,), generator=generator, device=self.device)
            * 0.025
        )[:, None]
        approach_pose[:, 2] += (
            0.015
            + torch.rand((groups,), generator=generator, device=self.device)
            * 0.020
        )
        ee_group = torch.where(
            (grasp_learning & (shell_group == 6))[:, None],
            approach_pose,
            ee_group,
        )

        motion_baseline = object_positions_group[:, 0].clone()
        motion_baseline[:, :2] += baseline_direction * 0.06
        initial_target_group = torch.where(
            held_group[:, None], motion_baseline, initial_target_group
        )
        initial_target_group = torch.where(
            grasp_learning[:, None],
            object_positions_group[:, 0],
            initial_target_group,
        )
        # Move-to may now name a slot other than zero, so its motion baseline
        # must follow the actual target slot rather than slot 0.
        initial_target_group = torch.where(
            is_move_to_group[:, None],
            object_positions_group[group_rows, target_slot_group],
            initial_target_group,
        )
        yaw_group = (
            torch.rand((groups,), generator=generator, device=self.device)
            * (2.0 * torch.pi)
            - torch.pi
        )

        object_positions = object_positions_group.repeat_interleave(
            group_size, dim=0
        )
        object_yaw = (
            torch.rand((groups, 4), generator=generator, device=self.device)
            * (2.0 * torch.pi)
            - torch.pi
        )
        object_yaw[:, 0] = torch.where(
            caught_group, yaw_group, object_yaw[:, 0]
        )
        object_quat_group = torch.zeros(
            (groups, 4, 4), dtype=torch.float32, device=self.device
        )
        object_quat_group[..., 0] = torch.cos(0.5 * object_yaw)
        object_quat_group[..., 3] = torch.sin(0.5 * object_yaw)
        object_quaternions = object_quat_group.repeat_interleave(
            group_size, dim=0
        )
        self.backend.set_free_body_poses(
            self.backend.object_body_ids,
            object_positions,
            object_quaternions,
        )
        self.backend.set_end_effector_poses(
            ee_group.repeat_interleave(group_size, dim=0),
            yaw_group.repeat_interleave(group_size, dim=0),
        )

        fitted = torch.tensor(
            _FITTED_GRIPPER, dtype=torch.float32, device=self.device
        ).index_select(0, target_catalog)
        opening_group = torch.where(
            is_push,
            torch.zeros_like(fitted),
            torch.where(
                caught_group,
                (fitted - (0.001 / 0.03)).clamp(0.0, 1.0),
                torch.ones_like(fitted),
            ),
        )
        self.backend.set_gripper_openings(
            opening_group.repeat_interleave(group_size, dim=0)
        )
        self.backend.set_free_body_poses(
            self.backend.object_body_ids,
            object_positions,
            object_quaternions,
        )

        texture_group = torch.randint(
            0, 7, (groups,), generator=generator, device=self.device
        )
        background_group = 0.65 + torch.rand(
            (groups, 4), generator=generator, device=self.device
        ) * 0.30
        background_group[:, 3] = 1.0
        shade_group = 0.55 + torch.rand(
            (groups,), generator=generator, device=self.device
        ) * 0.45
        self.backend.set_visual_variants(
            texture_group.repeat_interleave(group_size),
            background_group.repeat_interleave(group_size, dim=0),
            shade_group.repeat_interleave(group_size),
        )
        base_worlds = torch.arange(
            0, worlds, group_size, dtype=torch.int64, device=self.device
        )
        self.backend.broadcast_group_state(base_worlds)

        task_ids = task_group.repeat_interleave(group_size)
        target_slots = target_slot_group.repeat_interleave(group_size)
        reference_slots = reference_slot_group.repeat_interleave(group_size)
        second_reference_slots = second_reference_slot_group.repeat_interleave(
            group_size
        )
        caught = caught_group.repeat_interleave(group_size)
        initial_target = initial_target_group.repeat_interleave(group_size, dim=0)
        release_group = torch.maximum(
            torch.full_like(fitted, 0.55), (fitted + 0.04).clamp(max=1.0)
        )
        task_state = BatchedTaskState(
            instruction_ids=task_ids,
            target_slots=target_slots,
            reference_slots=reference_slots,
            second_reference_slots=second_reference_slots,
            initial_target_positions=initial_target,
            ever_grasped=caught.clone(),
            grasped=caught.clone(),
            step_count=torch.zeros(
                (worlds,), dtype=torch.int64, device=self.device
            ),
            release_threshold=release_group.repeat_interleave(group_size),
            support_surface_z=torch.full(
                (worlds,),
                self.support_surface_z,
                dtype=torch.float32,
                device=self.device,
            ),
            target_rest_height=(
                rest_height.repeat_interleave(group_size, dim=0)[
                    torch.arange(
                        worlds, dtype=torch.int64, device=self.device
                    ),
                    target_slots,
                ]
            ),
        )
        target_catalog_names = [
            ACTIVE_CDPR_CATALOGS[index]
            for index in target_catalog.detach().cpu().tolist()
        ]
        instruction_group: list[str] = []
        for index, task_id in enumerate(task_group.detach().cpu().tolist()):
            name = ACTIVE_INSTRUCTION_TYPES[task_id]
            target_name = OBJECT_VARIANTS[target_catalog_names[index]].label
            if name == "move_to_object":
                text = f"move to {target_name}"
            elif name == "push_left":
                text = f"push {target_name} left"
            elif name == "push_right":
                text = f"push {target_name} right"
            elif name == "put_into_bowl":
                text = f"put {target_name} into bowl"
            elif name == "put_into_plate":
                text = f"put {target_name} on the plate"
            elif name == "pick_up":
                text = f"pick up {target_name}"
            elif name == "move_left_of_object":
                text = f"move {target_name} left of the reference object"
            elif name == "move_right_of_object":
                text = f"move {target_name} right of the reference object"
            else:
                text = f"move {target_name} between the two reference objects"
            instruction_group.append(text)
        instructions = tuple(
            text
            for text in instruction_group
            for _ in range(group_size)
        )
        reset_low_dim = self.backend.low_dim_observations()
        world_rows = torch.arange(
            worlds, dtype=torch.int64, device=self.device
        )
        target_position_reset = reset_low_dim.object_positions[
            world_rows, target_slots
        ]
        target_quaternion_reset = reset_low_dim.object_quaternions[
            world_rows, target_slots
        ]
        previous_relative_position = (
            target_position_reset - reset_low_dim.ee_position
        )
        previous_relative_quaternion = _relative_quaternion(
            reset_low_dim.ee_quaternion, target_quaternion_reset
        )
        target_rest_height = (
            rest_height.repeat_interleave(group_size, dim=0)[
                world_rows, target_slots
            ]
        )
        return BatchedReset(
            instructions=instructions,
            task_state=task_state,
            group_instruction_ids=task_group,
            group_shell_ids=shell_group,
            horizons=horizon_group.repeat_interleave(group_size),
            physical_grasp=caught.clone(),
            grasp_eligible=(placement_task | pick_up_task).repeat_interleave(
                group_size
            ),
            bilateral_contact_steps=torch.zeros(
                (worlds,), dtype=torch.int64, device=self.device
            ),
            previous_relative_position=previous_relative_position,
            previous_relative_quaternion=previous_relative_quaternion,
            target_rest_height=target_rest_height,
            group_ids=group_ids.repeat_interleave(group_size),
            group_target_catalog_ids=target_catalog,
        )


@dataclass(frozen=True)
class CollectorRound:
    records: dict[str, Any]
    loss_mask: Any
    candidate_rewards: Any
    candidate_success: Any
    group_instruction_ids: Any
    group_shell_ids: Any
    metrics: dict[str, float]
    # Capped decision-0 subsample of SmolVLA inputs for the LoRA grad-through-VLA
    # update (None unless the collector was asked to store it).
    vla_records: dict[str, Any] | None = None


@dataclass(frozen=True)
class ValidationRound:
    candidate_rewards: Any
    candidate_success: Any
    final_xy_distance: Any
    group_target_catalog_ids: Any
    group_shell_ids: Any
    metrics: dict[str, float]
    # Which instruction each group ran. A run that mixes instructions of very
    # different difficulty (pick_up vs. the pre-grasped placement tasks) reports
    # a single blended success_rate that the easy tasks dominate, so a task stuck
    # at zero stays invisible. None keeps the aggregate-only behaviour.
    group_instruction_ids: Any | None = None


class RankLocalMJWarpGRPOCollector:
    def __init__(
        self,
        *,
        backend: Any,
        smolvla_runtime: Any,
        trainer: Any,
        resetter: BatchedReverseFrontierResetter,
        layout: RankLocalGroupLayout,
        actions_per_policy_decision: int = 4,
        smolvla_microbatch_size: int = 0,
        normalize_advantage: bool = True,
        advantage_clip_abs: float = 6.0,
        dynamic_min_pass_rate: float = 0.10,
        dynamic_max_pass_rate: float = 0.90,
        dynamic_sampling: bool = True,
        group_selection: str = "uniform",
        move_to_distance_reward: BatchedMoveToDistanceReward | None = None,
        catch_release_dense_reward: (
            BatchedCatchReleaseDenseReward | None
        ) = None,
        include_relative_target: bool = False,
        vision_feature_dim: int = 0,
        store_vla_records: bool = False,
        vla_update_max_records: int = 128,
        profile: bool = False,
    ) -> None:
        layout.validate()
        self.backend = backend
        self.runtime = smolvla_runtime
        self.trainer = trainer
        self.resetter = resetter
        self.layout = layout
        self.include_relative_target = bool(include_relative_target)
        # >0 appends a frozen fixed-projection SmolVLA vision feature of this
        # width to the residual state so the residual can localize the target.
        self.vision_feature_dim = max(0, int(vision_feature_dim))
        # Reset-on-drift is configured on the resetter (which parses task
        # metadata); the rollout loop lives here on the collector, so mirror the
        # settings across. getattr defaults keep it inert if the resetter type
        # does not define them.
        self.reset_on_drift = bool(getattr(resetter, "reset_on_drift", False))
        self.reset_on_drift_patience = max(
            1, int(getattr(resetter, "reset_on_drift_patience", 4))
        )
        self.reset_on_drift_penalty = float(
            getattr(resetter, "reset_on_drift_penalty", 0.0)
        )
        self.reset_on_drift_min_increase = max(
            0.0, float(getattr(resetter, "reset_on_drift_min_increase", 0.001))
        )
        self.store_vla_records = bool(store_vla_records)
        self.vla_update_max_records = max(0, int(vla_update_max_records))
        self.actions_per_policy_decision = max(
            1,
            min(
                int(actions_per_policy_decision),
                int(trainer.chunk_size),
            ),
        )
        self.smolvla_microbatch_size = max(0, int(smolvla_microbatch_size))
        self.normalize_advantage = bool(normalize_advantage)
        self.advantage_clip_abs = float(advantage_clip_abs)
        self.dynamic_min_pass_rate = float(dynamic_min_pass_rate)
        self.dynamic_max_pass_rate = float(dynamic_max_pass_rate)
        self.dynamic_sampling = bool(dynamic_sampling)
        self.move_to_distance_reward = move_to_distance_reward
        self.catch_release_dense_reward = catch_release_dense_reward
        self.group_selection = str(group_selection).lower()
        if self.group_selection not in {"uniform", "best", "softmax"}:
            raise ValueError(
                f"Unsupported rank-local GRPO selection: {group_selection!r}."
            )
        self.profile = bool(profile)
        self.torch = backend.torch
        self.device = backend.device
        self._sample_generator = self.torch.Generator(device=self.device)
        self._world_rows = self.torch.arange(
            self.layout.worlds_per_rank,
            dtype=self.torch.int64,
            device=self.device,
        )

    def _task_thresholds(self) -> BatchedTaskThresholds:
        move_to = self.move_to_distance_reward
        catch_release = self.catch_release_dense_reward
        return BatchedTaskThresholds(
            move_to_xy_low=(
                0.0 if move_to is None else float(move_to.xy_window_low)
            ),
            move_to_xy=(
                0.02 if move_to is None else float(move_to.xy_window_high)
            ),
            container_xy=(
                0.03
                if catch_release is None
                else max(
                    float(catch_release.plate_radius),
                    float(catch_release.bowl_radius),
                )
            ),
            container_z=(
                0.12
                if catch_release is None
                else float(catch_release.container_z_tolerance)
            ),
            minimum_target_motion=(
                0.04 if catch_release is None else 0.0
            ),
        )

    def _sync_for_profile(self) -> None:
        if self.profile:
            self.torch.cuda.synchronize(self.device)

    def _goal_slots(self, reset: BatchedReset) -> Any:
        """Slot the reward shapes toward, per world.

        Placement tasks are rewarded on the gripper->receptacle distance while
        the target object is already held, so their goal is the reference slot.
        Every other task drives toward the target object itself.
        """

        torch = self.torch
        instruction_ids = reset.task_state.instruction_ids
        reference_slots = reset.task_state.reference_slots
        is_container = (
            instruction_ids == INSTRUCTION_TO_ID["put_into_plate"]
        ) | (instruction_ids == INSTRUCTION_TO_ID["put_into_bowl"])
        return torch.where(
            is_container & (reference_slots >= 0),
            reference_slots,
            reset.task_state.target_slots,
        )

    def _update_physical_grasp(
        self,
        reset: BatchedReset,
        low_dim: Any,
        active_mask: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Evaluate a free-body grasp from persistent bilateral contact physics."""

        rows = self._world_rows
        target_slots = reset.task_state.target_slots
        target_position = low_dim.object_positions[rows, target_slots]
        target_quaternion = low_dim.object_quaternions[rows, target_slots]
        relative_position = target_position - low_dim.ee_position
        relative_quaternion = _relative_quaternion(
            low_dim.ee_quaternion, target_quaternion
        )
        relative_position_slip = self.torch.linalg.vector_norm(
            relative_position - reset.previous_relative_position, dim=-1
        )
        quaternion_dot = (
            relative_quaternion * reset.previous_relative_quaternion
        ).sum(dim=-1).abs().clamp(max=1.0)
        relative_orientation_slip = 2.0 * self.torch.acos(quaternion_dot)
        stable_relative_pose = (
            relative_position_slip
            <= float(_GRASP_MAX_RELATIVE_POSITION_SLIP_M)
        ) & (
            relative_orientation_slip
            <= float(_GRASP_MAX_RELATIVE_ORIENTATION_SLIP_RAD)
        )
        contacts = self.backend.finger_object_contact_metrics(target_slots)
        force_ok = (
            contacts.left_normal_force >= float(_GRASP_MIN_PAD_FORCE_N)
        ) & (
            contacts.right_normal_force >= float(_GRASP_MIN_PAD_FORCE_N)
        )
        persistent_candidate = (
            reset.grasp_eligible
            & active_mask
            & contacts.bilateral_contact
            & force_ok
            & stable_relative_pose
        )
        reset.bilateral_contact_steps.copy_(
            self.torch.where(
                persistent_candidate,
                reset.bilateral_contact_steps + 1,
                self.torch.zeros_like(reset.bilateral_contact_steps),
            )
        )
        lifted = target_position[:, 2] >= (
            reset.task_state.support_surface_z
            + reset.target_rest_height
            + float(_GRASP_MIN_LIFT_M)
        )
        release_open = (
            low_dim.gripper_opening >= reset.task_state.release_threshold
        )
        physical_grasp = (
            persistent_candidate
            & (
                reset.bilateral_contact_steps
                >= int(_GRASP_PERSISTENCE_STEPS)
            )
            & ~release_open
        )
        reset.physical_grasp.copy_(physical_grasp)
        reset.previous_relative_position.copy_(relative_position)
        reset.previous_relative_quaternion.copy_(relative_quaternion)
        diagnostics = {
            "bilateral_contact": contacts.bilateral_contact,
            "left_pad_force_n": contacts.left_normal_force,
            "right_pad_force_n": contacts.right_normal_force,
            "relative_position_slip_m": relative_position_slip,
            "relative_orientation_slip_rad": relative_orientation_slip,
            "stable_relative_pose": stable_relative_pose,
            "physically_lifted": lifted,
            "physical_grasp": physical_grasp,
            "physical_release": reset.task_state.ever_grasped
            & ~physical_grasp
            & release_open,
        }
        return low_dim, physical_grasp, diagnostics

    def collect_round(
        self,
        *,
        update_index: int,
        round_index: int,
    ) -> CollectorRound:
        torch = self.torch
        reset_start = time.perf_counter()
        reset = self.resetter.reset(
            update_index=update_index, round_index=round_index
        )
        self._sample_generator.manual_seed(
            self.resetter.base_seed
            + self.resetter.rank * 1_000_003
            + int(update_index) * 10_000_019
            + int(round_index) * 100_003
            + 71
        )
        self._sync_for_profile()
        reset_time = time.perf_counter() - reset_start

        worlds = int(self.layout.worlds_per_rank)
        group_size = int(self.layout.group_size)
        active = torch.ones(
            (worlds,), dtype=torch.bool, device=self.device
        )
        candidate_success = torch.zeros_like(active)
        candidate_rewards = torch.zeros(
            (worlds,), dtype=torch.float32, device=self.device
        )
        sampled_actions = torch.zeros((), dtype=torch.int64, device=self.device)
        actions_per_world = torch.zeros(
            (worlds,), dtype=torch.int64, device=self.device
        )
        record_lists: dict[str, list[Any]] = {
            "state": [],
            "prior": [],
            "action": [],
            "action_index": [],
            "old_log_prob": [],
            "world_index": [],
        }
        valid_masks: list[Any] = []
        timings = {
            "render_time_s": 0.0,
            "smolvla_time_s": 0.0,
            "policy_time_s": 0.0,
            "physics_time_s": 0.0,
            "reward_time_s": 0.0,
        }
        grasp_diagnostic_totals = {
            name: torch.zeros(
                (), dtype=torch.float32, device=self.device
            )
            for name in (
                "observations",
                "bilateral_contact",
                "left_pad_force_n",
                "right_pad_force_n",
                "relative_position_slip_m",
                "relative_orientation_slip_rad",
                "stable_relative_pose",
                "physically_lifted",
                "physical_grasp",
                "physical_release",
            )
        }
        max_decisions = int(reset.horizons.max().detach().cpu().item())
        goal_slots = self._goal_slots(reset)
        drift_counter = torch.zeros(
            (worlds,), dtype=torch.int64, device=self.device
        )
        prev_goal_distance = torch.full(
            (worlds,), float("inf"), dtype=torch.float32, device=self.device
        )
        drift_terminated_total = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        prior_target_cosine_first = torch.zeros(
            (worlds,), dtype=torch.float32, device=self.device
        )
        policy_target_cosine_first = torch.zeros_like(
            prior_target_cosine_first
        )
        vla_capture: dict[str, Any] | None = None

        for decision in range(max_decisions):
            self._sync_for_profile()
            started = time.perf_counter()
            cameras = self.backend.render_policy_cameras()
            self._sync_for_profile()
            timings["render_time_s"] += time.perf_counter() - started

            low_dim = self.backend.low_dim_observations()
            # Proprioception-only width (the vision feature is appended after the
            # SmolVLA forward, since it comes FROM that forward). SmolVLA sees the
            # narrow state; the residual sees narrow + vision.
            proprio_state_dim = int(self.trainer.state_dim) - self.vision_feature_dim
            state_tensor = build_smolvla_state_tensor(
                ee_position=low_dim.ee_position,
                ee_yaw=low_dim.ee_yaw,
                gripper_opening=low_dim.gripper_opening,
                object_positions=low_dim.object_positions,
                target_slots=reset.task_state.target_slots,
                state_dim=proprio_state_dim,
                include_relative_target=self.include_relative_target,
                goal_slots=goal_slots,
            )
            self._sync_for_profile()
            started = time.perf_counter()
            if self.vision_feature_dim > 0:
                prior, vision_feature = (
                    self.runtime.sample_cdpr_chunks_and_vision_from_tensors(
                        primary_images=cameras.overview,
                        wrist_images=cameras.wrist,
                        states=state_tensor,
                        instructions=reset.instructions,
                        vision_dim=self.vision_feature_dim,
                        microbatch_size=self.smolvla_microbatch_size,
                    )
                )
                state_tensor = torch.cat(
                    [state_tensor, vision_feature.to(dtype=state_tensor.dtype)],
                    dim=-1,
                )
            else:
                prior = self.runtime.sample_cdpr_chunks_from_tensors(
                    primary_images=cameras.overview,
                    wrist_images=cameras.wrist,
                    states=state_tensor,
                    instructions=reset.instructions,
                    microbatch_size=self.smolvla_microbatch_size,
                )
            self._sync_for_profile()
            timings["smolvla_time_s"] += time.perf_counter() - started

            started = time.perf_counter()
            actions, log_probs, _means = (
                self.trainer.sample_action_chunks_tensor(
                    states=state_tensor,
                    priors=prior,
                    action_count=self.actions_per_policy_decision,
                    generator=self._sample_generator,
                )
            )

            if decision == 0:
                # Real-scene goal-direction probe: does the first-step XY action
                # point toward the true target? cosine ~ +1 goal-directed,
                # ~0 task-blind, <0 anti-directed. Measured for the frozen VLA
                # prior and for the composed prior+residual policy so we can see
                # whether the residual is learning direction the prior lacks.
                target0 = gather_world_slots(
                    low_dim.object_positions, reset.task_state.target_slots
                )
                rel_xy0 = (target0 - low_dim.ee_position)[:, :2]
                prior_target_cosine_first = _cosine_2d(
                    prior[:, 0, :2], rel_xy0
                )
                policy_target_cosine_first = _cosine_2d(
                    actions[:, 0, :2], rel_xy0
                )
                if self.store_vla_records and self.vla_update_max_records > 0:
                    # Capped subsample of whole groups for the LoRA update:
                    # store the SmolVLA inputs + taken first action + behaviour
                    # log-prob + detached prior (KL reference). Images as fp16
                    # to bound memory; advantages are filled in after the round.
                    cap = min(self.vla_update_max_records, worlds)
                    cap -= cap % group_size
                    cap = max(group_size, cap)
                    idx = torch.arange(cap, device=self.device)
                    vla_capture = {
                        "world_index": idx,
                        "overview": cameras.overview[idx].to(torch.float16),
                        "wrist": cameras.wrist[idx].to(torch.float16),
                        "state": state_tensor[idx].detach(),
                        "instruction": [
                            reset.instructions[int(i)] for i in idx.tolist()
                        ],
                        "action": actions[idx, 0].detach(),
                        "action_index": torch.zeros(
                            (cap,), dtype=torch.int64, device=self.device
                        ),
                        "old_log_prob": log_probs[idx, 0].detach(),
                        "prior_ref": prior[idx].detach(),
                    }
            self._sync_for_profile()
            timings["policy_time_s"] += time.perf_counter() - started

            for action_index in range(self.actions_per_policy_decision):
                step_active = active & (decision < reset.horizons)
                record_lists["state"].append(state_tensor.detach())
                record_lists["prior"].append(prior.detach())
                record_lists["action"].append(actions[:, action_index].detach())
                record_lists["action_index"].append(
                    torch.full(
                        (worlds,),
                        action_index,
                        dtype=torch.int64,
                        device=self.device,
                    )
                )
                record_lists["old_log_prob"].append(
                    log_probs[:, action_index].detach()
                )
                record_lists["world_index"].append(
                    torch.arange(
                        worlds, dtype=torch.int64, device=self.device
                    )
                )
                valid_masks.append(step_active)
                sampled_actions += step_active.sum()
                actions_per_world += step_active.to(dtype=torch.int64)

                self._sync_for_profile()
                started = time.perf_counter()
                low_dim = self.backend.step(
                    actions[:, action_index], step_active
                )
                low_dim, caught_target, grasp_diagnostics = (
                    self._update_physical_grasp(
                        reset,
                        low_dim,
                        step_active,
                    )
                )
                diagnostic_bool = step_active & reset.grasp_eligible
                diagnostic_mask = diagnostic_bool.to(dtype=torch.float32)
                grasp_diagnostic_totals["observations"] += (
                    diagnostic_mask.sum()
                )
                for name, values in grasp_diagnostics.items():
                    # torch.where, not `values * mask`: a spurious inf contact
                    # force in a masked-out world would otherwise give inf*0=NaN
                    # and poison an observational mean (harmless in move-to, but
                    # it corrupts the real grasp diagnostics in stage 2). Zero
                    # the excluded worlds before they can contribute.
                    grasp_diagnostic_totals[name] += torch.where(
                        diagnostic_bool,
                        values.to(dtype=torch.float32),
                        torch.zeros_like(diagnostic_mask),
                    ).sum()
                self._sync_for_profile()
                timings["physics_time_s"] += time.perf_counter() - started

                started = time.perf_counter()
                result = evaluate_active_sparse_tasks(
                    state=reset.task_state,
                    ee_position=low_dim.ee_position,
                    object_positions=low_dim.object_positions,
                    gripper_opening=low_dim.gripper_opening,
                    caught_target=caught_target,
                    active_mask=step_active,
                    max_steps=10_000,
                    thresholds=self._task_thresholds(),
                    move_to_distance_reward=self.move_to_distance_reward,
                    catch_release_dense_reward=(
                        self.catch_release_dense_reward
                    ),
                    bilateral_contact=grasp_diagnostics["bilateral_contact"],
                )
                candidate_rewards.copy_(
                    torch.where(
                        step_active,
                        result.rewards.to(dtype=torch.float32),
                        candidate_rewards,
                    )
                )
                candidate_success.logical_or_(result.success)
                active.logical_and_(~result.terminated)
                if self.reset_on_drift:
                    goal_pos = low_dim.object_positions[
                        self._world_rows, goal_slots
                    ]
                    goal_distance = torch.linalg.vector_norm(
                        (goal_pos - low_dim.ee_position)[:, :2], dim=-1
                    )
                    drifted = goal_distance > (
                        prev_goal_distance + self.reset_on_drift_min_increase
                    )
                    # Count consecutive drift steps; reset the counter on any
                    # step that made progress. Only stepped worlds update.
                    drift_counter = torch.where(
                        step_active & drifted,
                        drift_counter + 1,
                        torch.where(
                            step_active,
                            torch.zeros_like(drift_counter),
                            drift_counter,
                        ),
                    )
                    prev_goal_distance = torch.where(
                        step_active, goal_distance, prev_goal_distance
                    )
                    drift_terminate = step_active & (
                        drift_counter >= self.reset_on_drift_patience
                    )
                    # Freeze the terminated trajectory's return to the penalty:
                    # the return is the last active step's reward, so without
                    # this a truncation would reward drifting (it stops the world
                    # at its closer pre-drift point).
                    candidate_rewards.copy_(
                        torch.where(
                            drift_terminate,
                            torch.full_like(
                                candidate_rewards,
                                float(self.reset_on_drift_penalty),
                            ),
                            candidate_rewards,
                        )
                    )
                    drift_terminated_total += drift_terminate.sum().to(
                        dtype=torch.float32
                    )
                    active.logical_and_(~drift_terminate)
                self._sync_for_profile()
                timings["reward_time_s"] += time.perf_counter() - started
            active.logical_and_((decision + 1) < reset.horizons)

        # Worlds the backend had to reset this round because their physics went
        # non-finite. The backend already contained them per step (a round-end
        # ee_position check would now always be zero), so this is the true
        # divergence rate: non-zero but small is tolerable, a climbing trend is
        # a real physics problem.
        non_finite_worlds = float(self.backend.pop_nonfinite_world_events())

        success_by_group = candidate_success.reshape(
            self.layout.groups_per_rank, group_size
        )
        rewards_by_group = candidate_rewards.reshape(
            self.layout.groups_per_rank, group_size
        )
        action_counts_by_group = actions_per_world.reshape(
            self.layout.groups_per_rank, group_size
        )
        if self.group_selection == "best":
            selected_candidate = rewards_by_group.argmax(dim=1)
        elif self.group_selection == "softmax":
            selected_candidate = torch.multinomial(
                torch.softmax(rewards_by_group, dim=1),
                num_samples=1,
                replacement=True,
                generator=self._sample_generator,
            ).squeeze(1)
        else:
            selected_candidate = torch.randint(
                0,
                group_size,
                (self.layout.groups_per_rank,),
                generator=self._sample_generator,
                device=self.device,
            )
        selected_actions = action_counts_by_group.gather(
            1, selected_candidate[:, None]
        ).sum()
        advantages_by_group = torch_group_advantages(
            rewards_by_group,
            normalize=self.normalize_advantage,
            clip_abs=self.advantage_clip_abs,
        )
        world_advantage = advantages_by_group.reshape(-1)
        pass_rate = success_by_group.to(dtype=torch.float32).mean(dim=1)
        if not self.dynamic_sampling:
            informative_group = torch.ones_like(pass_rate, dtype=torch.bool)
        elif (
            self.move_to_distance_reward is not None
            or self.catch_release_dense_reward is not None
        ):
            reward_span = rewards_by_group.amax(dim=1) - rewards_by_group.amin(
                dim=1
            )
            informative_group = reward_span > 1.0e-6
        else:
            informative_group = (
                (pass_rate > self.dynamic_min_pass_rate)
                & (pass_rate < self.dynamic_max_pass_rate)
            )
        informative_world = informative_group.repeat_interleave(group_size)

        records = {
            key: torch.cat(values, dim=0)
            for key, values in record_lists.items()
        }
        record_valid = torch.cat(valid_masks, dim=0)
        record_world = records.pop("world_index")
        records["advantage"] = world_advantage.index_select(0, record_world)
        loss_mask = record_valid & informative_world.index_select(
            0, record_world
        )
        vla_records = None
        if vla_capture is not None:
            world_idx = vla_capture.pop("world_index")
            vla_capture["advantage"] = world_advantage.index_select(
                0, world_idx
            )
            vla_records = vla_capture
        self._sync_for_profile()
        total_time = reset_time + sum(timings.values())
        grasp_observations = float(
            grasp_diagnostic_totals["observations"].item()
        )
        grasp_denominator = max(1.0, grasp_observations)
        metrics = {
            **timings,
            "reset_time_s": float(reset_time),
            "rollout_time_s": float(total_time),
            "sampled_environment_actions": float(sampled_actions.item()),
            "selected_environment_actions": float(selected_actions.item()),
            "trajectory_work_amplification": float(
                sampled_actions.item() / max(1, selected_actions.item())
            ),
            "sampled_actions_per_second": float(
                sampled_actions.item() / max(total_time, 1.0e-9)
            ),
            "selected_actions_per_second": float(
                selected_actions.item() / max(total_time, 1.0e-9)
            ),
            "smolvla_batch_size": float(worlds),
            "smolvla_inference_microbatch_size": float(
                worlds
                if self.smolvla_microbatch_size <= 0
                else min(worlds, self.smolvla_microbatch_size)
            ),
            "timers_cuda_synchronized": float(self.profile),
            "complete_groups_per_rank": float(self.layout.groups_per_rank),
            "informative_groups": float(informative_group.sum().item()),
            "group_pass_rate_mean": float(pass_rate.mean().item()),
            "non_finite_ee_worlds": non_finite_worlds,
            "drift_terminations": float(drift_terminated_total.item()),
            "curriculum/horizon_decisions": float(max_decisions),
            "candidate_reward_mean": float(rewards_by_group.mean().item()),
            "candidate_reward_std": float(
                rewards_by_group.std(unbiased=False).item()
            ),
            # Real-scene goal-direction probe (first decision). cosine ~ +1
            # means the action points at the target, ~0 task-blind, <0 away.
            # prior_* is the frozen VLA; policy_* is prior+residual.
            "prior_target_cosine_mean": float(
                prior_target_cosine_first.mean().item()
            ),
            "prior_target_alignment_rate": float(
                (prior_target_cosine_first > 0.0).float().mean().item()
            ),
            "policy_target_cosine_mean": float(
                policy_target_cosine_first.mean().item()
            ),
            "policy_target_alignment_rate": float(
                (policy_target_cosine_first > 0.0).float().mean().item()
            ),
            "dense_move_to_distance_reward": float(
                self.move_to_distance_reward is not None
            ),
            "dense_catch_release_reward": float(
                self.catch_release_dense_reward is not None
            ),
            "records_total": float(loss_mask.numel()),
            "records_informative": float(loss_mask.sum().item()),
            "grasp_diagnostic_observations": grasp_observations,
            "bilateral_pad_contact_rate": float(
                grasp_diagnostic_totals["bilateral_contact"].item()
                / grasp_denominator
            ),
            "left_pad_normal_force_mean_n": float(
                grasp_diagnostic_totals["left_pad_force_n"].item()
                / grasp_denominator
            ),
            "right_pad_normal_force_mean_n": float(
                grasp_diagnostic_totals["right_pad_force_n"].item()
                / grasp_denominator
            ),
            "relative_position_slip_mean_m": float(
                grasp_diagnostic_totals[
                    "relative_position_slip_m"
                ].item()
                / grasp_denominator
            ),
            "relative_orientation_slip_mean_rad": float(
                grasp_diagnostic_totals[
                    "relative_orientation_slip_rad"
                ].item()
                / grasp_denominator
            ),
            "stable_relative_pose_rate": float(
                grasp_diagnostic_totals["stable_relative_pose"].item()
                / grasp_denominator
            ),
            "physical_lift_rate": float(
                grasp_diagnostic_totals["physically_lifted"].item()
                / grasp_denominator
            ),
            "physical_grasp_rate": float(
                grasp_diagnostic_totals["physical_grasp"].item()
                / grasp_denominator
            ),
            "physical_release_rate": float(
                grasp_diagnostic_totals["physical_release"].item()
                / grasp_denominator
            ),
        }
        return CollectorRound(
            records=records,
            loss_mask=loss_mask.to(dtype=torch.float32),
            candidate_rewards=rewards_by_group,
            candidate_success=success_by_group,
            group_instruction_ids=reset.group_instruction_ids,
            group_shell_ids=reset.group_shell_ids,
            metrics=metrics,
            vla_records=vla_records,
        )

    def validate_round(
        self,
        *,
        round_index: int,
    ) -> ValidationRound:
        """Run one fixed-seed, inference-only validation batch on this GPU."""

        torch = self.torch
        if (
            self.move_to_distance_reward is None
            and self.catch_release_dense_reward is None
        ):
            raise RuntimeError(
                "MJWarp held-out dense validation requires a tensorized reward."
            )

        validation_started = time.perf_counter()
        reset = self.resetter.reset(update_index=0, round_index=round_index)
        if reset.group_target_catalog_ids is None:
            raise RuntimeError("Validation reset did not expose target catalogs.")

        worlds = int(self.layout.worlds_per_rank)
        group_size = int(self.layout.group_size)
        active = torch.ones(
            (worlds,), dtype=torch.bool, device=self.device
        )
        candidate_success = torch.zeros_like(active)
        candidate_rewards = torch.zeros(
            (worlds,), dtype=torch.float32, device=self.device
        )
        final_xy_distance = torch.full(
            (worlds,),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        sampled_actions = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )
        timings = {
            "validation/render_time_s": 0.0,
            "validation/smolvla_time_s": 0.0,
            "validation/policy_time_s": 0.0,
            "validation/physics_time_s": 0.0,
            "validation/reward_time_s": 0.0,
        }
        max_decisions = int(reset.horizons.max().detach().cpu().item())
        goal_slots = self._goal_slots(reset)

        with torch.inference_mode():
            for decision in range(max_decisions):
                self._sync_for_profile()
                started = time.perf_counter()
                cameras = self.backend.render_policy_cameras()
                self._sync_for_profile()
                timings["validation/render_time_s"] += (
                    time.perf_counter() - started
                )

                low_dim = self.backend.low_dim_observations()
                proprio_state_dim = (
                    int(self.trainer.state_dim) - self.vision_feature_dim
                )
                state_tensor = build_smolvla_state_tensor(
                    ee_position=low_dim.ee_position,
                    ee_yaw=low_dim.ee_yaw,
                    gripper_opening=low_dim.gripper_opening,
                    object_positions=low_dim.object_positions,
                    target_slots=reset.task_state.target_slots,
                    state_dim=proprio_state_dim,
                    include_relative_target=self.include_relative_target,
                    goal_slots=goal_slots,
                )
                self._sync_for_profile()
                started = time.perf_counter()
                if self.vision_feature_dim > 0:
                    prior, vision_feature = (
                        self.runtime.sample_cdpr_chunks_and_vision_from_tensors(
                            primary_images=cameras.overview,
                            wrist_images=cameras.wrist,
                            states=state_tensor,
                            instructions=reset.instructions,
                            vision_dim=self.vision_feature_dim,
                            microbatch_size=self.smolvla_microbatch_size,
                        )
                    )
                    state_tensor = torch.cat(
                        [state_tensor, vision_feature.to(dtype=state_tensor.dtype)],
                        dim=-1,
                    )
                else:
                    prior = self.runtime.sample_cdpr_chunks_from_tensors(
                        primary_images=cameras.overview,
                        wrist_images=cameras.wrist,
                        states=state_tensor,
                        instructions=reset.instructions,
                        microbatch_size=self.smolvla_microbatch_size,
                    )
                self._sync_for_profile()
                timings["validation/smolvla_time_s"] += (
                    time.perf_counter() - started
                )

                started = time.perf_counter()
                actions = self.trainer.deterministic_action_chunks_tensor(
                    states=state_tensor,
                    priors=prior,
                    action_count=self.actions_per_policy_decision,
                )
                self._sync_for_profile()
                timings["validation/policy_time_s"] += (
                    time.perf_counter() - started
                )

                for action_index in range(self.actions_per_policy_decision):
                    step_active = active & (decision < reset.horizons)
                    sampled_actions += step_active.sum()

                    self._sync_for_profile()
                    started = time.perf_counter()
                    low_dim = self.backend.step(
                        actions[:, action_index], step_active
                    )
                    low_dim, caught_target, grasp_diagnostics = (
                        self._update_physical_grasp(
                            reset,
                            low_dim,
                            step_active,
                        )
                    )
                    self._sync_for_profile()
                    timings["validation/physics_time_s"] += (
                        time.perf_counter() - started
                    )

                    started = time.perf_counter()
                    result = evaluate_active_sparse_tasks(
                        state=reset.task_state,
                        ee_position=low_dim.ee_position,
                        object_positions=low_dim.object_positions,
                        gripper_opening=low_dim.gripper_opening,
                        caught_target=caught_target,
                        active_mask=step_active,
                        max_steps=10_000,
                        thresholds=self._task_thresholds(),
                        move_to_distance_reward=self.move_to_distance_reward,
                        catch_release_dense_reward=(
                            self.catch_release_dense_reward
                        ),
                        bilateral_contact=grasp_diagnostics[
                            "bilateral_contact"
                        ],
                    )
                    candidate_rewards.copy_(
                        torch.where(
                            step_active,
                            result.rewards.to(dtype=torch.float32),
                            candidate_rewards,
                        )
                    )
                    final_xy_distance.copy_(
                        torch.where(
                            step_active,
                            result.diagnostics["dense_target_distance"].to(
                                dtype=torch.float32
                            ),
                            final_xy_distance,
                        )
                    )
                    candidate_success.logical_or_(result.success)
                    active.logical_and_(~result.terminated)
                    self._sync_for_profile()
                    timings["validation/reward_time_s"] += (
                        time.perf_counter() - started
                    )
                active.logical_and_((decision + 1) < reset.horizons)

        self._sync_for_profile()
        validation_time = time.perf_counter() - validation_started
        return ValidationRound(
            candidate_rewards=candidate_rewards.reshape(
                self.layout.groups_per_rank, group_size
            ),
            candidate_success=candidate_success.reshape(
                self.layout.groups_per_rank, group_size
            ),
            final_xy_distance=final_xy_distance.reshape(
                self.layout.groups_per_rank, group_size
            ),
            group_target_catalog_ids=reset.group_target_catalog_ids,
            group_shell_ids=reset.group_shell_ids,
            group_instruction_ids=reset.group_instruction_ids,
            metrics={
                **timings,
                "validation/time_s": float(validation_time),
                "validation/environment_actions": float(
                    sampled_actions.item()
                ),
                "validation/episodes_per_rank": float(worlds),
            },
        )


def concatenate_collector_rounds(
    rounds: Sequence[CollectorRound],
) -> tuple[dict[str, Any], Any, Any, Any, Any, Any, dict[str, float]]:
    if not rounds:
        raise ValueError("At least one collector round is required.")
    import torch

    keys = tuple(rounds[0].records)
    records = {
        key: torch.cat([item.records[key] for item in rounds], dim=0)
        for key in keys
    }
    mask = torch.cat([item.loss_mask for item in rounds], dim=0)
    rewards = torch.cat([item.candidate_rewards for item in rounds], dim=0)
    successes = torch.cat([item.candidate_success for item in rounds], dim=0)
    task_ids = torch.cat([item.group_instruction_ids for item in rounds], dim=0)
    shell_ids = torch.cat([item.group_shell_ids for item in rounds], dim=0)
    metrics: dict[str, float] = {}
    for key in rounds[0].metrics:
        values = [float(item.metrics.get(key, 0.0)) for item in rounds]
        if key.endswith("_time_s") or "actions" in key or "records" in key or key == "informative_groups":
            metrics[key] = float(sum(values))
        else:
            metrics[key] = float(sum(values) / len(values))
    rollout_time = max(metrics.get("rollout_time_s", 0.0), 1.0e-9)
    metrics["sampled_actions_per_second"] = (
        metrics.get("sampled_environment_actions", 0.0) / rollout_time
    )
    metrics["selected_actions_per_second"] = (
        metrics.get("selected_environment_actions", 0.0) / rollout_time
    )
    metrics["trajectory_work_amplification"] = (
        metrics.get("sampled_environment_actions", 0.0)
        / max(1.0, metrics.get("selected_environment_actions", 0.0))
    )
    return records, mask, rewards, successes, task_ids, shell_ids, metrics
