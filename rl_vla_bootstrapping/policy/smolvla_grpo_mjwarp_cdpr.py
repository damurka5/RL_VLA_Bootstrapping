#!/usr/bin/env python3
"""Rank-local, GPU-resident CDPR GRPO training on MJLab/MuJoCo Warp.

This is deliberately a separate entrypoint from the established CPU trainer.
Each torchrun rank owns a complete simulator batch and complete GRPO groups;
the only distributed communication in the rollout/update loop is update-level
schedule, curriculum, metric, and DDP gradient synchronization.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - the pinned MJLab env includes tensorboard.
    SummaryWriter = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - the pinned MJLab env includes tqdm.
    tqdm = None

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    BatchedReverseFrontierResetter,
    RankLocalCurriculum,
    RankLocalMJWarpGRPOCollector,
    ValidationRound,
    concatenate_collector_rounds,
)
from rl_vla_bootstrapping.policy.rank_local_grpo import (
    RankLocalGroupLayout,
    synchronize_equal_ddp_schedule,
)
from rl_vla_bootstrapping.policy.smolvla_cdpr import load_smolvla_runtime
from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import (
    _configure_distributed,
    _destroy_distributed,
    _log,
    _make_run_dir,
    _require_torch,
    _set_quiet_env,
    _set_seed,
    _silence_output,
    _write_json,
)
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
    SmolVLAGRPOTrainer,
    _resolve_checkpoint,
    parse_args,
)
from rl_vla_bootstrapping.simulation.cdpr_backend import (
    CDPRBackendConfig,
    create_cdpr_backend,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    BatchedCatchReleaseDenseReward,
    BatchedMoveToDistanceReward,
)
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
    ACTIVE_CDPR_CATALOGS,
    OBJECT_VARIANTS,
)


def _metadata_flag(
    metadata: Mapping[str, Any], key: str, default: bool = False
) -> bool:
    value = metadata.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _metadata_bounds(
    metadata: Mapping[str, Any],
    key: str,
    default: tuple[float, float],
    *,
    allow_equal: bool = False,
) -> tuple[float, float]:
    raw = metadata.get(key, default)
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must contain exactly two numbers.") from exc
    if len(values) != 2:
        raise ValueError(f"{key} must contain exactly two numbers.")
    low, high = values
    if low > high or (low == high and not allow_equal):
        relation = "nondecreasing" if allow_equal else "increasing"
        raise ValueError(f"{key} bounds must be {relation}.")
    return low, high


def _sample_random_workspace_ee_group(
    *,
    torch: Any,
    device: Any,
    generator: Any,
    target_xy: Any,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    min_target_xy_distance: float,
    attempts: int = 64,
) -> Any:
    """Sample one guaranteed non-near-target pose per GRPO group."""

    group_count = int(target_xy.shape[0])
    candidate_count = max(1, int(attempts))
    random_values = torch.rand(
        (group_count, candidate_count, 3),
        generator=generator,
        device=device,
    )
    low = torch.tensor(
        (x_bounds[0], y_bounds[0], z_bounds[0]),
        dtype=torch.float32,
        device=device,
    )
    high = torch.tensor(
        (x_bounds[1], y_bounds[1], z_bounds[1]),
        dtype=torch.float32,
        device=device,
    )
    candidates = low + random_values * (high - low)
    distances = torch.linalg.vector_norm(
        candidates[:, :, :2] - target_xy[:, None, :], dim=-1
    )
    valid = distances >= max(0.0, float(min_target_xy_distance))
    first_valid = valid.to(dtype=torch.int64).argmax(dim=1)
    rows = torch.arange(
        group_count, dtype=torch.int64, device=device
    )
    sampled = candidates[rows, first_valid]

    missing = ~valid.any(dim=1)
    if bool(missing.any().item()):
        corners = torch.tensor(
            (
                (x_bounds[0], y_bounds[0]),
                (x_bounds[0], y_bounds[1]),
                (x_bounds[1], y_bounds[0]),
                (x_bounds[1], y_bounds[1]),
            ),
            dtype=torch.float32,
            device=device,
        )
        corner_distances = torch.linalg.vector_norm(
            corners[None, :, :] - target_xy[:, None, :], dim=-1
        )
        farthest = corners.index_select(0, corner_distances.argmax(dim=1))
        sampled[:, :2] = torch.where(
            missing[:, None], farthest, sampled[:, :2]
        )
        final_distances = torch.linalg.vector_norm(
            sampled[:, :2] - target_xy, dim=-1
        )
        impossible = final_distances < max(
            0.0, float(min_target_xy_distance)
        )
        if bool(impossible.any().item()):
            raise ValueError(
                "The configured EE workspace cannot satisfy "
                "random_workspace_min_goal_xy_distance."
            )
    return sampled


class BatchedRandomWorkspaceMoveToResetter(BatchedReverseFrontierResetter):
    """Replace Reverse Frontier poses with full-workspace move-to starts."""

    def __init__(
        self,
        *,
        task_metadata: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(task_metadata=task_metadata, **kwargs)
        metadata = dict(task_metadata or {})
        if not _metadata_flag(
            metadata, "random_workspace_gripper_start", False
        ):
            raise ValueError(
                "Random workspace move-to reset requires "
                "random_workspace_gripper_start=true."
            )
        instruction_ids = tuple(
            int(value) for value in self.instruction_ids.detach().cpu().tolist()
        )
        if instruction_ids != (0,):
            raise ValueError(
                "Random workspace move-to reset supports only "
                "instruction_types=[move_to_object]."
            )
        self.workspace_x_bounds = _metadata_bounds(
            metadata, "ee_workspace_x_bounds", (-0.24, 0.24)
        )
        self.workspace_y_bounds = _metadata_bounds(
            metadata, "ee_workspace_y_bounds", (-0.24, 0.24)
        )
        self.workspace_z_bounds = _metadata_bounds(
            metadata,
            "ee_workspace_z_bounds",
            (0.27, 0.27),
            allow_equal=True,
        )
        self.min_target_xy_distance = max(
            0.0,
            float(
                metadata.get(
                    "random_workspace_min_goal_xy_distance", 0.12
                )
            ),
        )
        low = max(
            1, int(metadata.get("random_workspace_horizon_low", 32))
        )
        high = max(
            low, int(metadata.get("random_workspace_horizon_high", 32))
        )
        self.random_horizon_bounds = (low, high)

    def reset(self, *, update_index: int, round_index: int) -> Any:
        reset = super().reset(
            update_index=update_index, round_index=round_index
        )
        torch = self.torch
        groups = int(self.layout.groups_per_rank)
        group_size = int(self.layout.group_size)
        base_worlds = torch.arange(
            0,
            int(self.layout.worlds_per_rank),
            group_size,
            dtype=torch.int64,
            device=self.device,
        )
        low_dim = self.backend.low_dim_observations()
        target_slots = reset.task_state.target_slots.index_select(
            0, base_worlds
        )
        target_xy = low_dim.object_positions[
            base_worlds, target_slots, :2
        ]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(
            self.base_seed
            + self.rank * 1_000_003
            + int(update_index) * 10_000_019
            + int(round_index) * 100_003
            + 70_001
        )
        ee_group = _sample_random_workspace_ee_group(
            torch=torch,
            device=self.device,
            generator=generator,
            target_xy=target_xy,
            x_bounds=self.workspace_x_bounds,
            y_bounds=self.workspace_y_bounds,
            z_bounds=self.workspace_z_bounds,
            min_target_xy_distance=self.min_target_xy_distance,
        )
        yaw_group = (
            torch.rand((groups,), generator=generator, device=self.device)
            * (2.0 * torch.pi)
            - torch.pi
        )
        self.backend.set_end_effector_poses(
            ee_group.repeat_interleave(group_size, dim=0),
            yaw_group.repeat_interleave(group_size),
        )
        self.backend.set_gripper_openings(
            torch.ones(
                (int(self.layout.worlds_per_rank),),
                dtype=torch.float32,
                device=self.device,
            )
        )
        horizon_low, horizon_high = self.random_horizon_bounds
        group_horizons = torch.randint(
            horizon_low,
            horizon_high + 1,
            (groups,),
            generator=generator,
            dtype=torch.int64,
            device=self.device,
        )
        reset.horizons.copy_(
            group_horizons.repeat_interleave(group_size)
        )
        reset.group_shell_ids.fill_(-1)
        return reset


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True, default=str))
        stream.write("\n")


def _log_tensorboard_metrics(
    writer: Any,
    metrics: Mapping[str, Any],
    step: int,
) -> None:
    """Mirror every finite numeric metric without failing on JSON labels."""

    if writer is None:
        return
    for key, value in metrics.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            writer.add_scalar(str(key), scalar, int(step))
    writer.flush()


def _make_mjwarp_progress_bar(
    *,
    args: Any,
    is_main: bool,
    start_step: int,
) -> Any | None:
    """Create a rank-zero bar even when the remote launcher pipes via tee."""

    if not bool(args.progress) or not is_main or tqdm is None:
        return None
    total = max(0, int(args.max_train_steps))
    initial = min(max(0, int(start_step)), total)
    return tqdm(
        total=total,
        initial=initial,
        desc="[smolvla-mjwarp]",
        unit=" selected-step",
        dynamic_ncols=True,
        mininterval=max(0.1, float(args.progress_refresh_seconds)),
        maxinterval=max(10.0, float(args.progress_refresh_seconds) * 2.0),
        miniters=1,
        smoothing=0.1,
        leave=True,
        disable=False,
        file=sys.__stderr__,
    )


def _end_to_end_time_metrics(
    *,
    start_step: int,
    global_step: int,
    max_train_steps: int,
    elapsed_seconds: float,
) -> dict[str, float]:
    """Estimate completion from full update wall time, including validation."""

    elapsed = max(0.0, float(elapsed_seconds))
    completed = max(
        0,
        min(int(global_step), int(max_train_steps)) - int(start_step),
    )
    remaining = max(0, int(max_train_steps) - int(global_step))
    rate = float(completed) / max(elapsed, 1.0e-9)
    eta = float(remaining) / max(rate, 1.0e-9)
    return {
        "training/elapsed_time_s": elapsed,
        "training/end_to_end_selected_actions_per_second": rate,
        "training/estimated_remaining_time_s": eta,
        "training/estimated_total_time_s": elapsed + eta,
    }


def _update_mjwarp_progress_bar(
    progress: Any | None,
    *,
    previous_display_step: int,
    global_step: int,
    max_train_steps: int,
    update_index: int,
    metrics: Mapping[str, Any],
) -> int:
    """Advance by global selected actions and retain rollout diagnostics."""

    displayed_step = min(
        max(0, int(global_step)), max(0, int(max_train_steps))
    )
    if progress is None:
        return displayed_step
    progress.set_postfix(
        {
            "update": int(update_index),
            "sampled/s": (
                f"{float(metrics.get('sampled_actions_per_second_global', 0.0)):.1f}"
            ),
            "rollout-selected/s": (
                f"{float(metrics.get('selected_actions_per_second_global', 0.0)):.1f}"
            ),
            "success": (
                f"{float(metrics.get('candidate_successes', 0.0)):.0f}/"
                f"{float(metrics.get('candidate_worlds', 0.0)):.0f}"
            ),
            "records": (
                f"{float(metrics.get('informative_records', 0.0)):.0f}"
            ),
        },
        refresh=False,
    )
    progress.update(max(0, displayed_step - int(previous_display_step)))
    return displayed_step


def _validation_enabled(args: Any) -> bool:
    return (
        int(getattr(args, "validation_every_steps", 0)) > 0
        and int(getattr(args, "validation_episodes_per_instruction", 0)) > 0
    )


def _validation_due(
    args: Any,
    *,
    global_step: int,
    last_validation_step: int,
) -> bool:
    if not _validation_enabled(args) or int(global_step) <= 0:
        return False
    every = max(1, int(args.validation_every_steps))
    return (
        int(global_step) // every
        > int(last_validation_step) // every
    )


def _distributed_barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _synchronize_validation_rounds(
    rounds: Sequence[ValidationRound],
    *,
    device: Any,
) -> dict[str, float]:
    """Aggregate held-out candidate results once across all GPU ranks."""

    if not rounds:
        raise ValueError("At least one validation round is required.")
    import torch
    import torch.distributed as dist

    catalog_count = len(ACTIVE_CDPR_CATALOGS)
    counts = torch.zeros(
        (catalog_count,), dtype=torch.float64, device=device
    )
    successes = torch.zeros_like(counts)
    rewards = torch.zeros_like(counts)
    distances = torch.zeros_like(counts)
    environment_actions = torch.zeros(
        (), dtype=torch.float64, device=device
    )
    timing_keys = tuple(
        sorted(
            {
                key
                for item in rounds
                for key in item.metrics
                if key.endswith("_time_s") or key == "validation/time_s"
            }
        )
    )
    timing_values = torch.tensor(
        [
            sum(float(item.metrics.get(key, 0.0)) for item in rounds)
            for key in timing_keys
        ],
        dtype=torch.float64,
        device=device,
    )
    group_size = int(rounds[0].candidate_success.shape[1])
    for item in rounds:
        catalog_ids = item.group_target_catalog_ids.repeat_interleave(
            group_size
        )
        flat_success = item.candidate_success.reshape(-1).to(
            dtype=torch.float64
        )
        flat_rewards = item.candidate_rewards.reshape(-1).to(
            dtype=torch.float64
        )
        flat_distances = item.final_xy_distance.reshape(-1).to(
            dtype=torch.float64
        )
        one = torch.ones_like(flat_success)
        counts.index_add_(0, catalog_ids, one)
        successes.index_add_(0, catalog_ids, flat_success)
        rewards.index_add_(0, catalog_ids, flat_rewards)
        distances.index_add_(0, catalog_ids, flat_distances)
        environment_actions += float(
            item.metrics.get("validation/environment_actions", 0.0)
        )

    if dist.is_available() and dist.is_initialized():
        for tensor in (counts, successes, rewards, distances):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(environment_actions, op=dist.ReduceOp.SUM)
        if timing_values.numel() > 0:
            dist.all_reduce(timing_values, op=dist.ReduceOp.MAX)

    total_count = counts.sum().clamp_min(1.0)
    metrics = {
        "validation/episodes": float(counts.sum().item()),
        "validation/success_rate": float(
            (successes.sum() / total_count).item()
        ),
        "validation/reward_mean": float(
            (rewards.sum() / total_count).item()
        ),
        "validation/final_xy_distance_mean_m": float(
            (distances.sum() / total_count).item()
        ),
        "validation/environment_actions": float(environment_actions.item()),
        "validation/rounds_per_rank": float(len(rounds)),
    }
    for key, value in zip(
        timing_keys, timing_values.detach().cpu().tolist()
    ):
        metrics[key] = float(value)
    for catalog_index, catalog_name in enumerate(ACTIVE_CDPR_CATALOGS):
        count = float(counts[catalog_index].item())
        if count <= 0.0:
            continue
        label = OBJECT_VARIANTS[catalog_name].label.replace(" ", "_")
        prefix = f"validation/by_object/{label}"
        metrics[f"{prefix}/episodes"] = count
        metrics[f"{prefix}/success_rate"] = float(
            successes[catalog_index].item() / count
        )
        metrics[f"{prefix}/reward_mean"] = float(
            rewards[catalog_index].item() / count
        )
        metrics[f"{prefix}/final_xy_distance_mean_m"] = float(
            distances[catalog_index].item() / count
        )
    return metrics


def _run_gpu_validation(
    *,
    args: Any,
    collector: RankLocalMJWarpGRPOCollector,
    trainer: Any,
    device: Any,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    """Evaluate fixed held-out scenes while preserving the training RNG."""

    import torch

    requested = int(args.validation_episodes_per_instruction)
    global_worlds_per_round = (
        int(collector.layout.worlds_per_rank) * int(world_size)
    )
    round_count = max(
        1,
        int(math.ceil(requested / max(1, global_worlds_per_round))),
    )
    device_index = int(device.index or torch.cuda.current_device())
    validation_seed = int(args.validation_seed) + int(rank) * 1_000_003
    was_training = bool(trainer.actor.training)
    trainer.actor.eval()
    try:
        with torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(validation_seed)
            torch.cuda.manual_seed(validation_seed)
            rounds = [
                collector.validate_round(round_index=round_index)
                for round_index in range(round_count)
            ]
    finally:
        trainer.actor.train(was_training)
    metrics = _synchronize_validation_rounds(rounds, device=device)
    metrics["validation/requested_episodes"] = float(requested)
    metrics["validation/seed"] = float(args.validation_seed)
    return metrics


def _scene_object_curriculum_steps(
    metadata: Mapping[str, Any]
) -> tuple[int, ...]:
    """Global steps at which one more scene object becomes available."""

    raw = metadata.get("scene_object_curriculum_steps") or ()
    if isinstance(raw, (str, bytes)):
        raise ValueError(
            "scene_object_curriculum_steps must be a list of global steps."
        )
    return tuple(sorted(int(value) for value in raw))


def _apply_scene_object_curriculum(
    resetter: Any,
    *,
    curriculum_steps: tuple[int, ...],
    global_step: int,
) -> tuple[int, int]:
    """Unlock one additional scene object per passed threshold."""

    scene_min, scene_max = resetter.scene_object_bounds
    if curriculum_steps:
        unlocked = sum(
            1 for threshold in curriculum_steps if global_step >= threshold
        )
        resetter.set_scene_object_range(scene_min, scene_min + unlocked)
    return resetter.scene_object_range


def _task_metadata(args: Any) -> dict[str, Any]:
    raw = os.environ.get("RLVLA_TASK_METADATA_JSON", "").strip()
    if raw:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("RLVLA_TASK_METADATA_JSON must encode an object.")
        return dict(payload)
    if args.config:
        config = load_project_config(Path(args.config).expanduser().resolve())
        return dict(config.task.metadata or {})
    return {}


def _synchronize_update_metrics_once(
    metrics: Mapping[str, float],
    *,
    device: Any,
) -> dict[str, float]:
    """Synchronize all metrics at the update boundary, never per group."""

    import torch
    import torch.distributed as dist

    keys = tuple(sorted(metrics))
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    world_size = 1
    wall_keys = ("rollout_time_s", "update_time_s")
    wall_values = torch.tensor(
        [float(metrics.get(key, 0.0)) for key in wall_keys],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(wall_values, op=dist.ReduceOp.MAX)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    summed = {
        key: float(value)
        for key, value in zip(keys, values.detach().cpu().tolist())
    }
    summed["distributed_world_size"] = float(world_size)
    # Counts and work are global sums. Component times and scalar optimizer
    # diagnostics are reported as rank means from the same collective.
    for key in keys:
        if (
            key.endswith("_time_s")
            or key.endswith("_mean")
            or key.endswith("_max")
            or key.endswith("_std")
            or key.endswith("_rate")
            or "_mean_" in key
            or key.startswith("loss_")
            or key
            in {
                "entropy_mean",
                "approx_kl_mean",
                "clip_fraction_mean",
                "smolvla_batch_size",
                "smolvla_inference_microbatch_size",
                "complete_groups_per_rank",
                "group_pass_rate_mean",
                "padded_records",
                "backward_collectives",
                "optimizer_steps",
                "timers_cuda_synchronized",
                "profiled_update",
                "dense_move_to_distance_reward",
                "dense_catch_release_reward",
                # Per-rank LoRA diagnostics: report the rank mean, not the sum
                # (vla_lora/records stays a sum -- it is a global count).
                "vla_lora/ppo_loss",
                "vla_lora/kl",
                "vla_lora/grad_norm",
            }
        ):
            summed[key] /= float(world_size)
    for key, value in zip(
        wall_keys, wall_values.detach().cpu().tolist()
    ):
        summed[key] = float(value)
    rollout_wall = max(summed.get("rollout_time_s", 0.0), 1.0e-9)
    summed["sampled_actions_per_second_global"] = (
        summed.get("sampled_environment_actions", 0.0) / rollout_wall
    )
    summed["selected_actions_per_second_global"] = (
        summed.get("selected_environment_actions", 0.0) / rollout_wall
    )
    summed["trajectory_work_amplification"] = (
        summed.get("sampled_environment_actions", 0.0)
        / max(1.0, summed.get("selected_environment_actions", 0.0))
    )
    return summed


def _runtime_metadata(args: Any, backend: Any) -> dict[str, Any]:
    metadata = backend.metadata()
    metadata.update(
        {
            "entrypoint": "smolvla_grpo_mjwarp_cdpr",
            "global_step_definition": (
                "cumulative selected environment actions summed across ranks"
            ),
            "smolvla_precision": str(args.mixed_precision),
            "smolvla_inference_microbatch_size": int(
                args.smolvla_inference_microbatch_size
            ),
        }
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if str(args.simulator_backend) != "mjlab_mjwarp":
        raise SystemExit(
            "This entrypoint requires --simulator-backend mjlab_mjwarp. "
            "Use smolvla_grpo_finetune_cdpr.py for the CPU backend."
        )
    if not bool(args.grpo_trajectory_groups):
        raise SystemExit(
            "MJWarp training requires --grpo-trajectory-groups true because "
            "worlds are terminal continuation candidates."
        )

    _require_torch()
    import torch

    dist_ctx = _configure_distributed(args)
    _set_quiet_env(args, dist_ctx)
    _set_seed(int(args.seed) + int(dist_ctx.rank) * 1_000_003)
    device = torch.device(str(dist_ctx.device))
    if device.type != "cuda":
        _destroy_distributed(dist_ctx)
        raise RuntimeError("MJWarp production training requires one CUDA GPU per rank.")

    layout = RankLocalGroupLayout(
        worlds_per_rank=int(args.worlds_per_rank),
        groups_per_rank=int(args.groups_per_rank),
        group_size=int(args.grpo_group_size),
    )
    layout.validate()
    layout.assert_no_cross_rank_group(dist_ctx.rank, dist_ctx.world_size)
    run_dir = _make_run_dir(args.run_root_dir, args.run_id)
    metrics_path = run_dir / "metrics.jsonl"
    validation_path = run_dir / "validation.jsonl"
    writer = None
    progress = None
    if dist_ctx.is_main:
        _write_json(run_dir / "config.json", vars(args))
        if SummaryWriter is None:
            _log(
                dist_ctx,
                "[smolvla-mjwarp] TensorBoard is unavailable; install the "
                "pinned cdpr-mjlab requirements.",
            )
        else:
            writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    backend = None
    runtime = None
    try:
        backend_config = CDPRBackendConfig(
            backend="mjlab_mjwarp",
            worlds_per_rank=int(args.worlds_per_rank),
            groups_per_rank=int(args.groups_per_rank),
            grpo_group_size=int(args.grpo_group_size),
            hold_steps=int(args.hold_steps),
            action_step_xyz=float(args.action_step_xyz),
            action_step_yaw=float(args.action_step_yaw),
            action_step_gripper=float(args.action_step_gripper),
            lock_non_commanded_axes=bool(args.lock_non_commanded_axes),
            lock_non_commanded_axes_threshold=float(
                args.lock_non_commanded_axes_threshold
            ),
            render_width=int(args.render_width),
            render_height=int(args.render_height),
            object_slots=int(args.object_slots),
            nconmax=int(args.mjwarp_nconmax),
            njmax=int(args.mjwarp_njmax),
            nccdmax=args.mjwarp_nccdmax,
            device=str(device),
            xml_path=Path(args.mjwarp_xml_path),
        )
        _log(
            dist_ctx,
            "[smolvla-mjwarp] allocating "
            f"{layout.worlds_per_rank} worlds / {layout.groups_per_rank} "
            f"complete groups on rank {dist_ctx.rank} ({device})",
        )
        backend = create_cdpr_backend(backend_config)

        _log(
            dist_ctx,
            f"[smolvla-mjwarp] loading frozen SmolVLA replica on {device}: "
            f"{args.base_checkpoint}",
        )
        with _silence_output(not dist_ctx.is_main):
            runtime = load_smolvla_runtime(
                checkpoint=str(args.base_checkpoint),
                device=str(device),
                mixed_precision=str(args.mixed_precision),
                image_size=int(args.image_size),
                state_dim=int(args.state_dim),
                image_feature_keys=(
                    None
                    if args.image_feature_keys is None
                    else tuple(args.image_feature_keys)
                ),
                include_wrist=bool(args.include_wrist),
                include_aux_camera=bool(args.include_aux_camera),
                mask_empty_aux_camera=bool(args.mask_empty_aux_camera),
                chunk_size=int(args.chunk_size),
                action_dim=int(args.action_dim),
                action_indices=(
                    None
                    if args.smolvla_action_indices is None
                    else tuple(int(value) for value in args.smolvla_action_indices)
                ),
                action_normalization=str(args.smolvla_action_normalization),
                model_image_size=(
                    None
                    if int(args.smolvla_model_image_size) <= 0
                    else int(args.smolvla_model_image_size)
                ),
                # Compilation stays on even with LoRA: the runtime swaps in the
                # eager sample_actions only for the small grad-enabled
                # microbatches, so the no_grad rollout keeps its speedup.
                compile_model=bool(args.smolvla_compile_model),
                compile_mode=str(args.smolvla_compile_mode),
            )

        include_relative_target = bool(
            getattr(args, "residual_relative_target", False)
        )
        # The frozen SmolVLA replica keeps its native state width; only the
        # trainable residual is widened by the 3-D target-relative vector.
        residual_state_dim = int(args.state_dim) + (
            3 if include_relative_target else 0
        )
        trainer = SmolVLAGRPOTrainer(
            args=args,
            state_dim=residual_state_dim,
            action_dim=int(args.action_dim),
            chunk_size=int(args.chunk_size),
            run_dir=run_dir,
            device=device,
            distributed=dist_ctx,
        )
        train_vla_lora = bool(getattr(args, "train_vla_lora", False))
        lora_info: dict[str, float] = {}
        if train_vla_lora:
            lora_info = trainer.attach_vla_lora(runtime)
            _log(
                dist_ctx,
                "[smolvla-mjwarp] LoRA on action expert: "
                f"{lora_info['vla_lora/modules']:.0f} modules, "
                f"{lora_info['vla_lora/trainable_params']:.0f} trainable params",
            )
        simulator_metadata = _runtime_metadata(args, backend)
        global_step = 0
        if args.resume_checkpoint:
            checkpoint = _resolve_checkpoint(args.resume_checkpoint)
            global_step = trainer.load(
                checkpoint,
                expected_simulator_metadata=simulator_metadata,
                allow_legacy_simulator_metadata=bool(
                    args.allow_legacy_simulator_checkpoint
                ),
            )
            _log(
                dist_ctx,
                f"[smolvla-mjwarp] resumed {checkpoint} at global step "
                f"{global_step}",
            )

        curriculum = RankLocalCurriculum(
            device=device,
            promotion_success=float(args.reverse_frontier_promotion_success),
            demotion_success=float(args.reverse_frontier_demotion_success),
            validation_rollouts_per_shell=int(
                args.reverse_frontier_validation_episodes
            ),
            min_updates=int(args.reverse_frontier_min_train_updates),
            saturation_abort_threshold=float(
                args.reverse_frontier_saturation_abort_threshold
            ),
        )
        curriculum_state = trainer.loaded_extra_state.get("curriculum")
        if not isinstance(curriculum_state, Mapping):
            legacy_complex = trainer.loaded_extra_state.get("complex_runtime")
            if isinstance(legacy_complex, Mapping):
                curriculum_state = legacy_complex
        if isinstance(curriculum_state, Mapping):
            curriculum.restore(curriculum_state)
        task_metadata = _task_metadata(args)
        reverse_frontier_active = (
            str(args.complex_training_approach) == "reverse_frontier"
        )
        reward_mode = str(
            task_metadata.get("reward_mode", "sparse_binary")
        ).strip().lower()
        move_to_distance_reward = None
        catch_release_dense_reward = None
        if reward_mode == "dense":
            configured_instructions = tuple(args.instruction_types or ())
            supported_dense_instructions = {
                "move_to_object",
                "put_into_plate",
                "put_into_bowl",
                "pick_up",
            }
            unsupported = tuple(
                name
                for name in configured_instructions
                if name not in supported_dense_instructions
            )
            if unsupported:
                raise RuntimeError(
                    "The MJWarp dense reward path supports move_to_object, "
                    "put_into_plate, put_into_bowl, and pick_up; received "
                    f"unsupported values {unsupported!r}."
                )
            if "move_to_object" in configured_instructions:
                move_to_distance_reward = (
                    BatchedMoveToDistanceReward.from_metadata(task_metadata)
                )
            if {
                "put_into_plate",
                "put_into_bowl",
                "pick_up",
            }.intersection(configured_instructions):
                catch_release_dense_reward = (
                    BatchedCatchReleaseDenseReward.from_metadata(
                        task_metadata
                    )
                )
        resetter_kwargs = {
            "backend": backend,
            "layout": layout,
            "curriculum": curriculum,
            "rank": dist_ctx.rank,
            "base_seed": int(args.seed),
            "instruction_types": args.instruction_types,
            "allowed_objects": args.allowed_objects,
            "frontier_probability": float(
                args.reverse_frontier_sample_probability
            ),
            "rehearsal_probability": float(
                args.reverse_frontier_rehearsal_probability
            ),
        }
        random_workspace_move_to = (
            not reverse_frontier_active
            and tuple(args.instruction_types or ()) == ("move_to_object",)
        )
        if random_workspace_move_to:
            resetter = BatchedRandomWorkspaceMoveToResetter(
                **resetter_kwargs,
                task_metadata=task_metadata,
            )
        else:
            resetter = BatchedReverseFrontierResetter(
                **resetter_kwargs,
                task_metadata=task_metadata,
            )
        collector = RankLocalMJWarpGRPOCollector(
            backend=backend,
            smolvla_runtime=runtime,
            trainer=trainer,
            resetter=resetter,
            layout=layout,
            actions_per_policy_decision=int(args.replan_every),
            smolvla_microbatch_size=int(args.smolvla_inference_microbatch_size),
            normalize_advantage=bool(args.grpo_normalize_group_advantage),
            advantage_clip_abs=float(args.grpo_clip_advantage_abs),
            dynamic_min_pass_rate=float(args.grpo_dynamic_min_pass_rate),
            dynamic_max_pass_rate=float(args.grpo_dynamic_max_pass_rate),
            dynamic_sampling=bool(args.grpo_dynamic_sampling),
            group_selection=str(args.grpo_group_selection),
            move_to_distance_reward=move_to_distance_reward,
            catch_release_dense_reward=catch_release_dense_reward,
            include_relative_target=include_relative_target,
            store_vla_records=train_vla_lora,
            vla_update_max_records=int(
                getattr(args, "vla_update_max_records", 128)
            ),
            profile=bool(args.mjwarp_profile_timers),
        )
        validation_collector = None
        if _validation_enabled(args):
            validation_resetter_kwargs = {
                **resetter_kwargs,
                "base_seed": int(args.validation_seed),
                "frontier_probability": 1.0,
                "rehearsal_probability": 0.0,
                "balanced_target_catalogs": True,
            }
            if random_workspace_move_to:
                validation_resetter = (
                    BatchedRandomWorkspaceMoveToResetter(
                        **validation_resetter_kwargs,
                        task_metadata=task_metadata,
                    )
                )
            else:
                validation_resetter = BatchedReverseFrontierResetter(
                    **validation_resetter_kwargs,
                    task_metadata=task_metadata,
                )
            validation_collector = RankLocalMJWarpGRPOCollector(
                backend=backend,
                smolvla_runtime=runtime,
                trainer=trainer,
                resetter=validation_resetter,
                layout=layout,
                actions_per_policy_decision=int(args.replan_every),
                smolvla_microbatch_size=int(
                    args.smolvla_inference_microbatch_size
                ),
                normalize_advantage=bool(
                    args.grpo_normalize_group_advantage
                ),
                advantage_clip_abs=float(args.grpo_clip_advantage_abs),
                dynamic_min_pass_rate=float(
                    args.grpo_dynamic_min_pass_rate
                ),
                dynamic_max_pass_rate=float(
                    args.grpo_dynamic_max_pass_rate
                ),
                dynamic_sampling=False,
                group_selection="uniform",
                move_to_distance_reward=move_to_distance_reward,
                catch_release_dense_reward=catch_release_dense_reward,
                include_relative_target=include_relative_target,
                profile=bool(args.mjwarp_profile_timers),
            )

        scene_curriculum_steps = _scene_object_curriculum_steps(task_metadata)
        update_index = int(curriculum.updates)
        start_update_index = int(update_index)
        last_saved_step = int(global_step)
        validation_state = trainer.loaded_extra_state.get("validation")
        if isinstance(validation_state, Mapping):
            last_validation_step = int(
                validation_state.get("last_step", global_step)
            )
        else:
            last_validation_step = int(global_step)
        run_start_step = int(global_step)
        training_started = time.perf_counter()
        progress = _make_mjwarp_progress_bar(
            args=args,
            is_main=dist_ctx.is_main,
            start_step=run_start_step,
        )
        progress_display_step = min(
            run_start_step, int(args.max_train_steps)
        )
        while (
            global_step < int(args.max_train_steps)
            and (
                int(args.mjwarp_max_updates) <= 0
                or update_index - start_update_index
                < int(args.mjwarp_max_updates)
            )
        ):
            scene_object_range = _apply_scene_object_curriculum(
                resetter,
                curriculum_steps=scene_curriculum_steps,
                global_step=global_step,
            )
            profile_limit = int(args.mjwarp_profile_updates)
            profile_this_update = bool(args.mjwarp_profile_timers) and (
                profile_limit <= 0
                or update_index - start_update_index < profile_limit
            )
            collector.profile = profile_this_update
            trainer.profile_update = profile_this_update
            update_started = time.perf_counter()
            rounds = []
            local_informative = 0
            max_rounds = max(
                1,
                (
                    int(args.grpo_max_groups_per_update)
                    + int(layout.groups_per_rank)
                    - 1
                )
                // int(layout.groups_per_rank),
            )
            for round_index in range(max_rounds):
                item = collector.collect_round(
                    update_index=update_index,
                    round_index=round_index,
                )
                rounds.append(item)
                local_informative += int(item.loss_mask.sum().item())
                if (
                    int(args.grpo_target_records_per_update) <= 0
                    or local_informative
                    >= int(args.grpo_target_records_per_update)
                ):
                    break
                if (
                    float(args.grpo_max_collection_seconds_per_update) > 0.0
                    and time.perf_counter() - update_started
                    >= float(args.grpo_max_collection_seconds_per_update)
                ):
                    break

            (
                records,
                loss_mask,
                candidate_rewards,
                successes,
                task_ids,
                shell_ids,
                rollout_metrics,
            ) = (
                concatenate_collector_rounds(rounds)
            )
            synchronization_time = 0.0
            synchronization_started = time.perf_counter()
            schedule = synchronize_equal_ddp_schedule(
                local_informative_records=int(loss_mask.sum().item()),
                records_per_minibatch=int(args.minibatch_size),
                ppo_epochs=int(args.ppo_epochs),
                device=device,
            )
            synchronization_time += (
                time.perf_counter() - synchronization_started
            )
            update_timer = time.perf_counter()
            update_metrics = trainer.update_tensor_records(
                records,
                loss_mask=loss_mask,
                schedule=schedule,
            )
            torch.cuda.synchronize(device)
            update_metrics["update_time_s"] = time.perf_counter() - update_timer

            if train_vla_lora:
                # Every rank must call update_vla_lora (it issues a collective to
                # sync LoRA grads), even with an empty batch, to stay in lockstep.
                vla_batches = [
                    item.vla_records
                    for item in rounds
                    if item.vla_records is not None
                ]
                merged_vla: dict[str, Any] = {}
                if vla_batches:
                    tensor_keys = (
                        "overview",
                        "wrist",
                        "state",
                        "action",
                        "action_index",
                        "old_log_prob",
                        "prior_ref",
                        "advantage",
                    )
                    for key in tensor_keys:
                        merged_vla[key] = torch.cat(
                            [batch[key] for batch in vla_batches], dim=0
                        )
                    instructions: list[str] = []
                    for batch in vla_batches:
                        instructions.extend(batch["instruction"])
                    merged_vla["instruction"] = instructions
                    cap = int(getattr(args, "vla_update_max_records", 128))
                    if cap > 0 and int(merged_vla["advantage"].shape[0]) > cap:
                        for key in tensor_keys:
                            merged_vla[key] = merged_vla[key][:cap]
                        merged_vla["instruction"] = merged_vla["instruction"][
                            :cap
                        ]
                vla_metrics = trainer.update_vla_lora(merged_vla)
                torch.cuda.synchronize(device)
                update_metrics.update(vla_metrics)

            synchronization_started = time.perf_counter()
            if reverse_frontier_active:
                curriculum_metrics = (
                    curriculum.update_once_per_optimizer_update(
                        group_instruction_ids=task_ids,
                        group_shell_ids=shell_ids,
                        candidate_success=successes,
                    )
                )
            else:
                curriculum_metrics = {
                    "curriculum/enabled": 0.0,
                    "curriculum/random_workspace_reset": 1.0,
                }
            synchronization_time += (
                time.perf_counter() - synchronization_started
            )
            capacity = backend.capacity_status()
            local_metrics = {
                **rollout_metrics,
                **update_metrics,
                "candidate_successes": float(successes.sum().item()),
                "candidate_worlds": float(successes.numel()),
                "candidate_reward_sum": float(candidate_rewards.sum().item()),
                "candidate_reward_count": float(candidate_rewards.numel()),
                "groups_collected": float(successes.shape[0]),
                "contacts_rank_sum": float(capacity["contacts"]),
                "max_constraints_per_world_rank_sum": float(
                    capacity["max_constraints_per_world"]
                ),
                "contact_capacity_overflow_ranks": float(
                    capacity["contact_overflow"]
                ),
                "constraint_capacity_overflow_ranks": float(
                    capacity["constraint_overflow"]
                ),
                "updates": 1.0,
            }
            synchronization_started = time.perf_counter()
            synchronized_metrics = _synchronize_update_metrics_once(
                local_metrics, device=device
            )
            synchronization_time += (
                time.perf_counter() - synchronization_started
            )
            global_selected = int(
                synchronized_metrics.get("selected_environment_actions", 0.0)
            )
            global_step += global_selected
            update_index += 1
            if (
                synchronized_metrics["contact_capacity_overflow_ranks"] > 0
                or synchronized_metrics["constraint_capacity_overflow_ranks"] > 0
            ):
                raise RuntimeError(
                    "MJWarp contact/constraint capacity overflow: "
                    f"local={capacity}, synchronized={synchronized_metrics}. "
                    "Increase simulator.nconmax/mjwarp_nconmax or "
                    "simulator.njmax/mjwarp_njmax; the backend will not continue "
                    "with truncated physics."
                )
            synchronized_metrics.update(curriculum_metrics)
            synchronized_metrics.update(
                {
                    "curriculum/scene_objects_min": float(
                        scene_object_range[0]
                    ),
                    "curriculum/scene_objects_max": float(
                        scene_object_range[1]
                    ),
                    # Attach-time LoRA facts are repeated on every row so any
                    # tool reading the latest metrics (benchmarks, TensorBoard)
                    # can report them without parsing the startup log.
                    **lora_info,
                    "global_step": float(global_step),
                    "update_index": float(update_index),
                    "global_step_increment": float(global_selected),
                    "worlds_per_rank": float(layout.worlds_per_rank),
                    "groups_per_rank": float(layout.groups_per_rank),
                    "grpo_group_size": float(layout.group_size),
                    "synchronization_time_s_rank0": float(
                        synchronization_time
                    ),
                    "contact_capacity": float(capacity["contact_capacity"]),
                    "constraint_capacity_per_world": float(
                        capacity["constraint_capacity_per_world"]
                    ),
                    "profiled_update": float(profile_this_update),
                }
            )
            if profile_this_update:
                profiled_components = {
                    "smolvla_inference": float(
                        synchronized_metrics.get("smolvla_time_s", 0.0)
                    ),
                    "environment_step": float(
                        synchronized_metrics.get("physics_time_s", 0.0)
                    ),
                    "scene_reset": float(
                        synchronized_metrics.get("reset_time_s", 0.0)
                    ),
                    "backpropagation": float(
                        synchronized_metrics.get(
                            "backpropagation_time_s", 0.0
                        )
                    ),
                }
                selected_denominator = max(1.0, float(global_selected))
                for name, seconds in profiled_components.items():
                    synchronized_metrics[
                        f"profile/{name}_time_s"
                    ] = seconds
                    synchronized_metrics[
                        f"profile/{name}_ms_per_selected_action"
                    ] = 1000.0 * seconds / selected_denominator
                dominant_name, dominant_seconds = max(
                    profiled_components.items(), key=lambda item: item[1]
                )
                synchronized_metrics[
                    "profile/dominant_stage"
                ] = dominant_name
                synchronized_metrics[
                    "profile/dominant_stage_time_s"
                ] = dominant_seconds

            validation_due = _validation_due(
                args,
                global_step=global_step,
                last_validation_step=last_validation_step,
            )
            if validation_due:
                if validation_collector is None:
                    raise RuntimeError(
                        "Validation became due without a GPU validation collector."
                    )
                validation_metrics = _run_gpu_validation(
                    args=args,
                    collector=validation_collector,
                    trainer=trainer,
                    device=device,
                    rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                )
                validation_metrics.update(
                    {
                        "global_step": float(global_step),
                        "update_index": float(update_index),
                        "validation/current_move_to_shell": float(
                            curriculum.current_shell[0].detach().item()
                            if reverse_frontier_active
                            else -1
                        ),
                        "validation/random_workspace_reset": float(
                            not reverse_frontier_active
                        ),
                    }
                )
                last_validation_step = int(global_step)
                if dist_ctx.is_main:
                    _append_jsonl(validation_path, validation_metrics)
                    _log_tensorboard_metrics(
                        writer, validation_metrics, global_step
                    )
                    _log(
                        dist_ctx,
                        "[smolvla-mjwarp] validation "
                        f"step={global_step} "
                        f"success={validation_metrics['validation/success_rate']:.4f} "
                        f"reward={validation_metrics['validation/reward_mean']:.4f} "
                        f"episodes={validation_metrics['validation/episodes']:.0f}",
                        progress=progress,
                    )

            save_due = (
                int(args.save_every_steps) > 0
                and global_step - last_saved_step >= int(args.save_every_steps)
            )
            final_update = global_step >= int(args.max_train_steps)
            final_update = final_update or (
                int(args.mjwarp_max_updates) > 0
                and update_index - start_update_index
                >= int(args.mjwarp_max_updates)
            )
            checkpoint_due = bool(
                save_due or validation_due or final_update
            )
            if dist_ctx.is_main and checkpoint_due:
                checkpoint_path = trainer.save(
                    global_step=global_step,
                    args=args,
                    latest=False,
                    extra_state={
                        "curriculum": curriculum.snapshot(),
                        "validation": {
                            "last_step": int(last_validation_step),
                        },
                    },
                    simulator_metadata=simulator_metadata,
                )
                _log(
                    dist_ctx,
                    f"[smolvla-mjwarp] checkpoint={checkpoint_path}",
                    progress=progress,
                )
            if checkpoint_due:
                last_saved_step = int(global_step)
                _distributed_barrier()

            synchronized_metrics.update(
                _end_to_end_time_metrics(
                    start_step=run_start_step,
                    global_step=global_step,
                    max_train_steps=int(args.max_train_steps),
                    elapsed_seconds=(
                        time.perf_counter() - training_started
                    ),
                )
            )
            if dist_ctx.is_main:
                _append_jsonl(metrics_path, synchronized_metrics)
                _log_tensorboard_metrics(
                    writer, synchronized_metrics, global_step
                )
                if profile_this_update:
                    _write_json(
                        run_dir / "latest_profile.json",
                        {
                            key: value
                            for key, value in synchronized_metrics.items()
                            if key.startswith("profile/")
                            or key.startswith("training/")
                            or key
                            in {
                                "global_step",
                                "update_index",
                                "sampled_environment_actions",
                                "selected_environment_actions",
                            }
                        },
                    )
                progress_display_step = _update_mjwarp_progress_bar(
                    progress,
                    previous_display_step=progress_display_step,
                    global_step=global_step,
                    max_train_steps=int(args.max_train_steps),
                    update_index=update_index,
                    metrics=synchronized_metrics,
                )
                if progress is None:
                    eta_seconds = synchronized_metrics[
                        "training/estimated_remaining_time_s"
                    ]
                    _log(
                        dist_ctx,
                        "[smolvla-mjwarp] "
                        f"step={global_step}/{int(args.max_train_steps)} "
                        f"update={update_index} "
                        f"sampled={synchronized_metrics['sampled_actions_per_second_global']:.1f} "
                        f"selected={synchronized_metrics['selected_actions_per_second_global']:.1f} "
                        f"e2e={synchronized_metrics['training/end_to_end_selected_actions_per_second']:.1f} "
                        f"eta={eta_seconds / 3600.0:.2f}h "
                        f"success={synchronized_metrics['candidate_successes']:.0f}/"
                        f"{synchronized_metrics['candidate_worlds']:.0f} "
                        f"records={synchronized_metrics.get('informative_records', 0.0):.0f}",
                    )

            # MJWarp allocates through Warp's own CUDA allocator, while
            # PyTorch's caching allocator never hands freed blocks back to the
            # driver. Without this the cache creeps up (validation rounds and
            # the LoRA backward both raise the high-water mark) until Warp
            # cannot get its transient physics buffers and dies with
            # "Warp CUDA error 2: out of memory". Once per update is
            # negligible next to a ~100 s update.
            torch.cuda.empty_cache()
    finally:
        if progress is not None:
            progress.close()
        if writer is not None:
            writer.close()
        if backend is not None:
            backend.close()
        runtime = None
        _destroy_distributed(dist_ctx)


if __name__ == "__main__":
    main()
