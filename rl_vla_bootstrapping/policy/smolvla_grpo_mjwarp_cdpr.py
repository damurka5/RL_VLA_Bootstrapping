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
import warnings
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
    ACTIVE_INSTRUCTION_TYPES,
    INSTRUCTION_TO_ID,
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


class BatchedRandomWorkspaceMoveToResetter(BatchedReverseFrontierResetter):
    """Move-to starts drawn from the whole EE workspace, with no shells.

    The base resetter already owns this start distribution: with
    ``random_workspace_gripper_start`` it samples the EE uniformly inside
    ``ee_workspace_{x,y,z}_bounds`` and then applies everything the approach
    curriculum needs -- the per-instruction start-distance cap, the min-goal
    floor relaxed to half the cap, the pull-into-annulus toward the *sampled*
    target slot, the cap-includes-Z height confinement, and the
    curriculum-coupled horizon. So this subclass only validates the
    configuration and clears the Reverse Frontier shell id.

    It must NOT re-sample the start pose. It used to: it called
    ``super().reset()`` and then overwrote both the EE poses (from a sampler
    that took a MINIMUM goal distance and no maximum) and the horizons (from a
    fixed ``randint``). That silently discarded every curriculum signal --
    ``curriculum/start_max_goal_distance_m`` logged a cap that never reached
    the simulator and ``curriculum_horizon_coupling_enabled`` did nothing. In
    the 2M-step move-to run the measured pass rate was then independent of the
    cap (0.044 at 0.03 m vs 0.037 at 0.34 m), so the success gate promoted on a
    distance-independent background rate and marched the cap 0.03 -> 0.34 m in
    1.6M steps while the policy never once saw a close start.
    """

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

    def reset(self, *, update_index: int, round_index: int) -> Any:
        reset = super().reset(
            update_index=update_index, round_index=round_index
        )
        # This schedule runs no Reverse Frontier shells; -1 is the "no shell"
        # sentinel the validation and video tooling reads.
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
    skipped: list[str] = []
    for key, value in metrics.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            writer.add_scalar(str(key), scalar, int(step))
        else:
            skipped.append(str(key))
    if skipped:
        # Silently dropping these is how a NaN run masquerades as "TensorBoard
        # stopped logging some curves": the scalars simply stop appearing while
        # the job keeps burning GPU hours.
        warnings.warn(
            "Non-finite metrics not written to TensorBoard at step "
            f"{int(step)}: {sorted(skipped)}",
            RuntimeWarning,
        )
    writer.flush()


def _assert_finite_training_metrics(
    metrics: Mapping[str, Any], *, update_index: int, global_step: int
) -> None:
    """Abort as soon as the rollout goes numerically bad.

    A NaN reward propagates silently: reward_span becomes NaN, ``NaN > 1e-6``
    is False, so every GRPO group is judged non-informative, the loss mask is
    empty and nothing trains -- while the progress bar keeps advancing. Fail
    here instead, so the run stops minutes after the corruption rather than
    hours.
    """

    watched = (
        "candidate_reward_mean",
        "candidate_reward_std",
        "group_pass_rate_mean",
        "vla_lora/grad_norm",
        "vla_lora/kl",
        "vla_lora/ppo_loss",
    )
    broken = {}
    for key in watched:
        if key not in metrics:
            continue
        try:
            scalar = float(metrics[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(scalar):
            broken[key] = scalar
    if broken:
        raise RuntimeError(
            f"Non-finite training metrics at update {int(update_index)} "
            f"(global step {int(global_step)}): {broken}. The rollout or the "
            "policy has diverged; every later update would silently train on "
            "an empty loss mask. Checkpoints from this point are unusable."
        )


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
    reward_counts = torch.zeros_like(counts)
    distance_counts = torch.zeros_like(counts)
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
    instruction_count = len(ACTIVE_INSTRUCTION_TYPES)
    instruction_counts = torch.zeros(
        (instruction_count,), dtype=torch.float64, device=device
    )
    instruction_successes = torch.zeros_like(instruction_counts)
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
        zero = torch.zeros_like(flat_rewards)
        one = torch.ones_like(flat_success)
        # A single diverged episode used to poison the whole mean (plain sums
        # over NaN). Accumulate only finite values, with their own counts, and
        # track how many episodes went bad so instability stays visible instead
        # of erasing the curve.
        finite_rewards = torch.isfinite(flat_rewards)
        finite_distances = torch.isfinite(flat_distances)
        counts.index_add_(0, catalog_ids, one)
        successes.index_add_(0, catalog_ids, flat_success)
        reward_counts.index_add_(0, catalog_ids, finite_rewards.to(counts.dtype))
        distance_counts.index_add_(
            0, catalog_ids, finite_distances.to(counts.dtype)
        )
        rewards.index_add_(
            0, catalog_ids, torch.where(finite_rewards, flat_rewards, zero)
        )
        distances.index_add_(
            0, catalog_ids, torch.where(finite_distances, flat_distances, zero)
        )
        if item.group_instruction_ids is not None:
            instruction_ids = item.group_instruction_ids.repeat_interleave(
                group_size
            ).to(dtype=torch.int64)
            instruction_counts.index_add_(0, instruction_ids, one)
            instruction_successes.index_add_(0, instruction_ids, flat_success)
        environment_actions += float(
            item.metrics.get("validation/environment_actions", 0.0)
        )

    if dist.is_available() and dist.is_initialized():
        for tensor in (
            counts,
            successes,
            rewards,
            distances,
            reward_counts,
            distance_counts,
            instruction_counts,
            instruction_successes,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(environment_actions, op=dist.ReduceOp.SUM)
        if timing_values.numel() > 0:
            dist.all_reduce(timing_values, op=dist.ReduceOp.MAX)

    total_count = counts.sum().clamp_min(1.0)
    finite_reward_total = reward_counts.sum()
    finite_distance_total = distance_counts.sum()
    metrics = {
        "validation/episodes": float(counts.sum().item()),
        "validation/success_rate": float(
            (successes.sum() / total_count).item()
        ),
        # Episodes whose reward/distance diverged. Non-zero means the physics
        # or policy went unstable for those rollouts; success_rate already
        # counts them as failures (NaN comparisons are False).
        "validation/non_finite_reward_episodes": float(
            (counts.sum() - finite_reward_total).item()
        ),
        "validation/non_finite_distance_episodes": float(
            (counts.sum() - finite_distance_total).item()
        ),
        "validation/reward_mean": float(
            (rewards.sum() / finite_reward_total.clamp_min(1.0)).item()
        ),
        "validation/final_xy_distance_mean_m": float(
            (distances.sum() / finite_distance_total.clamp_min(1.0)).item()
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
        finite_rewards_here = max(1.0, float(reward_counts[catalog_index].item()))
        finite_distances_here = max(
            1.0, float(distance_counts[catalog_index].item())
        )
        metrics[f"{prefix}/episodes"] = count
        metrics[f"{prefix}/success_rate"] = float(
            successes[catalog_index].item() / count
        )
        metrics[f"{prefix}/reward_mean"] = float(
            rewards[catalog_index].item() / finite_rewards_here
        )
        metrics[f"{prefix}/final_xy_distance_mean_m"] = float(
            distances[catalog_index].item() / finite_distances_here
        )
    # Per-instruction success. Without this, a run mixing pick_up with the two
    # pre-grasped placement tasks reports one blended number: placement can carry
    # it while pick_up sits at zero for millions of steps unnoticed.
    for instruction_index, instruction_name in enumerate(
        ACTIVE_INSTRUCTION_TYPES
    ):
        episodes = float(instruction_counts[instruction_index].item())
        if episodes <= 0.0:
            continue
        prefix = f"validation/by_instruction/{instruction_name}"
        metrics[f"{prefix}/episodes"] = episodes
        metrics[f"{prefix}/success_rate"] = float(
            instruction_successes[instruction_index].item() / episodes
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


class ApproachDistanceCurriculum:
    """Success-gated cap on the EE start's XY distance to its goal.

    The cap widens only when the policy is actually solving the current
    difficulty -- the training group pass rate (terminal success at the current
    cap) rises above a threshold -- and narrows if it regresses, so exploration
    always starts from a distance the policy can reach. A fixed step schedule
    fails here because it widens on wall-clock regardless of mastery; by the
    time the policy sees a hard start it has had no easy wins to learn from.

    The gate reads the global (cross-rank) pass rate, which is identical on
    every rank, so the cap stays in lockstep without extra collectives. State is
    persisted so a resume continues at the reached difficulty instead of
    restarting from the initial cap. Disabled -> the cap is inf (full-workspace
    starts, byte-for-byte the historical sampling).
    """

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        def number(key: str, default: float) -> float:
            try:
                return float(metadata.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        self.enabled = bool(
            metadata.get(
                "random_workspace_start_distance_curriculum_enabled", False
            )
        )
        self.initial = max(number("random_workspace_start_distance_initial", 0.06), 0.0)
        self.final = max(number("random_workspace_start_distance_final", 0.34), self.initial)
        self.increment = max(number("random_workspace_start_distance_increment", 0.02), 1.0e-6)
        self.promote_threshold = number(
            "random_workspace_start_distance_promote_pass_rate", 0.20
        )
        self.demote_threshold = number(
            "random_workspace_start_distance_demote_pass_rate", 0.05
        )
        self.ema_decay = min(max(number(
            "random_workspace_start_distance_pass_rate_ema_decay", 0.9
        ), 0.0), 0.999)
        self.cooldown_updates = max(
            int(number("random_workspace_start_distance_cooldown_updates", 5)), 1
        )
        self.cap = self.initial
        self.pass_rate_ema = 0.0
        self._cooldown = 0
        self._reseed_ema = False

    def current_cap(self) -> float:
        return self.cap if self.enabled else float("inf")

    def observe(self, pass_rate: float) -> None:
        """Fold in this update's global pass rate; promote/demote when clear."""

        if not self.enabled:
            return
        rate = min(max(float(pass_rate), 0.0), 1.0)
        if self._reseed_ema:
            # First update at a new cap: start the average from what this
            # difficulty actually scores instead of blending it into the old
            # level's value. Carrying it over biases the next decision toward
            # repeating the last one -- after a promotion the EMA still holds
            # the easier level's higher rate, which promotes again, and the
            # curriculum ratchets away from the policy on its own momentum.
            self.pass_rate_ema = rate
            self._reseed_ema = False
        else:
            self.pass_rate_ema = (
                self.ema_decay * self.pass_rate_ema
                + (1.0 - self.ema_decay) * rate
            )
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        changed = False
        if self.pass_rate_ema >= self.promote_threshold and self.cap < self.final:
            self.cap = min(self.final, self.cap + self.increment)
            changed = True
        elif self.pass_rate_ema <= self.demote_threshold and self.cap > self.initial:
            self.cap = max(self.initial, self.cap - self.increment)
            changed = True
        if changed:
            self._cooldown = self.cooldown_updates
            self._reseed_ema = True

    def restart(self) -> bool:
        """Drop the cap back to ``initial`` and forget the pass-rate history.

        Called when something outside this curriculum makes the task harder --
        today, when the scene-object curriculum unlocks a distractor. The cap
        the policy earned was earned on the EASIER scene, so carrying it over
        hands the policy two difficulty increases at once: it must suddenly pick
        the named object out of several AND do it from the far starts it had
        just mastered on a single-object scene. That is the failure this run
        already paid for once (grounding degraded, cosine 0.20 -> 0.05, when the
        second object arrived at 5.7M on a fixed schedule).

        Returns whether anything actually changed, so the caller can log it.
        """

        if not self.enabled or self.cap <= self.initial:
            return False
        self.cap = self.initial
        # Not a re-seed: the next observation must not be treated as "the first
        # sample at a promoted level" either, since the whole history is void.
        self.pass_rate_ema = 0.0
        self._reseed_ema = True
        self._cooldown = self.cooldown_updates
        return True

    def state_dict(self) -> dict[str, float]:
        return {
            "cap": float(self.cap),
            "pass_rate_ema": float(self.pass_rate_ema),
            "cooldown": float(self._cooldown),
            "reseed_ema": float(self._reseed_ema),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        self.cap = min(self.final, max(self.initial, float(state.get("cap", self.cap))))
        self.pass_rate_ema = float(state.get("pass_rate_ema", self.pass_rate_ema))
        self._cooldown = int(float(state.get("cooldown", 0)))
        self._reseed_ema = bool(float(state.get("reseed_ema", 0.0)))


class PerInstructionApproachCurriculum:
    """One success-gated approach curriculum per configured instruction.

    A single shared cap is wrong as soon as the run trains more than one
    instruction: the gate reads the pass rate averaged over every task, so the
    easiest one drags the cap up for all of them. Concretely, put_into_plate
    starts with the object already in the gripper and only has to servo to a
    receptacle, while pick_up has to descend, close, and lift -- mixing them,
    placement successes promote the cap and pick_up loses exactly the close
    starts it needs to get its first grasp. Each instruction now advances only
    on its own successes.

    A run with one instruction behaves identically to the single curriculum.
    """

    def __init__(
        self,
        metadata: Mapping[str, Any],
        *,
        instruction_types: Sequence[str],
    ) -> None:
        names = tuple(instruction_types) or (ACTIVE_INSTRUCTION_TYPES[0],)
        self.instruction_names = names
        self._by_name = {
            name: ApproachDistanceCurriculum(metadata) for name in names
        }

    @property
    def enabled(self) -> bool:
        return any(item.enabled for item in self._by_name.values())

    def caps_by_instruction_id(self) -> dict[int, float]:
        return {
            int(INSTRUCTION_TO_ID[name]): item.current_cap()
            for name, item in self._by_name.items()
        }

    def observe(self, pass_rates: Mapping[str, float]) -> None:
        """Fold in each instruction's own pass rate for this update.

        An instruction absent from ``pass_rates`` collected no groups this
        update (instruction sampling is random per group), so it is skipped
        rather than fed a zero -- a spurious zero would ratchet its EMA down and
        eventually demote a cap the policy had actually earned.
        """

        for name, item in self._by_name.items():
            if name in pass_rates:
                item.observe(float(pass_rates[name]))

    def restart(self) -> tuple[str, ...]:
        """Restart every instruction's cap; returns the ones that moved."""

        return tuple(
            name for name, item in self._by_name.items() if item.restart()
        )

    def metrics(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, item in self._by_name.items():
            cap = item.current_cap()
            # -1 sentinel means the cap is disabled (uncapped start).
            values[f"curriculum/start_max_goal_distance_m/{name}"] = (
                float(cap) if cap != float("inf") else -1.0
            )
            values[f"curriculum/approach_pass_rate_ema/{name}"] = float(
                item.pass_rate_ema
            )
        return values

    def state_dict(self) -> dict[str, Any]:
        return {
            name: item.state_dict() for name, item in self._by_name.items()
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        # Legacy single-curriculum checkpoints stored a flat {cap, pass_rate_ema,
        # cooldown}; replay it into every instruction so a resume keeps the
        # difficulty it had reached.
        if "cap" in state:
            for item in self._by_name.values():
                item.load_state_dict(state)
            return
        for name, item in self._by_name.items():
            entry = state.get(name)
            if isinstance(entry, Mapping):
                item.load_state_dict(entry)


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

    # MJWarp allocates through Warp's own CUDA allocator, separate from
    # PyTorch's caching allocator, and dies with "Warp CUDA error 2: out of
    # memory" if PyTorch's cache has grown to fill the card. Cap PyTorch to a
    # fraction of the device so a fixed slice is always available to Warp for
    # its physics and render buffers. Opt-in and tunable; 0 disables it.
    memory_fraction = float(os.environ.get("RLVLA_TORCH_MEMORY_FRACTION", "0.82"))
    if 0.0 < memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
        _log(
            dist_ctx,
            "[smolvla-mjwarp] PyTorch capped at "
            f"{memory_fraction:.2f} of GPU memory; the remainder is reserved "
            "for the MJWarp/Warp allocator.",
        )

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
        # Vision-aware residual: a frozen fixed-projection of SmolVLA's connector
        # tokens (which DO encode target position -- linear-probe R^2 ~ 0.44)
        # appended to the residual state so the trainable residual can localize
        # the object. 0 disables it (the frozen SmolVLA action head cannot).
        vision_feature_dim = (
            int(getattr(args, "residual_vision_dim", 0))
            if bool(getattr(args, "residual_vision_features", False))
            else 0
        )
        # The frozen SmolVLA replica keeps its native state width; only the
        # trainable residual is widened (target-relative vector + vision feature).
        residual_state_dim = (
            int(args.state_dim)
            + (3 if include_relative_target else 0)
            + vision_feature_dim
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
        warmstart_checkpoint = os.environ.get(
            "RLVLA_SMOLVLA_WARMSTART_CHECKPOINT", ""
        ).strip()
        if warmstart_checkpoint and not args.resume_checkpoint:
            resolved_warmstart = _resolve_checkpoint(warmstart_checkpoint)
            trainer.load_weights_only(resolved_warmstart)
            _log(
                dist_ctx,
                f"[smolvla-mjwarp] warm-started weights from {resolved_warmstart} "
                "(fresh optimizer / curriculum / global step 0)",
            )
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
            vision_feature_dim=vision_feature_dim,
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
                vision_feature_dim=vision_feature_dim,
                profile=bool(args.mjwarp_profile_timers),
            )

        scene_curriculum_steps = _scene_object_curriculum_steps(task_metadata)
        configured_instruction_names = tuple(
            args.instruction_types or (ACTIVE_INSTRUCTION_TYPES[0],)
        )
        approach_curriculum = PerInstructionApproachCurriculum(
            task_metadata, instruction_types=configured_instruction_names
        )
        approach_curriculum.load_state_dict(
            trainer.loaded_extra_state.get("approach_curriculum")
        )
        # Seed from the count this run STARTS at, so a resume past an unlock
        # threshold does not read as a fresh unlock and wipe the earned cap.
        previous_scene_object_max = _apply_scene_object_curriculum(
            resetter,
            curriculum_steps=scene_curriculum_steps,
            global_step=global_step,
        )[1]
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
            # A distractor unlock is an outer-curriculum step, so the inner
            # start-distance curriculum has to start over: the cap was earned on
            # the easier scene, and keeping it would demand target selection AND
            # the far starts at the same moment. Restarting costs the climb but
            # re-manufactures the close starts that make the new grounding
            # problem learnable.
            if scene_object_range[1] > previous_scene_object_max:
                restarted = approach_curriculum.restart()
                if dist_ctx.is_main and restarted:
                    print(
                        "[smolvla-mjwarp] scene objects "
                        f"{previous_scene_object_max} -> "
                        f"{scene_object_range[1]} at step {global_step}; "
                        "restarted approach curriculum for "
                        f"{', '.join(restarted)}",
                        flush=True,
                    )
                previous_scene_object_max = scene_object_range[1]
            start_distance_caps = approach_curriculum.caps_by_instruction_id()
            resetter.set_random_start_max_goal_distance(start_distance_caps)
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
            # Per-instruction success counts. These are plain sums, so the
            # update-boundary all-reduce turns them into global counts and every
            # rank derives the same pass rate -- which keeps the per-instruction
            # approach curricula in lockstep without an extra collective.
            instruction_counts: dict[str, float] = {}
            for name in configured_instruction_names:
                selected = task_ids == int(INSTRUCTION_TO_ID[name])
                worlds_for_name = float(
                    selected.sum().item() * successes.shape[1]
                )
                instruction_counts[
                    f"instruction_successes/{name}"
                ] = float(successes[selected].sum().item())
                instruction_counts[
                    f"instruction_worlds/{name}"
                ] = worlds_for_name
            local_metrics = {
                **rollout_metrics,
                **update_metrics,
                **instruction_counts,
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
            # Success-gate each instruction's approach curriculum on ITS OWN
            # global pass rate, so an easy task cannot promote a hard one's cap.
            # The counts are global sums, so every rank computes the same rates
            # and moves the caps identically. Affects the next update's caps.
            instruction_pass_rates = {}
            for name in configured_instruction_names:
                worlds_for_name = synchronized_metrics.get(
                    f"instruction_worlds/{name}", 0.0
                )
                if worlds_for_name > 0.0:
                    instruction_pass_rates[name] = (
                        synchronized_metrics.get(
                            f"instruction_successes/{name}", 0.0
                        )
                        / worlds_for_name
                    )
            approach_curriculum.observe(instruction_pass_rates)
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
                    # Per-instruction caps and pass-rate EMAs. The unsuffixed
                    # aggregate below stays for dashboards that predate the
                    # split; it is the widest cap in the run.
                    **approach_curriculum.metrics(),
                    # -1 sentinel means the cap is disabled (uncapped start).
                    "curriculum/start_max_goal_distance_m": (
                        float(max(start_distance_caps.values()))
                        if start_distance_caps
                        and max(start_distance_caps.values()) != float("inf")
                        else -1.0
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

            _assert_finite_training_metrics(
                synchronized_metrics,
                update_index=update_index,
                global_step=global_step,
            )

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
                # Validation runs its own rollout, another peak on top of the
                # update just finished; release the cache first so Warp keeps
                # its reserve through it.
                torch.cuda.empty_cache()
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
                        "approach_curriculum": approach_curriculum.state_dict(),
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
