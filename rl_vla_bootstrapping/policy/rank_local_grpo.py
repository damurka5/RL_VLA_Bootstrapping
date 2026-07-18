from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RankLocalGroupLayout:
    """Contiguous, complete GRPO groups owned by exactly one rank."""

    worlds_per_rank: int
    groups_per_rank: int
    group_size: int = 8

    def validate(self) -> None:
        if self.group_size < 2:
            raise ValueError("GRPO group_size must be at least two.")
        if self.groups_per_rank < 1:
            raise ValueError("groups_per_rank must be positive.")
        expected = int(self.groups_per_rank) * int(self.group_size)
        if int(self.worlds_per_rank) != expected:
            raise ValueError(
                "Every rank must own complete groups: "
                f"worlds_per_rank={self.worlds_per_rank}, expected "
                f"{self.groups_per_rank} * {self.group_size} = {expected}."
            )

    @property
    def candidate_indices(self) -> np.ndarray:
        self.validate()
        return np.arange(self.worlds_per_rank, dtype=np.int64).reshape(
            self.groups_per_rank, self.group_size
        )

    @property
    def group_ids(self) -> np.ndarray:
        return np.repeat(
            np.arange(self.groups_per_rank, dtype=np.int64), self.group_size
        )

    @property
    def base_world_indices(self) -> np.ndarray:
        return self.candidate_indices[:, 0]

    def assert_no_cross_rank_group(self, rank: int, world_size: int) -> None:
        self.validate()
        if not 0 <= int(rank) < int(world_size):
            raise ValueError(f"Invalid rank {rank}/{world_size}.")
        # Group identity includes rank.  No group id is meaningful globally.
        keys = {
            (int(rank), int(group_id))
            for group_id in range(self.groups_per_rank)
        }
        if len(keys) != self.groups_per_rank:
            raise AssertionError("Rank-local group keys are not unique.")


def deterministic_group_seeds(
    *,
    base_seed: int,
    rank: int,
    update_index: int,
    groups_per_rank: int,
) -> np.ndarray:
    """Disjoint deterministic streams for reset scenes and policy sampling."""

    if groups_per_rank < 1 or rank < 0 or update_index < 0:
        raise ValueError("rank, update_index, and groups_per_rank must be non-negative.")
    # Large odd strides make collisions impossible for realistic run lengths.
    start = (
        int(base_seed)
        + int(rank) * 1_000_003
        + int(update_index) * 10_000_019
    )
    return start + np.arange(groups_per_rank, dtype=np.int64) * 97


def deterministic_candidate_seeds(
    group_seeds: np.ndarray,
    *,
    group_size: int,
) -> np.ndarray:
    seeds = np.asarray(group_seeds, dtype=np.int64).reshape(-1, 1)
    candidates = np.arange(int(group_size), dtype=np.int64).reshape(1, -1)
    return seeds * 131 + candidates * 17


def numpy_group_advantages(
    outcomes: np.ndarray,
    *,
    normalize: bool = True,
    eps: float = 1.0e-6,
    clip_abs: float | None = None,
) -> np.ndarray:
    values = np.asarray(outcomes, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(
            f"GRPO outcomes must have shape [groups, candidates>=2], got {values.shape}."
        )
    centered = values - values.mean(axis=1, keepdims=True)
    if normalize:
        centered = centered / np.maximum(
            values.std(axis=1, keepdims=True), float(eps)
        )
    if clip_abs is not None and float(clip_abs) > 0:
        centered = np.clip(centered, -float(clip_abs), float(clip_abs))
    return centered.astype(np.float32, copy=False)


def torch_group_advantages(
    outcomes: Any,
    *,
    normalize: bool = True,
    eps: float = 1.0e-6,
    clip_abs: float | None = None,
) -> Any:
    import torch

    if not isinstance(outcomes, torch.Tensor) or outcomes.ndim != 2:
        raise ValueError("Torch GRPO outcomes must have shape [groups, candidates].")
    centered = outcomes - outcomes.mean(dim=1, keepdim=True)
    if normalize:
        centered = centered / outcomes.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(float(eps))
    if clip_abs is not None and float(clip_abs) > 0:
        centered = centered.clamp(-float(clip_abs), float(clip_abs))
    return centered


@dataclass(frozen=True)
class EqualDDPSchedule:
    """Fixed optimizer schedule derived once per update across ranks."""

    records_per_minibatch: int
    ppo_epochs: int
    global_max_records: int

    @property
    def minibatches_per_epoch(self) -> int:
        return max(
            1,
            math.ceil(
                max(1, int(self.global_max_records))
                / max(1, int(self.records_per_minibatch))
            ),
        )

    @property
    def backward_collectives(self) -> int:
        return self.minibatches_per_epoch * max(1, int(self.ppo_epochs))

    @property
    def padded_records_per_rank(self) -> int:
        return self.minibatches_per_epoch * max(1, int(self.records_per_minibatch))


def synchronize_equal_ddp_schedule(
    *,
    local_informative_records: int,
    records_per_minibatch: int,
    ppo_epochs: int,
    device: Any,
) -> EqualDDPSchedule:
    """One update-level MAX collective fixes both ranks' backward count."""

    import torch
    import torch.distributed as dist

    count = torch.tensor(
        [max(0, int(local_informative_records))],
        dtype=torch.int64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
    return EqualDDPSchedule(
        records_per_minibatch=max(1, int(records_per_minibatch)),
        ppo_epochs=max(1, int(ppo_epochs)),
        global_max_records=int(count.item()),
    )


def pad_tensor_records(
    records: dict[str, Any],
    *,
    target_records: int,
) -> tuple[dict[str, Any], Any]:
    """Pad fixed-shape update tensors; padded rows receive a zero loss mask."""

    import torch

    if not records:
        raise ValueError("At least one record tensor is required.")
    lengths = {int(value.shape[0]) for value in records.values()}
    if len(lengths) != 1:
        raise ValueError(f"Record tensor lengths differ: {sorted(lengths)}.")
    current = lengths.pop()
    target = max(int(target_records), current)
    mask = torch.zeros(
        (target,), dtype=torch.float32, device=next(iter(records.values())).device
    )
    mask[:current] = 1.0
    padded: dict[str, Any] = {}
    for key, value in records.items():
        if target == current:
            padded[key] = value
            continue
        shape = (target - current, *value.shape[1:])
        filler = torch.zeros(shape, dtype=value.dtype, device=value.device)
        padded[key] = torch.cat((value, filler), dim=0)
    return padded, mask


def aggregated_global_step(
    *,
    prior_global_step: int,
    local_selected_environment_actions: int,
    world_size: int,
) -> int:
    """Definition used in logs/checkpoints: work aggregated across ranks."""

    if min(prior_global_step, local_selected_environment_actions) < 0:
        raise ValueError("Step counts must be non-negative.")
    if world_size < 1:
        raise ValueError("world_size must be positive.")
    return int(prior_global_step) + int(local_selected_environment_actions) * int(
        world_size
    )
