#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception as exc:  # pragma: no cover - optional local dependency
    _TORCH_IMPORT_ERROR = exc
    torch = None
    F = None
    nn = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from rl_vla_bootstrapping.policy.octo_cdpr_adapter import (
    CDPROctoObservationAdapter,
    CDPRStateLayout,
    DEFAULT_OCTO_SMALL_CHECKPOINT,
    OctoActionAdapterSpec,
    OctoObservationSpec,
    adapt_octo_actions_to_cdpr,
    load_octo_runtime,
)


def _bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}",
        dest=name,
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _float_pair(values: Sequence[str]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two float values.")
    return float(values[0]), float(values[1])


def _default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        detail = f" Original import error: {_TORCH_IMPORT_ERROR!r}" if _TORCH_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "Octo CDPR adapter training requires PyTorch. Install it in the remote `octo` "
            f"environment before executing the RL stage.{detail}"
        )


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False
    device: str = "cpu"

    @property
    def is_main(self) -> bool:
        return int(self.rank) == 0


_MOVE_TO_OBJECT_INSTRUCTIONS = frozenset({"move_to_object"})
_NEAR_GRIPPER_START_INSTRUCTIONS = frozenset({"catch_object", "grip_object", "grab_object"})
_CAUGHT_OBJECT_START_INSTRUCTIONS = frozenset(
    {
        "release_object",
        "free_object",
        "pick_up",
        "put_into_plate",
        "put_into_bowl",
        "move_object_left",
        "move_object_right",
        "move_object_up",
        "move_object_down",
        "move_left_of_object",
        "move_right_of_object",
        "move_in_front_of_object",
        "move_behind_object",
        "put_in_front_of_object",
        "put_behind_object",
        "move_between_objects",
    }
)


class AdaptiveInstructionSampler:
    def __init__(self, args: argparse.Namespace, *, seed: int) -> None:
        self.args = args
        self.rng = np.random.default_rng(int(seed))
        self.history_size = max(1, int(getattr(args, "mixed_curriculum_history_episodes", 100)))
        self.history: dict[str, deque[float]] = {}
        self.recent_instructions: deque[str] = deque(maxlen=self.history_size)
        self.total_episode_counts: dict[str, int] = {}
        self.last_probabilities: dict[str, float] = {}

    def record(self, instruction_type: str | None, success: bool) -> None:
        name = str(instruction_type or "").strip()
        if not name:
            return
        if name not in self.history:
            self.history[name] = deque(maxlen=self.history_size)
        self.history[name].append(1.0 if bool(success) else 0.0)
        self.recent_instructions.append(name)
        self.total_episode_counts[name] = int(self.total_episode_counts.get(name, 0)) + 1

    def reset_options(self, stage_index: int) -> dict[str, Any]:
        if not _mixed_curriculum_enabled(self.args) or int(stage_index) < 2:
            self.last_probabilities = {}
            return {}

        candidates = list(_active_dense_stage_instruction_types(self.args, stage_index))
        if not candidates:
            self.last_probabilities = {}
            return {}

        probabilities = self._probabilities(candidates)
        selected = str(candidates[int(self.rng.choice(len(candidates), p=probabilities))])
        self.last_probabilities = {
            str(name): float(probabilities[index])
            for index, name in enumerate(candidates)
        }
        options: dict[str, Any] = {"instruction_type": selected}
        if selected in set(_dedupe_instruction_names(getattr(self.args, "near_success_instruction_types", None))):
            if selected in _NEAR_GRIPPER_START_INSTRUCTIONS:
                options["start_with_target_at_gripper"] = True
            if selected in _CAUGHT_OBJECT_START_INSTRUCTIONS:
                options["start_with_caught_object"] = True
        return options

    def snapshot(self) -> dict[str, Any]:
        rates: dict[str, float | None] = {}
        counts: dict[str, int] = {}
        for name, values in self.history.items():
            counts[name] = int(len(values))
            rates[name] = float(sum(values) / len(values)) if values else None
        recent_counts: dict[str, int] = {}
        for name in self.recent_instructions:
            recent_counts[name] = int(recent_counts.get(name, 0)) + 1
        recent_total = max(1, sum(recent_counts.values()))
        recent_fractions = {
            name: float(count / recent_total)
            for name, count in sorted(recent_counts.items())
        }
        return {
            "mixed_curriculum_enabled": bool(_mixed_curriculum_enabled(self.args)),
            "mixed_curriculum_last_probabilities": dict(self.last_probabilities),
            "mixed_curriculum_success_rates": rates,
            "mixed_curriculum_episode_counts": counts,
            "mixed_curriculum_recent_episode_fractions": recent_fractions,
            "mixed_curriculum_total_episode_counts": dict(sorted(self.total_episode_counts.items())),
        }

    def _probabilities(self, candidates: list[str]) -> np.ndarray:
        if not bool(getattr(self.args, "mixed_curriculum_adaptive", True)):
            probs = np.ones((len(candidates),), dtype=np.float64)
        else:
            target = float(np.clip(getattr(self.args, "mixed_curriculum_success_target", 0.80), 0.0, 1.0))
            min_gap = max(1e-3, float(getattr(self.args, "mixed_curriculum_min_gap_weight", 0.05)))
            probs = np.zeros((len(candidates),), dtype=np.float64)
            for index, name in enumerate(candidates):
                history = self.history.get(str(name))
                if history:
                    success_rate = float(sum(history) / len(history))
                    uncertainty = 1.0 / np.sqrt(float(len(history)) + 1.0)
                    probs[index] = max(min_gap, target - success_rate) + 0.10 * uncertainty
                else:
                    probs[index] = max(min_gap, target)

        probs = self._normalize(probs)
        stage_one = set(_dense_stage_instruction_types(self.args, 1))
        manipulation = set(candidates) - stage_one - _MOVE_TO_OBJECT_INSTRUCTIONS
        floors = [
            (stage_one, float(getattr(self.args, "mixed_curriculum_rehearsal_min_prob", 0.0))),
            (_MOVE_TO_OBJECT_INSTRUCTIONS, float(getattr(self.args, "mixed_curriculum_move_to_object_min_prob", 0.0))),
            (manipulation, float(getattr(self.args, "mixed_curriculum_manipulation_min_prob", 0.0))),
        ]
        floor_sum = sum(max(0.0, floor) for _, floor in floors)
        scale = 1.0 / floor_sum if floor_sum > 0.95 else 1.0
        for members, floor in floors:
            probs = self._apply_group_floor(candidates, probs, members, max(0.0, floor) * scale)
        probs = self._apply_min_max_bounds(probs)
        return self._normalize(probs)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        total = float(np.sum(arr))
        if arr.size == 0:
            return arr
        if total <= 1e-12 or not np.isfinite(total):
            return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
        return arr / total

    def _apply_group_floor(
        self,
        candidates: list[str],
        probs: np.ndarray,
        members: set[str] | frozenset[str],
        floor: float,
    ) -> np.ndarray:
        floor = float(np.clip(floor, 0.0, 0.95))
        if floor <= 0.0:
            return probs
        mask = np.asarray([name in members for name in candidates], dtype=bool)
        if not bool(np.any(mask)):
            return probs
        current = float(np.sum(probs[mask]))
        if current >= floor:
            return probs
        out = probs.copy()
        other = ~mask
        other_total = float(np.sum(out[other]))
        if current <= 1e-12:
            out[mask] = floor / float(np.sum(mask))
        else:
            out[mask] *= floor / current
        if other_total > 1e-12:
            out[other] *= max(0.0, 1.0 - floor) / other_total
        return self._normalize(out)

    def _apply_min_max_bounds(self, probs: np.ndarray) -> np.ndarray:
        count = int(probs.size)
        if count <= 1:
            return self._normalize(probs)
        min_prob = max(0.0, float(getattr(self.args, "mixed_curriculum_min_prob", 0.0)))
        max_prob = max(0.0, float(getattr(self.args, "mixed_curriculum_max_prob", 1.0)))
        effective_min = min(min_prob, 0.95 / float(count))
        effective_max = max(max_prob, 1.0 / float(count))
        out = self._normalize(probs)
        for _ in range(4):
            low = out < effective_min
            if bool(np.any(low)):
                missing = float(np.sum(effective_min - out[low]))
                out[low] = effective_min
                high = ~low
                high_total = float(np.sum(out[high]))
                if high_total > 1e-12:
                    out[high] *= max(0.0, 1.0 - float(np.sum(out[low]))) / high_total
            over = out > effective_max
            if bool(np.any(over)):
                excess = float(np.sum(out[over] - effective_max))
                out[over] = effective_max
                under = ~over
                under_total = float(np.sum(out[under]))
                if under_total > 1e-12:
                    out[under] += excess * out[under] / under_total
            out = self._normalize(out)
        return out


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _configure_distributed(args: argparse.Namespace) -> DistributedContext:
    _require_torch()
    world_size = max(1, _env_int("WORLD_SIZE", 1))
    rank = max(0, _env_int("RANK", 0))
    local_rank = max(0, _env_int("LOCAL_RANK", 0))
    enabled = bool(args.distributed and world_size > 1)

    requested_device = str(args.device)
    cuda_requested = requested_device.startswith("cuda")
    device = requested_device
    if torch.cuda.is_available() and cuda_requested:
        if enabled:
            local_rank = local_rank % max(1, torch.cuda.device_count())
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
            if bool(args.bind_jax_to_local_rank):
                os.environ["JAX_VISIBLE_DEVICES"] = str(local_rank)
        elif requested_device == "cuda":
            device = "cuda:0"
            torch.cuda.set_device(0)
    args.device = device

    if enabled:
        import torch.distributed as dist

        if not dist.is_initialized():
            backend = str(args.distributed_backend)
            if backend == "nccl" and not torch.cuda.is_available():
                backend = "gloo"
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        enabled=enabled,
        device=device,
    )


def _destroy_distributed(ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _rank_seed(seed: int, ctx: DistributedContext) -> int:
    return int(seed) + int(ctx.rank) * 100_003


def _set_quiet_env(args: argparse.Namespace, ctx: DistributedContext) -> None:
    if bool(args.progress_only) or not ctx.is_main:
        os.environ["RLVLA_CDPR_QUIET"] = "1"
        os.environ["RLVLA_CDPR_WRAPPER_LOG"] = "0"


@contextlib.contextmanager
def _silence_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _log(ctx: DistributedContext, message: str, *, progress: Any | None = None) -> None:
    if not ctx.is_main:
        return
    if progress is not None:
        progress.write(message)
    else:
        print(message, flush=True)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(float(seconds)) or float(seconds) < 0.0:
        return "unknown"
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _format_step_progress(
    *,
    global_step: int,
    max_train_steps: int,
    start_step: int,
    elapsed_seconds: float,
) -> str:
    total = max(1, int(max_train_steps))
    current = max(0, int(global_step))
    run_total = max(0, total - int(start_step))
    completed_since_start = max(0, current - int(start_step))
    elapsed = max(1e-9, float(elapsed_seconds))
    rate = float(completed_since_start) / elapsed
    remaining = max(0, total - current)
    eta = None if rate <= 1e-9 else float(remaining) / rate
    pct = 100.0 * min(1.0, float(current) / float(total))
    return (
        f"progress={current}/{total} ({pct:.1f}%) "
        f"run={completed_since_start}/{run_total} "
        f"rate={rate:.2f} step/s eta={_format_duration(eta)}"
    )


def _make_progress_bar(*, args: argparse.Namespace, ctx: DistributedContext, start_step: int) -> Any | None:
    if not bool(args.progress) or not ctx.is_main:
        return None
    if not sys.__stderr__.isatty():
        # Remote launchers pipe through tee; tqdm carriage-return redraws become noisy log text there.
        return None
    if tqdm is None:
        if not bool(args.progress_only):
            print("[octo-cdpr] tqdm is unavailable; falling back to status prints.", flush=True)
        return None
    total = max(0, int(args.max_train_steps) - int(start_step))
    return tqdm(
        total=total,
        initial=0,
        desc="octo-cdpr",
        unit="step",
        dynamic_ncols=True,
        mininterval=float(args.progress_refresh_seconds),
        maxinterval=max(float(args.progress_refresh_seconds) * 2.0, 10.0),
        miniters=max(1, int(args.status_every_steps)),
        leave=True,
        file=sys.__stderr__,
    )


def _progress_postfix(
    progress: Any | None,
    *,
    episode: int,
    episode_length: int,
    episode_reward: float,
    reward: float,
    buffer_size: int,
    instruction: str,
    world_size: int,
) -> None:
    if progress is None:
        return
    progress.set_postfix(
        ep=int(episode),
        ep_len=int(episode_length),
        ep_reward=f"{float(episode_reward):+.3f}",
        reward=f"{float(reward):+.3f}",
        buffer=int(buffer_size),
        gpus=int(world_size),
        instr=str(instruction)[:24],
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight CDPR residual/readout adapter around frozen pretrained Octo-Small."
        )
    )
    parser.add_argument("--config", default=None, help="Optional project config path for manifest provenance.")
    parser.add_argument("--base-checkpoint", default=DEFAULT_OCTO_SMALL_CHECKPOINT)
    parser.add_argument("--run-root-dir", default="runs")
    parser.add_argument("--run-id", default="octo_cdpr_rl")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=_default_device())
    _bool_arg(parser, "distributed", default=True, help_text="Enable torch.distributed when launched by torchrun.")
    parser.add_argument("--distributed-backend", default="nccl")
    _bool_arg(
        parser,
        "bind_jax_to_local_rank",
        default=True,
        help_text="When distributed, set JAX_VISIBLE_DEVICES to the torchrun local rank before importing JAX.",
    )
    _bool_arg(parser, "progress", default=True, help_text="Show a tqdm training progress bar on rank 0.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=10.0)
    _bool_arg(
        parser,
        "progress_only",
        default=False,
        help_text="Suppress CDPR wrapper/simulator chatter and leave only the progress bar plus critical trainer logs.",
    )

    parser.add_argument("--catalog-path", default=None)
    parser.add_argument("--cdpr-dataset-root", default=None)
    parser.add_argument("--cdpr-mujoco-root", default=None)
    parser.add_argument("--desk-textures-dir", default=None)
    parser.add_argument("--desk-geom-regex", default=r"(table|desk|workbench|counter|surface)")
    parser.add_argument("--desk-texrepeat", nargs=2, type=int, default=(20, 20))
    parser.add_argument("--allowed-objects", nargs="+", default=None)
    parser.add_argument("--instruction-types", nargs="+", default=None)
    parser.add_argument("--max-objects", type=int, default=4)

    parser.add_argument("--max-env-steps", type=int, default=32)
    parser.add_argument("--max-train-steps", type=int, default=120000)
    parser.add_argument("--action-step-xyz", type=float, default=0.015)
    parser.add_argument("--action-step-yaw", type=float, default=0.08)
    parser.add_argument("--action-step-gripper", type=float, default=0.05)
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument("--move-distance", type=float, default=0.40)
    parser.add_argument("--lift-distance", type=float, default=0.10)
    _bool_arg(parser, "lock_non_commanded_axes", default=True, help_text="Forwarded to the CDPR env.")
    parser.add_argument("--lock-non-commanded-axes-threshold", type=float, default=0.05)
    _bool_arg(parser, "randomize_ee_start", default=True, help_text="Forwarded to the CDPR env.")
    parser.add_argument("--ee-start-x-bounds", nargs=2, type=float, default=(-0.25, 0.25))
    parser.add_argument("--ee-start-y-bounds", nargs=2, type=float, default=(-0.25, 0.25))
    parser.add_argument("--ee-start-z", type=float, default=None)
    _bool_arg(parser, "randomize_ee_yaw", default=True, help_text="Forwarded to the CDPR env.")
    parser.add_argument("--ee-yaw-bounds", nargs=2, type=float, default=(-3.141592653589793, 3.141592653589793))
    _bool_arg(parser, "capture_frames", default=False, help_text="Forwarded to the CDPR env.")
    _bool_arg(parser, "wrapper_cleanup", default=False, help_text="Forwarded to the CDPR env.")
    _bool_arg(parser, "use_wrapper_cache", default=True, help_text="Forwarded to the CDPR env.")
    _bool_arg(
        parser,
        "reuse_existing_wrapper_variants",
        default=True,
        help_text="Prefer existing compatible wrapper variants.",
    )

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--history", type=int, default=1)
    _bool_arg(parser, "include_wrist", default=True, help_text="Include the CDPR wrist camera in Octo observations.")
    _bool_arg(parser, "include_proprio", default=True, help_text="Include 5D CDPR proprio in Octo observations.")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--replan-every", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--octo-action-indices", nargs=5, type=int, default=None)
    parser.add_argument("--octo-action-normalization", choices=("tanh", "clip", "none"), default="tanh")
    _bool_arg(
        parser,
        "use_dataset_action_unnorm",
        default=False,
        help_text="Pass Octo dataset action statistics to sample_actions before CDPR normalization.",
    )

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--actor-lr", type=float, default=3.0e-4)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--update-after", type=int, default=512)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--exploration-noise", type=float, default=0.15)
    parser.add_argument("--noise-decay-steps", type=int, default=60000)
    parser.add_argument("--min-exploration-noise", type=float, default=0.03)
    parser.add_argument("--action-l2", type=float, default=1.0e-3)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--status-every-steps", type=int, default=250)
    parser.add_argument(
        "--metrics-window-episodes",
        type=int,
        default=100,
        help="Number of completed episodes used for rolling reward/success TensorBoard metrics.",
    )
    parser.add_argument(
        "--dense-stage-one-instruction-types",
        nargs="+",
        default=None,
        help="Optional first-stage dense curriculum instruction types.",
    )
    parser.add_argument(
        "--dense-stage-two-instruction-types",
        nargs="+",
        default=None,
        help="Optional second-stage dense curriculum instruction types.",
    )
    parser.add_argument(
        "--dense-stage-switch-success-rate",
        type=float,
        default=0.50,
        help="Rolling mean success rate required to switch from dense stage 1 to stage 2.",
    )
    parser.add_argument(
        "--dense-stage-min-episodes",
        type=int,
        default=0,
        help="Minimum completed stage-1 episodes before the success-rate switch can fire.",
    )
    parser.add_argument(
        "--dense-stage-gate-metric",
        choices=("rollout", "validation"),
        default="rollout",
        help="Metric used to promote dense stage 1 to stage 2.",
    )
    parser.add_argument(
        "--dense-stage-gate-aggregation",
        choices=("mean", "worst"),
        default="mean",
        help="Aggregate used for validation-gated promotion. `worst` uses the lowest per-instruction success rate.",
    )
    _bool_arg(
        parser,
        "mixed_curriculum_enabled",
        default=False,
        help_text="After stage promotion, sample a mixed stage with rehearsal, object navigation, and near-success manipulation.",
    )
    _bool_arg(
        parser,
        "mixed_curriculum_adaptive",
        default=True,
        help_text="Oversample instructions whose recent episode success is below the configured target.",
    )
    parser.add_argument("--mixed-curriculum-success-target", type=float, default=0.80)
    parser.add_argument("--mixed-curriculum-history-episodes", type=int, default=100)
    parser.add_argument("--mixed-curriculum-min-gap-weight", type=float, default=0.05)
    parser.add_argument("--mixed-curriculum-min-prob", type=float, default=0.03)
    parser.add_argument("--mixed-curriculum-max-prob", type=float, default=0.35)
    parser.add_argument("--mixed-curriculum-rehearsal-min-prob", type=float, default=0.30)
    parser.add_argument("--mixed-curriculum-move-to-object-min-prob", type=float, default=0.25)
    parser.add_argument("--mixed-curriculum-manipulation-min-prob", type=float, default=0.25)
    parser.add_argument(
        "--near-success-instruction-types",
        nargs="+",
        default=None,
        help="Instruction types that should be reset near success during the mixed stage.",
    )
    parser.add_argument(
        "--validation-every-steps",
        type=int,
        default=0,
        help="Run a distinct held-out validation rollout every N env steps. Disabled when <= 0.",
    )
    parser.add_argument(
        "--validation-episodes-per-instruction",
        type=int,
        default=0,
        help="Held-out validation episodes per active instruction. Disabled when <= 0.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=1000000,
        help="Base seed for deterministic held-out validation episodes.",
    )
    parser.add_argument(
        "--validation-video-count",
        type=int,
        default=3,
        help="Number of overview MP4 validation episodes to save per held-out validation run.",
    )
    parser.add_argument("--validation-video-fps", type=float, default=10.0)
    return parser.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    _require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _make_run_dir(root: str | Path, run_id: str) -> Path:
    path = Path(root).expanduser().resolve() / str(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _build_env(args: argparse.Namespace, *, seed: int):
    from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv

    return CDPRLanguageRLEnv(
        catalog_path=args.catalog_path,
        max_steps=args.max_env_steps,
        max_objects=args.max_objects,
        action_step_xyz=args.action_step_xyz,
        action_step_yaw=args.action_step_yaw,
        action_step_gripper=args.action_step_gripper,
        hold_steps=args.hold_steps,
        lock_non_commanded_axes=args.lock_non_commanded_axes,
        lock_non_commanded_axes_threshold=args.lock_non_commanded_axes_threshold,
        randomize_ee_start=args.randomize_ee_start,
        ee_start_x_bounds=args.ee_start_x_bounds,
        ee_start_y_bounds=args.ee_start_y_bounds,
        ee_start_z=args.ee_start_z,
        randomize_ee_yaw=args.randomize_ee_yaw,
        ee_yaw_bounds=args.ee_yaw_bounds,
        move_distance=args.move_distance,
        lift_distance=args.lift_distance,
        capture_frames=args.capture_frames,
        record_trajectory=args.capture_frames,
        instruction_types=args.instruction_types,
        allowed_objects=args.allowed_objects,
        desk_textures_dir=args.desk_textures_dir,
        desk_geom_regex=args.desk_geom_regex,
        desk_texrepeat=args.desk_texrepeat,
        wrapper_cleanup=args.wrapper_cleanup,
        use_wrapper_cache=args.use_wrapper_cache,
        reuse_existing_wrapper_variants=args.reuse_existing_wrapper_variants,
        seed=int(seed),
    )


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, chunk_size: int, action_dim: int):
        self.capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.prior = np.zeros((capacity, chunk_size, action_dim), dtype=np.float32)
        self.action_index = np.zeros((capacity, 1), dtype=np.int64)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_prior = np.zeros((capacity, chunk_size, action_dim), dtype=np.float32)
        self.next_action_index = np.zeros((capacity, 1), dtype=np.int64)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        *,
        state: np.ndarray,
        prior: np.ndarray,
        action_index: int,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_prior: np.ndarray,
        next_action_index: int,
        done: bool,
    ) -> None:
        self.state[self.ptr] = state
        self.prior[self.ptr] = prior
        self.action_index[self.ptr, 0] = int(action_index)
        self.action[self.ptr] = action
        self.reward[self.ptr, 0] = float(reward)
        self.next_state[self.ptr] = next_state
        self.next_prior[self.ptr] = next_prior
        self.next_action_index[self.ptr, 0] = int(next_action_index)
        self.done[self.ptr, 0] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "state": torch.as_tensor(self.state[idx], device=device),
            "prior": torch.as_tensor(self.prior[idx], device=device),
            "action_index": torch.as_tensor(self.action_index[idx], device=device),
            "action": torch.as_tensor(self.action[idx], device=device),
            "reward": torch.as_tensor(self.reward[idx], device=device),
            "next_state": torch.as_tensor(self.next_state[idx], device=device),
            "next_prior": torch.as_tensor(self.next_prior[idx], device=device),
            "next_action_index": torch.as_tensor(self.next_action_index[idx], device=device),
            "done": torch.as_tensor(self.done[idx], device=device),
        }


if nn is not None:
    class MLP(nn.Module):
        def __init__(self, dims: Sequence[int]):
            super().__init__()
            layers: list[nn.Module] = []
            for index in range(len(dims) - 1):
                layers.append(nn.Linear(int(dims[index]), int(dims[index + 1])))
                if index < len(dims) - 2:
                    layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class ResidualChunkActor(nn.Module):
        def __init__(
            self,
            *,
            state_dim: int,
            chunk_size: int,
            action_dim: int,
            hidden_dim: int,
            residual_scale: float,
        ) -> None:
            super().__init__()
            self.chunk_size = int(chunk_size)
            self.action_dim = int(action_dim)
            self.residual_scale = float(residual_scale)
            input_dim = int(state_dim) + int(chunk_size) * int(action_dim)
            output_dim = int(chunk_size) * int(action_dim)
            self.net = MLP((input_dim, hidden_dim, hidden_dim, output_dim))

        def forward(self, state: torch.Tensor, prior_chunk: torch.Tensor) -> torch.Tensor:
            prior = prior_chunk.reshape(prior_chunk.shape[0], self.chunk_size, self.action_dim)
            features = torch.cat([state, prior.reshape(prior.shape[0], -1)], dim=-1)
            residual = torch.tanh(self.net(features)).reshape_as(prior)
            return torch.tanh(prior + self.residual_scale * residual)

        def action_at(
            self,
            state: torch.Tensor,
            prior_chunk: torch.Tensor,
            action_index: torch.Tensor,
        ) -> torch.Tensor:
            chunk = self.forward(state, prior_chunk)
            idx = action_index.reshape(-1).long().clamp(0, self.chunk_size - 1)
            return chunk[torch.arange(chunk.shape[0], device=chunk.device), idx]


    class QNetwork(nn.Module):
        def __init__(self, *, state_dim: int, action_dim: int, hidden_dim: int):
            super().__init__()
            self.net = MLP((int(state_dim) + int(action_dim), hidden_dim, hidden_dim, 1))

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([state, action], dim=-1))
else:
    class ResidualChunkActor:  # pragma: no cover - dependency guard
        def __init__(self, *args, **kwargs):
            _require_torch()


    class QNetwork:  # pragma: no cover - dependency guard
        def __init__(self, *args, **kwargs):
            _require_torch()


class ResidualTrainer:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        run_dir: Path,
        device: torch.device,
        distributed: DistributedContext | None = None,
    ) -> None:
        self.args = args
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.run_dir = run_dir
        self.device = device
        self.distributed = distributed or DistributedContext(device=str(device))

        actor = ResidualChunkActor(
            state_dim=state_dim,
            chunk_size=chunk_size,
            action_dim=action_dim,
            hidden_dim=int(args.hidden_dim),
            residual_scale=float(args.residual_scale),
        ).to(device)

        critic1 = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=int(args.hidden_dim)).to(device)
        critic2 = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=int(args.hidden_dim)).to(device)

        if self.distributed.enabled:
            from torch.nn.parallel import DistributedDataParallel as DDP

            ddp_kwargs: dict[str, Any] = {}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [int(device.index or 0)]
                ddp_kwargs["output_device"] = int(device.index or 0)
            actor = DDP(actor, **ddp_kwargs)
            critic1 = DDP(critic1, **ddp_kwargs)
            critic2 = DDP(critic2, **ddp_kwargs)

        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2

        self.actor_target = ResidualChunkActor(
            state_dim=state_dim,
            chunk_size=chunk_size,
            action_dim=action_dim,
            hidden_dim=int(args.hidden_dim),
            residual_scale=float(args.residual_scale),
        ).to(device)
        self.actor_target.load_state_dict(self._unwrap(self.actor).state_dict())
        self._set_requires_grad(self.actor_target, False)

        self.critic1_target = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=int(args.hidden_dim)).to(device)
        self.critic2_target = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=int(args.hidden_dim)).to(device)
        self.critic1_target.load_state_dict(self._unwrap(self.critic1).state_dict())
        self.critic2_target.load_state_dict(self._unwrap(self.critic2).state_dict())
        self._set_requires_grad(self.critic1_target, False)
        self._set_requires_grad(self.critic2_target, False)

        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=float(args.actor_lr))
        self.critic_optim = torch.optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=float(args.critic_lr),
        )
        self.gradient_step = 0

    @staticmethod
    def _unwrap(module: nn.Module) -> nn.Module:
        return module.module if hasattr(module, "module") else module

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        base = ResidualTrainer._unwrap(module)
        for param in base.parameters():
            param.requires_grad_(bool(enabled))

    def select_chunk(self, state: np.ndarray, prior_chunk: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        prior_t = torch.as_tensor(prior_chunk, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_chunk = self.actor(state_t, prior_t)[0].cpu().numpy()
        return np.clip(action_chunk, -1.0, 1.0).astype(np.float32, copy=False)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        with torch.no_grad():
            next_action = self.actor_target.action_at(
                batch["next_state"],
                batch["next_prior"],
                batch["next_action_index"],
            )
            target_q = torch.min(
                self.critic1_target(batch["next_state"], next_action),
                self.critic2_target(batch["next_state"], next_action),
            )
            target = batch["reward"] + (1.0 - batch["done"]) * float(self.args.gamma) * target_q

        q1 = self.critic1(batch["state"], batch["action"])
        q2 = self.critic2(batch["state"], batch["action"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()

        self._set_requires_grad(self.critic1, False)
        try:
            pred_chunk = self.actor(batch["state"], batch["prior"])
            idx = batch["action_index"].reshape(-1).long().clamp(0, self.chunk_size - 1)
            pred_action = pred_chunk[torch.arange(pred_chunk.shape[0], device=pred_chunk.device), idx]
            actor_loss = -self._unwrap(self.critic1)(batch["state"], pred_action).mean()
            if float(self.args.action_l2) > 0.0:
                actor_loss = actor_loss + float(self.args.action_l2) * pred_action.pow(2).mean()
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optim.step()
        finally:
            self._set_requires_grad(self.critic1, True)

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        self.gradient_step += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q1": float(q1.mean().item()),
            "target_q": float(target.mean().item()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = float(self.args.tau)
        for src, dst in zip(source.parameters(), target.parameters()):
            dst.data.mul_(1.0 - tau).add_(tau * src.data)

    @staticmethod
    def _adapt_input_layer_for_state_dim(
        checkpoint_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        *,
        old_state_dim: int,
        new_state_dim: int,
        old_tail_dim: int,
        new_tail_dim: int,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        key = "net.net.0.weight"
        if key not in checkpoint_state or key not in target_state:
            return checkpoint_state, False

        old_weight = checkpoint_state[key]
        new_weight = target_state[key]
        if tuple(old_weight.shape) == tuple(new_weight.shape):
            return checkpoint_state, False
        if old_weight.ndim != 2 or new_weight.ndim != 2:
            return checkpoint_state, False

        adapted_state = dict(checkpoint_state)
        adapted_weight = torch.zeros_like(new_weight)
        rows = min(int(old_weight.shape[0]), int(new_weight.shape[0]))

        state_cols = min(int(old_state_dim), int(new_state_dim), int(old_weight.shape[1]), int(new_weight.shape[1]))
        if rows > 0 and state_cols > 0:
            adapted_weight[:rows, :state_cols] = old_weight[:rows, :state_cols].to(adapted_weight.device)

        old_tail_start = int(old_state_dim)
        new_tail_start = int(new_state_dim)
        tail_cols = min(
            int(old_tail_dim),
            int(new_tail_dim),
            max(0, int(old_weight.shape[1]) - old_tail_start),
            max(0, int(new_weight.shape[1]) - new_tail_start),
        )
        if rows > 0 and tail_cols > 0:
            adapted_weight[:rows, new_tail_start : new_tail_start + tail_cols] = old_weight[
                :rows,
                old_tail_start : old_tail_start + tail_cols,
            ].to(adapted_weight.device)

        adapted_state[key] = adapted_weight
        return adapted_state, True

    def load(self, checkpoint_path: Path) -> int:
        payload = torch.load(checkpoint_path, map_location=self.device)
        old_state_dim = int(payload.get("state_dim", self.state_dim))
        old_action_dim = int(payload.get("action_dim", self.action_dim))
        old_chunk_size = int(payload.get("chunk_size", self.chunk_size))
        state_dim_changed = bool(old_state_dim != int(self.state_dim))

        actor_old_tail = int(old_chunk_size * old_action_dim)
        actor_new_tail = int(self.chunk_size * self.action_dim)
        critic_old_tail = int(old_action_dim)
        critic_new_tail = int(self.action_dim)

        actor_state, actor_adapted = self._adapt_input_layer_for_state_dim(
            dict(payload["actor"]),
            self._unwrap(self.actor).state_dict(),
            old_state_dim=old_state_dim,
            new_state_dim=int(self.state_dim),
            old_tail_dim=actor_old_tail,
            new_tail_dim=actor_new_tail,
        )
        actor_target_state, actor_target_adapted = self._adapt_input_layer_for_state_dim(
            dict(payload.get("actor_target", payload["actor"])),
            self.actor_target.state_dict(),
            old_state_dim=old_state_dim,
            new_state_dim=int(self.state_dim),
            old_tail_dim=actor_old_tail,
            new_tail_dim=actor_new_tail,
        )
        critic1_state, critic1_adapted = self._adapt_input_layer_for_state_dim(
            dict(payload["critic1"]),
            self._unwrap(self.critic1).state_dict(),
            old_state_dim=old_state_dim,
            new_state_dim=int(self.state_dim),
            old_tail_dim=critic_old_tail,
            new_tail_dim=critic_new_tail,
        )
        critic2_state, critic2_adapted = self._adapt_input_layer_for_state_dim(
            dict(payload["critic2"]),
            self._unwrap(self.critic2).state_dict(),
            old_state_dim=old_state_dim,
            new_state_dim=int(self.state_dim),
            old_tail_dim=critic_old_tail,
            new_tail_dim=critic_new_tail,
        )
        critic1_target_state, critic1_target_adapted = self._adapt_input_layer_for_state_dim(
            dict(payload.get("critic1_target", payload["critic1"])),
            self.critic1_target.state_dict(),
            old_state_dim=old_state_dim,
            new_state_dim=int(self.state_dim),
            old_tail_dim=critic_old_tail,
            new_tail_dim=critic_new_tail,
        )
        critic2_target_state, critic2_target_adapted = self._adapt_input_layer_for_state_dim(
            dict(payload.get("critic2_target", payload["critic2"])),
            self.critic2_target.state_dict(),
            old_state_dim=old_state_dim,
            new_state_dim=int(self.state_dim),
            old_tail_dim=critic_old_tail,
            new_tail_dim=critic_new_tail,
        )

        self._unwrap(self.actor).load_state_dict(actor_state)
        self.actor_target.load_state_dict(actor_target_state)
        self._unwrap(self.critic1).load_state_dict(critic1_state)
        self._unwrap(self.critic2).load_state_dict(critic2_state)
        self.critic1_target.load_state_dict(critic1_target_state)
        self.critic2_target.load_state_dict(critic2_target_state)

        adapted = any(
            (
                state_dim_changed,
                actor_adapted,
                actor_target_adapted,
                critic1_adapted,
                critic2_adapted,
                critic1_target_adapted,
                critic2_target_adapted,
            )
        )
        if not adapted:
            self.actor_optim.load_state_dict(payload["actor_optim"])
            self.critic_optim.load_state_dict(payload["critic_optim"])
        self.gradient_step = int(payload.get("gradient_step", 0))
        return int(payload.get("global_step", 0))

    def save(self, *, global_step: int, args: argparse.Namespace, latest: bool = False) -> Path:
        payload = {
            "policy_type": "octo_small_cdpr",
            "base_checkpoint": str(args.base_checkpoint),
            "global_step": int(global_step),
            "gradient_step": int(self.gradient_step),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "chunk_size": int(self.chunk_size),
            "residual_scale": float(args.residual_scale),
            "hidden_dim": int(args.hidden_dim),
            "actor": self._unwrap(self.actor).state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1": self._unwrap(self.critic1).state_dict(),
            "critic2": self._unwrap(self.critic2).state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "args": vars(args),
        }
        if latest:
            output_path = self.run_dir / "latest.pt"
        else:
            step_dir = self.run_dir / f"step_{int(global_step):07d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            output_path = step_dir / "octo_cdpr_adapter.pt"
        torch.save(payload, output_path)
        if not latest:
            torch.save(payload, self.run_dir / "latest.pt")
        return output_path


def _module_parameter_summary(module: nn.Module) -> dict[str, int]:
    base = ResidualTrainer._unwrap(module)
    total = 0
    trainable = 0
    for param in base.parameters():
        count = int(param.numel())
        total += count
        if bool(param.requires_grad):
            trainable += count
    return {"total": int(total), "trainable": int(trainable)}


def _optimizer_parameter_count(*optimizers: Any) -> int:
    seen: set[int] = set()
    total = 0
    for optimizer in optimizers:
        for group in getattr(optimizer, "param_groups", []):
            for param in group.get("params", []):
                ident = id(param)
                if ident in seen:
                    continue
                seen.add(ident)
                total += int(param.numel())
    return int(total)


def _residual_trainer_parameter_summary(
    trainer: ResidualTrainer,
    *,
    pretrained_prior_name: str,
    base_checkpoint: str,
    pretrained_prior_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    online_modules = {
        "residual_actor": _module_parameter_summary(trainer.actor),
        "critic1": _module_parameter_summary(trainer.critic1),
        "critic2": _module_parameter_summary(trainer.critic2),
    }
    target_modules = {
        "actor_target": _module_parameter_summary(trainer.actor_target),
        "critic1_target": _module_parameter_summary(trainer.critic1_target),
        "critic2_target": _module_parameter_summary(trainer.critic2_target),
    }
    return {
        "optimized_surface": "residual_actor_plus_online_q_critics",
        "optimized_parameter_count": _optimizer_parameter_count(
            trainer.actor_optim,
            trainer.critic_optim,
        ),
        "online_modules": online_modules,
        "target_modules": target_modules,
        "target_update": "soft_update_only_not_optimizer_step",
        "pretrained_prior": {
            "name": str(pretrained_prior_name),
            "checkpoint": str(base_checkpoint),
            "role": "frozen_prior_action_chunk_generator",
            "optimized": False,
            **(pretrained_prior_parameters or {}),
        },
    }


def _octo_pretrained_parameter_summary(runtime: Any) -> dict[str, Any]:
    model = getattr(runtime, "model", None)
    params = getattr(model, "params", None)
    if params is None:
        variables = getattr(model, "variables", None)
        if isinstance(variables, dict):
            params = variables.get("params")
    if params is None:
        return {"parameter_count_available": False}
    try:
        leaves = runtime.jax.tree_util.tree_leaves(params)
        total = 0
        for leaf in leaves:
            size = getattr(leaf, "size", None)
            total += int(size if size is not None else np.asarray(leaf).size)
        return {
            "parameter_count_available": True,
            "parameter_count": int(total),
            "trainable_parameter_count": 0,
        }
    except Exception as exc:
        return {
            "parameter_count_available": False,
            "parameter_count_error": repr(exc),
        }


def _resolve_checkpoint(raw: str | Path) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.is_file():
        return path
    for name in ("octo_cdpr_adapter.pt", "latest.pt"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find an Octo CDPR checkpoint in {path}")


def _predict_prior_chunk(
    *,
    runtime: Any,
    obs_adapter: CDPROctoObservationAdapter,
    action_spec: OctoActionAdapterSpec,
    env: Any,
    obs: dict[str, np.ndarray],
    info: dict[str, Any],
    instruction: str,
) -> np.ndarray:
    octo_obs = obs_adapter.from_env(sim=env.sim, obs=obs, info=info)
    raw_actions = runtime.sample_actions(octo_obs, instruction)
    return adapt_octo_actions_to_cdpr(raw_actions, spec=action_spec)


def _exploration_noise(args: argparse.Namespace, global_step: int) -> float:
    start = float(args.exploration_noise)
    end = float(args.min_exploration_noise)
    decay_steps = max(1, int(args.noise_decay_steps))
    alpha = min(1.0, max(0.0, float(global_step) / decay_steps))
    return float(start + alpha * (end - start))


def _log_scalars(writer: Any, metrics: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar(key, float(value), int(step))
    writer.flush()


def _tensorboard_series_key(raw: Any) -> str:
    text = str(raw).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return text or "unknown"


def _log_grouped_scalars(writer: Any, main_tag: str, values: dict[str, Any], step: int) -> None:
    if writer is None or not values:
        return
    cleaned: dict[str, float] = {}
    for key, value in values.items():
        if value is None:
            continue
        try:
            cleaned[_tensorboard_series_key(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return
    add_scalars = getattr(writer, "add_scalars", None)
    if callable(add_scalars):
        add_scalars(str(main_tag), cleaned, int(step))
    else:
        for key, value in cleaned.items():
            writer.add_scalar(f"{main_tag}/{key}", value, int(step))
    writer.flush()


def _log_mixed_curriculum_scalars(writer: Any, metrics: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    writer.add_scalar(
        "curriculum/mixed_enabled",
        1.0 if bool(metrics.get("mixed_curriculum_enabled", False)) else 0.0,
        int(step),
    )
    _log_grouped_scalars(
        writer,
        "curriculum/instruction_sampling_probability",
        dict(metrics.get("mixed_curriculum_last_probabilities") or {}),
        step,
    )
    _log_grouped_scalars(
        writer,
        "curriculum/instruction_success_rate_recent",
        dict(metrics.get("mixed_curriculum_success_rates") or {}),
        step,
    )
    _log_grouped_scalars(
        writer,
        "curriculum/instruction_episode_count_recent",
        dict(metrics.get("mixed_curriculum_episode_counts") or {}),
        step,
    )
    _log_grouped_scalars(
        writer,
        "curriculum/instruction_episode_fraction_recent",
        dict(metrics.get("mixed_curriculum_recent_episode_fractions") or {}),
        step,
    )
    _log_grouped_scalars(
        writer,
        "curriculum/instruction_episode_count_total",
        dict(metrics.get("mixed_curriculum_total_episode_counts") or {}),
        step,
    )
    writer.flush()


def _episode_metric_snapshot(
    *,
    rewards: deque[float],
    successes: deque[float],
    episode_count: int,
    success_count: int,
) -> dict[str, float | int | None]:
    window_count = len(rewards)
    reward_mean = float(sum(rewards) / window_count) if window_count else None
    success_rate = float(sum(successes) / window_count) if window_count else None
    lifetime_success_rate = float(success_count / episode_count) if episode_count else None
    return {
        "episode_window_count": int(window_count),
        "episode_count": int(episode_count),
        "episode_success_count": int(success_count),
        "episode_reward_mean": reward_mean,
        "success_rate": success_rate,
        "success_rate_lifetime": lifetime_success_rate,
    }


def _episode_metric_scalars(snapshot: dict[str, float | int | None]) -> dict[str, float]:
    scalars: dict[str, float] = {
        "rollout/episode_window_count": float(snapshot["episode_window_count"] or 0),
        "rollout/episode_count": float(snapshot["episode_count"] or 0),
    }
    reward_mean = snapshot.get("episode_reward_mean")
    if reward_mean is not None:
        scalars["rollout/episode_reward_mean"] = float(reward_mean)
        scalars["rollout_step/reward_env_mean"] = float(reward_mean)
    success_rate = snapshot.get("success_rate")
    if success_rate is not None:
        scalars["rollout/success_rate"] = float(success_rate)
        scalars["rollout_step/success_rate_mean"] = float(success_rate)
        scalars["rollout_step/episode_success_rate_mean"] = float(success_rate)
    lifetime_success_rate = snapshot.get("success_rate_lifetime")
    if lifetime_success_rate is not None:
        scalars["rollout/success_rate_lifetime"] = float(lifetime_success_rate)
    return scalars


def _dedupe_instruction_names(raw: Sequence[str] | None) -> tuple[str, ...]:
    if not raw:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def _dense_curriculum_enabled(args: argparse.Namespace) -> bool:
    return bool(
        _dedupe_instruction_names(getattr(args, "dense_stage_one_instruction_types", None))
        and _dedupe_instruction_names(getattr(args, "dense_stage_two_instruction_types", None))
    )


def _mixed_curriculum_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "mixed_curriculum_enabled", False) and _dense_curriculum_enabled(args))


def _dense_stage_instruction_types(args: argparse.Namespace, stage_index: int) -> tuple[str, ...]:
    if int(stage_index) <= 1:
        return _dedupe_instruction_names(getattr(args, "dense_stage_one_instruction_types", None))
    return _dedupe_instruction_names(getattr(args, "dense_stage_two_instruction_types", None))


def _active_dense_stage_instruction_types(args: argparse.Namespace, stage_index: int) -> tuple[str, ...]:
    if int(stage_index) <= 1 or not _mixed_curriculum_enabled(args):
        return _dense_stage_instruction_types(args, stage_index)
    return _dedupe_instruction_names(
        [
            *_dense_stage_instruction_types(args, 1),
            *_dense_stage_instruction_types(args, 2),
        ]
    )


def _apply_dense_stage_to_envs(
    envs: Sequence[Any],
    args: argparse.Namespace,
    stage_index: int,
) -> tuple[str, ...]:
    instruction_types = _active_dense_stage_instruction_types(args, stage_index)
    if not instruction_types:
        return ()
    for env in envs:
        setattr(env, "instruction_types", tuple(instruction_types))
    return instruction_types


def _dense_stage_snapshot(
    *,
    args: argparse.Namespace,
    stage_index: int,
    stage_instruction_types: Sequence[str],
    successes: deque[float],
    episode_count: int,
    success_count: int,
) -> dict[str, Any]:
    enabled = _dense_curriculum_enabled(args)
    window_count = len(successes)
    success_rate = float(sum(successes) / window_count) if window_count else None
    lifetime_success_rate = float(success_count / episode_count) if episode_count else None
    return {
        "dense_stage_enabled": bool(enabled),
        "dense_stage_index": int(stage_index) if enabled else 0,
        "dense_stage_name": f"stage_{int(stage_index)}" if enabled else "disabled",
        "dense_stage_instruction_types": list(stage_instruction_types) if enabled else [],
        "dense_stage_episode_window_count": int(window_count),
        "dense_stage_episode_count": int(episode_count),
        "dense_stage_success_count": int(success_count),
        "dense_stage_success_rate": success_rate,
        "dense_stage_success_rate_lifetime": lifetime_success_rate,
        "dense_stage_switch_success_rate": float(getattr(args, "dense_stage_switch_success_rate", 0.50)),
        "dense_stage_min_episodes": int(getattr(args, "dense_stage_min_episodes", 0)),
    }


def _dense_stage_metric_scalars(snapshot: dict[str, Any]) -> dict[str, float]:
    if not bool(snapshot.get("dense_stage_enabled", False)):
        return {}
    scalars: dict[str, float] = {
        "stage/dense/index": float(snapshot.get("dense_stage_index", 0)),
        "stage/dense/episode_window_count": float(snapshot.get("dense_stage_episode_window_count", 0)),
        "stage/dense/episode_count": float(snapshot.get("dense_stage_episode_count", 0)),
        "stage/dense/switch_success_rate": float(snapshot.get("dense_stage_switch_success_rate", 0.0)),
        "stage/dense/min_episodes": float(snapshot.get("dense_stage_min_episodes", 0)),
    }
    success_rate = snapshot.get("dense_stage_success_rate")
    if success_rate is not None:
        scalars["stage/dense/success_rate"] = float(success_rate)
    lifetime_success_rate = snapshot.get("dense_stage_success_rate_lifetime")
    if lifetime_success_rate is not None:
        scalars["stage/dense/success_rate_lifetime"] = float(lifetime_success_rate)
    return scalars


def _distributed_completed_successes(
    *,
    ctx: DistributedContext,
    device: torch.device,
    done: bool,
    success: bool,
) -> list[float]:
    local = [1.0 if done else 0.0, 1.0 if success else 0.0]
    if (
        ctx.enabled
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        tensor = torch.as_tensor(local, dtype=torch.float32, device=device)
        gathered = [torch.zeros_like(tensor) for _ in range(int(ctx.world_size))]
        torch.distributed.all_gather(gathered, tensor)
        if not ctx.is_main:
            return []
        successes: list[float] = []
        for item in gathered:
            values = item.detach().cpu().tolist()
            if float(values[0]) >= 0.5:
                successes.append(1.0 if float(values[1]) >= 0.5 else 0.0)
        return successes
    return [1.0 if success else 0.0] if done else []


def _broadcast_dense_stage(
    *,
    ctx: DistributedContext,
    device: torch.device,
    stage_index: int,
) -> int:
    if not (
        ctx.enabled
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return int(stage_index)
    tensor = torch.as_tensor([int(stage_index)], dtype=torch.int64, device=device)
    torch.distributed.broadcast(tensor, src=0)
    return int(tensor.item())


def _distributed_barrier(ctx: DistributedContext) -> None:
    if (
        ctx.enabled
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()


def _validation_enabled(args: argparse.Namespace) -> bool:
    return int(getattr(args, "validation_every_steps", 0)) > 0 and int(
        getattr(args, "validation_episodes_per_instruction", 0)
    ) > 0


def _validation_due(args: argparse.Namespace, *, global_step: int, last_validation_step: int) -> bool:
    if not _validation_enabled(args) or int(global_step) <= 0:
        return False
    every = max(1, int(args.validation_every_steps))
    return int(global_step) // every > int(last_validation_step) // every


def _validation_instruction_types(
    args: argparse.Namespace,
    *,
    dense_curriculum_active: bool,
    dense_stage_index: int,
) -> tuple[str, ...]:
    if dense_curriculum_active:
        return _active_dense_stage_instruction_types(args, dense_stage_index)
    configured = _dedupe_instruction_names(getattr(args, "instruction_types", None))
    return configured


def _validation_gate_success_rate(summary: dict[str, Any], args: argparse.Namespace) -> float | None:
    if str(getattr(args, "dense_stage_gate_aggregation", "mean")) != "worst":
        raw = summary.get("validation_success_rate")
        return None if raw is None else float(raw)
    instruction_results = summary.get("validation_instruction_results")
    if not isinstance(instruction_results, dict):
        raw = summary.get("validation_success_rate")
        return None if raw is None else float(raw)
    rates: list[float] = []
    for item in instruction_results.values():
        if not isinstance(item, dict):
            continue
        if int(item.get("episodes") or 0) <= 0:
            continue
        rate = item.get("success_rate")
        if rate is not None:
            rates.append(float(rate))
    if not rates:
        return None
    return float(min(rates))


def _info_instruction_type(info: dict[str, Any], fallback_instruction: str | None = None) -> str:
    raw = info.get("instruction_type")
    if raw:
        return str(raw)
    from robots.cdpr.cdpr_dataset.rl_cdpr_env import _infer_instruction_type_from_text

    inferred = _infer_instruction_type_from_text(str(fallback_instruction or info.get("language_instruction") or ""))
    return str(inferred or "")


def _safe_filename_token(value: Any) -> str:
    token = str(value or "").strip().lower().replace(" ", "_")
    token = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in token)
    return token.strip("_") or "unknown"


def _clear_sim_recording_buffers(sim: Any) -> None:
    try:
        from robots.cdpr.cdpr_dataset.synthetic_tasks import clear_sim_recording_buffers

        clear_sim_recording_buffers(sim)
        return
    except Exception:
        pass
    for attr in ("trajectory_data", "overview_frames", "ee_camera_frames", "frame_capture_timestamps"):
        if hasattr(sim, attr):
            try:
                setattr(sim, attr, [])
            except Exception:
                pass


def _begin_validation_recording(env: Any, enabled: bool) -> dict[str, Any]:
    sim = getattr(env, "sim", None)
    state = {
        "enabled": bool(enabled),
        "capture_frames": getattr(env, "capture_frames", None),
        "sim": sim,
        "record_trajectory": getattr(sim, "record_trajectory", None) if sim is not None else None,
    }
    if not enabled:
        return state
    try:
        env.capture_frames = True
    except Exception:
        pass
    if sim is not None:
        try:
            sim.record_trajectory = True
        except Exception:
            pass
        _clear_sim_recording_buffers(sim)
    return state


def _finish_validation_recording(env: Any, state: dict[str, Any]) -> None:
    if not bool(state.get("enabled", False)):
        return
    sim = state.get("sim")
    if sim is not None:
        _clear_sim_recording_buffers(sim)
        previous_recording = state.get("record_trajectory")
        if previous_recording is not None:
            try:
                sim.record_trajectory = bool(previous_recording)
            except Exception:
                pass
    previous_capture = state.get("capture_frames")
    if previous_capture is not None:
        try:
            env.capture_frames = bool(previous_capture)
        except Exception:
            pass


def _validation_video_frames(sim: Any, episode_length: int) -> list[Any]:
    frames = list(getattr(sim, "overview_frames", []) or [])
    if len(frames) != 1:
        return frames
    target_frame_count = max(2, min(int(episode_length), 600))
    return frames * target_frame_count


def _save_training_validation_video(
    *,
    run_dir: Path,
    sim: Any,
    global_step: int,
    instruction_type: str,
    episode_index: int,
    seed: int,
    success: bool,
    episode_reward: float,
    episode_length: int,
    fps: float,
) -> dict[str, Any] | None:
    frames = _validation_video_frames(sim, int(episode_length))
    if not frames or not hasattr(sim, "save_video"):
        return None
    step_dir = run_dir / "validation_videos" / f"step_{int(global_step):07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    outcome = "success" if bool(success) else "failure"
    output_path = step_dir / (
        f"{_safe_filename_token(instruction_type)}_{outcome}_episode_{int(episode_index):03d}_overview.mp4"
    )
    fps = max(1.0, float(fps))
    summary: dict[str, Any] = {
        "video_path": output_path.as_posix(),
        "global_step": int(global_step),
        "instruction_type": str(instruction_type),
        "episode_index": int(episode_index),
        "seed": int(seed),
        "success": bool(success),
        "episode_reward": float(episode_reward),
        "episode_length": int(episode_length),
        "frame_count": int(len(frames)),
        "fps": float(fps),
        "duration_sec": float(len(frames) / fps),
    }
    try:
        sim.save_video(frames, str(output_path), fps=fps)
        summary["saved"] = True
        summary["size_bytes"] = int(output_path.stat().st_size) if output_path.is_file() else 0
    except Exception as exc:
        summary["saved"] = False
        summary["error"] = repr(exc)
    summary_path = output_path.with_name(output_path.name.replace("_overview.mp4", "_summary.json"))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _validation_video_episode_slots(
    *,
    instruction_count: int,
    episode_count: int,
    video_limit: int,
    global_step: int,
    validation_every_steps: int,
) -> set[tuple[int, int]]:
    if int(instruction_count) <= 0 or int(episode_count) <= 0 or int(video_limit) <= 0:
        return set()

    total_slots = int(instruction_count) * int(episode_count)
    target_count = min(int(video_limit), total_slots)
    validation_period = max(1, int(validation_every_steps))
    start_instruction = (int(global_step) // validation_period) % int(instruction_count)

    slots: set[tuple[int, int]] = set()
    for episode_index in range(int(episode_count)):
        for instruction_offset in range(int(instruction_count)):
            instruction_index = (start_instruction + instruction_offset) % int(instruction_count)
            slots.add((instruction_index, episode_index))
            if len(slots) >= target_count:
                return slots
    return slots


def _select_validation_chunk(
    actor: nn.Module,
    *,
    device: torch.device,
    state: np.ndarray,
    prior_chunk: np.ndarray,
) -> np.ndarray:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    prior_t = torch.as_tensor(prior_chunk, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_chunk = actor(state_t, prior_t)[0].detach().cpu().numpy()
    return np.clip(action_chunk, -1.0, 1.0).astype(np.float32, copy=False)


def _summarize_validation_results(
    *,
    global_step: int,
    stage_index: int,
    instruction_types: Sequence[str],
    results: list[dict[str, Any]],
    videos: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    episode_count = len(results)
    success_count = sum(1 for item in results if bool(item.get("success", False)))
    reward_values = [float(item.get("episode_reward", 0.0)) for item in results]
    instruction_results: dict[str, dict[str, Any]] = {}
    for instruction_type in instruction_types:
        subset = [item for item in results if str(item.get("instruction_type")) == str(instruction_type)]
        subset_successes = sum(1 for item in subset if bool(item.get("success", False)))
        subset_rewards = [float(item.get("episode_reward", 0.0)) for item in subset]
        instruction_results[str(instruction_type)] = {
            "episodes": int(len(subset)),
            "successes": int(subset_successes),
            "success_rate": float(subset_successes / len(subset)) if subset else None,
            "episode_reward_mean": float(sum(subset_rewards) / len(subset_rewards)) if subset_rewards else None,
        }
    return {
        "event": "validation",
        "global_step": int(global_step),
        "validation_stage_index": int(stage_index),
        "validation_instruction_types": list(instruction_types),
        "validation_episode_count": int(episode_count),
        "validation_success_count": int(success_count),
        "validation_success_rate": float(success_count / episode_count) if episode_count else None,
        "validation_episode_reward_mean": float(sum(reward_values) / len(reward_values)) if reward_values else None,
        "validation_instruction_results": instruction_results,
        "validation_videos": list(videos or []),
    }


def _validation_summary_scalars(summary: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {
        "validation/episode_count": float(summary.get("validation_episode_count") or 0),
        "validation/success_count": float(summary.get("validation_success_count") or 0),
        "validation/stage_index": float(summary.get("validation_stage_index") or 0),
    }
    success_rate = summary.get("validation_success_rate")
    if success_rate is not None:
        scalars["validation/success_rate"] = float(success_rate)
    reward_mean = summary.get("validation_episode_reward_mean")
    if reward_mean is not None:
        scalars["validation/episode_reward_mean"] = float(reward_mean)
    instruction_results = summary.get("validation_instruction_results")
    if isinstance(instruction_results, dict):
        for instruction_type, item in instruction_results.items():
            if not isinstance(item, dict):
                continue
            instr_success_rate = item.get("success_rate")
            if instr_success_rate is not None:
                scalars[f"validation/instruction_success_rate/{instruction_type}"] = float(instr_success_rate)
            instr_reward_mean = item.get("episode_reward_mean")
            if instr_reward_mean is not None:
                scalars[f"validation/instruction_episode_reward_mean/{instruction_type}"] = float(instr_reward_mean)
    return scalars


def _write_validation_summary(
    *,
    metrics_path: Path,
    writer: Any,
    summary: dict[str, Any],
) -> None:
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")
    _log_scalars(writer, _validation_summary_scalars(summary), int(summary["global_step"]))
    instruction_results = summary.get("validation_instruction_results")
    if isinstance(instruction_results, dict):
        _log_grouped_scalars(
            writer,
            "validation/instruction_success_rate_grouped",
            {
                name: item.get("success_rate")
                for name, item in instruction_results.items()
                if isinstance(item, dict)
            },
            int(summary["global_step"]),
        )
        _log_grouped_scalars(
            writer,
            "validation/instruction_episode_reward_mean_grouped",
            {
                name: item.get("episode_reward_mean")
                for name, item in instruction_results.items()
                if isinstance(item, dict)
            },
            int(summary["global_step"]),
        )


def _run_octo_distinct_validation(
    *,
    validation_env: Any,
    runtime: Any,
    obs_adapter: CDPROctoObservationAdapter,
    action_spec: OctoActionAdapterSpec,
    trainer: ResidualTrainer,
    layout: CDPRStateLayout,
    args: argparse.Namespace,
    global_step: int,
    stage_index: int,
    instruction_types: Sequence[str],
) -> dict[str, Any]:
    episode_count = max(1, int(args.validation_episodes_per_instruction))
    replan_every = max(1, min(int(args.replan_every), int(args.chunk_size)))
    actor = trainer._unwrap(trainer.actor)
    was_training = bool(actor.training)
    previous_instruction_types = getattr(validation_env, "instruction_types", None)
    results: list[dict[str, Any]] = []
    saved_videos: list[dict[str, Any]] = []
    video_limit = max(0, int(getattr(args, "validation_video_count", 0)))
    video_slots = _validation_video_episode_slots(
        instruction_count=len(instruction_types),
        episode_count=episode_count,
        video_limit=video_limit,
        global_step=global_step,
        validation_every_steps=int(getattr(args, "validation_every_steps", 1)),
    )
    actor.eval()
    try:
        for instruction_index, instruction_type in enumerate(instruction_types):
            validation_env.instruction_types = (str(instruction_type),)
            for episode_index in range(episode_count):
                seed = (
                    int(args.validation_seed)
                    + int(global_step) * 17
                    + int(instruction_index) * 1009
                    + int(episode_index)
                )
                with _silence_output(True):
                    obs, info = validation_env.reset(seed=seed)
                should_record_video = (int(instruction_index), int(episode_index)) in video_slots
                recording_state = _begin_validation_recording(validation_env, should_record_video)
                video_summary: dict[str, Any] | None = None
                try:
                    state = layout.flatten(obs)
                    instruction = _safe_instruction(info)
                    with _silence_output(True):
                        prior_chunk = _predict_prior_chunk(
                            runtime=runtime,
                            obs_adapter=obs_adapter,
                            action_spec=action_spec,
                            env=validation_env,
                            obs=obs,
                            info=info,
                            instruction=instruction,
                        )
                    action_chunk = _select_validation_chunk(
                        actor,
                        device=trainer.device,
                        state=state,
                        prior_chunk=prior_chunk,
                    )
                    chunk_idx = 0
                    episode_reward = 0.0
                    episode_length = 0
                    terminated = False
                    truncated = False
                    final_info = dict(info)
                    while not (terminated or truncated):
                        if chunk_idx >= replan_every:
                            instruction = _safe_instruction(final_info)
                            with _silence_output(True):
                                prior_chunk = _predict_prior_chunk(
                                    runtime=runtime,
                                    obs_adapter=obs_adapter,
                                    action_spec=action_spec,
                                    env=validation_env,
                                    obs=obs,
                                    info=final_info,
                                    instruction=instruction,
                                )
                            action_chunk = _select_validation_chunk(
                                actor,
                                device=trainer.device,
                                state=state,
                                prior_chunk=prior_chunk,
                            )
                            chunk_idx = 0
                        action = np.asarray(action_chunk[chunk_idx], dtype=np.float32).reshape(int(args.action_dim))
                        action = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)
                        with _silence_output(True):
                            obs, reward, terminated, truncated, final_info = validation_env.step(action)
                        state = layout.flatten(obs)
                        episode_reward += float(reward)
                        episode_length += 1
                        chunk_idx += 1
                    result = {
                        "instruction_type": str(instruction_type),
                        "episode_index": int(episode_index),
                        "seed": int(seed),
                        "success": bool(final_info.get("success", False)),
                        "episode_reward": float(episode_reward),
                        "episode_length": int(episode_length),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                    if should_record_video:
                        video_summary = _save_training_validation_video(
                            run_dir=trainer.run_dir,
                            sim=validation_env.sim,
                            global_step=global_step,
                            instruction_type=str(instruction_type),
                            episode_index=episode_index,
                            seed=seed,
                            success=bool(result["success"]),
                            episode_reward=float(episode_reward),
                            episode_length=int(episode_length),
                            fps=float(getattr(args, "validation_video_fps", 10.0)),
                        )
                        if video_summary is not None:
                            saved_videos.append(video_summary)
                    results.append(result)
                finally:
                    _finish_validation_recording(validation_env, recording_state)
    finally:
        actor.train(was_training)
        validation_env.instruction_types = previous_instruction_types

    return _summarize_validation_results(
        global_step=global_step,
        stage_index=stage_index,
        instruction_types=instruction_types,
        results=results,
        videos=saved_videos,
    )


def _safe_instruction(info: dict[str, Any]) -> str:
    return str(info.get("language_instruction") or info.get("instruction_type") or "move left")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dist_ctx = _configure_distributed(args)
    _set_quiet_env(args, dist_ctx)
    _set_seed(int(args.seed))
    rollout_seed = _rank_seed(int(args.seed), dist_ctx)

    run_dir = _make_run_dir(args.run_root_dir, args.run_id)
    if dist_ctx.is_main:
        _write_json(run_dir / "config.json", vars(args))
    writer = (
        SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        if dist_ctx.is_main and SummaryWriter is not None
        else None
    )
    metrics_window = max(1, int(args.metrics_window_episodes))
    dense_curriculum_active = _dense_curriculum_enabled(args)
    dense_stage_index = 1 if dense_curriculum_active else 0
    dense_stage_successes: deque[float] = deque(maxlen=metrics_window)
    dense_stage_episode_count = 0
    dense_stage_success_count = 0
    dense_stage_instruction_types = (
        _dense_stage_instruction_types(args, dense_stage_index) if dense_curriculum_active else ()
    )
    mixed_sampler = AdaptiveInstructionSampler(args, seed=int(rollout_seed))

    obs_adapter = CDPROctoObservationAdapter(
        OctoObservationSpec(
            image_size=int(args.image_size),
            history=int(args.history),
            include_wrist=bool(args.include_wrist),
            include_proprio=bool(args.include_proprio),
        )
    )
    action_spec = OctoActionAdapterSpec(
        action_dim=int(args.action_dim),
        chunk_size=int(args.chunk_size),
        action_indices=None if args.octo_action_indices is None else tuple(int(v) for v in args.octo_action_indices),
        normalization=str(args.octo_action_normalization),
    )

    startup_t0 = time.perf_counter()
    _log(
        dist_ctx,
        f"[octo-cdpr] Loading Octo checkpoint: {args.base_checkpoint} "
        f"(rank={dist_ctx.rank}, world_size={dist_ctx.world_size}, device={dist_ctx.device})",
    )
    load_t0 = time.perf_counter()
    with _silence_output(not dist_ctx.is_main):
        runtime = load_octo_runtime(
            checkpoint=str(args.base_checkpoint),
            seed=int(rollout_seed),
            use_dataset_action_unnorm=bool(args.use_dataset_action_unnorm),
        )
    _log(
        dist_ctx,
        f"[octo-cdpr] Loaded Octo checkpoint in {time.perf_counter() - load_t0:.1f}s; "
        f"{runtime.device_summary()}; distributed_world_size={dist_ctx.world_size}",
    )
    obs_adapter = obs_adapter.with_example_observation(runtime.example_observation)
    if obs_adapter.example_observation is not None and dist_ctx.is_main and not bool(args.progress_only):
        _log(dist_ctx, f"[octo-cdpr] Using Octo observation schema: {obs_adapter.expected_shape_summary()}")

    env = None
    validation_env = None
    progress = None
    try:
        env_t0 = time.perf_counter()
        _log(dist_ctx, "[octo-cdpr] Building CDPR environment...")
        with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
            env = _build_env(args, seed=rollout_seed)
        if dense_curriculum_active:
            dense_stage_instruction_types = _apply_dense_stage_to_envs(
                [env],
                args,
                dense_stage_index,
            )
            _log(
                dist_ctx,
                "[octo-cdpr] Dense curriculum stage 1 active: "
                f"{', '.join(dense_stage_instruction_types)}",
            )
        _log(dist_ctx, f"[octo-cdpr] Built CDPR environment in {time.perf_counter() - env_t0:.1f}s")
        reset_t0 = time.perf_counter()
        _log(dist_ctx, "[octo-cdpr] Resetting CDPR environment...")
        with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
            obs, info = env.reset(
                seed=int(rollout_seed),
                options=mixed_sampler.reset_options(dense_stage_index),
            )
        _log(dist_ctx, f"[octo-cdpr] Reset CDPR environment in {time.perf_counter() - reset_t0:.1f}s")
        layout = CDPRStateLayout.from_observation(obs)
        if dist_ctx.is_main and _validation_enabled(args):
            _log(
                dist_ctx,
                "[octo-cdpr] Building held-out validation environment "
                f"(every {int(args.validation_every_steps)} steps, "
                f"{int(args.validation_episodes_per_instruction)} episode(s)/instruction)...",
            )
            with _silence_output(bool(args.progress_only)):
                validation_env = _build_env(args, seed=int(args.validation_seed))
        device = torch.device(args.device)
        trainer = ResidualTrainer(
            args=args,
            state_dim=layout.state_dim,
            action_dim=int(args.action_dim),
            chunk_size=int(args.chunk_size),
            run_dir=run_dir,
            device=device,
            distributed=dist_ctx,
        )
        start_step = 0
        if args.resume_checkpoint:
            resume_path = _resolve_checkpoint(args.resume_checkpoint)
            start_step = trainer.load(resume_path)
            _log(dist_ctx, f"[octo-cdpr] Resumed adapter checkpoint {resume_path} at step {start_step}")

        random.seed(int(rollout_seed))
        np.random.seed(int(rollout_seed))
        torch.manual_seed(int(rollout_seed))

        buffer = ReplayBuffer(
            capacity=int(args.replay_size),
            state_dim=layout.state_dim,
            chunk_size=int(args.chunk_size),
            action_dim=int(args.action_dim),
        )
        metrics_path = run_dir / "metrics.jsonl"
        manifest = {
            "policy_type": "octo_small_cdpr",
            "base_checkpoint": str(args.base_checkpoint),
            "run_dir": run_dir.as_posix(),
            "config": str(args.config or ""),
            "action_keys": ["x", "y", "z", "yaw", "gripper"],
            "chunk_size": int(args.chunk_size),
            "prior_action_adapter": {
                "source": "Octo action chunk",
                "target": "CDPR normalized 5D [x, y, z, yaw, gripper]",
                "default_source_indices_for_7d": [0, 1, 2, 5, 6],
                "configured_source_indices": (
                    list(args.octo_action_indices)
                    if args.octo_action_indices is not None
                    else [0, 1, 2, 5, 6]
                ),
                "normalization": str(args.octo_action_normalization),
            },
            "trainable_surface": "torch_residual_chunk_head_and_q_critics",
            "frozen_octo": True,
            "parameter_training": _residual_trainer_parameter_summary(
                trainer,
                pretrained_prior_name="Octo-Small",
                base_checkpoint=str(args.base_checkpoint),
                pretrained_prior_parameters=_octo_pretrained_parameter_summary(runtime),
            ),
            "distributed_world_size": int(dist_ctx.world_size),
            "rank_device": str(dist_ctx.device),
            "success_threshold_to_beat_openvla": {
                "overall_simple_success_rate": 0.167,
                "move_to_object_success_rate": 0.09,
            },
            "dense_success_curriculum": {
                "enabled": bool(dense_curriculum_active),
                "stage_one_instruction_types": list(
                    _dense_stage_instruction_types(args, 1)
                ),
                "stage_two_instruction_types": list(
                    _dense_stage_instruction_types(args, 2)
                ),
                "switch_success_rate": float(args.dense_stage_switch_success_rate),
                "min_stage_one_episodes": int(args.dense_stage_min_episodes),
                "metric_window_episodes": int(metrics_window),
                "gate_metric": str(args.dense_stage_gate_metric),
                "gate_aggregation": str(args.dense_stage_gate_aggregation),
                "mixed_curriculum": {
                    "enabled": bool(_mixed_curriculum_enabled(args)),
                    "adaptive": bool(args.mixed_curriculum_adaptive),
                    "success_target": float(args.mixed_curriculum_success_target),
                    "rehearsal_min_prob": float(args.mixed_curriculum_rehearsal_min_prob),
                    "move_to_object_min_prob": float(args.mixed_curriculum_move_to_object_min_prob),
                    "manipulation_min_prob": float(args.mixed_curriculum_manipulation_min_prob),
                    "near_success_instruction_types": list(
                        _dedupe_instruction_names(args.near_success_instruction_types)
                    ),
                },
            },
            "heldout_validation": {
                "enabled": bool(_validation_enabled(args)),
                "every_steps": int(args.validation_every_steps),
                "episodes_per_instruction": int(args.validation_episodes_per_instruction),
                "seed": int(args.validation_seed),
                "video_count": int(args.validation_video_count),
                "video_fps": float(args.validation_video_fps),
            },
        }
        if dist_ctx.is_main:
            _write_json(run_dir / "octo_manifest.json", manifest)

        global_step = int(start_step)
        episode = 0
        episode_reward = 0.0
        episode_length = 0
        episode_stage_index = int(dense_stage_index)
        completed_episode_rewards: deque[float] = deque(maxlen=metrics_window)
        completed_episode_successes: deque[float] = deque(maxlen=metrics_window)
        completed_episode_count = 0
        completed_episode_success_count = 0
        state = layout.flatten(obs)
        instruction = _safe_instruction(info)
        _log(
            dist_ctx,
            "[octo-cdpr] Sampling first Octo prior action chunk "
            "(first JAX call can spend minutes compiling on CPU before GPU work starts)...",
        )
        prior_t0 = time.perf_counter()
        with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
            prior_chunk = _predict_prior_chunk(
                runtime=runtime,
                obs_adapter=obs_adapter,
                action_spec=action_spec,
                env=env,
                obs=obs,
                info=info,
                instruction=instruction,
            )
        _log(
            dist_ctx,
            f"[octo-cdpr] First Octo prior chunk ready in {time.perf_counter() - prior_t0:.1f}s; "
            f"startup total {time.perf_counter() - startup_t0:.1f}s",
        )
        action_chunk = trainer.select_chunk(state, prior_chunk)
        chunk_idx = 0
        replan_every = max(1, min(int(args.replan_every), int(args.chunk_size)))
        last_metrics: dict[str, float] = {}
        progress = _make_progress_bar(args=args, ctx=dist_ctx, start_step=start_step)
        status_start_t = time.perf_counter()
        last_validation_step = int(start_step)

        while global_step < int(args.max_train_steps):
            if chunk_idx >= replan_every:
                with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                    prior_chunk = _predict_prior_chunk(
                        runtime=runtime,
                        obs_adapter=obs_adapter,
                        action_spec=action_spec,
                        env=env,
                        obs=obs,
                        info=info,
                        instruction=instruction,
                    )
                action_chunk = trainer.select_chunk(state, prior_chunk)
                chunk_idx = 0

            action_index = int(chunk_idx)
            action = np.asarray(action_chunk[action_index], dtype=np.float32).reshape(int(args.action_dim))
            noise_std = _exploration_noise(args, global_step)
            if noise_std > 0.0:
                action = action + np.random.normal(0.0, noise_std, size=action.shape).astype(np.float32)
            action = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)

            with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_state = layout.flatten(next_obs)
            done = bool(terminated or truncated)
            next_instruction = _safe_instruction(next_info)
            next_idx = min(action_index + 1, int(args.chunk_size) - 1)
            next_prior = prior_chunk
            if done or next_idx >= replan_every:
                next_prior = prior_chunk

            buffer.add(
                state=state,
                prior=prior_chunk,
                action_index=action_index,
                action=action,
                reward=float(reward),
                next_state=next_state,
                next_prior=next_prior,
                next_action_index=next_idx,
                done=done,
            )

            global_step += 1
            episode_length += 1
            episode_reward += float(reward)
            episode_success = bool(next_info.get("success", False)) if done else False
            dense_stage_switched = False
            if dense_curriculum_active:
                count_for_dense_stage = done and int(episode_stage_index) == int(dense_stage_index)
                completed_successes = _distributed_completed_successes(
                    ctx=dist_ctx,
                    device=device,
                    done=count_for_dense_stage,
                    success=episode_success,
                )
                proposed_stage_index = int(dense_stage_index)
                if dist_ctx.is_main:
                    for success_value in completed_successes:
                        dense_stage_successes.append(float(success_value))
                        dense_stage_episode_count += 1
                        dense_stage_success_count += int(float(success_value) >= 0.5)
                    if int(dense_stage_index) == 1 and str(args.dense_stage_gate_metric) == "rollout":
                        required_episodes = max(1, int(args.dense_stage_min_episodes))
                        stage_success_rate = (
                            float(sum(dense_stage_successes) / len(dense_stage_successes))
                            if dense_stage_successes
                            else 0.0
                        )
                        if (
                            len(dense_stage_successes) >= required_episodes
                            and stage_success_rate >= float(args.dense_stage_switch_success_rate)
                        ):
                            proposed_stage_index = 2
                            dense_stage_successes.clear()
                            dense_stage_episode_count = 0
                            dense_stage_success_count = 0
                previous_stage_index = int(dense_stage_index)
                dense_stage_index = _broadcast_dense_stage(
                    ctx=dist_ctx,
                    device=device,
                    stage_index=proposed_stage_index,
                )
                if int(dense_stage_index) != previous_stage_index:
                    dense_stage_switched = True
                    dense_stage_successes.clear()
                    dense_stage_episode_count = 0
                    dense_stage_success_count = 0
                    dense_stage_instruction_types = _apply_dense_stage_to_envs(
                        [env],
                        args,
                        dense_stage_index,
                    )
                    _log(
                        dist_ctx,
                        "[octo-cdpr] Dense curriculum switched to stage "
                        f"{dense_stage_index}: {', '.join(dense_stage_instruction_types)}",
                        progress=progress,
                    )

            if buffer.size >= int(args.update_after):
                for _ in range(int(args.updates_per_step)):
                    batch = buffer.sample(int(args.batch_size), device=device)
                    last_metrics = trainer.update(batch)
                    _log_scalars(writer, {f"train/{k}": v for k, v in last_metrics.items()}, trainer.gradient_step)

            if dist_ctx.is_main and global_step % max(1, int(args.log_every_steps)) == 0:
                episode_metrics = _episode_metric_snapshot(
                    rewards=completed_episode_rewards,
                    successes=completed_episode_successes,
                    episode_count=completed_episode_count,
                    success_count=completed_episode_success_count,
                )
                dense_stage_metrics = _dense_stage_snapshot(
                    args=args,
                    stage_index=dense_stage_index,
                    stage_instruction_types=dense_stage_instruction_types,
                    successes=dense_stage_successes,
                    episode_count=dense_stage_episode_count,
                    success_count=dense_stage_success_count,
                )
                mixed_curriculum_metrics = mixed_sampler.snapshot()
                row = {
                    "event": "rollout_step",
                    "global_step": int(global_step),
                    "episode": int(episode),
                    "episode_length": int(episode_length),
                    "episode_reward_running": float(episode_reward),
                    "reward": float(reward),
                    "done": bool(done),
                    "buffer_size": int(buffer.size),
                    "instruction": str(instruction),
                    "episode_dense_stage_index": int(episode_stage_index),
                    "noise_std": float(noise_std),
                    **episode_metrics,
                    **dense_stage_metrics,
                    **mixed_curriculum_metrics,
                    **last_metrics,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                rollout_scalars = {
                    "rollout/reward": float(reward),
                    "rollout/episode_reward_running": float(episode_reward),
                    "rollout/noise_std": float(noise_std),
                    "rollout/buffer_size": float(buffer.size),
                }
                rollout_scalars.update(_episode_metric_scalars(episode_metrics))
                rollout_scalars.update(_dense_stage_metric_scalars(dense_stage_metrics))
                _log_scalars(writer, rollout_scalars, global_step)
                _log_mixed_curriculum_scalars(writer, mixed_curriculum_metrics, global_step)

            if progress is not None:
                progress.update(1)
                if global_step % max(1, int(args.status_every_steps)) == 0 or done:
                    _progress_postfix(
                        progress,
                        episode=episode,
                        episode_length=episode_length,
                        episode_reward=episode_reward,
                        reward=float(reward),
                        buffer_size=buffer.size,
                        instruction=instruction,
                        world_size=dist_ctx.world_size,
                    )
            elif dist_ctx.is_main and global_step % max(1, int(args.status_every_steps)) == 0:
                _log(
                    dist_ctx,
                    f"[octo-cdpr] step={global_step:07d} episode={episode:05d} "
                    f"ep_len={episode_length:03d} reward_running={episode_reward:+.3f} "
                    f"last_reward={float(reward):+.3f} buffer={buffer.size} instruction={instruction}",
                )
                _log(
                    dist_ctx,
                    "[octo-cdpr] "
                    + _format_step_progress(
                        global_step=global_step,
                        max_train_steps=int(args.max_train_steps),
                        start_step=int(start_step),
                        elapsed_seconds=time.perf_counter() - status_start_t,
                    ),
                )

            if dist_ctx.is_main and global_step % max(1, int(args.save_every_steps)) == 0:
                checkpoint = trainer.save(global_step=global_step, args=args, latest=False)
                _log(dist_ctx, f"[octo-cdpr] Saved checkpoint: {checkpoint}", progress=progress)

            if _validation_due(args, global_step=global_step, last_validation_step=last_validation_step):
                proposed_stage_index = int(dense_stage_index)
                if dist_ctx.is_main:
                    validation_instruction_types = _validation_instruction_types(
                        args,
                        dense_curriculum_active=dense_curriculum_active,
                        dense_stage_index=dense_stage_index,
                    )
                    if validation_env is not None and validation_instruction_types:
                        validation_summary = _run_octo_distinct_validation(
                            validation_env=validation_env,
                            runtime=runtime,
                            obs_adapter=obs_adapter,
                            action_spec=action_spec,
                            trainer=trainer,
                            layout=layout,
                            args=args,
                            global_step=global_step,
                            stage_index=dense_stage_index,
                            instruction_types=validation_instruction_types,
                        )
                        _write_validation_summary(
                            metrics_path=metrics_path,
                            writer=writer,
                            summary=validation_summary,
                        )
                        _log(
                            dist_ctx,
                            "[octo-cdpr] Held-out validation "
                            f"step={global_step:07d} stage={dense_stage_index} "
                            f"episodes={validation_summary['validation_episode_count']} "
                            f"success_rate={float(validation_summary['validation_success_rate'] or 0.0):.3f} "
                            f"gate={float(_validation_gate_success_rate(validation_summary, args) or 0.0):.3f}",
                            progress=progress,
                        )
                        gate_success_rate = _validation_gate_success_rate(validation_summary, args)
                        if (
                            dense_curriculum_active
                            and str(args.dense_stage_gate_metric) == "validation"
                            and int(dense_stage_index) == 1
                            and gate_success_rate is not None
                            and float(gate_success_rate)
                            >= float(args.dense_stage_switch_success_rate)
                        ):
                            proposed_stage_index = 2
                previous_stage_index = int(dense_stage_index)
                dense_stage_index = _broadcast_dense_stage(
                    ctx=dist_ctx,
                    device=device,
                    stage_index=proposed_stage_index,
                )
                if int(dense_stage_index) != previous_stage_index:
                    dense_stage_switched = True
                    dense_stage_successes.clear()
                    dense_stage_episode_count = 0
                    dense_stage_success_count = 0
                    dense_stage_instruction_types = _apply_dense_stage_to_envs(
                        [env],
                        args,
                        dense_stage_index,
                    )
                    _log(
                        dist_ctx,
                        "[octo-cdpr] Dense curriculum validation gate switched to stage "
                        f"{dense_stage_index}: {', '.join(dense_stage_instruction_types)}",
                        progress=progress,
                    )
                last_validation_step = int(global_step)

            if done:
                completed_episode_rewards.append(float(episode_reward))
                completed_episode_successes.append(1.0 if episode_success else 0.0)
                completed_episode_count += 1
                completed_episode_success_count += 1 if episode_success else 0
                episode_metrics = _episode_metric_snapshot(
                    rewards=completed_episode_rewards,
                    successes=completed_episode_successes,
                    episode_count=completed_episode_count,
                    success_count=completed_episode_success_count,
                )
                dense_stage_metrics = _dense_stage_snapshot(
                    args=args,
                    stage_index=dense_stage_index,
                    stage_instruction_types=dense_stage_instruction_types,
                    successes=dense_stage_successes,
                    episode_count=dense_stage_episode_count,
                    success_count=dense_stage_success_count,
                )
                if dist_ctx.is_main:
                    mixed_curriculum_metrics = mixed_sampler.snapshot()
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "event": "episode_end",
                                    "global_step": int(global_step),
                                    "episode": int(episode),
                                    "episode_length": int(episode_length),
                                    "episode_reward": float(episode_reward),
                                    "success": episode_success,
                                    "terminated": bool(terminated),
                                    "truncated": bool(truncated),
                                    "instruction": str(instruction),
                                    "episode_dense_stage_index": int(episode_stage_index),
                                    "dense_stage_switched": bool(dense_stage_switched),
                                    **episode_metrics,
                                    **dense_stage_metrics,
                                    **mixed_curriculum_metrics,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    done_scalars = {
                        "rollout/episode_reward": float(episode_reward),
                        "rollout/success": 1.0 if episode_success else 0.0,
                    }
                    done_scalars.update(_episode_metric_scalars(episode_metrics))
                    done_scalars.update(_dense_stage_metric_scalars(dense_stage_metrics))
                    _log_scalars(writer, done_scalars, global_step)
                    _log_mixed_curriculum_scalars(writer, mixed_curriculum_metrics, global_step)
                episode += 1
                episode_reward = 0.0
                episode_length = 0
                mixed_sampler.record(_info_instruction_type(next_info, instruction), episode_success)
                episode_stage_index = int(dense_stage_index)
                with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                    obs, info = env.reset(options=mixed_sampler.reset_options(dense_stage_index))
                state = layout.flatten(obs)
                instruction = _safe_instruction(info)
                with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                    prior_chunk = _predict_prior_chunk(
                        runtime=runtime,
                        obs_adapter=obs_adapter,
                        action_spec=action_spec,
                        env=env,
                        obs=obs,
                        info=info,
                        instruction=instruction,
                    )
                action_chunk = trainer.select_chunk(state, prior_chunk)
                chunk_idx = 0
                continue

            obs = next_obs
            info = next_info
            state = next_state
            instruction = next_instruction
            chunk_idx += 1

        if dist_ctx.is_main:
            latest = trainer.save(global_step=global_step, args=args, latest=True)
            _log(dist_ctx, f"[octo-cdpr] Final latest checkpoint: {latest}", progress=progress)
    finally:
        if progress is not None:
            progress.close()
        if writer is not None:
            writer.close()
        if validation_env is not None:
            try:
                with _silence_output(bool(getattr(args, "progress_only", False))):
                    validation_env.close()
            except Exception:
                pass
        if env is not None:
            try:
                with _silence_output(bool(getattr(args, "progress_only", False)) or not dist_ctx.is_main):
                    env.close()
            except Exception:
                pass
        _destroy_distributed(dist_ctx)


if __name__ == "__main__":
    main()
