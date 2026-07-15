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
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - optional local dependency
    _TORCH_IMPORT_ERROR = exc
    torch = None
    nn = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from rl_vla_bootstrapping.policy.octo_cdpr_adapter import CDPRStateLayout
from rl_vla_bootstrapping.policy.octo_finetune_cdpr import (
    AdaptiveInstructionSampler,
    ReplayBuffer,
    ResidualTrainer,
    _apply_dense_stage_to_envs,
    _begin_validation_recording,
    _broadcast_dense_stage,
    _dense_curriculum_enabled,
    _dedupe_instruction_names,
    _dense_stage_instruction_types,
    _dense_stage_metric_scalars,
    _dense_stage_snapshot,
    _distributed_completed_successes,
    _finish_validation_recording,
    _format_step_progress,
    _info_instruction_type,
    _log_mixed_curriculum_scalars,
    _mixed_curriculum_enabled,
    _residual_trainer_parameter_summary,
    _save_training_validation_video,
    _summarize_validation_results,
    _validation_due,
    _validation_enabled,
    _validation_gate_success_rate,
    _validation_instruction_types,
    _validation_video_episode_slots,
    _write_validation_summary,
)
from rl_vla_bootstrapping.policy.smolvla_cdpr import (
    DEFAULT_SMOLVLA_CHECKPOINT,
    SmolVLAActionAdapterSpec,
    load_smolvla_runtime,
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


def _default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _require_torch() -> None:
    if torch is None or nn is None:
        detail = f" Original import error: {_TORCH_IMPORT_ERROR!r}" if _TORCH_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "SmolVLA CDPR adapter training requires PyTorch. Install it in the remote "
            f"`smolvla` environment before executing the RL stage.{detail}"
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


@dataclass
class EnvSlot:
    env: Any
    obs: dict[str, np.ndarray]
    info: dict[str, Any]
    state: np.ndarray
    instruction: str
    stage_index: int = 0
    prior_chunk: np.ndarray | None = None
    action_chunk: np.ndarray | None = None
    chunk_idx: int = 0
    episode: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0


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
            timeout_seconds = max(
                0, int(getattr(args, "distributed_timeout_seconds", 0) or 0)
            )
            init_kwargs: dict[str, Any] = {
                "backend": backend,
                "rank": rank,
                "world_size": world_size,
            }
            if timeout_seconds > 0:
                init_kwargs["timeout"] = timedelta(seconds=timeout_seconds)
            dist.init_process_group(**init_kwargs)

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


def _make_progress_bar(*, args: argparse.Namespace, ctx: DistributedContext, start_step: int) -> Any | None:
    if not bool(args.progress) or not ctx.is_main:
        return None
    if not sys.__stderr__.isatty():
        # Remote launchers pipe through tee; tqdm carriage-return redraws become noisy log text there.
        return None
    if tqdm is None:
        if not bool(args.progress_only):
            print("[smolvla-cdpr] tqdm is unavailable; falling back to status prints.", flush=True)
        return None
    max_train_steps = int(args.max_train_steps)
    total = max(0, max_train_steps - int(start_step))
    return tqdm(
        total=total,
        initial=0,
        desc=f"smolvla-cdpr {int(start_step)}->{max_train_steps}",
        unit="env-step",
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
    num_envs: int,
    global_step: int,
    start_step: int,
    max_train_steps: int,
) -> None:
    if progress is None:
        return
    run_total = max(0, int(max_train_steps) - int(start_step))
    run_done = max(0, int(global_step) - int(start_step))
    progress.set_postfix(
        step=f"{int(global_step)}/{int(max_train_steps)}",
        run=f"{run_done}/{run_total}",
        ep=int(episode),
        ep_len=int(episode_length),
        ep_reward=f"{float(episode_reward):+.3f}",
        reward=f"{float(reward):+.3f}",
        buffer=int(buffer_size),
        gpus=int(world_size),
        envs=int(num_envs),
        instr=str(instruction)[:24],
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight CDPR residual/readout adapter around frozen pretrained SmolVLA."
    )
    parser.add_argument("--config", default=None, help="Optional project config path for manifest provenance.")
    parser.add_argument("--base-checkpoint", default=DEFAULT_SMOLVLA_CHECKPOINT)
    parser.add_argument("--run-root-dir", default="runs")
    parser.add_argument("--run-id", default="smolvla_cdpr_rl")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=_default_device())
    _bool_arg(parser, "distributed", default=True, help_text="Enable torch.distributed under torchrun.")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--mixed-precision", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    _bool_arg(parser, "progress", default=True, help_text="Show a tqdm training progress bar on rank 0.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=10.0)
    _bool_arg(
        parser,
        "progress_only",
        default=False,
        help_text="Suppress CDPR wrapper/simulator chatter and leave only progress plus trainer logs.",
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
    parser.add_argument("--num-envs-per-rank", type=int, default=2)

    parser.add_argument("--max-env-steps", type=int, default=64)
    parser.add_argument("--max-train-steps", type=int, default=250000)
    parser.add_argument("--action-step-xyz", type=float, default=0.015)
    parser.add_argument("--action-step-yaw", type=float, default=0.08)
    parser.add_argument("--action-step-gripper", type=float, default=0.05)
    parser.add_argument("--hold-steps", type=int, default=6)
    parser.add_argument("--move-distance", type=float, default=0.40)
    parser.add_argument("--lift-distance", type=float, default=0.10)
    _bool_arg(parser, "lock_non_commanded_axes", default=True, help_text="Forwarded to the CDPR env.")
    parser.add_argument("--lock-non-commanded-axes-threshold", type=float, default=0.05)
    _bool_arg(parser, "randomize_ee_start", default=True, help_text="Forwarded to the CDPR env.")
    parser.add_argument("--ee-start-x-bounds", nargs=2, type=float, default=(-0.20, 0.20))
    parser.add_argument("--ee-start-y-bounds", nargs=2, type=float, default=(-0.20, 0.20))
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
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--image-feature-keys", nargs="+", default=None)
    _bool_arg(parser, "include_wrist", default=True, help_text="Include the CDPR wrist camera.")
    _bool_arg(parser, "include_aux_camera", default=True, help_text="Fill SmolVLA's third camera input.")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--smolvla-action-indices", nargs=5, type=int, default=None)
    parser.add_argument("--smolvla-action-normalization", choices=("tanh", "clip", "none"), default="tanh")

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--actor-lr", type=float, default=3.0e-4)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--update-after", type=int, default=1024)
    parser.add_argument("--updates-per-step", type=int, default=1)
    _bool_arg(
        parser,
        "materialize_optimizer_state",
        default=False,
        help_text="Allocate AdamW optimizer state immediately so high-memory GPU profiles are visible before the first update.",
    )
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


class SmolVLAResidualTrainer(ResidualTrainer):
    def materialize_optimizer_state(self) -> None:
        for optimizer in (self.actor_optim, self.critic_optim):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue
                    state = optimizer.state[param]
                    if state:
                        continue
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def save(self, *, global_step: int, args: argparse.Namespace, latest: bool = False) -> Path:
        payload = {
            "policy_type": "smolvla_cdpr",
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
            output_path = step_dir / "smolvla_cdpr_adapter.pt"
        torch.save(payload, output_path)
        if not latest:
            torch.save(payload, self.run_dir / "latest.pt")
        return output_path


def _resolve_checkpoint(raw: str | Path) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.is_file():
        return path
    for name in ("smolvla_cdpr_adapter.pt", "latest.pt"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find a SmolVLA CDPR checkpoint in {path}")


def _safe_instruction(info: dict[str, Any]) -> str:
    return str(info.get("language_instruction") or info.get("instruction_type") or "move left")


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


def _gpu_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    idx = int(device.index or torch.cuda.current_device())
    return {
        "gpu_allocated_gb": float(torch.cuda.memory_allocated(idx) / (1024**3)),
        "gpu_reserved_gb": float(torch.cuda.memory_reserved(idx) / (1024**3)),
        "gpu_max_allocated_gb": float(torch.cuda.max_memory_allocated(idx) / (1024**3)),
    }


def _torch_pretrained_parameter_summary(policy: Any) -> dict[str, Any]:
    parameters = getattr(policy, "parameters", None)
    if not callable(parameters):
        return {"parameter_count_available": False}
    total = 0
    trainable = 0
    for param in parameters():
        count = int(param.numel())
        total += count
        if bool(param.requires_grad):
            trainable += count
    return {
        "parameter_count_available": True,
        "parameter_count": int(total),
        "trainable_parameter_count": int(trainable),
    }


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


def _select_chunks(
    trainer: SmolVLAResidualTrainer,
    *,
    states: Sequence[np.ndarray],
    priors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    state_t = torch.as_tensor(np.stack(states, axis=0), dtype=torch.float32, device=device)
    prior_t = torch.as_tensor(priors, dtype=torch.float32, device=device)
    with torch.no_grad():
        chunks = trainer.actor(state_t, prior_t).detach().to(dtype=torch.float32).cpu().numpy()
    return np.clip(chunks, -1.0, 1.0).astype(np.float32, copy=False)


def _refresh_policy_chunks(
    *,
    runtime: Any,
    trainer: SmolVLAResidualTrainer,
    slots: list[EnvSlot],
    indices: Sequence[int],
    device: torch.device,
    progress_only: bool,
) -> None:
    if not indices:
        return
    selected = [slots[idx] for idx in indices]
    with _silence_output(bool(progress_only)):
        priors = runtime.sample_cdpr_chunks_from_envs(
            envs=[slot.env for slot in selected],
            observations=[slot.obs for slot in selected],
            infos=[slot.info for slot in selected],
            instructions=[slot.instruction for slot in selected],
        )
    chunks = _select_chunks(
        trainer,
        states=[slot.state for slot in selected],
        priors=priors,
        device=device,
    )
    for local_idx, slot_idx in enumerate(indices):
        slots[slot_idx].prior_chunk = priors[local_idx]
        slots[slot_idx].action_chunk = chunks[local_idx]
        slots[slot_idx].chunk_idx = 0


def _reset_slot(
    *,
    slot: EnvSlot,
    layout: CDPRStateLayout,
    seed: int | None = None,
    reset_options: dict[str, Any] | None = None,
    progress_only: bool,
    stage_index: int | None = None,
) -> None:
    with _silence_output(bool(progress_only)):
        obs, info = slot.env.reset(seed=seed, options=dict(reset_options or {}))
    slot.obs = obs
    slot.info = dict(info)
    slot.state = layout.flatten(obs)
    slot.instruction = _safe_instruction(slot.info)
    slot.prior_chunk = None
    slot.action_chunk = None
    slot.chunk_idx = 0
    slot.episode_reward = 0.0
    slot.episode_length = 0
    if stage_index is not None:
        slot.stage_index = int(stage_index)


def _select_smolvla_validation_chunk(
    actor: nn.Module,
    *,
    device: torch.device,
    state: np.ndarray,
    prior_chunk: np.ndarray,
) -> np.ndarray:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    prior_t = torch.as_tensor(prior_chunk, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_chunk = actor(state_t, prior_t)[0].detach().to(dtype=torch.float32).cpu().numpy()
    return np.clip(action_chunk, -1.0, 1.0).astype(np.float32, copy=False)


def _run_smolvla_distinct_validation(
    *,
    validation_env: Any,
    runtime: Any,
    trainer: SmolVLAResidualTrainer,
    layout: CDPRStateLayout,
    args: argparse.Namespace,
    global_step: int,
    stage_index: int,
    instruction_types: Sequence[str],
    reset_options_by_instruction: Mapping[str, Mapping[str, Any]] | None = None,
    episodes_per_instruction: int | None = None,
    record_videos: bool = True,
) -> dict[str, Any]:
    episode_count = max(
        1,
        int(
            args.validation_episodes_per_instruction
            if episodes_per_instruction is None
            else episodes_per_instruction
        ),
    )
    replan_every = max(1, min(int(args.replan_every), int(args.chunk_size)))
    actor = trainer._unwrap(trainer.actor)
    was_training = bool(actor.training)
    previous_instruction_types = getattr(validation_env, "instruction_types", None)
    results: list[dict[str, Any]] = []
    saved_videos: list[dict[str, Any]] = []
    video_limit = (
        max(0, int(getattr(args, "validation_video_count", 0))) if record_videos else 0
    )
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
                reset_options = dict(
                    (reset_options_by_instruction or {}).get(str(instruction_type), {})
                )
                reset_options.setdefault("instruction_type", str(instruction_type))
                with _silence_output(True):
                    obs, info = validation_env.reset(seed=seed, options=reset_options)
                should_record_video = (int(instruction_index), int(episode_index)) in video_slots
                recording_state = _begin_validation_recording(validation_env, should_record_video)
                video_summary: dict[str, Any] | None = None
                try:
                    state = layout.flatten(obs)
                    instruction = _safe_instruction(info)
                    with _silence_output(True):
                        priors = runtime.sample_cdpr_chunks_from_envs(
                            envs=[validation_env],
                            observations=[obs],
                            infos=[dict(info)],
                            instructions=[instruction],
                        )
                    prior_chunk = np.asarray(priors[0], dtype=np.float32)
                    action_chunk = _select_smolvla_validation_chunk(
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
                                priors = runtime.sample_cdpr_chunks_from_envs(
                                    envs=[validation_env],
                                    observations=[obs],
                                    infos=[dict(final_info)],
                                    instructions=[instruction],
                                )
                            prior_chunk = np.asarray(priors[0], dtype=np.float32)
                            action_chunk = _select_smolvla_validation_chunk(
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
                        "curriculum_shell": int(info.get("curriculum_shell", -1)),
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _require_torch()
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

    startup_t0 = time.perf_counter()
    _log(
        dist_ctx,
        f"[smolvla-cdpr] Loading SmolVLA checkpoint: {args.base_checkpoint} "
        f"(rank={dist_ctx.rank}, world_size={dist_ctx.world_size}, device={dist_ctx.device}, "
        f"mixed_precision={args.mixed_precision})",
    )
    load_t0 = time.perf_counter()
    with _silence_output(not dist_ctx.is_main):
        runtime = load_smolvla_runtime(
            checkpoint=str(args.base_checkpoint),
            device=str(dist_ctx.device),
            mixed_precision=str(args.mixed_precision),
            image_size=int(args.image_size),
            state_dim=int(args.state_dim),
            image_feature_keys=None if args.image_feature_keys is None else tuple(args.image_feature_keys),
            include_wrist=bool(args.include_wrist),
            include_aux_camera=bool(args.include_aux_camera),
            chunk_size=int(args.chunk_size),
            action_dim=int(args.action_dim),
            action_indices=None
            if args.smolvla_action_indices is None
            else tuple(int(v) for v in args.smolvla_action_indices),
            action_normalization=str(args.smolvla_action_normalization),
        )
    _log(
        dist_ctx,
        f"[smolvla-cdpr] Loaded SmolVLA in {time.perf_counter() - load_t0:.1f}s; "
        f"{runtime.device_summary()}; distributed_world_size={dist_ctx.world_size}",
    )
    _log(
        dist_ctx,
        "[smolvla-cdpr] Camera inputs: "
        f"{runtime.obs_spec.image_feature_keys[0]}=overview, "
        f"{runtime.obs_spec.image_feature_keys[1] if len(runtime.obs_spec.image_feature_keys) > 1 else 'camera2'}=wrist, "
        f"{runtime.obs_spec.image_feature_keys[2] if len(runtime.obs_spec.image_feature_keys) > 2 else 'camera3'}=wrist/aux fallback",
    )

    slots: list[EnvSlot] = []
    validation_env = None
    progress = None
    try:
        env_count = max(1, int(args.num_envs_per_rank))
        _log(dist_ctx, f"[smolvla-cdpr] Building {env_count} CDPR env(s) on rank {dist_ctx.rank}...")
        env_t0 = time.perf_counter()
        for env_idx in range(env_count):
            seed = int(rollout_seed) + env_idx * 997
            with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                env = _build_env(args, seed=seed)
                if dense_curriculum_active:
                    dense_stage_instruction_types = _apply_dense_stage_to_envs(
                        [env],
                        args,
                        dense_stage_index,
                    )
                obs, info = env.reset(
                    seed=seed,
                    options=mixed_sampler.reset_options(dense_stage_index),
                )
            layout = CDPRStateLayout.from_observation(obs) if env_idx == 0 else layout
            slots.append(
                EnvSlot(
                    env=env,
                    obs=obs,
                    info=dict(info),
                    state=layout.flatten(obs),
                    instruction=_safe_instruction(dict(info)),
                    stage_index=int(dense_stage_index),
                )
            )
        if dense_curriculum_active:
            _log(
                dist_ctx,
                "[smolvla-cdpr] Dense curriculum stage 1 active: "
                f"{', '.join(dense_stage_instruction_types)}",
            )
        _log(dist_ctx, f"[smolvla-cdpr] Built env batch in {time.perf_counter() - env_t0:.1f}s")
        if dist_ctx.is_main and _validation_enabled(args):
            _log(
                dist_ctx,
                "[smolvla-cdpr] Building held-out validation environment "
                f"(every {int(args.validation_every_steps)} steps, "
                f"{int(args.validation_episodes_per_instruction)} episode(s)/instruction)...",
            )
            with _silence_output(bool(args.progress_only)):
                validation_env = _build_env(args, seed=int(args.validation_seed))

        device = torch.device(args.device)
        trainer = SmolVLAResidualTrainer(
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
            _log(dist_ctx, f"[smolvla-cdpr] Resumed adapter checkpoint {resume_path} at step {start_step}")
        elif bool(args.materialize_optimizer_state):
            trainer.materialize_optimizer_state()
            _log(
                dist_ctx,
                f"[smolvla-cdpr] Materialized AdamW optimizer state on {device}; "
                f"gpu_metrics={_gpu_metrics(device)}",
            )

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
            "policy_type": "smolvla_cdpr",
            "base_checkpoint": str(args.base_checkpoint),
            "run_dir": run_dir.as_posix(),
            "config": str(args.config or ""),
            "action_keys": ["x", "y", "z", "yaw", "gripper"],
            "chunk_size": int(args.chunk_size),
            "native_smolvla_chunk_size": int(getattr(getattr(runtime.policy, "config", None), "chunk_size", 50)),
            "prior_action_adapter": {
                "source": "SmolVLA action chunk",
                "target": "CDPR normalized 5D [x, y, z, yaw, gripper]",
                "default_source_indices_for_6plus_d": [0, 1, 2, 3, "last"],
                "configured_source_indices": (
                    list(args.smolvla_action_indices)
                    if args.smolvla_action_indices is not None
                    else [0, 1, 2, 3, "last"]
                ),
                "normalization": str(args.smolvla_action_normalization),
            },
            "image_feature_keys": list(runtime.obs_spec.image_feature_keys),
            "camera_mapping": {
                "camera1": "CDPR overview camera",
                "camera2": "CDPR wrist/end-effector camera",
                "camera3": "aux image when provided, otherwise CDPR wrist/end-effector camera duplicate",
            },
            "trainable_surface": "torch_residual_chunk_head_and_q_critics",
            "frozen_smolvla": True,
            "parameter_training": _residual_trainer_parameter_summary(
                trainer,
                pretrained_prior_name="SmolVLA",
                base_checkpoint=str(args.base_checkpoint),
                pretrained_prior_parameters=_torch_pretrained_parameter_summary(runtime.policy),
            ),
            "materialized_optimizer_state": bool(args.materialize_optimizer_state),
            "online_dense_rl": True,
            "num_envs_per_rank": int(env_count),
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
            "unavoidable_cpu_work": [
                "MuJoCo simulation stepping and camera readback",
                "short language tokenization cache misses",
            ],
        }
        if dist_ctx.is_main:
            _write_json(run_dir / "smolvla_manifest.json", manifest)

        global_step = int(start_step)
        last_metrics: dict[str, float] = {}
        replan_every = max(1, min(int(args.replan_every), int(args.chunk_size)))
        completed_episode_rewards: deque[float] = deque(maxlen=metrics_window)
        completed_episode_successes: deque[float] = deque(maxlen=metrics_window)
        completed_episode_count = 0
        completed_episode_success_count = 0

        _log(
            dist_ctx,
            "[smolvla-cdpr] Sampling first batched SmolVLA prior action chunks...",
        )
        prior_t0 = time.perf_counter()
        _refresh_policy_chunks(
            runtime=runtime,
            trainer=trainer,
            slots=slots,
            indices=list(range(len(slots))),
            device=device,
            progress_only=bool(args.progress_only) or not dist_ctx.is_main,
        )
        _log(
            dist_ctx,
            f"[smolvla-cdpr] First prior batch ready in {time.perf_counter() - prior_t0:.1f}s; "
            f"startup total {time.perf_counter() - startup_t0:.1f}s",
        )
        progress = _make_progress_bar(args=args, ctx=dist_ctx, start_step=start_step)
        status_start_t = time.perf_counter()
        last_validation_step = int(start_step)

        while global_step < int(args.max_train_steps):
            need_replan = [
                idx
                for idx, slot in enumerate(slots)
                if slot.prior_chunk is None or slot.action_chunk is None or slot.chunk_idx >= replan_every
            ]
            _refresh_policy_chunks(
                runtime=runtime,
                trainer=trainer,
                slots=slots,
                indices=need_replan,
                device=device,
                progress_only=bool(args.progress_only) or not dist_ctx.is_main,
            )

            for slot_idx, slot in enumerate(slots):
                if global_step >= int(args.max_train_steps):
                    break
                assert slot.prior_chunk is not None
                assert slot.action_chunk is not None

                action_index = int(slot.chunk_idx)
                action = np.asarray(slot.action_chunk[action_index], dtype=np.float32).reshape(int(args.action_dim))
                noise_std = _exploration_noise(args, global_step)
                if noise_std > 0.0:
                    action = action + np.random.normal(0.0, noise_std, size=action.shape).astype(np.float32)
                action = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)

                with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                    next_obs, reward, terminated, truncated, next_info = slot.env.step(action)
                next_state = layout.flatten(next_obs)
                done = bool(terminated or truncated)
                next_idx = min(action_index + 1, int(args.chunk_size) - 1)

                buffer.add(
                    state=slot.state,
                    prior=slot.prior_chunk,
                    action_index=action_index,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    next_prior=slot.prior_chunk,
                    next_action_index=next_idx,
                    done=done,
                )

                global_step += 1
                slot.episode_length += 1
                slot.episode_reward += float(reward)
                episode_success = bool(next_info.get("success", False)) if done else False
                dense_stage_switched = False
                if dense_curriculum_active:
                    count_for_dense_stage = done and int(slot.stage_index) == int(dense_stage_index)
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
                            [item.env for item in slots],
                            args,
                            dense_stage_index,
                        )
                        _log(
                            dist_ctx,
                            "[smolvla-cdpr] Dense curriculum switched to stage "
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
                        "rank": int(dist_ctx.rank),
                        "slot": int(slot_idx),
                        "episode": int(slot.episode),
                        "episode_length": int(slot.episode_length),
                        "episode_reward_running": float(slot.episode_reward),
                        "reward": float(reward),
                        "done": bool(done),
                        "buffer_size": int(buffer.size),
                        "instruction": str(slot.instruction),
                        "episode_dense_stage_index": int(slot.stage_index),
                        "noise_std": float(noise_std),
                        **episode_metrics,
                        **dense_stage_metrics,
                        **mixed_curriculum_metrics,
                        **last_metrics,
                        **_gpu_metrics(device),
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    rollout_scalars = {
                        "rollout/reward": float(reward),
                        "rollout/episode_reward_running": float(slot.episode_reward),
                        "rollout/noise_std": float(noise_std),
                        "rollout/buffer_size": float(buffer.size),
                        **{f"gpu/{k}": v for k, v in _gpu_metrics(device).items()},
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
                            episode=slot.episode,
                            episode_length=slot.episode_length,
                            episode_reward=slot.episode_reward,
                            reward=float(reward),
                            buffer_size=buffer.size,
                            instruction=slot.instruction,
                            world_size=dist_ctx.world_size,
                            num_envs=len(slots),
                            global_step=global_step,
                            start_step=int(start_step),
                            max_train_steps=int(args.max_train_steps),
                        )
                elif dist_ctx.is_main and global_step % max(1, int(args.status_every_steps)) == 0:
                    _log(
                        dist_ctx,
                        f"[smolvla-cdpr] step={global_step:07d} slot={slot_idx} "
                        f"episode={slot.episode:05d} ep_len={slot.episode_length:03d} "
                        f"reward_running={slot.episode_reward:+.3f} last_reward={float(reward):+.3f} "
                        f"buffer={buffer.size} instruction={slot.instruction}",
                    )
                    _log(
                        dist_ctx,
                        "[smolvla-cdpr] "
                        + _format_step_progress(
                            global_step=global_step,
                            max_train_steps=int(args.max_train_steps),
                            start_step=int(start_step),
                            elapsed_seconds=time.perf_counter() - status_start_t,
                        ),
                    )

                if dist_ctx.is_main and global_step % max(1, int(args.save_every_steps)) == 0:
                    checkpoint = trainer.save(global_step=global_step, args=args, latest=False)
                    _log(dist_ctx, f"[smolvla-cdpr] Saved checkpoint: {checkpoint}", progress=progress)

                if _validation_due(args, global_step=global_step, last_validation_step=last_validation_step):
                    proposed_stage_index = int(dense_stage_index)
                    if dist_ctx.is_main:
                        validation_instruction_types = _validation_instruction_types(
                            args,
                            dense_curriculum_active=dense_curriculum_active,
                            dense_stage_index=dense_stage_index,
                        )
                        if validation_env is not None and validation_instruction_types:
                            validation_summary = _run_smolvla_distinct_validation(
                                validation_env=validation_env,
                                runtime=runtime,
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
                                "[smolvla-cdpr] Held-out validation "
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
                            [item.env for item in slots],
                            args,
                            dense_stage_index,
                        )
                        _log(
                            dist_ctx,
                            "[smolvla-cdpr] Dense curriculum validation gate switched to stage "
                            f"{dense_stage_index}: {', '.join(dense_stage_instruction_types)}",
                            progress=progress,
                        )
                    last_validation_step = int(global_step)

                if done:
                    completed_episode_rewards.append(float(slot.episode_reward))
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
                                        "rank": int(dist_ctx.rank),
                                        "slot": int(slot_idx),
                                        "episode": int(slot.episode),
                                        "episode_length": int(slot.episode_length),
                                        "episode_reward": float(slot.episode_reward),
                                        "success": episode_success,
                                        "terminated": bool(terminated),
                                        "truncated": bool(truncated),
                                        "instruction": str(slot.instruction),
                                        "episode_dense_stage_index": int(slot.stage_index),
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
                            "rollout/episode_reward": float(slot.episode_reward),
                            "rollout/success": 1.0 if episode_success else 0.0,
                        }
                        done_scalars.update(_episode_metric_scalars(episode_metrics))
                        done_scalars.update(_dense_stage_metric_scalars(dense_stage_metrics))
                        _log_scalars(writer, done_scalars, global_step)
                        _log_mixed_curriculum_scalars(writer, mixed_curriculum_metrics, global_step)
                    slot.episode += 1
                    mixed_sampler.record(_info_instruction_type(next_info, slot.instruction), episode_success)
                    _reset_slot(
                        slot=slot,
                        layout=layout,
                        seed=None,
                        reset_options=mixed_sampler.reset_options(dense_stage_index),
                        progress_only=bool(args.progress_only) or not dist_ctx.is_main,
                        stage_index=dense_stage_index,
                    )
                    continue

                slot.obs = next_obs
                slot.info = dict(next_info)
                slot.state = next_state
                slot.instruction = _safe_instruction(slot.info)
                slot.chunk_idx += 1

        if dist_ctx.is_main:
            latest = trainer.save(global_step=global_step, args=args, latest=True)
            _log(dist_ctx, f"[smolvla-cdpr] Final latest checkpoint: {latest}", progress=progress)
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
        for slot in slots:
            try:
                with _silence_output(bool(getattr(args, "progress_only", False)) or not dist_ctx.is_main):
                    slot.env.close()
            except Exception:
                pass
        _destroy_distributed(dist_ctx)


if __name__ == "__main__":
    main()
