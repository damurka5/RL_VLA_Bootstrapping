#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import deque
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

from rl_vla_bootstrapping.policy.octo_cdpr_adapter import CDPRStateLayout
from rl_vla_bootstrapping.lchol.smolvla_complex import SmolVLAComplexRuntime
from rl_vla_bootstrapping.policy.octo_finetune_cdpr import (
    ResidualChunkActor,
    _apply_dense_stage_to_envs,
    _dedupe_instruction_names,
    _dense_curriculum_enabled,
    _dense_stage_instruction_types,
    _dense_stage_metric_scalars,
    _dense_stage_snapshot,
    _distributed_completed_successes,
    _format_step_progress,
    _info_instruction_type,
    _log_mixed_curriculum_scalars,
    _mixed_curriculum_enabled,
    _validation_due,
    _validation_enabled,
    _validation_instruction_types,
    _write_validation_summary,
    AdaptiveInstructionSampler,
)
from rl_vla_bootstrapping.policy.smolvla_cdpr import DEFAULT_SMOLVLA_CHECKPOINT, load_smolvla_runtime
from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import (
    DistributedContext,
    EnvSlot,
    _bool_arg,
    _build_env,
    _configure_distributed,
    _default_device,
    _destroy_distributed,
    _episode_metric_scalars,
    _episode_metric_snapshot,
    _gpu_metrics,
    _log,
    _log_scalars,
    _make_progress_bar,
    _make_run_dir,
    _progress_postfix,
    _rank_seed,
    _require_torch,
    _run_smolvla_distinct_validation,
    _safe_instruction,
    _set_quiet_env,
    _set_seed,
    _silence_output,
    _torch_pretrained_parameter_summary,
    _write_json,
)


if nn is not None:

    class SmolVLAGRPOPolicy(nn.Module):
        def __init__(
            self,
            *,
            state_dim: int,
            chunk_size: int,
            action_dim: int,
            hidden_dim: int,
            residual_scale: float,
            init_log_std: float,
            min_log_std: float,
            max_log_std: float,
        ) -> None:
            super().__init__()
            self.chunk_size = int(chunk_size)
            self.action_dim = int(action_dim)
            self.min_log_std = float(min_log_std)
            self.max_log_std = float(max_log_std)
            self.actor = ResidualChunkActor(
                state_dim=int(state_dim),
                chunk_size=int(chunk_size),
                action_dim=int(action_dim),
                hidden_dim=int(hidden_dim),
                residual_scale=float(residual_scale),
            )
            self.log_std = nn.Parameter(
                torch.full((int(chunk_size), int(action_dim)), float(init_log_std), dtype=torch.float32)
            )

        def forward(self, state: torch.Tensor, prior_chunk: torch.Tensor) -> torch.Tensor:
            return self.actor(state, prior_chunk)

        def clamped_log_std(self) -> torch.Tensor:
            return self.log_std.clamp(float(self.min_log_std), float(self.max_log_std))

else:

    class SmolVLAGRPOPolicy:  # pragma: no cover - dependency guard
        def __init__(self, *args, **kwargs):
            _require_torch()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SmolVLA CDPR residual policy with grouped relative policy optimization."
    )
    parser.add_argument("--config", default=None, help="Optional project config path for manifest provenance.")
    parser.add_argument("--base-checkpoint", default=DEFAULT_SMOLVLA_CHECKPOINT)
    parser.add_argument("--run-root-dir", default="runs")
    parser.add_argument("--run-id", default="smolvla_cdpr_grpo")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=_default_device())
    _bool_arg(parser, "distributed", default=True, help_text="Enable torch.distributed under torchrun.")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--mixed-precision", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    _bool_arg(parser, "progress", default=True, help_text="Show a tqdm training progress bar on rank 0.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=10.0)
    _bool_arg(parser, "progress_only", default=False, help_text="Suppress CDPR wrapper/simulator chatter.")

    parser.add_argument("--catalog-path", default=None)
    parser.add_argument("--cdpr-dataset-root", default=None)
    parser.add_argument("--cdpr-mujoco-root", default=None)
    parser.add_argument("--desk-textures-dir", default=None)
    parser.add_argument("--desk-geom-regex", default=r"(table|desk|workbench|counter|surface)")
    parser.add_argument("--desk-texrepeat", nargs=2, type=int, default=(20, 20))
    parser.add_argument("--allowed-objects", nargs="+", default=None)
    parser.add_argument("--instruction-types", nargs="+", default=None)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--num-envs-per-rank", "--num-parallel-envs", dest="num_envs_per_rank", type=int, default=2)

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
    _bool_arg(parser, "reuse_existing_wrapper_variants", default=True, help_text="Prefer compatible wrapper variants.")

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--image-feature-keys", nargs="+", default=None)
    _bool_arg(parser, "include_wrist", default=True, help_text="Include the CDPR wrist camera.")
    _bool_arg(parser, "include_aux_camera", default=True, help_text="Fill SmolVLA's third camera input.")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--replan-every", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--smolvla-action-indices", nargs=5, type=int, default=None)
    parser.add_argument("--smolvla-action-normalization", choices=("tanh", "clip", "none"), default="tanh")

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--residual-scale", type=float, default=0.30)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--adam-eps", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-log-std", type=float, default=-1.2)
    parser.add_argument("--min-log-std", type=float, default=-5.0)
    parser.add_argument("--max-log-std", type=float, default=1.0)
    parser.add_argument("--clip-range", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--action-l2", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grpo-group-size", type=int, default=2)
    parser.add_argument("--grpo-group-selection", choices=("uniform", "best", "softmax"), default="uniform")
    _bool_arg(parser, "grpo_normalize_group_advantage", default=True, help_text="Normalize rewards inside each group.")
    parser.add_argument("--grpo-clip-advantage-abs", type=float, default=6.0)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=100000)
    parser.add_argument("--log-every-steps", type=int, default=250)
    parser.add_argument("--status-every-steps", type=int, default=500)
    parser.add_argument("--metrics-window-episodes", type=int, default=100)

    parser.add_argument(
        "--complex-training-approach",
        choices=("none", "reverse_frontier", "lchol_hindsight"),
        default="none",
    )
    parser.add_argument("--reverse-frontier-promotion-success", type=float, default=0.80)
    parser.add_argument("--reverse-frontier-demotion-success", type=float, default=-1.0)
    parser.add_argument("--reverse-frontier-validation-episodes", type=int, default=50)
    parser.add_argument("--reverse-frontier-min-train-updates", type=int, default=1)
    parser.add_argument("--reverse-frontier-saturation-abort-threshold", type=float, default=1.01)
    parser.add_argument("--reverse-frontier-sample-probability", type=float, default=0.80)
    parser.add_argument("--reverse-frontier-rehearsal-probability", type=float, default=0.20)
    parser.add_argument("--lchol-hindsight-replay-capacity", type=int, default=20000)
    parser.add_argument("--lchol-hindsight-replay-ratio", type=float, default=0.25)
    parser.add_argument("--lchol-hindsight-prefix-max-steps", type=int, default=16)
    parser.add_argument("--lchol-hindsight-bc-coef", type=float, default=0.20)
    parser.add_argument("--put-stage-promotion-success", type=float, default=0.80)
    parser.add_argument("--put-stage-min-episodes", type=int, default=30)
    parser.add_argument("--put-stage-history-episodes", type=int, default=50)

    parser.add_argument("--dense-stage-one-instruction-types", nargs="+", default=None)
    parser.add_argument("--dense-stage-two-instruction-types", nargs="+", default=None)
    parser.add_argument("--dense-stage-switch-success-rate", type=float, default=0.50)
    parser.add_argument("--dense-stage-min-episodes", type=int, default=0)
    parser.add_argument("--dense-stage-gate-metric", choices=("rollout", "validation"), default="rollout")
    parser.add_argument("--dense-stage-gate-aggregation", choices=("mean", "worst"), default="mean")
    _bool_arg(parser, "mixed_curriculum_enabled", default=False, help_text="Enable mixed curriculum after promotion.")
    _bool_arg(parser, "mixed_curriculum_adaptive", default=True, help_text="Oversample weaker instructions.")
    parser.add_argument("--mixed-curriculum-success-target", type=float, default=0.80)
    parser.add_argument("--mixed-curriculum-history-episodes", type=int, default=100)
    parser.add_argument("--mixed-curriculum-min-gap-weight", type=float, default=0.05)
    parser.add_argument("--mixed-curriculum-min-prob", type=float, default=0.03)
    parser.add_argument("--mixed-curriculum-max-prob", type=float, default=0.35)
    parser.add_argument("--mixed-curriculum-rehearsal-min-prob", type=float, default=0.30)
    parser.add_argument("--mixed-curriculum-move-to-object-min-prob", type=float, default=0.25)
    parser.add_argument("--mixed-curriculum-manipulation-min-prob", type=float, default=0.25)
    parser.add_argument("--near-success-instruction-types", nargs="+", default=None)

    parser.add_argument("--validation-every-steps", type=int, default=0)
    parser.add_argument("--validation-episodes-per-instruction", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=1000000)
    parser.add_argument("--validation-video-count", type=int, default=3)
    parser.add_argument("--validation-video-fps", type=float, default=10.0)
    parser.add_argument("--comparison-validation-episodes-per-instruction", type=int, default=10)

    args = parser.parse_args(argv)
    if args.batch_size is not None and args.minibatch_size is None:
        args.minibatch_size = int(args.batch_size)
    if args.minibatch_size is None:
        args.minibatch_size = 256
    if args.microbatch_size is None:
        args.microbatch_size = int(args.minibatch_size)
    args.grpo_group_size = max(2, int(args.grpo_group_size))
    args.replan_every = max(1, min(int(args.replan_every), int(args.chunk_size)))
    return args


def _normal_log_prob(action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2.0 * log_std)
    return (-0.5 * (((action - mean).pow(2) / var) + 2.0 * log_std + math.log(2.0 * math.pi))).sum(dim=-1)


def _normal_entropy(log_std: torch.Tensor) -> torch.Tensor:
    return (log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))).sum(dim=-1)


def _gather_chunk_values(values: torch.Tensor, action_index: torch.Tensor) -> torch.Tensor:
    idx = action_index.reshape(-1).long().clamp(0, values.shape[1] - 1)
    return values[torch.arange(values.shape[0], device=values.device), idx]


class SmolVLAGRPOTrainer:
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
        _require_torch()
        self.args = args
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.run_dir = Path(run_dir)
        self.device = device
        self.distributed = distributed or DistributedContext(device=str(device))
        policy = SmolVLAGRPOPolicy(
            state_dim=int(state_dim),
            chunk_size=int(chunk_size),
            action_dim=int(action_dim),
            hidden_dim=int(args.hidden_dim),
            residual_scale=float(args.residual_scale),
            init_log_std=float(args.init_log_std),
            min_log_std=float(args.min_log_std),
            max_log_std=float(args.max_log_std),
        ).to(device)
        if self.distributed.enabled:
            from torch.nn.parallel import DistributedDataParallel as DDP

            ddp_kwargs: dict[str, Any] = {}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [int(device.index or 0)]
                ddp_kwargs["output_device"] = int(device.index or 0)
            policy = DDP(policy, **ddp_kwargs)
        self.actor = policy
        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=float(args.learning_rate),
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
        )
        self.gradient_step = 0
        self.loaded_extra_state: dict[str, Any] = {}
        self.bootstrap_source = "fresh_grpo"

    @staticmethod
    def _unwrap(module: nn.Module) -> nn.Module:
        return module.module if hasattr(module, "module") else module

    def _mean_and_log_std(
        self,
        state: torch.Tensor,
        prior: torch.Tensor,
        action_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_chunk = self.actor(state, prior)
        mean = _gather_chunk_values(mean_chunk, action_index)
        base = self._unwrap(self.actor)
        log_std = base.clamped_log_std()[action_index.reshape(-1).long().clamp(0, self.chunk_size - 1)]
        return mean, log_std

    def sample_action_group(
        self,
        *,
        state: np.ndarray,
        prior: np.ndarray,
        action_index: int,
        group_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        prior_t = torch.as_tensor(prior, dtype=torch.float32, device=self.device).unsqueeze(0)
        idx_t = torch.full((1,), int(action_index), dtype=torch.long, device=self.device)
        with torch.no_grad():
            mean, log_std = self._mean_and_log_std(state_t, prior_t, idx_t)
            std = torch.exp(log_std)
            mean_g = mean.expand(int(group_size), -1)
            log_std_g = log_std.expand(int(group_size), -1)
            eps = torch.randn_like(mean_g)
            actions = torch.clamp(mean_g + eps * std.expand_as(mean_g), -1.0, 1.0)
            log_probs = _normal_log_prob(actions, mean_g, log_std_g)
        return (
            actions.detach().to(dtype=torch.float32).cpu().numpy(),
            log_probs.detach().to(dtype=torch.float32).cpu().numpy(),
            mean.detach().to(dtype=torch.float32).cpu().numpy()[0],
        )

    def update(
        self,
        records: list[dict[str, Any]],
        *,
        hindsight_records: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, float]:
        if not records:
            return {}
        states = torch.as_tensor(np.stack([row["state"] for row in records]), dtype=torch.float32, device=self.device)
        priors = torch.as_tensor(np.stack([row["prior"] for row in records]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([row["action"] for row in records]), dtype=torch.float32, device=self.device)
        action_indices = torch.as_tensor([int(row["action_index"]) for row in records], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor([float(row["old_log_prob"]) for row in records], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor([float(row["advantage"]) for row in records], dtype=torch.float32, device=self.device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)

        n_items = int(states.shape[0])
        minibatch = max(1, min(int(self.args.minibatch_size), n_items))
        microbatch = max(1, min(int(self.args.microbatch_size), minibatch))
        policy_losses: list[float] = []
        entropy_values: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []

        for _epoch in range(max(1, int(self.args.ppo_epochs))):
            order = torch.randperm(n_items, device=self.device)
            for start in range(0, n_items, minibatch):
                mb_idx = order[start : start + minibatch]
                if mb_idx.numel() == 0:
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                micro_count = int(math.ceil(float(mb_idx.numel()) / float(microbatch)))
                for micro_start in range(0, int(mb_idx.numel()), microbatch):
                    idx = mb_idx[micro_start : micro_start + microbatch]
                    mean, log_std = self._mean_and_log_std(states[idx], priors[idx], action_indices[idx])
                    log_prob = _normal_log_prob(actions[idx], mean, log_std)
                    entropy = _normal_entropy(log_std)
                    log_ratio = (log_prob - old_log_probs[idx]).clamp(-20.0, 20.0)
                    ratio = torch.exp(log_ratio)
                    clipped = torch.clamp(ratio, 1.0 - float(self.args.clip_range), 1.0 + float(self.args.clip_range))
                    adv = advantages[idx]
                    policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                    entropy_loss = -float(self.args.entropy_coef) * entropy.mean()
                    l2_loss = float(self.args.action_l2) * mean.pow(2).mean()
                    loss = (policy_loss + entropy_loss + l2_loss) / max(1, micro_count)
                    loss.backward()
                    with torch.no_grad():
                        policy_losses.append(float(policy_loss.detach().item()))
                        entropy_values.append(float(entropy.mean().detach().item()))
                        approx_kls.append(float((old_log_probs[idx] - log_prob).mean().detach().item()))
                        clip_fracs.append(float(((ratio - 1.0).abs() > float(self.args.clip_range)).float().mean().detach().item()))
                if float(self.args.max_grad_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.args.max_grad_norm))
                self.optimizer.step()
                self.gradient_step += 1

        base = self._unwrap(self.actor)
        metrics = {
            "loss_policy_mean": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "entropy_mean": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "approx_kl_mean": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction_mean": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "advantage_mean": float(advantages.mean().detach().item()),
            "advantage_std": float(advantages.std(unbiased=False).detach().item()) if advantages.numel() > 1 else 0.0,
            "log_std_mean": float(base.clamped_log_std().detach().mean().item()),
            "train_records": float(n_items),
        }

        hindsight_losses: list[float] = []
        bc_coef = max(0.0, float(getattr(self.args, "lchol_hindsight_bc_coef", 0.0)))
        if hindsight_records and bc_coef > 0.0:
            replay_states = torch.as_tensor(
                np.stack([row["state"] for row in hindsight_records]),
                dtype=torch.float32,
                device=self.device,
            )
            replay_priors = torch.as_tensor(
                np.stack([row["prior"] for row in hindsight_records]),
                dtype=torch.float32,
                device=self.device,
            )
            replay_actions = torch.as_tensor(
                np.stack([row["action"] for row in hindsight_records]),
                dtype=torch.float32,
                device=self.device,
            )
            replay_indices = torch.as_tensor(
                [int(row.get("action_index", 0)) for row in hindsight_records],
                dtype=torch.long,
                device=self.device,
            )
            replay_minibatch = max(1, min(int(self.args.minibatch_size), len(hindsight_records)))
            order = torch.randperm(len(hindsight_records), device=self.device)
            for start in range(0, len(hindsight_records), replay_minibatch):
                idx = order[start : start + replay_minibatch]
                self.optimizer.zero_grad(set_to_none=True)
                mean, log_std = self._mean_and_log_std(
                    replay_states[idx], replay_priors[idx], replay_indices[idx]
                )
                nll = -_normal_log_prob(replay_actions[idx], mean, log_std).mean()
                (bc_coef * nll).backward()
                if float(self.args.max_grad_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.args.max_grad_norm))
                self.optimizer.step()
                self.gradient_step += 1
                hindsight_losses.append(float(nll.detach().item()))
        metrics.update(
            {
                "hindsight_records": float(len(hindsight_records)),
                "hindsight_bc_loss": float(np.mean(hindsight_losses)) if hindsight_losses else 0.0,
                "hindsight_bc_coef": float(bc_coef),
            }
        )
        return metrics

    def load(self, checkpoint_path: Path) -> int:
        try:
            payload = torch.load(
                Path(checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:  # PyTorch < 2.6 has no weights_only keyword.
            payload = torch.load(Path(checkpoint_path), map_location=self.device)
        base = self._unwrap(self.actor)
        if "policy" in payload:
            base.load_state_dict(payload["policy"])
            if "optimizer" in payload:
                self.optimizer.load_state_dict(payload["optimizer"])
            self.gradient_step = int(payload.get("gradient_step", 0))
            self.loaded_extra_state = dict(payload.get("extra_state") or {})
            self.bootstrap_source = "grpo_resume"
        elif "actor" in payload:
            base.actor.load_state_dict(payload["actor"])
            self.gradient_step = 0
            self.loaded_extra_state = {}
            self.bootstrap_source = "smolvla_td3_actor"
        else:
            raise KeyError(
                f"Unsupported SmolVLA checkpoint {checkpoint_path}: expected 'policy' or 'actor'."
            )
        return int(payload.get("global_step", 0))

    def save(
        self,
        *,
        global_step: int,
        args: argparse.Namespace,
        latest: bool = False,
        extra_state: Mapping[str, Any] | None = None,
    ) -> Path:
        payload = {
            "policy_type": "smolvla_cdpr_grpo",
            "base_checkpoint": str(args.base_checkpoint),
            "global_step": int(global_step),
            "gradient_step": int(self.gradient_step),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "chunk_size": int(self.chunk_size),
            "residual_scale": float(args.residual_scale),
            "hidden_dim": int(args.hidden_dim),
            "policy": self._unwrap(self.actor).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "extra_state": dict(extra_state or {}),
            "args": vars(args),
        }
        if latest:
            output_path = self.run_dir / "latest.pt"
        else:
            step_dir = self.run_dir / f"step_{int(global_step):07d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            output_path = step_dir / "smolvla_grpo_adapter.pt"
        torch.save(payload, output_path)
        if not latest:
            torch.save(payload, self.run_dir / "latest.pt")
        return output_path


def _resolve_checkpoint(raw: str | Path) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.is_file():
        return path
    for name in ("smolvla_grpo_adapter.pt", "smolvla_cdpr_adapter.pt", "latest.pt"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find a SmolVLA GRPO checkpoint in {path}")


def _group_advantages(rewards: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float32)
    centered = values - float(values.mean())
    if bool(args.grpo_normalize_group_advantage):
        centered = centered / max(float(values.std()), 1e-6)
    clip_abs = float(args.grpo_clip_advantage_abs)
    if clip_abs > 0.0:
        centered = np.clip(centered, -clip_abs, clip_abs)
    return centered.astype(np.float32, copy=False)


def _select_candidate(rewards: np.ndarray, args: argparse.Namespace) -> int:
    mode = str(args.grpo_group_selection)
    if mode == "best":
        return int(np.argmax(rewards))
    if mode == "softmax":
        shifted = np.asarray(rewards, dtype=np.float64) - float(np.max(rewards))
        probs = np.exp(shifted)
        probs = probs / max(float(probs.sum()), 1e-12)
        return int(np.random.choice(len(rewards), p=probs))
    return int(np.random.randint(0, len(rewards)))


def _sample_prior(runtime: Any, slot: EnvSlot, *, progress_only: bool) -> np.ndarray:
    with _silence_output(bool(progress_only)):
        priors = runtime.sample_cdpr_chunks_from_envs(
            envs=[slot.env],
            observations=[slot.obs],
            infos=[slot.info],
            instructions=[slot.instruction],
        )
    return np.asarray(priors[0], dtype=np.float32)


def _evaluate_candidate_group(
    *,
    trainer: SmolVLAGRPOTrainer,
    slot: EnvSlot,
    action_index: int,
    prior: np.ndarray,
    args: argparse.Namespace,
    progress_only: bool,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    snapshot = slot.env.capture_state()
    actions, old_log_probs, means = trainer.sample_action_group(
        state=slot.state,
        prior=prior,
        action_index=int(action_index),
        group_size=int(args.grpo_group_size),
    )
    rewards: list[float] = []
    infos: list[dict[str, Any]] = []
    dones: list[bool] = []
    for candidate_idx, action in enumerate(actions):
        slot.env.restore_state(snapshot)
        with _silence_output(bool(progress_only)):
            _obs, reward, terminated, truncated, info = slot.env.step(action)
        rewards.append(float(reward))
        infos.append(dict(info))
        dones.append(bool(terminated or truncated))
    reward_arr = np.asarray(rewards, dtype=np.float32)
    advantages = _group_advantages(reward_arr, args)
    selected = _select_candidate(reward_arr, args)
    slot.env.restore_state(snapshot)
    records: list[dict[str, Any]] = []
    for idx, action in enumerate(actions):
        records.append(
            {
                "state": np.asarray(slot.state, dtype=np.float32).copy(),
                "prior": np.asarray(prior, dtype=np.float32).copy(),
                "action_index": int(action_index),
                "action": np.asarray(action, dtype=np.float32).copy(),
                "old_log_prob": float(old_log_probs[idx]),
                "advantage": float(advantages[idx]),
                "reward": float(reward_arr[idx]),
                "selected": bool(idx == selected),
                "candidate_success": bool(infos[idx].get("success", False)) if idx < len(infos) else False,
                "candidate_done": bool(dones[idx]) if idx < len(dones) else False,
                "mean_action": np.asarray(means, dtype=np.float32).copy(),
            }
        )
    group_stats = {
        "candidate_reward_mean": float(reward_arr.mean()),
        "candidate_reward_std": float(reward_arr.std()),
        "candidate_reward_max": float(reward_arr.max()),
        "candidate_reward_min": float(reward_arr.min()),
        "candidate_selected_reward": float(reward_arr[selected]),
        "candidate_selected_index": int(selected),
        "candidate_success_count": float(np.sum(reward_arr >= 1.0 - 1e-6)),
        "candidate_binary_reward_rate": float(
            np.mean(np.logical_or(np.isclose(reward_arr, 0.0), np.isclose(reward_arr, 1.0)))
        ),
        "zero_advantage_group": float(float(reward_arr.std()) <= 1e-8),
    }
    return records, selected, group_stats


def _parameter_summary(trainer: SmolVLAGRPOTrainer, runtime_policy: Any, args: argparse.Namespace) -> dict[str, Any]:
    base = trainer._unwrap(trainer.actor)
    total = 0
    trainable = 0
    for param in base.parameters():
        count = int(param.numel())
        total += count
        if bool(param.requires_grad):
            trainable += count
    return {
        "optimized_surface": "residual_actor_plus_grpo_log_std_no_td3_critics",
        "optimized_parameter_count": int(sum(param.numel() for group in trainer.optimizer.param_groups for param in group["params"])),
        "online_modules": {
            "grpo_policy": {
                "total": int(total),
                "trainable": int(trainable),
            }
        },
        "pretrained_prior": {
            "name": "SmolVLA",
            "base_checkpoint": str(args.base_checkpoint),
            **_torch_pretrained_parameter_summary(runtime_policy),
        },
    }


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
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard")) if dist_ctx.is_main and SummaryWriter is not None else None
    metrics_path = run_dir / "metrics.jsonl"

    metrics_window = max(1, int(args.metrics_window_episodes))
    complex_runtime = SmolVLAComplexRuntime(args=args, seed=int(rollout_seed))
    complex_training_active = str(args.complex_training_approach) != "none"
    dense_curriculum_active = _dense_curriculum_enabled(args) and not complex_training_active
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
        f"[smolvla-grpo] Loading SmolVLA checkpoint: {args.base_checkpoint} "
        f"(rank={dist_ctx.rank}, world_size={dist_ctx.world_size}, device={dist_ctx.device}, "
        f"mixed_precision={args.mixed_precision})",
    )
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
    _log(dist_ctx, f"[smolvla-grpo] Loaded SmolVLA; {runtime.device_summary()}")

    slots: list[EnvSlot] = []
    episode_initial_object_positions: dict[int, np.ndarray] = {}
    validation_env = None
    progress = None
    try:
        env_count = max(1, int(args.num_envs_per_rank))
        _log(dist_ctx, f"[smolvla-grpo] Building {env_count} CDPR env(s) on rank {dist_ctx.rank}...")
        for env_idx in range(env_count):
            seed = int(rollout_seed) + env_idx * 997
            with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                env = _build_env(args, seed=seed)
                if dense_curriculum_active:
                    dense_stage_instruction_types = _apply_dense_stage_to_envs([env], args, dense_stage_index)
                reset_options = (
                    complex_runtime.reset_options()
                    if complex_training_active
                    else mixed_sampler.reset_options(dense_stage_index)
                )
                obs, info = env.reset(seed=seed, options=reset_options)
            layout = CDPRStateLayout.from_observation(obs) if env_idx == 0 else layout
            episode_initial_object_positions[env_idx] = np.asarray(
                obs["all_object_positions"], dtype=np.float32
            ).copy()
            complex_runtime.reset_episode(env_idx)
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
            _log(dist_ctx, f"[smolvla-grpo] Dense curriculum stage 1 active: {', '.join(dense_stage_instruction_types)}")
        if dist_ctx.is_main and _validation_enabled(args):
            with _silence_output(bool(args.progress_only)):
                validation_env = _build_env(args, seed=int(args.validation_seed))

        device = torch.device(args.device)
        trainer = SmolVLAGRPOTrainer(
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
            complex_runtime.load_state_dict(
                dict(trainer.loaded_extra_state.get("complex_runtime") or {})
            )
            _log(
                dist_ctx,
                f"[smolvla-grpo] Loaded checkpoint {resume_path} at step {start_step} "
                f"(source={trainer.bootstrap_source})",
            )

        manifest = {
            "policy_type": "smolvla_cdpr_grpo",
            "base_checkpoint": str(args.base_checkpoint),
            "run_dir": run_dir.as_posix(),
            "config": str(args.config or ""),
            "action_keys": ["x", "y", "z", "yaw", "gripper"],
            "chunk_size": int(args.chunk_size),
            "frozen_smolvla": True,
            "trainable_surface": "residual_chunk_head_and_log_std",
            "no_td3_critics": True,
            "checkpoint_bootstrap_source": str(trainer.bootstrap_source),
            "complex_training_approach": str(args.complex_training_approach),
            "policy_gradient_reward": "exact_sparse_binary_0_or_1",
            "hindsight_replay_role": (
                "separate_auxiliary_behavior_cloning"
                if complex_runtime.hindsight_enabled
                else "disabled"
            ),
            "grpo": {
                "group_size": int(args.grpo_group_size),
                "group_selection": str(args.grpo_group_selection),
                "normalize_group_advantage": bool(args.grpo_normalize_group_advantage),
                "clip_advantage_abs": float(args.grpo_clip_advantage_abs),
                "clip_range": float(args.clip_range),
                "ppo_epochs": int(args.ppo_epochs),
                "minibatch_size": int(args.minibatch_size),
                "microbatch_size": int(args.microbatch_size),
            },
            "parameter_training": _parameter_summary(trainer, runtime.policy, args),
            "instruction_types": list(args.instruction_types or []),
            "num_envs_per_rank": int(env_count),
            "distributed_world_size": int(dist_ctx.world_size),
            "rank_device": str(dist_ctx.device),
        }
        if dist_ctx.is_main:
            _write_json(run_dir / "smolvla_grpo_manifest.json", manifest)
            _write_json(run_dir / "complex_curriculum_state.json", complex_runtime.json_state())

        global_step = int(start_step)
        last_validation_step = int(start_step)
        last_metrics: dict[str, float] = {}
        completed_episode_rewards: deque[float] = deque(maxlen=metrics_window)
        completed_episode_successes: deque[float] = deque(maxlen=metrics_window)
        completed_episode_count = 0
        completed_episode_success_count = 0
        progress = _make_progress_bar(args=args, ctx=dist_ctx, start_step=start_step)
        status_start_t = time.perf_counter()
        _log(dist_ctx, f"[smolvla-grpo] Startup ready in {time.perf_counter() - startup_t0:.1f}s")

        while global_step < int(args.max_train_steps):
            rollout_records: list[dict[str, Any]] = []
            rollout_group_stats: list[dict[str, Any]] = []
            for _rollout_idx in range(max(1, int(args.rollout_steps))):
                for slot_idx, slot in enumerate(slots):
                    if global_step >= int(args.max_train_steps):
                        break
                    prior = _sample_prior(
                        runtime,
                        slot,
                        progress_only=bool(args.progress_only) or not dist_ctx.is_main,
                    )
                    records, selected_idx, group_stats = _evaluate_candidate_group(
                        trainer=trainer,
                        slot=slot,
                        action_index=0,
                        prior=prior,
                        args=args,
                        progress_only=bool(args.progress_only) or not dist_ctx.is_main,
                    )
                    source_instruction_type = _info_instruction_type(slot.info, slot.instruction)
                    for record in records:
                        record["instruction_type"] = str(source_instruction_type)
                    selected_action = np.asarray(records[selected_idx]["action"], dtype=np.float32)
                    pre_action_snapshot = slot.env.capture_state()
                    with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                        next_obs, reward, terminated, truncated, next_info = slot.env.step(selected_action)
                    next_state = layout.flatten(next_obs)
                    done = bool(terminated or truncated)
                    episode_success = bool(next_info.get("success", False)) if done else False
                    rollout_records.extend(records)
                    rollout_group_stats.append(group_stats)
                    global_step += 1
                    slot.episode_length += 1
                    slot.episode_reward += float(reward)

                    if complex_runtime.hindsight_enabled:
                        trajectory_step = dict(next_info)
                        trajectory_step.update(
                            {
                                "action": selected_action.copy(),
                                "source_instruction": str(slot.instruction),
                                "instruction_type": str(source_instruction_type),
                                "all_object_positions": np.asarray(
                                    next_obs["all_object_positions"], dtype=np.float32
                                ).copy(),
                                "initial_all_object_positions": episode_initial_object_positions[
                                    slot_idx
                                ].copy(),
                                "source_rollout_id": f"rank{dist_ctx.rank}-slot{slot_idx}-episode{slot.episode}",
                                "source_policy_version": int(trainer.gradient_step),
                            }
                        )
                        new_relabels = complex_runtime.append_trajectory_step(
                            slot_idx, trajectory_step
                        )
                        eligible_relabels = [
                            hindsight
                            for hindsight in new_relabels
                            if int(hindsight.first_timestep) == int(slot.episode_length - 1)
                        ]
                        if eligible_relabels:
                            post_action_snapshot = slot.env.capture_state()
                            try:
                                slot.env.restore_state(pre_action_snapshot)
                                with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                                    relabeled_priors = runtime.sample_cdpr_chunks_from_envs(
                                        envs=[slot.env] * len(eligible_relabels),
                                        observations=[slot.obs] * len(eligible_relabels),
                                        infos=[slot.info] * len(eligible_relabels),
                                        instructions=[
                                            str(hindsight.instruction)
                                            for hindsight in eligible_relabels
                                        ],
                                    )
                                for hindsight, relabeled_prior in zip(
                                    eligible_relabels, relabeled_priors
                                ):
                                    complex_runtime.add_hindsight_record(
                                        {
                                            "option_name": str(hindsight.option_name),
                                            "instruction": str(hindsight.instruction),
                                            "source_instruction": str(hindsight.source_instruction),
                                            "state": np.asarray(slot.state, dtype=np.float32).copy(),
                                            "prior": np.asarray(
                                                relabeled_prior, dtype=np.float32
                                            ).copy(),
                                            "action_index": 0,
                                            "action": selected_action.copy(),
                                            "first_timestep": int(hindsight.first_timestep),
                                            "metadata": dict(hindsight.metadata),
                                        }
                                    )
                            finally:
                                slot.env.restore_state(post_action_snapshot)

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
                            required_episodes = max(1, int(args.dense_stage_min_episodes))
                            stage_success_rate = (
                                float(sum(dense_stage_successes) / len(dense_stage_successes))
                                if dense_stage_successes
                                else 0.0
                            )
                            if (
                                int(dense_stage_index) == 1
                                and str(args.dense_stage_gate_metric) == "rollout"
                                and len(dense_stage_successes) >= required_episodes
                                and stage_success_rate >= float(args.dense_stage_switch_success_rate)
                            ):
                                proposed_stage_index = 2
                        previous_stage_index = int(dense_stage_index)
                        dense_stage_index = proposed_stage_index
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
                                f"[smolvla-grpo] Dense curriculum switched to stage {dense_stage_index}: "
                                f"{', '.join(dense_stage_instruction_types)}",
                                progress=progress,
                            )

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
                        group_reward_mean = float(np.mean([item["candidate_reward_mean"] for item in rollout_group_stats])) if rollout_group_stats else 0.0
                        zero_advantage_rate = float(
                            np.mean([item["zero_advantage_group"] for item in rollout_group_stats])
                        ) if rollout_group_stats else 0.0
                        binary_reward_rate = float(
                            np.mean([item["candidate_binary_reward_rate"] for item in rollout_group_stats])
                        ) if rollout_group_stats else 0.0
                        complex_metrics = complex_runtime.metrics()
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
                            "instruction": str(slot.instruction),
                            "candidate_reward_mean": group_reward_mean,
                            "zero_advantage_group_rate": zero_advantage_rate,
                            "candidate_binary_reward_rate": binary_reward_rate,
                            **episode_metrics,
                            **dense_stage_metrics,
                            **complex_metrics,
                            **last_metrics,
                            **_gpu_metrics(device),
                        }
                        with metrics_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                        rollout_scalars = {
                            "rollout/reward": float(reward),
                            "rollout/episode_reward_running": float(slot.episode_reward),
                            "rollout/candidate_reward_mean": group_reward_mean,
                            "rollout/zero_advantage_group_rate": zero_advantage_rate,
                            "rollout/candidate_binary_reward_rate": binary_reward_rate,
                            "rollout/grpo_records_pending": float(len(rollout_records)),
                            **{f"gpu/{k}": v for k, v in _gpu_metrics(device).items()},
                        }
                        rollout_scalars.update(_episode_metric_scalars(episode_metrics))
                        rollout_scalars.update(_dense_stage_metric_scalars(dense_stage_metrics))
                        rollout_scalars.update(complex_runtime.metrics(consume_interval_counts=True))
                        _log_scalars(writer, rollout_scalars, global_step)
                        if not complex_training_active:
                            _log_mixed_curriculum_scalars(writer, mixed_sampler.snapshot(), global_step)

                    if progress is not None:
                        progress.update(1)
                        if global_step % max(1, int(args.status_every_steps)) == 0 or done:
                            _progress_postfix(
                                progress,
                                episode=slot.episode,
                                episode_length=slot.episode_length,
                                episode_reward=slot.episode_reward,
                                reward=float(reward),
                                buffer_size=len(rollout_records),
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
                            f"[smolvla-grpo] step={global_step:07d} slot={slot_idx} "
                            f"episode={slot.episode:05d} ep_len={slot.episode_length:03d} "
                            f"reward_running={slot.episode_reward:+.3f} last_reward={float(reward):+.3f} "
                            f"instruction={slot.instruction}",
                        )
                        _log(
                            dist_ctx,
                            "[smolvla-grpo] "
                            + _format_step_progress(
                                global_step=global_step,
                                max_train_steps=int(args.max_train_steps),
                                start_step=int(start_step),
                                elapsed_seconds=time.perf_counter() - status_start_t,
                            ),
                        )

                    if dist_ctx.is_main and global_step % max(1, int(args.save_every_steps)) == 0:
                        checkpoint = trainer.save(
                            global_step=global_step,
                            args=args,
                            latest=False,
                            extra_state={"complex_runtime": complex_runtime.state_dict()},
                        )
                        _write_json(
                            run_dir / "complex_curriculum_state.json",
                            complex_runtime.json_state(),
                        )
                        _log(dist_ctx, f"[smolvla-grpo] Saved checkpoint: {checkpoint}", progress=progress)

                    if _validation_due(args, global_step=global_step, last_validation_step=last_validation_step):
                        if dist_ctx.is_main:
                            validation_instruction_types = _validation_instruction_types(
                                args,
                                dense_curriculum_active=dense_curriculum_active,
                                dense_stage_index=dense_stage_index,
                            )
                            if validation_env is not None and validation_instruction_types:
                                reverse_plan = dict(complex_runtime.reverse_validation_plan())
                                if reverse_plan:
                                    active_instruction_types = tuple(reverse_plan)
                                    active_reset_options = {
                                        instruction: {
                                            "instruction_type": instruction,
                                            "curriculum_mode": "reverse_frontier",
                                            "curriculum_shell": int(shell),
                                            "start_with_caught_object": False,
                                            "start_with_target_at_gripper": False,
                                        }
                                        for instruction, shell in reverse_plan.items()
                                    }
                                    active_summary = _run_smolvla_distinct_validation(
                                        validation_env=validation_env,
                                        runtime=runtime,
                                        trainer=trainer,
                                        layout=layout,
                                        args=args,
                                        global_step=global_step,
                                        stage_index=dense_stage_index,
                                        instruction_types=active_instruction_types,
                                        reset_options_by_instruction=active_reset_options,
                                        episodes_per_instruction=int(
                                            args.reverse_frontier_validation_episodes
                                        ),
                                        record_videos=False,
                                    )
                                    active_summary["event"] = "reverse_frontier_validation"
                                    active_summary["reverse_frontier_shells"] = reverse_plan
                                    with metrics_path.open("a", encoding="utf-8") as handle:
                                        handle.write(json.dumps(active_summary, sort_keys=True) + "\n")
                                    per_instruction = active_summary.get(
                                        "validation_instruction_results", {}
                                    )
                                    complex_runtime.record_reverse_validation(
                                        [
                                            {
                                                "instruction_id": instruction,
                                                "shell_id": int(shell),
                                                "success_rate": float(
                                                    per_instruction.get(instruction, {}).get(
                                                        "success_rate", 0.0
                                                    )
                                                    or 0.0
                                                ),
                                                "rollouts": int(
                                                    per_instruction.get(instruction, {}).get(
                                                        "episodes", 0
                                                    )
                                                    or 0
                                                ),
                                                "action_saturation_rate": 0.0,
                                            }
                                            for instruction, shell in reverse_plan.items()
                                        ]
                                    )
                                    _write_json(
                                        run_dir / "complex_curriculum_state.json",
                                        complex_runtime.json_state(),
                                    )
                                    _log_scalars(
                                        writer,
                                        complex_runtime.metrics(),
                                        global_step,
                                    )
                                validation_summary = _run_smolvla_distinct_validation(
                                    validation_env=validation_env,
                                    runtime=runtime,
                                    trainer=trainer,
                                    layout=layout,
                                    args=args,
                                    global_step=global_step,
                                    stage_index=dense_stage_index,
                                    instruction_types=validation_instruction_types,
                                    reset_options_by_instruction=None,
                                    episodes_per_instruction=int(
                                        args.comparison_validation_episodes_per_instruction
                                    ),
                                    record_videos=True,
                                )
                                validation_summary["event"] = "full_task_comparison_validation"
                                _write_validation_summary(
                                    metrics_path=metrics_path,
                                    writer=writer,
                                    summary=validation_summary,
                                )
                                _log(
                                    dist_ctx,
                                    "[smolvla-grpo] Held-out validation "
                                    f"step={global_step:07d} stage={dense_stage_index} "
                                    f"success_rate={float(validation_summary['validation_success_rate'] or 0.0):.3f}",
                                    progress=progress,
                                )
                        last_validation_step = int(global_step)

                    if done:
                        episode_instruction_type = _info_instruction_type(
                            next_info, slot.instruction
                        )
                        episode_put_stage = int(
                            slot.info.get(
                                "curriculum_put_start_stage",
                                slot.info.get("curriculum_shell", -1),
                            )
                        )
                        put_stage_promoted = complex_runtime.record_episode(
                            instruction_type=episode_instruction_type,
                            success=episode_success,
                            episode_put_stage=episode_put_stage,
                        )
                        completed_episode_rewards.append(float(slot.episode_reward))
                        completed_episode_successes.append(1.0 if episode_success else 0.0)
                        completed_episode_count += 1
                        completed_episode_success_count += 1 if episode_success else 0
                        if dist_ctx.is_main:
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
                                            "put_stage_promoted": bool(put_stage_promoted),
                                            **episode_metrics,
                                            **dense_stage_metrics,
                                            **complex_runtime.metrics(),
                                            **(
                                                {}
                                                if complex_training_active
                                                else mixed_sampler.snapshot()
                                            ),
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
                            done_scalars.update(complex_runtime.metrics())
                            _log_scalars(writer, done_scalars, global_step)
                            if not complex_training_active:
                                _log_mixed_curriculum_scalars(writer, mixed_sampler.snapshot(), global_step)
                        slot.episode += 1
                        if not complex_training_active:
                            mixed_sampler.record(episode_instruction_type, episode_success)
                        reset_options = (
                            complex_runtime.reset_options()
                            if complex_training_active
                            else mixed_sampler.reset_options(dense_stage_index)
                        )
                        with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                            obs, info = slot.env.reset(
                                seed=None,
                                options=reset_options,
                            )
                        episode_initial_object_positions[slot_idx] = np.asarray(
                            obs["all_object_positions"], dtype=np.float32
                        ).copy()
                        complex_runtime.reset_episode(slot_idx)
                        slot.obs = obs
                        slot.info = dict(info)
                        slot.state = layout.flatten(obs)
                        slot.instruction = _safe_instruction(slot.info)
                        slot.stage_index = int(dense_stage_index)
                        slot.episode_reward = 0.0
                        slot.episode_length = 0
                        continue

                    slot.obs = next_obs
                    slot.info = dict(next_info)
                    slot.state = next_state
                    slot.instruction = _safe_instruction(slot.info)

            if rollout_records:
                hindsight_records = complex_runtime.sample_hindsight(len(rollout_records))
                last_metrics = trainer.update(
                    rollout_records,
                    hindsight_records=hindsight_records,
                )
                complex_runtime.record_train_update(
                    [str(item.get("instruction_type", "")) for item in rollout_records]
                )
                if dist_ctx.is_main:
                    _log_scalars(writer, {f"train/{k}": v for k, v in last_metrics.items()}, trainer.gradient_step)

        if dist_ctx.is_main:
            latest = trainer.save(
                global_step=global_step,
                args=args,
                latest=True,
                extra_state={"complex_runtime": complex_runtime.state_dict()},
            )
            _write_json(run_dir / "complex_curriculum_state.json", complex_runtime.json_state())
            _log(dist_ctx, f"[smolvla-grpo] Final latest checkpoint: {latest}", progress=progress)
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
