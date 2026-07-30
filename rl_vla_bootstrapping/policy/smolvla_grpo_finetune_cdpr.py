#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import warnings
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
from rl_vla_bootstrapping.policy.rank_local_grpo import (
    EqualDDPSchedule,
    pad_tensor_records,
)
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
    parser.add_argument(
        "--distributed-timeout-seconds",
        type=int,
        default=0,
        help="Process-group timeout; increase it when rank 0 performs long validation runs.",
    )
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
    parser.add_argument(
        "--simulator-backend",
        choices=("mujoco_cpu", "mjlab_mjwarp"),
        default="mujoco_cpu",
    )
    parser.add_argument("--worlds-per-rank", type=int, default=1)
    parser.add_argument("--groups-per-rank", type=int, default=1)
    parser.add_argument("--mjwarp-xml-path", default=None)
    parser.add_argument("--mjwarp-nconmax", type=int, default=256)
    parser.add_argument("--mjwarp-njmax", type=int, default=1024)
    parser.add_argument("--mjwarp-nccdmax", type=int, default=None)
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    parser.add_argument("--object-slots", type=int, default=4)
    # The floor here clamps EVERY commanded target, so it decides whether the
    # gripper can physically reach an object at all -- a separate and stricter
    # thing than the task's spawn band. Default matches CDPRBackendConfig.
    parser.add_argument(
        "--controller-workspace-z-bounds",
        nargs=2,
        type=float,
        default=None,
    )
    parser.add_argument(
        "--smolvla-inference-microbatch-size",
        type=int,
        default=0,
        help="Maximum GPU image batch per frozen SmolVLA forward; zero uses all local worlds.",
    )
    parser.add_argument(
        "--mjwarp-max-updates",
        type=int,
        default=0,
        help=(
            "Optional MJWarp-only optimizer-update limit for smoke/benchmark "
            "runs; zero trains to max_train_steps."
        ),
    )
    _bool_arg(
        parser,
        "mjwarp_profile_timers",
        default=False,
        help_text=(
            "Synchronize CUDA at MJWarp rollout timing boundaries. Enable for "
            "benchmarks; leave disabled for production throughput."
        ),
    )
    parser.add_argument(
        "--mjwarp-profile-updates",
        type=int,
        default=0,
        help=(
            "When synchronized MJWarp timers are enabled, profile only this "
            "many optimizer updates; zero profiles every update."
        ),
    )
    _bool_arg(
        parser,
        "allow_legacy_simulator_checkpoint",
        default=False,
        help_text=(
            "Explicitly permit loading an older policy checkpoint that has no "
            "simulator metadata; simulator/controller state is never restored."
        ),
    )

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
    parser.add_argument(
        "--smolvla-model-image-size",
        type=int,
        default=0,
        help=(
            "Override LeRobot's model-side padded image size. Zero preserves the "
            "checkpoint default (commonly 512); 256 avoids silently upsampling the "
            "already-resized CDPR frames before the vision encoder."
        ),
    )
    _bool_arg(
        parser,
        "smolvla_compile_model",
        default=False,
        help_text="Compile frozen SmolVLA action sampling with torch.compile on CUDA.",
    )
    parser.add_argument(
        "--smolvla-compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune",
    )
    parser.add_argument("--state-dim", type=int, default=6)
    _bool_arg(
        parser,
        "residual_relative_target",
        default=False,
        help_text=(
            "Append the 3-D end-effector->target vector to the trainable "
            "residual's state (SmolVLA still receives the truncated 6-dim "
            "state). Gives the residual an explicit direction to the object."
        ),
    )
    _bool_arg(
        parser,
        "residual_vision_features",
        default=False,
        help_text=(
            "Append a frozen fixed-projection of SmolVLA's connector vision "
            "tokens to the trainable residual's state so it can localize the "
            "target (the connector feature encodes target XY: linear-probe R^2 "
            "~ 0.44). SmolVLA still receives only the proprioceptive state."
        ),
    )
    parser.add_argument("--residual-vision-dim", type=int, default=512)
    # --- SmolVLA action-expert LoRA fine-tune (off by default) ---
    _bool_arg(
        parser,
        "train_vla_lora",
        default=False,
        help_text=(
            "Attach LoRA to SmolVLA's action-expert attention and train it with "
            "a grad-through-VLA GRPO pass (vision + VLM stay frozen)."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    _bool_arg(
        parser,
        "lora_include_mlp",
        default=False,
        help_text="Also LoRA the action-expert MLP (gate/up/down_proj).",
    )
    parser.add_argument(
        "--lora-expert-name-contains",
        default="lm_expert",
        help="Qualified-path substring selecting the action expert (not the VLM).",
    )
    parser.add_argument("--vla-lr", type=float, default=1.0e-5)
    parser.add_argument("--vla-kl-coef", type=float, default=0.1)
    parser.add_argument("--vla-microbatch-size", type=int, default=16)
    parser.add_argument("--vla-update-max-records", type=int, default=128)
    parser.add_argument("--image-feature-keys", nargs="+", default=None)
    _bool_arg(parser, "include_wrist", default=True, help_text="Include the CDPR wrist camera.")
    _bool_arg(parser, "include_aux_camera", default=True, help_text="Fill SmolVLA's third camera input.")
    _bool_arg(
        parser,
        "mask_empty_aux_camera",
        default=False,
        help_text=(
            "Without a real auxiliary camera, feed slot three as a black frame "
            "with a zero padding mask instead of duplicating the wrist view."
        ),
    )
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
    parser.add_argument("--clip-range-low", type=float, default=None)
    parser.add_argument("--clip-range-high", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--action-l2", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grpo-group-size", type=int, default=2)
    parser.add_argument("--grpo-group-selection", choices=("uniform", "best", "softmax"), default="uniform")
    _bool_arg(parser, "grpo_normalize_group_advantage", default=True, help_text="Normalize rewards inside each group.")
    parser.add_argument("--grpo-clip-advantage-abs", type=float, default=6.0)
    _bool_arg(
        parser,
        "grpo_trajectory_groups",
        default=False,
        help_text="Compare complete continuations from one cloned environment state.",
    )
    _bool_arg(
        parser,
        "grpo_dynamic_sampling",
        default=False,
        help_text="Only optimize trajectory groups whose pass rate is inside the configured bounds.",
    )
    parser.add_argument("--grpo-dynamic-min-pass-rate", type=float, default=0.10)
    parser.add_argument("--grpo-dynamic-max-pass-rate", type=float, default=0.90)
    parser.add_argument("--grpo-trajectory-max-decisions", type=int, default=0)
    _bool_arg(
        parser,
        "grpo_trajectory_shell_aware_horizon",
        default=False,
        help_text="Shorten failed Reverse Frontier continuations according to the sampled shell horizon.",
    )
    parser.add_argument("--grpo-trajectory-horizon-multiplier", type=float, default=1.5)
    parser.add_argument("--grpo-trajectory-horizon-grace-decisions", type=int, default=8)
    parser.add_argument(
        "--grpo-candidate-inference-batch-size",
        type=int,
        default=0,
        help=(
            "Batch divergent trajectory candidates into each frozen-VLA forward. "
            "Zero batches every candidate assigned to the local rank; one restores "
            "the legacy serial inference path."
        ),
    )
    parser.add_argument("--grpo-target-records-per-update", type=int, default=0)
    parser.add_argument("--grpo-max-groups-per-update", type=int, default=64)
    parser.add_argument(
        "--grpo-max-collection-seconds-per-update",
        type=float,
        default=0.0,
        help="Stop trajectory collection after this many seconds and update on accepted records; disabled at 0.",
    )
    _bool_arg(
        parser,
        "grpo_all_success_sample_harder",
        default=True,
        help_text="Retry an all-success Reverse Frontier group from the next harder shell when available.",
    )
    _bool_arg(
        parser,
        "grpo_shell0_retry_easiest",
        default=True,
        help_text="Retry shell-0 all-failure groups at the minimum shell-0 decision horizon.",
    )
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
    args.clip_range_low = float(args.clip_range if args.clip_range_low is None else args.clip_range_low)
    args.clip_range_high = float(args.clip_range if args.clip_range_high is None else args.clip_range_high)
    if args.clip_range_low < 0.0 or args.clip_range_high < 0.0:
        parser.error("--clip-range-low and --clip-range-high must be non-negative")
    args.grpo_dynamic_min_pass_rate = float(np.clip(args.grpo_dynamic_min_pass_rate, 0.0, 1.0))
    args.grpo_dynamic_max_pass_rate = float(np.clip(args.grpo_dynamic_max_pass_rate, 0.0, 1.0))
    if args.grpo_dynamic_min_pass_rate >= args.grpo_dynamic_max_pass_rate:
        parser.error("--grpo-dynamic-min-pass-rate must be smaller than --grpo-dynamic-max-pass-rate")
    args.grpo_trajectory_max_decisions = max(0, int(args.grpo_trajectory_max_decisions))
    args.grpo_trajectory_horizon_multiplier = max(
        1.0, float(args.grpo_trajectory_horizon_multiplier)
    )
    args.grpo_trajectory_horizon_grace_decisions = max(
        0, int(args.grpo_trajectory_horizon_grace_decisions)
    )
    args.grpo_candidate_inference_batch_size = max(
        0, int(args.grpo_candidate_inference_batch_size)
    )
    args.smolvla_model_image_size = max(0, int(args.smolvla_model_image_size))
    args.grpo_target_records_per_update = max(0, int(args.grpo_target_records_per_update))
    args.grpo_max_groups_per_update = max(1, int(args.grpo_max_groups_per_update))
    args.grpo_max_collection_seconds_per_update = max(
        0.0, float(args.grpo_max_collection_seconds_per_update)
    )
    args.replan_every = max(1, min(int(args.replan_every), int(args.chunk_size)))
    args.worlds_per_rank = max(1, int(args.worlds_per_rank))
    args.groups_per_rank = max(1, int(args.groups_per_rank))
    args.smolvla_inference_microbatch_size = max(
        0, int(args.smolvla_inference_microbatch_size)
    )
    args.mjwarp_max_updates = max(0, int(args.mjwarp_max_updates))
    args.mjwarp_profile_updates = max(0, int(args.mjwarp_profile_updates))
    if args.simulator_backend == "mjlab_mjwarp":
        expected_worlds = int(args.groups_per_rank) * int(args.grpo_group_size)
        if int(args.worlds_per_rank) != expected_worlds:
            parser.error(
                "--worlds-per-rank must equal --groups-per-rank * "
                f"--grpo-group-size ({expected_worlds}) for mjlab_mjwarp"
            )
        if not args.mjwarp_xml_path:
            parser.error("--mjwarp-xml-path is required for mjlab_mjwarp")
        if int(args.object_slots) != 4:
            parser.error("The active MJWarp CDPR topology requires --object-slots 4")
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
            # Rollout ranks may execute different numbers of inference forwards because
            # successful candidates terminate early. Buffer broadcasts would turn those
            # independent forwards into mismatched collectives.
            ddp_kwargs["broadcast_buffers"] = False
            policy = DDP(policy, **ddp_kwargs)
        self.actor = policy
        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=float(args.learning_rate),
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
        )
        self.gradient_step = 0
        self._warned_zero_vla_grad = False
        self.loaded_extra_state: dict[str, Any] = {}
        self.bootstrap_source = "fresh_grpo"
        self.profile_update = bool(args.mjwarp_profile_timers)

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

    def sample_action_chunk(
        self,
        *,
        state: np.ndarray,
        prior: np.ndarray,
        action_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample one open-loop action chunk from a single policy decision."""
        actions, log_probs, means = self.sample_action_chunks_batch(
            states=np.asarray(state, dtype=np.float32)[None, ...],
            priors=np.asarray(prior, dtype=np.float32)[None, ...],
            action_count=action_count,
        )
        return actions[0], log_probs[0], means[0]

    def sample_action_chunks_batch(
        self,
        *,
        states: np.ndarray,
        priors: np.ndarray,
        action_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample open-loop chunks for several candidate states in one forward."""
        count = max(1, min(int(action_count), int(self.chunk_size)))
        state_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        prior_t = torch.as_tensor(priors, dtype=torch.float32, device=self.device)
        if state_t.ndim != 2 or prior_t.ndim != 3:
            raise ValueError(
                "Batched GRPO sampling expects states [B,D] and priors [B,H,A], "
                f"got {tuple(state_t.shape)} and {tuple(prior_t.shape)}."
            )
        if int(state_t.shape[0]) != int(prior_t.shape[0]):
            raise ValueError("Batched GRPO states and priors must have matching batch sizes.")
        with torch.no_grad():
            # Rollout inference does not need the DDP wrapper. Parameters remain
            # synchronized by DDP optimizer updates, while bypassing the wrapper
            # avoids per-forward reducer bookkeeping on divergent rank rollouts.
            base = self._unwrap(self.actor)
            mean_chunk = base(state_t, prior_t)[:, :count]
            log_std = base.clamped_log_std()[:count].unsqueeze(0)
            actions = torch.clamp(
                mean_chunk + torch.randn_like(mean_chunk) * torch.exp(log_std),
                -1.0,
                1.0,
            )
            log_probs = _normal_log_prob(actions, mean_chunk, log_std)
        return (
            actions.detach().to(dtype=torch.float32).cpu().numpy(),
            log_probs.detach().to(dtype=torch.float32).cpu().numpy(),
            mean_chunk.detach().to(dtype=torch.float32).cpu().numpy(),
        )

    def sample_action_chunks_tensor(
        self,
        *,
        states: torch.Tensor,
        priors: torch.Tensor,
        action_count: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU-resident stochastic sampling for every active local world."""

        count = max(1, min(int(action_count), int(self.chunk_size)))
        if states.device != self.device or priors.device != self.device:
            raise RuntimeError("Rank-local GRPO sampling tensors must stay on the trainer GPU.")
        if states.ndim != 2 or priors.ndim != 3:
            raise ValueError(
                f"Expected states [B,D] and priors [B,H,A], got "
                f"{tuple(states.shape)} and {tuple(priors.shape)}."
            )
        if int(states.shape[0]) != int(priors.shape[0]):
            raise ValueError("GRPO state and prior batch sizes differ.")
        with torch.no_grad():
            base = self._unwrap(self.actor)
            mean_chunk = base(states, priors)[:, :count]
            log_std = base.clamped_log_std()[:count].unsqueeze(0)
            noise = torch.randn(
                mean_chunk.shape,
                dtype=mean_chunk.dtype,
                device=mean_chunk.device,
                generator=generator,
            )
            actions = torch.clamp(
                mean_chunk + noise * torch.exp(log_std), -1.0, 1.0
            )
            log_probs = _normal_log_prob(actions, mean_chunk, log_std)
        return actions, log_probs, mean_chunk

    def deterministic_action_chunks_tensor(
        self,
        *,
        states: torch.Tensor,
        priors: torch.Tensor,
        action_count: int,
    ) -> torch.Tensor:
        """Return the residual-policy mean for CUDA-resident validation."""

        count = max(1, min(int(action_count), int(self.chunk_size)))
        if states.device != self.device or priors.device != self.device:
            raise RuntimeError(
                "Rank-local GRPO validation tensors must stay on the trainer GPU."
            )
        if states.ndim != 2 or priors.ndim != 3:
            raise ValueError(
                f"Expected states [B,D] and priors [B,H,A], got "
                f"{tuple(states.shape)} and {tuple(priors.shape)}."
            )
        if int(states.shape[0]) != int(priors.shape[0]):
            raise ValueError("GRPO state and prior batch sizes differ.")
        with torch.inference_mode():
            base = self._unwrap(self.actor)
            return torch.clamp(base(states, priors)[:, :count], -1.0, 1.0)

    def update_tensor_records(
        self,
        records: Mapping[str, torch.Tensor],
        *,
        loss_mask: torch.Tensor,
        schedule: EqualDDPSchedule,
    ) -> dict[str, float]:
        """DDP-safe PPO update with an identical collective schedule per rank."""

        required = {
            "state",
            "prior",
            "action",
            "action_index",
            "old_log_prob",
            "advantage",
        }
        missing = sorted(required.difference(records))
        if missing:
            raise KeyError(f"Missing GPU GRPO record tensors: {missing}.")
        input_lengths = {int(value.shape[0]) for value in records.values()}
        if len(input_lengths) != 1:
            raise ValueError(
                f"GPU GRPO record tensor lengths differ: {sorted(input_lengths)}."
            )
        input_count = input_lengths.pop()
        input_mask = torch.as_tensor(
            loss_mask, dtype=torch.float32, device=self.device
        ).reshape(-1)
        if int(input_mask.numel()) != input_count:
            raise ValueError(
                f"loss_mask has {input_mask.numel()} rows for {input_count} records."
            )
        informative_indices = torch.nonzero(
            input_mask > 0.0, as_tuple=False
        ).reshape(-1)
        if int(informative_indices.numel()) > 0:
            compact = {
                key: value.index_select(0, informative_indices)
                for key, value in records.items()
            }
            compact_mask = torch.ones(
                (int(informative_indices.numel()),),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            # Retain one graph-shaped placeholder. It receives zero loss, but
            # lets an all-padding rank traverse the same DDP reducer graph.
            compact = {key: value[:1] for key, value in records.items()}
            compact_mask = torch.zeros(
                (1,), dtype=torch.float32, device=self.device
            )
        target = int(schedule.padded_records_per_rank)
        padded, default_mask = pad_tensor_records(
            compact, target_records=target
        )
        actual_target = int(default_mask.numel())
        if actual_target != target:
            raise AssertionError(
                "Synchronized DDP schedule is smaller than a rank's informative "
                f"record count: schedule={target}, compact={actual_target}."
            )
        mask = compact_mask
        if int(mask.numel()) < actual_target:
            mask = torch.cat(
                (
                    mask,
                    torch.zeros(
                        (actual_target - int(mask.numel()),),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
            )
        mask = mask * default_mask
        if any(value.device != self.device for value in padded.values()):
            raise RuntimeError("GPU record tensors left the rank-local device.")

        states = padded["state"].to(dtype=torch.float32)
        priors = padded["prior"].to(dtype=torch.float32)
        actions = padded["action"].to(dtype=torch.float32)
        action_indices = padded["action_index"].to(dtype=torch.long)
        old_log_probs = padded["old_log_prob"].to(dtype=torch.float32)
        advantages = padded["advantage"].to(dtype=torch.float32)
        valid = mask > 0.0
        valid_count = valid.sum().clamp_min(1)
        valid_advantages = advantages[valid]
        if int(valid_advantages.numel()) > 1:
            adv_mean = valid_advantages.mean()
            adv_std = valid_advantages.std(unbiased=False).clamp_min(1.0e-6)
            advantages = torch.where(
                valid, (advantages - adv_mean) / adv_std, advantages
            )

        minibatch = int(schedule.records_per_minibatch)
        if target % minibatch:
            raise AssertionError("Padded records must be a whole number of minibatches.")
        microbatch = max(1, min(int(self.args.microbatch_size), minibatch))
        if minibatch % microbatch:
            raise ValueError(
                "DDP-safe MJWarp updates require minibatch_size to be divisible "
                "by microbatch_size so every rank executes the same backward count."
            )

        base = self._unwrap(self.actor)
        initial_log_std = base.clamped_log_std().detach().clone()
        policy_loss_total = torch.zeros((), dtype=torch.float32, device=self.device)
        entropy_total = torch.zeros_like(policy_loss_total)
        kl_total = torch.zeros_like(policy_loss_total)
        clip_total = torch.zeros_like(policy_loss_total)
        metric_weight = torch.zeros_like(policy_loss_total)
        gradient_norms: list[float] = []
        optimizer_steps = 0
        update_forward_time_s = 0.0
        backpropagation_time_s = 0.0
        optimizer_time_s = 0.0

        def synchronize_profile() -> None:
            if self.profile_update and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        for _epoch in range(int(schedule.ppo_epochs)):
            order = torch.randperm(target, device=self.device)
            for start in range(0, target, minibatch):
                mb_idx = order[start : start + minibatch]
                self.optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, minibatch, microbatch):
                    idx = mb_idx[micro_start : micro_start + microbatch]
                    synchronize_profile()
                    forward_started = time.perf_counter()
                    weight = mask[idx]
                    denominator = weight.sum().clamp_min(1.0)
                    mean, log_std = self._mean_and_log_std(
                        states[idx], priors[idx], action_indices[idx]
                    )
                    log_prob = _normal_log_prob(actions[idx], mean, log_std)
                    entropy = _normal_entropy(log_std)
                    ratio = torch.exp(
                        (log_prob - old_log_probs[idx]).clamp(-20.0, 20.0)
                    )
                    clipped = torch.clamp(
                        ratio,
                        1.0 - float(self.args.clip_range_low),
                        1.0 + float(self.args.clip_range_high),
                    )
                    surrogate = torch.minimum(
                        ratio * advantages[idx], clipped * advantages[idx]
                    )
                    policy_loss = -(surrogate * weight).sum() / denominator
                    entropy_mean = (entropy * weight).sum() / denominator
                    l2_per_row = mean.pow(2).mean(dim=-1)
                    l2_mean = (l2_per_row * weight).sum() / denominator
                    loss = (
                        policy_loss
                        - float(self.args.entropy_coef) * entropy_mean
                        + float(self.args.action_l2) * l2_mean
                    ) / (minibatch // microbatch)
                    synchronize_profile()
                    if self.profile_update:
                        update_forward_time_s += (
                            time.perf_counter() - forward_started
                        )
                    # Even an all-padding rank traverses the DDP graph and
                    # participates in the same reducer collective with zero grads.
                    backward_started = time.perf_counter()
                    loss.backward()
                    synchronize_profile()
                    if self.profile_update:
                        backpropagation_time_s += (
                            time.perf_counter() - backward_started
                        )
                    with torch.no_grad():
                        metric_weight += weight.sum()
                        policy_loss_total += policy_loss * weight.sum()
                        entropy_total += entropy_mean * weight.sum()
                        kl_total += (
                            ((old_log_probs[idx] - log_prob) * weight).sum()
                        )
                        outside = (
                            (ratio < 1.0 - float(self.args.clip_range_low))
                            | (ratio > 1.0 + float(self.args.clip_range_high))
                        )
                        clip_total += (outside.to(torch.float32) * weight).sum()
                optimizer_started = time.perf_counter()
                grad_limit = (
                    float(self.args.max_grad_norm)
                    if float(self.args.max_grad_norm) > 0.0
                    else float("inf")
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), grad_limit
                )
                gradient_norms.append(float(grad_norm.detach().item()))
                self.optimizer.step()
                synchronize_profile()
                if self.profile_update:
                    optimizer_time_s += (
                        time.perf_counter() - optimizer_started
                    )
                self.gradient_step += 1
                optimizer_steps += 1

        metric_denominator = metric_weight.clamp_min(1.0)
        valid_advantages = advantages[valid]
        return {
            "loss_policy_mean": float(
                (policy_loss_total / metric_denominator).detach().item()
            ),
            "entropy_mean": float(
                (entropy_total / metric_denominator).detach().item()
            ),
            "approx_kl_mean": float(
                (kl_total / metric_denominator).detach().item()
            ),
            "clip_fraction_mean": float(
                (clip_total / metric_denominator).detach().item()
            ),
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "gradient_norm_max": float(np.max(gradient_norms)),
            "optimizer_steps": float(optimizer_steps),
            "update_forward_time_s": float(update_forward_time_s),
            "backpropagation_time_s": float(backpropagation_time_s),
            "optimizer_time_s": float(optimizer_time_s),
            "backward_collectives": float(
                schedule.backward_collectives * (minibatch // microbatch)
            ),
            "informative_records": float(valid.sum().detach().item()),
            "padded_records": float(target),
            "advantage_mean": float(
                valid_advantages.mean().detach().item()
                if int(valid_advantages.numel())
                else 0.0
            ),
            "advantage_std": float(
                valid_advantages.std(unbiased=False).detach().item()
                if int(valid_advantages.numel()) > 1
                else 0.0
            ),
            "log_std_mean": float(base.clamped_log_std().detach().mean().item()),
            "log_std_update_abs_mean": float(
                (base.clamped_log_std().detach() - initial_log_std)
                .abs()
                .mean()
                .item()
            ),
        }

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

        base = self._unwrap(self.actor)
        initial_log_std = base.clamped_log_std().detach().clone()
        n_items = int(states.shape[0])
        minibatch = max(1, min(int(self.args.minibatch_size), n_items))
        microbatch = max(1, min(int(self.args.microbatch_size), minibatch))
        policy_losses: list[float] = []
        entropy_values: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []
        clip_low_fracs: list[float] = []
        clip_high_fracs: list[float] = []
        gradient_norms: list[float] = []
        optimizer_steps = 0

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
                    clip_low = float(self.args.clip_range_low)
                    clip_high = float(self.args.clip_range_high)
                    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
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
                        outside_clip = torch.logical_or(
                            ratio < 1.0 - clip_low,
                            ratio > 1.0 + clip_high,
                        )
                        clip_fracs.append(
                            float(outside_clip.float().mean().detach().item())
                        )
                        clip_low_fracs.append(
                            float((ratio < 1.0 - clip_low).float().mean().detach().item())
                        )
                        clip_high_fracs.append(
                            float((ratio > 1.0 + clip_high).float().mean().detach().item())
                        )
                grad_limit = (
                    float(self.args.max_grad_norm)
                    if float(self.args.max_grad_norm) > 0.0
                    else float("inf")
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_limit)
                gradient_norms.append(float(grad_norm.detach().item()))
                self.optimizer.step()
                self.gradient_step += 1
                optimizer_steps += 1

        metrics = {
            "loss_policy_mean": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "entropy_mean": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "approx_kl_mean": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction_mean": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "clip_fraction_low_mean": float(np.mean(clip_low_fracs)) if clip_low_fracs else 0.0,
            "clip_fraction_high_mean": float(np.mean(clip_high_fracs)) if clip_high_fracs else 0.0,
            "gradient_norm_mean": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
            "gradient_norm_max": float(np.max(gradient_norms)) if gradient_norms else 0.0,
            "optimizer_steps": float(optimizer_steps),
            "advantage_mean": float(advantages.mean().detach().item()),
            "advantage_std": float(advantages.std(unbiased=False).detach().item()) if advantages.numel() > 1 else 0.0,
            "log_std_mean": float(base.clamped_log_std().detach().mean().item()),
            "log_std_update_abs_mean": float(
                (base.clamped_log_std().detach() - initial_log_std).abs().mean().item()
            ),
            "train_records": float(n_items),
        }
        for chunk_action_index in range(int(self.chunk_size)):
            action_index_count = int(
                (action_indices == int(chunk_action_index)).sum().detach().item()
            )
            metrics[
                f"train_records_action_index_{chunk_action_index}"
            ] = float(action_index_count)
            metrics[
                f"train_record_action_index_{chunk_action_index}_rate"
            ] = float(action_index_count / max(1, n_items))
            metrics[f"log_std_action_index_{chunk_action_index}"] = float(
                base.clamped_log_std()[chunk_action_index].detach().mean().item()
            )

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

    def attach_vla_lora(self, runtime: Any) -> dict[str, float]:
        """Attach LoRA to the frozen SmolVLA action expert and register it.

        Vision encoder and VLM stay frozen; only the action-expert attention
        (and optionally its MLP) get trainable low-rank adapters, optimized by a
        separate AdamW at ``--vla-lr``. The residual actor and its optimizer are
        untouched.
        """

        from rl_vla_bootstrapping.policy.lora import (
            attach_lora,
            count_trainable,
            freeze_all_but_lora,
            lora_parameters,
        )

        args = self.args
        leaves = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if bool(getattr(args, "lora_include_mlp", False)):
            leaves += ["gate_proj", "up_proj", "down_proj"]
        replaced = attach_lora(
            runtime.policy,
            target_leaf_names=tuple(leaves),
            name_contains=(str(args.lora_expert_name_contains),),
            rank=int(args.lora_rank),
            alpha=float(args.lora_alpha),
            dropout=float(args.lora_dropout),
        )
        if not replaced:
            raise RuntimeError(
                "LoRA attach matched no action-expert linears; check "
                "--lora-expert-name-contains against the SmolVLA module names."
            )
        freeze_all_but_lora(runtime.policy)
        runtime.policy.to(self.device)
        self.vla_runtime = runtime
        self.vla_lora_params = lora_parameters(runtime.policy)
        self.vla_optimizer = torch.optim.AdamW(
            self.vla_lora_params,
            lr=float(args.vla_lr),
            eps=float(args.adam_eps),
            weight_decay=0.0,
        )
        return {
            "vla_lora/modules": float(len(replaced)),
            "vla_lora/trainable_params": float(count_trainable(runtime.policy)),
        }

    def update_vla_lora(self, records: Mapping[str, Any]) -> dict[str, float]:
        """Grad-through-VLA PPO+KL step on the action-expert LoRA.

        ``records`` are a capped decision-0 subsample carrying the SmolVLA inputs
        (images/state/instruction), the taken action + behaviour log-prob, the
        GRPO advantage, and the detached rollout prior as the KL reference.
        """

        import torch.distributed as dist

        runtime = getattr(self, "vla_runtime", None)
        if runtime is None:
            raise RuntimeError("attach_vla_lora must run before update_vla_lora.")
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1

        advantages = records.get("advantage") if records else None
        n = 0 if advantages is None else int(advantages.shape[0])
        base = self._unwrap(self.actor)
        self.vla_optimizer.zero_grad(set_to_none=True)
        base.zero_grad(set_to_none=True)

        ppo_sum = 0.0
        kl_sum = 0.0
        counted = 0
        if n > 0:
            overview = records["overview"]
            wrist = records["wrist"]
            states = records["state"].to(dtype=torch.float32)
            instructions = list(records["instruction"])
            actions = records["action"].to(dtype=torch.float32)
            action_indices = records["action_index"].to(dtype=torch.long)
            old_log_probs = records["old_log_prob"].to(dtype=torch.float32)
            advantages = advantages.to(dtype=torch.float32)
            prior_ref = records["prior_ref"].to(dtype=torch.float32)
            if n > 1:
                advantages = (
                    advantages - advantages.mean()
                ) / advantages.std(unbiased=False).clamp_min(1.0e-6)

            micro = max(1, min(int(self.args.vla_microbatch_size), n))
            clip_low = float(self.args.clip_range_low)
            clip_high = float(self.args.clip_range_high)
            kl_coef = float(self.args.vla_kl_coef)
            for start in range(0, n, micro):
                end = min(n, start + micro)
                sl = slice(start, end)
                prior_grad = runtime.sample_cdpr_chunks_from_tensors(
                    primary_images=overview[sl].to(dtype=torch.float32),
                    wrist_images=wrist[sl].to(dtype=torch.float32),
                    states=states[sl],
                    instructions=instructions[start:end],
                    microbatch_size=0,
                    enable_grad=True,
                )
                mean_chunk = base(states[sl], prior_grad)
                mean = _gather_chunk_values(mean_chunk, action_indices[sl])
                log_std = base.clamped_log_std()[
                    action_indices[sl].clamp(0, self.chunk_size - 1)
                ]
                log_prob = _normal_log_prob(actions[sl], mean, log_std)
                ratio = torch.exp(
                    (log_prob - old_log_probs[sl]).clamp(-20.0, 20.0)
                )
                adv = advantages[sl]
                unclipped = ratio * adv
                clipped = (
                    torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * adv
                )
                ppo = -torch.min(unclipped, clipped).mean()
                kl = ((prior_grad - prior_ref[sl]) ** 2).mean()
                weight = float(end - start) / float(n)
                (weight * (ppo + kl_coef * kl)).backward()
                ppo_sum += float(ppo.detach().item()) * (end - start)
                kl_sum += float(kl.detach().item()) * (end - start)
                counted += end - start

        # LoRA params live on the frozen runtime, outside the residual's DDP
        # wrapper, so sync their grads manually. Zero-fill missing grads first
        # so every rank issues the identical collective schedule.
        if distributed:
            for param in self.vla_lora_params:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
            # Coalesce into ONE collective. Reducing each tensor separately was
            # a few hundred small NCCL calls per update, each with its own
            # transient buffer -- slow, and it churned the allocator on a device
            # already competing with Warp for memory.
            grads = [param.grad for param in self.vla_lora_params]
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat /= float(world_size)
            for grad, reduced in zip(
                grads, torch._utils._unflatten_dense_tensors(flat, grads)
            ):
                grad.copy_(reduced)
            del flat
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.vla_lora_params, float(self.args.max_grad_norm)
        )
        # An exactly-zero norm on a non-empty batch means nothing reached the
        # adapters (lora_b starts at zero but must still receive gradient), so
        # the run would train the VLA not at all. Say so once, loudly, rather
        # than let hours of "training" pass with a flat grad_norm curve.
        if n > 0 and float(grad_norm) == 0.0 and not self._warned_zero_vla_grad:
            self._warned_zero_vla_grad = True
            warnings.warn(
                "vla_lora/grad_norm is exactly 0 on a non-empty LoRA batch: no "
                "gradient is reaching the action-expert adapters, so the VLA is "
                "not being trained. Check that the SmolVLA forward used for the "
                "update is not wrapped in torch.no_grad()/inference_mode().",
                RuntimeWarning,
            )
        self.vla_optimizer.step()
        self.vla_optimizer.zero_grad(set_to_none=True)
        base.zero_grad(set_to_none=True)
        denom = max(1, counted)
        return {
            "vla_lora/records": float(n),
            "vla_lora/ppo_loss": ppo_sum / denom,
            "vla_lora/kl": kl_sum / denom,
            "vla_lora/grad_norm": float(grad_norm),
        }

    def load(
        self,
        checkpoint_path: Path,
        *,
        expected_simulator_metadata: Mapping[str, Any] | None = None,
        allow_legacy_simulator_metadata: bool = False,
    ) -> int:
        try:
            payload = torch.load(
                Path(checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:  # PyTorch < 2.6 has no weights_only keyword.
            payload = torch.load(Path(checkpoint_path), map_location=self.device)
        expected_metadata = dict(expected_simulator_metadata or {})
        stored_metadata = dict(payload.get("simulator_metadata") or {})
        if expected_metadata:
            if not stored_metadata and not bool(allow_legacy_simulator_metadata):
                raise RuntimeError(
                    "Checkpoint predates simulator metadata. Pass the explicit "
                    "legacy-checkpoint opt-in to load policy weights only; no "
                    "simulator/controller state will be restored."
                )
            if stored_metadata:
                compatibility_keys = (
                    "backend",
                    "versions",
                    "worlds_per_rank",
                    "groups_per_rank",
                    "grpo_group_size",
                    "physics_substeps_per_action",
                    "physics_dtype",
                    "controller_implementation",
                    "action_step_xyz",
                    "action_step_yaw",
                    "action_step_gripper",
                    "lock_non_commanded_axes",
                    "lock_non_commanded_axes_threshold",
                    "xml_sha256",
                    "object_assets_sha256",
                    "object_geometry",
                    "nconmax_per_world",
                    "njmax_per_world",
                    "nccdmax_per_world",
                    "render_width",
                    "render_height",
                    "object_slots",
                    "object_catalogs",
                    "camera_order",
                )
                differences = {
                    key: (stored_metadata.get(key), expected_metadata.get(key))
                    for key in compatibility_keys
                    if key in expected_metadata
                    and stored_metadata.get(key) != expected_metadata.get(key)
                }
                if differences:
                    details = ", ".join(
                        f"{key}: checkpoint={old!r}, runtime={new!r}"
                        for key, (old, new) in differences.items()
                    )
                    raise RuntimeError(
                        "Checkpoint simulator assumptions are incompatible: "
                        + details
                    )
        base = self._unwrap(self.actor)
        if "policy" in payload:
            base.load_state_dict(payload["policy"])
            if "optimizer" in payload:
                self.optimizer.load_state_dict(payload["optimizer"])
            self._load_vla_lora_state(payload)
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

    def load_weights_only(self, checkpoint_path: "Path | str") -> None:
        """Warm-start from a checkpoint's WEIGHTS only.

        Loads the residual (+ log_std) and the LoRA tensors, but deliberately
        discards the optimizer state, LoRA optimizer, curriculum/extra state, and
        global step. The learned behaviour carries over while the training run
        starts fresh at step 0 with the current schedule and hyperparameters --
        so a re-tuned curriculum/gate takes effect instead of being overwritten
        by the checkpoint's stalled state. Simulator-metadata is not checked
        because only architecture-shaped weights are transferred; a mismatched
        state_dim / chunk / LoRA rank surfaces as a load_state_dict error.
        """

        try:
            payload = torch.load(
                Path(checkpoint_path), map_location=self.device, weights_only=False
            )
        except TypeError:  # PyTorch < 2.6
            payload = torch.load(Path(checkpoint_path), map_location=self.device)
        if "policy" not in payload:
            raise KeyError(
                f"Warm-start checkpoint {checkpoint_path} has no 'policy' weights."
            )
        self._unwrap(self.actor).load_state_dict(payload["policy"])
        lora_state = payload.get("vla_lora")
        runtime = getattr(self, "vla_runtime", None)
        if lora_state:
            if runtime is None:
                raise RuntimeError(
                    "Warm-start checkpoint carries LoRA weights but no LoRA is "
                    "attached; re-run with matching --train-vla-lora settings."
                )
            runtime.policy.load_state_dict(lora_state, strict=False)
        # Fresh training state: step 0, empty curriculum, no optimizer carry-over.
        self.gradient_step = 0
        self.loaded_extra_state = {}
        self.bootstrap_source = "grpo_warmstart_weights"

    def _vla_lora_state_dict(self) -> dict[str, Any] | None:
        """Only the LoRA tensors from the frozen runtime, or None if unused."""

        runtime = getattr(self, "vla_runtime", None)
        if runtime is None:
            return None
        return {
            name: tensor.detach().cpu()
            for name, tensor in runtime.policy.state_dict().items()
            if "lora_" in name
        }

    def _load_vla_lora_state(self, payload: Mapping[str, Any]) -> None:
        """Restore LoRA weights/optimizer; refuse to silently lose them."""

        lora_state = payload.get("vla_lora")
        runtime = getattr(self, "vla_runtime", None)
        if not lora_state:
            if runtime is not None:
                warnings.warn(
                    "Resuming with --train-vla-lora, but the checkpoint holds "
                    "no LoRA weights: the action expert restarts from a no-op "
                    "adapter and prior VLA adaptation is lost.",
                    RuntimeWarning,
                )
            return
        if runtime is None:
            raise RuntimeError(
                "Checkpoint carries action-expert LoRA weights but this run has "
                "no LoRA attached. Re-run with --train-vla-lora (and matching "
                "--lora-rank/--lora-include-mlp) or the adapted prior is lost."
            )
        # strict=False: the checkpoint deliberately holds only lora_* keys.
        runtime.policy.load_state_dict(lora_state, strict=False)
        optimizer_state = payload.get("vla_lora_optimizer")
        if optimizer_state and getattr(self, "vla_optimizer", None) is not None:
            self.vla_optimizer.load_state_dict(optimizer_state)

    def save(
        self,
        *,
        global_step: int,
        args: argparse.Namespace,
        latest: bool = False,
        extra_state: Mapping[str, Any] | None = None,
        simulator_metadata: Mapping[str, Any] | None = None,
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
            # The action-expert LoRA lives on the frozen SmolVLA runtime, not
            # on self.actor, so it would otherwise be silently dropped: a
            # resumed phase would restart from a zero (no-op) adapter and throw
            # away every step of VLA adaptation, and evaluation would run the
            # un-adapted prior. Store only the lora_* tensors (a few MB).
            "vla_lora": self._vla_lora_state_dict(),
            "vla_lora_optimizer": (
                self.vla_optimizer.state_dict()
                if getattr(self, "vla_optimizer", None) is not None
                else None
            ),
            "optimizer": self.optimizer.state_dict(),
            "extra_state": dict(extra_state or {}),
            "simulator_metadata": dict(simulator_metadata or {}),
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
    return _sample_prior_for_observation(
        runtime,
        env=slot.env,
        obs=slot.obs,
        info=slot.info,
        instruction=slot.instruction,
        progress_only=progress_only,
    )


def _sample_prior_for_observation(
    runtime: Any,
    *,
    env: Any,
    obs: Mapping[str, Any],
    info: Mapping[str, Any],
    instruction: str,
    progress_only: bool,
) -> np.ndarray:
    with _silence_output(bool(progress_only)):
        priors = runtime.sample_cdpr_chunks_from_envs(
            envs=[env],
            observations=[obs],
            infos=[info],
            instructions=[instruction],
        )
    return np.asarray(priors[0], dtype=np.float32)


def _distributed_candidate_indices(
    group_size: int,
    world_size: int,
    rank: int,
) -> list[int]:
    """Assign global GRPO candidates to ranks without overlap."""
    return list(range(int(rank), max(0, int(group_size)), max(1, int(world_size))))


def _dist_all_gather_object(value: Any, ctx: DistributedContext) -> list[Any]:
    if not ctx.enabled:
        return [value]
    import torch.distributed as dist

    gathered: list[Any] = [None for _ in range(int(ctx.world_size))]
    dist.all_gather_object(gathered, value)
    return gathered


def _dist_broadcast_object(
    value: Any,
    ctx: DistributedContext,
    *,
    src: int = 0,
) -> Any:
    if not ctx.enabled:
        return value
    import torch.distributed as dist

    payload = [value if int(ctx.rank) == int(src) else None]
    dist.broadcast_object_list(payload, src=int(src))
    return payload[0]


def _observation_reset_signature(
    observation: Mapping[str, Any],
    info: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact deterministic signature used to verify rank-synchronized resets."""
    digest = hashlib.sha256()
    key_digests: dict[str, str] = {}
    for key in sorted(observation):
        value = np.asarray(observation[key])
        value_bytes = np.ascontiguousarray(value).tobytes()
        key_digests[str(key)] = hashlib.sha256(value_bytes).hexdigest()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value_bytes)
    return {
        "observation_sha256": digest.hexdigest(),
        "observation_key_sha256": key_digests,
        "scene": str(info.get("scene", "")),
        "scene_objects": tuple(str(item) for item in info.get("scene_objects", ())),
        "instruction_type": str(info.get("instruction_type", "")),
        "language_instruction": str(info.get("language_instruction", "")),
        "target_object_catalog": str(info.get("target_object_catalog", "")),
        "curriculum_mode": str(info.get("curriculum_mode", "")),
        "curriculum_shell": int(info.get("curriculum_shell", -1)),
        "curriculum_shell_target_policy_steps": int(
            info.get("curriculum_shell_target_policy_steps", 0) or 0
        ),
        "curriculum_shell_target_action_steps": int(
            info.get("curriculum_shell_target_action_steps", 0) or 0
        ),
    }


def _assert_synchronized_reset(
    observation: Mapping[str, Any],
    info: Mapping[str, Any],
    ctx: DistributedContext,
) -> None:
    if not ctx.enabled:
        return
    signatures = _dist_all_gather_object(
        _observation_reset_signature(observation, info), ctx
    )
    expected = signatures[0]
    if any(signature != expected for signature in signatures[1:]):
        raise RuntimeError(
            "Synchronized Reverse Frontier reset produced different rank states: "
            f"{signatures}"
        )


def _synchronize_environment_reset_state(
    *,
    env: Any,
    observation: Mapping[str, Any],
    info: Mapping[str, Any],
    ctx: DistributedContext,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Restore every rank from rank 0's exact post-reset simulator snapshot."""
    if not ctx.enabled:
        return (
            {
                str(key): np.asarray(value).copy()
                for key, value in observation.items()
            },
            dict(info),
        )

    payload = None
    if ctx.is_main:
        payload = {
            "snapshot": env.capture_state(),
            "observation": {
                str(key): np.asarray(value).copy()
                for key, value in observation.items()
            },
            "info": dict(info),
        }
    canonical = dict(_dist_broadcast_object(payload, ctx, src=0))

    # Restore rank 0 as well so both ranks execute the identical mj_setState +
    # mj_forward path before the byte-for-byte verification below.
    env.restore_state(canonical["snapshot"])
    get_observation = getattr(env, "_get_obs", None)
    if callable(get_observation):
        synchronized_observation = {
            str(key): np.asarray(value).copy()
            for key, value in dict(get_observation()).items()
        }
    else:
        synchronized_observation = {
            str(key): np.asarray(value).copy()
            for key, value in dict(canonical["observation"]).items()
        }
    synchronized_info = dict(canonical["info"])
    _assert_synchronized_reset(synchronized_observation, synchronized_info, ctx)
    return synchronized_observation, synchronized_info


def _synchronized_reset_payload(
    *,
    ctx: DistributedContext,
    options: Mapping[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "options": None if options is None else dict(options),
        "seed": int(seed),
    }
    return dict(_dist_broadcast_object(payload, ctx, src=0))


def _trajectory_group_is_informative(
    outcomes: np.ndarray,
    args: argparse.Namespace,
) -> bool:
    if not bool(args.grpo_dynamic_sampling):
        return True
    values = np.asarray(outcomes, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return False
    pass_rate = float(values.mean())
    return bool(
        pass_rate >= float(args.grpo_dynamic_min_pass_rate)
        and pass_rate <= float(args.grpo_dynamic_max_pass_rate)
        and float(values.std()) > 1e-8
    )


def _dynamic_frontier_retry_options(
    group_stats: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str]:
    """Choose the next training reset after DAPO rejects a homogeneous group."""
    if not bool(args.grpo_trajectory_groups and args.grpo_dynamic_sampling):
        return None, "none"
    if float(group_stats.get("informative_group", 0.0)) >= 0.5:
        return None, "none"

    instruction = str(group_stats.get("instruction_type", "") or "")
    shell = int(group_stats.get("curriculum_shell", -1))
    shell_count = max(0, int(group_stats.get("curriculum_shell_count", 0)))
    successes = int(group_stats.get("candidate_success_count", 0))
    group_size = max(1, int(group_stats.get("group_size", args.grpo_group_size)))
    if not instruction or shell < 0 or shell_count <= 0:
        return None, "none"

    options = {
        "instruction_type": instruction,
        "curriculum_mode": "reverse_frontier",
        "start_with_caught_object": False,
        "start_with_target_at_gripper": False,
    }
    if successes <= 0:
        if shell > 0:
            options.update(
                {
                    "curriculum_shell": int(shell - 1),
                    "curriculum_sample_source": "dapo_easier_shell_retry",
                }
            )
            return options, "easier_shell"
        options.update(
            {
                "curriculum_shell": 0,
                "curriculum_sample_source": "dapo_shell0_easiest_retry",
            }
        )
        if bool(args.grpo_shell0_retry_easiest):
            low = max(1, int(group_stats.get("curriculum_shell_policy_steps_low", 1)))
            options["curriculum_target_policy_steps"] = low
        return options, "shell0_easiest"

    if successes >= group_size and bool(args.grpo_all_success_sample_harder):
        harder_shell = int(shell + 1)
        if harder_shell < shell_count:
            options.update(
                {
                    "curriculum_shell": harder_shell,
                    "curriculum_sample_source": "dapo_harder_shell_retry",
                }
            )
            return options, "harder_shell"
    return None, "none"


def _trajectory_group_metric_scalars(
    group_stats: Sequence[Mapping[str, Any]],
    *,
    group_size: int,
) -> dict[str, float]:
    if not group_stats:
        return {}
    stats = list(group_stats)
    informative_count = int(
        sum(float(item.get("informative_group", 0.0)) >= 0.5 for item in stats)
    )
    total_group_wall_time_s = max(
        1e-9, sum(float(item.get("group_wall_time_s", 0.0)) for item in stats)
    )
    total_sampled_decisions = sum(
        float(item.get("sampled_policy_decisions", 0.0)) for item in stats
    )
    total_sampled_actions = sum(
        float(item.get("sampled_environment_actions", 0.0)) for item in stats
    )
    total_selected_actions = sum(
        float(
            item.get(
                "selected_environment_actions",
                float(item.get("sampled_environment_actions", 0.0))
                / max(1, int(group_size)),
            )
        )
        for item in stats
    )
    total_camera_render_time_s = sum(
        float(item.get("camera_render_time_s", 0.0)) for item in stats
    )
    total_prior_model_time_s = sum(
        float(item.get("prior_model_time_s", 0.0)) for item in stats
    )
    total_residual_inference_time_s = sum(
        float(item.get("residual_inference_time_s", 0.0)) for item in stats
    )
    total_env_step_time_s = sum(
        float(item.get("env_step_time_s", 0.0)) for item in stats
    )
    total_snapshot_time_s = sum(
        float(item.get("snapshot_time_s", 0.0)) for item in stats
    )
    total_distributed_sync_time_s = sum(
        float(item.get("distributed_sync_time_s", 0.0)) for item in stats
    )
    total_candidate_inference_batches = sum(
        float(item.get("candidate_inference_batches", 0.0)) for item in stats
    )
    total_candidate_inference_batch_items = sum(
        float(item.get("candidate_inference_batch_size_mean", 0.0))
        * float(item.get("candidate_inference_batches", 0.0))
        for item in stats
    )
    total_accepted_records = sum(
        float(item.get("accepted_policy_records", 0.0)) for item in stats
    )
    out = {
        "rollout/grpo_groups_attempted": float(len(stats)),
        "rollout/grpo_groups_accepted": float(informative_count),
        "rollout/grpo_groups_rejected": float(len(stats) - informative_count),
        "rollout/grpo_trajectories_sampled": float(len(stats) * max(1, int(group_size))),
        "rollout/grpo_informative_group_rate": float(
            np.mean([float(item.get("informative_group", 0.0)) for item in stats])
        ),
        "rollout/grpo_all_fail_group_rate": float(
            np.mean([float(item.get("all_fail_group", 0.0)) for item in stats])
        ),
        "rollout/grpo_all_success_group_rate": float(
            np.mean([float(item.get("all_success_group", 0.0)) for item in stats])
        ),
        "rollout/grpo_sampled_policy_decisions_total": float(
            total_sampled_decisions
        ),
        "rollout/grpo_sampled_policy_decisions_per_group": float(
            np.mean([float(item.get("sampled_policy_decisions", 0.0)) for item in stats])
        ),
        "rollout/grpo_sampled_environment_actions_total": float(
            total_sampled_actions
        ),
        "rollout/grpo_sampled_environment_actions_per_group": float(
            np.mean(
                [float(item.get("sampled_environment_actions", 0.0)) for item in stats]
            )
        ),
        "rollout/grpo_selected_environment_actions_total": float(
            total_selected_actions
        ),
        "rollout/grpo_trajectory_work_amplification": float(
            total_sampled_actions / max(1.0, total_selected_actions)
        ),
        "rollout/grpo_accepted_policy_records": float(
            total_accepted_records
        ),
        "rollout/grpo_trajectory_length_mean": float(
            np.mean([float(item.get("trajectory_length_mean", 0.0)) for item in stats])
        ),
        "rollout/grpo_trajectory_length_min": float(
            min(float(item.get("trajectory_length_min", 0.0)) for item in stats)
        ),
        "rollout/grpo_trajectory_length_max": float(
            max(float(item.get("trajectory_length_max", 0.0)) for item in stats)
        ),
        "rollout/grpo_trajectory_policy_decisions_mean": float(
            np.mean(
                [
                    float(item.get("trajectory_policy_decisions_mean", 0.0))
                    for item in stats
                ]
            )
        ),
        "rollout/grpo_trajectory_policy_decisions_min": float(
            min(
                float(item.get("trajectory_policy_decisions_min", 0.0))
                for item in stats
            )
        ),
        "rollout/grpo_trajectory_policy_decisions_max": float(
            max(
                float(item.get("trajectory_policy_decisions_max", 0.0))
                for item in stats
            )
        ),
        "rollout/grpo_executed_actions_per_policy_decision": float(
            total_sampled_actions / max(1.0, total_sampled_decisions)
        ),
        "rollout/grpo_trajectory_decision_horizon_mean": float(
            np.mean([float(item.get("trajectory_decision_horizon", 0.0)) for item in stats])
        ),
        "rollout/grpo_trajectory_decision_horizon_max": float(
            max(float(item.get("trajectory_decision_horizon", 0.0)) for item in stats)
        ),
        "rollout/grpo_trajectory_action_horizon_mean": float(
            np.mean([float(item.get("trajectory_action_horizon", 0.0)) for item in stats])
        ),
        "rollout/grpo_trajectory_action_horizon_max": float(
            max(float(item.get("trajectory_action_horizon", 0.0)) for item in stats)
        ),
        "rollout/grpo_group_wall_time_s_mean": float(
            np.mean([float(item.get("group_wall_time_s", 0.0)) for item in stats])
        ),
        "rollout/grpo_group_wall_time_s_total": float(total_group_wall_time_s),
        "rollout/grpo_prior_inference_time_s_total": float(
            sum(float(item.get("prior_inference_time_s", 0.0)) for item in stats)
        ),
        "rollout/grpo_camera_render_time_s_total": float(
            total_camera_render_time_s
        ),
        "rollout/grpo_prior_model_time_s_total": float(total_prior_model_time_s),
        "rollout/grpo_residual_inference_time_s_total": float(
            total_residual_inference_time_s
        ),
        "rollout/grpo_env_step_time_s_total": float(total_env_step_time_s),
        "rollout/grpo_snapshot_time_s_total": float(total_snapshot_time_s),
        "rollout/grpo_distributed_sync_time_s_total": float(
            total_distributed_sync_time_s
        ),
        "rollout/grpo_camera_render_wall_fraction": float(
            total_camera_render_time_s / total_group_wall_time_s
        ),
        "rollout/grpo_prior_model_wall_fraction": float(
            total_prior_model_time_s / total_group_wall_time_s
        ),
        "rollout/grpo_residual_inference_wall_fraction": float(
            total_residual_inference_time_s / total_group_wall_time_s
        ),
        "rollout/grpo_env_step_wall_fraction": float(
            total_env_step_time_s / total_group_wall_time_s
        ),
        "rollout/grpo_snapshot_wall_fraction": float(
            total_snapshot_time_s / total_group_wall_time_s
        ),
        "rollout/grpo_distributed_sync_wall_fraction": float(
            total_distributed_sync_time_s / total_group_wall_time_s
        ),
        "rollout/grpo_candidate_inference_batch_size_mean": float(
            total_candidate_inference_batch_items
            / max(1.0, total_candidate_inference_batches)
        ),
        "rollout/grpo_candidate_inference_batch_size_max": float(
            max(
                float(item.get("candidate_inference_batch_size_max", 1.0))
                for item in stats
            )
        ),
        "rollout/grpo_sampled_policy_decisions_per_second": float(
            total_sampled_decisions / total_group_wall_time_s
        ),
        "rollout/grpo_sampled_environment_actions_per_second": float(
            total_sampled_actions / total_group_wall_time_s
        ),
        "rollout/grpo_selected_environment_actions_per_second": float(
            total_selected_actions / total_group_wall_time_s
        ),
        "rollout/grpo_accepted_policy_records_per_second": float(
            total_accepted_records / total_group_wall_time_s
        ),
        "rollout/grpo_easier_shell_retry_rate": float(
            np.mean([float(item.get("easier_shell_retry", 0.0)) for item in stats])
        ),
        "rollout/grpo_shell0_easiest_retry_rate": float(
            np.mean([float(item.get("shell0_easiest_retry", 0.0)) for item in stats])
        ),
        "rollout/grpo_harder_shell_retry_rate": float(
            np.mean([float(item.get("harder_shell_retry", 0.0)) for item in stats])
        ),
        "rollout/grpo_distributed_candidate_parallelism": float(
            np.mean(
                [float(item.get("distributed_candidate_parallelism", 1.0)) for item in stats]
            )
        ),
        "rollout/grpo_candidates_per_rank_min": float(
            min(float(item.get("candidates_per_rank_min", group_size)) for item in stats)
        ),
        "rollout/grpo_candidates_per_rank_max": float(
            max(float(item.get("candidates_per_rank_max", group_size)) for item in stats)
        ),
        "rollout/grpo_candidate_load_imbalance_mean": float(
            np.mean(
                [float(item.get("candidate_load_imbalance", 0.0)) for item in stats]
            )
        ),
        "rollout/grpo_decisions_per_rank_min": float(
            min(float(item.get("decisions_per_rank_min", 0.0)) for item in stats)
        ),
        "rollout/grpo_decisions_per_rank_max": float(
            max(float(item.get("decisions_per_rank_max", 0.0)) for item in stats)
        ),
        "rollout/grpo_rank_rollout_time_s_min": float(
            min(float(item.get("rank_rollout_time_s_min", 0.0)) for item in stats)
        ),
        "rollout/grpo_rank_rollout_time_s_max": float(
            max(float(item.get("rank_rollout_time_s_max", 0.0)) for item in stats)
        ),
        "rollout/grpo_rank_straggler_ratio_mean": float(
            np.mean([float(item.get("rank_straggler_ratio", 1.0)) for item in stats])
        ),
    }
    out["rollout/grpo_dynamic_sampling_efficiency"] = out[
        "rollout/grpo_informative_group_rate"
    ]
    for success_count in range(max(1, int(group_size)) + 1):
        out[f"rollout/grpo_group_success_count_{success_count}_rate"] = float(
            np.mean(
                [
                    int(round(float(item.get("candidate_success_count", -1))))
                    == success_count
                    for item in stats
                ]
            )
        )

    by_shell: dict[tuple[str, int], list[float]] = {}
    for item in stats:
        instruction = str(item.get("instruction_type", "") or "")
        shell = int(item.get("curriculum_shell", -1))
        if not instruction or shell < 0:
            continue
        key = (instruction, shell)
        by_shell.setdefault(key, []).append(float(item.get("candidate_reward_mean", 0.0)))
    for (instruction, shell), values in sorted(by_shell.items()):
        tag = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in instruction)
        out[f"rollout/grpo_group_pass_rate/{tag}/shell_{shell:02d}"] = float(np.mean(values))
    return out


def _step_interval_due(*, global_step: int, last_step: int, every: int) -> bool:
    interval = max(1, int(every))
    return int(global_step) // interval > int(last_step) // interval


def _trajectory_decision_horizon(
    info: Mapping[str, Any],
    args: argparse.Namespace,
) -> int:
    actions_per_policy_decision = max(
        1,
        int(
            info.get(
                "curriculum_actions_per_policy_decision",
                getattr(args, "replan_every", 1),
            )
            or 1
        ),
    )
    configured_cap = int(args.grpo_trajectory_max_decisions)
    cap = (
        configured_cap * actions_per_policy_decision
        if configured_cap > 0
        else int(args.max_env_steps)
    )
    cap = max(1, cap)
    if not bool(args.grpo_trajectory_shell_aware_horizon):
        return cap
    if str(info.get("curriculum_mode", "")) != "reverse_frontier":
        return cap
    configured_actions_per_decision = int(
        info.get("curriculum_actions_per_policy_decision", 0) or 0
    )
    if (
        configured_actions_per_decision > 0
        and configured_actions_per_decision != int(args.replan_every)
    ):
        raise RuntimeError(
            "Reverse Frontier action-chunk mismatch: reset metadata expects "
            f"{configured_actions_per_decision} actions per policy decision, but "
            f"training uses replan_every={int(args.replan_every)}."
        )

    target_policy = int(info.get("curriculum_shell_target_policy_steps", 0) or 0)
    high_policy = int(
        info.get("curriculum_shell_policy_steps_high", target_policy)
        or target_policy
    )
    target = int(
        info.get(
            "curriculum_shell_target_action_steps",
            target_policy * actions_per_policy_decision,
        )
        or 0
    )
    high = int(
        info.get(
            "curriculum_shell_action_steps_high",
            high_policy * actions_per_policy_decision,
        )
        or target
    )
    if target <= 0 and high <= 0:
        return cap
    high = max(1, high, target)
    multiplier = max(1.0, float(args.grpo_trajectory_horizon_multiplier))
    grace = (
        max(0, int(args.grpo_trajectory_horizon_grace_decisions))
        * actions_per_policy_decision
    )
    shell_horizon = max(
        target,
        int(math.ceil(float(high) * multiplier)),
        int(high + grace),
    )
    return max(1, min(cap, shell_horizon))


def _evaluate_trajectory_group(
    *,
    trainer: SmolVLAGRPOTrainer,
    runtime: Any,
    slot: EnvSlot,
    layout: CDPRStateLayout,
    args: argparse.Namespace,
    progress_only: bool,
) -> tuple[list[dict[str, Any]], int, dict[str, Any], dict[str, Any]]:
    """Roll out complete stochastic continuations from one identical state snapshot."""
    group_started = time.perf_counter()
    snapshot = slot.env.capture_state()
    group_size = max(2, int(args.grpo_group_size))
    max_action_steps = _trajectory_decision_horizon(slot.info, args)
    trajectories: list[list[dict[str, Any]]] = []
    outcomes: list[float] = []
    final_states: list[dict[str, Any]] = []
    prior_inference_time_s = 0.0
    env_step_time_s = 0.0

    for candidate_idx in range(group_size):
        slot.env.restore_state(snapshot)
        obs = slot.obs
        info = dict(slot.info)
        state = np.asarray(slot.state, dtype=np.float32).copy()
        instruction = str(slot.instruction)
        trajectory: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        final_reward = 0.0
        policy_decision_count = 0
        while len(trajectory) < max_action_steps:
            prior_started = time.perf_counter()
            prior = _sample_prior_for_observation(
                runtime,
                env=slot.env,
                obs=obs,
                info=info,
                instruction=instruction,
                progress_only=progress_only,
            )
            prior_inference_time_s += float(time.perf_counter() - prior_started)
            plan_state = state.copy()
            action_count = min(
                int(args.replan_every), max_action_steps - len(trajectory)
            )
            actions, old_log_probs, means = trainer.sample_action_chunk(
                state=state,
                prior=prior,
                action_count=action_count,
            )
            current_policy_decision = int(policy_decision_count)
            policy_decision_count += 1
            for chunk_action_index in range(action_count):
                action = np.asarray(
                    actions[chunk_action_index], dtype=np.float32
                )
                env_step_started = time.perf_counter()
                with _silence_output(bool(progress_only)):
                    next_obs, reward, terminated, truncated, next_info = slot.env.step(
                        action
                    )
                env_step_time_s += float(time.perf_counter() - env_step_started)
                trajectory.append(
                    {
                        # Every action in an open-loop chunk is conditioned on the
                        # observation at which that chunk was planned.
                        "state": plan_state.copy(),
                        "prior": prior.copy(),
                        "action_index": int(chunk_action_index),
                        "action": action.copy(),
                        "old_log_prob": float(old_log_probs[chunk_action_index]),
                        "reward": 0.0,
                        "selected": False,
                        "candidate_success": False,
                        "candidate_done": bool(terminated or truncated),
                        "mean_action": np.asarray(
                            means[chunk_action_index], dtype=np.float32
                        ).copy(),
                        "trajectory_index": int(candidate_idx),
                        "trajectory_step": int(len(trajectory) - 1),
                        "policy_decision_index": current_policy_decision,
                        "chunk_action_index": int(chunk_action_index),
                    }
                )
                obs = next_obs
                info = dict(next_info)
                state = layout.flatten(next_obs)
                instruction = _safe_instruction(info)
                final_reward = float(reward)
                if bool(terminated or truncated):
                    break
            if bool(terminated or truncated):
                break

        success = bool(info.get("success", False))
        outcome = 1.0 if success else 0.0
        if not bool(terminated or truncated):
            truncated = True
            info.update(
                {
                    "truncated": True,
                    "env_done": True,
                    "trajectory_horizon_truncated": True,
                }
            )
        for record in trajectory:
            record["reward"] = float(outcome)
            record["candidate_success"] = bool(success)
            record["trajectory_length"] = int(len(trajectory))
        trajectories.append(trajectory)
        outcomes.append(float(outcome))
        final_states.append(
            {
                "snapshot": slot.env.capture_state(),
                "obs": obs,
                "info": info,
                "state": state,
                "instruction": instruction,
                "reward": float(final_reward),
                "outcome": float(outcome),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "length": int(len(trajectory)),
                "policy_decisions": int(policy_decision_count),
                "last_action": (
                    np.asarray(trajectory[-1]["action"], dtype=np.float32).copy()
                    if trajectory
                    else np.zeros((int(args.action_dim),), dtype=np.float32)
                ),
            }
        )

    outcome_arr = np.asarray(outcomes, dtype=np.float32)
    advantages = _group_advantages(outcome_arr, args)
    informative = _trajectory_group_is_informative(outcome_arr, args)
    selected = _select_candidate(outcome_arr, args)
    records: list[dict[str, Any]] = []
    for trajectory_idx, trajectory in enumerate(trajectories):
        for record in trajectory:
            record["advantage"] = float(advantages[trajectory_idx])
            record["selected"] = bool(trajectory_idx == selected)
            if informative:
                records.append(record)

    selected_state = final_states[selected]
    slot.env.restore_state(selected_state["snapshot"])
    lengths = np.asarray([len(trajectory) for trajectory in trajectories], dtype=np.int64)
    policy_decision_counts = np.asarray(
        [int(item["policy_decisions"]) for item in final_states], dtype=np.int64
    )
    success_count = int(outcome_arr.sum())
    group_wall_time_s = max(1e-9, float(time.perf_counter() - group_started))
    group_stats = {
        "group_size": int(group_size),
        "instruction_type": str(_info_instruction_type(slot.info, slot.instruction)),
        "curriculum_shell": int(slot.info.get("curriculum_shell", -1)),
        "curriculum_shell_count": int(slot.info.get("curriculum_shell_count", 0)),
        "curriculum_shell_policy_steps_low": int(
            slot.info.get("curriculum_shell_policy_steps_low", 1)
        ),
        "candidate_reward_mean": float(outcome_arr.mean()),
        "candidate_reward_std": float(outcome_arr.std()),
        "candidate_reward_max": float(outcome_arr.max()),
        "candidate_reward_min": float(outcome_arr.min()),
        "candidate_selected_reward": float(outcome_arr[selected]),
        "candidate_selected_index": int(selected),
        "candidate_success_count": float(success_count),
        "candidate_binary_reward_rate": 1.0,
        "zero_advantage_group": float(float(outcome_arr.std()) <= 1e-8),
        "informative_group": float(informative),
        "all_fail_group": float(success_count == 0),
        "all_success_group": float(success_count == group_size),
        "sampled_policy_decisions": float(policy_decision_counts.sum()),
        "sampled_environment_actions": float(lengths.sum()),
        "selected_environment_actions": float(lengths[selected]),
        "trajectory_work_amplification": float(
            lengths.sum() / max(1, int(lengths[selected]))
        ),
        "accepted_policy_records": float(len(records)),
        "trajectory_length_mean": float(lengths.mean()),
        "trajectory_length_min": float(lengths.min()),
        "trajectory_length_max": float(lengths.max()),
        "trajectory_policy_decisions_mean": float(policy_decision_counts.mean()),
        "trajectory_policy_decisions_min": float(policy_decision_counts.min()),
        "trajectory_policy_decisions_max": float(policy_decision_counts.max()),
        "executed_actions_per_policy_decision": float(
            lengths.sum() / max(1, int(policy_decision_counts.sum()))
        ),
        "trajectory_decision_horizon": float(
            math.ceil(max_action_steps / max(1, int(args.replan_every)))
        ),
        "trajectory_action_horizon": float(max_action_steps),
        "group_wall_time_s": float(group_wall_time_s),
        "prior_inference_time_s": float(prior_inference_time_s),
        "env_step_time_s": float(env_step_time_s),
        "sampled_policy_decisions_per_second": float(
            policy_decision_counts.sum() / group_wall_time_s
        ),
        "sampled_environment_actions_per_second": float(
            lengths.sum() / group_wall_time_s
        ),
        "selected_environment_actions_per_second": float(
            lengths[selected] / group_wall_time_s
        ),
        "candidate_inference_batches": float(policy_decision_counts.sum()),
        "candidate_inference_batch_size_mean": 1.0,
        "candidate_inference_batch_size_max": 1.0,
        "trajectory_terminated_count": float(
            sum(bool(item["terminated"]) for item in final_states)
        ),
        "trajectory_truncated_count": float(
            sum(bool(item["truncated"]) for item in final_states)
        ),
        "distributed_candidate_parallelism": 1.0,
        "candidates_per_rank_min": float(group_size),
        "candidates_per_rank_max": float(group_size),
        "candidate_load_imbalance": 0.0,
        "decisions_per_rank_min": float(policy_decision_counts.sum()),
        "decisions_per_rank_max": float(policy_decision_counts.sum()),
        "rank_rollout_time_s_min": float(group_wall_time_s),
        "rank_rollout_time_s_max": float(group_wall_time_s),
        "rank_straggler_ratio": 1.0,
    }
    _retry_options, retry_kind = _dynamic_frontier_retry_options(group_stats, args)
    group_stats.update(
        {
            "easier_shell_retry": float(retry_kind == "easier_shell"),
            "shell0_easiest_retry": float(retry_kind == "shell0_easiest"),
            "harder_shell_retry": float(retry_kind == "harder_shell"),
        }
    )
    return records, selected, group_stats, selected_state


def _rollout_candidate_partition_batched(
    *,
    trainer: SmolVLAGRPOTrainer,
    runtime: Any,
    slot: EnvSlot,
    layout: CDPRStateLayout,
    args: argparse.Namespace,
    progress_only: bool,
    base_snapshot: Mapping[str, Any],
    candidate_indices: Sequence[int],
    max_action_steps: int,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, float],
] | None:
    """Roll out local candidates in lockstep and batch their VLA forwards.

    One MuJoCo instance is time-multiplexed through captured snapshots, so this
    does not require extra simulator memory. Camera rendering and environment
    stepping remain serial, but all locally assigned candidate observations are
    fed through one frozen-SmolVLA forward per policy decision.
    """
    configured_batch = int(getattr(args, "grpo_candidate_inference_batch_size", 0))
    required_methods = (
        callable(getattr(runtime, "capture_cdpr_images", None)),
        callable(getattr(runtime, "sample_cdpr_chunks_from_images", None)),
        callable(getattr(trainer, "sample_action_chunks_batch", None)),
    )
    if configured_batch == 1 or not all(required_methods) or not candidate_indices:
        return None

    batch_limit = (
        len(candidate_indices)
        if configured_batch <= 0
        else max(2, configured_batch)
    )
    candidates: dict[int, dict[str, Any]] = {
        int(candidate_idx): {
            "candidate_index": int(candidate_idx),
            "snapshot": base_snapshot,
            "obs": slot.obs,
            "info": dict(slot.info),
            "state": np.asarray(slot.state, dtype=np.float32).copy(),
            "instruction": str(slot.instruction),
            "trajectory": [],
            "terminated": False,
            "truncated": False,
            "final_reward": 0.0,
            "policy_decisions": 0,
        }
        for candidate_idx in candidate_indices
    }
    camera_render_time_s = 0.0
    prior_model_time_s = 0.0
    residual_inference_time_s = 0.0
    env_step_time_s = 0.0
    snapshot_time_s = 0.0
    inference_batches = 0
    inference_batch_items = 0
    inference_batch_size_max = 0

    while True:
        active = [
            candidate
            for candidate in candidates.values()
            if not bool(candidate["terminated"] or candidate["truncated"])
            and len(candidate["trajectory"]) < int(max_action_steps)
        ]
        if not active:
            break

        # Candidates that terminated inside an earlier chunk can leave the
        # survivors at different offsets. Group equal remaining chunk lengths
        # so every tensor in a batched actor call has the same [H,A] shape.
        by_action_count: dict[int, list[dict[str, Any]]] = {}
        for candidate in active:
            action_count = min(
                int(args.replan_every),
                int(max_action_steps) - len(candidate["trajectory"]),
            )
            by_action_count.setdefault(int(action_count), []).append(candidate)

        for action_count, same_count_candidates in by_action_count.items():
            for batch_start in range(0, len(same_count_candidates), batch_limit):
                batch_candidates = same_count_candidates[
                    batch_start : batch_start + batch_limit
                ]
                primary_images: list[np.ndarray] = []
                wrist_images: list[np.ndarray | None] = []

                render_started = time.perf_counter()
                for candidate in batch_candidates:
                    slot.env.restore_state(candidate["snapshot"])
                    primary, wrist = runtime.capture_cdpr_images(slot.env)
                    primary_images.append(primary)
                    wrist_images.append(wrist)
                camera_render_time_s += float(time.perf_counter() - render_started)

                prior_started = time.perf_counter()
                priors = runtime.sample_cdpr_chunks_from_images(
                    primary_images=primary_images,
                    wrist_images=wrist_images,
                    observations=[candidate["obs"] for candidate in batch_candidates],
                    infos=[candidate["info"] for candidate in batch_candidates],
                    instructions=[
                        str(candidate["instruction"])
                        for candidate in batch_candidates
                    ],
                )
                prior_model_time_s += float(time.perf_counter() - prior_started)

                residual_started = time.perf_counter()
                actions, old_log_probs, means = trainer.sample_action_chunks_batch(
                    states=np.stack(
                        [candidate["state"] for candidate in batch_candidates]
                    ).astype(np.float32, copy=False),
                    priors=np.asarray(priors, dtype=np.float32),
                    action_count=int(action_count),
                )
                residual_inference_time_s += float(
                    time.perf_counter() - residual_started
                )
                inference_batches += 1
                inference_batch_items += len(batch_candidates)
                inference_batch_size_max = max(
                    inference_batch_size_max, len(batch_candidates)
                )

                for batch_idx, candidate in enumerate(batch_candidates):
                    snapshot_started = time.perf_counter()
                    slot.env.restore_state(candidate["snapshot"])
                    snapshot_time_s += float(time.perf_counter() - snapshot_started)
                    plan_state = np.asarray(
                        candidate["state"], dtype=np.float32
                    ).copy()
                    policy_decision = int(candidate["policy_decisions"])
                    candidate["policy_decisions"] = policy_decision + 1

                    for chunk_action_index in range(int(action_count)):
                        action = np.asarray(
                            actions[batch_idx, chunk_action_index],
                            dtype=np.float32,
                        )
                        env_step_started = time.perf_counter()
                        with _silence_output(bool(progress_only)):
                            (
                                next_obs,
                                reward,
                                terminated,
                                truncated,
                                next_info,
                            ) = slot.env.step(action)
                        env_step_time_s += float(
                            time.perf_counter() - env_step_started
                        )
                        trajectory = candidate["trajectory"]
                        trajectory.append(
                            {
                                "state": plan_state.copy(),
                                "prior": np.asarray(
                                    priors[batch_idx], dtype=np.float32
                                ).copy(),
                                "action_index": int(chunk_action_index),
                                "action": action.copy(),
                                "old_log_prob": float(
                                    old_log_probs[batch_idx, chunk_action_index]
                                ),
                                "reward": 0.0,
                                "selected": False,
                                "candidate_success": False,
                                "candidate_done": bool(terminated or truncated),
                                "mean_action": np.asarray(
                                    means[batch_idx, chunk_action_index],
                                    dtype=np.float32,
                                ).copy(),
                                "trajectory_index": int(
                                    candidate["candidate_index"]
                                ),
                                "trajectory_step": int(len(trajectory) - 1),
                                "policy_decision_index": policy_decision,
                                "chunk_action_index": int(chunk_action_index),
                            }
                        )
                        candidate["obs"] = next_obs
                        candidate["info"] = dict(next_info)
                        candidate["state"] = layout.flatten(next_obs)
                        candidate["instruction"] = _safe_instruction(next_info)
                        candidate["final_reward"] = float(reward)
                        candidate["terminated"] = bool(terminated)
                        candidate["truncated"] = bool(truncated)
                        if bool(terminated or truncated):
                            break

                    snapshot_started = time.perf_counter()
                    candidate["snapshot"] = slot.env.capture_state()
                    snapshot_time_s += float(time.perf_counter() - snapshot_started)

    local_trajectories: dict[int, list[dict[str, Any]]] = {}
    local_summaries: list[dict[str, Any]] = []
    for candidate_idx in candidate_indices:
        candidate = candidates[int(candidate_idx)]
        trajectory = candidate["trajectory"]
        info = dict(candidate["info"])
        terminated = bool(candidate["terminated"])
        truncated = bool(candidate["truncated"])
        success = bool(info.get("success", False))
        outcome = 1.0 if success else 0.0
        if not bool(terminated or truncated):
            truncated = True
            info.update(
                {
                    "truncated": True,
                    "env_done": True,
                    "trajectory_horizon_truncated": True,
                }
            )
        for record in trajectory:
            record["reward"] = float(outcome)
            record["candidate_success"] = bool(success)
            record["trajectory_length"] = int(len(trajectory))
        local_trajectories[int(candidate_idx)] = trajectory
        local_summaries.append(
            {
                "candidate_index": int(candidate_idx),
                "outcome": float(outcome),
                "reward": float(candidate["final_reward"]),
                "info": info,
                "instruction": str(candidate["instruction"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "length": int(len(trajectory)),
                "policy_decisions": int(candidate["policy_decisions"]),
                "last_action": (
                    np.asarray(trajectory[-1]["action"], dtype=np.float32).copy()
                    if trajectory
                    else np.zeros((int(args.action_dim),), dtype=np.float32)
                ),
            }
        )

    timings = {
        "prior_inference_time_s": float(
            camera_render_time_s + prior_model_time_s
        ),
        "camera_render_time_s": float(camera_render_time_s),
        "prior_model_time_s": float(prior_model_time_s),
        "residual_inference_time_s": float(residual_inference_time_s),
        "env_step_time_s": float(env_step_time_s),
        "snapshot_time_s": float(snapshot_time_s),
        "candidate_inference_batches": float(inference_batches),
        "candidate_inference_batch_items": float(inference_batch_items),
        "candidate_inference_batch_size_max": float(inference_batch_size_max),
        "candidate_inference_batch_size_mean": float(
            inference_batch_items / max(1, inference_batches)
        ),
    }
    return local_trajectories, local_summaries, timings


def _evaluate_distributed_trajectory_group(
    *,
    trainer: SmolVLAGRPOTrainer,
    runtime: Any,
    slot: EnvSlot,
    layout: CDPRStateLayout,
    args: argparse.Namespace,
    progress_only: bool,
    dist_ctx: DistributedContext,
) -> tuple[list[dict[str, Any]], int, dict[str, Any], dict[str, Any]]:
    """Evaluate disjoint continuations on each rank and form one global GRPO group."""
    if not dist_ctx.enabled:
        return _evaluate_trajectory_group(
            trainer=trainer,
            runtime=runtime,
            slot=slot,
            layout=layout,
            args=args,
            progress_only=progress_only,
        )

    group_started = time.perf_counter()
    snapshot = slot.env.capture_state()
    group_size = max(2, int(args.grpo_group_size))
    max_action_steps = _trajectory_decision_horizon(slot.info, args)
    candidate_indices = _distributed_candidate_indices(
        group_size, dist_ctx.world_size, dist_ctx.rank
    )
    local_trajectories: dict[int, list[dict[str, Any]]] = {}
    local_summaries: list[dict[str, Any]] = []
    prior_inference_time_s = 0.0
    env_step_time_s = 0.0
    candidate_timings: dict[str, float] = {
        "camera_render_time_s": 0.0,
        "prior_model_time_s": 0.0,
        "residual_inference_time_s": 0.0,
        "snapshot_time_s": 0.0,
        "candidate_inference_batches": 0.0,
        "candidate_inference_batch_items": 0.0,
        "candidate_inference_batch_size_max": 1.0,
        "candidate_inference_batch_size_mean": 1.0,
    }

    batched_partition = _rollout_candidate_partition_batched(
        trainer=trainer,
        runtime=runtime,
        slot=slot,
        layout=layout,
        args=args,
        progress_only=progress_only,
        base_snapshot=snapshot,
        candidate_indices=candidate_indices,
        max_action_steps=max_action_steps,
    )
    candidate_indices_to_roll = list(candidate_indices)
    if batched_partition is not None:
        local_trajectories, local_summaries, candidate_timings = batched_partition
        prior_inference_time_s = float(
            candidate_timings["prior_inference_time_s"]
        )
        env_step_time_s = float(candidate_timings["env_step_time_s"])
        candidate_indices_to_roll = []

    for candidate_idx in candidate_indices_to_roll:
        slot.env.restore_state(snapshot)
        obs = slot.obs
        info = dict(slot.info)
        state = np.asarray(slot.state, dtype=np.float32).copy()
        instruction = str(slot.instruction)
        trajectory: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        final_reward = 0.0
        policy_decision_count = 0
        while len(trajectory) < max_action_steps:
            prior_started = time.perf_counter()
            prior = _sample_prior_for_observation(
                runtime,
                env=slot.env,
                obs=obs,
                info=info,
                instruction=instruction,
                progress_only=progress_only,
            )
            prior_inference_time_s += float(time.perf_counter() - prior_started)
            plan_state = state.copy()
            action_count = min(
                int(args.replan_every), max_action_steps - len(trajectory)
            )
            actions, old_log_probs, means = trainer.sample_action_chunk(
                state=state,
                prior=prior,
                action_count=action_count,
            )
            current_policy_decision = int(policy_decision_count)
            policy_decision_count += 1
            for chunk_action_index in range(action_count):
                action = np.asarray(
                    actions[chunk_action_index], dtype=np.float32
                )
                env_step_started = time.perf_counter()
                with _silence_output(bool(progress_only)):
                    next_obs, reward, terminated, truncated, next_info = slot.env.step(
                        action
                    )
                env_step_time_s += float(time.perf_counter() - env_step_started)
                trajectory.append(
                    {
                        "state": plan_state.copy(),
                        "prior": prior.copy(),
                        "action_index": int(chunk_action_index),
                        "action": action.copy(),
                        "old_log_prob": float(old_log_probs[chunk_action_index]),
                        "reward": 0.0,
                        "selected": False,
                        "candidate_success": False,
                        "candidate_done": bool(terminated or truncated),
                        "mean_action": np.asarray(
                            means[chunk_action_index], dtype=np.float32
                        ).copy(),
                        "trajectory_index": int(candidate_idx),
                        "trajectory_step": int(len(trajectory) - 1),
                        "policy_decision_index": current_policy_decision,
                        "chunk_action_index": int(chunk_action_index),
                    }
                )
                obs = next_obs
                info = dict(next_info)
                state = layout.flatten(next_obs)
                instruction = _safe_instruction(info)
                final_reward = float(reward)
                if bool(terminated or truncated):
                    break
            if bool(terminated or truncated):
                break

        success = bool(info.get("success", False))
        outcome = 1.0 if success else 0.0
        if not bool(terminated or truncated):
            truncated = True
            info.update(
                {
                    "truncated": True,
                    "env_done": True,
                    "trajectory_horizon_truncated": True,
                }
            )
        for record in trajectory:
            record["reward"] = float(outcome)
            record["candidate_success"] = bool(success)
            record["trajectory_length"] = int(len(trajectory))
        local_trajectories[int(candidate_idx)] = trajectory
        local_summaries.append(
            {
                "candidate_index": int(candidate_idx),
                "outcome": float(outcome),
                "reward": float(final_reward),
                "info": info,
                "instruction": instruction,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "length": int(len(trajectory)),
                "policy_decisions": int(policy_decision_count),
                "last_action": (
                    np.asarray(trajectory[-1]["action"], dtype=np.float32).copy()
                    if trajectory
                    else np.zeros((int(args.action_dim),), dtype=np.float32)
                ),
            }
        )

    local_payload = {
        "rank": int(dist_ctx.rank),
        "candidate_indices": list(candidate_indices),
        "candidates": local_summaries,
        "prior_inference_time_s": float(prior_inference_time_s),
        "env_step_time_s": float(env_step_time_s),
        "camera_render_time_s": float(
            candidate_timings["camera_render_time_s"]
        ),
        "prior_model_time_s": float(candidate_timings["prior_model_time_s"]),
        "residual_inference_time_s": float(
            candidate_timings["residual_inference_time_s"]
        ),
        "snapshot_time_s": float(candidate_timings["snapshot_time_s"]),
        "candidate_inference_batches": float(
            candidate_timings["candidate_inference_batches"]
        ),
        "candidate_inference_batch_items": float(
            candidate_timings["candidate_inference_batch_items"]
        ),
        "candidate_inference_batch_size_max": float(
            candidate_timings["candidate_inference_batch_size_max"]
        ),
        "candidate_inference_batch_size_mean": float(
            candidate_timings["candidate_inference_batch_size_mean"]
        ),
        "local_rollout_time_s": float(time.perf_counter() - group_started),
        "sampled_policy_decisions": int(
            sum(int(item["policy_decisions"]) for item in local_summaries)
        ),
        "sampled_environment_actions": int(
            sum(int(item["length"]) for item in local_summaries)
        ),
    }
    rank_payloads = _dist_all_gather_object(local_payload, dist_ctx)
    summaries = sorted(
        [
            dict(candidate)
            for payload in rank_payloads
            for candidate in payload["candidates"]
        ],
        key=lambda item: int(item["candidate_index"]),
    )
    observed_indices = [int(item["candidate_index"]) for item in summaries]
    expected_indices = list(range(group_size))
    if observed_indices != expected_indices:
        raise RuntimeError(
            "Distributed GRPO candidate partition is incomplete or overlapping: "
            f"expected={expected_indices}, observed={observed_indices}"
        )

    outcome_arr = np.asarray([item["outcome"] for item in summaries], dtype=np.float32)
    advantages = _group_advantages(outcome_arr, args)
    informative = _trajectory_group_is_informative(outcome_arr, args)
    selected_value = _select_candidate(outcome_arr, args) if dist_ctx.is_main else None
    selected = int(_dist_broadcast_object(selected_value, dist_ctx, src=0))

    records: list[dict[str, Any]] = []
    for candidate_idx, trajectory in local_trajectories.items():
        for record in trajectory:
            record["advantage"] = float(advantages[candidate_idx])
            record["selected"] = bool(candidate_idx == selected)
            if informative:
                records.append(record)

    slot.env.restore_state(snapshot)
    selected_summary = dict(summaries[selected])
    selected_state = {
        **selected_summary,
        # Trajectory groups are terminal continuations and reset immediately. Keeping
        # the compact base observation avoids gathering camera-sized data every group.
        "obs": slot.obs,
        "state": np.asarray(slot.state, dtype=np.float32).copy(),
    }
    lengths = np.asarray([int(item["length"]) for item in summaries], dtype=np.int64)
    policy_decision_counts = np.asarray(
        [int(item["policy_decisions"]) for item in summaries], dtype=np.int64
    )
    success_count = int(outcome_arr.sum())
    group_wall_times = _dist_all_gather_object(
        float(time.perf_counter() - group_started), dist_ctx
    )
    group_wall_time_s = max(1e-9, max(float(value) for value in group_wall_times))
    candidate_counts = [len(payload["candidate_indices"]) for payload in rank_payloads]
    decision_counts = [
        int(payload["sampled_policy_decisions"]) for payload in rank_payloads
    ]
    rank_rollout_times = [
        float(payload["local_rollout_time_s"]) for payload in rank_payloads
    ]
    min_rank_rollout_time = max(1e-9, min(rank_rollout_times))
    candidate_inference_batches = sum(
        float(payload["candidate_inference_batches"])
        for payload in rank_payloads
    )
    candidate_inference_batch_items = sum(
        float(payload["candidate_inference_batch_items"])
        for payload in rank_payloads
    )
    accepted_policy_records = int(lengths.sum()) if informative else 0
    group_stats = {
        "group_size": int(group_size),
        "instruction_type": str(_info_instruction_type(slot.info, slot.instruction)),
        "curriculum_shell": int(slot.info.get("curriculum_shell", -1)),
        "curriculum_shell_count": int(slot.info.get("curriculum_shell_count", 0)),
        "curriculum_shell_policy_steps_low": int(
            slot.info.get("curriculum_shell_policy_steps_low", 1)
        ),
        "candidate_reward_mean": float(outcome_arr.mean()),
        "candidate_reward_std": float(outcome_arr.std()),
        "candidate_reward_max": float(outcome_arr.max()),
        "candidate_reward_min": float(outcome_arr.min()),
        "candidate_selected_reward": float(outcome_arr[selected]),
        "candidate_selected_index": int(selected),
        "candidate_success_count": float(success_count),
        "candidate_binary_reward_rate": 1.0,
        "zero_advantage_group": float(float(outcome_arr.std()) <= 1e-8),
        "informative_group": float(informative),
        "all_fail_group": float(success_count == 0),
        "all_success_group": float(success_count == group_size),
        "sampled_policy_decisions": float(policy_decision_counts.sum()),
        "sampled_environment_actions": float(lengths.sum()),
        "selected_environment_actions": float(lengths[selected]),
        "trajectory_work_amplification": float(
            lengths.sum() / max(1, int(lengths[selected]))
        ),
        "accepted_policy_records": float(accepted_policy_records),
        "trajectory_length_mean": float(lengths.mean()),
        "trajectory_length_min": float(lengths.min()),
        "trajectory_length_max": float(lengths.max()),
        "trajectory_policy_decisions_mean": float(policy_decision_counts.mean()),
        "trajectory_policy_decisions_min": float(policy_decision_counts.min()),
        "trajectory_policy_decisions_max": float(policy_decision_counts.max()),
        "executed_actions_per_policy_decision": float(
            lengths.sum() / max(1, int(policy_decision_counts.sum()))
        ),
        "trajectory_decision_horizon": float(
            math.ceil(max_action_steps / max(1, int(args.replan_every)))
        ),
        "trajectory_action_horizon": float(max_action_steps),
        "group_wall_time_s": float(group_wall_time_s),
        "prior_inference_time_s": float(
            max(float(payload["prior_inference_time_s"]) for payload in rank_payloads)
        ),
        "env_step_time_s": float(
            max(float(payload["env_step_time_s"]) for payload in rank_payloads)
        ),
        "camera_render_time_s": float(
            max(float(payload["camera_render_time_s"]) for payload in rank_payloads)
        ),
        "prior_model_time_s": float(
            max(float(payload["prior_model_time_s"]) for payload in rank_payloads)
        ),
        "residual_inference_time_s": float(
            max(
                float(payload["residual_inference_time_s"])
                for payload in rank_payloads
            )
        ),
        "snapshot_time_s": float(
            max(float(payload["snapshot_time_s"]) for payload in rank_payloads)
        ),
        "distributed_sync_time_s": float(
            max(0.0, group_wall_time_s - max(rank_rollout_times))
        ),
        "candidate_inference_batches": float(candidate_inference_batches),
        "candidate_inference_batch_size_mean": float(
            candidate_inference_batch_items
            / max(1.0, candidate_inference_batches)
        ),
        "candidate_inference_batch_size_max": float(
            max(
                float(payload["candidate_inference_batch_size_max"])
                for payload in rank_payloads
            )
        ),
        "sampled_policy_decisions_per_second": float(
            policy_decision_counts.sum() / group_wall_time_s
        ),
        "sampled_environment_actions_per_second": float(
            lengths.sum() / group_wall_time_s
        ),
        "selected_environment_actions_per_second": float(
            lengths[selected] / group_wall_time_s
        ),
        "trajectory_terminated_count": float(
            sum(bool(item["terminated"]) for item in summaries)
        ),
        "trajectory_truncated_count": float(
            sum(bool(item["truncated"]) for item in summaries)
        ),
        "distributed_candidate_parallelism": float(dist_ctx.world_size),
        "candidates_per_rank_min": float(min(candidate_counts)),
        "candidates_per_rank_max": float(max(candidate_counts)),
        "candidate_load_imbalance": float(max(candidate_counts) - min(candidate_counts)),
        "decisions_per_rank_min": float(min(decision_counts)),
        "decisions_per_rank_max": float(max(decision_counts)),
        "rank_rollout_time_s_min": float(min(rank_rollout_times)),
        "rank_rollout_time_s_max": float(max(rank_rollout_times)),
        "rank_straggler_ratio": float(max(rank_rollout_times) / min_rank_rollout_time),
    }
    _retry_options, retry_kind = _dynamic_frontier_retry_options(group_stats, args)
    group_stats.update(
        {
            "easier_shell_retry": float(retry_kind == "easier_shell"),
            "shell0_easiest_retry": float(retry_kind == "shell0_easiest"),
            "harder_shell_retry": float(retry_kind == "harder_shell"),
        }
    )
    return records, selected, group_stats, selected_state


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
    synchronized_trajectory_groups = bool(
        args.grpo_trajectory_groups and dist_ctx.enabled
    )

    run_dir = _make_run_dir(args.run_root_dir, args.run_id)
    if dist_ctx.is_main:
        _write_json(run_dir / "config.json", vars(args))
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard")) if dist_ctx.is_main and SummaryWriter is not None else None
    metrics_path = run_dir / "metrics.jsonl"

    metrics_window = max(1, int(args.metrics_window_episodes))
    curriculum_seed = int(args.seed) if synchronized_trajectory_groups else int(rollout_seed)
    complex_runtime = SmolVLAComplexRuntime(args=args, seed=curriculum_seed)
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
            mask_empty_aux_camera=bool(
                getattr(args, "mask_empty_aux_camera", False)
            ),
            chunk_size=int(args.chunk_size),
            action_dim=int(args.action_dim),
            action_indices=None
            if args.smolvla_action_indices is None
            else tuple(int(v) for v in args.smolvla_action_indices),
            action_normalization=str(args.smolvla_action_normalization),
            model_image_size=(
                None
                if int(args.smolvla_model_image_size) <= 0
                else int(args.smolvla_model_image_size)
            ),
            compile_model=bool(args.smolvla_compile_model),
            compile_mode=str(args.smolvla_compile_mode),
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
            seed = (
                int(args.seed) + env_idx * 997
                if synchronized_trajectory_groups
                else int(rollout_seed) + env_idx * 997
            )
            with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                env = _build_env(args, seed=seed)
                if dense_curriculum_active:
                    dense_stage_instruction_types = _apply_dense_stage_to_envs([env], args, dense_stage_index)
            if synchronized_trajectory_groups:
                main_options = None
                if dist_ctx.is_main:
                    main_options = (
                        complex_runtime.reset_options()
                        if complex_training_active
                        else mixed_sampler.reset_options(dense_stage_index)
                    )
                reset_payload = _synchronized_reset_payload(
                    ctx=dist_ctx,
                    options=main_options,
                    seed=seed,
                )
                reset_options = reset_payload["options"]
                seed = int(reset_payload["seed"])
            else:
                reset_options = (
                    complex_runtime.reset_options()
                    if complex_training_active
                    else mixed_sampler.reset_options(dense_stage_index)
                )
            with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                obs, info = env.reset(seed=seed, options=reset_options)
            if synchronized_trajectory_groups:
                obs, info = _synchronize_environment_reset_state(
                    env=env,
                    observation=obs,
                    info=info,
                    ctx=dist_ctx,
                )
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
        if bool(args.grpo_trajectory_groups) and complex_runtime.hindsight_enabled:
            raise ValueError(
                "Trajectory-group GRPO is currently scoped to Reverse Frontier; "
                "disable it for LC-HOL hindsight collection."
            )
        if synchronized_trajectory_groups:
            # Model construction and checkpoint loading need identical initialization.
            # Candidate sampling needs rank-local RNG streams afterwards.
            _set_seed(int(rollout_seed))
            candidate_plan = {
                rank: _distributed_candidate_indices(
                    int(args.grpo_group_size), int(dist_ctx.world_size), rank
                )
                for rank in range(int(dist_ctx.world_size))
            }
            _log(
                dist_ctx,
                "[smolvla-grpo] Synchronized distributed trajectory collection active: "
                f"{candidate_plan}",
            )

        manifest = {
            "policy_type": "smolvla_cdpr_grpo",
            "base_checkpoint": str(args.base_checkpoint),
            "run_dir": run_dir.as_posix(),
            "config": str(args.config or ""),
            "action_keys": ["x", "y", "z", "yaw", "gripper"],
            "chunk_size": int(args.chunk_size),
            "replan_every": int(args.replan_every),
            "smolvla_model_image_size": int(args.smolvla_model_image_size),
            "smolvla_compile_model": bool(args.smolvla_compile_model),
            "smolvla_compile_mode": str(args.smolvla_compile_mode),
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
                "trajectory_groups": bool(args.grpo_trajectory_groups),
                "trajectory_max_decisions": int(args.grpo_trajectory_max_decisions),
                "trajectory_shell_aware_horizon": bool(
                    args.grpo_trajectory_shell_aware_horizon
                ),
                "trajectory_horizon_multiplier": float(
                    args.grpo_trajectory_horizon_multiplier
                ),
                "trajectory_horizon_grace_decisions": int(
                    args.grpo_trajectory_horizon_grace_decisions
                ),
                "candidate_inference_batch_size": int(
                    args.grpo_candidate_inference_batch_size
                ),
                "dynamic_sampling": bool(args.grpo_dynamic_sampling),
                "dynamic_min_pass_rate": float(args.grpo_dynamic_min_pass_rate),
                "dynamic_max_pass_rate": float(args.grpo_dynamic_max_pass_rate),
                "target_records_per_update": int(args.grpo_target_records_per_update),
                "max_groups_per_update": int(args.grpo_max_groups_per_update),
                "max_collection_seconds_per_update": float(
                    args.grpo_max_collection_seconds_per_update
                ),
                "normalize_group_advantage": bool(args.grpo_normalize_group_advantage),
                "clip_advantage_abs": float(args.grpo_clip_advantage_abs),
                "clip_range": float(args.clip_range),
                "clip_range_low": float(args.clip_range_low),
                "clip_range_high": float(args.clip_range_high),
                "entropy_coef": float(args.entropy_coef),
                "ppo_epochs": int(args.ppo_epochs),
                "minibatch_size": int(args.minibatch_size),
                "microbatch_size": int(args.microbatch_size),
                "distributed_candidate_collection": bool(
                    synchronized_trajectory_groups
                ),
                "candidates_per_rank": {
                    str(rank): _distributed_candidate_indices(
                        int(args.grpo_group_size), int(dist_ctx.world_size), rank
                    )
                    for rank in range(int(dist_ctx.world_size))
                },
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
        last_log_step = int(start_step)
        last_status_step = int(start_step)
        last_save_step = int(start_step)
        trajectory_groups_enabled = bool(args.grpo_trajectory_groups)
        _log(dist_ctx, f"[smolvla-grpo] Startup ready in {time.perf_counter() - startup_t0:.1f}s")

        while global_step < int(args.max_train_steps):
            rollout_records: list[dict[str, Any]] = []
            rollout_group_stats: list[dict[str, Any]] = []
            rollout_iteration = 0
            groups_attempted = 0
            accepted_records_collected = 0
            collection_started = time.perf_counter()
            collection_time_limit_reached = False
            target_records = int(args.grpo_target_records_per_update)
            if target_records <= 0:
                target_records = int(args.batch_size or args.rollout_steps)
            while True:
                if global_step >= int(args.max_train_steps):
                    break
                if trajectory_groups_enabled:
                    if collection_time_limit_reached:
                        break
                    if groups_attempted >= int(args.grpo_max_groups_per_update):
                        break
                    if accepted_records_collected >= max(1, target_records):
                        break
                elif rollout_iteration >= max(1, int(args.rollout_steps)):
                    break
                rollout_iteration += 1
                for slot_idx, slot in enumerate(slots):
                    if global_step >= int(args.max_train_steps):
                        break
                    if trajectory_groups_enabled and (
                        collection_time_limit_reached
                        or groups_attempted >= int(args.grpo_max_groups_per_update)
                        or accepted_records_collected >= max(1, target_records)
                    ):
                        break
                    source_instruction_type = _info_instruction_type(slot.info, slot.instruction)
                    if trajectory_groups_enabled:
                        trajectory_evaluator = (
                            _evaluate_distributed_trajectory_group
                            if synchronized_trajectory_groups
                            else _evaluate_trajectory_group
                        )
                        evaluator_kwargs: dict[str, Any] = {
                            "trainer": trainer,
                            "runtime": runtime,
                            "slot": slot,
                            "layout": layout,
                            "args": args,
                            "progress_only": bool(args.progress_only)
                            or not dist_ctx.is_main,
                        }
                        if synchronized_trajectory_groups:
                            evaluator_kwargs["dist_ctx"] = dist_ctx
                        records, selected_idx, group_stats, selected_state = (
                            trajectory_evaluator(
                                **evaluator_kwargs,
                            )
                        )
                        groups_attempted += 1
                        accepted_records_collected += int(
                            group_stats.get("accepted_policy_records", 0)
                        )
                        collection_limit_seconds = float(
                            args.grpo_max_collection_seconds_per_update
                        )
                        if collection_limit_seconds > 0.0:
                            limit_reached_value = (
                                time.perf_counter() - collection_started
                                >= collection_limit_seconds
                                if dist_ctx.is_main
                                else None
                            )
                            collection_time_limit_reached = bool(
                                _dist_broadcast_object(
                                    limit_reached_value,
                                    dist_ctx,
                                    src=0,
                                )
                            )
                        selected_action = np.asarray(
                            selected_state["last_action"], dtype=np.float32
                        )
                        next_obs = selected_state["obs"]
                        next_info = dict(selected_state["info"])
                        next_state = np.asarray(selected_state["state"], dtype=np.float32)
                        reward = float(selected_state["outcome"])
                        terminated = bool(selected_state["terminated"])
                        truncated = bool(selected_state["truncated"])
                        done = True
                        episode_success = bool(selected_state["outcome"] >= 0.5)
                        step_increment = max(1, int(selected_state["length"]))
                        pre_action_snapshot = None
                    else:
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
                        selected_action = np.asarray(
                            records[selected_idx]["action"], dtype=np.float32
                        )
                        pre_action_snapshot = slot.env.capture_state()
                        with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                            next_obs, reward, terminated, truncated, next_info = slot.env.step(
                                selected_action
                            )
                        next_state = layout.flatten(next_obs)
                        done = bool(terminated or truncated)
                        episode_success = bool(next_info.get("success", False)) if done else False
                        step_increment = 1
                    for record in records:
                        record["instruction_type"] = str(source_instruction_type)
                    rollout_records.extend(records)
                    rollout_group_stats.append(group_stats)
                    global_step += int(step_increment)
                    slot.episode_length += int(step_increment)
                    slot.episode_reward += float(reward)

                    if complex_runtime.hindsight_enabled:
                        if pre_action_snapshot is None:
                            raise RuntimeError(
                                "LC-HOL hindsight collection requires one-step GRPO rollouts."
                            )
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

                    if dist_ctx.is_main and _step_interval_due(
                        global_step=global_step,
                        last_step=last_log_step,
                        every=int(args.log_every_steps),
                    ):
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
                        trajectory_group_scalars = _trajectory_group_metric_scalars(
                            rollout_group_stats,
                            group_size=int(args.grpo_group_size),
                        ) if trajectory_groups_enabled else {}
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
                            **{
                                key.removeprefix("rollout/"): value
                                for key, value in trajectory_group_scalars.items()
                            },
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
                            "rollout/grpo_records_pending": float(
                                accepted_records_collected
                                if trajectory_groups_enabled
                                else len(rollout_records)
                            ),
                            **trajectory_group_scalars,
                            **{f"gpu/{k}": v for k, v in _gpu_metrics(device).items()},
                        }
                        rollout_scalars.update(_episode_metric_scalars(episode_metrics))
                        rollout_scalars.update(_dense_stage_metric_scalars(dense_stage_metrics))
                        rollout_scalars.update(complex_runtime.metrics(consume_interval_counts=True))
                        _log_scalars(writer, rollout_scalars, global_step)
                        if trajectory_group_scalars:
                            metric = trajectory_group_scalars.get
                            _log(
                                dist_ctx,
                                "[smolvla-grpo] rollout profile "
                                f"selected_step_s={metric('rollout/grpo_selected_environment_actions_per_second', 0.0):.2f} "
                                f"sampled_action_s={metric('rollout/grpo_sampled_environment_actions_per_second', 0.0):.2f} "
                                f"work_amp={metric('rollout/grpo_trajectory_work_amplification', 0.0):.2f}x "
                                f"vla_batch={metric('rollout/grpo_candidate_inference_batch_size_mean', 0.0):.2f} "
                                f"render={metric('rollout/grpo_camera_render_wall_fraction', 0.0):.1%} "
                                f"prior={metric('rollout/grpo_prior_model_wall_fraction', 0.0):.1%} "
                                f"env={metric('rollout/grpo_env_step_wall_fraction', 0.0):.1%} "
                                f"sync={metric('rollout/grpo_distributed_sync_wall_fraction', 0.0):.1%}",
                                progress=progress,
                            )
                        if not complex_training_active:
                            _log_mixed_curriculum_scalars(writer, mixed_sampler.snapshot(), global_step)
                        last_log_step = int(global_step)

                    if progress is not None:
                        progress.update(int(step_increment))
                        if _step_interval_due(
                            global_step=global_step,
                            last_step=last_status_step,
                            every=int(args.status_every_steps),
                        ) or done:
                            _progress_postfix(
                                progress,
                                episode=slot.episode,
                                episode_length=slot.episode_length,
                                episode_reward=slot.episode_reward,
                                reward=float(reward),
                                buffer_size=(
                                    accepted_records_collected
                                    if trajectory_groups_enabled
                                    else len(rollout_records)
                                ),
                                instruction=slot.instruction,
                                world_size=dist_ctx.world_size,
                                num_envs=len(slots),
                                global_step=global_step,
                                start_step=int(start_step),
                                max_train_steps=int(args.max_train_steps),
                            )
                            last_status_step = int(global_step)
                    elif dist_ctx.is_main and _step_interval_due(
                        global_step=global_step,
                        last_step=last_status_step,
                        every=int(args.status_every_steps),
                    ):
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
                        last_status_step = int(global_step)

                    if dist_ctx.is_main and _step_interval_due(
                        global_step=global_step,
                        last_step=last_save_step,
                        every=int(args.save_every_steps),
                    ):
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
                        last_save_step = int(global_step)

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
                        if synchronized_trajectory_groups:
                            synchronized_runtime_state = _dist_broadcast_object(
                                complex_runtime.state_dict()
                                if dist_ctx.is_main
                                else None,
                                dist_ctx,
                                src=0,
                            )
                            if not dist_ctx.is_main:
                                complex_runtime.load_state_dict(
                                    dict(synchronized_runtime_state or {})
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
                        retry_options, _retry_kind = _dynamic_frontier_retry_options(
                            group_stats,
                            args,
                        )
                        if synchronized_trajectory_groups:
                            main_options = None
                            if dist_ctx.is_main:
                                if retry_options is not None:
                                    main_options = retry_options
                                else:
                                    main_options = (
                                        complex_runtime.reset_options()
                                        if complex_training_active
                                        else mixed_sampler.reset_options(dense_stage_index)
                                    )
                            reset_payload = _synchronized_reset_payload(
                                ctx=dist_ctx,
                                options=main_options,
                                seed=(
                                    int(args.seed)
                                    + 1_000_003
                                    + int(slot_idx) * 100_003
                                    + int(global_step) * 997
                                )
                                % (2**32 - 1),
                            )
                            reset_options = reset_payload["options"]
                            reset_seed = int(reset_payload["seed"])
                        elif retry_options is not None:
                            reset_options = retry_options
                            reset_seed = None
                        else:
                            reset_options = (
                                complex_runtime.reset_options()
                                if complex_training_active
                                else mixed_sampler.reset_options(dense_stage_index)
                            )
                            reset_seed = None
                        with _silence_output(bool(args.progress_only) or not dist_ctx.is_main):
                            obs, info = slot.env.reset(
                                seed=reset_seed,
                                options=reset_options,
                            )
                        if synchronized_trajectory_groups:
                            obs, info = _synchronize_environment_reset_state(
                                env=slot.env,
                                observation=obs,
                                info=info,
                                ctx=dist_ctx,
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

            collection_wall_time_s = float(time.perf_counter() - collection_started)
            if dist_ctx.is_main:
                _log_scalars(
                    writer,
                    {
                        "rollout/grpo_collection_wall_time_s": collection_wall_time_s,
                        "rollout/grpo_collection_groups": float(groups_attempted),
                        "rollout/grpo_collection_accepted_records": float(
                            accepted_records_collected
                        ),
                        "rollout/grpo_collection_time_limit_reached": float(
                            collection_time_limit_reached
                        ),
                    },
                    global_step,
                )

            if synchronized_trajectory_groups and accepted_records_collected > 0:
                rank_record_batches = _dist_all_gather_object(
                    rollout_records, dist_ctx
                )
                rollout_records = [
                    record
                    for rank_records in rank_record_batches
                    for record in rank_records
                ]
                if len(rollout_records) != int(accepted_records_collected):
                    raise RuntimeError(
                        "Distributed GRPO record gather disagrees with accepted count: "
                        f"accepted={accepted_records_collected}, gathered={len(rollout_records)}"
                    )

            if rollout_records:
                hindsight_records = complex_runtime.sample_hindsight(len(rollout_records))
                update_started = time.perf_counter()
                last_metrics = trainer.update(
                    rollout_records,
                    hindsight_records=hindsight_records,
                )
                update_wall_time_s = float(time.perf_counter() - update_started)
                last_metrics.update(
                    {
                        "optimizer_update_wall_time_s": update_wall_time_s,
                        "collection_wall_time_s": collection_wall_time_s,
                        "collection_to_update_time_ratio": float(
                            collection_wall_time_s / max(1e-9, update_wall_time_s)
                        ),
                        "collection_groups": float(groups_attempted),
                        "collection_accepted_records": float(
                            accepted_records_collected
                        ),
                        "collection_time_limit_reached": float(
                            collection_time_limit_reached
                        ),
                    }
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
