#!/usr/bin/env python3
from __future__ import annotations

import atexit
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import types
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CDPR_CURRICULUM_OPTIONS = (
    "move_to_object",
    "grab_object",
    "pick_up",
    "push_left",
    "push_right",
    "push_forward",
    "push_backward",
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_in_front_of_object",
    "move_behind_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
)


@dataclass(frozen=True)
class _FastWrapperArgs:
    tensorboard_rollout_every_global_steps: int = 0
    tensorboard_metric_profile: str = "compact"
    rollout_image_size: int = 0
    resume_actor_stats: bool = True
    first_stage_grpo_actor_stats_path: Path | None = None
    second_stage_grpo_actor_stats_path: Path | None = None
    sparse_stage_init_log_std: float | None = None
    ddp_timeout_seconds: int = 0
    ddp_rollout_sync_interval: int = 0
    lr_scheduler: str = "constant"
    lr_warmup_updates: int = 0
    lr_min_factor: float = 1.0
    max_train_reset_attempts: int = 10
    lchol: "_LCHOLWrapperArgs | None" = None


@dataclass(frozen=True)
class _LCHOLWrapperArgs:
    enabled: bool = False
    group_score: str = "phase_shaped"
    hindsight_bc_coef: float = 0.20
    hindsight_done_weight: float = 0.20
    hindsight_replay_capacity: int = 20_000
    hindsight_replay_ratio: float = 0.50
    hindsight_prefix_max_steps: int = 16
    option_prior_bc_coef: float = 0.20
    option_prior_min_coef: float = 0.035
    option_prior_decay_updates: int = 80
    curriculum: str = "strict_staged"
    strict_min_success_samples: int = 24
    weakest_mode_oversample_strength: float = 2.5
    newest_stage_weight: float = 1.4
    reverse_promotion_success: float = 0.50
    reverse_demotion_success: float = 0.20
    reverse_validation_rollouts_per_shell: int = 50
    reverse_min_train_updates_before_validation: int = 5
    reverse_max_shell_jump: int = 1
    reverse_saturation_abort_threshold: float = 0.30
    reverse_sample_frontier_probability: float = 0.80
    reverse_sample_rehearsal_probability: float = 0.20


@dataclass(frozen=True)
class _ResumeArtifacts:
    checkpoint_dir: Path | None = None
    actor_stats_path: Path | None = None


_LCHOL_VALUE_FIELDS: dict[str, tuple[str, type]] = {
    "lchol_group_score": ("group_score", str),
    "lchol_hindsight_bc_coef": ("hindsight_bc_coef", float),
    "lchol_hindsight_done_weight": ("hindsight_done_weight", float),
    "lchol_hindsight_replay_capacity": ("hindsight_replay_capacity", int),
    "lchol_hindsight_replay_ratio": ("hindsight_replay_ratio", float),
    "lchol_hindsight_prefix_max_steps": ("hindsight_prefix_max_steps", int),
    "lchol_option_prior_bc_coef": ("option_prior_bc_coef", float),
    "lchol_option_prior_min_coef": ("option_prior_min_coef", float),
    "lchol_option_prior_decay_updates": ("option_prior_decay_updates", int),
    "lchol_curriculum": ("curriculum", str),
    "lchol_strict_min_success_samples": ("strict_min_success_samples", int),
    "lchol_weakest_mode_oversample_strength": ("weakest_mode_oversample_strength", float),
    "lchol_newest_stage_weight": ("newest_stage_weight", float),
    "lchol_reverse_promotion_success": ("reverse_promotion_success", float),
    "lchol_reverse_demotion_success": ("reverse_demotion_success", float),
    "lchol_reverse_validation_rollouts_per_shell": ("reverse_validation_rollouts_per_shell", int),
    "lchol_reverse_min_train_updates_before_validation": ("reverse_min_train_updates_before_validation", int),
    "lchol_reverse_max_shell_jump": ("reverse_max_shell_jump", int),
    "lchol_reverse_saturation_abort_threshold": ("reverse_saturation_abort_threshold", float),
    "lchol_reverse_sample_frontier_probability": ("reverse_sample_frontier_probability", float),
    "lchol_reverse_sample_rehearsal_probability": ("reverse_sample_rehearsal_probability", float),
}


def _strip_lchol_prefix(flag: str) -> str:
    out = str(flag).lstrip("-")
    return out.replace("-", "_")


def _parse_lchol_bool(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean LC-HOL value, got {raw!r}.")


def _split_wrapper_argv(argv: Sequence[str]) -> tuple[Path | None, list[str], _FastWrapperArgs]:
    forwarded: list[str] = []
    external_script: Path | None = None
    tensorboard_rollout_every_global_steps = 0
    tensorboard_metric_profile = "compact"
    rollout_image_size = 0
    resume_actor_stats = True
    first_stage_grpo_actor_stats_path: Path | None = None
    second_stage_grpo_actor_stats_path: Path | None = None
    sparse_stage_init_log_std: float | None = None
    lr_scheduler = "constant"
    lr_warmup_updates = 0
    lr_min_factor = 1.0
    max_train_reset_attempts = 10
    try:
        ddp_timeout_seconds = max(0, int(os.environ.get("RLVLA_DDP_TIMEOUT_SECONDS", "0")))
    except ValueError:
        ddp_timeout_seconds = 0
    try:
        ddp_rollout_sync_interval = max(0, int(os.environ.get("RLVLA_DDP_ROLLOUT_SYNC_INTERVAL", "0")))
    except ValueError:
        ddp_rollout_sync_interval = 0
    lchol_values = dict(_LCHOLWrapperArgs().__dict__)

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--external_grpo_script":
            if idx + 1 >= len(argv):
                raise SystemExit("--external_grpo_script expects a path.")
            external_script = Path(argv[idx + 1]).expanduser().resolve()
            idx += 2
            continue
        if arg == "--tensorboard_rollout_every_global_steps":
            if idx + 1 >= len(argv):
                raise SystemExit("--tensorboard_rollout_every_global_steps expects an integer.")
            try:
                tensorboard_rollout_every_global_steps = max(0, int(argv[idx + 1]))
            except ValueError as exc:
                raise SystemExit("--tensorboard_rollout_every_global_steps expects an integer.") from exc
            idx += 2
            continue
        if arg in ("--tensorboard_metric_profile", "--tensorboard-metric-profile"):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects `compact` or `full`.")
            tensorboard_metric_profile = str(argv[idx + 1]).strip().lower()
            if tensorboard_metric_profile not in {"compact", "full"}:
                raise SystemExit(f"{arg} expects `compact` or `full`, got {argv[idx + 1]!r}.")
            idx += 2
            continue
        if arg in ("--rollout_image_size", "--rollout-image-size"):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects an integer image edge length, or 0 to disable.")
            try:
                rollout_image_size = max(0, int(argv[idx + 1]))
            except ValueError as exc:
                raise SystemExit(f"{arg} expects an integer image edge length, or 0 to disable.") from exc
            idx += 2
            continue
        if arg in ("--resume_actor_stats", "--resume-actor-stats"):
            resume_actor_stats = True
            idx += 1
            continue
        if arg in ("--no-resume_actor_stats", "--no-resume-actor-stats"):
            resume_actor_stats = False
            idx += 1
            continue
        if arg in (
            "--first_stage_grpo_actor_stats_path",
            "--first-stage-grpo-actor-stats-path",
            "--stage1_grpo_actor_stats_path",
            "--stage1-grpo-actor-stats-path",
            "--dense_stage_grpo_actor_stats_path",
            "--dense-stage-grpo-actor-stats-path",
        ):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects a path.")
            first_stage_grpo_actor_stats_path = Path(argv[idx + 1]).expanduser()
            idx += 2
            continue
        if arg in ("--sparse_stage_init_log_std", "--sparse-stage-init-log-std"):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects a float.")
            try:
                sparse_stage_init_log_std = float(argv[idx + 1])
            except ValueError as exc:
                raise SystemExit(f"{arg} expects a float.") from exc
            idx += 2
            continue
        if arg in (
            "--second_stage_grpo_actor_stats_path",
            "--second-stage-grpo-actor-stats-path",
            "--stage2_grpo_actor_stats_path",
            "--stage2-grpo-actor-stats-path",
            "--sparse_stage_grpo_actor_stats_path",
            "--sparse-stage-grpo-actor-stats-path",
            "--lchol_stage_grpo_actor_stats_path",
            "--lchol-stage-grpo-actor-stats-path",
        ):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects a path.")
            second_stage_grpo_actor_stats_path = Path(argv[idx + 1]).expanduser()
            idx += 2
            continue
        if arg in ("--ddp_timeout_seconds", "--ddp-timeout-seconds"):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects an integer.")
            try:
                ddp_timeout_seconds = max(0, int(argv[idx + 1]))
            except ValueError as exc:
                raise SystemExit(f"{arg} expects an integer.") from exc
            idx += 2
            continue
        if arg in ("--ddp_rollout_sync_interval", "--ddp-rollout-sync-interval"):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects an integer.")
            try:
                ddp_rollout_sync_interval = max(0, int(argv[idx + 1]))
            except ValueError as exc:
                raise SystemExit(f"{arg} expects an integer.") from exc
            idx += 2
            continue
        if arg in ("--lr_scheduler", "--lr-scheduler"):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects `constant`, `cosine`, or `linear`.")
            lr_scheduler = str(argv[idx + 1]).strip().lower()
            if lr_scheduler not in {"constant", "none", "cosine", "linear"}:
                raise SystemExit(f"{arg} expects `constant`, `cosine`, or `linear`, got {argv[idx + 1]!r}.")
            idx += 2
            continue
        if arg in ("--lr_warmup_updates", "--lr-warmup-updates"):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects an integer.")
            try:
                lr_warmup_updates = max(0, int(argv[idx + 1]))
            except ValueError as exc:
                raise SystemExit(f"{arg} expects an integer.") from exc
            idx += 2
            continue
        if arg in ("--lr_min_factor", "--lr-min-factor"):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects a float.")
            try:
                lr_min_factor = float(argv[idx + 1])
            except ValueError as exc:
                raise SystemExit(f"{arg} expects a float.") from exc
            if not math.isfinite(lr_min_factor):
                raise SystemExit(f"{arg} expects a finite float.")
            lr_min_factor = float(min(max(lr_min_factor, 0.0), 1.0))
            idx += 2
            continue
        if arg in ("--max_train_reset_attempts", "--max-train-reset-attempts"):
            if idx + 1 >= len(argv):
                raise SystemExit(f"{arg} expects a positive integer.")
            try:
                max_train_reset_attempts = max(1, int(argv[idx + 1]))
            except ValueError as exc:
                raise SystemExit(f"{arg} expects a positive integer.") from exc
            idx += 2
            continue

        normalized_arg = _strip_lchol_prefix(arg.split("=", 1)[0])
        if normalized_arg == "ddp_timeout_seconds":
            raw_value = arg.split("=", 1)[1] if "=" in arg else None
            if raw_value is None:
                if idx + 1 >= len(argv):
                    raise SystemExit(f"{arg} expects an integer.")
                idx += 1
                raw_value = argv[idx]
            try:
                ddp_timeout_seconds = max(0, int(raw_value))
            except ValueError as exc:
                raise SystemExit(f"{arg} expects an integer.") from exc
            idx += 1
            continue
        if normalized_arg == "ddp_rollout_sync_interval":
            raw_value = arg.split("=", 1)[1] if "=" in arg else None
            if raw_value is None:
                if idx + 1 >= len(argv):
                    raise SystemExit(f"{arg} expects an integer.")
                idx += 1
                raw_value = argv[idx]
            try:
                ddp_rollout_sync_interval = max(0, int(raw_value))
            except ValueError as exc:
                raise SystemExit(f"{arg} expects an integer.") from exc
            idx += 1
            continue
        if normalized_arg in {"lchol_enabled", "no_lchol_enabled"}:
            if "=" in arg:
                lchol_values["enabled"] = _parse_lchol_bool(arg.split("=", 1)[1])
            elif normalized_arg == "no_lchol_enabled":
                lchol_values["enabled"] = False
            elif idx + 1 < len(argv) and not str(argv[idx + 1]).startswith("--"):
                lchol_values["enabled"] = _parse_lchol_bool(str(argv[idx + 1]))
                idx += 1
            else:
                lchol_values["enabled"] = True
            idx += 1
            continue

        if normalized_arg in _LCHOL_VALUE_FIELDS:
            attr, caster = _LCHOL_VALUE_FIELDS[normalized_arg]
            if "=" in arg:
                raw_value = arg.split("=", 1)[1]
            else:
                if idx + 1 >= len(argv):
                    raise SystemExit(f"{arg} expects a value.")
                idx += 1
                raw_value = argv[idx]
            try:
                lchol_values[attr] = caster(raw_value)
            except ValueError as exc:
                raise SystemExit(f"{arg} received invalid value {raw_value!r}.") from exc
            idx += 1
            continue
        forwarded.append(arg)
        idx += 1

    return external_script, forwarded, _FastWrapperArgs(
        tensorboard_rollout_every_global_steps=tensorboard_rollout_every_global_steps,
        tensorboard_metric_profile=tensorboard_metric_profile,
        rollout_image_size=rollout_image_size,
        resume_actor_stats=resume_actor_stats,
        first_stage_grpo_actor_stats_path=first_stage_grpo_actor_stats_path,
        second_stage_grpo_actor_stats_path=second_stage_grpo_actor_stats_path,
        sparse_stage_init_log_std=sparse_stage_init_log_std,
        ddp_timeout_seconds=ddp_timeout_seconds,
        ddp_rollout_sync_interval=ddp_rollout_sync_interval,
        lr_scheduler=lr_scheduler,
        lr_warmup_updates=lr_warmup_updates,
        lr_min_factor=lr_min_factor,
        max_train_reset_attempts=max_train_reset_attempts,
        lchol=_LCHOLWrapperArgs(**lchol_values),
    )


def _extract_cli_arg_value(argv: Sequence[str], flag: str) -> str | None:
    inline_prefix = flag + "="
    for idx in range(len(argv) - 1, -1, -1):
        arg = str(argv[idx])
        if arg == flag:
            if idx + 1 < len(argv):
                return str(argv[idx + 1])
            return None
        if arg.startswith(inline_prefix):
            return arg[len(inline_prefix) :]
    return None


def _candidate_checkpoint_dirs(raw_path: str | Path) -> list[Path]:
    base = Path(raw_path).expanduser().resolve()
    if base.is_file():
        return [base.parent]

    candidates: list[Path] = []
    if base.name == "vla_cdpr_adapter":
        candidates.append(base.parent)
    candidates.append(base)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _resolve_grpo_actor_stats_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    base = Path(raw_path).expanduser().resolve()
    candidates: list[Path] = []
    if base.name == "grpo_actor_stats.pt":
        candidates.append(base)
    if base.is_file():
        candidates.append(base)
    for checkpoint_dir in _candidate_checkpoint_dirs(base):
        candidates.append(checkpoint_dir / "grpo_actor_stats.pt")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate.resolve()
    return None


def _infer_resume_artifacts(
    argv: Sequence[str],
    *,
    resume_actor_stats: bool = True,
    actor_stats_path: str | Path | None = None,
    require_actor_stats: bool = False,
) -> _ResumeArtifacts:
    if not resume_actor_stats:
        return _ResumeArtifacts()

    if actor_stats_path is not None:
        resolved_actor_stats_path = _resolve_grpo_actor_stats_path(actor_stats_path)
        if resolved_actor_stats_path is None:
            if require_actor_stats:
                raise FileNotFoundError(f"Could not find GRPO actor stats at {actor_stats_path}")
            return _ResumeArtifacts()
        return _ResumeArtifacts(
            checkpoint_dir=resolved_actor_stats_path.parent,
            actor_stats_path=resolved_actor_stats_path,
        )

    raw_values = [
        _extract_cli_arg_value(argv, "--action_head_path"),
        _extract_cli_arg_value(argv, "--adapter_path"),
    ]
    for raw_value in raw_values:
        if not raw_value:
            continue
        for checkpoint_dir in _candidate_checkpoint_dirs(raw_value):
            grpo_actor_stats_path = checkpoint_dir / "grpo_actor_stats.pt"
            if grpo_actor_stats_path.is_file():
                return _ResumeArtifacts(
                    checkpoint_dir=checkpoint_dir,
                    actor_stats_path=grpo_actor_stats_path.resolve(),
                )
    return _ResumeArtifacts()


def _candidate_external_scripts() -> list[Path]:
    env_candidate = os.environ.get("RLVLA_EXTERNAL_GRPO_SCRIPT")
    candidates: list[Path] = []
    if env_candidate:
        candidates.append(Path(env_candidate).expanduser().resolve())

    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        root = Path(entry).expanduser().resolve()
        candidates.append(root / "vla-scripts" / "grpo_finetune_cdpr.py")
        candidates.append(root.parent / "openvla-oft" / "vla-scripts" / "grpo_finetune_cdpr.py")

    return candidates


def _resolve_external_script(cli_path: Path | None) -> Path:
    candidates = [cli_path] if cli_path is not None else []
    candidates.extend(_candidate_external_scripts())
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    attempted = ", ".join(str(path) for path in seen) or "<none>"
    raise FileNotFoundError(
        "Could not locate external OpenVLA GRPO trainer. "
        "Pass --external_grpo_script or set RLVLA_EXTERNAL_GRPO_SCRIPT. "
        f"Checked: {attempted}"
    )


def _transform_external_grpo_source_for_lchol(source: str) -> str:
    transformed = source
    replacements = [
        (
            "@dataclass\n"
            "class Transition:\n"
            "    img_primary: np.ndarray\n"
            "    img_wrist: Optional[np.ndarray]\n"
            "    instruction: str\n"
            "    action: np.ndarray\n"
            "    logprob: float\n"
            "    env_reward: float\n"
            "    reward: float\n"
            "    advantage: float\n",
            "@dataclass\n"
            "class Transition:\n"
            "    img_primary: np.ndarray\n"
            "    img_wrist: Optional[np.ndarray]\n"
            "    instruction: str\n"
            "    action: np.ndarray\n"
            "    logprob: float\n"
            "    env_reward: float\n"
            "    reward: float\n"
            "    advantage: float\n"
            "    source: str\n"
            "    collection_prompt: str = \"\"\n"
            "    policy_version: int = 0\n"
            "    was_relabelled: bool = False\n"
            "    from_replay: bool = False\n",
        ),
        (
            "    branch_rng = np.random.default_rng(args.seed + 90_000 + rank)\n",
            "    branch_rng = np.random.default_rng(args.seed + 90_000 + rank)\n"
            "    _rlvla_lchol_set_runtime(\n"
            "        _rlvla_lchol_build_runtime(args, is_main=is_main, rank=rank, seed=args.seed + rank)\n"
            "    )\n",
        ),
        (
            "        for update in range(1, args.total_updates + 1):\n"
            "            policy.eval()\n",
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_lchol_pre_update(policy, optimizer=optimizer, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n",
        ),
        (
            "            loss_total_values: List[float] = []\n",
            "            loss_total_values: List[float] = []\n"
            "            loss_lchol_bc_values: List[float] = []\n",
        ),
        (
            "                        reward_components = _extract_reward_components(step_info if isinstance(step_info, dict) else {})\n",
            "                        lchol_group_score = _rlvla_lchol_phase_score(step_info, fallback=reward)\n"
            "                        if isinstance(step_info, dict):\n"
            "                            step_info[\"lchol_group_score\"] = float(lchol_group_score)\n"
            "                        reward_components = _extract_reward_components(step_info if isinstance(step_info, dict) else {})\n",
        ),
        (
            "                        candidate_results.append(candidate)\n"
            "                        candidate_rewards.append(float(reward))\n"
            "                        env.restore_state(base_state)\n",
            "                        candidate_results.append(candidate)\n"
            "                        candidate_group_score = (\n"
            "                            step_info.get(\"lchol_group_score\", reward)\n"
            "                            if isinstance(step_info, dict)\n"
            "                            else reward\n"
            "                        )\n"
            "                        candidate_rewards.append(float(candidate_group_score))\n"
            "                        _rlvla_lchol_capture_candidate(\n"
            "                            obs=obs,\n"
            "                            step_info=step_info if isinstance(step_info, dict) else {},\n"
            "                            sampled_action=sampled_actions_group[group_idx][env_idx],\n"
            "                            group_score=float(candidate_group_score),\n"
            "                            update=update,\n"
            "                            global_step=global_step + 1,\n"
            "                        )\n"
            "                        env.restore_state(base_state)\n",
        ),
        (
            "                            \"reward_shaped\": float(selected.reward),\n",
            "                            \"reward_shaped\": float(selected.reward),\n"
            "                            \"lchol_group_score\": float(selected.step_info.get(\"lchol_group_score\", selected.reward)),\n",
        ),
        (
            "            advantages = np.asarray([transition.advantage for transition in transitions], dtype=np.float32)\n\n"
            "            policy.train()\n",
            "            advantages = np.asarray([transition.advantage for transition in transitions], dtype=np.float32)\n"
            "            _rlvla_lchol_validate_grpo_transitions(transitions)\n"
            "            _rlvla_lchol_after_rollout(update=update)\n\n"
            "            policy.train()\n",
        ),
        (
            "                                advantage=float(group_advantages[group_idx]),\n"
            "                            )\n",
            "                                advantage=float(group_advantages[group_idx]),\n"
            "                                source=\"pg\",\n"
            "                                collection_prompt=str(obs.get(\"instruction\", \"\")),\n"
            "                                policy_version=int(update),\n"
            "                                was_relabelled=False,\n"
            "                                from_replay=False,\n"
            "                            )\n",
        ),
        (
            "                            loss = policy_loss + args.ent_coef * entropy_loss\n",
            "                            lchol_bc_loss = _rlvla_lchol_bc_loss(\n"
            "                                policy,\n"
            "                                ppo,\n"
            "                                device,\n"
            "                                args,\n"
            "                                num_actions_chunk=NUM_ACTIONS_CHUNK,\n"
            "                            )\n"
            "                            loss = policy_loss + args.ent_coef * entropy_loss + lchol_bc_loss\n",
        ),
        (
            "                                loss_total_values.append(float(loss.item()))\n",
            "                                loss_total_values.append(float(loss.item()))\n"
            "                                loss_lchol_bc_values.append(float(lchol_bc_loss.detach().item()))\n",
        ),
        (
            "            avg_total_loss = float(np.mean(loss_total_values)) if loss_total_values else 0.0\n",
            "            avg_total_loss = float(np.mean(loss_total_values)) if loss_total_values else 0.0\n"
            "            avg_lchol_bc_loss = float(np.mean(loss_lchol_bc_values)) if loss_lchol_bc_values else 0.0\n",
        ),
        (
            "                    f\"loss_total_mean={avg_total_loss:.4f} \"\n",
            "                    f\"loss_total_mean={avg_total_loss:.4f} \"\n"
            "                    f\"loss_lchol_bc_mean={avg_lchol_bc_loss:.4f} \"\n",
        ),
        (
            "                tb_writer.add_scalar(\"train/loss_total_mean\", avg_total_loss, global_step)\n",
            "                tb_writer.add_scalar(\"train/loss_total_mean\", avg_total_loss, global_step)\n"
            "                tb_writer.add_scalar(\"train/loss_lchol_bc_mean\", avg_lchol_bc_loss, global_step)\n",
        ),
        (
            "                tb_writer.flush()\n\n"
            "            if (\n"
            "                is_main\n"
            "                and args.rollout_tap_every_updates > 0\n",
            "                tb_writer.flush()\n\n"
            "            _rlvla_lchol_log_update(\n"
            "                update=update,\n"
            "                global_step=global_step,\n"
            "                tb_writer=tb_writer,\n"
            "                is_main=is_main,\n"
            "            )\n\n"
            "            if (\n"
            "                is_main\n"
            "                and args.rollout_tap_every_updates > 0\n",
        ),
        (
            "            if is_main and update % args.save_every == 0:\n",
            "            _rlvla_lchol_after_update(\n"
            "                update=update,\n"
            "                global_step=global_step,\n"
            "                run_dir=run_dir if is_main else None,\n"
            "            )\n"
            "            if _rlvla_lchol_should_stop_training(update=update):\n"
            "                break\n\n"
            "            if is_main and update % args.save_every == 0:\n",
        ),
    ]
    for old, new in replacements:
        if old not in transformed:
            raise RuntimeError(
                "Could not apply LC-HOL patch to external GRPO trainer; "
                f"missing source anchor: {old[:80]!r}"
            )
        transformed = transformed.replace(old, new, 1)
    return transformed


def _indent_source_block(block: str, *, spaces: int = 4) -> str:
    prefix = " " * int(spaces)
    return "\n".join(prefix + line if line else line for line in block.splitlines())


def _transform_external_grpo_source_for_memory_safety(source: str) -> str:
    transformed = source
    rollout_records_anchor = "            rollout_records: List[Dict[str, Any]] = []\n"
    rollout_records_replacement = (
        "            record_rollout_tap = bool(\n"
        "                is_main\n"
        "                and args.rollout_tap_every_updates > 0\n"
        "                and (update % args.rollout_tap_every_updates == 0)\n"
        "            )\n"
        "            rollout_records: List[Dict[str, Any]] = []\n"
    )
    if rollout_records_anchor not in transformed:
        raise RuntimeError(
            "Could not apply GRPO memory-safety patch; missing rollout_records anchor."
        )
    transformed = transformed.replace(rollout_records_anchor, rollout_records_replacement, 1)

    append_anchor = "                    rollout_records.append(\n"
    next_line_anchor = "\n\n                    steps_since_reset[env_idx] = next_step_count\n"
    append_start = transformed.find(append_anchor)
    if append_start < 0:
        raise RuntimeError(
            "Could not apply GRPO memory-safety patch; missing rollout_records append anchor."
        )
    append_end = transformed.find(next_line_anchor, append_start)
    if append_end < 0:
        raise RuntimeError(
            "Could not apply GRPO memory-safety patch; missing rollout_records append terminator."
        )
    append_block = transformed[append_start:append_end]
    guarded_append_block = (
        "                    if record_rollout_tap:\n"
        f"{_indent_source_block(append_block, spaces=4)}"
    )
    transformed = transformed[:append_start] + guarded_append_block + transformed[append_end:]

    post_rollout_anchor = (
        "            advantages = np.asarray([transition.advantage for transition in transitions], dtype=np.float32)\n"
    )
    post_rollout_replacement = (
        post_rollout_anchor +
        "            _rlvla_log_memory(\"post_rollout\", update=update, is_main=is_main)\n"
    )
    if post_rollout_anchor not in transformed:
        raise RuntimeError(
            "Could not apply GRPO memory-safety patch; missing post-rollout memory anchor."
        )
    transformed = transformed.replace(post_rollout_anchor, post_rollout_replacement, 1)

    post_train_anchor = (
        "            if train_pbar is not None:\n"
        "                train_pbar.close()\n\n"
    )
    post_train_replacement = (
        post_train_anchor +
        "            _rlvla_log_memory(\"post_train\", update=update, is_main=is_main)\n\n"
    )
    if post_train_anchor not in transformed:
        raise RuntimeError(
            "Could not apply GRPO memory-safety patch; missing post-train memory anchor."
        )
    transformed = transformed.replace(post_train_anchor, post_train_replacement, 1)
    return transformed


def _transform_external_grpo_source_for_ddp_sync(source: str) -> str:
    transformed = source
    legacy_ppo_actor_stats_anchor = (
        "    # Keep the familiar filename too so tooling can inspect log_std consistently.\n"
        "    torch.save(actor_stats, ckpt_dir / \"ppo_actor_stats.pt\")\n"
    )
    if legacy_ppo_actor_stats_anchor in transformed:
        transformed = transformed.replace(legacy_ppo_actor_stats_anchor, "", 1)

    init_anchor = "    rank, local_rank, world_size = ppo._init_distributed()\n"
    if init_anchor in transformed:
        transformed = transformed.replace(
            init_anchor,
            "    rank, local_rank, world_size = _rlvla_init_distributed()\n",
            1,
        )

    ddp_kwargs_anchor = "            gradient_as_bucket_view=True,\n"
    if ddp_kwargs_anchor in transformed:
        transformed = transformed.replace(
            ddp_kwargs_anchor,
            ddp_kwargs_anchor + "            broadcast_buffers=False,\n",
            1,
        )

    ddp_init_anchor = (
        "        if \"static_graph\" in ddp_params:\n"
        "            ddp_kwargs[\"static_graph\"] = bool(args.ddp_static_graph)\n"
        "        policy = DDP(policy, **ddp_kwargs)\n"
    )
    if ddp_init_anchor in transformed:
        transformed = transformed.replace(
            ddp_init_anchor,
            "        if \"static_graph\" in ddp_params:\n"
            "            ddp_kwargs[\"static_graph\"] = bool(args.ddp_static_graph)\n"
            "        if \"init_sync\" in ddp_params:\n"
            "            ddp_kwargs[\"init_sync\"] = False\n"
            "        print(f\"[rlvla-ddp] rank={rank} entering DDP policy wrap\", flush=True)\n"
            "        policy = DDP(policy, **ddp_kwargs)\n"
            "        print(f\"[rlvla-ddp] rank={rank} DDP policy ready\", flush=True)\n",
            1,
        )

    update_anchors = (
        (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_lchol_pre_update(policy, optimizer=optimizer, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n",
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_ddp_sync(\"pre_update\", update=update, run_dir=run_dir)\n"
            "            _rlvla_lchol_pre_update(policy, optimizer=optimizer, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n",
        ),
        (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_lchol_pre_update(policy, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n",
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_ddp_sync(\"pre_update\", update=update, run_dir=run_dir)\n"
            "            _rlvla_lchol_pre_update(policy, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n",
        ),
        (
            "        for update in range(1, args.total_updates + 1):\n            policy.eval()\n",
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_ddp_sync(\"pre_update\", update=update, run_dir=run_dir)\n"
            "            policy.eval()\n",
        ),
    )
    for update_anchor, update_replacement in update_anchors:
        if update_anchor not in transformed:
            continue
        transformed = transformed.replace(
            update_anchor,
            update_replacement,
            1,
        )
        break

    rollout_step_anchor = (
        "                if rollout_pbar is not None:\n"
        "                    rollout_pbar.update(1)\n\n"
        "            if rollout_pbar is not None:\n"
    )
    if rollout_step_anchor in transformed:
        transformed = transformed.replace(
            rollout_step_anchor,
            "                if rollout_pbar is not None:\n"
            "                    rollout_pbar.update(1)\n"
            "                _rlvla_ddp_sync_rollout(update=update, rollout_step=rollout_step)\n\n"
            "            if rollout_pbar is not None:\n",
            1,
        )

    train_anchor = "            policy.train()\n"
    if train_anchor in transformed:
        transformed = transformed.replace(
            train_anchor,
            "            _rlvla_ddp_sync(\"pre_train\", update=update)\n"
            "            policy.train()\n",
            1,
        )

    update_complete_anchor = (
        "                )\n\n"
        "        if is_main:\n"
        "            save_checkpoint(\n"
    )
    if update_complete_anchor in transformed:
        transformed = transformed.replace(
            update_complete_anchor,
            "                )\n\n"
            "            _rlvla_ddp_mark_update_complete(update=update, run_dir=run_dir)\n\n"
            "        if is_main:\n"
            "            save_checkpoint(\n",
            1,
        )

    return transformed


def _transform_external_grpo_source_for_lr_scheduler(source: str) -> str:
    transformed = source
    loop_anchors = (
        (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_ddp_sync(\"pre_update\", update=update, run_dir=run_dir)\n"
            "            _rlvla_lchol_pre_update(policy, optimizer=optimizer, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n"
        ),
        (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_ddp_sync(\"pre_update\", update=update, run_dir=run_dir)\n"
            "            _rlvla_lchol_pre_update(policy, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n"
        ),
        (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_lchol_pre_update(policy, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n"
        ),
        (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_ddp_sync(\"pre_update\", update=update, run_dir=run_dir)\n"
            "            policy.eval()\n"
        ),
        (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_lchol_pre_update(policy, optimizer=optimizer, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n"
        ),
        "        for update in range(1, args.total_updates + 1):\n            policy.eval()\n",
    )
    loop_replacement = None
    for anchor in loop_anchors:
        if anchor in transformed:
            loop_replacement = anchor.replace(
                "            policy.eval()\n",
                "            _rlvla_apply_lr_schedule(optimizer, update=update, total_updates=args.total_updates)\n"
                "            policy.eval()\n",
                1,
            )
            transformed = transformed.replace(anchor, loop_replacement, 1)
            break
    if loop_replacement is None:
        raise RuntimeError(
            "Could not apply LR scheduler patch to external GRPO trainer; missing update-loop anchor."
        )

    tb_anchor = '                tb_writer.add_scalar("train/loss_total_mean", avg_total_loss, global_step)\n'
    if tb_anchor in transformed:
        transformed = transformed.replace(
            tb_anchor,
            tb_anchor
            + '                tb_writer.add_scalar("train/learning_rate", _rlvla_current_lr(optimizer), global_step)\n',
            1,
        )
    else:
        raise RuntimeError(
            "Could not apply LR scheduler patch to external GRPO trainer; missing TensorBoard train/loss_total_mean anchor."
        )
    return transformed


def _load_external_module(
    script_path: Path,
    *,
    enable_lchol: bool = False,
    enable_ddp_sync: bool = False,
    enable_lr_scheduler: bool = False,
):
    spec = importlib.util.spec_from_file_location("rlvla_external_grpo_finetune_cdpr", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {script_path}")
    if enable_lchol or enable_ddp_sync or enable_lr_scheduler:
        source = script_path.read_text(encoding="utf-8")
        if enable_lchol:
            source = _transform_external_grpo_source_for_lchol(source)
        source = _transform_external_grpo_source_for_memory_safety(source)
        if enable_ddp_sync:
            source = _transform_external_grpo_source_for_ddp_sync(source)
        if enable_lr_scheduler:
            source = _transform_external_grpo_source_for_lr_scheduler(source)
        module = types.ModuleType(spec.name)
        module.__file__ = str(script_path)
        module.__package__ = ""
        sys.modules[spec.name] = module
        exec(compile(source, str(script_path), "exec"), module.__dict__)
        return module
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _to_pil_rgb(image) -> Image.Image:
    return Image.fromarray(image.astype("uint8")).convert("RGB")


def _pil_resample_filter():
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return resampling.BILINEAR
    return getattr(Image, "BILINEAR", 2)


def _resize_uint8_rgb(image: Any, *, image_size: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return arr
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    target = int(image_size)
    if target <= 0 or (int(arr.shape[0]) == target and int(arr.shape[1]) == target):
        return arr
    pil_image = Image.fromarray(arr).convert("RGB")
    resized = pil_image.resize((target, target), resample=_pil_resample_filter())
    return np.asarray(resized, dtype=np.uint8).copy()


def _patch_rollout_image_resize(module, *, image_size: int) -> None:
    target = max(0, int(image_size))
    if target <= 0:
        return
    ppo_module = getattr(module, "ppo", None)
    original_latest_image = getattr(ppo_module, "_latest_image_from_sim", None)
    if not callable(original_latest_image):
        return
    if getattr(original_latest_image, "_rlvla_resize_wrapped", False):
        return

    def _latest_image_from_sim_resized(sim, fallback_hw=(224, 224), wrist: bool = False):
        raw = original_latest_image(sim, fallback_hw=(target, target), wrist=wrist)
        return _resize_uint8_rgb(raw, image_size=target)

    _latest_image_from_sim_resized._rlvla_resize_wrapped = True  # type: ignore[attr-defined]
    _latest_image_from_sim_resized._rlvla_original = original_latest_image  # type: ignore[attr-defined]
    ppo_module._latest_image_from_sim = _latest_image_from_sim_resized
    if hasattr(module, "_latest_image_from_sim"):
        module._latest_image_from_sim = _latest_image_from_sim_resized
    if os.environ.get("RANK", "0") == "0":
        print(
            f"[rlvla-fast] Rollout image resize enabled: {target}x{target}",
            flush=True,
        )


def _current_process_rss_mib() -> float | None:
    status_path = Path("/proc/self/status")
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        pass
    try:
        import resource

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _patch_memory_logging(module) -> None:
    def _log_memory(label: str, *, update: int, is_main: bool) -> None:
        if not bool(is_main):
            return
        rss_mib = _current_process_rss_mib()
        if rss_mib is None:
            return
        print(
            f"[rlvla-memory] update={int(update):05d} label={label} rss_mib={rss_mib:.1f}",
            flush=True,
        )

    module._rlvla_log_memory = _log_memory


def _patch_prepare_inputs(module) -> None:
    policy_cls = module.OpenVLAGRPOPolicy
    original_prepare_inputs = policy_cls._prepare_inputs

    def _prepare_inputs_batched(self, images_primary, images_wrist, instructions):
        prompts = [f"In: What action should the robot take to {text.lower()}?\nOut:" for text in instructions]
        try:
            primary_images = [_to_pil_rgb(image) for image in images_primary]
            inputs = self.processor(prompts, primary_images, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"]

            if self.num_images_in_input > 1 and images_wrist is not None:
                wrist_images = [_to_pil_rgb(image) for image in images_wrist]
                wrist_inputs = self.processor(prompts, wrist_images, return_tensors="pt", padding=True)
                pixel_values = torch.cat([pixel_values, wrist_inputs["pixel_values"]], dim=1)

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            pixel_values = pixel_values.to(
                self.device,
                dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            )
            return input_ids, attention_mask, pixel_values
        except Exception as exc:
            if not getattr(self, "_rlvla_fast_prepare_warned", False):
                print(
                    f"[rlvla-fast] Falling back to original input preparation path: {exc}",
                    flush=True,
                )
                self._rlvla_fast_prepare_warned = True
            return original_prepare_inputs(self, images_primary, images_wrist, instructions)

    policy_cls._prepare_inputs = _prepare_inputs_batched


def _load_wrapper_bundle_checker():
    for module_name in ("cdpr_dataset.rl_cdpr_env", "robots.cdpr.cdpr_dataset.rl_cdpr_env"):
        try:
            helper_module = importlib.import_module(module_name)
        except Exception:
            continue
        helper = getattr(helper_module, "_wrapper_bundle_exists", None)
        if callable(helper):
            return helper
    return None


def _load_wrapper_object_checker():
    for module_name in ("cdpr_dataset.rl_cdpr_env", "robots.cdpr.cdpr_dataset.rl_cdpr_env"):
        try:
            helper_module = importlib.import_module(module_name)
        except Exception:
            continue
        helper = getattr(helper_module, "_wrapper_contains_requested_objects", None)
        if callable(helper):
            return helper
    return None


def _extract_cli_bool(argv: Sequence[str], flag: str, *, default: bool = False) -> bool:
    names = {flag, flag.replace("_", "-")}
    negative_names = {
        "--no-" + name.lstrip("-")
        for name in names
    }
    for idx in range(len(argv) - 1, -1, -1):
        arg = str(argv[idx])
        key, sep, value = arg.partition("=")
        if key in negative_names:
            return False
        if key in names:
            if sep:
                return str(value).strip().lower() not in {"0", "false", "no", "off"}
            if idx + 1 < len(argv):
                next_arg = str(argv[idx + 1])
                if not next_arg.startswith("--"):
                    return next_arg.strip().lower() not in {"0", "false", "no", "off"}
            return True
    return bool(default)


class _SceneCacheProgressReporter:
    def __init__(self, *, expected_total: int, enabled: bool = True):
        self.expected_total = max(0, int(expected_total))
        self.enabled = bool(enabled)
        self.count = 0
        self._bar = None
        self._started = False
        self._closed = False
        atexit.register(self.close)

    def _is_main_process(self) -> bool:
        try:
            return int(os.environ.get("RANK", "0")) == 0
        except ValueError:
            return True

    def _ensure_started(self) -> None:
        if self._started or not self.enabled or not self._is_main_process():
            return
        self._started = True
        try:
            from tqdm.auto import tqdm

            total = self.expected_total if self.expected_total > 0 else None
            self._bar = tqdm(
                total=total,
                desc="scene-cache prebuild",
                dynamic_ncols=True,
                file=sys.__stderr__,
                leave=True,
            )
        except Exception:
            total = f"/{self.expected_total}" if self.expected_total > 0 else ""
            print(f"[scene-cache] prebuild started: 0{total} wrapper builds", flush=True)

    def set_expected_total(self, expected_total: int) -> None:
        expected = max(0, int(expected_total))
        if expected <= 0 or expected == self.expected_total:
            return
        self.expected_total = expected
        if self._bar is not None:
            self._bar.total = expected
            self._bar.refresh()

    def update(self, *, scene_name: str = "") -> None:
        if self._closed or not self.enabled or not self._is_main_process():
            return
        self._ensure_started()
        self.count += 1
        if self._bar is not None:
            if scene_name:
                self._bar.set_postfix_str(str(scene_name))
            self._bar.update(1)
            return
        if self.count == 1 or self.count % 10 == 0:
            total = f"/{self.expected_total}" if self.expected_total > 0 else ""
            suffix = f" scene={scene_name}" if scene_name else ""
            print(f"[scene-cache] prebuild progress: {self.count}{total}{suffix}", flush=True)

    def close(self, *, cached_total: int | None = None) -> None:
        if self._closed or not self.enabled or not self._is_main_process():
            self._closed = True
            return
        self._closed = True
        cached_suffix = "" if cached_total is None else f"; cached_variants={int(cached_total)}"
        if self._bar is not None:
            if cached_total is not None:
                self._bar.set_postfix_str(f"cached={int(cached_total)}")
            self._bar.close()
        elif self._started:
            total = f"/{self.expected_total}" if self.expected_total > 0 else ""
            print(f"[scene-cache] prebuild complete: {self.count}{total} wrapper builds{cached_suffix}", flush=True)


def _patch_scene_cache_prebuild_progress(module, forwarded_argv: Sequence[str]) -> None:
    if not _extract_cli_bool(forwarded_argv, "--prebuild_scene_cache", default=False):
        return

    scene_pool_size = _extract_cli_arg_value(forwarded_argv, "--scene_pool_size")
    texture_pool_size = _extract_cli_arg_value(forwarded_argv, "--texture_pool_size")
    try:
        scene_count = max(1, int(scene_pool_size or 0))
    except ValueError:
        scene_count = 0
    try:
        texture_count = max(1, int(texture_pool_size or 1))
    except ValueError:
        texture_count = 1
    reporter = _SceneCacheProgressReporter(expected_total=scene_count * texture_count)

    patched_classes: set[type] = set()
    for module_name in ("cdpr_dataset.rl_cdpr_env", "robots.cdpr.cdpr_dataset.rl_cdpr_env"):
        try:
            env_module = importlib.import_module(module_name)
        except Exception:
            continue
        env_cls = getattr(env_module, "CDPRLanguageRLEnv", None)
        if env_cls is None or env_cls in patched_classes:
            continue
        original_build_wrapper = getattr(env_cls, "_build_wrapper", None)
        if not callable(original_build_wrapper) or getattr(original_build_wrapper, "_rlvla_progress_wrapped", False):
            continue

        def _build_wrapper_with_progress(self, scene, *args, _original=original_build_wrapper, **kwargs):
            live_textures = len(getattr(self, "desk_texture_files", []) or [])
            if live_textures > 0 and scene_count > 0:
                reporter.set_expected_total(scene_count * min(texture_count, live_textures))
            try:
                return _original(self, scene, *args, **kwargs)
            finally:
                reporter.update(scene_name=str(getattr(scene, "name", "")))

        _build_wrapper_with_progress._rlvla_progress_wrapped = True  # type: ignore[attr-defined]
        env_cls._build_wrapper = _build_wrapper_with_progress
        patched_classes.add(env_cls)

    env_cls = getattr(module, "CDPRVisionLanguageEnv", None)
    original_activate = getattr(env_cls, "_activate_scene_wrapper_cache", None)
    if not callable(original_activate) or getattr(original_activate, "_rlvla_progress_wrapped", False):
        return

    def _activate_scene_wrapper_cache_with_progress(self, scene_wrapper_cache, texture_name_by_wrapper):
        try:
            return original_activate(self, scene_wrapper_cache, texture_name_by_wrapper)
        finally:
            cached_total = 0
            if isinstance(scene_wrapper_cache, dict):
                cached_total = sum(len(list(paths or [])) for paths in scene_wrapper_cache.values())
            reporter.close(cached_total=cached_total)

    _activate_scene_wrapper_cache_with_progress._rlvla_progress_wrapped = True  # type: ignore[attr-defined]
    env_cls._activate_scene_wrapper_cache = _activate_scene_wrapper_cache_with_progress


def _patch_scene_wrapper_cache(module) -> None:
    wrapper_bundle_exists = _load_wrapper_bundle_checker()
    if wrapper_bundle_exists is None:
        return
    wrapper_contains_requested_objects = _load_wrapper_object_checker()

    env_cls = module.CDPRVisionLanguageEnv
    original_activate = env_cls._activate_scene_wrapper_cache

    def _activate_scene_wrapper_cache_checked(self, scene_wrapper_cache, texture_name_by_wrapper):
        out = original_activate(self, scene_wrapper_cache, texture_name_by_wrapper)
        rl = self.env
        if not hasattr(rl, "_build_wrapper_original"):
            return out

        def _variant_supports_scene(path: Path, scene) -> bool:
            path = Path(path).resolve()
            if not wrapper_bundle_exists(path):
                return False
            if callable(wrapper_contains_requested_objects):
                return bool(wrapper_contains_requested_objects(path, tuple(getattr(scene, "objects", ()) or ())))
            return True

        def _call_original_builder(this, scene, ee_start=None):
            try:
                if ee_start is not None:
                    return this._build_wrapper_original(scene, ee_start=ee_start)
            except TypeError as exc:
                if "ee_start" not in str(exc):
                    raise
            return this._build_wrapper_original(scene)

        def _build_wrapper_checked(this, scene, ee_start=None):
            scene_name = str(getattr(scene, "name", ""))
            variants_local = list(self._scene_wrapper_cache.get(scene_name) or [])
            if variants_local:
                object_key = tuple(sorted(str(item) for item in (getattr(scene, "objects", ()) or ())))
                lookup = getattr(self, "_rlvla_compatible_wrapper_cache", {})
                lookup_key = (scene_name, object_key)
                available_variants = lookup.get(lookup_key)
                if available_variants is None:
                    available_variants = [
                        Path(path).resolve()
                        for path in variants_local
                        if _variant_supports_scene(Path(path), scene)
                    ]
                    lookup[lookup_key] = available_variants
                    self._rlvla_compatible_wrapper_cache = lookup
                if len(available_variants) != len(variants_local):
                    warned = getattr(self, "_rlvla_scene_cache_repair_warned", set())
                    warn_key = (scene_name, tuple(getattr(scene, "objects", ()) or ()))
                    if warn_key not in warned:
                        print(
                            f"[env_cache] Selected object-compatible wrappers for scene '{scene_name}' "
                            f"({len(available_variants)}/{len(variants_local)} variants).",
                            flush=True,
                        )
                        warned.add(warn_key)
                        self._rlvla_scene_cache_repair_warned = warned
                if available_variants:
                    idx = int(this.np_random.integers(0, len(available_variants)))
                    chosen = Path(available_variants[idx]).resolve()
                    texture_map = getattr(self, "_texture_name_by_wrapper", {})
                    this._desk_texture_name = str(texture_map.get(str(chosen), ""))
                    return chosen
            this._desk_texture_name = ""
            this._background_color = ""
            return _call_original_builder(this, scene, ee_start=ee_start)

        rl._build_wrapper = module.types.MethodType(_build_wrapper_checked, rl)
        return out

    env_cls._activate_scene_wrapper_cache = _activate_scene_wrapper_cache_checked


def _patch_fresh_scene_cache_prebuild(module, forwarded_argv: Sequence[str]) -> None:
    if not _extract_cli_bool(forwarded_argv, "--prebuild_scene_cache", default=False):
        return
    if _extract_cli_bool(forwarded_argv, "--use_wrapper_cache", default=False):
        return

    env_cls = getattr(module, "CDPRVisionLanguageEnv", None)
    original_enable = getattr(env_cls, "enable_prebuilt_scene_cache", None)
    if not callable(original_enable):
        return

    def _enable_fresh_scene_cache(self, scene_pool_size: int, texture_pool_size: int, seed: int):
        rl = self.env
        rl_mod = self._rl_env_module
        scenes_all = list(getattr(rl, "scenes", []) or [])
        if not scenes_all:
            return {"scenes": 0, "variants": 0, "textures": 0}

        rng = np.random.default_rng(int(seed))
        if int(scene_pool_size) > 0 and len(scenes_all) > int(scene_pool_size):
            chosen_idx = np.sort(rng.choice(len(scenes_all), size=int(scene_pool_size), replace=False))
            scenes = [scenes_all[int(idx)] for idx in chosen_idx]
        else:
            scenes = scenes_all

        texture_files = list(getattr(rl, "desk_texture_files", []) or [])
        if int(texture_pool_size) > 0 and len(texture_files) > int(texture_pool_size):
            texture_idx = np.sort(rng.choice(len(texture_files), size=int(texture_pool_size), replace=False))
            texture_files = [texture_files[int(idx)] for idx in texture_idx]

        palette_getter = getattr(rl_mod, "_metadata_color_palette", None)
        background_builder = getattr(rl_mod, "_build_background_color_variant", None)
        background_palette = (
            tuple(palette_getter(dict(getattr(rl, "_task_metadata", {}) or {})))
            if callable(palette_getter)
            else ()
        )

        cache: dict[str, list[Path]] = {}
        texture_name_by_wrapper: dict[str, str] = {}
        original_use_cache = bool(getattr(rl, "use_wrapper_cache", False))
        original_reuse = bool(getattr(rl, "reuse_existing_wrapper_variants", False))
        original_cleanup = bool(getattr(rl, "wrapper_cleanup", False))
        original_textures = list(getattr(rl, "desk_texture_files", []) or [])
        original_metadata = dict(getattr(rl, "_task_metadata", {}) or {})

        try:
            rl.use_wrapper_cache = False
            rl.reuse_existing_wrapper_variants = False
            rl.wrapper_cleanup = False
            for scene_idx, scene in enumerate(scenes):
                scene_name = str(getattr(scene, "name", ""))
                scene_objects = tuple(str(item) for item in (getattr(scene, "objects", ()) or ()))
                if not scene_name or not scene_objects:
                    continue

                ee_start = np.asarray(
                    rl.defaults.get("ee_start", (0.0, 0.0, 0.40)),
                    dtype=np.float32,
                ).reshape(-1)
                if ee_start.size < 3:
                    ee_start = np.pad(ee_start, (0, 3 - ee_start.size))
                ee_start[2] = max(float(ee_start[2]), 0.40)

                # Build one run-local, untextured base wrapper. Texture and
                # background variants are derived from it without consulting
                # any pre-existing wrapper bundle.
                rl.desk_texture_files = []
                base_metadata = dict(original_metadata)
                base_metadata.pop("background_color_palette", None)
                rl._task_metadata = base_metadata
                base_wrapper = Path(
                    rl._build_wrapper(scene, ee_start=ee_start[:3])
                ).resolve()

                rl.desk_texture_files = list(original_textures)
                rl._task_metadata = dict(original_metadata)
                variant_count = max(1, len(texture_files))
                variants: list[Path] = []
                for variant_idx in range(variant_count):
                    wrapper_path = base_wrapper
                    texture_name = ""
                    if texture_files:
                        texture_path = Path(texture_files[variant_idx]).resolve()
                        texture_hash = hashlib.sha1(
                            texture_path.as_posix().encode("utf-8")
                        ).hexdigest()[:8]
                        tag = self._safe_cache_tag(
                            f"fresh_s{scene_idx:03d}_t{variant_idx:03d}_{texture_hash}"
                        )
                        textured = rl_mod._build_textured_wrapper_variant(
                            base_wrapper_xml=wrapper_path,
                            chosen_texture=texture_path,
                            variant_tag=tag,
                            desk_geom_regex=rl.desk_geom_regex,
                            desk_texrepeat=rl.desk_texrepeat,
                        )
                        wrapper_path = Path(textured.wrapper_xml).resolve()
                        texture_name = texture_path.name
                    if background_palette and callable(background_builder):
                        color = background_palette[variant_idx % len(background_palette)]
                        colored = background_builder(wrapper_path, color)
                        wrapper_path = Path(colored.wrapper_xml).resolve()
                    variants.append(wrapper_path)
                    texture_name_by_wrapper[str(wrapper_path)] = texture_name

                cache.setdefault(scene_name, []).extend(variants)
        finally:
            rl.use_wrapper_cache = original_use_cache
            rl.reuse_existing_wrapper_variants = original_reuse
            rl.wrapper_cleanup = original_cleanup
            rl.desk_texture_files = original_textures
            rl._task_metadata = original_metadata

        if not cache:
            return {"scenes": 0, "variants": 0, "textures": 0}
        out = self._activate_scene_wrapper_cache(cache, texture_name_by_wrapper)
        print(
            "[env_cache] Built fresh run-local wrapper pool "
            f"variants={sum(len(items) for items in cache.values())}; old wrapper cache ignored.",
            flush=True,
        )
        return out

    env_cls.enable_prebuilt_scene_cache = _enable_fresh_scene_cache


def _patch_training_reset_retries(module, *, max_attempts: int) -> None:
    env_cls = getattr(module, "CDPRVisionLanguageEnv", None)
    original_reset = getattr(env_cls, "reset", None)
    if not callable(original_reset):
        return

    retry_markers = (
        "Invalid CDPR state after episode reset",
        "invalid reset state",
        "Failed to initialize wrapper",
        "simulation_state_valid",
    )

    def _reset_with_retries(self, options=None):
        errors: list[str] = []
        attempts = max(1, int(max_attempts))
        for attempt in range(1, attempts + 1):
            try:
                return original_reset(self, options=options)
            except RuntimeError as exc:
                message = str(exc)
                if not any(marker in message for marker in retry_markers):
                    raise
                errors.append(f"attempt {attempt}: {message}")
                if attempt < attempts:
                    print(
                        f"[env-reset] retrying after attempt {attempt}/{attempts}: {message}",
                        flush=True,
                    )
        raise RuntimeError(
            f"CDPR training reset failed after {attempts} attempts. "
            + " | ".join(errors)
        )

    env_cls.reset = _reset_with_retries


def _patch_desk_texture_prepare(module) -> None:
    ppo_module = getattr(module, "ppo", None)
    owner = ppo_module if ppo_module is not None else module
    original_prepare = getattr(owner, "_prepare_desk_textures_dir", None)
    broadcast_object = getattr(owner, "_broadcast_object", None)
    if not callable(original_prepare) or not callable(broadcast_object):
        return

    def _prepare_desk_textures_dir_single_writer(src_dir, run_dir, is_main, rank, max_textures):
        rank_int = int(rank)
        if rank_int != 0:
            return broadcast_object(None, rank_int)
        return original_prepare(src_dir, run_dir, is_main, rank_int, max_textures)

    owner._prepare_desk_textures_dir = _prepare_desk_textures_dir_single_writer
    if owner is not module and hasattr(module, "_prepare_desk_textures_dir"):
        module._prepare_desk_textures_dir = _prepare_desk_textures_dir_single_writer


def _wrap_process_group_init_timeout(dist_module, *, timeout: timedelta | None) -> None:
    if timeout is None:
        return
    original_init_process_group = getattr(dist_module, "init_process_group", None)
    if not callable(original_init_process_group):
        return
    if getattr(original_init_process_group, "_rlvla_timeout_wrapped", False):
        return

    def _init_process_group_with_timeout(*args, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = timeout
        return original_init_process_group(*args, **kwargs)

    _init_process_group_with_timeout._rlvla_timeout_wrapped = True  # type: ignore[attr-defined]
    _init_process_group_with_timeout._rlvla_original = original_init_process_group  # type: ignore[attr-defined]
    dist_module.init_process_group = _init_process_group_with_timeout


def _patch_global_distributed_timeout(*, timeout_seconds: int) -> None:
    if timeout_seconds <= 0:
        return
    try:
        import torch.distributed as dist_module
    except Exception:
        return
    _wrap_process_group_init_timeout(
        dist_module,
        timeout=timedelta(seconds=max(1, int(timeout_seconds))),
    )


def _patch_distributed_timeout(module, *, timeout_seconds: int) -> None:
    ppo_module = getattr(module, "ppo", None)
    original_init = (
        getattr(ppo_module, "_init_distributed", None)
        if ppo_module is not None
        else getattr(module, "_init_distributed", None)
    )
    timeout = timedelta(seconds=max(1, int(timeout_seconds))) if timeout_seconds > 0 else None
    for dist_candidate in (
        getattr(ppo_module, "dist", None) if ppo_module is not None else None,
        getattr(module, "dist", None),
    ):
        if dist_candidate is not None:
            _wrap_process_group_init_timeout(dist_candidate, timeout=timeout)
    if timeout is not None and os.environ.get("RANK", "0") == "0":
        print(
            f"[rlvla-ddp] wrapper requested process-group timeout {int(timeout.total_seconds())}s",
            flush=True,
        )
    if ppo_module is None:
        if callable(original_init):
            module._rlvla_init_distributed = original_init
        return
    dist_module = getattr(ppo_module, "dist", None) or getattr(module, "dist", None)
    torch_module = getattr(ppo_module, "torch", None) or getattr(module, "torch", torch)
    if dist_module is None or not callable(getattr(dist_module, "init_process_group", None)):
        if callable(original_init):
            module._rlvla_init_distributed = original_init
        return

    def _init_distributed_with_timeout():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if world_size > 1 and not dist_module.is_initialized():
            backend = "nccl" if torch_module.cuda.is_available() else "gloo"
            kwargs = {"backend": backend}
            if timeout is not None:
                kwargs["timeout"] = timeout
            dist_module.init_process_group(**kwargs)
            if timeout is not None and rank == 0:
                print(
                    f"[rlvla-ddp] process-group timeout set to {int(timeout.total_seconds())}s",
                    flush=True,
                )
        return rank, local_rank, world_size

    module._rlvla_init_distributed = _init_distributed_with_timeout
    ppo_module._init_distributed = _init_distributed_with_timeout


def _patch_ddp_sync(module, *, rollout_sync_interval: int = 0) -> None:
    dist_module = getattr(module, "dist", None)
    if dist_module is None:
        return
    rollout_interval = max(0, int(rollout_sync_interval))

    def _rank() -> int:
        try:
            return int(os.environ.get("RANK", "0"))
        except ValueError:
            return 0

    def _world_size() -> int:
        try:
            return int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            return 1

    def _update_ready_path(run_dir: Path | str, *, update: int, rank: int) -> Path:
        return Path(run_dir) / ".rlvla_ddp" / f"update_{int(update):05d}_rank_{int(rank):05d}.ready"

    def _wait_for_update_markers(run_dir: Path | str, *, update: int) -> None:
        world_size = _world_size()
        if update <= 0 or world_size <= 1:
            return
        import time

        started = time.monotonic()
        last_log = started
        missing = [
            _update_ready_path(run_dir, update=int(update), rank=rank)
            for rank in range(world_size)
        ]
        while True:
            remaining = [path for path in missing if not path.exists()]
            if not remaining:
                waited = time.monotonic() - started
                if waited >= 30.0 and _rank() == 0:
                    print(
                        f"[rlvla-ddp] waited {waited:.1f}s for update {int(update)} filesystem markers",
                        flush=True,
                    )
                return
            now = time.monotonic()
            if now - last_log >= 60.0 and _rank() == 0:
                preview = ", ".join(path.name for path in remaining[:4])
                extra = "" if len(remaining) <= 4 else f", ... +{len(remaining) - 4}"
                print(
                    f"[rlvla-ddp] waiting for update {int(update)} markers: {preview}{extra}",
                    flush=True,
                )
                last_log = now
            time.sleep(5.0)

    def _rlvla_ddp_sync(label: str, *, update: int, run_dir=None) -> None:
        if not (dist_module.is_available() and dist_module.is_initialized()):
            return
        import time

        if label == "pre_update" and run_dir is not None:
            _wait_for_update_markers(run_dir, update=int(update) - 1)

        start = time.monotonic()
        dist_module.barrier()
        waited = time.monotonic() - start
        if waited >= 30.0 and os.environ.get("RANK", "0") == "0":
            print(
                f"[rlvla-ddp] waited {waited:.1f}s at {label} sync for update {int(update)}",
                flush=True,
            )

    def _rlvla_ddp_sync_rollout(*, update: int, rollout_step: int) -> None:
        if rollout_interval <= 0:
            return
        step = int(rollout_step) + 1
        if step % rollout_interval != 0:
            return
        _rlvla_ddp_sync(f"rollout_step_{step}", update=int(update))

    def _rlvla_ddp_mark_update_complete(*, update: int, run_dir) -> None:
        if _world_size() <= 1 or run_dir is None:
            return
        path = _update_ready_path(run_dir, update=int(update), rank=_rank())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text("ready\n", encoding="utf-8")
        tmp_path.replace(path)

    module._rlvla_ddp_sync = _rlvla_ddp_sync
    module._rlvla_ddp_sync_rollout = _rlvla_ddp_sync_rollout
    module._rlvla_ddp_mark_update_complete = _rlvla_ddp_mark_update_complete
    if rollout_interval > 0 and os.environ.get("RANK", "0") == "0":
        print(
            f"[rlvla-ddp] rollout sync interval set to every {rollout_interval} rollout steps",
            flush=True,
        )


def _find_log_std_tensor(payload: Any) -> torch.Tensor | None:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        direct = payload.get("log_std")
        if isinstance(direct, torch.Tensor):
            return direct
        for value in payload.values():
            tensor = _find_log_std_tensor(value)
            if tensor is not None:
                return tensor
    return None


def _copy_grpo_log_std_from_actor_stats(policy_core: Any, actor_stats_path: Path) -> None:
    state_raw = torch.load(actor_stats_path, map_location="cpu")
    log_std = _find_log_std_tensor(state_raw)
    if log_std is None:
        raise RuntimeError(f"Could not locate `log_std` tensor in {actor_stats_path}")
    target_log_std = getattr(policy_core, "log_std")
    flat_log_std = log_std.detach().reshape(-1)
    if flat_log_std.numel() != target_log_std.numel():
        raise RuntimeError(
            "Resumed GRPO actor stats shape mismatch. "
            f"checkpoint={tuple(flat_log_std.shape)}, target={tuple(target_log_std.shape)}"
        )
    device = getattr(policy_core, "device", getattr(target_log_std, "device", None))
    target = flat_log_std.to(device=device, dtype=target_log_std.dtype).reshape(target_log_std.shape)
    with torch.no_grad():
        target_log_std.copy_(target)


def _patch_resume_artifacts(module, forwarded_argv: Sequence[str], fast_args: _FastWrapperArgs) -> None:
    artifacts = _infer_resume_artifacts(
        forwarded_argv,
        resume_actor_stats=fast_args.resume_actor_stats,
        actor_stats_path=fast_args.first_stage_grpo_actor_stats_path,
        require_actor_stats=fast_args.first_stage_grpo_actor_stats_path is not None,
    )
    if artifacts.checkpoint_dir is None or artifacts.actor_stats_path is None:
        return

    original_policy_init = module.OpenVLAGRPOPolicy.__init__

    def _policy_init_with_actor_resume(self, *args, **kwargs):
        original_policy_init(self, *args, **kwargs)
        try:
            self._rlvla_grpo_initial_log_std = self.log_std.detach().clone()
        except Exception:
            self._rlvla_grpo_initial_log_std = None
        _copy_grpo_log_std_from_actor_stats(self, artifacts.actor_stats_path)
        print(f"[grpo_actor_stats] Loaded first-stage stats from {artifacts.actor_stats_path}", flush=True)

    module.OpenVLAGRPOPolicy.__init__ = _policy_init_with_actor_resume
    print(
        "[rlvla-fast] Resume artifacts inferred from checkpoint dir: "
        f"{artifacts.checkpoint_dir} (actor_stats)",
        flush=True,
    )


def _lr_schedule_factor(
    *,
    scheduler: str,
    update: int,
    total_updates: int,
    warmup_updates: int,
    min_factor: float,
) -> float:
    mode = str(scheduler).strip().lower()
    if mode in {"", "constant", "none"}:
        return 1.0

    update_i = max(1, int(update))
    total_i = max(1, int(total_updates))
    warmup_i = max(0, int(warmup_updates))
    min_f = float(min(max(float(min_factor), 0.0), 1.0))

    if warmup_i > 0 and update_i <= warmup_i:
        return float(update_i / max(1, warmup_i))

    denom = max(1, total_i - warmup_i)
    progress = float(min(max((update_i - warmup_i) / denom, 0.0), 1.0))
    if mode == "linear":
        return float(min_f + (1.0 - min_f) * (1.0 - progress))
    if mode == "cosine":
        return float(min_f + (1.0 - min_f) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    return 1.0


def _patch_lr_scheduler(module, fast_args: _FastWrapperArgs) -> None:
    config = {
        "scheduler": str(fast_args.lr_scheduler).strip().lower(),
        "warmup_updates": int(fast_args.lr_warmup_updates),
        "min_factor": float(fast_args.lr_min_factor),
    }

    def _current_lr(optimizer) -> float:
        try:
            return float(optimizer.param_groups[0]["lr"])
        except Exception:
            return 0.0

    def _apply_lr_schedule(optimizer, *, update: int, total_updates: int) -> float:
        factor = _lr_schedule_factor(
            scheduler=str(config["scheduler"]),
            update=int(update),
            total_updates=int(total_updates),
            warmup_updates=int(config["warmup_updates"]),
            min_factor=float(config["min_factor"]),
        )
        for group in getattr(optimizer, "param_groups", []):
            if "initial_lr" not in group:
                group["initial_lr"] = float(group.get("lr", 0.0))
            group["lr"] = float(group["initial_lr"]) * float(factor)
        return _current_lr(optimizer)

    module._rlvla_current_lr = _current_lr
    module._rlvla_apply_lr_schedule = _apply_lr_schedule
    if str(config["scheduler"]) not in {"", "constant", "none"}:
        print(
            "[rlvla-fast] LR scheduler enabled: "
            f"{config['scheduler']} warmup_updates={config['warmup_updates']} "
            f"min_factor={config['min_factor']}",
            flush=True,
        )


_COMPACT_TENSORBOARD_SCALARS: set[str] = {
    "train/reward_env_mean",
    "train/episode_return_env_mean",
    "train/loss_policy_mean",
    "train/entropy_mean",
    "train/loss_total_mean",
    "train/loss_lchol_bc_mean",
    "train/approx_kl_mean",
    "train/clip_fraction_mean",
    "train/group_advantage_std",
    "train/log_std_mean",
    "train/learning_rate",
    "validation/env_return_mean",
    "validation/success_rate",
    "rollout_step/reward_env_mean",
    "rollout_step/reward_shaped_mean",
    "rollout_step/success_rate_mean",
    "rollout_step/episode_success_rate_mean",
    "rollout_step/episode_timeout_rate_mean",
    "rollout_step/target_grasped_rate_mean",
    "rollout_step/unstable_transition_rate_mean",
    "rollout_step/reward_clip_rate_mean",
    "rollout_step/reward_non_finite_rate_mean",
    "rollout_step/distance_ee_to_object_xy_mean",
    "rollout_step/sparse_success_mean",
    "rollout_step/caught_object_is_target_mean",
    "rollout_step/target_motion_xy_mean",
    "rollout_step/relation_error_mean",
    "rollout_step/action_saturation_rate_mean",
    "rollout_step/lchol_group_score_mean",
    "rollout_episode/instruction_success_rate_mean",
    "lchol/source/hindsight_new",
    "lchol/source/hindsight_replay",
    "lchol/replay/total_records",
    "lchol/replay/episodes_total",
    "lchol/curriculum/stage_index",
    "lchol/phase_score/mean",
}
_COMPACT_TENSORBOARD_PREFIXES: tuple[str, ...] = (
    "rollout_episode/instruction_success_rate/",
    "rollout_episode/shell_success_rate/",
    "rollout_episode/subgoal_success_rate/",
    "lchol/replay/episodes/",
    "lchol/curriculum/success_rate/",
    "lchol/curriculum/reverse_frontier/",
    "lchol/reverse_frontier/shell_success_rate/",
)


def _strip_tensorboard_stage_prefix(tag: str) -> str:
    parts = str(tag).split("/", 2)
    if len(parts) == 3 and parts[0] == "stage" and parts[1] in {"dense", "sparse"}:
        return parts[2]
    return str(tag)


def _tensorboard_tag_allowed(tag: str, *, profile: str = "compact", kind: str = "scalar") -> bool:
    if str(profile).strip().lower() == "full":
        return True
    if str(kind) != "scalar":
        return False
    raw_tag = str(tag)
    if raw_tag.startswith("stage/dense/"):
        dense_tag = raw_tag[len("stage/dense/") :]
        if dense_tag in {
            "active",
            "complete",
            "threshold",
            "mean_success",
            "mean_reward",
            "updates_completed",
            "max_updates",
        }:
            return True
        if dense_tag.startswith(("success_rate/", "rollouts/", "reward/")):
            return True
    if raw_tag.startswith("stage/sparse/"):
        sparse_tag = raw_tag[len("stage/sparse/") :]
        if sparse_tag in {"updates_completed", "max_updates"}:
            return True
    tag = _strip_tensorboard_stage_prefix(raw_tag)
    return tag in _COMPACT_TENSORBOARD_SCALARS or any(
        tag.startswith(prefix) for prefix in _COMPACT_TENSORBOARD_PREFIXES
    )


def _current_tensorboard_stage(module) -> str | None:
    runtime = getattr(module, "_rlvla_lchol_runtime", None)
    if runtime is None:
        return None
    dense_gate_active = getattr(runtime, "dense_gate_active", None)
    if not callable(dense_gate_active):
        return None
    try:
        return "dense" if bool(dense_gate_active()) else "sparse"
    except Exception:
        return None


def _stage_tensorboard_tag(module, tag: str) -> str | None:
    tag = str(tag)
    if tag.startswith("stage/"):
        return None
    stage = _current_tensorboard_stage(module)
    if not stage:
        return None
    return f"stage/{stage}/{tag}"


def _patch_tensorboard_metric_filter(module, *, profile: str) -> None:
    writer_cls = getattr(module, "SummaryWriter", None)
    if writer_cls is None:
        return
    profile_name = str(profile).strip().lower()

    class _FilteredSummaryWriter:
        def __init__(self, *args, **kwargs):
            self._inner = writer_cls(*args, **kwargs)

        def add_scalar(self, tag, scalar_value, *args, **kwargs):
            tag_str = str(tag)
            if not _tensorboard_tag_allowed(tag_str, profile=profile_name, kind="scalar"):
                return None
            result = self._inner.add_scalar(tag, scalar_value, *args, **kwargs)
            stage_tag = _stage_tensorboard_tag(module, tag_str)
            if stage_tag is not None and _tensorboard_tag_allowed(stage_tag, profile=profile_name, kind="scalar"):
                self._inner.add_scalar(stage_tag, scalar_value, *args, **kwargs)
            return result

        def add_histogram(self, tag, values, *args, **kwargs):
            if _tensorboard_tag_allowed(str(tag), profile=profile_name, kind="histogram"):
                return self._inner.add_histogram(tag, values, *args, **kwargs)
            return None

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    module.SummaryWriter = _FilteredSummaryWriter
    if profile_name == "full":
        print("[rlvla-fast] TensorBoard metric profile: full (stage mirrors enabled)", flush=True)
    else:
        print("[rlvla-fast] TensorBoard metric profile: compact (stage mirrors enabled)", flush=True)


class _RolloutTensorboardLogger:
    def __init__(self, summary_writer_cls, every_global_steps: int, stage_fn=None):
        self.summary_writer_cls = summary_writer_cls
        self.every_global_steps = max(0, int(every_global_steps))
        self.stage_fn = stage_fn
        self.run_dir: Path | None = None
        self.writer = None
        self.global_step = 0
        self.enabled = self.every_global_steps > 0
        self.training_enabled = True
        self._registered_atexit = False
        self._pending_reward: dict[str, float] | None = None
        self._episode_buffers: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self._windows: dict[str, deque[float]] = {
            "reward_env": deque(maxlen=self.every_global_steps or 1),
            "reward_shaped": deque(maxlen=self.every_global_steps or 1),
            "success_rate": deque(maxlen=self.every_global_steps or 1),
            "episode_success_rate": deque(maxlen=self.every_global_steps or 1),
            "episode_timeout_rate": deque(maxlen=self.every_global_steps or 1),
            "target_grasped_rate": deque(maxlen=self.every_global_steps or 1),
            "unstable_transition_rate": deque(maxlen=self.every_global_steps or 1),
            "reward_clip_rate": deque(maxlen=self.every_global_steps or 1),
            "reward_non_finite_rate": deque(maxlen=self.every_global_steps or 1),
            "distance_ee_to_object_xy": deque(maxlen=self.every_global_steps or 1),
            "sparse_success": deque(maxlen=self.every_global_steps or 1),
            "caught_object_is_target": deque(maxlen=self.every_global_steps or 1),
            "target_motion_xy": deque(maxlen=self.every_global_steps or 1),
            "relation_error": deque(maxlen=self.every_global_steps or 1),
            "action_saturation_rate": deque(maxlen=self.every_global_steps or 1),
            "lchol_group_score": deque(maxlen=self.every_global_steps or 1),
        }
        self._instruction_episode_windows: dict[str, deque[float]] = {
            name: deque(maxlen=self.every_global_steps or 1) for name in _CDPR_CURRICULUM_OPTIONS
        }
        self._subgoal_episode_windows: dict[str, deque[float]] = {
            name: deque(maxlen=self.every_global_steps or 1) for name in _CDPR_CURRICULUM_OPTIONS
        }
        self._shell_episode_windows: dict[str, deque[float]] = {}

    def set_run_dir(self, run_dir: Path | str | None) -> None:
        if run_dir is None:
            return
        self.run_dir = Path(run_dir)

    def set_training_enabled(self, enabled: bool) -> None:
        self.training_enabled = bool(enabled)
        if not self.training_enabled:
            self._pending_reward = None

    def capture_reward(
        self,
        *,
        env_reward: float,
        shaped_reward: float,
        closer_bonus: float,
        farther_penalty: float,
        distance_delta_raw: float,
    ) -> None:
        if not self.enabled or not self.training_enabled or not self._is_main_process():
            return
        self._pending_reward = {
            "reward_env": float(env_reward),
            "reward_shaped": float(shaped_reward),
            "closer_bonus": float(closer_bonus),
            "farther_penalty": float(farther_penalty),
            "distance_delta_raw": float(distance_delta_raw),
        }

    def finalize_step(self, info: dict[str, Any], reward_components: dict[str, float]) -> None:
        if not self.enabled or not self.training_enabled or not self._is_main_process():
            return

        pending = self._pending_reward or {
            "reward_env": 0.0,
            "reward_shaped": 0.0,
            "closer_bonus": 0.0,
            "farther_penalty": 0.0,
            "distance_delta_raw": 0.0,
        }
        self._pending_reward = None
        self.global_step += 1

        self._append("reward_env", pending["reward_env"])
        self._append("reward_shaped", pending["reward_shaped"])
        success_value = self._success_value(info)
        done_value = self._done_value(info, success_value=success_value)
        self._record_episode_progress(info, success_value=success_value, done_value=done_value)
        self._append("success_rate", success_value)
        if done_value:
            self._append("episode_success_rate", success_value)
            self._append("episode_timeout_rate", 1.0 if self._truthy(info.get("episode_timeout")) else 0.0)
        self._append("target_grasped_rate", 1.0 if bool(info.get("target_grasped", False)) else 0.0)
        self._append("unstable_transition_rate", 1.0 if bool(info.get("unstable_transition", False)) else 0.0)
        self._append("reward_clip_rate", 1.0 if bool(info.get("reward_env_clipped", False)) else 0.0)
        self._append("reward_non_finite_rate", 1.0 if bool(info.get("reward_env_non_finite", False)) else 0.0)
        self._append_optional("distance_ee_to_object_xy", info.get("distance_ee_to_object_xy"))
        self._append_optional("sparse_success", info.get("sparse_success"))
        self._append_optional("caught_object_is_target", info.get("caught_object_is_target"))
        self._append_optional("target_motion_xy", info.get("target_motion_xy"))
        self._append_optional("relation_error", info.get("relation_error"))
        self._append_optional("action_saturation_rate", info.get("action_saturation_rate"))
        self._append_optional("lchol_group_score", info.get("lchol_group_score"))

        if self.global_step % self.every_global_steps != 0:
            return

        writer = self._ensure_writer()
        if writer is None:
            return

        for key, values in self._windows.items():
            if not values:
                continue
            writer.add_scalar(
                f"rollout_step/{key}_mean",
                float(sum(values) / len(values)),
                self.global_step,
            )
        self._write_episode_windows(writer)
        writer.flush()

    def close(self) -> None:
        writer = self.writer
        if writer is None:
            return
        self.writer = None
        try:
            writer.flush()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass

    def _append(self, key: str, value: float) -> None:
        self._windows[key].append(float(value))

    def _append_optional(self, key: str, value: Any) -> None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(numeric):
            return
        self._append(key, numeric)

    def _record_episode_progress(self, info: dict[str, Any], *, success_value: float, done_value: bool) -> None:
        key = self._episode_key(info)
        buffer = self._episode_buffers.setdefault(key, [])
        buffer.append(dict(info))
        if not done_value:
            return

        trajectory = self._episode_buffers.pop(key, buffer)
        instruction = self._safe_tag_token(str(info.get("instruction_type") or "unknown"))
        if instruction:
            window = self._instruction_episode_windows.setdefault(
                instruction,
                deque(maxlen=self.every_global_steps or 1),
            )
            window.append(float(success_value))
        if self._current_stage() == "dense":
            return
        shell_key = self._shell_success_key(info)
        if shell_key:
            window = self._shell_episode_windows.setdefault(
                shell_key,
                deque(maxlen=self.every_global_steps or 1),
            )
            window.append(float(success_value))

        achieved = self._subgoal_successes(trajectory)
        for option in _CDPR_CURRICULUM_OPTIONS:
            window = self._subgoal_episode_windows.setdefault(
                option,
                deque(maxlen=self.every_global_steps or 1),
            )
            window.append(1.0 if option in achieved else 0.0)

    def _write_episode_windows(self, writer) -> None:
        instruction_means: list[float] = []
        for instruction, values in sorted(self._instruction_episode_windows.items()):
            if not values:
                continue
            mean_value = float(sum(values) / len(values))
            instruction_means.append(mean_value)
            writer.add_scalar(
                f"rollout_episode/instruction_success_rate/{instruction}",
                mean_value,
                self.global_step,
            )
        if instruction_means:
            writer.add_scalar(
                "rollout_episode/instruction_success_rate_mean",
                float(sum(instruction_means) / len(instruction_means)),
                self.global_step,
            )
        if self._current_stage() == "dense":
            return
        for shell_key, values in sorted(self._shell_episode_windows.items()):
            if not values:
                continue
            writer.add_scalar(
                f"rollout_episode/shell_success_rate/{shell_key}",
                float(sum(values) / len(values)),
                self.global_step,
            )
        for option, values in sorted(self._subgoal_episode_windows.items()):
            if not values:
                continue
            writer.add_scalar(
                f"rollout_episode/subgoal_success_rate/{option}",
                float(sum(values) / len(values)),
                self.global_step,
            )

    def _episode_key(self, info: dict[str, Any]) -> tuple[Any, ...]:
        return (
            info.get("env_instance_id", ""),
            info.get("episode_index", ""),
            info.get("instruction_type", ""),
            info.get("curriculum_shell", ""),
            info.get("target_object_body", ""),
            info.get("reference_object_body", ""),
            info.get("second_reference_object_body", ""),
            info.get("scene", ""),
        )

    def _shell_success_key(self, info: dict[str, Any]) -> str:
        raw_shell = info.get("curriculum_shell")
        if raw_shell is None or raw_shell == "":
            return ""
        try:
            shell_id = int(raw_shell)
        except (TypeError, ValueError):
            return ""
        instruction = self._safe_tag_token(
            str(info.get("curriculum_instruction_id") or info.get("instruction_type") or "unknown")
        )
        if not instruction:
            instruction = "unknown"
        return f"{instruction}/shell_{max(0, shell_id):02d}"

    def _subgoal_successes(self, trajectory: Sequence[dict[str, Any]]) -> set[str]:
        achieved: set[str] = set()
        for info in trajectory:
            achieved.update(self._instant_subgoal_successes(info))
        return achieved

    def _instant_subgoal_successes(self, info: dict[str, Any]) -> set[str]:
        achieved: set[str] = set()
        if self._move_to_object_success(info):
            achieved.add("move_to_object")
        if self._grab_object_success(info):
            achieved.add("grab_object")
        if self._pick_up_success(info):
            achieved.add("pick_up")

        push = self._push_success(info)
        if push:
            achieved.add(push)

        relation = self._relation_success(info)
        if relation:
            achieved.add(relation)
        return achieved

    def _move_to_object_success(self, info: dict[str, Any]) -> bool:
        distance = self._finite_float(
            info.get("move_to_object_validation_distance_xy"),
            fallback=self._finite_float(
                info.get("move_to_object_xy_distance"),
                fallback=self._finite_float(info.get("distance_ee_to_object_xy"), fallback=float("nan")),
            ),
        )
        threshold = self._finite_float(
            info.get("move_to_object_validation_distance_threshold"),
            fallback=self._finite_float(info.get("move_to_object_xy_tolerance"), fallback=0.025),
        )
        return bool(math.isfinite(distance) and distance <= max(float(threshold), 1e-6))

    def _grab_object_success(self, info: dict[str, Any]) -> bool:
        if self._wrong_object_contact(info) >= 0.20:
            return False
        if self._truthy(info.get("caught_object_is_target")) or self._truthy(info.get("target_grasped")):
            return True
        if self._truthy(info.get("grab_require_caught", True)):
            return False
        distance_xy = self._finite_float(info.get("distance_ee_to_object_xy"), fallback=float("inf"))
        threshold = self._finite_float(info.get("grab_xy_tolerance"), fallback=0.025)
        return bool(self._truthy(info.get("gripper_closed")) and distance_xy <= max(float(threshold), 1e-6))

    def _pick_up_success(self, info: dict[str, Any]) -> bool:
        if self._wrong_object_contact(info) >= 0.20:
            return False
        if not (
            self._truthy(info.get("grasped"))
            or self._truthy(info.get("target_grasped"))
            or self._truthy(info.get("caught_object_is_target"))
        ):
            return False
        lift = self._finite_float(info.get("pick_target_lift"), fallback=0.0)
        threshold = max(self._finite_float(info.get("pick_lift_success_height"), fallback=0.05), 1e-6)
        return bool(lift >= threshold)

    def _push_success(self, info: dict[str, Any]) -> str:
        if self._wrong_object_contact(info) >= 0.20:
            return ""
        motion_x = self._finite_float(info.get("target_motion_x"), fallback=0.0)
        motion_y = self._finite_float(info.get("target_motion_y"), fallback=0.0)
        threshold = max(self._finite_float(info.get("push_success_displacement"), fallback=0.08), 0.02)
        if motion_x >= threshold:
            return "push_right"
        if motion_x <= -threshold:
            return "push_left"
        if motion_y >= threshold:
            return "push_forward"
        if motion_y <= -threshold:
            return "push_backward"
        return ""

    def _relation_success(self, info: dict[str, Any]) -> str:
        instruction = str(info.get("instruction_type") or "")
        if instruction == "put_into_plate" and self._success_value(info) >= 0.5:
            return "put_into_plate"
        if instruction == "move_between_objects" and self._success_value(info) >= 0.5:
            return "move_between_objects"
        if instruction in {
            "move_left_of_object",
            "move_right_of_object",
            "move_in_front_of_object",
            "move_behind_object",
            "put_in_front_of_object",
            "put_behind_object",
        } and self._success_value(info) >= 0.5:
            return instruction

        signed = self._finite_float(info.get("signed_relation_offset"), fallback=0.0)
        axis = int(round(self._finite_float(info.get("relation_axis"), fallback=0.0)))
        sign = self._finite_float(info.get("relation_axis_sign"), fallback=1.0)
        offset_key = "relation_front_behind_offset" if axis == 1 else "relation_left_right_offset"
        offset = max(
            self._finite_float(
                info.get(offset_key),
                fallback=self._finite_float(info.get("relation_left_right_offset"), fallback=0.08),
            ),
            1e-6,
        )
        motion_ok = self._truthy(info.get("relation_motion_ok", True))
        if signed >= offset and motion_ok:
            if axis == 1:
                return "move_behind_object" if sign > 0.0 else "move_in_front_of_object"
            return "move_right_of_object" if sign > 0.0 else "move_left_of_object"
        if signed <= -offset and motion_ok:
            if axis == 1:
                return "move_in_front_of_object" if sign > 0.0 else "move_behind_object"
            return "move_left_of_object" if sign > 0.0 else "move_right_of_object"
        return ""

    def _success_value(self, info: dict[str, Any]) -> float:
        if self._truthy(info.get("success")):
            return 1.0
        try:
            sparse_success = float(info.get("sparse_success"))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(sparse_success):
            return 0.0
        return 1.0 if sparse_success >= 0.5 else 0.0

    def _done_value(self, info: dict[str, Any], *, success_value: float) -> bool:
        if success_value >= 0.5:
            return True
        return bool(
            self._truthy(info.get("env_done"))
            or self._truthy(info.get("terminated"))
            or self._truthy(info.get("truncated"))
        )

    def _wrong_object_contact(self, info: dict[str, Any]) -> float:
        if self._truthy(info.get("caught_object_is_target")):
            return 0.0
        return max(
            0.0,
            self._finite_float(info.get("caught_object_score"), fallback=0.0),
        )

    @staticmethod
    def _finite_float(value: Any, *, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        return float(numeric) if math.isfinite(numeric) else float(fallback)

    @staticmethod
    def _safe_tag_token(value: str) -> str:
        token = str(value).strip().lower().replace(" ", "_")
        token = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in token)
        return token.strip("_")

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return bool(value)
        return bool(math.isfinite(numeric) and numeric >= 0.5)

    def _ensure_writer(self):
        if self.writer is not None:
            return self.writer
        if self.summary_writer_cls is None or self.run_dir is None:
            return None
        logdir = self.run_dir / "tensorboard"
        logdir.mkdir(parents=True, exist_ok=True)
        self.writer = self.summary_writer_cls(log_dir=str(logdir), flush_secs=10)
        if not self._registered_atexit:
            atexit.register(self.close)
            self._registered_atexit = True
        print(
            "[rlvla-fast] Rollout TensorBoard metrics: "
            f"every {self.every_global_steps} global steps -> {logdir} "
            "(tags: rollout_step/*, rollout_episode/*)",
            flush=True,
        )
        return self.writer

    def _is_main_process(self) -> bool:
        try:
            return int(os.environ.get("RANK", "0")) == 0
        except ValueError:
            return True

    def _window_len(self) -> int:
        if not self._windows:
            return 0
        return max(len(values) for values in self._windows.values())

    def _current_stage(self) -> str:
        if not callable(self.stage_fn):
            return ""
        try:
            return str(self.stage_fn() or "")
        except Exception:
            return ""


def _patch_rollout_tensorboard(module, *, every_global_steps: int) -> None:
    if every_global_steps <= 0:
        return

    logger = _RolloutTensorboardLogger(
        summary_writer_cls=getattr(module, "SummaryWriter", None),
        every_global_steps=every_global_steps,
        stage_fn=lambda: _current_tensorboard_stage(module),
    )

    original_make_run_dir = module.make_run_dir

    def _make_run_dir_with_rollout_tb(args):
        run_dir = original_make_run_dir(args)
        logger.set_run_dir(run_dir)
        return run_dir

    module.make_run_dir = _make_run_dir_with_rollout_tb

    original_shape_reward = module._shape_reward_with_delta_progress

    def _shape_reward_with_rollout_tb(
        env_reward: float,
        distance_before,
        distance_after,
        delta_closer_reward_coef: float,
        delta_farther_penalty_coef: float,
    ):
        shaped_reward, closer_bonus, farther_penalty, raw_delta = original_shape_reward(
            env_reward,
            distance_before,
            distance_after,
            delta_closer_reward_coef,
            delta_farther_penalty_coef,
        )
        logger.capture_reward(
            env_reward=float(env_reward),
            shaped_reward=float(shaped_reward),
            closer_bonus=float(closer_bonus),
            farther_penalty=float(farther_penalty),
            distance_delta_raw=float(raw_delta),
        )
        return shaped_reward, closer_bonus, farther_penalty, raw_delta

    module._shape_reward_with_delta_progress = _shape_reward_with_rollout_tb

    original_extract_reward_components = module._extract_reward_components

    def _extract_reward_components_with_rollout_tb(info: dict[str, Any]) -> dict[str, float]:
        reward_components = original_extract_reward_components(info)
        logger.finalize_step(info if isinstance(info, dict) else {}, reward_components)
        return reward_components

    module._extract_reward_components = _extract_reward_components_with_rollout_tb

    original_run_validation_rollouts = module.run_validation_rollouts

    def _run_validation_rollouts_without_rollout_tb(*args, **kwargs):
        logger.set_training_enabled(False)
        try:
            return original_run_validation_rollouts(*args, **kwargs)
        finally:
            logger.set_training_enabled(True)

    module.run_validation_rollouts = _run_validation_rollouts_without_rollout_tb


def _lchol_args_to_runtime_config(lchol_args: _LCHOLWrapperArgs):
    from rl_vla_bootstrapping.lchol.grpo_runtime import LCHOLGRPOConfig

    return LCHOLGRPOConfig(
        enabled=bool(lchol_args.enabled),
        group_score=str(lchol_args.group_score),
        hindsight_bc_coef=float(lchol_args.hindsight_bc_coef),
        hindsight_done_weight=float(lchol_args.hindsight_done_weight),
        hindsight_replay_capacity=int(lchol_args.hindsight_replay_capacity),
        hindsight_replay_ratio=float(lchol_args.hindsight_replay_ratio),
        hindsight_prefix_max_steps=int(lchol_args.hindsight_prefix_max_steps),
        option_prior_bc_coef=float(lchol_args.option_prior_bc_coef),
        option_prior_min_coef=float(lchol_args.option_prior_min_coef),
        option_prior_decay_updates=int(lchol_args.option_prior_decay_updates),
        curriculum=str(lchol_args.curriculum),
        strict_min_success_samples=int(lchol_args.strict_min_success_samples),
        weakest_mode_oversample_strength=float(lchol_args.weakest_mode_oversample_strength),
        newest_stage_weight=float(lchol_args.newest_stage_weight),
        reverse_promotion_success=float(lchol_args.reverse_promotion_success),
        reverse_demotion_success=float(lchol_args.reverse_demotion_success),
        reverse_validation_rollouts_per_shell=int(lchol_args.reverse_validation_rollouts_per_shell),
        reverse_min_train_updates_before_validation=int(lchol_args.reverse_min_train_updates_before_validation),
        reverse_max_shell_jump=int(lchol_args.reverse_max_shell_jump),
        reverse_saturation_abort_threshold=float(lchol_args.reverse_saturation_abort_threshold),
        reverse_sample_frontier_probability=float(lchol_args.reverse_sample_frontier_probability),
        reverse_sample_rehearsal_probability=float(lchol_args.reverse_sample_rehearsal_probability),
    )


def _patch_lchol_runtime(
    module,
    lchol_args: _LCHOLWrapperArgs | None,
    *,
    fast_args: _FastWrapperArgs | None = None,
) -> None:
    lchol_args = lchol_args or _LCHOLWrapperArgs()
    second_stage_grpo_actor_stats_path = (
        fast_args.second_stage_grpo_actor_stats_path if fast_args is not None else None
    )
    sparse_stage_init_log_std = (
        fast_args.sparse_stage_init_log_std if fast_args is not None else None
    )
    module._rlvla_lchol_runtime = None

    original_parse_args = module.parse_args

    def _parse_args_with_lchol():
        args = original_parse_args()
        setattr(args, "lchol_enabled", bool(lchol_args.enabled))
        for field_name, value in lchol_args.__dict__.items():
            setattr(args, f"lchol_{field_name}", value)
        return args

    module.parse_args = _parse_args_with_lchol

    def _build_runtime(args, *, is_main: bool, rank: int, seed: int):
        if not bool(getattr(args, "lchol_enabled", False)):
            return None
        try:
            from robots.cdpr.cdpr_dataset.cdpr_lchol_spec import CDPRLCHOLSpec
        except ModuleNotFoundError:
            from cdpr_dataset.cdpr_lchol_spec import CDPRLCHOLSpec
        from rl_vla_bootstrapping.lchol.grpo_runtime import LCHOLGRPORuntime

        runtime = LCHOLGRPORuntime(
            config=_lchol_args_to_runtime_config(lchol_args),
            spec=CDPRLCHOLSpec(),
            available_options=getattr(args, "instruction_types", None) or CDPRLCHOLSpec.option_names,
            seed=int(seed),
        )
        if is_main:
            print(
                "[lchol] enabled "
                f"group_score={lchol_args.group_score} "
                f"hindsight_bc_coef={lchol_args.hindsight_bc_coef} "
                f"replay_capacity={lchol_args.hindsight_replay_capacity} "
                f"curriculum={lchol_args.curriculum} "
                f"reverse_validation_rollouts={lchol_args.reverse_validation_rollouts_per_shell} "
                f"rank={rank}",
                flush=True,
            )
        return runtime

    def _set_runtime(runtime) -> None:
        module._rlvla_lchol_runtime = runtime

    def _runtime():
        return getattr(module, "_rlvla_lchol_runtime", None)

    def _phase_score(step_info: dict[str, Any], *, fallback: float) -> float:
        runtime = _runtime()
        if runtime is None:
            return float(fallback)
        return float(runtime.phase_score(step_info if isinstance(step_info, dict) else {}, fallback=float(fallback)))

    def _capture_candidate(
        *,
        obs: dict[str, Any],
        step_info: dict[str, Any],
        sampled_action: Any,
        group_score: float,
        update: int,
        global_step: int,
    ) -> None:
        runtime = _runtime()
        if runtime is None:
            return
        runtime.capture_candidate(
            obs=obs if isinstance(obs, dict) else {},
            step_info=step_info if isinstance(step_info, dict) else {},
            sampled_action=sampled_action,
            group_score=float(group_score),
            update=int(update),
            global_step=int(global_step),
        )

    def _after_rollout(*, update: int) -> None:
        runtime = _runtime()
        if runtime is None:
            return
        after_rollout = getattr(runtime, "after_rollout", None)
        if callable(after_rollout):
            after_rollout(update=int(update))

    def _bc_loss(policy, ppo_module, device, args, *, num_actions_chunk: int):
        runtime = _runtime()
        if runtime is None:
            return torch.zeros((), dtype=torch.float32, device=device)
        return runtime.bc_loss(
            policy=policy,
            ppo_module=ppo_module,
            device=device,
            args=args,
            num_actions_chunk=int(num_actions_chunk),
        )

    def _validate_grpo_transitions(transitions) -> None:
        invalid: list[str] = []
        non_pg = 0
        for idx, transition in enumerate(transitions):
            source = str(getattr(transition, "source", ""))
            if source != "pg":
                non_pg += 1
                invalid.append(f"idx={idx} source={source!r}")
            if bool(getattr(transition, "was_relabelled", False)):
                invalid.append(f"idx={idx} was_relabelled=True")
            if bool(getattr(transition, "from_replay", False)):
                invalid.append(f"idx={idx} from_replay=True")
            collection_prompt = str(getattr(transition, "collection_prompt", "") or "")
            instruction = str(getattr(transition, "instruction", "") or "")
            if collection_prompt and collection_prompt != instruction:
                invalid.append(f"idx={idx} collection_prompt_mismatch")
            try:
                old_logprob = float(getattr(transition, "logprob"))
            except (TypeError, ValueError):
                invalid.append(f"idx={idx} old_logprob_missing")
            else:
                if not math.isfinite(old_logprob):
                    invalid.append(f"idx={idx} old_logprob_non_finite")

        runtime = _runtime()
        if runtime is not None:
            recorder = getattr(runtime, "record_grpo_batch_audit", None)
            if callable(recorder):
                recorder(total=len(transitions), non_pg=non_pg)
        if invalid:
            preview = ", ".join(invalid[:8])
            extra = "" if len(invalid) <= 8 else f", ... +{len(invalid) - 8} more"
            raise RuntimeError(f"LC-HOL GRPO batch audit failed: {preview}{extra}")

    def _log_update(*, update: int, global_step: int, tb_writer, is_main: bool) -> None:
        runtime = _runtime()
        if runtime is None:
            return
        runtime.log_update(
            update=int(update),
            global_step=int(global_step),
            tb_writer=tb_writer,
            is_main=bool(is_main),
        )

    def _after_update(*, update: int, global_step: int, run_dir) -> None:
        runtime = _runtime()
        if runtime is None:
            return
        after_update = getattr(runtime, "after_update", None)
        if callable(after_update):
            after_update(update=int(update), global_step=int(global_step), run_dir=run_dir)

    def _should_stop_training(*, update: int) -> bool:
        del update
        runtime = _runtime()
        if runtime is None:
            return False
        should_stop = getattr(runtime, "should_stop_training", None)
        return bool(should_stop()) if callable(should_stop) else False

    def _policy_core(policy):
        return getattr(policy, "module", policy)

    def _log_std_values(log_std) -> list[float]:
        if log_std is None:
            return []
        values = getattr(log_std, "values", None)
        if values is not None:
            return [float(value) for value in values]
        try:
            tensor = log_std.detach().reshape(-1)
            if hasattr(tensor, "cpu"):
                tensor = tensor.cpu()
            if hasattr(tensor, "tolist"):
                return [float(value) for value in tensor.tolist()]
        except Exception:
            pass
        return []

    def _reset_policy_grpo_stats(policy, args) -> str | None:
        core = _policy_core(policy)
        log_std = getattr(core, "log_std", None)
        if log_std is None:
            return None
        initial = getattr(core, "_rlvla_grpo_initial_log_std", None)
        try:
            with torch.no_grad():
                if sparse_stage_init_log_std is not None:
                    log_std.fill_(float(sparse_stage_init_log_std))
                    source = "sparse_stage_init_log_std"
                elif initial is not None and hasattr(initial, "to"):
                    target = initial.to(device=log_std.device, dtype=log_std.dtype).reshape(log_std.shape)
                    log_std.copy_(target)
                    source = "policy_initial_log_std"
                else:
                    raw_init = float(getattr(args, "init_log_std", -1.2))
                    log_std.fill_(raw_init)
                    source = "trainer_init_log_std"
        except Exception as exc:
            print(f"[grpo_actor_stats] Failed to re-initialize at dense-to-sparse switch: {exc}", flush=True)
            return None
        return source

    def _load_or_reset_policy_grpo_stats(policy, args, *, update: int) -> str | None:
        core = _policy_core(policy)
        if second_stage_grpo_actor_stats_path is not None:
            resolved = _resolve_grpo_actor_stats_path(second_stage_grpo_actor_stats_path)
            if resolved is None:
                raise FileNotFoundError(
                    f"Could not find second-stage GRPO actor stats at {second_stage_grpo_actor_stats_path}"
                )
            _copy_grpo_log_std_from_actor_stats(core, resolved)
            print(
                "[grpo_actor_stats] Loaded second-stage stats at dense-to-sparse switch "
                f"update={int(update)} path={resolved}",
                flush=True,
            )
            return f"checkpoint:{resolved}"
        return _reset_policy_grpo_stats(policy, args)

    def _pre_update(policy, *, optimizer=None, args, update: int, run_dir) -> None:
        runtime = _runtime()
        if runtime is None:
            return
        sync_state = getattr(runtime, "sync_dense_gate_state", None)
        if callable(sync_state):
            sync_state(run_dir=run_dir)
        before_update = getattr(runtime, "before_update", None)
        if callable(before_update):
            before_update(update=int(update))
        consume_reset = getattr(runtime, "consume_grpo_stats_reset_request", None)
        if not callable(consume_reset) or not bool(consume_reset()):
            return
        core = _policy_core(policy)
        log_std = getattr(core, "log_std", None)
        before = _log_std_values(log_std)
        source = _load_or_reset_policy_grpo_stats(policy, args, update=int(update))
        if source is None:
            return
        optimizer_state_cleared = False
        if optimizer is not None and log_std is not None:
            state = getattr(optimizer, "state", None)
            if isinstance(state, dict) and log_std in state:
                state.pop(log_std, None)
                optimizer_state_cleared = True
        after = _log_std_values(log_std)
        event = {
            "update": int(update),
            "source": str(source),
            "before_log_std": before,
            "after_log_std": after,
            "optimizer_state_cleared": bool(optimizer_state_cleared),
        }
        event_path = Path(run_dir) / "grpo_stage_transition.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        print(
            "[grpo_actor_stats] Sparse-stage exploration initialized "
            f"update={int(update)} source={source} before={before} after={after} "
            f"optimizer_state_cleared={optimizer_state_cleared}",
            flush=True,
        )

    module._rlvla_lchol_build_runtime = _build_runtime
    module._rlvla_lchol_set_runtime = _set_runtime
    module._rlvla_lchol_pre_update = _pre_update
    module._rlvla_lchol_phase_score = _phase_score
    module._rlvla_lchol_capture_candidate = _capture_candidate
    module._rlvla_lchol_after_rollout = _after_rollout
    module._rlvla_lchol_after_update = _after_update
    module._rlvla_lchol_should_stop_training = _should_stop_training
    module._rlvla_lchol_bc_loss = _bc_loss
    module._rlvla_lchol_validate_grpo_transitions = _validate_grpo_transitions
    module._rlvla_lchol_log_update = _log_update

    original_reset = module.CDPRVisionLanguageEnv.reset

    def _reset_with_lchol_curriculum(self, options=None):
        runtime = _runtime()
        if runtime is not None:
            current_metadata = getattr(runtime, "current_task_metadata", None)
            if callable(current_metadata):
                os.environ["RLVLA_TASK_METADATA_JSON"] = json.dumps(current_metadata(), sort_keys=True)
            configure_env = getattr(runtime, "configure_env_for_current_stage", None)
            if callable(configure_env):
                configure_env(self)
            options = dict(options or {})
            if "instruction_type" not in options and "curriculum_shell" not in options:
                sampled_options = runtime.sample_reset_options()
                options.update({key: value for key, value in sampled_options.items() if value is not None})
        return original_reset(self, options=options)

    module.CDPRVisionLanguageEnv.reset = _reset_with_lchol_curriculum

    original_run_validation_rollouts = module.run_validation_rollouts

    def _action_saturation_rate(summary: dict[str, Any]) -> float:
        action_dim_stats = summary.get("action_dim_stats", {}) if isinstance(summary, dict) else {}
        values = []
        if isinstance(action_dim_stats, dict):
            for stats in action_dim_stats.values():
                if isinstance(stats, dict) and "sat_frac_abs_ge_0_99" in stats:
                    try:
                        values.append(float(stats["sat_frac_abs_ge_0_99"]))
                    except (TypeError, ValueError):
                        pass
        return float(max(values) if values else 0.0)

    def _run_validation_rollouts_with_reverse_frontier(*args, **kwargs):
        runtime = _runtime()
        dense_plan = runtime.dense_validation_plan() if runtime is not None else ()
        if dense_plan:
            base_next_reset_options = kwargs.get("next_reset_options")
            if base_next_reset_options is None:
                return original_run_validation_rollouts(*args, **kwargs)

            run_dir = kwargs.get("run_dir")
            update = kwargs.get("update")
            requested_num_episodes = int(kwargs.get("num_episodes", 1))
            validation_rollouts = int(runtime.dense_validation_episodes_per_instruction(requested_num_episodes))
            summaries: list[dict[str, Any]] = []
            dense_results: list[dict[str, Any]] = []
            for instruction_id in dense_plan:
                def _dense_next_reset_options(
                    instruction_id=instruction_id,
                    base_next_reset_options=base_next_reset_options,
                ):
                    options = dict(base_next_reset_options() if callable(base_next_reset_options) else {})
                    options.update({"instruction_type": str(instruction_id), "lchol_dense_stage": True})
                    return options

                dense_kwargs = dict(kwargs)
                dense_kwargs["num_episodes"] = validation_rollouts
                dense_kwargs["next_reset_options"] = _dense_next_reset_options
                summary = original_run_validation_rollouts(*args, **dense_kwargs)
                summary = dict(summary) if isinstance(summary, dict) else {}
                summary["instruction_id"] = str(instruction_id)
                summaries.append(summary)
                dense_results.append(
                    {
                        "instruction_id": str(instruction_id),
                        "success_rate": float(summary.get("success_rate", 0.0)),
                        "rollouts": int(summary.get("episodes", validation_rollouts)),
                        "mean_reward": float(summary.get("mean_env_return", 0.0)),
                    }
                )

            runtime.record_dense_validation(dense_results, run_dir=run_dir, update=update)
            mean_success = float(np.mean([item["success_rate"] for item in dense_results])) if dense_results else 0.0
            mean_env_return = float(np.mean([float(item.get("mean_env_return", 0.0)) for item in summaries])) if summaries else 0.0
            mean_shaped_return = (
                float(np.mean([float(item.get("mean_shaped_return", 0.0)) for item in summaries]))
                if summaries
                else 0.0
            )
            return {
                **(summaries[-1] if summaries else {}),
                "episodes": int(sum(item["rollouts"] for item in dense_results)),
                "mean_env_return": mean_env_return,
                "mean_shaped_return": mean_shaped_return,
                "success_rate": mean_success,
                "dense_gate_results": dense_results,
                "dense_gate_summaries": summaries,
            }

        plan = runtime.reverse_validation_plan() if runtime is not None else []
        if not plan:
            return original_run_validation_rollouts(*args, **kwargs)

        base_next_reset_options = kwargs.get("next_reset_options")
        if base_next_reset_options is None:
            return original_run_validation_rollouts(*args, **kwargs)

        run_dir = kwargs.get("run_dir")
        update = kwargs.get("update")
        requested_num_episodes = int(kwargs.get("num_episodes", 1))
        validation_rollouts = max(
            1,
            int(
                getattr(
                    getattr(runtime, "config", None),
                    "reverse_validation_rollouts_per_shell",
                    requested_num_episodes,
                )
            ),
        )
        summaries: list[dict[str, Any]] = []
        scheduler_results: list[dict[str, Any]] = []
        for instruction_id, shell_id in plan:
            def _shell_next_reset_options(
                instruction_id=instruction_id,
                shell_id=shell_id,
                base_next_reset_options=base_next_reset_options,
            ):
                options = dict(base_next_reset_options() if callable(base_next_reset_options) else {})
                options.update(runtime.reverse_validation_options(instruction_id, int(shell_id)))
                return options

            shell_kwargs = dict(kwargs)
            shell_kwargs["num_episodes"] = validation_rollouts
            shell_kwargs["next_reset_options"] = _shell_next_reset_options
            summary = original_run_validation_rollouts(*args, **shell_kwargs)
            summary = dict(summary) if isinstance(summary, dict) else {}
            summary["instruction_id"] = str(instruction_id)
            summary["curriculum_shell"] = int(shell_id)
            summaries.append(summary)
            scheduler_results.append(
                {
                    "instruction_id": str(instruction_id),
                    "shell_id": int(shell_id),
                    "success_rate": float(summary.get("success_rate", 0.0)),
                    "rollouts": int(summary.get("episodes", validation_rollouts)),
                    "action_saturation_rate": _action_saturation_rate(summary),
                }
            )

        runtime.record_reverse_validation(scheduler_results, run_dir=run_dir, update=update)
        mean_success = float(np.mean([item["success_rate"] for item in scheduler_results])) if scheduler_results else 0.0
        mean_env_return = float(np.mean([float(item.get("mean_env_return", 0.0)) for item in summaries])) if summaries else 0.0
        mean_shaped_return = (
            float(np.mean([float(item.get("mean_shaped_return", 0.0)) for item in summaries]))
            if summaries
            else 0.0
        )
        return {
            **(summaries[-1] if summaries else {}),
            "episodes": int(sum(item["rollouts"] for item in scheduler_results)),
            "mean_env_return": mean_env_return,
            "mean_shaped_return": mean_shaped_return,
            "success_rate": mean_success,
            "reverse_frontier_results": scheduler_results,
            "reverse_frontier_summaries": summaries,
        }

    module.run_validation_rollouts = _run_validation_rollouts_with_reverse_frontier


def _enable_fast_runtime_flags() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def main() -> None:
    external_arg, forwarded_argv, fast_args = _split_wrapper_argv(sys.argv[1:])
    external_script = _resolve_external_script(external_arg)
    _patch_global_distributed_timeout(timeout_seconds=fast_args.ddp_timeout_seconds)
    module = _load_external_module(
        external_script,
        enable_lchol=bool(fast_args.lchol and fast_args.lchol.enabled),
        enable_ddp_sync=True,
        enable_lr_scheduler=str(fast_args.lr_scheduler).strip().lower() not in {"", "constant", "none"},
    )

    _enable_fast_runtime_flags()
    _patch_rollout_image_resize(module, image_size=fast_args.rollout_image_size)
    _patch_memory_logging(module)
    _patch_prepare_inputs(module)
    _patch_scene_wrapper_cache(module)
    _patch_scene_cache_prebuild_progress(module, forwarded_argv)
    _patch_fresh_scene_cache_prebuild(module, forwarded_argv)
    _patch_desk_texture_prepare(module)
    _patch_distributed_timeout(module, timeout_seconds=fast_args.ddp_timeout_seconds)
    _patch_ddp_sync(module, rollout_sync_interval=fast_args.ddp_rollout_sync_interval)
    _patch_resume_artifacts(module, forwarded_argv, fast_args)
    _patch_lr_scheduler(module, fast_args)
    _patch_training_reset_retries(
        module,
        max_attempts=fast_args.max_train_reset_attempts,
    )
    _patch_lchol_runtime(module, fast_args.lchol, fast_args=fast_args)
    _patch_tensorboard_metric_filter(module, profile=fast_args.tensorboard_metric_profile)
    _patch_rollout_tensorboard(
        module,
        every_global_steps=fast_args.tensorboard_rollout_every_global_steps,
    )

    sys.argv = [str(external_script)] + forwarded_argv
    module.main()


if __name__ == "__main__":
    main()
