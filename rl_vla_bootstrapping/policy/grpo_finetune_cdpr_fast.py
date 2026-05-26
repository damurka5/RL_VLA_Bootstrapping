#!/usr/bin/env python3
from __future__ import annotations

import atexit
import importlib
import importlib.util
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
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
)


@dataclass(frozen=True)
class _FastWrapperArgs:
    tensorboard_rollout_every_global_steps: int = 0
    tensorboard_metric_profile: str = "compact"
    resume_actor_stats: bool = True
    ddp_timeout_seconds: int = 0
    lr_scheduler: str = "constant"
    lr_warmup_updates: int = 0
    lr_min_factor: float = 1.0
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
    resume_actor_stats = True
    lr_scheduler = "constant"
    lr_warmup_updates = 0
    lr_min_factor = 1.0
    try:
        ddp_timeout_seconds = max(0, int(os.environ.get("RLVLA_DDP_TIMEOUT_SECONDS", "0")))
    except ValueError:
        ddp_timeout_seconds = 0
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
        if arg in ("--resume_actor_stats", "--resume-actor-stats"):
            resume_actor_stats = True
            idx += 1
            continue
        if arg in ("--no-resume_actor_stats", "--no-resume-actor-stats"):
            resume_actor_stats = False
            idx += 1
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
        resume_actor_stats=resume_actor_stats,
        ddp_timeout_seconds=ddp_timeout_seconds,
        lr_scheduler=lr_scheduler,
        lr_warmup_updates=lr_warmup_updates,
        lr_min_factor=lr_min_factor,
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


def _infer_resume_artifacts(
    argv: Sequence[str],
    *,
    resume_actor_stats: bool = True,
) -> _ResumeArtifacts:
    if not resume_actor_stats:
        return _ResumeArtifacts()

    raw_values = [
        _extract_cli_arg_value(argv, "--action_head_path"),
        _extract_cli_arg_value(argv, "--adapter_path"),
    ]
    for raw_value in raw_values:
        if not raw_value:
            continue
        for checkpoint_dir in _candidate_checkpoint_dirs(raw_value):
            grpo_actor_stats_path = checkpoint_dir / "grpo_actor_stats.pt"
            ppo_actor_stats_path = checkpoint_dir / "ppo_actor_stats.pt"
            resolved_actor_stats_path = None
            if grpo_actor_stats_path.is_file():
                resolved_actor_stats_path = grpo_actor_stats_path
            elif ppo_actor_stats_path.is_file():
                resolved_actor_stats_path = ppo_actor_stats_path
            if resolved_actor_stats_path is not None:
                return _ResumeArtifacts(
                    checkpoint_dir=checkpoint_dir,
                    actor_stats_path=resolved_actor_stats_path,
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
    ]
    for old, new in replacements:
        if old not in transformed:
            raise RuntimeError(
                "Could not apply LC-HOL patch to external GRPO trainer; "
                f"missing source anchor: {old[:80]!r}"
            )
        transformed = transformed.replace(old, new, 1)
    return transformed


def _transform_external_grpo_source_for_ddp_sync(source: str) -> str:
    transformed = source
    update_anchor = "        for update in range(1, args.total_updates + 1):\n            policy.eval()\n"
    if update_anchor in transformed:
        transformed = transformed.replace(
            update_anchor,
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_ddp_sync(\"pre_update\", update=update)\n"
            "            policy.eval()\n",
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
    return transformed


def _transform_external_grpo_source_for_lr_scheduler(source: str) -> str:
    transformed = source
    loop_anchors = (
        (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_ddp_sync(\"pre_update\", update=update)\n"
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
        cached_builder = getattr(rl, "_build_wrapper", None)
        if cached_builder is None or not hasattr(rl, "_build_wrapper_original"):
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
                available_variants = [
                    Path(path).resolve()
                    for path in variants_local
                    if _variant_supports_scene(Path(path), scene)
                ]
                if len(available_variants) != len(variants_local):
                    self._scene_wrapper_cache[scene_name] = available_variants
                    warned = getattr(self, "_rlvla_scene_cache_repair_warned", set())
                    warn_key = (scene_name, tuple(getattr(scene, "objects", ()) or ()))
                    if warn_key not in warned:
                        print(
                            f"[env_cache] Repaired unavailable cached wrappers for scene '{scene_name}' "
                            f"({len(variants_local)} -> {len(available_variants)} variants).",
                            flush=True,
                        )
                        warned.add(warn_key)
                        self._rlvla_scene_cache_repair_warned = warned
                if not available_variants:
                    this._desk_texture_name = ""
                    return _call_original_builder(this, scene, ee_start=ee_start)
            try:
                if ee_start is not None:
                    return cached_builder(scene, ee_start=ee_start)
            except TypeError as exc:
                if "ee_start" not in str(exc):
                    raise
            return cached_builder(scene)

        rl._build_wrapper = module.types.MethodType(_build_wrapper_checked, rl)
        return out

    env_cls._activate_scene_wrapper_cache = _activate_scene_wrapper_cache_checked


def _patch_desk_texture_prepare(module) -> None:
    original_prepare = getattr(module, "_prepare_desk_textures_dir", None)
    broadcast_object = getattr(module, "_broadcast_object", None)
    if not callable(original_prepare) or not callable(broadcast_object):
        return

    def _prepare_desk_textures_dir_single_writer(src_dir, run_dir, is_main, rank, max_textures):
        rank_int = int(rank)
        if rank_int != 0:
            return broadcast_object(None, rank_int)
        return original_prepare(src_dir, run_dir, is_main, rank_int, max_textures)

    module._prepare_desk_textures_dir = _prepare_desk_textures_dir_single_writer


def _patch_distributed_timeout(module, *, timeout_seconds: int) -> None:
    if timeout_seconds <= 0:
        return
    ppo_module = getattr(module, "ppo", None)
    if ppo_module is None:
        return
    dist_module = getattr(ppo_module, "dist", None)
    torch_module = getattr(ppo_module, "torch", torch)
    if dist_module is None or not callable(getattr(dist_module, "init_process_group", None)):
        return

    timeout = timedelta(seconds=max(1, int(timeout_seconds)))

    def _init_distributed_with_timeout():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if world_size > 1 and not dist_module.is_initialized():
            backend = "nccl" if torch_module.cuda.is_available() else "gloo"
            dist_module.init_process_group(backend=backend, timeout=timeout)
            if rank == 0:
                print(
                    f"[rlvla-ddp] process-group timeout set to {int(timeout.total_seconds())}s",
                    flush=True,
                )
        return rank, local_rank, world_size

    ppo_module._init_distributed = _init_distributed_with_timeout


def _patch_ddp_sync(module) -> None:
    dist_module = getattr(module, "dist", None)
    if dist_module is None:
        return

    def _rlvla_ddp_sync(label: str, *, update: int) -> None:
        if not (dist_module.is_available() and dist_module.is_initialized()):
            return
        import time

        start = time.monotonic()
        dist_module.barrier()
        waited = time.monotonic() - start
        if waited >= 30.0 and os.environ.get("RANK", "0") == "0":
            print(
                f"[rlvla-ddp] waited {waited:.1f}s at {label} sync for update {int(update)}",
                flush=True,
            )

    module._rlvla_ddp_sync = _rlvla_ddp_sync


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


def _patch_resume_artifacts(module, forwarded_argv: Sequence[str], fast_args: _FastWrapperArgs) -> None:
    artifacts = _infer_resume_artifacts(
        forwarded_argv,
        resume_actor_stats=fast_args.resume_actor_stats,
    )
    if artifacts.checkpoint_dir is None or artifacts.actor_stats_path is None:
        return

    original_policy_init = module.OpenVLAGRPOPolicy.__init__

    def _policy_init_with_actor_resume(self, *args, **kwargs):
        original_policy_init(self, *args, **kwargs)
        state_raw = torch.load(artifacts.actor_stats_path, map_location="cpu")
        log_std = _find_log_std_tensor(state_raw)
        if log_std is None:
            raise RuntimeError(f"Could not locate `log_std` tensor in {artifacts.actor_stats_path}")
        flat_log_std = log_std.detach().reshape(-1)
        if flat_log_std.numel() != self.log_std.numel():
            raise RuntimeError(
                "Resumed GRPO actor stats shape mismatch. "
                f"checkpoint={tuple(flat_log_std.shape)}, target={tuple(self.log_std.shape)}"
            )
        target = flat_log_std.to(device=self.device, dtype=self.log_std.dtype).reshape(self.log_std.shape)
        with torch.no_grad():
            self.log_std.copy_(target)
        print(f"[grpo_actor_stats] Loaded from {artifacts.actor_stats_path}", flush=True)

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


def _tensorboard_tag_allowed(tag: str, *, profile: str = "compact", kind: str = "scalar") -> bool:
    if str(profile).strip().lower() == "full":
        return True
    if str(kind) != "scalar":
        return False
    tag = str(tag)
    return tag in _COMPACT_TENSORBOARD_SCALARS or any(
        tag.startswith(prefix) for prefix in _COMPACT_TENSORBOARD_PREFIXES
    )


def _patch_tensorboard_metric_filter(module, *, profile: str) -> None:
    if str(profile).strip().lower() == "full":
        return
    writer_cls = getattr(module, "SummaryWriter", None)
    if writer_cls is None:
        return

    class _FilteredSummaryWriter:
        def __init__(self, *args, **kwargs):
            self._inner = writer_cls(*args, **kwargs)

        def add_scalar(self, tag, scalar_value, *args, **kwargs):
            if _tensorboard_tag_allowed(str(tag), profile=profile, kind="scalar"):
                return self._inner.add_scalar(tag, scalar_value, *args, **kwargs)
            return None

        def add_histogram(self, tag, values, *args, **kwargs):
            if _tensorboard_tag_allowed(str(tag), profile=profile, kind="histogram"):
                return self._inner.add_histogram(tag, values, *args, **kwargs)
            return None

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    module.SummaryWriter = _FilteredSummaryWriter
    print("[rlvla-fast] TensorBoard metric profile: compact", flush=True)


class _RolloutTensorboardLogger:
    def __init__(self, summary_writer_cls, every_global_steps: int):
        self.summary_writer_cls = summary_writer_cls
        self.every_global_steps = max(0, int(every_global_steps))
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
        for instruction, values in sorted(self._instruction_episode_windows.items()):
            if not values:
                continue
            writer.add_scalar(
                f"rollout_episode/instruction_success_rate/{instruction}",
                float(sum(values) / len(values)),
                self.global_step,
            )
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
        threshold = max(self._finite_float(info.get("push_success_displacement"), fallback=0.08), 0.02)
        if motion_x >= threshold:
            return "push_right"
        if motion_x <= -threshold:
            return "push_left"
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
            "put_in_front_of_object",
            "put_behind_object",
        } and self._success_value(info) >= 0.5:
            return instruction

        signed = self._finite_float(info.get("signed_relation_offset"), fallback=0.0)
        offset = max(self._finite_float(info.get("relation_left_right_offset"), fallback=0.08), 1e-6)
        motion_ok = self._truthy(info.get("relation_motion_ok", True))
        if signed >= offset and motion_ok:
            return "move_right_of_object"
        if signed <= -offset and motion_ok:
            return "move_left_of_object"
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


def _patch_rollout_tensorboard(module, *, every_global_steps: int) -> None:
    if every_global_steps <= 0:
        return

    logger = _RolloutTensorboardLogger(
        summary_writer_cls=getattr(module, "SummaryWriter", None),
        every_global_steps=every_global_steps,
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


def _patch_lchol_runtime(module, lchol_args: _LCHOLWrapperArgs | None) -> None:
    lchol_args = lchol_args or _LCHOLWrapperArgs()
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

    module._rlvla_lchol_build_runtime = _build_runtime
    module._rlvla_lchol_set_runtime = _set_runtime
    module._rlvla_lchol_phase_score = _phase_score
    module._rlvla_lchol_capture_candidate = _capture_candidate
    module._rlvla_lchol_after_rollout = _after_rollout
    module._rlvla_lchol_bc_loss = _bc_loss
    module._rlvla_lchol_validate_grpo_transitions = _validate_grpo_transitions
    module._rlvla_lchol_log_update = _log_update

    original_reset = module.CDPRVisionLanguageEnv.reset

    def _reset_with_lchol_curriculum(self, options=None):
        runtime = _runtime()
        if runtime is not None:
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
    module = _load_external_module(
        external_script,
        enable_lchol=bool(fast_args.lchol and fast_args.lchol.enabled),
        enable_ddp_sync=True,
        enable_lr_scheduler=str(fast_args.lr_scheduler).strip().lower() not in {"", "constant", "none"},
    )

    _enable_fast_runtime_flags()
    _patch_prepare_inputs(module)
    _patch_scene_wrapper_cache(module)
    _patch_desk_texture_prepare(module)
    _patch_distributed_timeout(module, timeout_seconds=fast_args.ddp_timeout_seconds)
    _patch_ddp_sync(module)
    _patch_resume_artifacts(module, forwarded_argv, fast_args)
    _patch_lr_scheduler(module, fast_args)
    _patch_lchol_runtime(module, fast_args.lchol)
    _patch_tensorboard_metric_filter(module, profile=fast_args.tensorboard_metric_profile)
    _patch_rollout_tensorboard(
        module,
        every_global_steps=fast_args.tensorboard_rollout_every_global_steps,
    )

    sys.argv = [str(external_script)] + forwarded_argv
    module.main()


if __name__ == "__main__":
    main()
