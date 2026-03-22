#!/usr/bin/env python3
from __future__ import annotations

import atexit
import importlib
import importlib.util
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image


@dataclass(frozen=True)
class _FastWrapperArgs:
    tensorboard_rollout_every_global_steps: int = 0


def _split_wrapper_argv(argv: Sequence[str]) -> tuple[Path | None, list[str], _FastWrapperArgs]:
    forwarded: list[str] = []
    external_script: Path | None = None
    tensorboard_rollout_every_global_steps = 0

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--external_ppo_script":
            if idx + 1 >= len(argv):
                raise SystemExit("--external_ppo_script expects a path.")
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
        forwarded.append(arg)
        idx += 1

    return external_script, forwarded, _FastWrapperArgs(
        tensorboard_rollout_every_global_steps=tensorboard_rollout_every_global_steps
    )


def _candidate_external_scripts() -> list[Path]:
    env_candidate = os.environ.get("RLVLA_EXTERNAL_PPO_SCRIPT")
    candidates: list[Path] = []
    if env_candidate:
        candidates.append(Path(env_candidate).expanduser().resolve())

    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        root = Path(entry).expanduser().resolve()
        candidates.append(root / "vla-scripts" / "ppo_finetune_cdpr.py")
        candidates.append(root.parent / "openvla-oft" / "vla-scripts" / "ppo_finetune_cdpr.py")

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
        "Could not locate external OpenVLA PPO trainer. "
        "Pass --external_ppo_script or set RLVLA_EXTERNAL_PPO_SCRIPT. "
        f"Checked: {attempted}"
    )


def _load_external_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("rlvla_external_ppo_finetune_cdpr", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _to_pil_rgb(image) -> Image.Image:
    return Image.fromarray(image.astype("uint8")).convert("RGB")


def _patch_prepare_inputs(module) -> None:
    policy_cls = module.OpenVLAPPOPolicy
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


def _patch_scene_wrapper_cache(module) -> None:
    wrapper_bundle_exists = _load_wrapper_bundle_checker()
    if wrapper_bundle_exists is None:
        return

    env_cls = module.CDPRVisionLanguageEnv
    original_activate = env_cls._activate_scene_wrapper_cache

    def _activate_scene_wrapper_cache_checked(self, scene_wrapper_cache, texture_name_by_wrapper):
        out = original_activate(self, scene_wrapper_cache, texture_name_by_wrapper)
        rl = self.env
        cached_builder = getattr(rl, "_build_wrapper", None)
        if cached_builder is None or not hasattr(rl, "_build_wrapper_original"):
            return out

        def _build_wrapper_checked(this, scene):
            scene_name = str(getattr(scene, "name", ""))
            variants_local = list(self._scene_wrapper_cache.get(scene_name) or [])
            if variants_local:
                available_variants = [Path(path).resolve() for path in variants_local if wrapper_bundle_exists(Path(path))]
                if len(available_variants) != len(variants_local):
                    self._scene_wrapper_cache[scene_name] = available_variants
                    warned = getattr(self, "_rlvla_scene_cache_repair_warned", set())
                    if scene_name not in warned:
                        print(
                            f"[env_cache] Repaired unavailable cached wrappers for scene '{scene_name}' "
                            f"({len(variants_local)} -> {len(available_variants)} variants).",
                            flush=True,
                        )
                        warned.add(scene_name)
                        self._rlvla_scene_cache_repair_warned = warned
                if not available_variants:
                    this._desk_texture_name = ""
                    return this._build_wrapper_original(scene)
            return cached_builder(scene)

        rl._build_wrapper = module.types.MethodType(_build_wrapper_checked, rl)
        return out

    env_cls._activate_scene_wrapper_cache = _activate_scene_wrapper_cache_checked


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
        self._windows: dict[str, deque[float]] = {
            "reward_env": deque(maxlen=self.every_global_steps or 1),
            "reward_shaped": deque(maxlen=self.every_global_steps or 1),
            "closer_bonus": deque(maxlen=self.every_global_steps or 1),
            "farther_penalty": deque(maxlen=self.every_global_steps or 1),
            "distance_delta_raw": deque(maxlen=self.every_global_steps or 1),
            "reward_component_r_xyz": deque(maxlen=self.every_global_steps or 1),
            "reward_component_r_orient": deque(maxlen=self.every_global_steps or 1),
            "reward_component_r_obj": deque(maxlen=self.every_global_steps or 1),
            "reward_component_r_success": deque(maxlen=self.every_global_steps or 1),
            "success_rate": deque(maxlen=self.every_global_steps or 1),
            "target_grasped_rate": deque(maxlen=self.every_global_steps or 1),
            "unstable_transition_rate": deque(maxlen=self.every_global_steps or 1),
            "reward_clip_rate": deque(maxlen=self.every_global_steps or 1),
            "reward_non_finite_rate": deque(maxlen=self.every_global_steps or 1),
            "distance_to_goal": deque(maxlen=self.every_global_steps or 1),
            "action_saturation_penalty": deque(maxlen=self.every_global_steps or 1),
            "action_saturation_rate": deque(maxlen=self.every_global_steps or 1),
        }

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
        self._append("closer_bonus", pending["closer_bonus"])
        self._append("farther_penalty", pending["farther_penalty"])
        self._append("distance_delta_raw", pending["distance_delta_raw"])
        self._append("reward_component_r_xyz", reward_components.get("r_xyz", 0.0))
        self._append("reward_component_r_orient", reward_components.get("r_orient", 0.0))
        self._append("reward_component_r_obj", reward_components.get("r_obj", 0.0))
        self._append("reward_component_r_success", reward_components.get("r_success", 0.0))
        self._append("success_rate", 1.0 if bool(info.get("success", False)) else 0.0)
        self._append("target_grasped_rate", 1.0 if bool(info.get("target_grasped", False)) else 0.0)
        self._append("unstable_transition_rate", 1.0 if bool(info.get("unstable_transition", False)) else 0.0)
        self._append("reward_clip_rate", 1.0 if bool(info.get("reward_env_clipped", False)) else 0.0)
        self._append("reward_non_finite_rate", 1.0 if bool(info.get("reward_env_non_finite", False)) else 0.0)
        self._append_optional("distance_to_goal", info.get("distance_to_goal"))
        self._append_optional("action_saturation_penalty", info.get("action_saturation_penalty"))
        self._append_optional("action_saturation_rate", info.get("action_saturation_rate"))

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
        writer.add_scalar("rollout_step/window_size", float(self._window_len()), self.global_step)
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
            f"every {self.every_global_steps} global steps -> {logdir} (tags: rollout_step/*)",
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
    module = _load_external_module(external_script)

    _enable_fast_runtime_flags()
    _patch_prepare_inputs(module)
    _patch_scene_wrapper_cache(module)
    _patch_rollout_tensorboard(
        module,
        every_global_steps=fast_args.tensorboard_rollout_every_global_steps,
    )

    sys.argv = [str(external_script)] + forwarded_argv
    module.main()


if __name__ == "__main__":
    main()
