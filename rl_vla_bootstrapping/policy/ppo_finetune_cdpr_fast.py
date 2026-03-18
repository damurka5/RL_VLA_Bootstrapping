#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


def _split_wrapper_argv(argv: Sequence[str]) -> tuple[Path | None, list[str]]:
    forwarded: list[str] = []
    external_script: Path | None = None

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--external_ppo_script":
            if idx + 1 >= len(argv):
                raise SystemExit("--external_ppo_script expects a path.")
            external_script = Path(argv[idx + 1]).expanduser().resolve()
            idx += 2
            continue
        forwarded.append(arg)
        idx += 1

    return external_script, forwarded


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
    external_arg, forwarded_argv = _split_wrapper_argv(sys.argv[1:])
    external_script = _resolve_external_script(external_arg)
    module = _load_external_module(external_script)

    _enable_fast_runtime_flags()
    _patch_prepare_inputs(module)
    _patch_scene_wrapper_cache(module)

    sys.argv = [str(external_script)] + forwarded_argv
    module.main()


if __name__ == "__main__":
    main()
