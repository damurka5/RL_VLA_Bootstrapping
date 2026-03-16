from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rl_vla_bootstrapping.core.commands import ensure_directory
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.embodiments.mujoco import MujocoEmbodiment
from rl_vla_bootstrapping.simulation.scene_builder import build_scene_xml, preview_selection
from robots.cdpr.cdpr_dataset.rl_instruction_tasks import canonical_object_name
from robots.cdpr.cdpr_mujoco.policy_control import (
    CDPRPolicyControlSpec,
    apply_normalized_cdpr_action,
    policy_action_frequency_hz,
    policy_action_period_s,
)


@dataclass
class _FallbackGenerateConfig:
    # Minimal OpenVLA config surface needed by `experiments.robot.openvla_utils`.
    model_family: str = "openvla"
    pretrained_checkpoint: str | Path = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = False
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: str | Path | None = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an OpenVLA/OFT policy on the CDPR embodiment.")
    parser.add_argument("--config", required=True, help="Path to bootstrap YAML/JSON/TOML config.")
    parser.add_argument("--adapter-path", required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--action-head-path", required=True, help="Path to the RL action-head checkpoint (.pt).")
    parser.add_argument("--base-ckpt", default=None, help="Optional override for the base VLA checkpoint.")
    parser.add_argument("--scene", default=None, help="Scene name override. Defaults to config preview scene.")
    parser.add_argument("--target-object", default=None, help="Primary target object.")
    parser.add_argument(
        "--distractor",
        action="append",
        default=[],
        help="Additional distractor object. Repeat the flag to add more than one.",
    )
    parser.add_argument("--instruction", default=None, help="Instruction text. Defaults to 'pick up <target>'.")
    parser.add_argument("--steps", type=int, default=150, help="Maximum number of policy actions to execute.")
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=None,
        help="Open-loop chunk length. Defaults to config action codec chunk size.",
    )
    parser.add_argument("--hold-steps", type=int, default=None, help="Extra simulator substeps per policy action.")
    parser.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass through OpenVLA center-crop behavior.",
    )
    parser.add_argument("--run-dir", default=None, help="Optional output directory.")
    parser.add_argument("--run-name", default="policy_rollout", help="Artifact name prefix for saved results.")
    parser.add_argument(
        "--action-guard",
        type=float,
        default=1.25,
        help="Warn and clip if the policy outputs absolute values larger than this.",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Logging cadence in policy steps.")
    parser.add_argument("--no-save", action="store_true", help="Skip simulator-side artifact export.")
    return parser


def _load_generate_config() -> tuple[type[Any], str | None]:
    try:
        module = importlib.import_module("experiments.robot.libero.run_libero_eval")
        GenerateConfig = module.GenerateConfig
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)
        if not missing:
            missing = str(exc).replace("No module named", "").strip().strip("'\"") or "unknown"
        note = (
            "LIBERO eval dependencies were not importable "
            f"(`{missing}`); using the runner-local GenerateConfig fallback."
        )
        return _FallbackGenerateConfig, note
    return GenerateConfig, None


def _load_openvla_modules(policy_repo: Path):
    if str(policy_repo) not in sys.path:
        sys.path.insert(0, str(policy_repo))
    try:
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
        )
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Could not import OpenVLA/OFT runtime dependencies. "
            f"Ensure the policy repo exists and the active environment has its requirements installed: {exc}"
        ) from exc
    generate_config, note = _load_generate_config()
    return generate_config, get_action_head, get_processor, get_proprio_projector, get_vla, PeftModel, note


def _default_objects(config, target_object: str | None, distractors: list[str]) -> list[str]:
    if target_object:
        names = [target_object]
        names.extend(str(item) for item in distractors if item)
        return names
    _, preview_objects = preview_selection(config)
    names = list(preview_objects)
    if distractors:
        names.extend(str(item) for item in distractors if item)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _default_instruction(target_object: str | None, object_names: list[str]) -> str:
    target = target_object or (object_names[0] if object_names else "object")
    return f"pick up {canonical_object_name(target)}"


def _control_spec_from_config(config, hold_steps: int | None) -> CDPRPolicyControlSpec:
    limits = config.embodiment.action_adapter.controller_limits
    return CDPRPolicyControlSpec(
        xyz_limits=(
            limits["x"],
            limits["y"],
            limits["z"],
        ),
        action_step_xyz=float(config.training.rl.args.get("action_step_xyz", config.embodiment.action_adapter.controller_scales.get("x", 0.005))),
        action_step_yaw=float(config.training.rl.args.get("action_step_yaw", config.embodiment.action_adapter.controller_scales.get("yaw", 0.25))),
        open_gripper_threshold=float(config.embodiment.action_adapter.open_gripper_threshold),
        close_gripper_threshold=float(config.embodiment.action_adapter.close_gripper_threshold),
        hold_steps=int(config.training.rl.args.get("hold_steps", 0) if hold_steps is None else hold_steps),
    )


def _make_observation(sim, instruction: str, gripper_range: tuple[float, float]) -> tuple[dict[str, np.ndarray], int]:
    full_rgb = sim.capture_frame(sim.overview_cam, "overview")
    wrist_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")

    ee = np.asarray(sim.get_end_effector_position(), dtype=np.float32).reshape(-1)[:3]
    yaw = float(sim.get_yaw()) if hasattr(sim, "get_yaw") else 0.0
    grip_phys = float(sim.get_gripper_opening()) if hasattr(sim, "get_gripper_opening") else float(gripper_range[1])
    g_lo, g_hi = gripper_range
    grip_norm = float(np.clip((grip_phys - g_lo) / max(g_hi - g_lo, 1e-6), 0.0, 1.0))

    state = np.array([ee[0], ee[1], ee[2], yaw, grip_norm], dtype=np.float32)
    obs = {
        "full_image": np.ascontiguousarray(full_rgb),
        "wrist_image": np.ascontiguousarray(wrist_rgb),
        "state": state,
        "task_description": instruction,
    }
    return obs, state.size


def _iter_model_candidates(model: Any) -> list[Any]:
    out: list[Any] = []
    queue: list[Any] = [model]
    seen: set[int] = set()

    while queue:
        cur = queue.pop(0)
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        out.append(cur)

        for attr in ("module", "model", "base_model"):
            child = getattr(cur, attr, None)
            if child is not None and id(child) not in seen:
                queue.append(child)

    return out


def _resolve_vision_backbone(vla: Any) -> Any:
    for obj in _iter_model_candidates(vla):
        backbone = getattr(obj, "vision_backbone", None)
        if backbone is not None:
            return backbone
    return None


def _set_num_images_in_input(vla: Any, num_images: int) -> int:
    requested = int(num_images)

    for obj in _iter_model_candidates(vla):
        if hasattr(obj, "set_num_images_in_input"):
            obj.set_num_images_in_input(requested)
            return requested
        if hasattr(obj, "num_images_in_input"):
            setattr(obj, "num_images_in_input", requested)
            return requested

    backbone = _resolve_vision_backbone(vla)
    if backbone is not None:
        if hasattr(backbone, "set_num_images_in_input"):
            backbone.set_num_images_in_input(requested)
            return requested
        if hasattr(backbone, "num_images_in_input"):
            setattr(backbone, "num_images_in_input", requested)
            return requested
        backbone_cfg = getattr(backbone, "config", None)
        if backbone_cfg is not None and hasattr(backbone_cfg, "num_images_in_input"):
            setattr(backbone_cfg, "num_images_in_input", requested)
            return requested

    if requested != 1:
        print("[warn] OpenVLA image-count control API not found; falling back to single-image mode.")
    return 1


def _motion_diagnostics_for_log(
    action: np.ndarray,
    control_spec: CDPRPolicyControlSpec,
    ee_before: np.ndarray,
    result: dict[str, Any],
) -> dict[str, np.ndarray | float]:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    ee_before_arr = np.asarray(ee_before, dtype=np.float32).reshape(-1)[:3]
    clipped = np.clip(action_arr, -1.0, 1.0)
    commanded_raw = np.asarray(clipped[:3] * float(control_spec.action_step_xyz), dtype=np.float32)

    target_xyz = np.asarray(
        result.get("target_xyz", ee_before_arr + commanded_raw),
        dtype=np.float32,
    ).reshape(-1)[:3]
    commanded_effective = (target_xyz - ee_before_arr).astype(np.float32)

    ee_after = np.asarray(
        result.get("ee_position", ee_before_arr),
        dtype=np.float32,
    ).reshape(-1)[:3]
    realized = (ee_after - ee_before_arr).astype(np.float32)

    cmd_norm = float(np.linalg.norm(commanded_effective))
    real_norm = float(np.linalg.norm(realized))
    gain = float(real_norm / max(cmd_norm, 1e-8)) if cmd_norm > 1e-8 else float("nan")
    cosine = float("nan")
    if cmd_norm > 1e-8 and real_norm > 1e-8:
        cosine = float(np.dot(commanded_effective, realized) / (cmd_norm * real_norm))

    return {
        "commanded_xyz_delta_raw": commanded_raw,
        "commanded_xyz_delta_effective": commanded_effective,
        "realized_xyz_delta": realized,
        "target_xyz": target_xyz,
        "realized_vs_command_gain": gain,
        "realized_vs_command_cosine": cosine,
    }


def _resolve_llm_dim(vla: Any) -> int | None:
    for obj in _iter_model_candidates(vla):
        llm_dim = getattr(obj, "llm_dim", None)
        if llm_dim is not None:
            return int(llm_dim)

        cfg = getattr(obj, "config", None)
        if cfg is not None:
            text_cfg = getattr(cfg, "text_config", None)
            hidden = getattr(text_cfg, "hidden_size", None) if text_cfg is not None else None
            if hidden is not None:
                return int(hidden)
            hidden = getattr(cfg, "hidden_size", None)
            if hidden is not None:
                return int(hidden)

        language_model = getattr(obj, "language_model", None)
        language_cfg = getattr(language_model, "config", None) if language_model is not None else None
        hidden = getattr(language_cfg, "hidden_size", None) if language_cfg is not None else None
        if hidden is not None:
            return int(hidden)

    return None


def _core_model(vla: Any) -> Any:
    if hasattr(vla, "_prepare_input_for_action_prediction"):
        return vla
    base = getattr(vla, "base_model", None)
    if base is not None and hasattr(base, "model"):
        return base.model
    return vla


def _prepare_input_for_action_prediction_compat(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    action_dim: int,
    chunk_length: int,
) -> tuple[Any, Any]:
    if hasattr(model, "_prepare_input_for_action_prediction"):
        return model._prepare_input_for_action_prediction(input_ids, attention_mask)

    import torch
    from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, STOP_INDEX

    placeholder_count = int(action_dim) * int(chunk_length)
    arbitrary_action_token_idx = int(ACTION_TOKEN_BEGIN_IDX) + 1
    placeholder_action_token_ids = torch.full(
        (input_ids.shape[0], placeholder_count),
        fill_value=arbitrary_action_token_idx,
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

    stop_token_id = torch.full(
        (input_ids.shape[0], 1),
        fill_value=int(STOP_INDEX),
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

    mask_extension = torch.ones(
        (attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)
    return input_ids, attention_mask


def _prepare_labels_for_action_prediction_compat(
    model: Any,
    labels: Any,
    input_ids: Any,
    *,
    action_dim: int,
    chunk_length: int,
) -> Any:
    if hasattr(model, "_prepare_labels_for_action_prediction"):
        return model._prepare_labels_for_action_prediction(labels, input_ids)

    import torch
    from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, STOP_INDEX

    arbitrary_action_token_idx = int(ACTION_TOKEN_BEGIN_IDX) + 1
    labels_extension = torch.full(
        (labels.shape[0], input_ids.shape[-1] - labels.shape[-1]),
        fill_value=arbitrary_action_token_idx,
        device=labels.device,
        dtype=labels.dtype,
    )
    labels = torch.cat([labels, labels_extension], dim=-1)
    labels[:, -1] = int(STOP_INDEX)
    return labels


def _process_action_masks_compat(model: Any, labels: Any, *, action_dim: int, chunk_length: int) -> Any:
    if hasattr(model, "_process_action_masks"):
        return model._process_action_masks(labels)

    import torch
    from prismatic.vla.constants import IGNORE_INDEX

    batch_size, seq_len = labels.shape
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=labels.device)
    expected_total = int(action_dim) * int(chunk_length)

    for batch_idx in range(batch_size):
        positions = torch.nonzero(labels[batch_idx] != int(IGNORE_INDEX), as_tuple=False).squeeze(1)
        if positions.numel() > expected_total:
            positions = positions[:expected_total]
        if positions.numel() > 0:
            mask[batch_idx, positions] = True

    return mask


def _process_vision_features_compat(
    model: Any,
    pixel_values: Any,
    language_embeddings: Any,
    *,
    use_film: bool,
) -> Any:
    if hasattr(model, "_process_vision_features"):
        return model._process_vision_features(pixel_values, language_embeddings, use_film=use_film)

    if use_film:
        patch_features = model.vision_backbone(pixel_values, language_embeddings)
    else:
        patch_features = model.vision_backbone(pixel_values)
    return model.projector(patch_features)


def _build_multimodal_attention_compat(
    model: Any,
    input_embeddings: Any,
    projected_patch_embeddings: Any,
    attention_mask: Any,
) -> tuple[Any, Any]:
    if hasattr(model, "_build_multimodal_attention"):
        return model._build_multimodal_attention(input_embeddings, projected_patch_embeddings, attention_mask)

    import torch

    projected_patch_attention_mask = None
    if attention_mask is not None:
        projected_patch_attention_mask = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            fill_value=True,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

    multimodal_embeddings = torch.cat(
        [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]],
        dim=1,
    )

    multimodal_attention_mask = None
    if attention_mask is not None:
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]],
            dim=1,
        )

    return multimodal_embeddings, multimodal_attention_mask


def _predict_normalized_action_chunk(
    *,
    vla: Any,
    processor: Any,
    action_head: Any,
    obs: dict[str, np.ndarray],
    instruction: str,
    chunk_length: int,
    num_images_in_input: int,
    device: Any,
    pixel_dtype: Any,
) -> np.ndarray:
    import torch
    from PIL import Image

    with torch.inference_mode():
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        primary_image = Image.fromarray(obs["full_image"].astype(np.uint8)).convert("RGB")
        inputs = processor(prompt, primary_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        if int(num_images_in_input) > 1 and "wrist_image" in obs:
            wrist_image = Image.fromarray(obs["wrist_image"].astype(np.uint8)).convert("RGB")
            wrist_inputs = processor(prompt, wrist_image, return_tensors="pt")
            pixel_values = torch.cat([pixel_values, wrist_inputs["pixel_values"]], dim=1)

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        pixel_values = pixel_values.to(device=device, dtype=pixel_dtype)

        model = _core_model(vla)
        labels = torch.full_like(input_ids, fill_value=-100)
        prompt_len = input_ids.shape[1]
        action_dim = int(getattr(action_head, "action_dim", 5))

        input_ids_prep, attn_prep = _prepare_input_for_action_prediction_compat(
            model,
            input_ids,
            attention_mask,
            action_dim=action_dim,
            chunk_length=chunk_length,
        )
        labels = _prepare_labels_for_action_prediction_compat(
            model,
            labels,
            input_ids_prep,
            action_dim=action_dim,
            chunk_length=chunk_length,
        )

        input_embeddings = model.get_input_embeddings()(input_ids_prep)
        all_actions_mask = _process_action_masks_compat(
            model,
            labels,
            action_dim=action_dim,
            chunk_length=chunk_length,
        )

        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        projected_patch_embeddings = _process_vision_features_compat(
            model,
            pixel_values,
            language_embeddings,
            use_film=False,
        )

        all_actions_mask_expanded = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask_expanded

        multimodal_embeddings, multimodal_attention_mask = _build_multimodal_attention_compat(
            model,
            input_embeddings,
            projected_patch_embeddings,
            attn_prep,
        )

        language_model_output = model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_states = language_model_output.hidden_states[-1]
        patch_token_count = projected_patch_embeddings.shape[1]
        text_hidden_states = torch.cat(
            [last_hidden_states[:, :1, :], last_hidden_states[:, 1 + patch_token_count :, :]],
            dim=1,
        )
        total_action_tokens = int(action_dim) * int(chunk_length)
        action_hidden_states = text_hidden_states[:, prompt_len : prompt_len + total_action_tokens, :]

        pred_pre = action_head.predict_action(action_hidden_states)
        predicted_actions = torch.tanh(pred_pre)
        return predicted_actions[0].detach().to(dtype=torch.float32).cpu().numpy()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_project_config(args.config)
    policy_repo = config.resolve_path(config.policy.repo_path)
    if policy_repo is None:
        raise RuntimeError("Config is missing `policy.repo_path`.")

    run_dir = (
        ensure_directory(Path(args.run_dir).expanduser().resolve())
        if args.run_dir
        else ensure_directory(
            (config.resolve_path(config.project.output_root) or Path("runs"))
            / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )

    object_names = _default_objects(config, args.target_object, args.distractor)
    if not object_names:
        raise RuntimeError("No objects selected for the rollout scene.")
    scene_name = args.scene or config.simulation.preview_scene or "desk"
    instruction = args.instruction or _default_instruction(args.target_object, object_names)

    xml_path = build_scene_xml(
        config,
        output_dir=run_dir / "scene",
        scene_name=scene_name,
        object_names=object_names,
    )
    embodiment = MujocoEmbodiment(config=config, spec=config.embodiment)
    sim = embodiment.instantiate_controller(xml_path=xml_path, run_dir=run_dir)
    sim.initialize()
    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=10)
    setattr(sim, "language_instruction", instruction)

    control_spec = _control_spec_from_config(config, args.hold_steps)
    sim_dt = float(getattr(getattr(sim, "controller", None), "dt", 1.0 / 60.0))
    action_period = policy_action_period_s(sim_dt, control_spec.hold_steps)
    action_hz = policy_action_frequency_hz(sim_dt, control_spec.hold_steps)
    gripper_range = (
        float(getattr(sim, "gripper_min", config.embodiment.action_adapter.controller_limits["gripper"][0])),
        float(getattr(sim, "gripper_max", config.embodiment.action_adapter.controller_limits["gripper"][1])),
    )

    print(f"Run directory: {run_dir}")
    print(f"Scene: {scene_name}")
    print(f"Objects: {object_names}")
    print(f"Instruction: {instruction}")
    print(
        "Control contract: "
        f"dt={sim_dt:.6f}s, hold_steps={control_spec.hold_steps}, "
        f"sim_steps_per_action={control_spec.sim_steps_per_policy_action}, "
        f"policy_period={action_period:.6f}s (~{action_hz:.2f} Hz), "
        f"action_step_xyz={control_spec.action_step_xyz}, action_step_yaw={control_spec.action_step_yaw}"
    )

    (
        GenerateConfig,
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
        PeftModel,
        generate_config_note,
    ) = _load_openvla_modules(policy_repo)
    if generate_config_note:
        print(f"[info] {generate_config_note}")

    cfg = GenerateConfig(
        pretrained_checkpoint=args.base_ckpt or config.policy.base_checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=int(config.policy.num_images_in_input),
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=bool(args.center_crop),
        num_open_loop_steps=int(args.chunk_length or config.policy.action_codec.chunk_size),
        unnorm_key=None,
    )
    cfg.cdpr_action_head_path = str(Path(args.action_head_path).expanduser().resolve())

    vla_base = get_vla(cfg)
    vla_base.eval()
    if Path(args.adapter_path).expanduser().is_dir():
        vla = PeftModel.from_pretrained(vla_base, str(Path(args.adapter_path).expanduser().resolve()))
        vla.eval()
    else:
        raise RuntimeError(f"Adapter path is not a directory: {args.adapter_path}")

    cfg.num_images_in_input = _set_num_images_in_input(vla, int(cfg.num_images_in_input))
    llm_dim = _resolve_llm_dim(vla)
    if llm_dim is None:
        raise RuntimeError("Could not resolve llm_dim from the wrapped OpenVLA model.")

    processor = get_processor(cfg)
    param = next(vla.parameters())
    action_head = get_action_head(cfg, llm_dim=llm_dim).to(device=param.device, dtype=param.dtype)
    action_head.eval()

    initial_obs, proprio_dim = _make_observation(sim, instruction, gripper_range)
    proprio_projector = None
    if bool(getattr(cfg, "use_proprio", False)):
        proprio_projector = get_proprio_projector(cfg, llm_dim=llm_dim, proprio_dim=proprio_dim)
        proprio_projector = proprio_projector.to(device=param.device, dtype=param.dtype)
        proprio_projector.eval()

    current_chunk = np.asarray(
        _predict_normalized_action_chunk(
            vla=vla,
            processor=processor,
            action_head=action_head,
            obs=initial_obs,
            instruction=instruction,
            chunk_length=int(cfg.num_open_loop_steps),
            num_images_in_input=int(cfg.num_images_in_input),
            device=param.device,
            pixel_dtype=param.dtype,
        ),
        dtype=np.float32,
    )
    if current_chunk.ndim == 1:
        current_chunk = current_chunk.reshape(1, -1)
    if current_chunk.shape[1] < 5:
        raise RuntimeError(f"Expected at least 5 action dimensions, got chunk shape {current_chunk.shape}")
    current_chunk = current_chunk[:, :5]
    chunk_idx = 0
    log_every = max(1, int(args.log_every))

    for step in range(int(args.steps)):
        if chunk_idx >= len(current_chunk):
            obs, _ = _make_observation(sim, instruction, gripper_range)
            current_chunk = np.asarray(
                _predict_normalized_action_chunk(
                    vla=vla,
                    processor=processor,
                    action_head=action_head,
                    obs=obs,
                    instruction=instruction,
                    chunk_length=int(cfg.num_open_loop_steps),
                    num_images_in_input=int(cfg.num_images_in_input),
                    device=param.device,
                    pixel_dtype=param.dtype,
                ),
                dtype=np.float32,
            )
            if current_chunk.ndim == 1:
                current_chunk = current_chunk.reshape(1, -1)
            current_chunk = current_chunk[:, :5]
            chunk_idx = 0

        action = np.asarray(current_chunk[chunk_idx], dtype=np.float32).reshape(5)
        chunk_idx += 1
        max_abs = float(np.max(np.abs(action)))
        if max_abs > float(args.action_guard):
            print(f"[warn] Step {step}: policy action max abs {max_abs:.4f} > {args.action_guard}; clipping to [-1, 1].")

        ee_before = np.asarray(sim.get_end_effector_position(), dtype=np.float32).reshape(-1)[:3]
        result = apply_normalized_cdpr_action(
            sim,
            action,
            control_spec,
            capture_last_frame=True,
        )

        if step % log_every == 0:
            motion_diag = _motion_diagnostics_for_log(
                action=action,
                control_spec=control_spec,
                ee_before=ee_before,
                result=result,
            )
            ee = np.asarray(result.get("ee_position", sim.get_end_effector_position()), dtype=np.float32).reshape(-1)[:3]
            yaw = float(result.get("ee_yaw", sim.get_yaw() if hasattr(sim, "get_yaw") else 0.0))
            grip = float(result.get("gripper_opening", sim.get_gripper_opening() if hasattr(sim, "get_gripper_opening") else 0.0))
            print(
                f"step={step:03d} action={np.clip(action, -1.0, 1.0).round(4).tolist()} "
                f"ee=({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}) yaw={yaw:.4f} grip={grip:.4f}"
            )
            cmd_raw = np.asarray(motion_diag["commanded_xyz_delta_raw"], dtype=np.float32)
            cmd_eff = np.asarray(motion_diag["commanded_xyz_delta_effective"], dtype=np.float32)
            realized = np.asarray(motion_diag["realized_xyz_delta"], dtype=np.float32)
            target_xyz = np.asarray(motion_diag["target_xyz"], dtype=np.float32)
            gain = float(motion_diag["realized_vs_command_gain"])
            cosine = float(motion_diag["realized_vs_command_cosine"])
            print(
                "  motion "
                f"cmd_raw=({cmd_raw[0]:+.4f}, {cmd_raw[1]:+.4f}, {cmd_raw[2]:+.4f}) "
                f"cmd_eff=({cmd_eff[0]:+.4f}, {cmd_eff[1]:+.4f}, {cmd_eff[2]:+.4f}) "
                f"real=({realized[0]:+.4f}, {realized[1]:+.4f}, {realized[2]:+.4f}) "
                f"target=({target_xyz[0]:+.4f}, {target_xyz[1]:+.4f}, {target_xyz[2]:+.4f}) "
                f"gain={gain:.4f} cosine={cosine:.4f}"
            )

    if not args.no_save and hasattr(sim, "save_trajectory_results"):
        sim.save_trajectory_results(str(run_dir), args.run_name)
    if hasattr(sim, "cleanup"):
        sim.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
