from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency surface
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency surface
    torch = None
    nn = None


def prepare_prompt(text: str) -> str:
    return f"In: What action should the robot take to {str(text).lower()}?\nOut:"


def iter_model_candidates(model: Any) -> list[Any]:
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


def resolve_vision_backbone(vla: Any) -> Any:
    for obj in iter_model_candidates(vla):
        backbone = getattr(obj, "vision_backbone", None)
        if backbone is not None:
            return backbone
    return None


def resolve_llm_dim(vla: Any) -> int | None:
    for obj in iter_model_candidates(vla):
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


def set_num_images_in_input(vla: Any, num_images: int) -> int:
    requested = int(num_images)

    for obj in iter_model_candidates(vla):
        if hasattr(obj, "set_num_images_in_input"):
            obj.set_num_images_in_input(requested)
            return requested
        if hasattr(obj, "num_images_in_input"):
            setattr(obj, "num_images_in_input", requested)
            return requested

    backbone = resolve_vision_backbone(vla)
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

    return 1 if requested != 1 else requested


def load_openvla_runtime(policy_repo: str | Path) -> tuple[Any, Any, Any, Any, Any, Any]:
    repo = Path(policy_repo).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from experiments.robot.openvla_utils import (  # type: ignore[import-not-found]
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
    )
    from peft import PeftModel  # type: ignore[import-not-found]

    return get_action_head, get_processor, get_proprio_projector, get_vla, PeftModel, repo


def load_generate_config() -> tuple[type[Any], str | None]:
    try:
        module = importlib.import_module("experiments.robot.libero.run_libero_eval")
        return module.GenerateConfig, None
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        note = f"LIBERO eval dependencies were not importable (`{missing}`); using a minimal fallback config surface."

        @dataclass
        class _FallbackGenerateConfig:
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

        return _FallbackGenerateConfig, note


if torch is not None:  # pragma: no branch - runtime feature gate
    import torch.nn.functional as F
    from PIL import Image

    def _core_model(vla: Any) -> Any:
        if hasattr(vla, "_prepare_input_for_action_prediction"):
            return vla
        base = getattr(vla, "base_model", None)
        if base is not None and hasattr(base, "model"):
            return base.model
        return vla


    def _to_pil_rgb(image: np.ndarray) -> Image.Image:
        return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


    def prepare_openvla_batch(
        *,
        processor: Any,
        observations: Sequence[dict[str, np.ndarray]],
        instructions: Sequence[str],
        num_images_in_input: int,
        device: Any,
        pixel_dtype: Any,
    ) -> tuple[Any, Any, Any]:
        prompts = [prepare_prompt(text) for text in instructions]
        primary_images = [_to_pil_rgb(obs["full_image"]) for obs in observations]
        inputs = processor(prompts, primary_images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"]

        if int(num_images_in_input) > 1:
            wrist_images = [_to_pil_rgb(obs["wrist_image"]) for obs in observations]
            wrist_inputs = processor(prompts, wrist_images, return_tensors="pt", padding=True)
            pixel_values = torch.cat([pixel_values, wrist_inputs["pixel_values"]], dim=1)

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        pixel_values = pixel_values.to(device=device, dtype=pixel_dtype)
        return input_ids, attention_mask, pixel_values


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


    def extract_action_hidden_states(
        *,
        vla: Any,
        processor: Any,
        observations: Sequence[dict[str, np.ndarray]],
        instructions: Sequence[str],
        chunk_length: int,
        action_dim: int,
        num_images_in_input: int,
        device: Any,
        pixel_dtype: Any,
    ) -> torch.Tensor:
        input_ids, attention_mask, pixel_values = prepare_openvla_batch(
            processor=processor,
            observations=observations,
            instructions=instructions,
            num_images_in_input=num_images_in_input,
            device=device,
            pixel_dtype=pixel_dtype,
        )

        model = _core_model(vla)
        labels = torch.full_like(input_ids, fill_value=-100)
        prompt_len = input_ids.shape[1]

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
            input_embeddings.shape[0],
            -1,
            input_embeddings.shape[2],
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
        return text_hidden_states[:, prompt_len : prompt_len + total_action_tokens, :]


    class ChunkCritic(nn.Module):
        def __init__(self, llm_dim: int, chunk_length: int, action_dim: int, hidden_dim: int):
            super().__init__()
            feature_dim = int(llm_dim) + int(chunk_length) * int(action_dim)
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, pooled_features: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
            flat_action = action_chunk.reshape(action_chunk.shape[0], -1)
            return self.net(torch.cat([pooled_features, flat_action], dim=-1))


    class OpenVLAActorCriticStack(nn.Module):
        def __init__(
            self,
            *,
            vla: Any,
            processor: Any,
            action_head: Any,
            chunk_length: int,
            action_dim: int,
            num_images_in_input: int,
            llm_dim: int | None = None,
            critic_hidden_dim: int = 1024,
            freeze_vla_backbone: bool = True,
        ) -> None:
            super().__init__()
            self.vla = vla
            self.processor = processor
            self.action_head = action_head
            self.chunk_length = int(chunk_length)
            self.action_dim = int(action_dim)
            self.num_images_in_input = int(num_images_in_input)
            self.llm_dim = int(resolve_llm_dim(vla) if llm_dim is None else llm_dim)
            self.critic1 = ChunkCritic(self.llm_dim, self.chunk_length, self.action_dim, critic_hidden_dim)
            self.critic2 = ChunkCritic(self.llm_dim, self.chunk_length, self.action_dim, critic_hidden_dim)

            if freeze_vla_backbone:
                self.freeze_backbone()

        def freeze_backbone(self) -> None:
            for module in (self.vla,):
                for param in getattr(module, "parameters", lambda: [])():
                    param.requires_grad = False

        def encode(
            self,
            *,
            observations: Sequence[dict[str, np.ndarray]],
            instructions: Sequence[str],
            device: Any,
            pixel_dtype: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = extract_action_hidden_states(
                vla=self.vla,
                processor=self.processor,
                observations=observations,
                instructions=instructions,
                chunk_length=self.chunk_length,
                action_dim=self.action_dim,
                num_images_in_input=self.num_images_in_input,
                device=device,
                pixel_dtype=pixel_dtype,
            )
            pooled = hidden.mean(dim=1)
            return hidden, pooled

        def actor(
            self,
            *,
            observations: Sequence[dict[str, np.ndarray]],
            instructions: Sequence[str],
            device: Any,
            pixel_dtype: Any,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden, pooled = self.encode(
                observations=observations,
                instructions=instructions,
                device=device,
                pixel_dtype=pixel_dtype,
            )
            action_pre = self.action_head.predict_action(hidden)
            action = torch.tanh(action_pre)
            return action, action_pre, pooled

        def critics(
            self,
            *,
            observations: Sequence[dict[str, np.ndarray]],
            instructions: Sequence[str],
            actions: torch.Tensor | None,
            device: Any,
            pixel_dtype: Any,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden, pooled = self.encode(
                observations=observations,
                instructions=instructions,
                device=device,
                pixel_dtype=pixel_dtype,
            )
            if actions is None:
                actions = torch.tanh(self.action_head.predict_action(hidden))
            q1 = self.critic1(pooled, actions)
            q2 = self.critic2(pooled, actions)
            return q1, q2, pooled

        def act(
            self,
            *,
            observation: dict[str, np.ndarray],
            instruction: str,
            device: Any,
            pixel_dtype: Any,
        ) -> np.ndarray:
            with torch.inference_mode():
                action, _, _ = self.actor(
                    observations=[observation],
                    instructions=[instruction],
                    device=device,
                    pixel_dtype=pixel_dtype,
                )
            return action[0].detach().to(dtype=torch.float32).cpu().numpy()

else:
    class OpenVLAActorCriticStack:  # pragma: no cover - runtime dependency gate
        def __init__(self, *args, **kwargs):
            raise ImportError("OpenVLAActorCriticStack requires torch, PIL, and the OpenVLA runtime dependencies.")
