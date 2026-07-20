from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - optional runtime dependency
    import torch
    import torch.nn.functional as F
    if not hasattr(torch, "float32") or not hasattr(torch, "device"):
        raise ImportError("Incomplete PyTorch compatibility stub")
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    F = None

from rl_vla_bootstrapping.policy.octo_cdpr_adapter import (
    CDPRStateLayout,
    cdpr_proprio_from_observation,
    flatten_cdpr_observation,
)


DEFAULT_SMOLVLA_CHECKPOINT = "lerobot/smolvla_base"
SMOLVLA_STATE_KEY = "observation.state"
SMOLVLA_TASK_KEY = "task"
SMOLVLA_LANGUAGE_TOKEN_KEY = "observation.language.tokens"
SMOLVLA_LANGUAGE_MASK_KEY = "observation.language.attention_mask"
DEFAULT_SMOLVLA_IMAGE_KEYS: tuple[str, ...] = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
CDPR_ACTION_KEYS: tuple[str, ...] = ("x", "y", "z", "yaw", "gripper")


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected an RGB-like image with shape HxWxC, got {arr.shape}.")
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _nearest_resize_hwc_uint8(image: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    arr = _as_uint8_rgb(image)
    if isinstance(size, int):
        target_h, target_w = int(size), int(size)
    else:
        target_h, target_w = int(size[0]), int(size[1])
    if arr.shape[:2] == (target_h, target_w):
        return arr
    y_idx = np.linspace(0, arr.shape[0] - 1, target_h).round().astype(np.int64)
    x_idx = np.linspace(0, arr.shape[1] - 1, target_w).round().astype(np.int64)
    return np.ascontiguousarray(arr[y_idx][:, x_idx])


def _normalize_instruction(text: str) -> str:
    stripped = " ".join(str(text).strip().split())
    return stripped if stripped.endswith("\n") else stripped + "\n"


def _resolve_torch_device(device: str | Any) -> Any:
    if torch is None:
        raise RuntimeError("SmolVLA runtime requires PyTorch.")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is None and torch.cuda.is_available():
        torch_device = torch.device(f"cuda:{int(torch.cuda.current_device())}")
    return torch_device


def cdpr_state_from_observation(
    obs: dict[str, np.ndarray],
    info: dict[str, Any] | None = None,
    *,
    state_dim: int = 6,
) -> np.ndarray:
    """Build the compact proprio/state vector expected by SmolVLA."""
    info = dict(info or {})
    base = cdpr_proprio_from_observation(obs, info).astype(np.float32, copy=False).reshape(-1)
    target = np.asarray(
        obs.get("target_object_position", info.get("target_object_position", (0.0, 0.0, 0.0))),
        dtype=np.float32,
    ).reshape(-1)
    ee = np.asarray(obs.get("ee_position", base[:3]), dtype=np.float32).reshape(-1)
    rel = np.zeros(3, dtype=np.float32)
    if target.size >= 3 and ee.size >= 3:
        rel = target[:3] - ee[:3]
    extras = np.asarray(
        [
            float(np.linalg.norm(rel[:2])),
            float(rel[2]),
            float(info.get("step", 0.0)),
        ],
        dtype=np.float32,
    )
    merged = np.concatenate([base, extras], axis=0).astype(np.float32, copy=False)
    width = max(1, int(state_dim))
    out = np.zeros(width, dtype=np.float32)
    out[: min(width, merged.size)] = merged[: min(width, merged.size)]
    return out


@dataclass(frozen=True)
class SmolVLAObservationSpec:
    image_size: int = 256
    state_dim: int = 6
    image_feature_keys: tuple[str, ...] = DEFAULT_SMOLVLA_IMAGE_KEYS
    state_key: str = SMOLVLA_STATE_KEY
    task_key: str = SMOLVLA_TASK_KEY
    include_wrist: bool = True
    include_aux_camera: bool = True
    # When no true auxiliary camera exists, feed camera slot three as a black
    # frame with a zero padding mask (LeRobot's native empty-camera encoding)
    # instead of aliasing the wrist view.
    mask_empty_aux_camera: bool = False


def _image_sources_for_keys(
    *,
    primary_image: np.ndarray,
    wrist_image: np.ndarray | None,
    aux_image: np.ndarray | None,
    spec: SmolVLAObservationSpec,
) -> list[np.ndarray]:
    fallback_wrist = wrist_image if wrist_image is not None and spec.include_wrist else primary_image
    if aux_image is not None and spec.include_aux_camera:
        fallback_aux = aux_image
    elif spec.mask_empty_aux_camera:
        fallback_aux = np.zeros_like(_as_uint8_rgb(primary_image))
    else:
        fallback_aux = fallback_wrist
    sources = [primary_image, fallback_wrist, fallback_aux]
    while len(sources) < len(spec.image_feature_keys):
        sources.append(primary_image)
    return sources[: len(spec.image_feature_keys)]


def _numpy_image_batch(images: Sequence[np.ndarray], image_size: int) -> np.ndarray:
    frames = [_nearest_resize_hwc_uint8(image, image_size) for image in images]
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return np.transpose(arr, (0, 3, 1, 2)).astype(np.float32, copy=False)


def _torch_image_batch(
    images: Sequence[np.ndarray],
    *,
    image_size: int,
    device: Any,
    dtype: Any | None,
    non_blocking: bool,
) -> Any:
    if torch is None or F is None:
        raise RuntimeError("Torch is required for GPU SmolVLA image preprocessing.")
    frames = [_as_uint8_rgb(image) for image in images]
    arr = np.stack(frames, axis=0)
    tensor = torch.as_tensor(arr, device="cpu").permute(0, 3, 1, 2).contiguous()
    tensor = tensor.to(device=device, non_blocking=bool(non_blocking)).to(dtype=torch.float32).div_(255.0)
    if tensor.shape[-2:] != (int(image_size), int(image_size)):
        tensor = F.interpolate(
            tensor,
            size=(int(image_size), int(image_size)),
            mode="bilinear",
            align_corners=False,
        )
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def adapt_cdpr_observations_to_smolvla_batch(
    *,
    primary_images: Sequence[np.ndarray],
    wrist_images: Sequence[np.ndarray | None] | None = None,
    aux_images: Sequence[np.ndarray | None] | None = None,
    observations: Sequence[dict[str, np.ndarray]] | None = None,
    infos: Sequence[dict[str, Any] | None] | None = None,
    instructions: Sequence[str] | None = None,
    spec: SmolVLAObservationSpec | None = None,
    device: Any | None = None,
    dtype: Any | None = None,
    non_blocking: bool = True,
) -> dict[str, Any]:
    spec = spec or SmolVLAObservationSpec()
    batch_size = len(primary_images)
    if batch_size == 0:
        raise ValueError("SmolVLA batch needs at least one CDPR observation.")
    wrist_images = wrist_images or [None] * batch_size
    aux_images = aux_images or [None] * batch_size
    observations = observations or [{} for _ in range(batch_size)]
    infos = infos or [None] * batch_size
    instructions = instructions or ["move left"] * batch_size
    if not (
        len(wrist_images)
        == len(aux_images)
        == len(observations)
        == len(infos)
        == len(instructions)
        == batch_size
    ):
        raise ValueError("All SmolVLA CDPR batch inputs must have the same length.")

    image_columns: dict[str, list[np.ndarray]] = {key: [] for key in spec.image_feature_keys}
    for idx, primary in enumerate(primary_images):
        sources = _image_sources_for_keys(
            primary_image=primary,
            wrist_image=wrist_images[idx],
            aux_image=aux_images[idx],
            spec=spec,
        )
        for key, source in zip(spec.image_feature_keys, sources):
            image_columns[key].append(source)

    batch: dict[str, Any] = {
        spec.task_key: [_normalize_instruction(text) for text in instructions],
    }
    aux_padding_key = None
    aux_padding_np = None
    if spec.mask_empty_aux_camera and len(spec.image_feature_keys) >= 3:
        aux_padding_key = f"{spec.image_feature_keys[2]}_padding_mask"
        aux_padding_np = np.array(
            [
                image is not None and spec.include_aux_camera
                for image in aux_images
            ],
            dtype=bool,
        )

    states = [
        cdpr_state_from_observation(obs, info, state_dim=spec.state_dim)
        for obs, info in zip(observations, infos)
    ]
    state_np = np.stack(states, axis=0).astype(np.float32, copy=False)
    if device is not None and torch is not None:
        batch[spec.state_key] = torch.as_tensor(state_np, device="cpu").to(
            device=device,
            dtype=dtype or torch.float32,
            non_blocking=bool(non_blocking),
        )
        for key, images in image_columns.items():
            batch[key] = _torch_image_batch(
                images,
                image_size=int(spec.image_size),
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
        if aux_padding_key is not None:
            batch[aux_padding_key] = torch.as_tensor(
                aux_padding_np, device="cpu"
            ).to(device=device, non_blocking=bool(non_blocking))
    else:
        batch[spec.state_key] = state_np
        for key, images in image_columns.items():
            batch[key] = _numpy_image_batch(images, int(spec.image_size))
        if aux_padding_key is not None:
            batch[aux_padding_key] = aux_padding_np
    return batch


def _normalize_gpu_image_batch(
    images: Any,
    *,
    image_size: int,
    device: Any,
) -> Any:
    """Normalize a GPU RGB batch without NumPy/PIL or host synchronization."""

    if torch is None or F is None:
        raise RuntimeError("Torch is required for GPU-resident SmolVLA inputs.")
    if not isinstance(images, torch.Tensor):
        raise TypeError(
            "MJWarp camera inputs must already be torch.Tensor instances on GPU."
        )
    tensor = images
    if tensor.ndim != 4:
        raise ValueError(f"Expected a batched image tensor, got {tuple(tensor.shape)}.")
    if tensor.shape[1] in {3, 4}:
        tensor = tensor[:, :3]
    elif tensor.shape[-1] in {3, 4}:
        tensor = tensor[..., :3].permute(0, 3, 1, 2)
    else:
        raise ValueError(
            f"Expected BCHW or BHWC RGB images, got {tuple(tensor.shape)}."
        )
    if tensor.device != device:
        raise RuntimeError(
            f"Camera tensor is on {tensor.device}, expected rank-local {device}."
        )
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
    elif tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    else:
        raise TypeError(f"Unsupported camera dtype {tensor.dtype}.")
    tensor = tensor.clamp_(0.0, 1.0)
    target = (int(image_size), int(image_size))
    if tuple(tensor.shape[-2:]) != target:
        tensor = F.interpolate(
            tensor,
            size=target,
            mode="bilinear",
            align_corners=False,
        )
    return tensor


def adapt_cdpr_tensors_to_smolvla_batch(
    *,
    primary_images: Any,
    wrist_images: Any,
    states: Any,
    instructions: Sequence[str],
    aux_images: Any | None = None,
    spec: SmolVLAObservationSpec | None = None,
    device: Any | None = None,
) -> dict[str, Any]:
    """Build the active three-camera SmolVLA contract entirely on GPU.

    Without a true auxiliary camera, slot three either aliases the wrist input
    (the established CPU path) or, with ``spec.mask_empty_aux_camera``, becomes
    a black frame plus a zero ``*_padding_mask`` so LeRobot masks its tokens.
    """

    if torch is None:
        raise RuntimeError("Torch is required for GPU-resident SmolVLA inputs.")
    spec = spec or SmolVLAObservationSpec()
    resolved_device = _resolve_torch_device(
        device if device is not None else primary_images.device
    )
    primary = _normalize_gpu_image_batch(
        primary_images,
        image_size=int(spec.image_size),
        device=resolved_device,
    )
    wrist = _normalize_gpu_image_batch(
        wrist_images,
        image_size=int(spec.image_size),
        device=resolved_device,
    )
    aux_masked_empty = False
    if aux_images is not None:
        auxiliary = _normalize_gpu_image_batch(
            aux_images,
            image_size=int(spec.image_size),
            device=resolved_device,
        )
    elif spec.mask_empty_aux_camera:
        auxiliary = torch.zeros_like(primary)
        aux_masked_empty = True
    else:
        auxiliary = wrist
    state_tensor = torch.as_tensor(
        states, dtype=torch.float32, device=resolved_device
    )
    if state_tensor.ndim != 2:
        raise ValueError(
            f"SmolVLA state tensor must have shape [B, D], got {tuple(state_tensor.shape)}."
        )
    batch_size = int(primary.shape[0])
    if (
        int(wrist.shape[0]) != batch_size
        or int(auxiliary.shape[0]) != batch_size
        or int(state_tensor.shape[0]) != batch_size
        or len(instructions) != batch_size
    ):
        raise ValueError("All GPU SmolVLA inputs must have the same batch size.")
    if int(state_tensor.shape[1]) != int(spec.state_dim):
        if int(state_tensor.shape[1]) > int(spec.state_dim):
            state_tensor = state_tensor[:, : int(spec.state_dim)]
        else:
            state_tensor = F.pad(
                state_tensor, (0, int(spec.state_dim) - int(state_tensor.shape[1]))
            )

    sources = (primary, wrist, auxiliary)
    batch: dict[str, Any] = {
        spec.task_key: [_normalize_instruction(text) for text in instructions],
        spec.state_key: state_tensor,
    }
    for index, key in enumerate(spec.image_feature_keys):
        batch[key] = sources[index] if index < len(sources) else primary
    if aux_masked_empty and len(spec.image_feature_keys) >= 3:
        batch[f"{spec.image_feature_keys[2]}_padding_mask"] = torch.zeros(
            (batch_size,), dtype=torch.bool, device=resolved_device
        )
    return batch


@dataclass(frozen=True)
class SmolVLAActionAdapterSpec:
    action_dim: int = 5
    chunk_size: int = 8
    action_indices: tuple[int, ...] | None = None
    normalization: str = "tanh"


def _resolve_action_indices(source_dim: int, requested: tuple[int, ...] | None) -> tuple[int, ...]:
    if requested:
        if len(requested) != 5:
            raise ValueError("SmolVLA CDPR action_indices must contain exactly five entries.")
        resolved: list[int] = []
        for raw_idx in requested:
            idx = int(raw_idx)
            if idx < 0:
                idx = int(source_dim) + idx
            if idx < 0 or idx >= int(source_dim):
                raise ValueError(f"Action index {raw_idx} is out of range for source dim {source_dim}.")
            resolved.append(idx)
        return tuple(resolved)
    if source_dim >= 6:
        return (0, 1, 2, 3, source_dim - 1)
    if source_dim >= 5:
        return (0, 1, 2, 3, 4)
    return tuple(range(source_dim))


def adapt_smolvla_actions_to_cdpr(
    actions: np.ndarray,
    *,
    spec: SmolVLAActionAdapterSpec | None = None,
) -> np.ndarray:
    spec = spec or SmolVLAActionAdapterSpec()
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected SmolVLA actions with shape [H, D] or [B, H, D], got {arr.shape}.")

    source_dim = int(arr.shape[-1])
    indices = _resolve_action_indices(source_dim, spec.action_indices)
    out = np.zeros((arr.shape[0], int(spec.action_dim)), dtype=np.float32)
    for dst_idx, src_idx in enumerate(indices[: int(spec.action_dim)]):
        out[:, dst_idx] = arr[:, src_idx]

    mode = str(spec.normalization or "clip").lower()
    if mode == "tanh":
        out = np.tanh(out)
    elif mode == "clip":
        out = np.clip(out, -1.0, 1.0)
    elif mode in {"none", "identity"}:
        out = out.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported SmolVLA action normalization mode: {spec.normalization!r}")

    target_horizon = max(1, int(spec.chunk_size))
    if out.shape[0] < target_horizon:
        pad = np.repeat(out[-1:], target_horizon - out.shape[0], axis=0)
        out = np.concatenate([out, pad], axis=0)
    elif out.shape[0] > target_horizon:
        out = out[:target_horizon]
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def adapt_smolvla_action_tensors_to_cdpr(
    actions: Any,
    *,
    spec: SmolVLAActionAdapterSpec | None = None,
) -> Any:
    """Vectorized GPU action adapter preserving ``[B, H, 5]``."""

    if torch is None:
        raise RuntimeError("Torch is required for GPU-resident SmolVLA actions.")
    spec = spec or SmolVLAActionAdapterSpec()
    if not isinstance(actions, torch.Tensor):
        raise TypeError("SmolVLA GPU actions must be a torch.Tensor.")
    tensor = actions
    if tensor.ndim == 2:
        tensor = tensor[:, None, :]
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected SmolVLA actions with shape [B, H, D], got {tuple(tensor.shape)}."
        )
    source_dim = int(tensor.shape[-1])
    indices = _resolve_action_indices(source_dim, spec.action_indices)
    index_tensor = torch.tensor(
        indices, dtype=torch.int64, device=tensor.device
    )
    selected = tensor.index_select(-1, index_tensor).to(dtype=torch.float32)
    action_dim = int(spec.action_dim)
    if selected.shape[-1] < action_dim:
        selected = F.pad(selected, (0, action_dim - int(selected.shape[-1])))
    else:
        selected = selected[..., :action_dim]

    mode = str(spec.normalization or "clip").lower()
    if mode == "tanh":
        selected = torch.tanh(selected)
    elif mode == "clip":
        selected = selected.clamp(-1.0, 1.0)
    elif mode not in {"none", "identity"}:
        raise ValueError(
            f"Unsupported SmolVLA action normalization mode: {spec.normalization!r}"
        )

    horizon = max(1, int(spec.chunk_size))
    current_horizon = int(selected.shape[1])
    if current_horizon < horizon:
        selected = torch.cat(
            (
                selected,
                selected[:, -1:].expand(-1, horizon - current_horizon, -1),
            ),
            dim=1,
        )
    elif current_horizon > horizon:
        selected = selected[:, :horizon]
    return selected.clamp(-1.0, 1.0)


def _torch_dtype_from_name(name: str | None, *, device: Any) -> Any | None:
    if torch is None:
        return None
    value = str(name or "auto").lower()
    if value in {"none", "fp32", "float32"}:
        return torch.float32
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value == "auto":
        if str(device).startswith("cuda") and torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability(device)
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    raise ValueError(f"Unsupported SmolVLA mixed precision setting: {name!r}")


class SmolVLARuntime:
    def __init__(
        self,
        *,
        policy: Any,
        checkpoint: str,
        device: Any,
        dtype: Any | None,
        obs_spec: SmolVLAObservationSpec,
        action_spec: SmolVLAActionAdapterSpec,
        tokenizer: Any | None = None,
        model_image_size: int | None = None,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        original_sample_actions: Any | None = None,
    ) -> None:
        self.policy = policy
        self.checkpoint = str(checkpoint)
        self.device = device
        self.dtype = dtype
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.tokenizer = tokenizer or self._resolve_tokenizer(policy)
        self._token_cache: dict[tuple[str, ...], tuple[Any, Any]] = {}
        self.model_image_size = (
            None if model_image_size is None else int(model_image_size)
        )
        self.compile_model = bool(compile_model)
        self.compile_mode = str(compile_mode)
        self._original_sample_actions = original_sample_actions
        self._compile_fallback_used = False

    @classmethod
    def load(
        cls,
        checkpoint: str = DEFAULT_SMOLVLA_CHECKPOINT,
        *,
        device: str | Any = "cuda",
        mixed_precision: str = "auto",
        image_size: int = 256,
        state_dim: int | None = None,
        image_feature_keys: Sequence[str] | None = None,
        include_wrist: bool = True,
        include_aux_camera: bool = True,
        mask_empty_aux_camera: bool = False,
        chunk_size: int = 8,
        action_dim: int = 5,
        action_indices: tuple[int, ...] | None = None,
        action_normalization: str = "tanh",
        model_image_size: int | None = None,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
    ) -> "SmolVLARuntime":
        if torch is None:
            raise RuntimeError("SmolVLA runtime requires PyTorch plus `lerobot[smolvla]`.")
        try:
            module = importlib.import_module("lerobot.policies.smolvla.modeling_smolvla")
        except ImportError as exc:
            raise RuntimeError(
                "Could not import LeRobot SmolVLA. Install the remote environment with "
                '`pip install "lerobot[smolvla]"` before running SmolVLA CDPR.'
            ) from exc

        torch_device = _resolve_torch_device(device)
        if torch_device.type == "cuda":
            torch.cuda.set_device(torch_device)
        dtype = _torch_dtype_from_name(mixed_precision, device=torch_device)
        policy_cls = module.SmolVLAPolicy
        policy = policy_cls.from_pretrained(str(checkpoint)).to(torch_device)
        policy.eval()
        parameters = getattr(policy, "parameters", None)
        if callable(parameters):
            for param in parameters():
                param.requires_grad_(False)
        if hasattr(policy, "reset"):
            policy.reset()

        cfg = getattr(policy, "config", None)
        resolved_model_image_size = (
            None
            if model_image_size is None or int(model_image_size) <= 0
            else int(model_image_size)
        )
        if resolved_model_image_size is not None and hasattr(
            cfg, "resize_imgs_with_padding"
        ):
            # LeRobot otherwise upsamples the already-preprocessed 256px inputs
            # back to the checkpoint default (typically 512px). Keeping the
            # model-side resize explicit avoids 4x vision-token work hidden
            # behind a seemingly smaller runtime image_size.
            cfg.resize_imgs_with_padding = (
                resolved_model_image_size,
                resolved_model_image_size,
            )

        original_sample_actions = None
        compile_enabled = False
        if bool(compile_model) and torch_device.type == "cuda":
            model = getattr(policy, "model", None)
            sample_actions = getattr(model, "sample_actions", None)
            if model is None or not callable(sample_actions):
                warnings.warn(
                    "SmolVLA torch.compile was requested, but this LeRobot policy "
                    "does not expose model.sample_actions; continuing eagerly.",
                    RuntimeWarning,
                )
            elif not hasattr(torch, "compile"):
                warnings.warn(
                    "SmolVLA torch.compile was requested, but this PyTorch build "
                    "does not provide torch.compile; continuing eagerly.",
                    RuntimeWarning,
                )
            else:
                torch.set_float32_matmul_precision("high")
                original_sample_actions = sample_actions
                model.sample_actions = torch.compile(
                    sample_actions,
                    mode=str(compile_mode),
                )
                compile_enabled = True
        cfg_state_dim = _feature_dim(getattr(cfg, "robot_state_feature", None))
        cfg_action_dim = _feature_dim(getattr(cfg, "action_feature", None))
        cfg_image_keys = tuple(getattr(getattr(cfg, "image_features", {}), "keys", lambda: [])())
        obs_spec = SmolVLAObservationSpec(
            image_size=int(image_size),
            state_dim=int(state_dim or cfg_state_dim or 6),
            image_feature_keys=tuple(image_feature_keys or cfg_image_keys or DEFAULT_SMOLVLA_IMAGE_KEYS),
            include_wrist=bool(include_wrist),
            include_aux_camera=bool(include_aux_camera),
            mask_empty_aux_camera=bool(mask_empty_aux_camera),
        )
        action_spec = SmolVLAActionAdapterSpec(
            action_dim=int(action_dim),
            chunk_size=int(chunk_size),
            action_indices=action_indices,
            normalization=str(action_normalization),
        )
        return cls(
            policy=policy,
            checkpoint=str(checkpoint),
            device=torch_device,
            dtype=dtype,
            obs_spec=obs_spec,
            action_spec=action_spec,
            model_image_size=resolved_model_image_size,
            compile_model=compile_enabled,
            compile_mode=str(compile_mode),
            original_sample_actions=original_sample_actions,
        )

    @staticmethod
    def _resolve_tokenizer(policy: Any) -> Any | None:
        for path in (
            ("model", "vlm_with_expert", "processor", "tokenizer"),
            ("model", "processor", "tokenizer"),
            ("processor", "tokenizer"),
        ):
            obj = policy
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        return None

    def _tokenize(self, instructions: Sequence[str]) -> tuple[Any, Any]:
        if torch is None:
            raise RuntimeError("Torch is required to tokenize SmolVLA language inputs.")
        normalized = tuple(_normalize_instruction(text) for text in instructions)
        cached = self._token_cache.get(normalized)
        if cached is not None:
            return cached
        if self.tokenizer is None:
            raise RuntimeError("Could not resolve the SmolVLA tokenizer from the loaded policy.")
        max_length = int(getattr(getattr(self.policy, "config", None), "tokenizer_max_length", 48))
        tokens = self.tokenizer(
            list(normalized),
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device, non_blocking=True)
        attention_mask = tokens["attention_mask"].to(
            self.device,
            dtype=torch.bool,
            non_blocking=True,
        )
        self._token_cache[normalized] = (input_ids, attention_mask)
        return input_ids, attention_mask

    def _prepare_batch(
        self,
        *,
        primary_images: Sequence[np.ndarray],
        wrist_images: Sequence[np.ndarray | None] | None,
        aux_images: Sequence[np.ndarray | None] | None,
        observations: Sequence[dict[str, np.ndarray]],
        infos: Sequence[dict[str, Any] | None],
        instructions: Sequence[str],
    ) -> dict[str, Any]:
        batch = adapt_cdpr_observations_to_smolvla_batch(
            primary_images=primary_images,
            wrist_images=wrist_images,
            aux_images=aux_images,
            observations=observations,
            infos=infos,
            instructions=instructions,
            spec=self.obs_spec,
            device=self.device,
            dtype=torch.float32 if self.dtype is torch.float32 else None,
            non_blocking=True,
        )
        input_ids, attention_mask = self._tokenize(instructions)
        batch[SMOLVLA_LANGUAGE_TOKEN_KEY] = input_ids
        batch[SMOLVLA_LANGUAGE_MASK_KEY] = attention_mask
        return batch

    def _predict_actions_tensor(self, batch: dict[str, Any]) -> Any:
        if torch is None:
            raise RuntimeError("Torch is required for SmolVLA inference.")
        autocast_enabled = self.device.type == "cuda" and self.dtype in {
            torch.bfloat16,
            torch.float16,
        }

        def predict() -> Any:
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=bool(autocast_enabled),
                ):
                    if hasattr(self.policy, "predict_action_chunk"):
                        return self.policy.predict_action_chunk(batch)
                    return self.policy.select_action(batch)

        try:
            actions = predict()
        except Exception as compile_exc:
            # Compilation is an optimization, not a correctness requirement.
            # Backend/driver incompatibilities generally surface on the first
            # compiled invocation rather than at torch.compile(...).
            model = getattr(self.policy, "model", None)
            if (
                not self.compile_model
                or self._original_sample_actions is None
                or model is None
                or self._compile_fallback_used
            ):
                raise
            model.sample_actions = self._original_sample_actions
            self.compile_model = False
            self._compile_fallback_used = True
            warnings.warn(
                "Compiled SmolVLA inference failed; restored eager inference. "
                f"First compile error: {compile_exc}",
                RuntimeWarning,
            )
            actions = predict()
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3:
            raise RuntimeError(
                f"SmolVLA emitted unexpected action shape {tuple(actions.shape)}."
            )
        if actions.device != self.device:
            raise RuntimeError(
                f"SmolVLA actions are on {actions.device}, expected {self.device}."
            )
        return actions.detach().to(dtype=torch.float32)

    def sample_actions_from_images(
        self,
        *,
        primary_images: Sequence[np.ndarray],
        wrist_images: Sequence[np.ndarray | None] | None,
        aux_images: Sequence[np.ndarray | None] | None,
        observations: Sequence[dict[str, np.ndarray]],
        infos: Sequence[dict[str, Any] | None],
        instructions: Sequence[str],
    ) -> np.ndarray:
        if torch is None:
            raise RuntimeError("Torch is required for SmolVLA inference.")
        batch = self._prepare_batch(
            primary_images=primary_images,
            wrist_images=wrist_images,
            aux_images=aux_images,
            observations=observations,
            infos=infos,
            instructions=instructions,
        )
        return self._predict_actions_tensor(batch).cpu().numpy()

    def sample_actions_from_tensors(
        self,
        *,
        primary_images: Any,
        wrist_images: Any,
        states: Any,
        instructions: Sequence[str],
        aux_images: Any | None = None,
        microbatch_size: int = 0,
    ) -> Any:
        """Run batched SmolVLA without leaving the rank-local GPU."""

        if torch is None:
            raise RuntimeError("Torch is required for SmolVLA inference.")
        batch = adapt_cdpr_tensors_to_smolvla_batch(
            primary_images=primary_images,
            wrist_images=wrist_images,
            aux_images=aux_images,
            states=states,
            instructions=instructions,
            spec=self.obs_spec,
            device=self.device,
        )
        input_ids, attention_mask = self._tokenize(instructions)
        batch[SMOLVLA_LANGUAGE_TOKEN_KEY] = input_ids
        batch[SMOLVLA_LANGUAGE_MASK_KEY] = attention_mask
        batch_size = int(primary_images.shape[0])
        microbatch = int(microbatch_size)
        if microbatch <= 0 or microbatch >= batch_size:
            return self._predict_actions_tensor(batch)

        outputs: list[Any] = []
        for start in range(0, batch_size, microbatch):
            end = min(batch_size, start + microbatch)
            sliced: dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    sliced[key] = value[start:end]
                elif isinstance(value, list):
                    sliced[key] = value[start:end]
                else:
                    sliced[key] = value
            outputs.append(self._predict_actions_tensor(sliced))
        return torch.cat(outputs, dim=0)

    def sample_cdpr_chunks_from_tensors(
        self,
        *,
        primary_images: Any,
        wrist_images: Any,
        states: Any,
        instructions: Sequence[str],
        aux_images: Any | None = None,
        microbatch_size: int = 0,
    ) -> Any:
        raw = self.sample_actions_from_tensors(
            primary_images=primary_images,
            wrist_images=wrist_images,
            aux_images=aux_images,
            states=states,
            instructions=instructions,
            microbatch_size=microbatch_size,
        )
        return adapt_smolvla_action_tensors_to_cdpr(
            raw, spec=self.action_spec
        )

    def capture_cdpr_images(self, env: Any) -> tuple[np.ndarray, np.ndarray | None]:
        """Render the camera inputs for one CDPR environment state."""
        sim = env.sim
        primary = sim.capture_frame(sim.overview_cam, "overview")
        wrist = (
            sim.capture_frame(sim.ee_cam, "ee_camera")
            if hasattr(sim, "ee_cam")
            else None
        )
        return primary, wrist

    def sample_cdpr_chunks_from_images(
        self,
        *,
        primary_images: Sequence[np.ndarray],
        wrist_images: Sequence[np.ndarray | None] | None,
        observations: Sequence[dict[str, np.ndarray]],
        infos: Sequence[dict[str, Any] | None],
        instructions: Sequence[str],
    ) -> np.ndarray:
        """Run one batched SmolVLA forward for already-rendered CDPR views."""
        raw = self.sample_actions_from_images(
            primary_images=primary_images,
            wrist_images=wrist_images,
            aux_images=None,
            observations=observations,
            infos=infos,
            instructions=instructions,
        )
        chunks = [
            adapt_smolvla_actions_to_cdpr(item, spec=self.action_spec)
            for item in np.asarray(raw, dtype=np.float32)
        ]
        return np.stack(chunks, axis=0).astype(np.float32, copy=False)

    def sample_cdpr_chunks_from_envs(
        self,
        *,
        envs: Sequence[Any],
        observations: Sequence[dict[str, np.ndarray]],
        infos: Sequence[dict[str, Any] | None],
        instructions: Sequence[str],
    ) -> np.ndarray:
        primary_images = []
        wrist_images: list[np.ndarray | None] = []
        for env in envs:
            primary, wrist = self.capture_cdpr_images(env)
            primary_images.append(primary)
            wrist_images.append(wrist)
        return self.sample_cdpr_chunks_from_images(
            primary_images=primary_images,
            wrist_images=wrist_images,
            observations=observations,
            infos=infos,
            instructions=instructions,
        )

    def device_summary(self) -> str:
        if torch is None:
            return "torch=unavailable"
        if self.device.type != "cuda":
            return f"device={self.device}; dtype={self.dtype}"
        index = int(self.device.index or torch.cuda.current_device())
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / (1024**3)
        return (
            f"device={self.device}; name={props.name}; total_vram_gb={total_gb:.1f}; "
            f"dtype={self.dtype}; model_image_size={self.model_image_size or 'checkpoint'}; "
            f"compiled={self.compile_model}; compile_mode={self.compile_mode}"
        )


def _feature_dim(feature: Any) -> int | None:
    shape = getattr(feature, "shape", None)
    if not shape:
        return None
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def load_smolvla_runtime(
    checkpoint: str = DEFAULT_SMOLVLA_CHECKPOINT,
    **kwargs: Any,
) -> SmolVLARuntime:
    return SmolVLARuntime.load(checkpoint=checkpoint, **kwargs)
