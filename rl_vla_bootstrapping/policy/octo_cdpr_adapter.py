from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_OCTO_SMALL_CHECKPOINT = "hf://rail-berkeley/octo-small-1.5"
DEFAULT_OCTO_REPO_PATH = "/root/repo/octo"
CDPR_ACTION_KEYS: tuple[str, ...] = ("x", "y", "z", "yaw", "gripper")


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected an RGB-like image with shape HxWxC, got {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize_uint8_rgb(image: np.ndarray, size: int | tuple[int, int] | None) -> np.ndarray:
    arr = _as_uint8_rgb(image)
    if size is None:
        return arr
    if isinstance(size, int):
        target_hw = (int(size), int(size))
    else:
        target_hw = (int(size[0]), int(size[1]))
    if arr.shape[:2] == target_hw:
        return arr

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - only exercised without pillow
        raise RuntimeError("Pillow is required to resize Octo RGB observations.") from exc

    pil = Image.fromarray(arr, mode="RGB")
    pil = pil.resize((target_hw[1], target_hw[0]), resample=Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _history_batch(value: np.ndarray, history: int) -> np.ndarray:
    arr = np.asarray(value)
    if history <= 1:
        return arr[None, None]
    stacked = np.repeat(arr[None], int(history), axis=0)
    return stacked[None]


def _pad_mask(history: int, valid: bool = True) -> np.ndarray:
    return np.full((1, int(history)), bool(valid), dtype=bool)


@dataclass(frozen=True)
class CDPRStateLayout:
    state_dim: int
    max_objects: int
    ee_slice: slice
    target_slice: slice
    all_objects_slice: slice
    object_mask_slice: slice
    instruction_slice: slice
    goal_direction_slice: slice

    @classmethod
    def from_observation(cls, obs: dict[str, np.ndarray]) -> "CDPRStateLayout":
        ee = np.asarray(obs["ee_position"], dtype=np.float32).reshape(-1)
        target = np.asarray(obs["target_object_position"], dtype=np.float32).reshape(-1)
        all_objects = np.asarray(obs["all_object_positions"], dtype=np.float32).reshape(-1)
        object_mask = np.asarray(obs["object_position_mask"], dtype=np.float32).reshape(-1)
        instruction = np.asarray(obs["instruction_onehot"], dtype=np.float32).reshape(-1)
        goal_direction = np.asarray(obs["goal_direction"], dtype=np.float32).reshape(-1)

        offset = 0
        ee_slice = slice(offset, offset + ee.size)
        offset = ee_slice.stop
        target_slice = slice(offset, offset + target.size)
        offset = target_slice.stop
        all_objects_slice = slice(offset, offset + all_objects.size)
        offset = all_objects_slice.stop
        object_mask_slice = slice(offset, offset + object_mask.size)
        offset = object_mask_slice.stop
        instruction_slice = slice(offset, offset + instruction.size)
        offset = instruction_slice.stop
        goal_direction_slice = slice(offset, offset + goal_direction.size)
        offset = goal_direction_slice.stop

        return cls(
            state_dim=int(offset),
            max_objects=int(object_mask.size),
            ee_slice=ee_slice,
            target_slice=target_slice,
            all_objects_slice=all_objects_slice,
            object_mask_slice=object_mask_slice,
            instruction_slice=instruction_slice,
            goal_direction_slice=goal_direction_slice,
        )

    def flatten(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return flatten_cdpr_observation(obs)


def flatten_cdpr_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["ee_position"], dtype=np.float32).reshape(-1),
            np.asarray(obs["target_object_position"], dtype=np.float32).reshape(-1),
            np.asarray(obs["all_object_positions"], dtype=np.float32).reshape(-1),
            np.asarray(obs["object_position_mask"], dtype=np.float32).reshape(-1),
            np.asarray(obs["instruction_onehot"], dtype=np.float32).reshape(-1),
            np.asarray(obs["goal_direction"], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def cdpr_proprio_from_observation(
    obs: dict[str, np.ndarray],
    info: dict[str, Any] | None = None,
) -> np.ndarray:
    info = dict(info or {})
    ee = np.asarray(obs.get("ee_position", info.get("ee_position", (0.0, 0.0, 0.0))), dtype=np.float32)
    ee = ee.reshape(-1)[:3]
    yaw = float(info.get("ee_yaw", 0.0))
    gripper = float(info.get("gripper_opening", info.get("gripper_target", 1.0)))
    return np.asarray([ee[0], ee[1], ee[2], yaw, gripper], dtype=np.float32)


@dataclass(frozen=True)
class OctoObservationSpec:
    image_size: int = 256
    history: int = 1
    primary_image_key: str = "image_primary"
    wrist_image_key: str = "image_wrist"
    proprio_key: str = "proprio"
    include_wrist: bool = True
    include_proprio: bool = True


class CDPROctoObservationAdapter:
    def __init__(self, spec: OctoObservationSpec | None = None):
        self.spec = spec or OctoObservationSpec()

    @property
    def history(self) -> int:
        return max(1, int(self.spec.history))

    def from_images(
        self,
        *,
        primary_image: np.ndarray,
        wrist_image: np.ndarray | None = None,
        proprio: np.ndarray | None = None,
    ) -> dict[str, Any]:
        history = self.history
        primary = _resize_uint8_rgb(primary_image, int(self.spec.image_size))
        observation: dict[str, Any] = {
            self.spec.primary_image_key: _history_batch(primary, history),
            "timestep_pad_mask": _pad_mask(history, valid=True),
            "pad_mask_dict": {
                self.spec.primary_image_key: _pad_mask(history, valid=True),
            },
        }

        if self.spec.include_wrist and wrist_image is not None:
            wrist = _resize_uint8_rgb(wrist_image, int(self.spec.image_size))
            observation[self.spec.wrist_image_key] = _history_batch(wrist, history)
            observation["pad_mask_dict"][self.spec.wrist_image_key] = _pad_mask(history, valid=True)

        if self.spec.include_proprio and proprio is not None:
            prop = np.asarray(proprio, dtype=np.float32).reshape(-1)
            observation[self.spec.proprio_key] = _history_batch(prop, history).astype(np.float32, copy=False)
            observation["pad_mask_dict"][self.spec.proprio_key] = _pad_mask(history, valid=True)

        return observation

    def from_env(
        self,
        *,
        sim: Any,
        obs: dict[str, np.ndarray],
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        primary = sim.capture_frame(sim.overview_cam, "overview")
        wrist = None
        if self.spec.include_wrist and hasattr(sim, "ee_cam"):
            wrist = sim.capture_frame(sim.ee_cam, "ee_camera")
        return self.from_images(
            primary_image=primary,
            wrist_image=wrist,
            proprio=cdpr_proprio_from_observation(obs, info),
        )


@dataclass(frozen=True)
class OctoActionAdapterSpec:
    action_dim: int = 5
    chunk_size: int = 8
    action_indices: tuple[int, ...] | None = None
    normalization: str = "tanh"


def _resolve_action_indices(source_dim: int, requested: tuple[int, ...] | None) -> tuple[int, ...]:
    if requested:
        if len(requested) != 5:
            raise ValueError("Octo CDPR action_indices must contain exactly five entries.")
        resolved: list[int] = []
        for raw_idx in requested:
            idx = int(raw_idx)
            if idx < 0:
                idx = int(source_dim) + idx
            if idx < 0 or idx >= int(source_dim):
                raise ValueError(f"Action index {raw_idx} is out of range for source dim {source_dim}.")
            resolved.append(idx)
        return tuple(resolved)
    if source_dim >= 7:
        return (0, 1, 2, 5, 6)
    if source_dim >= 5:
        return (0, 1, 2, 3, 4)
    return tuple(range(source_dim))


def adapt_octo_actions_to_cdpr(
    actions: np.ndarray,
    *,
    spec: OctoActionAdapterSpec | None = None,
) -> np.ndarray:
    spec = spec or OctoActionAdapterSpec()
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected Octo actions with shape [H, D] or [B, H, D], got {arr.shape}.")

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
        raise ValueError(f"Unsupported Octo action normalization mode: {spec.normalization!r}")

    target_horizon = max(1, int(spec.chunk_size))
    if out.shape[0] < target_horizon:
        pad = np.repeat(out[-1:], target_horizon - out.shape[0], axis=0)
        out = np.concatenate([out, pad], axis=0)
    elif out.shape[0] > target_horizon:
        out = out[:target_horizon]
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


class OctoRuntime:
    def __init__(
        self,
        *,
        model: Any,
        jax: Any,
        checkpoint: str,
        seed: int = 0,
        use_dataset_action_unnorm: bool = False,
    ) -> None:
        self.model = model
        self.jax = jax
        self.checkpoint = str(checkpoint)
        self.rng = jax.random.PRNGKey(int(seed))
        self.use_dataset_action_unnorm = bool(use_dataset_action_unnorm)

    @classmethod
    def load(
        cls,
        checkpoint: str = DEFAULT_OCTO_SMALL_CHECKPOINT,
        *,
        seed: int = 0,
        use_dataset_action_unnorm: bool = False,
    ) -> "OctoRuntime":
        _prepare_octo_import_path()
        try:
            module = importlib.import_module("octo.model.octo_model")
            jax = importlib.import_module("jax")
        except ImportError as exc:
            found = importlib.util.find_spec("octo")
            origin = getattr(found, "origin", None) if found is not None else None
            locations = list(getattr(found, "submodule_search_locations", []) or []) if found is not None else []
            raise RuntimeError(
                "Could not import Berkeley Octo/JAX runtime. Expected a checkout at "
                f"`{os.environ.get('OCTO_REPO_PATH', DEFAULT_OCTO_REPO_PATH)}` containing "
                "`octo/model/octo_model.py`, plus JAX installed in the active `octo` env. "
                f"Python found octo origin={origin!r}, locations={locations!r}. "
                "If this points to a non-Berkeley package, uninstall that package or set "
                "OCTO_REPO_PATH=/root/repo/octo."
            ) from exc
        model = module.OctoModel.load_pretrained(str(checkpoint))
        return cls(
            model=model,
            jax=jax,
            checkpoint=str(checkpoint),
            seed=int(seed),
            use_dataset_action_unnorm=bool(use_dataset_action_unnorm),
        )

    def create_task(self, instruction: str) -> Any:
        text = str(instruction)
        try:
            return self.model.create_tasks(texts=[text])
        except TypeError:
            return self.model.create_tasks(texts=text)

    def sample_actions(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        self.rng, sample_rng = self.jax.random.split(self.rng)
        task = self.create_task(instruction)
        kwargs: dict[str, Any] = {"rng": sample_rng}
        if self.use_dataset_action_unnorm:
            stats = getattr(self.model, "dataset_statistics", None)
            if isinstance(stats, dict) and "action" in stats:
                kwargs["unnormalization_statistics"] = stats["action"]
        actions = self.model.sample_actions(observation, task, **kwargs)
        return np.asarray(self.jax.device_get(actions), dtype=np.float32)

    def pretty_spec(self) -> str:
        getter = getattr(self.model, "get_pretty_spec", None)
        if callable(getter):
            return str(getter())
        return f"OctoRuntime(checkpoint={self.checkpoint})"


def load_octo_runtime(
    checkpoint: str = DEFAULT_OCTO_SMALL_CHECKPOINT,
    *,
    seed: int = 0,
    use_dataset_action_unnorm: bool = False,
) -> OctoRuntime:
    return OctoRuntime.load(
        checkpoint=checkpoint,
        seed=int(seed),
        use_dataset_action_unnorm=bool(use_dataset_action_unnorm),
    )


def _prepare_octo_import_path() -> None:
    repo_path = Path(os.environ.get("OCTO_REPO_PATH", DEFAULT_OCTO_REPO_PATH)).expanduser()
    expected = repo_path / "octo" / "model" / "octo_model.py"
    if expected.is_file():
        repo_str = repo_path.resolve().as_posix()
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        loaded = sys.modules.get("octo")
        if loaded is not None and not hasattr(loaded, "__path__"):
            sys.modules.pop("octo", None)
            for name in list(sys.modules):
                if name.startswith("octo."):
                    sys.modules.pop(name, None)
