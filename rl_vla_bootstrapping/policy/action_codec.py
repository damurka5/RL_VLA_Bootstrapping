from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QuantizationSpec:
    enabled: bool = True
    bins: int = 256


@dataclass(frozen=True)
class ActionCodec:
    action_keys: tuple[str, ...]
    controller_limits: dict[str, tuple[float, float]]
    normalized_low: float = -1.0
    normalized_high: float = 1.0
    chunk_size: int = 8
    quantization: QuantizationSpec = QuantizationSpec()

    def __post_init__(self):
        missing = [key for key in self.action_keys if key not in self.controller_limits]
        if missing:
            raise ValueError(f"Missing controller limits for action keys: {missing}")

    @property
    def action_dim(self) -> int:
        return len(self.action_keys)

    def normalize(self, controller_action: np.ndarray) -> np.ndarray:
        action = np.asarray(controller_action, dtype=np.float32).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"Expected controller action dim {self.action_dim}, got {action.size}")

        out = np.zeros_like(action, dtype=np.float32)
        span = float(self.normalized_high - self.normalized_low)
        for idx, key in enumerate(self.action_keys):
            lo, hi = self.controller_limits[key]
            if hi <= lo:
                raise ValueError(f"Invalid controller limits for `{key}`: {(lo, hi)}")
            unit = (float(action[idx]) - float(lo)) / float(hi - lo)
            out[idx] = np.float32(self.normalized_low + unit * span)
        return np.clip(out, self.normalized_low, self.normalized_high)

    def denormalize(self, normalized_action: np.ndarray) -> np.ndarray:
        action = np.asarray(normalized_action, dtype=np.float32).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"Expected normalized action dim {self.action_dim}, got {action.size}")

        out = np.zeros_like(action, dtype=np.float32)
        span = float(self.normalized_high - self.normalized_low)
        for idx, key in enumerate(self.action_keys):
            lo, hi = self.controller_limits[key]
            clipped = float(np.clip(action[idx], self.normalized_low, self.normalized_high))
            unit = (clipped - self.normalized_low) / span
            out[idx] = np.float32(lo + unit * float(hi - lo))
        return out

    def quantize(self, normalized_action: np.ndarray) -> np.ndarray:
        if not self.quantization.enabled:
            raise RuntimeError("Quantization is disabled for this action codec.")
        action = np.asarray(normalized_action, dtype=np.float32).reshape(-1)
        bins = int(self.quantization.bins)
        clipped = np.clip(action, self.normalized_low, self.normalized_high)
        scaled = (clipped - self.normalized_low) / (self.normalized_high - self.normalized_low)
        return np.rint(scaled * (bins - 1)).astype(np.int32)

    def dequantize(self, token_ids: np.ndarray) -> np.ndarray:
        if not self.quantization.enabled:
            raise RuntimeError("Quantization is disabled for this action codec.")
        tokens = np.asarray(token_ids, dtype=np.float32).reshape(-1)
        bins = float(self.quantization.bins - 1)
        scaled = tokens / bins
        return np.asarray(
            self.normalized_low + scaled * (self.normalized_high - self.normalized_low),
            dtype=np.float32,
        )

    def export_manifest(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "action_keys": list(self.action_keys),
            "action_dim": self.action_dim,
            "chunk_size": int(self.chunk_size),
            "normalized_range": [float(self.normalized_low), float(self.normalized_high)],
            "controller_limits": {
                key: [float(bounds[0]), float(bounds[1])]
                for key, bounds in self.controller_limits.items()
            },
            "quantization": {
                "enabled": bool(self.quantization.enabled),
                "bins": int(self.quantization.bins),
            },
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return output_path
