from __future__ import annotations

import numpy as np


def group_relative_advantages(
    scores: np.ndarray,
    *,
    normalize: bool = True,
    eps: float = 1.0e-6,
    clip_abs: float = 6.0,
) -> np.ndarray:
    group_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    centered = group_scores - float(group_scores.mean())
    if normalize:
        centered = centered / max(float(group_scores.std(ddof=0)), float(eps))
    if clip_abs > 0.0:
        centered = np.clip(centered, -float(clip_abs), float(clip_abs))
    return centered.astype(np.float32)
