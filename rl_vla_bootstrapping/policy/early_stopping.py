"""Reading one stopping number out of an in-run validation record.

Kept free of torch and of the trainer's imports so the pipeline driver
(tools/train/run_widowx200_pipeline.py) can rank checkpoints by exactly the
number the trainer stopped on, without loading the training stack.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def parse_validation_caps(values: Sequence[str], instruction_ids: Mapping[str, int]) -> dict[int, float]:
    """Fixed evaluation distances, independent of the training curriculum."""

    caps: dict[int, float] = {}
    for value in values:
        name, separator, raw = str(value).partition("=")
        if not separator or name not in instruction_ids:
            raise ValueError(f"Invalid validation cap {value!r}; expected instruction=metres")
        cap = float(raw)
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError(f"Validation cap must be finite and positive: {value!r}")
        key = int(instruction_ids[name])
        if key in caps:
            raise ValueError(f"Duplicate validation cap for {name}")
        caps[key] = cap
    return caps


def validation_matches_caps(metrics: Mapping[str, Any], caps: Mapping[str, float]) -> bool:
    return metrics.get("validation/fixed_start_distance") == 1.0 and all(
        math.isclose(float(metrics.get(f"validation/start_distance_cap/{name}", -1)), cap)
        for name, cap in caps.items()
    )


def early_stop_metric_value(metrics: Mapping[str, Any], metric: str) -> float:
    """One number from a validation record, as the stop rule reads it.

    ``min_by_instruction`` is the worst instruction's success rate.
    ``min_by_instruction_composed`` is the same, except that a container
    instruction is read from the composed (uncaught-start) leg when the run has
    one: that is the full put_into task, where the plain leg is mostly
    carry-only episodes. Anything else is a literal key.
    """

    if metric in ("min_by_instruction", "min_by_instruction_composed"):
        rates: dict[str, float] = {}
        prefixes = ["validation/by_instruction/"]
        if metric == "min_by_instruction_composed":
            prefixes.append("validation_composed/by_instruction/")
        for prefix in prefixes:
            for key, value in metrics.items():
                if key.startswith(prefix) and key.endswith("/success_rate"):
                    rates[key[len(prefix) : -len("/success_rate")]] = float(value)
        return min(rates.values()) if rates else float("nan")
    return float(metrics.get(metric, float("nan")))
