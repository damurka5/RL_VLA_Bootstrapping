"""Reading one stopping number out of an in-run validation record.

Kept free of torch and of the trainer's imports so the pipeline driver
(tools/train/run_widowx200_pipeline.py) can rank checkpoints by exactly the
number the trainer stopped on, without loading the training stack.
"""

from __future__ import annotations

from typing import Any, Mapping


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
