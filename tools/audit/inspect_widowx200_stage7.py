#!/usr/bin/env python3
"""Read a standalone stage-7 run's stop reason, validation curves and health.

No models, CUDA or TensorBoard dependencies. Does not change training state.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_vla_bootstrapping.policy.early_stopping import (  # noqa: E402
    early_stop_metric_value, validation_matches_caps,
)

RATE_KEYS = {
    "move_to": "validation/by_instruction/move_to_object/success_rate",
    "pick_up": "validation/by_instruction/pick_up/success_rate",
    "ordinary_plate": "validation/by_instruction/put_into_plate/success_rate",
    "ordinary_bowl": "validation/by_instruction/put_into_bowl/success_rate",
    "composed_plate": "validation_composed/by_instruction/put_into_plate/success_rate",
    "composed_bowl": "validation_composed/by_instruction/put_into_bowl/success_rate",
}
HEALTH_KEYS = (
    "candidate_reward_mean", "candidate_reward_std", "group_pass_rate_mean",
    "physical_grasp_rate", "physical_lift_rate", "physical_release_rate",
    "left_pad_normal_force_mean_n", "right_pad_normal_force_mean_n",
    "non_finite_ee_worlds", "non_finite_live_episodes", "non_finite_live_episode_rate",
    "vla_lora/grad_norm", "vla_lora/kl", "vla_lora/ppo_loss",
)


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def inspect(run: Path) -> dict[str, Any]:
    run = run.expanduser().resolve()
    rl = run if run.name == "rl" else run / "rl"
    if not rl.is_dir():
        raise ValueError(f"Missing RL directory: {rl}")
    warnings = []

    def read_json(name):
        path = rl / name
        return json.loads(path.read_text()) if path.is_file() else {}

    def read_jsonl(name):
        path = rl / name
        rows = []
        if path.is_file():
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected an object")
                    rows.append(row)
                except ValueError:
                    warnings.append(f"Skipped incomplete/invalid {name} line {line_number}")
        return rows

    args = read_json("config.json")
    metric = args.get("early_stop_metric", "min_by_instruction_composed")
    caps = {name: float(raw) for item in args.get("validation_start_distance_caps", [])
            for name, raw in [item.split("=", 1)]}
    validations = read_jsonl("validation.jsonl")
    matching = [row for row in validations if not caps or validation_matches_caps(row, caps)]

    def validation_row(row):
        effective = {}
        prefixes = ["validation/by_instruction/"]
        if metric == "min_by_instruction_composed":
            prefixes.append("validation_composed/by_instruction/")
        for prefix in prefixes:
            for key, value in row.items():
                if key.startswith(prefix) and key.endswith("/success_rate"):
                    effective[key[len(prefix):-len("/success_rate")]] = value
        finite = {key: value for key, value in effective.items()
                  if isinstance(value, (int, float)) and math.isfinite(value)}
        step = int(row["global_step"])
        checkpoint = rl / f"step_{step:07d}" / "smolvla_grpo_adapter.pt"
        return {"step": step, "score": early_stop_metric_value(row, metric),
                "bottleneck": min(finite, key=finite.get) if finite else None,
                "rates": {label: row.get(key) for label, key in RATE_KEYS.items()},
                "episodes": {label: row.get(key.replace("/success_rate", "/episodes"))
                             for label, key in RATE_KEYS.items()},
                "checkpoint": str(checkpoint) if checkpoint.is_file() else None}

    curve = [validation_row(row) for row in matching]
    finite_curve = [row for row in curve if math.isfinite(row["score"])]
    metrics = read_jsonl("metrics.jsonl")
    latest = metrics[-1] if metrics else {}
    non_finite = {
        key: sum(isinstance(row.get(key), (int, float)) and not math.isfinite(row[key])
                 for row in metrics)
        for key in HEALTH_KEYS
    }
    return clean({
        "run_dir": str(rl.parent), "stop": read_json("early_stop.json") or None,
        "rules": {key: value for key, value in args.items()
                  if key.startswith("early_stop_") or key == "max_train_steps"},
        "metric": metric, "validation_caps": caps,
        "validation_rows": len(validations), "matching_validation_rows": len(matching),
        "best": max(finite_curve, key=lambda row: row["score"]) if finite_curve else None,
        "validation_curve": curve, "training_updates": len(metrics),
        "last_training_step": latest.get("global_step"),
        "last_training_metrics": {key: value for key, value in latest.items()
                                  if key in HEALTH_KEYS or key.startswith("curriculum/")},
        "non_finite_metric_updates": {key: count for key, count in non_finite.items() if count},
        "warnings": warnings,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report file")
    args = parser.parse_args(argv)
    report = inspect(args.run_dir)
    print("Stop:", json.dumps(report["stop"], sort_keys=True))
    print("Rules:", json.dumps(report["rules"], sort_keys=True))
    best = report["best"]
    if best:
        print(f"Best: step {best['step']}, {best['score']:.2%}, bottleneck {best['bottleneck']}")
        print("Best checkpoint:", best["checkpoint"])
    print("Last validations (ordinary placements can start holding):")
    print("     step   monitored    move_to    pick_up  ord_plate   ord_bowl comp_plate  comp_bowl")
    for row in report["validation_curve"][-6:]:
        def percent(value):
            return "-" if value is None else f"{value:.2%}"
        values = [row["score"], *row["rates"].values()]
        print(f"{row['step']:9d}  " + " ".join(f"{percent(value):>10}" for value in values))
    print("Last training metrics:", json.dumps(report["last_training_metrics"], sort_keys=True))
    print("Updates with non-finite health metrics:", json.dumps(report["non_finite_metric_updates"]))
    for warning in report["warnings"]:
        print("Warning:", warning)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("Report:", args.output.resolve())


if __name__ == "__main__":
    main()
