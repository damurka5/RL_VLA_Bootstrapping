#!/usr/bin/env python3
"""Compare saved residual weights and effective LoRA updates on CPU."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor))


def tensor_delta(before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
    if before.shape != after.shape:
        raise ValueError(f"Incompatible tensor shapes: {before.shape} and {after.shape}")
    before, after = before.double(), after.double()
    difference = after - before
    return {
        "before_norm": norm(before),
        "after_norm": norm(after),
        "delta_norm": norm(difference),
        "changed_elements": int(torch.count_nonzero(difference)),
        "elements": difference.numel(),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        key: sum(row[key] ** 2 for row in rows) ** 0.5
        for key in ("before_norm", "after_norm", "delta_norm")
    }
    result["relative_to_before"] = (
        result["delta_norm"] / result["before_norm"] if result["before_norm"] else None
    )
    return result


def lora_factors(payload: dict[str, Any]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    state = payload.get("vla_lora") or {}
    a_names = {key.removesuffix(".lora_a") for key in state if key.endswith(".lora_a")}
    b_names = {key.removesuffix(".lora_b") for key in state if key.endswith(".lora_b")}
    if a_names != b_names or len(state) != 2 * len(a_names):
        raise ValueError("Expected paired .lora_a and .lora_b tensors in vla_lora")
    return {name: (state[f"{name}.lora_a"], state[f"{name}.lora_b"]) for name in a_names}


def effective_lora(factors: tuple[torch.Tensor, torch.Tensor], alpha: float) -> torch.Tensor:
    a, b = factors
    return (b.double() @ a.double()) * (alpha / a.shape[0])


def lora_group(name: str) -> str:
    if ".lm_expert." in name:
        return "action_expert"
    if ".vision_model." in name:
        return "vision_encoder"
    return "other"


def describe(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: payload.get(key) for key in ("global_step", "base_checkpoint", "policy_type")},
        **{key: payload["args"].get(key) for key in ("lora_rank", "lora_alpha", "train_vla_vision_lora")},
        "sil_sft": payload.get("sil_sft"),
    }


def cdpr_residual_layout(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the input columns and chunk-major output rows of the CDPR MLP."""
    if before["policy_type"] != "smolvla_cdpr_grpo" or before["action_dim"] != 5:
        return None
    proprio = before["args"].get("state_dim")
    vision = before["args"].get("residual_vision_dim")
    executed = before["args"].get("replan_every")
    for field in ("state_dim", "residual_vision_dim", "replan_every", "residual_relative_target"):
        if before["args"].get(field) != after["args"].get(field):
            return None
    chunk = before["chunk_size"]
    state = before["state_dim"]
    if not (proprio and vision is not None and proprio + vision == state and executed):
        return None
    if not 0 < executed <= chunk:
        return None

    input_key, output_key = "actor.net.net.0.weight", "actor.net.net.4.weight"
    if input_key not in before["policy"] or output_key not in before["policy"]:
        return None
    weights = [payload["policy"] for payload in (before, after)]
    first = [p[input_key] for p in weights]
    output = [p[output_key] for p in weights]
    if first[0].ndim != 2 or first[0].shape[1] != state + chunk * 5:
        return None
    if output[0].ndim != 2 or output[0].shape[0] != chunk * 5:
        return None
    last = [tensor.reshape(chunk, 5, -1) for tensor in output]
    columns = {
        "state": slice(0, proprio),
        "pooled_vision": slice(proprio, state),
        "prior": slice(state, None),
    }
    actions = ("x", "y", "z", "yaw", "gripper")
    return {
        "actions_per_decision": executed,
        "input_weights": {
            name: tensor_delta(first[0][:, cols], first[1][:, cols]) for name, cols in columns.items()
        },
        "output_weights_by_step": [
            {name: tensor_delta(last[0][step, i], last[1][step, i]) for i, name in enumerate(actions)}
            for step in range(chunk)
        ],
        "output_weights_by_executed_action": {
            name: tensor_delta(last[0][:executed, i], last[1][:executed, i]) for i, name in enumerate(actions)
        },
        "output_weights_by_execution": {
            "executed": tensor_delta(last[0][:executed], last[1][:executed]),
            "unexecuted": tensor_delta(last[0][executed:], last[1][executed:]),
        },
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    for key in ("policy_type", "base_checkpoint", "state_dim", "action_dim", "chunk_size", "residual_scale"):
        if before.get(key) is None or before[key] != after.get(key):
            raise ValueError(f"Checkpoint {key} is missing or incompatible")
    if before["policy"].keys() != after["policy"].keys():
        raise ValueError("Saved policy tensors do not describe the same architecture")

    policy = {name: tensor_delta(tensor, after["policy"][name]) for name, tensor in before["policy"].items()}
    former, latter = lora_factors(before), lora_factors(after)
    modules = []
    expert_modules = []
    groups = defaultdict(list)
    expert_projections = defaultdict(list)
    for name in sorted(former.keys() | latter.keys()):
        a = effective_lora(former[name], float(before["args"]["lora_alpha"])) if name in former else None
        b = effective_lora(latter[name], float(after["args"]["lora_alpha"])) if name in latter else None
        row = tensor_delta(a if a is not None else torch.zeros_like(b), b if b is not None else torch.zeros_like(a))
        row.update(name=name, present_before=name in former, present_after=name in latter)
        modules.append(row)
        group = lora_group(name)
        groups[group].append(row)
        if group == "action_expert":
            expert_projections[name.rsplit(".", 1)[-1]].append(row)
            if a is not None and b is not None and row["delta_norm"] > 0:
                expert_modules.append(row)

    spectra = []
    for row in sorted(expert_modules, key=lambda item: -item["delta_norm"])[:3]:
        name = row["name"]
        old_weight = effective_lora(former[name], float(before["args"]["lora_alpha"]))
        new_weight = effective_lora(latter[name], float(after["args"]["lora_alpha"]))
        energy = torch.linalg.svdvals(new_weight - old_weight).square()
        spectra.append({
            "name": name,
            "top_1_squared_norm_share": float(energy[0] / energy.sum()),
            "top_4_squared_norm_share": float(energy[:4].sum() / energy.sum()),
        })

    return {
        "before": describe(before),
        "after": describe(after),
        "policy_tensors": policy,
        "residual_actor": aggregate([row for name, row in policy.items() if name.startswith("actor.")]),
        "cdpr_residual_layout": cdpr_residual_layout(before, after),
        "lora_groups": {
            key: {
                **aggregate(rows),
                "modules": len(rows),
                "modules_added": sum(not row["present_before"] for row in rows),
                "modules_removed": sum(not row["present_after"] for row in rows),
                "modules_changed": sum(row["changed_elements"] > 0 for row in rows),
            }
            for key, rows in groups.items()
        },
        "action_expert_projections": {name: aggregate(rows) for name, rows in expert_projections.items()},
        "action_expert_largest_updates_spectra": spectra,
        "lora_modules": modules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, help="Write a new JSON report to this file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    before = torch.load(args.before, map_location="cpu", weights_only=True)
    after = torch.load(args.after, map_location="cpu", weights_only=True)
    report = json.dumps(compare(before, after), indent=2) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        with args.output.open("x", encoding="utf-8") as file:
            file.write(report)


if __name__ == "__main__":
    main()
