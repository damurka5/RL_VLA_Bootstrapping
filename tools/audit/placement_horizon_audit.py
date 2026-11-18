#!/usr/bin/env python3
"""Inspect placement timeouts, optionally compare longer runs on matching scenes.

Reads policy_outcome_videos --grasp-audit outputs on CPU. Verdicts always come
from the recorded predicate. Nearness and motion are diagnostics, not success.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def load_round(path):
    with np.load(path, allow_pickle=False) as data:
        record = dict(data)
    trace_path = path.with_name(path.name.replace("record_", "grasp_trace_"))
    with np.load(trace_path, allow_pickle=False) as trace:
        if str(trace["recording_sha256"]) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Grasp trace belongs to a different recording: {trace_path}")
        columns = {str(name): i for i, name in enumerate(trace["columns"])}
        active = trace["values"][:, :, columns["active"]].astype(bool)
        metadata = json.loads(str(trace["metadata_json"]))
    if active.shape != record["active"].shape:
        raise ValueError(f"Misaligned grasp trace: {trace_path}")
    record["live_diverged"] = (record["active"] & ~active).any(axis=0)
    record["metadata"] = metadata
    return record


def episode_rows(record, radii, tail_steps=20):
    if not bool(record.get("contact_based_grasp", False)):
        raise ValueError("This audit requires contact-v2 placement recordings")
    active = record["active"].astype(bool)
    worlds = np.arange(active.shape[1])
    target = record["object_xyz"][:, worlds, record["target_slots"]]
    reference = record["object_xyz"][:, worlds, record["reference_slots"]]
    distance = np.linalg.norm(target[:, :, :2] - reference[:, :, :2], axis=-1)
    metadata = record["metadata"]
    result = []
    for world in worlds:
        indices = np.flatnonzero(active[:, world])
        if not len(indices):
            raise ValueError(f"World {world} has no active steps")
        last = int(indices[-1])
        previous = int(indices[max(0, len(indices) - 1 - tail_steps)])
        success_steps = np.flatnonzero(record["success"][:, world] & active[:, world])
        success = bool(len(success_steps))
        budget = int(record["horizons"][world] * record["actions_per_decision"])
        timeout = len(indices) >= budget
        held = bool(record["caught_target"][last, world])
        diverged = bool(record["live_diverged"][world])
        instruction = metadata["instructions"][int(record["instruction_ids"][world])]
        grasp = np.flatnonzero(record["caught_target"][:, world] & active[:, world])
        category = ("success" if success else "live_divergence" if diverged else
                    "early_failure" if not timeout else "timeout_holding" if held else "timeout_unheld")
        result.append({
            "round": int(record["round_index"]), "world": int(world),
            "scene": int(world) // int(metadata["group_size"]),
            "instruction": instruction,
            "object": metadata["catalogs"][int(record["target_catalog_ids"][world])],
            "success": success, "first_success_env_step": int(success_steps[0]) if success else -1,
            "first_grasp_env_step": int(grasp[0]) if len(grasp) else -1,
            "live_diverged": diverged, "category": category,
            "active_steps": int(len(indices)), "budget_env_steps": budget,
            "held_final": held, "ever_grasped": bool(len(grasp)),
            "final_xy_m": float(distance[last, world]),
            "within_radius_final": bool(distance[last, world] <= radii[instruction]),
            "xy_improvement_tail_m": float(distance[previous, world] - distance[last, world]),
            "target_displacement_tail_m": float(np.linalg.norm(target[last, world] - target[previous, world])),
        })
    return result


def summarize(rows):
    result = []
    for name in sorted({row["instruction"] for row in rows}):
        subset = [row for row in rows if row["instruction"] == name]
        holding = [row for row in subset if row["category"] == "timeout_holding"]
        result.append({
            "instruction": name, "episodes": len(subset),
            "distinct_reset_groups": len({(r["round"], r["scene"]) for r in subset}),
            "successes": sum(row["success"] for row in subset),
            "ever_grasped": sum(row["ever_grasped"] for row in subset),
            "categories": dict(Counter(row["category"] for row in subset)),
            "timeout_holding_within_radius": sum(row["within_radius_final"] for row in holding),
            "timeout_holding_improved_at_least_1cm_in_tail": sum(row["xy_improvement_tail_m"] >= .01 for row in holding),
            "timeout_holding_displaced_less_than_5mm_in_tail": sum(row["target_displacement_tail_m"] < .005 for row in holding),
        })
    return result


def compare_rounds(baseline, extended, baseline_rows, extended_rows):
    """Match reset state and audit the common prefix before attributing a rescue.

    Re-running a GPU simulation can change the trajectory before the old limit.
    Such outcomes are reported but never attributed to extra time.
    """
    if baseline["metadata"] != extended["metadata"]:
        raise ValueError("Grasp contracts/code hashes differ between runs")
    for key in ("round_index", "actions_per_decision", "start_distance_cap", "contact_based_grasp",
                "instruction_ids", "target_slots", "reference_slots", "second_reference_slots",
                "target_catalog_ids", "instructions", "release_threshold", "target_rest_height",
                "support_surface_z", "reset_object_xyz", "reset_ee_xyz",
                "reset_object_quaternions", "reset_ee_quaternion"):
        a, b = np.asarray(baseline[key]), np.asarray(extended[key])
        same = a.shape == b.shape and (np.allclose(a, b, rtol=0, atol=1e-6)
               if a.dtype.kind in "fc" else np.array_equal(a, b))
        if not same:
            raise ValueError(f"Different scenes/settings: {key}")
    if not np.all(extended["horizons"] > baseline["horizons"]):
        raise ValueError("Extended horizons must exceed the baseline in every world")
    if len(baseline_rows) != len(extended_rows):
        raise ValueError("Different world counts")
    common = min(len(baseline["active"]), len(extended["active"]))
    result = []
    for before, after in zip(baseline_rows, extended_rows):
        world = before["world"]
        end = min(common, before["budget_env_steps"])
        live = baseline["active"][:end, world].astype(bool)
        # Require the whole baseline episode prefix, including its end, to
        # agree. Checking only the intersection hides early-success reruns.
        mask_match = np.array_equal(live, extended["active"][:end, world].astype(bool))
        errors = {}
        for key in ("actions", "ee_xyz", "object_xyz"):
            delta = np.abs(baseline[key][:end, world][live] - extended[key][:end, world][live])
            errors[key] = float(np.max(delta)) if delta.size and np.isfinite(delta).all() else float("inf")
        outcomes_match = all(np.array_equal(baseline[key][:end, world][live],
                             extended[key][:end, world][live]) for key in ("success", "terminated"))
        prefix_matches = (mask_match and outcomes_match and errors["actions"] <= 1e-4
                          and errors["ee_xyz"] <= 1e-3 and errors["object_xyz"] <= 1e-3)
        rescued = (before["category"].startswith("timeout_") and after["success"]
                   and after["first_success_env_step"] >= before["budget_env_steps"] and prefix_matches)
        result.append({"round": before["round"], "world": world, "scene": before["scene"],
                       "instruction": before["instruction"], "object": before["object"],
                       "baseline_success": before["success"], "extended_success": after["success"],
                       "baseline_category": before["category"], "prefix_matches": prefix_matches,
                       "rescued_after_original_budget": bool(rescued),
                       "extended_first_success_env_step": after["first_success_env_step"],
                       **{f"prefix_max_{key}_error": value for key, value in errors.items()}})
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(baseline_dir, output, extended_dir=None, tail_steps=20):
    manifest = json.loads((baseline_dir / "manifest.json").read_text())
    metadata = yaml.safe_load((baseline_dir / "source_eval_config.yaml").read_text())["task"]["metadata"]
    radii = {"put_into_plate": float(metadata["put_plate_xy_tolerance"]),
             "put_into_bowl": float(metadata["put_bowl_xy_tolerance"])}
    paths = sorted(baseline_dir.glob("record_*.npz"))
    if not paths:
        raise ValueError("No baseline record_*.npz files")
    if extended_dir:
        other = json.loads((extended_dir / "manifest.json").read_text())
        for key in ("checkpoint", "eval_config_sha256", "worlds", "group_size", "seed_torch", "oracle"):
            if manifest[key] != other[key]:
                raise ValueError(f"Different run settings: {key}")
        if {p.name for p in paths} != {p.name for p in extended_dir.glob("record_*.npz")}:
            raise ValueError("Recording round sets differ")
    rows, extended_rows, pairs = [], [], []
    for path in paths:
        baseline = load_round(path)
        before = episode_rows(baseline, radii, tail_steps)
        rows.extend(before)
        if extended_dir:
            extended = load_round(extended_dir / path.name)
            after = episode_rows(extended, radii, tail_steps)
            extended_rows.extend(after)
            pairs.extend(compare_rounds(baseline, extended, before, after))
    report = {"baseline": str(baseline_dir.resolve()), "tail_env_steps": tail_steps,
              "baseline_summary": summarize(rows),
              "notes": ["Success is the recorded predicate verdict, never inferred from distance.",
                        "Copies within a reset group are correlated; episode counts are not independent scene counts.",
                        "A timeout while holding is a candidate for extra time, not a promised success.",
                        "For comparison, keep the checkpoint file unchanged; manifests identify its path, not its bytes.",
                        "Prefix matches allow 0.0001 action error and 0.001 m position error; GPU reruns may differ."]}
    if pairs:
        report.update(extended=str(extended_dir.resolve()), extended_summary=summarize(extended_rows),
                      paired_summary=[{"instruction": name, "episodes": len(subset),
                                       "prefix_matches": sum(r["prefix_matches"] for r in subset),
                                       "rescued_after_original_budget": sum(r["rescued_after_original_budget"] for r in subset),
                                       "baseline_success_lost_on_rerun": sum(r["baseline_success"] and not r["extended_success"] for r in subset)}
                                      for name in sorted({r["instruction"] for r in pairs})
                                      for subset in [[r for r in pairs if r["instruction"] == name]]])
    # Validate everything before creating the result directory.
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "episodes.csv", rows)
    if pairs:
        write_csv(output / "extended_episodes.csv", extended_rows)
        write_csv(output / "paired_episodes.csv", pairs)
    (output / "horizon_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--extended", type=Path)
    parser.add_argument("--tail-env-steps", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.tail_env_steps < 1:
        parser.error("tail-env-steps must be positive")
    analyze(args.baseline, args.output, args.extended, args.tail_env_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
