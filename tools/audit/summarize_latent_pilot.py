#!/usr/bin/env python3
"""Read-only, standard-library report of latent pilot artifacts (no GPU needed).

No arguments lists available runs; explicit directories print update/validation
tables. Completion is distinct from numerical health and experiment promotion.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARCHITECTURES = {
    "candidate": "frozen_reference_logit_correction_v1",
    "control": "bounded_residual_v0",
}
METRICS = {
    "update": "update_index",
    "selected": "pilot/selected_environment_actions",
    "KL": "approx_kl_mean",
    "PPO_clip": "clip_fraction_mean",
    "grad": "gradient_norm_mean",
    "hidden_grad": "correction/grad_norm_hidden_mean",
    "corr_z": "policy_components/abs_correction_z_mean",
    "corr_grip": "policy_components/abs_correction_gripper_mean",
    "latent_clip_z": "latent/clip_fraction_z",
    "log_std": "log_std_mean",
    "nonfinite_ep": "non_finite_live_episode_rate",
    "pickup": "three_stage/pickup_rate",
    "placement": "three_stage/placement_rate",
    "ref_change": "correction/reference_max_abs_change",
    "lora_change": "policy/lora_max_abs_change",
}


def read_objects(path: Path, issues: list[str], *, jsonl: bool = False) -> list[dict]:
    if not path.is_file():
        issues.append(f"missing {path.name}")
        return []
    texts = path.read_text(encoding="utf-8").splitlines() if jsonl else [path.read_text(encoding="utf-8")]
    objects = []
    for line, text in enumerate(texts, 1):
        if not text.strip():
            continue
        try:
            item = json.loads(text)
            if not isinstance(item, dict):
                raise ValueError("expected a JSON object")
            objects.append(item)
        except ValueError as error:
            issues.append(f"{path.name}:{line}: {error}")
    return objects


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def fmt(value) -> str:
    return f"{value:.6g}" if isinstance(value, (int, float)) else "-" if value is None else str(value)


def summarize(run: Path, expect_updates: int | None = None) -> dict:
    if run.name == "rl":
        run = run.parent
    issues: list[str] = []
    launch = next(iter(read_objects(run / "launch_provenance.json", issues)), {})
    protocol_name = "latent_pilot_protocol_resume.json" if launch.get("init_mode") == "resume" else "latent_pilot_protocol.json"
    protocol = next(iter(read_objects(run / "rl" / protocol_name, issues)), {})
    metrics = read_objects(run / "rl" / "metrics.jsonl", issues, jsonl=True)
    validation = read_objects(run / "rl" / "validation.jsonl", issues, jsonl=True)
    # Older runs have no exit record. A missing record does not prove failure.
    result_path = run / "launch_result.json"
    result = next(iter(read_objects(result_path, issues)), {}) if result_path.exists() else {}
    arm = launch.get("arm", "unknown")
    if arm not in ARCHITECTURES:
        issues.append("unknown arm; cannot check architecture")
    elif protocol.get("policy_architecture") != ARCHITECTURES[arm]:
        issues.append("arm and resolved architecture disagree")
    for key in ("policy_architecture", "action_likelihood", "max_train_steps", "ppo_epochs"):
        if launch.get(key) != protocol.get(key):
            issues.append(f"launch/resolved mismatch: {key}")
    if protocol.get("action_likelihood") != "latent_gaussian_conditional_offset_v1":
        issues.append("resolved likelihood is not the latent conditional likelihood")
    if launch.get("max_updates") != protocol.get("mjwarp_max_updates"):
        issues.append("launch/resolved mismatch: max_updates")

    last = metrics[-1] if metrics else {}
    selected = last.get("pilot/selected_environment_actions")
    start_updates = protocol.get("pilot_counters_at_start", {}).get("updates", 0)
    total_updates = last.get("pilot/updates")
    updates = total_updates - start_updates if finite(total_updates) and finite(start_updates) else None
    max_updates = launch.get("max_updates")
    max_steps = launch.get("max_train_steps")
    reasons = []
    if finite(selected) and finite(max_steps) and selected >= max_steps:
        reasons.append("selected-action cap")
    if finite(updates) and finite(max_updates) and max_updates > 0 and updates >= max_updates:
        reasons.append("update cap")
    status = "limit reached: " + ", ".join(reasons) if reasons else "incomplete / still running / interrupted"
    if last.get("three_stage/stopped_for_zero_signal", 0):
        status = "stopped for zero signal"
        issues.append("zero-signal stop; inspect training before extending")
    if result:
        if result.get("train_exit_code") != 0 or result.get("log_exit_code") != 0:
            status = "launcher failed"
            issues.append(f"exit codes: train={result.get('train_exit_code')} log={result.get('log_exit_code')}")
        elif last.get("three_stage/stopped_for_zero_signal", 0):
            pass  # Keep the zero-signal diagnosis even if a budget also expired.
        elif reasons:
            status = "completed: " + ", ".join(reasons)
        else:
            issues.append("launcher exited successfully before a configured limit")
    if not metrics:
        issues.append("no completed update rows (written after update, validation and checkpoint)")
    if expect_updates is not None and (updates is None or updates < expect_updates):
        issues.append(f"expected {expect_updates} updates in this invocation; found {fmt(updates)} (the action cap may stop it earlier)")
    if updates is not None and updates != len(metrics):
        issues.append(f"counter says {fmt(updates)} updates but file has {len(metrics)} rows")

    required = ["global_step", "update_index", "pilot/updates", "pilot/selected_environment_actions",
                "loss_policy_mean", "approx_kl_mean", "clip_fraction_mean", "gradient_norm_mean",
                "log_std_mean", "policy/lora_max_abs_change", "non_finite_live_episode_rate"]
    if arm == "candidate":
        required += ["correction/reference_max_abs_change", "correction/grad_norm_final_mean",
                     "correction/grad_norm_hidden_mean"]
    for index, row in enumerate(metrics, 1):
        missing = [key for key in required if key not in row]
        if missing:
            issues.append(f"row {index} missing: {', '.join(missing)}")
        nonnumeric = [key for key in required if key in row and not isinstance(row[key], (int, float))]
        if nonnumeric:
            issues.append(f"row {index} non-numeric metrics: {', '.join(nonnumeric)}")
        bad = [key for key, value in row.items() if isinstance(value, (float, int)) and not finite(value)]
        if bad:
            issues.append(f"row {index} non-finite metrics: {', '.join(bad)}")
        for key in ("correction/reference_max_abs_change", "policy/lora_max_abs_change"):
            if key in row and row[key] != 0:
                issues.append(f"row {index} frozen weights changed: {key}={row[key]}")
        if row.get("global_step") != row.get("pilot/selected_environment_actions"):
            issues.append(f"row {index} global_step disagrees with pilot selected actions")
    if reasons and (not validation or validation[-1].get("global_step") != last.get("global_step")):
        issues.append("no validation row at the final training step")
    for index, row in enumerate(validation, 1):
        if not finite(row.get("validation/success_rate")):
            issues.append(f"validation row {index} lacks a finite success rate")
    return dict(run=str(run), arm=arm, launch=launch, protocol=protocol, metrics=metrics,
                validation=validation, status=status, updates=updates, selected=selected,
                issues=issues, exit_record=bool(result))


def print_report(report: dict, *, brief: bool = False) -> None:
    launch = report["launch"]
    print(f"\n{report['run']}\n  arm={report['arm']} updates={fmt(report['updates'])}/{fmt(launch.get('max_updates'))} "
          f"selected={fmt(report['selected'])}/{fmt(launch.get('max_train_steps'))} | {report['status']}")
    if brief:
        return
    print(f"  init={launch.get('init_mode')} source_sha256={launch.get('checkpoint_sha256')} git={launch.get('git_commit')}")
    print(f"  architecture={report['protocol'].get('policy_architecture')} "
          f"LR={report['protocol'].get('optimizer', {}).get('lr_resolved')}")
    if not report["exit_record"]:
        print("  No launcher exit record; limits are inferred from saved metrics, not process state.")
    print(" | ".join(METRICS))
    for row in report["metrics"]:
        print(" | ".join(fmt(row.get(key)) for key in METRICS.values()))
    print("validation step | strict_success | episodes")
    for row in report["validation"]:
        print(" | ".join(fmt(row.get(key)) for key in ("global_step", "validation/success_rate", "validation/episodes")))
    if report["metrics"]:
        last = report["metrics"][-1]
        print("  cumulative pilot counters: " + ", ".join(
            f"{key}={fmt(last.get('pilot/' + key))}" for key in
            ("sampled_environment_actions", "episodes", "optimizer_steps", "wall_time_s")))
    for issue in report["issues"]:
        print(f"  CHECK: {issue}")
    print("  Review KL, clipping, correction/gradient trends, simulator non-finite episode rates and validation before extending.")
    print("  This report does not establish a success-rate gain or approve promotion.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="run directories (or their rl directories)")
    parser.add_argument("--list", action="store_true", help="list runs under --runs-root; no automatic run selection")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--expect-updates", type=int, help="flag fewer completed updates in this invocation")
    args = parser.parse_args()
    brief = args.list or not args.runs
    runs = args.runs or sorted(path.parent for path in args.runs_root.glob("*/launch_provenance.json"))
    reports = []
    for run in runs:
        try:
            report = summarize(run, args.expect_updates)
        except (OSError, TypeError, AttributeError) as error:
            print(f"{run}: cannot read report: {error}")
            return 2
        if brief and report["arm"] not in ARCHITECTURES:
            continue
        reports.append(report)
        print_report(report, brief=brief)
    if not reports:
        print("No latent pilot runs found. Pass the actual training run directory, not a preflight directory.")
        return 2
    return int(not brief and any(report["issues"] for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
