"""Verify the supplied snapshot and reproduce the compact evidence summary.

Run with Python's standard library only. No trajectory, model or GPU is used.
"""

import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent


def stats(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (float, int)) and math.isfinite(r[key])]
    if not values:
        return None
    return {"readings": len(values), "minimum": min(values), "median": statistics.median(values),
            "maximum": max(values), "last": values[-1], "nonzero_readings": sum(v != 0 for v in values)}


def main():
    archive = HERE / "results_snapshot.zip"
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), "Duplicate archive members"
        assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names)
        prefix = names[0].split("/")[0] + "/"

        def payload(name):
            return json.loads(z.read(prefix + name))

        manifest = payload("manifest.json")
        for source in manifest["sources"]:
            content = z.read(prefix + source["exported"])
            assert len(content) == source["bytes"], source["exported"]
            assert hashlib.sha256(content).hexdigest() == source["sha256"], source["exported"]
        state = payload("raw/pipeline/pipeline_state.json")
        summary = {"review_date": "2026-11-04", "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                   "verified_sources": len(manifest["sources"]), "archive_members": len(names),
                   "warnings": manifest["warnings"],
                   "provenance": {k: manifest[k] for k in ("git_commit_at_pipeline_start", "exporter_commit",
                       "snapshot_finished_utc", "live_snapshot_not_globally_atomic", "source_config_current_may_differ_from_launch")},
                   "references": state["references"], "stages": {}, "sft": {}, "retention": {}}
        keys = ("informative_groups", "usable_group_fraction", "group_reward_std_mean", "group_pass_rate_mean",
                "instruction_successes_normal_start/pick_up", "instruction_successes/pick_up",
                "successes_prelifted", "non_finite_live_episode_rate", "vla_lora/records", "vla_lora/grad_norm")
        for name, stage in state["stages"].items():
            cfg = payload(f"raw/stages/{name}/config.json")
            rows = [json.loads(line) for line in z.read(prefix + f"raw/stages/{name}/metrics.jsonl").splitlines() if line.strip()]
            summary["stages"][name] = {
                "best_step": stage["best_step"], "best_value": stage["best_value"],
                "selected_step": stage.get("selected_step", stage["best_step"]),
                "last_logged_step": rows[-1]["global_step"], "updates": len(rows),
                "last_logged_elapsed_hours": rows[-1].get("training/elapsed_time_s", 0) / 3600,
                "configuration": {k: cfg.get(k) for k in ("split_credit_at_grasp", "action_likelihood", "learning_rate",
                    "ppo_epochs", "train_vla_lora", "vla_lora_updates_enabled", "validation_start_distance_caps")},
                "metrics": {k: stats(rows, k) for k in keys if stats(rows, k) is not None},
            }
            summary["retention"][name] = {k: stage[k] for k in ("retention_rates", "retention_rates_after_sil") if k in stage}
            summary["retention"][name]["cycles"] = [
                {k: c.get(k) for k in ("accepted", "reason", "candidate_rates")} for c in stage.get("sil_cycles", [])]
        prep = state["composition_sft"]
        summary["composition"] = {k: prep[k] for k in ("id", "git_commit", "before_skill_rates", "after_skill_rates",
            "before_composed_rates", "after_composed_rates", "dataset_report", "sources")}
        for n in names:
            if n.endswith("/sft_report.json"):
                r = json.loads(z.read(n))
                summary["sft"][n[len(prefix):]] = {k: r.get(k) for k in (
                    "selected", "best_epoch", "best_val_mse", "best_selection_score", "initializer_selection_score",
                    "baseline_untrained", "rows_train", "rows_val", "scene_leakage", "split_by")}
                summary["sft"][n[len(prefix):]]["lora"] = r.get("lora")
            if n.endswith("/refreshed/refresh.json"):
                r = json.loads(z.read(n))
                summary.setdefault("refresh", {})[n[len(prefix):]] = {k: r.get(k) for k in ("frame_join", "resolved_fraction", "rows_in", "rows_out", "integrity")}
        # Retain readable audit tables and state alongside the original archive.
        for name in ("manifest.json", "report.md", "stages.csv", "retention.csv", "banks.csv", "replay_survival.csv"):
            (HERE / name).write_bytes(z.read(prefix + name))
        (HERE / "pipeline_state.json").write_bytes(z.read(prefix + "raw/pipeline/pipeline_state.json"))
        replay_rows = list(csv.DictReader(io.StringIO(z.read(prefix + "replay_survival.csv").decode())))
        summary["replay"] = {"entries": len(replay_rows),
            "recorded_successes": sum(int(r["recorded_successes"]) for r in replay_rows),
            "survived": sum(int(r["survived"]) for r in replay_rows)}
        (HERE / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"Verified {len(manifest['sources'])} source files; wrote audit_summary.json.")


if __name__ == "__main__":
    main()
