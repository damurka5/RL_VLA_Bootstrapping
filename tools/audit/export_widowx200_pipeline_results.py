#!/usr/bin/env python3
"""Snapshot a running WidowX pipeline's numerical evidence without loading models.

Uses only the Python standard library. Copies logs/configs/reports, never model,
image or trajectory arrays. Training and pipeline state are not changed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any


STAGES = (
    "s1_move_to_dense", "s2_pick_up_dense", "s3_put_into_dense",
    "s4_move_to_binary", "s5_pick_up_binary", "s6_put_into_binary", "s7_joint_staged",
)
RULE_KEYS = (
    "max_train_steps", "early_stop_metric", "early_stop_target", "early_stop_patience",
    "early_stop_min_steps", "early_stop_plateau_validations", "early_stop_min_delta",
    "validation_start_distance_caps", "validation_seed", "validation_episodes_per_instruction",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                              for k, v in row.items()} for row in rows)


def percent(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1%}"


class Snapshot:
    def __init__(self, pipeline: Path, output: Path) -> None:
        self.pipeline = pipeline.resolve()
        self.output = output.resolve()
        self.captured: dict[str, bytes] = {}
        self.sources: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    def capture(self, path: Path, destination: Path) -> bytes | None:
        """Read each source once; tables and archive use exactly those bytes."""
        key = str(path.resolve())
        if key in self.captured:
            return self.captured[key]
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        self.captured[key] = data
        target = self.output / "raw" / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.sources.append({"source": key, "exported": str(target.relative_to(self.output)),
                             "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return data

    def payload(self, path: Path, destination: Path) -> dict[str, Any]:
        data = self.capture(path, destination)
        if data is None:
            return {}
        try:
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value
        except (ValueError, UnicodeDecodeError) as error:
            self.warnings.append(f"Cannot parse {path}: {error}")
            return {}

    def jsonl(self, path: Path, destination: Path) -> list[dict[str, Any]]:
        data = self.capture(path, destination)
        if data is None:
            return []
        rows = []
        for index, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("expected a JSON object")
                rows.append(value)
            except ValueError:
                self.warnings.append(f"Skipped invalid JSONL row {index} in {path}; snapshot may overlap a write")
        return rows

    def stage_tables(self, state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        tables: dict[str, list[dict[str, Any]]] = {
            "stages": [], "validation": [], "validation_slices": [], "training": [], "retention": [],
        }
        for name in STAGES:
            record = state.get("stages", {}).get(name, {})
            rl = Path(record.get("rl_dir") or self.pipeline.parent / f"{self.pipeline.name}__{name}" / "rl")
            raw = Path("stages") / name
            config = self.payload(rl / "config.json", raw / "config.json")
            for path in sorted((rl.parent / "manifests").glob("*.json")):
                self.capture(path, raw / "manifests" / path.name)
            stop = self.payload(rl / "early_stop.json", raw / "early_stop.json")
            validations = self.jsonl(rl / "validation.jsonl", raw / "validation.jsonl")
            metrics = self.jsonl(rl / "metrics.jsonl", raw / "metrics.jsonl")
            last = metrics[-1] if metrics else {}
            tables["stages"].append({
                "stage": name, "status": record.get("status", "pending"),
                "started": record.get("started"), "finished": record.get("finished"),
                "best_rl_metric": record.get("metric"), "best_rl_value": record.get("best_value"),
                "best_rl_step": record.get("best_step"), "best_rl_checkpoint": record.get("best_checkpoint"),
                "handed_off_policy": record.get("policy"),
                "stop_reason": (record.get("early_stop") or stop).get("reason"),
                "last_step": last.get("global_step"), "last_update": last.get("update_index"),
                "sil_cycles_attempted": len(record.get("sil_cycles", [])),
                "sil_cycles_accepted": sum(bool(c.get("accepted")) for c in record.get("sil_cycles", [])),
                "empty_harvest": record.get("empty_harvest", False),
                **{key: config.get(key) for key in RULE_KEYS},
            })
            if record.get("status") != "pending" and record and not config:
                self.warnings.append(f"{name}: missing/unreadable trainer config at {rl / 'config.json'}")
            for index, row in enumerate(validations):
                identity = {"stage": name, "validation_index": index, "global_step": row.get("global_step"),
                            "selected_best_rl": row.get("global_step") == record.get("best_step"),
                            "fixed_start_distance": row.get("validation/fixed_start_distance")}
                tables["validation"].append({**identity, **row})
                for key, value in row.items():
                    for leg in ("validation", "validation_composed"):
                        for kind in ("instruction", "object"):
                            prefix = f"{leg}/by_{kind}/"
                            suffix = "/success_rate"
                            if key.startswith(prefix) and key.endswith(suffix):
                                label = key[len(prefix):-len(suffix)]
                                tables["validation_slices"].append({
                                    **identity, "leg": leg, "slice": kind, "label": label,
                                    "success_rate": value, "episodes": row.get(f"{prefix}{label}/episodes"),
                                    "start_distance_cap": row.get(f"validation/start_distance_cap/{label}")
                                    if kind == "instruction" else None,
                                })
            tables["training"].extend({"stage": name, **row} for row in metrics)
            evaluations = [("before_sil", None, record.get("retention_rates", {})),
                           ("after_sil", None, record.get("retention_rates_after_sil", {}))]
            for index, cycle in enumerate(record.get("sil_cycles", []), 1):
                evaluations.append((f"sil_{index}_candidate", cycle.get("accepted"), cycle.get("candidate_rates", {})))
            for phase, accepted, rates in evaluations:
                for instruction, stats in rates.items():
                    tables["retention"].append({"stage": name, "phase": phase, "accepted": accepted,
                                                "instruction": instruction, **stats})
        return tables

    def report_tables(self) -> dict[str, list[dict[str, Any]]]:
        tables: dict[str, list[dict[str, Any]]] = {"record_rounds": [], "replay_survival": []}
        # No NPZs/checkpoints/images are read. Keep episode CSVs for scene/object analysis.
        reports: dict[Path, dict[str, Any]] = {}
        for folder in ("banks", "eval", "sil"):
            for path in sorted((self.pipeline / folder).rglob("*")):
                if not path.is_file() or path.suffix not in (".json", ".jsonl", ".csv"):
                    continue
                relative = Path("pipeline") / path.relative_to(self.pipeline)
                if path.suffix == ".json":
                    reports[path] = self.payload(path, relative)
                else:
                    self.capture(path, relative)
        for path, report in reports.items():
            if not path.name.startswith("summary"):
                continue
            parent = reports.get(path.parent / "summary.json", {})
            # Only owning run summaries carry run_XX. Never also count pooled totals.
            if report.get("mode") == "record":
                complete = None
                if parent.get("sharded"):
                    children = [reports.get(path.parent / name, {}) for name in parent.get("shard_summaries", [])]
                    expected = {f"run_{i:02d}" for i in range(int(parent.get("round_index", 0)),
                                                           int(parent.get("round_index", 0)) + int(parent.get("rounds", 0)))}
                    found = [key for child in children for key in child if key.startswith("run_")]
                    complete = bool(expected) and not parent.get("failed_shards") and set(found) == expected and len(found) == len(expected)
                for run, entry in report.items():
                    if not run.startswith("run_") or not isinstance(entry, dict):
                        continue
                    for instruction, stats in (entry.get("by_instruction") or {}).items():
                        tables["record_rounds"].append({
                            "source": str(path.relative_to(self.pipeline)), "run": run,
                            "instruction": instruction, "checkpoint": report.get("checkpoint"),
                            "start_distance_cap": report.get("start_distance_cap"),
                            "sharded_request_complete": complete,
                            "diverged_worlds": entry.get("diverged_worlds"), **stats,
                        })
            for instruction, stats in (report.get("replay", {}).get("by_instruction") or {}).items():
                tables["replay_survival"].append({"source": str(path.relative_to(self.pipeline)),
                                                  "instruction": instruction, **stats})
        return tables

    def bank_table(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for skill, bank in state.get("banks", {}).items():
            replays, frames = bank.get("replays", []), bank.get("frames", [])
            markers = [self.payload(Path(path).with_suffix(".complete.json"),
                                   Path("bank_markers") / skill / (Path(path).stem + ".json")) for path in replays]
            for instruction, stats in bank.get("rates", {}).items():
                rows.append({"skill": skill, "instruction": instruction, "stage": bank.get("stage"),
                             "checkpoint": bank.get("checkpoint"), "provenance": bank.get("provenance"),
                             "start_distance_cap": bank.get("cap"), "replay_files": len(replays),
                             "frame_files": len(frames), "replay_files_present": sum(Path(p).is_file() for p in replays),
                             "frame_files_present": sum(Path(p).is_file() for p in frames),
                             "replay_markers_present": sum(bool(m) for m in markers),
                             "filmed_successes_marker_sum_all_instructions": sum(int(m.get("frame_successes", 0)) for m in markers),
                             "source_harvest_rate": stats.get("rate"),
                             "source_harvest_successes": stats.get("successes"),
                             "source_harvest_episodes": stats.get("episodes")})
        return rows


def markdown(state: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# WidowX-200 pipeline evidence snapshot", "",
             "This is an interim snapshot of one continuing training lineage. Best RL validation scores are checkpoint-selection results, not independent final-test estimates.",
             "Placement skill tests can start with an already-held object. Replay survival is conditional on original success; it is not task success.",
             "", "| Stage | Status | Best RL selection metric | Value | Step | Stop | SIL accepted/attempted |",
             "|---|---|---|---:|---:|---|---:|"]
    for row in tables["stages"]:
        lines.append(f"| {row['stage']} | {row['status']} | {row['best_rl_metric'] or '-'} | "
                     f"{percent(row['best_rl_value'])} | {row['best_rl_step'] if row['best_rl_step'] is not None else '-'} | {row['stop_reason'] or '-'} | "
                     f"{row['sil_cycles_accepted']}/{row['sil_cycles_attempted']} |")
    lines += ["", "## Retained banks", "",
              "| Skill | Source stage | Instruction | Source harvest successes/episodes | Replays | Frame files |",
              "|---|---|---|---:|---:|---:|"]
    for row in tables["banks"]:
        lines.append(f"| {row['skill']} | {row['stage']} | {row['instruction']} | "
                     f"{row['source_harvest_successes']}/{row['source_harvest_episodes']} | "
                     f"{row['replay_files']} | {row['frame_files']} |")
    lines += ["", "## Skill evaluations before/after retention SFT", "",
              "| Stage | Evaluation | Candidate accepted | Instruction | Successes/episodes | Rate |",
              "|---|---|---|---|---:|---:|"]
    for row in tables["retention"]:
        accepted = "-" if row["accepted"] is None else str(row["accepted"])
        lines.append(f"| {row['stage']} | {row['phase']} | {accepted} | {row['instruction']} | "
                     f"{row.get('successes', '-')}/{row.get('episodes', '-')} | {percent(row.get('rate'))} |")
    lines += ["", "## Interpretation and files", "",
              "- `stages.csv` separates the best RL checkpoint from the policy handed to the next stage after any accepted SIL.",
              "- `validation.csv` and `validation_slices.csv` contain learning curves, instruction/object slices, episode denominators, and composed legs when available.",
              "- `training.csv` holds the logged rollout/curriculum/throughput diagnostics; training success is not fixed-distance validation success.",
              "- `record_rounds.csv` contains per-round harvest and retention evaluations. Pooled sharded wrappers are not counted again. A false completeness flag identifies a partial request; blank means no sharded-completeness claim.",
              "- `replay_survival.csv` reports survivors among source successes. `banks.csv` inventories the banks retained by the controller. Marker sums cover all instructions in a bank and are not instruction-specific demo counts.",
              "- `raw/` preserves source JSON/JSONL/CSV and configs. `manifest.json` records their paths, SHA-256 digests, capture time and warnings. No checkpoints, images or trajectory arrays are included.",
              "- Successive validations reuse evaluation seeds. Do not pool them as independent tests or treat stages as independent training seeds. Perform a separate fixed-checkpoint evaluation for final reported performance.",
              "- Current automated pipeline: original-label skill retention SFT, then joint staged RL. No automatic relabelled full-task SFT before stage 7.",
              "", f"Controller final report recorded: {state.get('final_report') or 'not yet'}", ""]
    return "\n".join(lines)


def export(pipeline: Path, output: Path) -> Path:
    pipeline, output = pipeline.resolve(), output.resolve()
    if not (pipeline / "pipeline_state.json").is_file():
        raise FileNotFoundError(pipeline / "pipeline_state.json")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = Snapshot(pipeline, output)
    state = snapshot.payload(pipeline / "pipeline_state.json", Path("pipeline/pipeline_state.json"))
    if not state:
        raise ValueError("Cannot export an unreadable pipeline state")
    tables = snapshot.stage_tables(state)
    tables.update(snapshot.report_tables())
    tables["banks"] = snapshot.bank_table(state)
    for name in ("final_report.json", "final_report.md"):
        if (pipeline / name).is_file():
            snapshot.capture(pipeline / name, Path("pipeline") / name)
    # Generated stage YAMLs carry task metadata which the trainer args alone omit.
    repo = Path(__file__).resolve().parents[2]
    for config in sorted((repo / "configs/widowx200_pipeline").glob(f"{pipeline.name}__*.yaml")):
        snapshot.capture(config, Path("generated_configs") / config.name)
    source_config = Path(state.get("config", ""))
    if source_config.is_file():
        snapshot.capture(source_config, Path("source_config_current") / source_config.name)
    try:
        exporter_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                                                  stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        exporter_commit = None
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    report = markdown(state, tables)
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest = {"schema_version": 1, "snapshot_finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "pipeline": str(pipeline), "git_commit_at_pipeline_start": state.get("git_commit_at_start"),
                "exporter_commit": exporter_commit, "live_snapshot_not_globally_atomic": True,
                "source_config_current_may_differ_from_launch": True,
                "rows": {name: len(rows) for name, rows in tables.items()},
                "sources": snapshot.sources, "warnings": snapshot.warnings}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    archive = output.parent / f"{output.name}.zip"
    # Refuse to replace an earlier archive even if its expanded folder was removed.
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, str(Path(output.name) / path.relative_to(output)))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New export directory (must not already exist).")
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or args.pipeline_dir / f"results_snapshot_{stamp}"
    archive = export(args.pipeline_dir, output)
    print((output / "report.md").read_text())
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
