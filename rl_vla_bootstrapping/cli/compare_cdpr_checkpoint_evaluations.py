from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DENSE_STAGE_INSTRUCTIONS = (
    "move_left",
    "move_right",
    "move_top",
    "move_bottom",
    "move_up",
    "move_down",
    "move_to_object",
    "open_gripper",
    "close_gripper",
    "rotate_gripper_clockwise",
    "rotate_gripper_counterclockwise",
)

SPARSE_STAGE_INSTRUCTIONS = (
    "move_to_object",
    "grab_object",
    "pick_up",
    "push_left",
    "push_right",
    "push_forward",
    "push_backward",
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_in_front_of_object",
    "move_behind_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
)


@dataclass(frozen=True)
class RateSummary:
    successes: int
    episodes: int
    success_rate: float


@dataclass(frozen=True)
class EvaluationRunSummary:
    label: str
    run_dir: Path
    checkpoint_dir: str
    generated_at: str
    overall: RateSummary
    normal_canonical: RateSummary
    dense_simple: RateSummary
    dense_normal_canonical: RateSummary
    sparse_complex: RateSummary
    invalid_videos: int
    incomplete_video_coverage: int
    reset_retries: int
    simulation_instability_episodes: int


def _as_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _rate(successes: int, episodes: int) -> RateSummary:
    return RateSummary(
        successes=int(successes),
        episodes=int(episodes),
        success_rate=float(successes / episodes) if episodes else 0.0,
    )


def _manifest_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    return path / "validation_manifest.json"


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing validation manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_from_instruction_summaries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest.get("instruction_summaries", []) or []:
        rows.append(
            {
                "instruction_type": item.get("instruction_type", ""),
                "successes": item.get("successes", 0),
                "episodes": item.get("episodes", 0),
                "success_rate": item.get("success_rate", 0.0),
                "mean_reward": item.get("mean_reward", 0.0),
                "mean_steps": item.get("mean_steps", 0.0),
            }
        )
    return rows


def _instruction_rows(run_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _read_csv_rows(run_dir / "instruction_success_rates.csv")
    return rows if rows else _rows_from_instruction_summaries(manifest)


def _normal_canonical_rows(run_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _read_csv_rows(run_dir / "normal_scene_canonical_success_rates.csv")
    if rows:
        return rows
    return list(manifest.get("normal_scene_canonical_summaries", []) or [])


def _weighted_rate(
    rows: list[dict[str, Any]],
    *,
    instruction_types: set[str] | None = None,
) -> RateSummary:
    successes = 0
    episodes = 0
    for row in rows:
        instruction_type = str(row.get("instruction_type", ""))
        if instruction_types is not None and instruction_type not in instruction_types:
            continue
        successes += _as_int(row.get("successes", 0))
        episodes += _as_int(row.get("episodes", 0))
    return _rate(successes, episodes)


def _metric_rate_from_manifest(manifest: dict[str, Any]) -> RateSummary:
    rows = _rows_from_instruction_summaries(manifest)
    if rows:
        return _weighted_rate(rows)
    episode_groups = manifest.get("episodes", {}) or {}
    successes = 0
    episodes = 0
    for items in episode_groups.values():
        for item in items:
            if not bool(item.get("metric_episode", True)):
                continue
            episodes += 1
            successes += int(bool(item.get("success")))
    return _rate(successes, episodes)


def _count_invalid_videos(manifest: dict[str, Any]) -> int:
    return sum(
        1
        for item in manifest.get("video_validation", []) or []
        if not bool(item.get("valid"))
    )


def _count_incomplete_video_coverage(manifest: dict[str, Any]) -> int:
    return sum(
        1
        for item in manifest.get("video_coverage", []) or []
        if not bool(item.get("complete"))
    )


def _summarize_run(label: str, path: Path) -> EvaluationRunSummary:
    manifest = _read_manifest(path)
    run_dir = _manifest_path(path).parent
    instruction_rows = _instruction_rows(run_dir, manifest)
    normal_rows = _normal_canonical_rows(run_dir, manifest)
    dense_types = set(DENSE_STAGE_INSTRUCTIONS)
    sparse_types = set(SPARSE_STAGE_INSTRUCTIONS)
    return EvaluationRunSummary(
        label=label,
        run_dir=run_dir,
        checkpoint_dir=str(manifest.get("checkpoint_dir") or ""),
        generated_at=str(manifest.get("generated_at") or ""),
        overall=_metric_rate_from_manifest(manifest),
        normal_canonical=_weighted_rate(normal_rows),
        dense_simple=_weighted_rate(instruction_rows, instruction_types=dense_types),
        dense_normal_canonical=_weighted_rate(normal_rows, instruction_types=dense_types),
        sparse_complex=_weighted_rate(instruction_rows, instruction_types=sparse_types),
        invalid_videos=_count_invalid_videos(manifest),
        incomplete_video_coverage=_count_incomplete_video_coverage(manifest),
        reset_retries=_as_int(manifest.get("total_reset_retries", 0)),
        simulation_instability_episodes=_as_int(
            manifest.get("simulation_instability_episodes", 0)
        ),
    )


def _parse_labeled_path(raw_value: str) -> tuple[str, Path]:
    if "=" in raw_value:
        label, path = raw_value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in run specification: {raw_value!r}")
        return label, Path(path)
    path = Path(raw_value)
    return path.name or "evaluation", path


def _format_rate(summary: RateSummary) -> str:
    if summary.episodes <= 0:
        return "n/a"
    return f"{summary.successes}/{summary.episodes} ({summary.success_rate:.1%})"


def _markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(str(row.get(column, "")) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _instruction_rate_table(
    run_specs: list[tuple[str, Path]],
    *,
    instruction_types: tuple[str, ...],
) -> list[dict[str, str]]:
    per_run_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in run_specs:
        manifest = _read_manifest(path)
        run_dir = _manifest_path(path).parent
        rows = _instruction_rows(run_dir, manifest)
        per_run_rows[label] = {
            str(row.get("instruction_type", "")): row
            for row in rows
        }

    table_rows: list[dict[str, str]] = []
    for instruction_type in instruction_types:
        row: dict[str, str] = {"instruction_type": instruction_type}
        for label, _path in run_specs:
            run_row = per_run_rows[label].get(instruction_type)
            if run_row is None:
                row[label] = "n/a"
                continue
            rate = _weighted_rate([run_row])
            row[label] = _format_rate(rate)
        table_rows.append(row)
    return table_rows


def _delta_rows(
    summaries: list[EvaluationRunSummary],
    *,
    baseline: EvaluationRunSummary,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in summaries:
        if summary.label == baseline.label:
            continue
        rows.append(
            {
                "checkpoint": summary.label,
                "overall_delta": f"{summary.overall.success_rate - baseline.overall.success_rate:+.1%}",
                "normal_delta": (
                    f"{summary.normal_canonical.success_rate - baseline.normal_canonical.success_rate:+.1%}"
                ),
                "dense_simple_delta": (
                    f"{summary.dense_simple.success_rate - baseline.dense_simple.success_rate:+.1%}"
                ),
                "sparse_complex_delta": (
                    f"{summary.sparse_complex.success_rate - baseline.sparse_complex.success_rate:+.1%}"
                ),
            }
        )
    return rows


def _gate_recommendation(
    summaries: list[EvaluationRunSummary],
    *,
    dense_gate_threshold: float,
) -> str:
    eligible = [
        summary.dense_normal_canonical
        if summary.dense_normal_canonical.episodes
        else summary.dense_simple
        for summary in summaries
    ]
    best = max((item.success_rate for item in eligible), default=0.0)
    if best >= dense_gate_threshold:
        return (
            f"Best dense/simple score is {best:.1%}, meeting the "
            f"{dense_gate_threshold:.0%} gate for sparse-stage experiments."
        )
    return (
        f"Best dense/simple score is {best:.1%}, below the "
        f"{dense_gate_threshold:.0%} gate. Treat sparse-complex training as a "
        "diagnostic probe, not the next main run."
    )


def _render_report_markdown(
    summaries: list[EvaluationRunSummary],
    *,
    run_specs: list[tuple[str, Path]],
    dense_gate_threshold: float,
    baseline_label: str | None = None,
) -> str:
    if not summaries:
        raise ValueError("At least one evaluation summary is required.")
    baseline = next(
        (summary for summary in summaries if summary.label == baseline_label),
        summaries[0],
    )
    overview_rows = [
        {
            "checkpoint": summary.label,
            "overall": _format_rate(summary.overall),
            "normal_canonical": _format_rate(summary.normal_canonical),
            "dense_simple": _format_rate(summary.dense_simple),
            "dense_normal": _format_rate(summary.dense_normal_canonical),
            "sparse_complex": _format_rate(summary.sparse_complex),
            "video_gaps": summary.incomplete_video_coverage,
            "invalid_videos": summary.invalid_videos,
        }
        for summary in summaries
    ]
    dense_rows = _instruction_rate_table(
        run_specs,
        instruction_types=DENSE_STAGE_INSTRUCTIONS,
    )
    sparse_rows = _instruction_rate_table(
        run_specs,
        instruction_types=SPARSE_STAGE_INSTRUCTIONS,
    )
    columns = ("instruction_type", *[label for label, _path in run_specs])
    artifact_rows = [
        {
            "checkpoint": summary.label,
            "checkpoint_dir": summary.checkpoint_dir,
            "run_dir": summary.run_dir.as_posix(),
        }
        for summary in summaries
    ]
    lines = [
        "# CDPR checkpoint comparison",
        "",
        f"Generated at `{datetime.now().isoformat(timespec='seconds')}`.",
        "",
        "## Gate read",
        "",
        _gate_recommendation(
            summaries,
            dense_gate_threshold=float(dense_gate_threshold),
        ),
        "",
        "## Summary",
        "",
        _markdown_table(
            overview_rows,
            (
                "checkpoint",
                "overall",
                "normal_canonical",
                "dense_simple",
                "dense_normal",
                "sparse_complex",
                "video_gaps",
                "invalid_videos",
            ),
        ),
        "",
        f"Baseline for deltas: `{baseline.label}`.",
        "",
        _markdown_table(
            _delta_rows(summaries, baseline=baseline),
            (
                "checkpoint",
                "overall_delta",
                "normal_delta",
                "dense_simple_delta",
                "sparse_complex_delta",
            ),
        ),
        "",
        "## Dense/simple instruction rates",
        "",
        _markdown_table(dense_rows, columns),
        "",
        "## Sparse/complex instruction rates",
        "",
        _markdown_table(sparse_rows, columns),
        "",
        "## Artifacts",
        "",
        _markdown_table(
            artifact_rows,
            ("checkpoint", "checkpoint_dir", "run_dir"),
        ),
        "",
    ]
    return "\n".join(lines)


def _write_summary_csv(output_path: Path, summaries: list[EvaluationRunSummary]) -> None:
    columns = [
        "label",
        "checkpoint_dir",
        "run_dir",
        "overall_successes",
        "overall_episodes",
        "overall_success_rate",
        "normal_canonical_successes",
        "normal_canonical_episodes",
        "normal_canonical_success_rate",
        "dense_simple_successes",
        "dense_simple_episodes",
        "dense_simple_success_rate",
        "dense_normal_canonical_successes",
        "dense_normal_canonical_episodes",
        "dense_normal_canonical_success_rate",
        "sparse_complex_successes",
        "sparse_complex_episodes",
        "sparse_complex_success_rate",
        "invalid_videos",
        "incomplete_video_coverage",
        "reset_retries",
        "simulation_instability_episodes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "label": summary.label,
                    "checkpoint_dir": summary.checkpoint_dir,
                    "run_dir": summary.run_dir.as_posix(),
                    "overall_successes": summary.overall.successes,
                    "overall_episodes": summary.overall.episodes,
                    "overall_success_rate": f"{summary.overall.success_rate:.6f}",
                    "normal_canonical_successes": summary.normal_canonical.successes,
                    "normal_canonical_episodes": summary.normal_canonical.episodes,
                    "normal_canonical_success_rate": (
                        f"{summary.normal_canonical.success_rate:.6f}"
                    ),
                    "dense_simple_successes": summary.dense_simple.successes,
                    "dense_simple_episodes": summary.dense_simple.episodes,
                    "dense_simple_success_rate": f"{summary.dense_simple.success_rate:.6f}",
                    "dense_normal_canonical_successes": (
                        summary.dense_normal_canonical.successes
                    ),
                    "dense_normal_canonical_episodes": (
                        summary.dense_normal_canonical.episodes
                    ),
                    "dense_normal_canonical_success_rate": (
                        f"{summary.dense_normal_canonical.success_rate:.6f}"
                    ),
                    "sparse_complex_successes": summary.sparse_complex.successes,
                    "sparse_complex_episodes": summary.sparse_complex.episodes,
                    "sparse_complex_success_rate": (
                        f"{summary.sparse_complex.success_rate:.6f}"
                    ),
                    "invalid_videos": summary.invalid_videos,
                    "incomplete_video_coverage": summary.incomplete_video_coverage,
                    "reset_retries": summary.reset_retries,
                    "simulation_instability_episodes": (
                        summary.simulation_instability_episodes
                    ),
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare saved CDPR checkpoint validation artifacts."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Evaluation run directories or LABEL=RUN_DIR pairs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for comparison report artifacts. Defaults to the first run dir.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional label to use as the delta baseline. Defaults to the first run.",
    )
    parser.add_argument(
        "--dense-gate-threshold",
        type=float,
        default=0.70,
        help="Dense-stage success threshold used for the sparse-readiness note.",
    )
    parser.add_argument(
        "--report-name",
        default="checkpoint_comparison_report.md",
        help="Markdown report file name.",
    )
    parser.add_argument(
        "--csv-name",
        default="checkpoint_comparison_summary.csv",
        help="CSV summary file name.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_specs = [_parse_labeled_path(raw_value) for raw_value in args.runs]
    summaries = [
        _summarize_run(label, path)
        for label, path in run_specs
    ]
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else summaries[0].run_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _render_report_markdown(
        summaries,
        run_specs=run_specs,
        dense_gate_threshold=float(args.dense_gate_threshold),
        baseline_label=args.baseline,
    )
    report_path = output_dir / str(args.report_name)
    csv_path = output_dir / str(args.csv_name)
    report_path.write_text(report, encoding="utf-8")
    _write_summary_csv(csv_path, summaries)
    print(f"Comparison report: {report_path}")
    print(f"Comparison CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
