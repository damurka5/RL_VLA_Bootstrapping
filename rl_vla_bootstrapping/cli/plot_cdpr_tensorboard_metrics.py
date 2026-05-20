from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from rl_vla_bootstrapping.core.commands import ensure_directory

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:  # pragma: no cover - optional runtime dependency
    EventAccumulator = None

try:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional runtime dependency
    plt = None


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    y_label: str
    tags: tuple[str, ...]
    filename_stem: str
    preferred_direction: str
    analysis_note: str


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float
    wall_time: float


@dataclass(frozen=True)
class MetricSummary:
    point_count: int
    step_start: int | None
    step_end: int | None
    first_value: float | None
    last_value: float | None
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    start_window_mean: float | None
    end_window_mean: float | None
    delta: float | None
    window_points: int
    trend: str


_METRIC_SPECS: dict[str, MetricSpec] = {
    "action_saturation_rate": MetricSpec(
        key="action_saturation_rate",
        title="Action saturation rate",
        y_label="Rate",
        tags=("rollout_step/action_saturation_rate_mean",),
        filename_stem="action_saturation_rate",
        preferred_direction="lower",
        analysis_note=(
            "Lower values are usually better here because they suggest the policy is spending "
            "less time at action limits or in clipped control regimes."
        ),
    ),
    "env_reward_mean": MetricSpec(
        key="env_reward_mean",
        title="Environment reward mean",
        y_label="Reward",
        tags=("rollout_step/reward_env_mean",),
        filename_stem="env_reward_mean",
        preferred_direction="higher",
        analysis_note=(
            "Higher values are better when reward shaping is aligned with task progress."
        ),
    ),
    "success_rate": MetricSpec(
        key="success_rate",
        title="Success rate",
        y_label="Rate",
        tags=("rollout_step/success_rate_mean",),
        filename_stem="success_rate",
        preferred_direction="higher",
        analysis_note=(
            "Higher values are better because they reflect more successful rollouts in the "
            "logging window."
        ),
    ),
}


_METRIC_ALIASES: dict[str, str] = {
    "action_saturation": "action_saturation_rate",
    "action_saturation_rate": "action_saturation_rate",
    "action_daturation": "action_saturation_rate",
    "action_daturation_rate": "action_saturation_rate",
    "saturation": "action_saturation_rate",
    "saturation_rate": "action_saturation_rate",
    "env_reward": "env_reward_mean",
    "env_reward_mean": "env_reward_mean",
    "reward_env": "env_reward_mean",
    "reward_env_mean": "env_reward_mean",
    "success": "success_rate",
    "success_rate": "success_rate",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract rollout TensorBoard scalars from PPO and GRPO runs, then save comparison "
            "graphs and CSV exports."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Additional run in the form LABEL=/path/to/tensorboard_or_run_dir. May be repeated.",
    )
    parser.add_argument("--ppo-dir", default=None, help="Optional PPO TensorBoard or run directory.")
    parser.add_argument("--grpo-dir", default=None, help="Optional GRPO TensorBoard or run directory.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["action_saturation_rate", "env_reward_mean", "success_rate"],
        help=(
            "Metrics to export. Supported names include `action_saturation_rate`, "
            "`env_reward_mean`, `success_rate`. The saturation metric also accepts the typo "
            "`action_daturation_rate`."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Directory where plots and CSV files will be written.")
    parser.add_argument(
        "--title-prefix",
        default="CDPR Training Metrics",
        help="Prefix used in plot titles.",
    )
    return parser


def _normalize_name(raw_value: str) -> str:
    return str(raw_value).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_run_spec(raw_value: str) -> RunSpec:
    if "=" not in str(raw_value):
        raise ValueError(f"Expected LABEL=PATH for --run, got {raw_value!r}")
    raw_label, raw_path = str(raw_value).split("=", 1)
    label = str(raw_label).strip()
    if not label:
        raise ValueError(f"Run label is empty in {raw_value!r}")
    path = Path(raw_path).expanduser().resolve()
    return RunSpec(label=label, path=path)


def _resolve_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    runs: list[RunSpec] = []
    seen_labels: set[str] = set()

    def _append(label: str, raw_path: str | None) -> None:
        if not raw_path:
            return
        if label in seen_labels:
            raise ValueError(f"Duplicate run label: {label}")
        seen_labels.add(label)
        runs.append(RunSpec(label=label, path=Path(raw_path).expanduser().resolve()))

    _append("ppo", args.ppo_dir)
    _append("grpo", args.grpo_dir)
    for raw_value in args.run:
        spec = _parse_run_spec(raw_value)
        if spec.label in seen_labels:
            raise ValueError(f"Duplicate run label: {spec.label}")
        seen_labels.add(spec.label)
        runs.append(spec)

    if not runs:
        raise ValueError("Provide at least one run via --ppo-dir, --grpo-dir, or --run LABEL=PATH.")
    return runs


def _resolve_metric_specs(raw_metrics: list[str]) -> list[MetricSpec]:
    resolved: list[MetricSpec] = []
    seen: set[str] = set()
    for raw_metric in raw_metrics:
        normalized = _normalize_name(raw_metric)
        key = _METRIC_ALIASES.get(normalized, normalized)
        if key not in _METRIC_SPECS:
            supported = ", ".join(sorted(set(_METRIC_ALIASES) | set(_METRIC_SPECS)))
            raise ValueError(f"Unsupported metric {raw_metric!r}. Supported names: {supported}")
        if key in seen:
            continue
        seen.add(key)
        resolved.append(_METRIC_SPECS[key])
    if not resolved:
        raise ValueError("Metric selection removed every metric.")
    return resolved


def _find_event_files(root: Path) -> list[Path]:
    path = Path(root).expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"TensorBoard path does not exist: {path}")

    event_files = sorted(
        candidate
        for candidate in path.rglob("events.out.tfevents*")
        if candidate.is_file()
    )
    if event_files:
        return event_files

    # Fall back to any direct files if the caller points at a single event log with a non-standard name.
    direct_files = sorted(candidate for candidate in path.iterdir() if candidate.is_file())
    return direct_files


def _dedupe_scalar_points(points: list[ScalarPoint]) -> list[ScalarPoint]:
    deduped: dict[int, ScalarPoint] = {}
    for point in sorted(points, key=lambda item: (int(item.step), float(item.wall_time))):
        deduped[int(point.step)] = point
    return [deduped[step] for step in sorted(deduped)]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _analysis_window_len(point_count: int) -> int:
    if point_count <= 0:
        return 0
    if point_count == 1:
        return 1
    return min(max(1, point_count // 2), max(1, max(5, int(math.ceil(point_count * 0.1)))))


def _trend_threshold(metric: MetricSpec) -> float:
    if metric.key == "env_reward_mean":
        return 0.1
    return 0.01


def _summarize_metric_points(metric: MetricSpec, points: list[ScalarPoint]) -> MetricSummary:
    if not points:
        return MetricSummary(
            point_count=0,
            step_start=None,
            step_end=None,
            first_value=None,
            last_value=None,
            min_value=None,
            max_value=None,
            mean_value=None,
            start_window_mean=None,
            end_window_mean=None,
            delta=None,
            window_points=0,
            trend="no_data",
        )

    values = [float(point.value) for point in points]
    window_points = _analysis_window_len(len(values))
    start_window_mean = _mean(values[:window_points])
    end_window_mean = _mean(values[-window_points:])
    delta = None
    trend = "stable"
    if start_window_mean is not None and end_window_mean is not None:
        delta = float(end_window_mean - start_window_mean)
        threshold = _trend_threshold(metric)
        if abs(delta) < threshold:
            trend = "stable"
        else:
            improved = delta > 0.0 if metric.preferred_direction == "higher" else delta < 0.0
            trend = "improving" if improved else "worsening"

    return MetricSummary(
        point_count=len(points),
        step_start=int(points[0].step),
        step_end=int(points[-1].step),
        first_value=float(values[0]),
        last_value=float(values[-1]),
        min_value=float(min(values)),
        max_value=float(max(values)),
        mean_value=_mean(values),
        start_window_mean=start_window_mean,
        end_window_mean=end_window_mean,
        delta=delta,
        window_points=window_points,
        trend=trend,
    )


def _format_float(value: float | None, precision: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{precision}f}"


def _improvement_score(metric: MetricSpec, summary: MetricSummary) -> float | None:
    if summary.delta is None:
        return None
    return float(summary.delta) if metric.preferred_direction == "higher" else float(-summary.delta)


def _build_metric_observations(
    metric: MetricSpec,
    runs: list[RunSpec],
    summaries: dict[str, MetricSummary],
) -> list[str]:
    valid = [(run, summaries.get(run.label)) for run in runs]
    valid = [(run, summary) for run, summary in valid if summary is not None and summary.point_count > 0]
    if not valid:
        return ["No scalar data was found for this metric in the supplied TensorBoard logs."]

    reverse_final = metric.preferred_direction == "higher"
    best_final_run, best_final_summary = sorted(
        valid,
        key=lambda item: float(item[1].last_value if item[1].last_value is not None else 0.0),
        reverse=reverse_final,
    )[0]
    direction_label = "highest" if reverse_final else "lowest"
    observations = [
        (
            f"`{best_final_run.label}` finishes with the {direction_label} final "
            f"{metric.title.lower()} ({_format_float(best_final_summary.last_value)})."
        )
    ]

    if len(valid) == 1:
        run, summary = valid[0]
        if summary.trend == "no_data":
            return observations
        observations.append(
            (
                f"`{run.label}` trends `{summary.trend}` over the last analysis window: "
                f"{_format_float(summary.start_window_mean)} -> {_format_float(summary.end_window_mean)} "
                f"across {summary.window_points} points."
            )
        )
        return observations

    scored = []
    for run, summary in valid:
        score = _improvement_score(metric, summary)
        if score is None:
            continue
        scored.append((run, summary, score))

    threshold = _trend_threshold(metric)
    if scored:
        best_delta_run, best_delta_summary, best_score = max(scored, key=lambda item: item[2])
        if best_score >= threshold:
            observations.append(
                (
                    f"`{best_delta_run.label}` shows the strongest windowed improvement: "
                    f"{_format_float(best_delta_summary.start_window_mean)} -> "
                    f"{_format_float(best_delta_summary.end_window_mean)}."
                )
            )

        worst_delta_run, worst_delta_summary, worst_score = min(scored, key=lambda item: item[2])
        if worst_score <= -threshold:
            observations.append(
                (
                    f"`{worst_delta_run.label}` moves in the wrong direction over the same window: "
                    f"{_format_float(worst_delta_summary.start_window_mean)} -> "
                    f"{_format_float(worst_delta_summary.end_window_mean)}."
                )
            )

    if metric.key == "success_rate":
        peak_summary = max(valid, key=lambda item: float(item[1].max_value if item[1].max_value is not None else 0.0))
        peak_run, peak_values = peak_summary
        if (peak_values.max_value or 0.0) < 0.05:
            observations.append("All runs stay below a 5% logged success rate, so task completion remains rare.")
        elif (
            peak_values.max_value is not None
            and len(valid) > 1
            and all(
                other_summary.max_value is not None and peak_values.max_value >= other_summary.max_value
                for _, other_summary in valid
            )
        ):
            observations.append(
                (
                    f"`{peak_run.label}` is the only run to reach a clearly non-trivial peak success rate "
                    f"({_format_float(peak_values.max_value)})."
                )
            )

    return observations[:3]


def _build_overall_observations(
    metrics: list[MetricSpec],
    runs: list[RunSpec],
    metric_summaries: dict[str, dict[str, MetricSummary]],
) -> list[str]:
    observations: list[str] = []
    best_runs: dict[str, tuple[RunSpec, MetricSummary]] = {}

    for metric in metrics:
        valid = []
        for run in runs:
            summary = metric_summaries.get(metric.key, {}).get(run.label)
            if summary is None or summary.point_count <= 0 or summary.last_value is None:
                continue
            valid.append((run, summary))
        if not valid:
            continue
        reverse_final = metric.preferred_direction == "higher"
        best_runs[metric.key] = sorted(
            valid,
            key=lambda item: float(item[1].last_value if item[1].last_value is not None else 0.0),
            reverse=reverse_final,
        )[0]

    reward_best = best_runs.get("env_reward_mean")
    success_best = best_runs.get("success_rate")
    saturation_best = best_runs.get("action_saturation_rate")

    if reward_best is not None:
        run, summary = reward_best
        observations.append(
            f"Best final environment reward mean: `{run.label}` at {_format_float(summary.last_value)}."
        )
    if success_best is not None:
        run, summary = success_best
        observations.append(f"Best final success rate: `{run.label}` at {_format_float(summary.last_value)}.")
    if saturation_best is not None:
        run, summary = saturation_best
        observations.append(
            f"Lowest final action saturation rate: `{run.label}` at {_format_float(summary.last_value)}."
        )

    if reward_best is not None and success_best is not None and reward_best[0].label != success_best[0].label:
        observations.append(
            (
                f"Reward leadership and success leadership split across runs: reward is best in "
                f"`{reward_best[0].label}`, while success is best in `{success_best[0].label}`. "
                "That usually means the shaped reward is not perfectly aligned with completed tasks."
            )
        )

    return observations[:4]


def _render_report_markdown(
    *,
    runs: list[RunSpec],
    metrics: list[MetricSpec],
    metric_summaries: dict[str, dict[str, MetricSummary]],
    metric_tags: dict[str, dict[str, str | None]],
    output_dir: Path,
    combined_path: Path | None,
) -> str:
    lines: list[str] = [
        "# TensorBoard Metrics Report",
        "",
        f"Generated at: {datetime.now().isoformat()}",
        f"Output directory: `{output_dir.as_posix()}`",
        "",
        "## Runs",
        "",
        "| Run | TensorBoard path |",
        "| --- | --- |",
    ]
    for run in runs:
        lines.append(f"| `{run.label}` | `{run.path.as_posix()}` |")

    overall_observations = _build_overall_observations(metrics, runs, metric_summaries)
    if overall_observations:
        lines.extend(["", "## Overall Takeaways", ""])
        for observation in overall_observations:
            lines.append(f"- {observation}")

    lines.extend(["", "## Plot Files", "", "| Metric | Plot |", "| --- | --- |"])
    for metric in metrics:
        lines.append(
            f"| `{metric.key}` | `{(output_dir / f'{metric.filename_stem}.png').as_posix()}` |"
        )
    if combined_path is not None:
        lines.append(f"| `combined` | `{combined_path.as_posix()}` |")

    for metric in metrics:
        lines.extend(
            [
                "",
                f"## {metric.title}",
                "",
                metric.analysis_note,
                "",
                "| Run | Tag | Points | Step range | First | Last | Mean | Min | Max | Start window | End window | Delta | Trend |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for run in runs:
            summary = metric_summaries.get(metric.key, {}).get(run.label)
            tag = metric_tags.get(metric.key, {}).get(run.label) or ""
            if summary is None:
                summary = _summarize_metric_points(metric, [])
            step_range = ""
            if summary.step_start is not None and summary.step_end is not None:
                step_range = f"{summary.step_start} -> {summary.step_end}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{run.label}`",
                        f"`{tag}`" if tag else "",
                        str(summary.point_count),
                        step_range,
                        _format_float(summary.first_value),
                        _format_float(summary.last_value),
                        _format_float(summary.mean_value),
                        _format_float(summary.min_value),
                        _format_float(summary.max_value),
                        _format_float(summary.start_window_mean),
                        _format_float(summary.end_window_mean),
                        _format_float(summary.delta),
                        summary.trend,
                    ]
                )
                + " |"
            )
        lines.extend(["", "Observations:"])
        for observation in _build_metric_observations(
            metric, runs, metric_summaries.get(metric.key, {})
        ):
            lines.append(f"- {observation}")

    lines.append("")
    return "\n".join(lines)


def _load_scalar_points(path: Path, tag: str) -> list[ScalarPoint]:
    if EventAccumulator is None:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "TensorBoard is required to read event files. Install `tensorboard` in the remote environment."
        )

    points: list[ScalarPoint] = []
    for event_file in _find_event_files(path):
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
        except Exception:
            continue
        scalar_tags = set(accumulator.Tags().get("scalars", []))
        if tag not in scalar_tags:
            continue
        for event in accumulator.Scalars(tag):
            points.append(
                ScalarPoint(
                    step=int(event.step),
                    value=float(event.value),
                    wall_time=float(event.wall_time),
                )
            )
    return _dedupe_scalar_points(points)


def _load_metric_points(path: Path, metric: MetricSpec) -> tuple[str | None, list[ScalarPoint]]:
    for tag in metric.tags:
        points = _load_scalar_points(path, tag)
        if points:
            return tag, points
    return None, []


def _plot_metric(
    *,
    metric: MetricSpec,
    runs: list[RunSpec],
    run_points: dict[str, list[ScalarPoint]],
    output_path: Path,
    title_prefix: str,
) -> None:
    if plt is None:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("matplotlib is required to create plots in the remote environment.")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    plotted = False
    for run in runs:
        points = run_points.get(run.label, [])
        if not points:
            continue
        ax.plot(
            [point.step for point in points],
            [point.value for point in points],
            label=run.label,
            linewidth=2.0,
        )
        plotted = True

    ax.set_title(f"{title_prefix}: {metric.title}")
    ax.set_xlabel("Global step")
    ax.set_ylabel(metric.y_label)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No scalar data found", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_figure(
    *,
    metrics: list[MetricSpec],
    runs: list[RunSpec],
    metric_points: dict[str, dict[str, list[ScalarPoint]]],
    output_path: Path,
    title_prefix: str,
) -> None:
    if plt is None:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("matplotlib is required to create plots in the remote environment.")

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, max(4, 3.6 * len(metrics))), squeeze=False)
    for row_idx, metric in enumerate(metrics):
        ax = axes[row_idx][0]
        plotted = False
        for run in runs:
            points = metric_points.get(metric.key, {}).get(run.label, [])
            if not points:
                continue
            ax.plot(
                [point.step for point in points],
                [point.value for point in points],
                label=run.label,
                linewidth=2.0,
            )
            plotted = True
        ax.set_title(metric.title)
        ax.set_xlabel("Global step")
        ax.set_ylabel(metric.y_label)
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No scalar data found", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(title_prefix)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_scalar_csv(
    output_path: Path,
    *,
    runs: list[RunSpec],
    metrics: list[MetricSpec],
    metric_points: dict[str, dict[str, list[ScalarPoint]]],
    metric_tags: dict[str, dict[str, str | None]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_label",
                "run_path",
                "metric_key",
                "metric_title",
                "tag",
                "step",
                "value",
                "wall_time",
            ]
        )
        for metric in metrics:
            for run in runs:
                tag = metric_tags.get(metric.key, {}).get(run.label)
                for point in metric_points.get(metric.key, {}).get(run.label, []):
                    writer.writerow(
                        [
                            run.label,
                            run.path.as_posix(),
                            metric.key,
                            metric.title,
                            tag or "",
                            int(point.step),
                            f"{point.value:.10f}",
                            f"{point.wall_time:.6f}",
                        ]
                        )


def _write_report(
    output_path: Path,
    *,
    runs: list[RunSpec],
    metrics: list[MetricSpec],
    metric_summaries: dict[str, dict[str, MetricSummary]],
    metric_tags: dict[str, dict[str, str | None]],
    output_dir: Path,
    combined_path: Path | None,
) -> None:
    output_path.write_text(
        _render_report_markdown(
            runs=runs,
            metrics=metrics,
            metric_summaries=metric_summaries,
            metric_tags=metric_tags,
            output_dir=output_dir,
            combined_path=combined_path,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    runs = _resolve_run_specs(args)
    metrics = _resolve_metric_specs(args.metrics)
    output_dir = (
        ensure_directory(Path(args.output_dir).expanduser().resolve())
        if args.output_dir
        else ensure_directory(Path.cwd() / f"tensorboard_metric_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )

    metric_points: dict[str, dict[str, list[ScalarPoint]]] = {}
    metric_tags: dict[str, dict[str, str | None]] = {}
    metric_summaries: dict[str, dict[str, MetricSummary]] = {}
    for metric in metrics:
        metric_points[metric.key] = {}
        metric_tags[metric.key] = {}
        metric_summaries[metric.key] = {}
        for run in runs:
            tag, points = _load_metric_points(run.path, metric)
            metric_points[metric.key][run.label] = points
            metric_tags[metric.key][run.label] = tag
            metric_summaries[metric.key][run.label] = _summarize_metric_points(metric, points)

    for metric in metrics:
        _plot_metric(
            metric=metric,
            runs=runs,
            run_points=metric_points[metric.key],
            output_path=output_dir / f"{metric.filename_stem}.png",
            title_prefix=str(args.title_prefix),
        )

    combined_path = output_dir / "combined_metrics.png"
    _plot_combined_figure(
        metrics=metrics,
        runs=runs,
        metric_points=metric_points,
        output_path=combined_path,
        title_prefix=str(args.title_prefix),
    )

    csv_path = output_dir / "tensorboard_scalars.csv"
    _write_scalar_csv(
        csv_path,
        runs=runs,
        metrics=metrics,
        metric_points=metric_points,
        metric_tags=metric_tags,
    )
    report_path = output_dir / "tensorboard_metrics_report.md"
    _write_report(
        report_path,
        runs=runs,
        metrics=metrics,
        metric_summaries=metric_summaries,
        metric_tags=metric_tags,
        output_dir=output_dir,
        combined_path=combined_path,
    )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "output_dir": output_dir.as_posix(),
        "runs": [{"label": run.label, "path": run.path.as_posix()} for run in runs],
        "metrics": [asdict(metric) for metric in metrics],
        "metric_tags": metric_tags,
        "metric_summaries": {
            metric.key: {
                run.label: asdict(metric_summaries.get(metric.key, {}).get(run.label))
                for run in runs
            }
            for metric in metrics
        },
        "points_per_run": {
            metric.key: {
                run.label: len(metric_points.get(metric.key, {}).get(run.label, []))
                for run in runs
            }
            for metric in metrics
        },
        "plots": {
            "combined": combined_path.as_posix(),
            "per_metric": {
                metric.key: (output_dir / f"{metric.filename_stem}.png").as_posix()
                for metric in metrics
            },
        },
        "csv_path": csv_path.as_posix(),
        "report_path": report_path.as_posix(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved plots and CSV to: {output_dir}")
    print(f"Combined figure: {combined_path}")
    print(f"Scalar CSV: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
