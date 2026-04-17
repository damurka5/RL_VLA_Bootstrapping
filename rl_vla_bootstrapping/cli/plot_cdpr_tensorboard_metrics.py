from __future__ import annotations

import argparse
import csv
import json
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


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float
    wall_time: float


_METRIC_SPECS: dict[str, MetricSpec] = {
    "action_saturation_rate": MetricSpec(
        key="action_saturation_rate",
        title="Action saturation rate",
        y_label="Rate",
        tags=("rollout_step/action_saturation_rate_mean",),
        filename_stem="action_saturation_rate",
    ),
    "env_reward_mean": MetricSpec(
        key="env_reward_mean",
        title="Environment reward mean",
        y_label="Reward",
        tags=("rollout_step/reward_env_mean",),
        filename_stem="env_reward_mean",
    ),
    "success_rate": MetricSpec(
        key="success_rate",
        title="Success rate",
        y_label="Rate",
        tags=("rollout_step/success_rate_mean",),
        filename_stem="success_rate",
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
    for metric in metrics:
        metric_points[metric.key] = {}
        metric_tags[metric.key] = {}
        for run in runs:
            tag, points = _load_metric_points(run.path, metric)
            metric_points[metric.key][run.label] = points
            metric_tags[metric.key][run.label] = tag

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

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "output_dir": output_dir.as_posix(),
        "runs": [{"label": run.label, "path": run.path.as_posix()} for run in runs],
        "metrics": [asdict(metric) for metric in metrics],
        "metric_tags": metric_tags,
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
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved plots and CSV to: {output_dir}")
    print(f"Combined figure: {combined_path}")
    print(f"Scalar CSV: {csv_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
