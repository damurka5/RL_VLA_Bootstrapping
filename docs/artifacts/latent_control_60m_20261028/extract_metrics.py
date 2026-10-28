"""Reproduce this evidence bundle from the supplied TensorBoard event file."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.event_file
    accumulator = EventAccumulator(str(source), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    scalars = {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}

    def series(tag):
        entries = scalars[tag]
        by_step = {e.step: e for e in entries}
        # Duplicate scalar names are expected for global counters; retain the
        # last observation at each exact integer Event.step.
        return [by_step[step] for step in sorted(by_step)]

    def dump_csv(filename, tags, steps):
        values = {tag: {e.step: e.value for e in series(tag)} for tag in tags}
        wall_times = {e.step: e.wall_time for e in series(tags[0])}
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["step", "wall_time_utc", *tags])
            for step in steps:
                writer.writerow([step, datetime.fromtimestamp(wall_times[step], timezone.utc).isoformat(),
                                 *(values[tag].get(step, "") for tag in tags)])

    strict = series("validation/three_stage_strict_rate")
    validation_steps = [e.step for e in strict]
    first, final = strict[0], strict[-1]
    peak = max(strict, key=lambda e: e.value)
    training = series("loss_policy_mean")
    training_steps = [e.step for e in training]
    validation_tags = sorted(tag for tag in scalars if tag.startswith("validation/"))
    training_tags = [
        "loss_policy_mean", "entropy_mean", "approx_kl_mean", "clip_fraction_mean",
        "gradient_norm_mean", "log_std_mean", "optimizer_lr_mean", "policy/lora_max_abs_change",
        "policy/trainable_parameter_count", "three_stage/episodes", "three_stage/strict_rate",
        "non_finite_live_episodes", "non_finite_live_episode_rate", "optimizer_steps",
        "training/elapsed_time_s", "pilot/updates", "pilot/episodes", "pilot/optimizer_steps",
        "left_pad_normal_force_mean_n", "right_pad_normal_force_mean_n",
    ]
    dump_csv("validation.csv", validation_tags, validation_steps)
    dump_csv("training.csv", training_tags, training_steps)

    def at(tag, step):
        return next(e.value for e in series(tag) if e.step == step)

    # Verify counts, rates and instruction totals for every retained reading.
    for step in validation_steps:
        episodes = at("validation/three_stage_episodes", step)
        count = at("validation/three_stage_strict_count", step)
        assert math.isclose(count / episodes, at("validation/three_stage_strict_rate", step), abs_tol=1e-7)
        assert episodes == at("validation/episodes", step)
        assert count == sum(at(f"validation/three_stage_{dest}_strict_count", step) for dest in ("plate", "bowl"))
        assert episodes == sum(at(f"validation/three_stage_{dest}_episodes", step) for dest in ("plate", "bowl"))

    def snapshot(entry):
        result = {"step": entry.step,
                  "wall_time_moscow": datetime.fromtimestamp(entry.wall_time, timezone(timedelta(hours=3))).isoformat()}
        for metric in ["episodes", "strict_count", "strict_rate", "physical_grasp_count",
                       "physical_grasp_rate", "physical_held_lift_count", "physical_held_lift_rate",
                       "plate_episodes", "plate_strict_count", "plate_strict_rate", "bowl_episodes",
                       "bowl_strict_count", "bowl_strict_rate", "carry_slip_count", "carry_slip_rate",
                       "wrong_place_count", "wrong_place_rate", "non_finite_count", "non_finite_rate"]:
            result[metric] = at("validation/three_stage_" + metric, entry.step)
        return result

    finite_tags = training_tags[:8]
    for tag in finite_tags:
        assert {e.step for e in series(tag)} == set(training_steps)
        assert all(math.isfinite(e.value) for e in series(tag))
    episode_total = int(sum(e.value for e in series("three_stage/episodes")))
    nonfinite_total = int(sum(e.value for e in series("non_finite_live_episodes")))
    summary = {
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_size_bytes": source.stat().st_size,
        "review_date": "2026-10-28",
        "measurement_type": "in_run_validation_not_separate_checkpoint_evaluation",
        "run_directory": None, "checkpoint_sha256": None,
        "start_step": first.step, "final_step": final.step,
        "new_selected_actions": final.step - first.step,
        "new_updates": len(training_steps),
        "cumulative_updates": int(series("pilot/updates")[-1].value),
        "new_episodes": episode_total,
        "new_optimizer_steps": int(sum(e.value for e in series("optimizer_steps"))),
        "elapsed_hours": series("training/elapsed_time_s")[-1].value / 3600,
        "validation_rows": len(validation_steps),
        "validation_episode_counts": sorted({e.value for e in series("validation/episodes")}),
        "validation_seeds": sorted({e.value for e in series("validation/seed")}),
        "initial": snapshot(first), "peak": snapshot(peak), "final": snapshot(final),
        "first20_validation_strict_mean": statistics.mean(e.value for e in strict[:20]),
        "last20_validation_strict_mean": statistics.mean(e.value for e in strict[-20:]),
        "non_finite_live_episodes": nonfinite_total,
        "non_finite_count_derived_rate": nonfinite_total / episode_total,
        "updates_with_fewer_than_2048_episodes": [e.step for e in series("three_stage/episodes") if e.value < 2048],
        "finite_full_coverage_tags_checked": finite_tags,
        "force_scalar_rows": len(series("left_pad_normal_force_mean_n")),
        "force_missing_rows": len(training_steps) - len(series("left_pad_normal_force_mean_n")),
        "limitations": [
            "The source has no run-directory manifest, checkpoint bytes, scene IDs or per-episode outcomes.",
            "Event.step metadata is exact; float32 global_step values round and are not checkpoint identifiers.",
            "Repeated in-run validation readings and their moving averages are correlated and used during training.",
            "The peak is selected from 81 readings; no matched off-panel gain, confidence interval or promotion is established.",
            "The last separately evaluated development checkpoint remains the 40M control (61.43%, 1258/2048).",
            "Missing force scalars are unavailable diagnostics, not evidence that forces were finite.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cdpr-60m-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    x = [e.step / 1e6 for e in strict]
    axes[0].plot(x, [e.value * 100 for e in strict], color="#2f6fdd", linewidth=1.8, label="Strict completion (1,024 episodes/reading)")
    for metric, label, color in [("physical_grasp_rate", "Physical grasp", "#868e96"),
                                 ("physical_held_lift_rate", "Held lift", "#43a17c")]:
        axes[0].plot(x, [at("validation/three_stage_" + metric, e.step) * 100 for e in strict], color=color, linewidth=1.3, label=label)
    axes[0].scatter([peak.step / 1e6, final.step / 1e6], [peak.value * 100, final.value * 100], color="#2f6fdd", zorder=4)
    axes[0].annotate("Peak 71.09%", (peak.step / 1e6, peak.value * 100), xytext=(-105, 10), textcoords="offset points")
    axes[0].annotate("Final 70.61%", (final.step / 1e6, final.value * 100), xytext=(-70, -19), textcoords="offset points")
    axes[0].set(ylabel="In-run validation (%)", ylim=(55, 100), title="CDPR + SmolVLA: 40M to 60M continuation")
    axes[0].legend(loc="upper left", fontsize=9)
    for metric, label, color in [("carry_slip_rate", "Carry slip", "#d7832f"),
                                 ("wrong_place_rate", "Wrong placement", "#b66a8a"),
                                 ("non_finite_rate", "Numerical failure", "#7f3c8d")]:
        axes[1].plot(x, [at("validation/three_stage_" + metric, e.step) * 100 for e in strict], color=color, linewidth=1.4, label=label)
    axes[1].set(xlabel="Selected actions since latent-likelihood conversion (millions)", ylabel="In-run event rate (%)", ylim=(0, 28))
    axes[1].legend(loc="upper right", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("81 correlated panel readings; separate development evaluation pending.", fontsize=9)
    fig.savefig(args.output_dir / "training_review.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"final_strict_percent": final.value * 100, "peak_strict_percent": peak.value * 100,
                      "validation_rows": len(strict), "training_rows": len(training)}, indent=2))


if __name__ == "__main__":
    main()
