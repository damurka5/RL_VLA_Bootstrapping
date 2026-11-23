#!/usr/bin/env python3
"""Plot paired projection effects from pilot and confirmation recordings."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from smolvla_skill_interventions import paired_stats, verify_reset

TASKS = {
    "move_to": ("move_to_object",),
    "placement": ("put_into_bowl", "put_into_plate"),
}
TITLES = {"move_to_object": "Move to", "put_into_bowl": "Bowl", "put_into_plate": "Plate"}
ARMS = {
    "recipient_repeat": "Repeat B",
    "donor_full": "SIL (residual + expert)",
    "donor_residual": "Residual only",
    "all_expert": "All expert LoRA",
    "layer_13_gate_proj": "Layer 13 gate_proj",
    "layer_14_gate_proj": "Layer 14 gate_proj",
    "layer_14_q_proj": "Layer 14 q_proj",
}
DIRECT = "donor_full_vs_donor_residual"


def direct_effect(directory: Path, notes: dict) -> dict:
    from sil_record import _Recording, _instruction_name

    rollout = notes["rollout"]
    group_size = rollout["group_size"]
    chunks, labels = [], []
    for i in range(rollout["round_index"], rollout["round_index"] + rollout["rounds"]):
        filename = f"record_{i:02d}.npz"
        residual = _Recording.from_npz(directory / "donor_residual" / filename)
        full = _Recording.from_npz(directory / "donor_full" / filename)
        if residual.round_index != i or full.round_index != i:
            raise ValueError(f"Unexpected round number in {filename}")
        verify_reset(residual, full)
        ids = residual.instruction_ids.reshape(-1, group_size)
        if not np.equal(ids, ids[:, :1]).all():
            raise ValueError(f"Mixed instructions within reset group in {filename}")
        chunks.append(np.stack((residual.episode_success, full.episode_success), axis=-1)
                      .reshape(-1, group_size, 2).astype(float))
        labels.extend(_instruction_name(i) for i in ids[:, 0])
    outcomes, labels = np.concatenate(chunks), np.asarray(labels)
    return {name: paired_stats(outcomes[labels == name], group_size, rollout["seed"])
            for name in np.unique(labels)}


def table_row(run: str, instruction: str, arm: str, stats: dict) -> dict:
    n = stats["episodes"]
    low, high = stats["cluster_bootstrap_ci95_pp"]
    return {
        "run": run, "instruction": instruction, "arm": arm,
        "reference": "donor_residual" if arm == DIRECT else "recipient",
        "episodes": n, "reset_groups": stats["reset_groups"],
        "reference_successes": round(stats["recipient_success"] * n),
        "arm_successes": round(stats["arm_success"] * n),
        "delta_pp": stats["delta_pp"], "ci95_low_pp": low, "ci95_high_pp": high,
        "rescued": stats["rescued"], "lost": stats["lost"],
    }


def read_run(label: str, directory: Path) -> tuple[list[dict], dict, dict]:
    rows, contracts, rounds = [], {}, {}
    for task, instructions in TASKS.items():
        folder = directory / task
        notes = json.loads((folder / "paired.json").read_text())
        settings = notes["rollout"]
        rounds[task] = set(range(settings["round_index"], settings["round_index"] + settings["rounds"]))
        contracts[task] = {
            "manifest_sha256": notes["manifest_sha256"], "config_sha256": notes["config_sha256"],
            **{key: settings[key] for key in ("cap", "controller_workspace_from_checkpoint",
               "metadata_override", "group_size", "microbatch", "seed", "horizon")},
        }
        for arm in ARMS:
            for instruction in instructions:
                stats = notes["arms"].get(arm, {}).get("instruction", {}).get(instruction)
                if stats:
                    rows.append(table_row(label, instruction, arm, stats))
        if {"donor_full", "donor_residual"} <= notes["arms"].keys():
            for instruction, stats in direct_effect(folder, notes).items():
                rows.append(table_row(label, instruction, DIRECT, stats))
    if not rows:
        raise ValueError(f"No projection results in {directory}")
    return rows, contracts, rounds


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = list(dict.fromkeys(row["run"] for row in rows))
    arms = [key for key in ARMS if any(row["arm"] == key for row in rows)]
    positions = {arm: index for index, arm in enumerate(arms)}
    fig, axes = plt.subplots(1, len(TITLES), figsize=(13, 5), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10").colors
    shifts = np.linspace(-0.18, 0.18, len(runs)) if len(runs) > 1 else [0.0]
    for ax, (instruction, title) in zip(axes, TITLES.items()):
        ax.axvline(0, color="#7e8590", lw=1, zorder=1)
        match = [row for row in rows if row["instruction"] == instruction and row["arm"] in positions]
        for idx, (run, offset) in enumerate(zip(runs, shifts)):
            subset = [row for row in match if row["run"] == run]
            if not subset:
                continue
            y = [positions[row["arm"]] + offset for row in subset]
            color = colors[idx % len(colors)]
            ax.hlines(y, [row["ci95_low_pp"] for row in subset],
                      [row["ci95_high_pp"] for row in subset], color=color, lw=1.5)
            ax.scatter([row["delta_pp"] for row in subset], y, color=color, s=25, zorder=2)
        counts = sorted({(runs.index(row["run"]), row["reset_groups"]) for row in match})
        suffix = " / ".join(str(n) for _, n in counts)
        ax.set_title(f"{title}\nReset groups: {suffix or '–'}")
        ax.grid(axis="x", alpha=0.22)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(arms)), [ARMS[key] for key in arms])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([0], [0], color=colors[i % len(colors)], marker="o", label=run)
               for i, run in enumerate(runs)]
    fig.legend(handles=handles, loc="lower center", ncol=len(runs), frameon=False, bbox_to_anchor=(.5, .075))
    fig.suptitle("CDPR SmolVLA · paired effect of SIL weight transfers", x=.11, ha="left", fontweight="bold")
    fig.supxlabel("Success difference vs. pre-SIL checkpoint (percentage points)", y=.03)
    fig.text(.11, .92, "95% intervals from bootstrap resampling of reset groups", color="#4b525c")
    fig.tight_layout(rect=(0, .14, 1, .88))
    fig.savefig(output / "paired_projection_effects.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, baseline_contract = [], None
    seen = set()
    seen_rounds = {task: set() for task in TASKS}
    for entry in args.run:
        label, sep, directory = entry.partition("=")
        if not sep or not re.fullmatch(r"[\w .-]+", label) or not directory or label in seen:
            parser.error(f"Expected unique LABEL=DIRECTORY: {entry!r}")
        seen.add(label)
        results, contract, rounds = read_run(label, Path(directory))
        if baseline_contract is not None and contract != baseline_contract:
            raise ValueError(f"{label} uses different checkpoint or evaluation settings")
        for task, round_numbers in rounds.items():
            if overlap := seen_rounds[task] & round_numbers:
                raise ValueError(f"{label} shares {task} reset rounds: {sorted(overlap)}")
            seen_rounds[task].update(round_numbers)
        baseline_contract = contract
        rows.extend(results)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "paired_projection_effects.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
