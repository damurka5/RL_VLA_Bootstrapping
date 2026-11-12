"""Rebuild the skill-retention panels from the two simulation records."""
import csv
import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ral-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ("Reach", "Pickup", "Plate", "Bowl")


def cdpr_stages():
    rounds = json.loads((ROOT / "evidence/skill_retention_rounds.json").read_text())
    forget, recover = rounds["forgetting"], rounds["recovery"]
    # Dedicated sources and cycle 3 are in the instruction-level result index
    # in ../CDPR_CONSOLIDATED_PROGRESS_REPORT.md. They use separate protocols.
    stages = {"Skill references": [np.mean(forget["Reaching reference"]), .260, .710, .320],
              "After placement RL": [np.mean(forget["After placement RL"]), None, None, None]}
    for label, phase in [("Before SIL", "before"), ("After SIL", "after")]:
        stages[label] = [np.mean(recover["Reaching"][phase]), None,
                         np.mean(recover["Plate placement"][phase]),
                         np.mean(recover["Bowl placement"][phase])]
    # The separate Phase 5 pickup-seed evaluation (CDPR_PHASE5_REPORT.md, §3)
    # measures pickup on the preceding checkpoint and then after adding its bank.
    stages["After SIL"][1] = 0.0
    stages["Add pickup to SIL"] = [.3203, .0964, .5736, .2721]
    stages["Later shared policy"] = [.4779, .1191, .7383, .5353]
    return stages


def widowx_stages():
    path = ROOT.parent / "docs/artifacts/widowx200_snapshot_20261104/retention.csv"
    with path.open() as f:
        rates = {(r["stage"], r["phase"], r["instruction"]):
                 int(r["successes"]) / int(r["episodes"]) for r in csv.DictReader(f)}
    stages = [("Dense pickup", "s2_pick_up_dense", "before_sil"),
              ("Dense placement", "s3_put_into_dense", "before_sil"),
              ("SIL trial (rejected)", "s3_put_into_dense", "sil_1_candidate"),
              ("Binary reach", "s4_move_to_binary", "after_sil"),
              ("Binary pickup", "s5_pick_up_binary", "before_sil"),
              ("After SIL", "s5_pick_up_binary", "after_sil")]
    return {label: [rates.get((stage, phase, skill))
                    for skill in ("move_to_object", "pick_up", "put_into_plate", "put_into_bowl")]
            for label, stage, phase in stages}


def draw_panel(ax, stages, title):
    values = 100 * np.array(list(stages.values()), dtype=float)
    cmap = LinearSegmentedColormap.from_list("rates", ["#f5faf9", "#9fcfc4", "#005b59"])
    cmap.set_bad("#f0f0ef")
    ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    for row, col in np.ndindex(values.shape):
        v = values[row, col]
        label = "–" if np.isnan(v) else str(Decimal(str(round(v, 7))).quantize(
            Decimal(".1"), rounding=ROUND_HALF_UP))
        ax.text(col, row, label, ha="center", va="center", fontsize=8,
                color="white" if v >= 70 else "#172c34")
    ax.set_xticks(range(len(SKILLS)), SKILLS, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(stages)), list(stages), fontsize=8)
    ax.set_xticks(np.arange(.5, len(SKILLS)), minor=True)
    ax.set_yticks(np.arange(.5, len(stages)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="both", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)


if __name__ == "__main__":
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.16))
    fig.subplots_adjust(left=.158, right=.994, top=.73, bottom=.035, wspace=.52)
    draw_panel(axes[0], cdpr_stages(), "(a) Cable-driven robot")
    axes[0].axhline(1.5, color="#354e55", linewidth=1)
    draw_panel(axes[1], widowx_stages(), "(b) WidowX-200")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(ROOT / f"figures/skill_retention.{ext}", dpi=240)
    plt.close(fig)
