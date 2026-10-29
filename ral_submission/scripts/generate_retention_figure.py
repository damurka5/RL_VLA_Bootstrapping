"""Render the recovery-only Figure 3 from the recorded evaluation-round rates."""

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ral-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INK = "#243447"
MUTED = "#526377"
TEAL = "#147D77"
GRAY = "#E2E7ED"


def main():
    data = json.loads((ROOT / "evidence/skill_retention_rounds.json").read_text())
    recovery = data["recovery"]
    names = list(recovery)
    means = {
        condition: [100 * np.mean(recovery[name][condition]) for name in names]
        for condition in ("before", "after")
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": INK,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.16)
    x = np.arange(len(names))
    width = 0.23
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDE3EB", linewidth=0.6)
    for condition, offset, color, edge, label in (
        ("before", -width / 2, GRAY, MUTED, "Before SIL"),
        ("after", width / 2, TEAL, TEAL, "After SIL"),
    ):
        ax.bar(x + offset, means[condition], width, color=color,
               edgecolor=edge, linewidth=0.8, label=label, zorder=3)
        # Individual evaluation rounds, so the spread behind each mean is visible.
        for index, name in enumerate(names):
            rounds = [100 * v for v in recovery[name][condition]]
            jitter = np.linspace(-width / 5, width / 5, len(rounds))
            ax.scatter(x[index] + offset + jitter, rounds, s=14, color=INK,
                       edgecolor="white", linewidth=0.5, zorder=4,
                       label="Evaluation round" if (condition, index) == ("after", 0) else None)
            ax.text(x[index] + offset, max(max(rounds), means[condition][index]) + 2.5,
                    f"{means[condition][index]:.1f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 20))
    ax.set_ylabel("Skill success (%)", fontsize=10)
    ax.tick_params(axis="both", length=0, pad=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
        ax.spines[spine].set_linewidth(0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3,
              frameon=False, fontsize=10, handlelength=1.5, columnspacing=2.0)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(ROOT / f"figures/skill_retention.{ext}", dpi=240,
                    facecolor="white")
    plt.close(fig)
    print(json.dumps({"skills": names, "means_percent": means,
                      "source": data["source"]}, indent=2))


if __name__ == "__main__":
    main()
