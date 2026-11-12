"""Build the mixed-outcome plot and extract the four illustrated CDPR events."""
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ral-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def mixed_groups():
    fig, ax = plt.subplots(figsize=(3.37, 1.85))
    p = np.linspace(0, 1, 401)
    for size, color, style in [(2, "#834218", ":"), (4, "#4d5a86", "-."),
                               (8, "#00746e", "--"), (16, "#182b38", "-")]:
        ax.plot(p, 1 - p**size - (1-p)**size, style, label=f"{size}",
                color=color, linewidth=1.4)
    ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Success probability per attempt",
           ylabel="Mixed-group probability", xticks=[0, .25, .5, .75, 1],
           yticks=[0, .5, 1])
    ax.legend(title="Group size", loc="lower center", ncol=4, fontsize=7,
              title_fontsize=7, columnspacing=.65, handlelength=1.3, frameon=False)
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", color="#dee4e3", linewidth=.6)
    fig.tight_layout(pad=.35)
    fig.savefig(ROOT / "figures/mixed_groups.pdf")
    plt.close(fig)


def task_frames():
    media = ROOT.parent / "docs/media/put_into"
    metadata = json.loads((media / "apple_into_plate.json").read_text())
    times = [0] + [metadata["event_seconds"][e] for e in ("grasp", "lift", "release")]
    width, height, gap = 320, 240, 4
    montage = Image.new("RGB", (4 * width + 3 * gap, 2 * height + gap), "white")
    for i, seconds in enumerate(times):
        command = ["ffmpeg", "-loglevel", "error", "-i", str(media / "apple_into_plate.mp4"),
                   "-ss", str(seconds), "-frames:v", "1", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "pipe:1"]
        frame = Image.frombytes("RGB", (2 * width, height), subprocess.check_output(command))
        for view in (0, 1):
            panel = frame.crop((view * width, 0, (view+1)*width, height))
            montage.paste(panel, (i*(width+gap), view*(height+gap)))
    montage.save(ROOT / "figures/apple_sequence.png")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    mixed_groups()
    task_frames()
