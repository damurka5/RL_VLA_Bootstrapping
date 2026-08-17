#!/usr/bin/env python3
"""Does the demonstration set contain aiming, or one averaged action?

The worry this answers: a policy that emits roughly the same command
regardless of what is in front of it will still produce a dataset, and the
dataset will still train. Distilling a constant teaches a constant.

**The marginal distribution cannot answer it.** A wide spread of actions is
equally consistent with a policy that servos to each object and one that
emits noise around a fixed drift. Histograms and plane projections are worth
looking at, and this draws them, but on their own they are decoration. The
question is conditional: does the command covary with the geometry it is
supposed to be responding to?

So the numbers that decide it are:

``direction_concentration`` -- the length of the mean UNIT command. 1.0 means
every world was commanded the same way in world frame. Context, not a verdict:
where the arm sits systematically to one side of the goals, a genuine servo
scores 0.82 on it, because aiming at them means pointing one way most of the
time. High concentration is only damning when ``aim`` is also near zero.

``aim`` -- the cosine between the commanded XY and the direction to the goal,
minus the same cosine with the COMMANDS permuted across rows. The permutation
keeps both marginal distributions and breaks only the pairing, so it prices
exactly the alignment a policy gets for free from systematic geometry. The
raw cosine is not a verdict on its own: a fixed drift that points where goals
usually sit scores 0.78 on it while knowing nothing. This is the
discriminator; everything else is context.

``cosine_by_object`` -- the same cosine split by object catalog. If the policy
genuinely distinguishes objects, this is roughly flat across them. If one
object carries the whole signal, the "aiming" is a bias toward wherever that
object usually sits.

Calibrated on synthetic policies whose answer is known: ``aim`` reads +0.001
for a pure fixed drift, +0.163 for half servo half drift, +0.327 for a true
servo, -0.004 for uniform noise. Two earlier nulls -- pairing with another
world's goal, and rotating the goal about the arm -- both certified the pure
drift as aiming, and the docstring of ``_aiming`` records why.

Recordings, not the dataset, are the input: geometry is what makes the
question answerable, and ``demonstrations.npz`` stores the pooled vision
feature rather than object positions. Actions are read from the recordings
too, so this describes exactly what was demonstrated.

    python tools/audit/sil_action_stats.py \\
        --recordings 'tools/audit/out/smooth_placement/replay_*.npz' \\
        --output tools/audit/out/action_stats
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.sil_record import (  # noqa: E402
    _Recording,
    _catalog_name,
    _instruction_name,
)
from tools.audit.xy_approach_probe import _cosine, _unit  # noqa: E402

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (  # noqa: E402
    ACTIVE_INSTRUCTION_TYPES,
)

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

AXES = ("x", "y", "z", "yaw", "gripper")


# Instructions whose goal is the REFERENCE object, not the target object. A
# placement episode starts already holding its target, so the end effector is
# on top of it for the whole carry and "direction to the target" is a
# centimetre of gripper slop. The thing being aimed at is the receptacle.
_REFERENCE_GOAL_INSTRUCTIONS = frozenset(
    {
        "put_into_bowl",
        "put_into_plate",
        "move_left_of_object",
        "move_right_of_object",
        "move_between_objects",
    }
)


def _goal_slots(rec: _Recording, mode: str) -> np.ndarray:
    """Which object each world is aiming AT, per instruction."""

    if mode == "object":
        return rec.target_slots
    if mode == "receptacle":
        return rec.reference_slots
    goal = rec.target_slots.copy()
    for name in _REFERENCE_GOAL_INSTRUCTIONS:
        if name not in ACTIVE_INSTRUCTION_TYPES:
            continue
        mask = rec.instruction_ids == ACTIVE_INSTRUCTION_TYPES.index(name)
        valid = mask & (rec.reference_slots >= 0)
        goal[valid] = rec.reference_slots[valid]
    return goal


def _gather(
    paths: Sequence[str], *, successes_only: bool, goal: str
) -> dict[str, np.ndarray]:
    """Flatten every live step of every recording into one table of rows.

    One row per executed env step, carrying the command and the geometry it
    was issued against. Steps a world was not stepped for are dropped: the
    policy keeps emitting into a frozen world after it terminates, and those
    commands moved nothing.
    """

    actions: list[np.ndarray] = []
    relative: list[np.ndarray] = []
    instructions: list[np.ndarray] = []
    catalogs: list[np.ndarray] = []

    for path in paths:
        rec = _Recording.from_npz(Path(path))
        steps, worlds = rec.actions.shape[0], rec.worlds
        rows = np.arange(worlds)
        slots = _goal_slots(rec, goal)
        target = rec.object_xyz[:, rows, slots, :]  # [S, W, 3]
        rel = target - rec.ee_xyz
        live = rec.active.copy()
        if successes_only:
            live &= rec.episode_success[None, :]
        # An episode's last steps carry it onto the target, where the
        # direction is dominated by the residual offset rather than by any
        # approach. Excluded so the cosine is not diluted by arrival.
        far = np.linalg.norm(rel[..., :2], axis=-1) > 0.01
        live &= far

        actions.append(rec.actions[live])
        relative.append(rel[live])
        instructions.append(
            np.broadcast_to(rec.instruction_ids[None, :], (steps, worlds))[live]
        )
        catalogs.append(
            np.broadcast_to(
                (
                    rec.target_catalog_ids
                    if rec.target_catalog_ids is not None
                    else np.full((worlds,), -1, dtype=np.int64)
                )[None, :],
                (steps, worlds),
            )[live]
        )

    if not actions or not sum(a.shape[0] for a in actions):
        raise SystemExit("No live steps found in the given recordings.")
    return {
        "action": np.concatenate(actions).astype(np.float64),
        "relative": np.concatenate(relative).astype(np.float64),
        "instruction_id": np.concatenate(instructions),
        "catalog_id": np.concatenate(catalogs),
    }


def _aiming(
    action: np.ndarray,
    relative: np.ndarray,
    rng: np.random.Generator,
    permutations: int = 8,
) -> dict[str, Any]:
    """The conditional statistics. These are what decide the question.

    The null permutes the COMMANDS across rows, keeping every row's own
    geometry. That preserves both marginal distributions exactly -- the same
    commands, the same goal directions -- and breaks only the pairing between
    them, which is the thing being tested.

    Two earlier nulls in this file were wrong and are gone, because the way
    they were wrong is the lesson.

    Pairing each command with ANOTHER world's goal seemed like the object
    discrimination test, and is not: the arm tracks its own goal, so its own
    goal direction is short and variable while another world's is long and
    dominated by wherever the arm happens to sit. A pure fixed drift carrying
    no goal information at all scored 0.383 on it -- indistinguishable from
    the real recordings, which is how it nearly certified them.

    Rotating the goal direction about the arm has expectation exactly zero by
    symmetry, which sounds ideal and is useless: it destroys the systematic
    geometry along with the pairing, so a fixed drift that happens to point
    where goals usually are still scored 0.78 against it.

    On synthetic policies with known answers, the command permutation reads
    +0.001 for a pure drift, +0.163 for half servo half drift, +0.327 for a
    true servo and -0.004 for uniform noise.
    """

    command_xy = action[:, :2]
    goal_xy = relative[:, :2]
    mean_vector = command_xy.mean(axis=0)
    spread = float(np.linalg.norm(command_xy - mean_vector, axis=-1).mean())
    observed = float(np.mean(_cosine(command_xy, goal_xy)))
    rows = command_xy.shape[0]
    nulls = [
        float(np.mean(_cosine(command_xy[rng.permutation(rows)], goal_xy)))
        for _ in range(max(1, int(permutations)))
    ]
    null = float(np.mean(nulls))
    return {
        "rows": int(rows),
        "command_mean_vector": [round(float(v), 5) for v in mean_vector],
        "command_mean_norm": round(float(np.linalg.norm(mean_vector)), 5),
        "command_spread": round(spread, 5),
        # 1.0 = one direction for every world regardless of its goal.
        "direction_concentration": round(
            float(np.linalg.norm(_unit(command_xy).mean(axis=0))), 5
        ),
        # Not a verdict on its own: a fixed drift pointing where goals usually
        # sit scores high here while knowing nothing.
        "target_cosine": round(observed, 5),
        "permuted_cosine": round(null, 5),
        "permutation_spread": round(
            float(np.std(nulls)) if len(nulls) > 1 else 0.0, 5
        ),
        # The discriminator: alignment that survives breaking the pairing.
        "aim": round(observed - null, 5),
    }


def _per_axis(action: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for index, name in enumerate(AXES):
        column = action[:, index]
        stats[name] = {
            "mean": round(float(column.mean()), 5),
            "std": round(float(column.std()), 5),
            "p05": round(float(np.percentile(column, 5)), 5),
            "p50": round(float(np.percentile(column, 50)), 5),
            "p95": round(float(np.percentile(column, 95)), 5),
            "saturated_fraction": round(
                float(np.mean(np.abs(column) > 0.99)), 5
            ),
        }
    return stats


def _text_histogram(values: np.ndarray, *, bins: int = 21, width: int = 46) -> list[str]:
    counts, edges = np.histogram(values, bins=bins, range=(-1.0, 1.0))
    peak = max(int(counts.max()), 1)
    lines = []
    for count, low in zip(counts, edges[:-1]):
        bar = "#" * int(round(width * count / peak))
        lines.append(f"  {low:+.2f} {bar:<{width}} {int(count)}")
    return lines


def _plot(
    table: Mapping[str, np.ndarray], output: Path, label: str
) -> list[str]:
    """Per-axis histograms and the three plane projections."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - optional dependency
        print(f"[actions] plots skipped ({error})")
        return []

    action = table["action"]
    written: list[str] = []

    figure, axes = plt.subplots(1, 5, figsize=(20, 3.4))
    for index, name in enumerate(AXES):
        axes[index].hist(action[:, index], bins=61, range=(-1, 1), color="#3b6ea5")
        axes[index].set_title(f"{name}")
        axes[index].set_xlim(-1, 1)
    figure.suptitle(f"{label}: commanded action per axis ({action.shape[0]} steps)")
    figure.tight_layout()
    path = output / f"{label}_axes.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(str(path))

    planes = (("x", "y", 0, 1), ("z", "y", 2, 1), ("z", "x", 2, 0))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for position, (first, second, i, j) in enumerate(planes):
        axes[position].hexbin(
            action[:, i], action[:, j], gridsize=45,
            extent=(-1, 1, -1, 1), bins="log", cmap="viridis",
        )
        axes[position].set_xlabel(first)
        axes[position].set_ylabel(second)
        axes[position].set_title(f"{first}-{second}")
        # A constant-action policy collapses to a point at the crosshair; a
        # servo fills the square.
        axes[position].axhline(0.0, color="w", lw=0.5, alpha=0.4)
        axes[position].axvline(0.0, color="w", lw=0.5, alpha=0.4)
    figure.suptitle(f"{label}: action density by plane (log counts)")
    figure.tight_layout()
    path = output / f"{label}_planes.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(str(path))

    figure = plt.figure(figsize=(6.5, 5.5))
    ax = figure.add_subplot(111, projection="3d")
    sample = action
    if sample.shape[0] > 20000:
        pick = np.random.default_rng(0).choice(
            sample.shape[0], 20000, replace=False
        )
        sample = sample[pick]
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1.5, alpha=0.12,
               c="#3b6ea5", linewidths=0)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_title(f"{label}: xyz command cloud")
    figure.tight_layout()
    path = output / f"{label}_xyz.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(str(path))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--recordings", required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--successes-only",
        action="store_true",
        help=(
            "Restrict to episodes that succeeded, which is what the dataset "
            "keeps. Without it the answer describes the policy; with it, the "
            "demonstrations."
        ),
    )
    parser.add_argument(
        "--goal",
        choices=("auto", "object", "receptacle"),
        default="auto",
        help=(
            "What each world is aiming at. `auto` uses the reference object "
            "for put_into_* and the target object otherwise, because a "
            "placement episode starts already HOLDING its target: the end "
            "effector sits on it for the whole carry, so a cosine against it "
            "measures gripper slop and post-release retreat rather than "
            "aiming. The thing being aimed at there is the receptacle."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    paths: list[str] = []
    for pattern in args.recordings:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    if not paths:
        raise SystemExit("No recordings matched.")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260817)
    table = _gather(
        paths, successes_only=bool(args.successes_only), goal=str(args.goal)
    )
    report: dict[str, Any] = {
        "recordings": paths,
        "successes_only": bool(args.successes_only),
        "goal": str(args.goal),
        "overall": {
            **_aiming(table["action"], table["relative"], rng),
            "per_axis": _per_axis(table["action"]),
        },
    }

    by_instruction: dict[str, Any] = {}
    for instruction_id in sorted(set(table["instruction_id"].tolist())):
        mask = table["instruction_id"] == instruction_id
        if int(mask.sum()) < 50:
            continue
        name = _instruction_name(instruction_id)
        entry = _aiming(table["action"][mask], table["relative"][mask], rng)
        entry["per_axis"] = _per_axis(table["action"][mask])
        by_object: dict[str, Any] = {}
        for catalog_id in sorted(set(table["catalog_id"][mask].tolist())):
            if int(catalog_id) < 0:
                continue
            sub = mask & (table["catalog_id"] == catalog_id)
            if int(sub.sum()) < 50:
                continue
            by_object[_catalog_name(catalog_id)] = _aiming(
                table["action"][sub], table["relative"][sub], rng
            )
        entry["by_object"] = by_object
        by_instruction[name] = entry
    report["by_instruction"] = by_instruction

    if not args.no_plots:
        plots = _plot(table, output, "all")
        for instruction_id in sorted(set(table["instruction_id"].tolist())):
            name = _instruction_name(instruction_id)
            if name not in by_instruction:
                continue
            mask = table["instruction_id"] == instruction_id
            plots.extend(_plot({"action": table["action"][mask]}, output, name))
        report["plots"] = plots

    (output / "action_stats.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    overall = report["overall"]
    print(f"[actions] {overall['rows']} live approach steps")
    print(
        f"[actions] command mean {overall['command_mean_vector']} "
        f"|mean|={overall['command_mean_norm']} spread={overall['command_spread']}"
    )
    print(
        f"[actions] direction_concentration={overall['direction_concentration']} "
        "(1.0 = one fixed direction for every world)"
    )
    print(
        f"[actions] target_cosine={overall['target_cosine']} vs "
        f"permutation null {overall['permuted_cosine']} "
        f"-> aim {overall['aim']} (+-{overall['permutation_spread']})"
    )
    print()
    header = (
        f"{'slice':<26}{'rows':>8}{'|mean|':>8}{'spread':>8}"
        f"{'conc':>7}{'cos':>8}{'null':>8}{'aim':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, entry in sorted(by_instruction.items()):
        print(
            f"{name:<26}{entry['rows']:>8}{entry['command_mean_norm']:>8.3f}"
            f"{entry['command_spread']:>8.3f}{entry['direction_concentration']:>7.3f}"
            f"{entry['target_cosine']:>8.3f}{entry['permuted_cosine']:>8.3f}"
            f"{entry['aim']:>8.3f}"
        )
        for catalog, sub in sorted(entry["by_object"].items()):
            print(
                f"    {catalog:<22}{sub['rows']:>8}{sub['command_mean_norm']:>8.3f}"
                f"{sub['command_spread']:>8.3f}{sub['direction_concentration']:>7.3f}"
                f"{sub['target_cosine']:>8.3f}{sub['permuted_cosine']:>8.3f}"
                f"{sub['aim']:>8.3f}"
            )
    print()
    print("commanded x histogram (all rows):")
    for line in _text_histogram(table["action"][:, 0]):
        print(line)
    print(f"[actions] wrote {output / 'action_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
