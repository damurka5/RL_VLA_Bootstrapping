#!/usr/bin/env python3
"""Tabulate a smoothing sweep from the summaries ``sil_record`` wrote.

One survival rate is not a comparison. The question part 3 asks is which
filter buys the most smoothness for the least damage, and that needs every
arm of the sweep on one page with both columns visible.

Survival alone cannot answer it: the identity filter survives perfectly and
smooths nothing, so it would win. Reduction alone cannot either: a filter
wide enough to flatten the trajectory scores beautifully and destroys the
episodes. ``ratio`` is reduction bought per point of survival lost, which is
the trade the choice actually turns on.

``diverged`` is here because it is a different failure from "stopped
succeeding": a smoothed command can drive a world into the cable singularity
and out of the simulation entirely, and that world is not a demonstration
that got worse, it is one that stopped existing.

Pure stdlib, no GPU. Run it wherever the summaries are::

    python tools/audit/sil_sweep_table.py
    python tools/audit/sil_sweep_table.py --glob 'tools/audit/out/smooth_*/summary_*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Sequence

DEFAULT_GLOB = "tools/audit/out/sweep_*/summary_*.json"


def _rows(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"[sweep] skipping {path}: {error}")
            continue
        if payload.get("mode") != "replay":
            continue
        smoothing = payload.get("smoothing") or {}
        replay = payload.get("replay") or {}
        divergence = replay.get("divergence") or {}
        survival = replay.get("survival_rate")
        reduction = smoothing.get("step_delta_reduction")
        method = smoothing.get("method")
        if reduction is None or survival is None:
            ratio: Any = None
        elif survival >= 1.0:
            # Nothing was lost. Infinite trade for a real filter, and a
            # meaningless 0 for the identity one, which is the honest
            # distinction: `none` surviving at 1.0 is the control passing,
            # not a method winning.
            ratio = "inf" if reduction > 0.0 else 0.0
        else:
            ratio = round(reduction / (1.0 - survival), 2)
        rows.append(
            {
                "run": os.path.basename(os.path.dirname(path)),
                "source": os.path.basename(str(payload.get("source", ""))),
                "method": method,
                "param": (
                    smoothing.get("alpha")
                    if method == "ema"
                    else smoothing.get("window")
                ),
                "channels": smoothing.get("channels"),
                "before": smoothing.get("step_delta_before"),
                "after": smoothing.get("step_delta_after"),
                "reduction": reduction,
                "survival": survival,
                "kept": f"{replay.get('survived')}/"
                f"{replay.get('recorded_successes')}",
                "diverged": payload.get("replay_diverged_worlds"),
                "ee_active": divergence.get("max_ee_delta_m_active"),
                "ratio": ratio,
                "by_instruction": replay.get("by_instruction") or {},
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default=DEFAULT_GLOB)
    parser.add_argument(
        "--per-instruction",
        action="store_true",
        help=(
            "Break survival out by instruction. Worth it whenever a sweep "
            "mixes task families: placement clears 0.057/0.091 m radii while "
            "pick_up needs roughly 2 cm of grasp precision, so one filter can "
            "be free for one and fatal for the other."
        ),
    )
    args = parser.parse_args(argv)

    rows = _rows(sorted(glob.glob(args.glob)))
    if not rows:
        print(f"[sweep] no replay summaries matched {args.glob!r}")
        return 1

    header = (
        f"{'run':<32}{'method':<15}{'p':>6}{'step delta':>20}"
        f"{'red':>8}{'surv':>8}{'kept':>12}{'div':>5}{'ratio':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(
        rows, key=lambda item: (str(item["run"]), str(item["method"]))
    ):
        delta = f"{row['before']}->{row['after']}"
        print(
            f"{row['run']:<32}{str(row['method']):<15}{str(row['param']):>6}"
            f"{delta:>20}{str(row['reduction']):>8}"
            f"{str(row['survival']):>8}{row['kept']:>12}"
            f"{str(row['diverged']):>5}{str(row['ratio']):>8}"
        )
        if args.per_instruction:
            for name, entry in sorted(row["by_instruction"].items()):
                print(
                    f"    {name:<22}"
                    f"{entry.get('survived')}/{entry.get('recorded_successes')}"
                    f" = {entry.get('survival_rate')}"
                )
    print()
    print(
        "ratio = step-delta reduction per point of survival lost. `none` "
        "reads 0.0 by construction: it is the control, not a contender."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
