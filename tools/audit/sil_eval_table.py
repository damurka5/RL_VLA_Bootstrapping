#!/usr/bin/env python3
"""Compare two record-mode runs: RL-only against the self-imitation checkpoint.

This is the phase's verdict, and the SFT loss is not it. A lower action MSE
means the residual mimics a smoothed open-loop target more closely; whether
that makes the task succeed more often is a different question, answered only
by running the policy.

The comparison is cheap because the baseline already exists. The harvest that
produced the demonstrations was itself a record-mode run of the RL checkpoint
at the same caps, round indices and seed, so its summaries are a matched
control -- same harness, same resets, no extra GPU time.

Three things this does that a glance at the logs does not.

It pools across rounds on COUNTS, not on rates. Averaging four per-round
percentages weights a round with 152 episodes the same as one with 184, and
the instruction mix varies per round because instructions are sampled per
group.

It prints the per-round spread next to the pooled figure. Two identical runs
of this stack disagree on about 6.6% of episodes -- micron-scale physics noise
amplified through the closed loop -- so a pooled delta smaller than the spread
is not a result.

It refuses to compare runs whose episode counts differ, because that means
the two runs did not draw the same resets and the delta would be measuring
the reset distribution.

Pure stdlib, no GPU::

    python tools/audit/sil_eval_table.py \\
        --baseline tools/audit/out/sil_harvest_0.03 \\
        --candidate tools/audit/out/eval_sft_0.03
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any, Sequence


def _collect(directory: str) -> dict[str, Any]:
    """Pool per-instruction counts over every run_NN in a record summary."""

    paths = sorted(glob.glob(os.path.join(directory, "summary*.json")))
    if not paths:
        raise SystemExit(f"No summary*.json under {directory!r}.")
    totals: dict[str, dict[str, Any]] = {}
    rounds = 0
    for path in paths:
        payload = json.load(open(path, encoding="utf-8"))
        if payload.get("mode") != "record":
            continue
        for key, run in sorted(payload.items()):
            if not key.startswith("run_") or not isinstance(run, dict):
                continue
            rounds += 1
            for name, entry in (run.get("by_instruction") or {}).items():
                bucket = totals.setdefault(
                    name, {"episodes": 0, "successes": 0, "rates": []}
                )
                bucket["episodes"] += int(entry["episodes"])
                bucket["successes"] += int(entry["successes"])
                bucket["rates"].append(float(entry["source_success_rate"]))
    if not totals:
        raise SystemExit(f"No record-mode runs found under {directory!r}.")
    return {"rounds": rounds, "by_instruction": totals}


def _rate(bucket: dict[str, Any]) -> float:
    return (
        bucket["successes"] / bucket["episodes"] if bucket["episodes"] else 0.0
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.066,
        help=(
            "Measured per-episode verdict noise: the fraction that flips "
            "between two identical runs. Used only to widen the reported "
            "uncertainty, never to adjust a rate."
        ),
    )
    args = parser.parse_args(argv)

    base = _collect(args.baseline)
    cand = _collect(args.candidate)

    header = (
        f"{'instruction':<20}{'RL-only':>18}{'SIL-SFT':>18}"
        f"{'delta':>9}{'+-':>8}{'resolved':>10}"
    )
    print(f"baseline  {args.baseline}  ({base['rounds']} rounds)")
    print(f"candidate {args.candidate}  ({cand['rounds']} rounds)")
    print()
    print(header)
    print("-" * len(header))

    for name in sorted(
        set(base["by_instruction"]) | set(cand["by_instruction"])
    ):
        b = base["by_instruction"].get(name)
        c = cand["by_instruction"].get(name)
        if b is None or c is None:
            print(f"{name:<20}{'absent from one side':>63}")
            continue
        if b["episodes"] != c["episodes"]:
            # Different denominators mean different resets, and a delta across
            # them measures the reset distribution rather than the policy.
            print(
                f"{name:<20}{'episode counts differ':>40} "
                f"{b['episodes']} vs {c['episodes']} -- not comparable"
            )
            continue
        rb, rc = _rate(b), _rate(c)
        # Binomial spread on each side, widened by the measured verdict noise.
        # A lower bound on the uncertainty, not a confidence interval: the two
        # runs share resets, so their errors are correlated in ways this does
        # not model.
        var = (
            rb * (1.0 - rb) / max(b["episodes"], 1)
            + rc * (1.0 - rc) / max(c["episodes"], 1)
            + 2.0 * float(args.noise) ** 2 / max(base["rounds"], 1)
        )
        spread = math.sqrt(var)
        delta = rc - rb
        print(
            f"{name:<20}"
            f"{b['successes']:>6}/{b['episodes']:<5}{rb:>6.3f}"
            f"{c['successes']:>6}/{c['episodes']:<5}{rc:>6.3f}"
            f"{delta:>+9.3f}{spread:>8.3f}"
            f"{('yes' if abs(delta) > 2.0 * spread else 'no'):>10}"
        )
        print(
            f"{'':<20}per-round  "
            f"{[round(v, 3) for v in b['rates']]} -> "
            f"{[round(v, 3) for v in c['rates']]}"
        )

    print()
    print(
        "`resolved` is |delta| > 2 sigma with sigma widened by the measured "
        f"{args.noise:.3f} verdict-flip rate. Report these separately from "
        "the RL-only result; they are an extension, not a replacement."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
