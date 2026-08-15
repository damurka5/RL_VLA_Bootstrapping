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


# Two-sided 95% critical values of t, indexed by degrees of freedom. Four
# rounds gives df=3 and a critical value of 3.18, which is the honest cost of
# pairing on so few samples: the test is correct but underpowered, and saying
# so is better than borrowing power from a model that ignores round-to-round
# variance.
_T_CRITICAL = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
               7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def _t_critical(df: int) -> float:
    return _T_CRITICAL.get(int(df), 2.0)


def _collect(directory: str) -> dict[str, Any]:
    """Per-round and pooled counts per instruction, from a record summary."""

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
                    name,
                    {"episodes": 0, "successes": 0, "rates": [],
                     "per_round_episodes": []},
                )
                bucket["episodes"] += int(entry["episodes"])
                bucket["successes"] += int(entry["successes"])
                bucket["rates"].append(float(entry["source_success_rate"]))
                bucket["per_round_episodes"].append(int(entry["episodes"]))
    if not totals:
        raise SystemExit(f"No record-mode runs found under {directory!r}.")
    return {"rounds": rounds, "by_instruction": totals}


def _paired(base_rates: Sequence[float], cand_rates: Sequence[float]) -> dict[str, Any]:
    """Paired difference across rounds -- the correct test for shared resets.

    Round index seeds the reset, so round i of the baseline and round i of the
    candidate are the SAME episodes. Pairing removes the between-round
    variance, which is the dominant noise here: at cap 0.10 the plate rate
    swings 0.466 to 0.690 across rounds while the policy difference is a few
    points.

    The earlier version of this file added a constant derived from the 6.6%
    verdict-flip rate to every comparison. That is wrong at small rates -- it
    put a +-0.047 band around a task whose rate is 0.03, so a collapse from
    0.033 to exactly 0.000 in four independent rounds of 664 episodes read as
    "not resolved" when its probability under the null is about 2e-10.
    """

    n = min(len(base_rates), len(cand_rates))
    deltas = [cand_rates[i] - base_rates[i] for i in range(n)]
    if n < 2:
        return {"n": n, "mean": deltas[0] if deltas else 0.0, "t": None,
                "resolved": False, "all_same_sign": True, "deltas": deltas}
    mean = sum(deltas) / n
    variance = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    stderr = math.sqrt(variance / n) if variance > 0.0 else 0.0
    if stderr == 0.0:
        # Every round moved by the identical amount. Degenerate but decisive
        # when that amount is non-zero.
        t: float | None = None if mean == 0.0 else math.inf
    else:
        t = mean / stderr
    resolved = t is not None and abs(t) > _t_critical(n - 1)
    return {
        "n": n,
        "mean": mean,
        "stderr": stderr,
        "t": t,
        "resolved": resolved,
        "all_same_sign": all(d > 0 for d in deltas)
        or all(d < 0 for d in deltas),
        "deltas": deltas,
    }


def _rate(bucket: dict[str, Any]) -> float:
    return (
        bucket["successes"] / bucket["episodes"] if bucket["episodes"] else 0.0
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)

    base = _collect(args.baseline)
    cand = _collect(args.candidate)

    header = (
        f"{'instruction':<18}{'RL-only':>16}{'SIL-SFT':>16}"
        f"{'delta':>9}{'t':>8}{'signs':>7}{'resolved':>10}"
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
            print(f"{name:<18}{'absent from one side':>58}")
            continue
        if b["per_round_episodes"] != c["per_round_episodes"]:
            # Different denominators mean different resets, and a delta across
            # them measures the reset distribution rather than the policy.
            print(
                f"{name:<18}{'per-round episode counts differ':>50} "
                f"-- not comparable"
            )
            continue
        rb, rc = _rate(b), _rate(c)
        paired = _paired(b["rates"], c["rates"])
        t_text = (
            "inf" if paired["t"] == math.inf
            else ("--" if paired["t"] is None else f"{paired['t']:.2f}")
        )
        print(
            f"{name:<18}"
            f"{b['successes']:>5}/{b['episodes']:<4}{rb:>6.3f}"
            f"{c['successes']:>5}/{c['episodes']:<4}{rc:>6.3f}"
            f"{rc - rb:>+9.3f}{t_text:>8}"
            f"{('all' if paired['all_same_sign'] else 'mixed'):>7}"
            f"{('yes' if paired['resolved'] else 'no'):>10}"
        )
        print(
            f"{'':<18}per-round  "
            f"{[round(v, 3) for v in b['rates']]} -> "
            f"{[round(v, 3) for v in c['rates']]}"
        )
        # A skill that stopped existing, not one that got worse. Keyed on the
        # ratio rather than on exactly zero: at cap 0.01 the candidate scored
        # 1/664 against a 0.066 baseline, and an == 0 test would have stayed
        # silent on a 97% loss because of a single lucky episode.
        if b["successes"] >= 10 and rb > 0.0 and rc <= 0.1 * rb:
            print(
                f"{'':<18}COLLAPSE: {c['successes']}/{c['episodes']} = "
                f"{rc:.3f} against a {rb:.3f} baseline -- "
                f"{rc / rb * 100.0:.0f}% of the original rate."
            )

    print()
    print(
        "Paired by round index, because the round index seeds the reset and "
        "round i of each side is the same episodes. `t` is the paired t "
        f"statistic; `resolved` is |t| > {_t_critical(base['rounds'] - 1):.2f} "
        f"(95%, df={max(base['rounds'] - 1, 1)}). `signs` says whether every "
        "round moved the same way -- with four rounds the test is honest but "
        "underpowered, so a consistent sign at |t| just under the bar means "
        "run more rounds, not that there is no effect."
    )
    print(
        "Report these separately from the RL-only result; they are an "
        "extension, not a replacement."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
