#!/usr/bin/env python3
"""Paired comparison of two put_into evaluations run on the same scenes.

Two checkpoints scored on identical scenes should be compared scene by scene.
Only the scenes where they DISAGREE carry evidence about which is better; the
exact McNemar test asks whether those disagreements split more unevenly than a
fair coin would. Comparing the two rates as independent samples throws that
pairing away and needs a much larger gap to say the same thing.

Per-scene outcomes come from ``results.episodes`` in ``evaluation.json`` when the
evaluator wrote them. Evaluations made before that field existed are read from
``videos/videos.json`` instead, which is complete for ``strict`` only when the
run kept every strict episode (``--video-outcome strict --max-videos 0``); the
tool checks the kept count against the reported strict count and refuses a
partial index.

Usage::

    python3 tools/audit/compare_put_into_evaluations.py BASELINE_DIR CANDIDATE_DIR
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def mcnemar_exact(baseline_only: int, candidate_only: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""

    total = int(baseline_only) + int(candidate_only)
    if total == 0:
        return 1.0
    low = min(int(baseline_only), int(candidate_only))
    tail = sum(math.comb(total, k) for k in range(low + 1)) / 2.0 ** total
    return min(1.0, 2.0 * tail)


def _scene_outcomes(directory: Path, metric: str) -> dict[str, bool]:
    report = json.loads((directory / "evaluation.json").read_text("utf-8"))
    results = report["results"]
    episodes = results.get("episodes")
    if episodes:
        outcomes = {row["scene_uid"]: bool(row[metric]) for row in episodes}
        if len(outcomes) != len(episodes):
            raise SystemExit(f"{directory}: scenes repeat; this tool needs distinct scenes.")
        return outcomes
    videos = report.get("videos") or {}
    if metric != "strict" or videos.get("outcome_filter") != "strict":
        raise SystemExit(
            f"{directory}: no per-scene outcomes; the video index can only "
            "reconstruct `strict`, and only from a strict-filtered run."
        )
    kept = json.loads((directory / "videos" / "videos.json").read_text("utf-8"))
    expected = round(float(results["strict"]["rate"]) * int(results["chains"]))
    if len(kept) != expected:
        raise SystemExit(
            f"{directory}: the video index holds {len(kept)} strict episodes but "
            f"the report implies {expected}; it is not a complete record."
        )
    if int(results["scenes"]) != int(results["chains"]):
        raise SystemExit(f"{directory}: scenes repeat; this tool needs distinct scenes.")
    # The scene list itself is not in older reports; the successes are, and a
    # scene absent from them failed. The caller aligns on the union of both
    # runs' scenes plus the reported scene count.
    return {row["scene_uid"]: True for row in kept}


def compare(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    *,
    scenes: int | None = None,
) -> dict[str, Any]:
    both_have_all = scenes is None
    if both_have_all:
        if set(baseline) != set(candidate):
            raise SystemExit("The two evaluations did not run the same scenes.")
        keys: Sequence[str] = sorted(baseline)
    else:
        keys = sorted(set(baseline) | set(candidate))
    base = [bool(baseline.get(key, False)) for key in keys]
    cand = [bool(candidate.get(key, False)) for key in keys]
    total = len(keys) if scenes is None else int(scenes)
    both = sum(b and c for b, c in zip(base, cand))
    baseline_only = sum(b and not c for b, c in zip(base, cand))
    candidate_only = sum(c and not b for b, c in zip(base, cand))
    return {
        "scenes": total,
        "baseline_successes": both + baseline_only,
        "candidate_successes": both + candidate_only,
        "both": both,
        "neither": total - both - baseline_only - candidate_only,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "difference": (candidate_only - baseline_only) / max(1, total),
        "mcnemar_exact_p": round(mcnemar_exact(baseline_only, candidate_only), 5),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--metric", default="strict")
    args = parser.parse_args(argv)
    baseline_dir = args.baseline.expanduser().resolve()
    candidate_dir = args.candidate.expanduser().resolve()
    reports = [
        json.loads((path / "evaluation.json").read_text("utf-8"))
        for path in (baseline_dir, candidate_dir)
    ]
    for key in ("scene_manifest_sha256", "split", "worlds", "rounds", "distinct_scene_rounds", "decisions"):
        if reports[0].get(key) != reports[1].get(key):
            raise SystemExit(f"Protocols differ on {key}: {reports[0].get(key)!r} vs {reports[1].get(key)!r}.")
    baseline = _scene_outcomes(baseline_dir, args.metric)
    candidate = _scene_outcomes(candidate_dir, args.metric)
    complete = bool(reports[0]["results"].get("episodes")) and bool(reports[1]["results"].get("episodes"))
    result = compare(
        baseline,
        candidate,
        scenes=None if complete else int(reports[0]["results"]["chains"]),
    )
    result["metric"] = args.metric
    result["baseline"] = reports[0]["checkpoint"]
    result["candidate"] = reports[1]["checkpoint"]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
