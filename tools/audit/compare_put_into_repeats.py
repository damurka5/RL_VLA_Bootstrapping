#!/usr/bin/env python3
"""Compare put_into checkpoints over REPEATED evaluations on the same scenes.

One 256-scene evaluation of one checkpoint moves by up to 5.5 pp between runs
and flips 25-29% of scene verdicts (closed-loop chaos, 2026-09-28), so a single
paired McNemar cannot resolve the differences that now matter. Each checkpoint
is therefore evaluated R times on one fixed scene set, and the unit of analysis
is the SCENE:

* per scene, each checkpoint's success fraction over its repeats;
* the effect is the mean over scenes of the per-scene difference;
* its 95% interval resamples scenes (scene-clustered bootstrap), so repeats of
  one scene never count as independent evidence;
* its p-value is a within-scene permutation test: under "no difference" the
  checkpoint labels of a scene's episodes are exchangeable, so the candidate's
  success count per scene is hypergeometric given the scene's pooled count;
* the noise floor is each checkpoint's own disagreement between its repeats.

Pooling scene x repeat pairs into one McNemar would treat repeats as fresh
scenes and overstate the evidence; the per-repeat McNemar is printed only as a
readout of how single-pair comparisons would have scattered.

Usage::

    python3 tools/audit/compare_put_into_repeats.py \\
        --checkpoint step_56072006=DIR_A1,DIR_A2,DIR_A3 \\
        --checkpoint step_63525522=DIR_B1,DIR_B2,DIR_B3 \\
        --output comparison.json

The first ``--checkpoint`` is the baseline unless ``--baseline`` names another.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from tools.audit.compare_put_into_evaluations import mcnemar_exact
except ModuleNotFoundError:  # run as a plain script from tools/audit
    from compare_put_into_evaluations import mcnemar_exact  # type: ignore

PROTOCOL_KEYS = (
    "scene_manifest_sha256",
    "split",
    "worlds",
    "rounds",
    "distinct_scene_rounds",
    "decisions",
    "settle_decisions",
    "excluded_validation_panel",
    "prior_noise_scale",
    "stochastic_seed",
    "arm",
)
FLAGS = ("strict", "native", "grasped", "lifted", "released", "carry_slip", "wrong_place", "non_finite")
# Outcomes a candidate must not lose while gaining strict (grasp erosion).
PRIMARY = "strict"
RETENTION = ("grasped", "lifted")


def load_evaluation(directory: Path) -> dict[str, Any]:
    report = json.loads((directory / "evaluation.json").read_text("utf-8"))
    episodes = report["results"].get("episodes")
    if not episodes:
        raise SystemExit(f"{directory}: no per-scene outcomes (results.episodes).")
    uids = [row["scene_uid"] for row in episodes]
    if len(set(uids)) != len(uids):
        raise SystemExit(f"{directory}: scenes repeat inside one evaluation.")
    sha_file = directory / "checkpoint.sha256"
    return {
        "dir": str(directory),
        "report": report,
        "episodes": {row["scene_uid"]: row for row in episodes},
        "checkpoint_sha256": (
            sha_file.read_text("utf-8").split()[0] if sha_file.exists() else None
        ),
    }


def outcome_matrix(
    evaluations: Sequence[Mapping[str, Any]], scenes: Sequence[str], flag: str
) -> np.ndarray:
    """``(repeats, scenes)`` 0/1 array of one outcome flag."""

    return np.asarray(
        [[bool(ev["episodes"][uid].get(flag, False)) for uid in scenes] for ev in evaluations],
        dtype=np.float64,
    )


def scene_bootstrap(
    diff: np.ndarray, *, rng: np.random.Generator, resamples: int
) -> tuple[float, float]:
    """95% percentile interval of the mean of per-scene values."""

    n = diff.size
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(resamples, n))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(low), float(high))


def permutation_p(
    base: np.ndarray,
    cand: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> float:
    """Two-sided within-scene label-permutation p for mean(cand) - mean(base).

    ``base`` is ``(Ra, n)`` and ``cand`` ``(Rb, n)`` of 0/1 outcomes. Given a
    scene's pooled successes k out of Ra + Rb, a random relabelling gives the
    candidate a hypergeometric share, so the null is drawn exactly.
    """

    ra, rb = base.shape[0], cand.shape[0]
    k = (base.sum(axis=0) + cand.sum(axis=0)).astype(np.int64)
    observed = (cand.mean(axis=0) - base.mean(axis=0)).mean()
    informative = (k > 0) & (k < ra + rb)
    if not informative.any():
        return 1.0
    k = k[informative]
    n_all = base.shape[1]
    # Scenes with k in {0, Ra+Rb} contribute 0 under every relabelling.
    cand_hits = rng.hypergeometric(k, ra + rb - k, rb, size=(resamples, k.size))
    null = (cand_hits / rb - (k - cand_hits) / ra).sum(axis=1) / n_all
    extreme = np.abs(null) >= abs(observed) - 1e-12
    return float((extreme.sum() + 1) / (resamples + 1))


def repeat_discordance(matrix: np.ndarray) -> float | None:
    """Mean fraction of scenes whose verdict differs between two repeats."""

    if matrix.shape[0] < 2:
        return None
    pairs = [
        float(np.mean(matrix[i] != matrix[j]))
        for i, j in itertools.combinations(range(matrix.shape[0]), 2)
    ]
    return round(float(np.mean(pairs)), 4)


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=lambda key: pvalues[key])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def summarize_checkpoint(
    evaluations: Sequence[Mapping[str, Any]], scenes: Sequence[str]
) -> dict[str, Any]:
    strict = outcome_matrix(evaluations, scenes, "strict")
    grasped = outcome_matrix(evaluations, scenes, "grasped")
    lifted = outcome_matrix(evaluations, scenes, "lifted")
    out: dict[str, Any] = {
        "repeats": len(evaluations),
        "episodes": int(strict.size),
        "strict_per_repeat": [int(row.sum()) for row in strict],
        "strict_repeat_discordance": repeat_discordance(strict),
        # Scenes solved in every / no repeat: how much of the set is settled.
        "strict_scenes_always": int((strict.min(axis=0) == 1).sum()),
        "strict_scenes_never": int((strict.max(axis=0) == 0).sum()),
    }
    for flag in FLAGS:
        matrix = outcome_matrix(evaluations, scenes, flag)
        out[flag] = round(float(matrix.mean()), 4)
    out["lifted_given_grasped"] = (
        round(float(lifted.sum() / grasped.sum()), 4) if grasped.sum() else None
    )
    out["strict_given_lifted"] = (
        round(float((strict * lifted).sum() / lifted.sum()), 4) if lifted.sum() else None
    )
    return out


def compare_pair(
    base_evals: Sequence[Mapping[str, Any]],
    cand_evals: Sequence[Mapping[str, Any]],
    scenes: Sequence[str],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    first = base_evals[0]["episodes"]
    destination = np.asarray([first[uid]["destination"] for uid in scenes])
    target = np.asarray([first[uid]["target_catalog"] for uid in scenes])
    out: dict[str, Any] = {"metrics": {}, "strict_by_destination": {}, "strict_by_object": {}}

    def effect(base: np.ndarray, cand: np.ndarray, *, test: bool) -> dict[str, Any]:
        diff = cand.mean(axis=0) - base.mean(axis=0)
        low, high = scene_bootstrap(diff, rng=rng, resamples=resamples)
        row = {
            "scenes": int(diff.size),
            "baseline": round(float(base.mean()), 4),
            "candidate": round(float(cand.mean()), 4),
            "difference": round(float(diff.mean()), 4),
            "ci95": [round(low, 4), round(high, 4)],
        }
        if test:
            row["permutation_p"] = round(
                permutation_p(base, cand, rng=rng, resamples=resamples), 5
            )
        return row

    for flag in ("strict", "native", "grasped", "lifted", "carry_slip", "wrong_place"):
        base = outcome_matrix(base_evals, scenes, flag)
        cand = outcome_matrix(cand_evals, scenes, flag)
        out["metrics"][flag] = effect(base, cand, test=True)
    base = outcome_matrix(base_evals, scenes, PRIMARY)
    cand = outcome_matrix(cand_evals, scenes, PRIMARY)
    for name in np.unique(destination):
        mask = destination == name
        out["strict_by_destination"][str(name)] = effect(base[:, mask], cand[:, mask], test=True)
    for name in np.unique(target):
        mask = target == name
        out["strict_by_object"][str(name)] = effect(base[:, mask], cand[:, mask], test=False)
    # Readout only: how single matched pairs (repeat i vs repeat i) scatter.
    out["per_repeat_mcnemar"] = [
        {
            "baseline_successes": int(b.sum()),
            "candidate_successes": int(c.sum()),
            "baseline_only": int(((b == 1) & (c == 0)).sum()),
            "candidate_only": int(((c == 1) & (b == 0)).sum()),
            "p": round(
                mcnemar_exact(int(((b == 1) & (c == 0)).sum()), int(((c == 1) & (b == 0)).sum())),
                5,
            ),
        }
        for b, c in zip(base, cand)
    ]
    return out


def verdict(pair: Mapping[str, Any], *, alpha: float, adjusted_p: float) -> str:
    strict = pair["metrics"][PRIMARY]
    # Grasp erosion is the recurring failure of this lineage; a strict gain
    # bought by a significant loss of grasps or lifts is not a promotion.
    eroded = [
        flag
        for flag in RETENTION
        if pair["metrics"][flag]["difference"] < 0
        and pair["metrics"][flag]["permutation_p"] < alpha
    ]
    if adjusted_p >= alpha:
        return "no_significant_difference" + ("; retention_loss:" + ",".join(eroded) if eroded else "")
    if strict["difference"] > 0:
        return "candidate_better" if not eroded else "candidate_better_but_retention_loss:" + ",".join(eroded)
    return "baseline_better"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=DIR[,DIR...]",
        help="One checkpoint's repeated evaluation directories.",
    )
    parser.add_argument("--baseline", default=None, help="Label of the baseline (default: first).")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    groups: dict[str, list[dict[str, Any]]] = {}
    for spec in args.checkpoint:
        label, _, dirs = spec.partition("=")
        if not label or not dirs:
            raise SystemExit(f"Expected LABEL=DIR[,DIR...], got {spec!r}.")
        if label in groups:
            raise SystemExit(f"Checkpoint label {label!r} given twice.")
        groups[label] = [
            load_evaluation(Path(item).expanduser().resolve())
            for item in dirs.split(",")
            if item
        ]
    baseline_label = args.baseline or next(iter(groups))
    if baseline_label not in groups:
        raise SystemExit(f"Unknown baseline {baseline_label!r}.")

    everything = [ev for evs in groups.values() for ev in evs]
    reference = everything[0]["report"]
    for ev in everything[1:]:
        for key in PROTOCOL_KEYS:
            if ev["report"].get(key) != reference.get(key):
                raise SystemExit(
                    f"Protocols differ on {key}: {ev['dir']} has {ev['report'].get(key)!r}, "
                    f"{everything[0]['dir']} has {reference.get(key)!r}."
                )
    scene_set = set(everything[0]["episodes"])
    for ev in everything[1:]:
        if set(ev["episodes"]) != scene_set:
            raise SystemExit(f"{ev['dir']} did not run the same scenes.")
    scenes = sorted(scene_set)
    for label, evs in groups.items():
        paths = {ev["report"]["checkpoint"] for ev in evs}
        hashes = {ev["checkpoint_sha256"] for ev in evs} - {None}
        if len(paths) != 1 or len(hashes) > 1:
            raise SystemExit(f"{label}: its repeats evaluated different checkpoints: {paths} {hashes}.")

    rng = np.random.default_rng(int(args.seed))
    result: dict[str, Any] = {
        "tool": "compare_put_into_repeats.py",
        "scenes": len(scenes),
        "protocol": {key: reference.get(key) for key in PROTOCOL_KEYS},
        "baseline": baseline_label,
        "alpha": float(args.alpha),
        "checkpoints": {
            label: {
                "path": evs[0]["report"]["checkpoint"],
                "sha256": evs[0]["checkpoint_sha256"],
                "dirs": [ev["dir"] for ev in evs],
                **summarize_checkpoint(evs, scenes),
            }
            for label, evs in groups.items()
        },
        "comparisons": {},
    }
    for label, evs in groups.items():
        if label == baseline_label:
            continue
        result["comparisons"][label] = compare_pair(
            groups[baseline_label], evs, scenes, rng=rng, resamples=int(args.resamples)
        )
    adjusted = holm(
        {
            label: pair["metrics"][PRIMARY]["permutation_p"]
            for label, pair in result["comparisons"].items()
        }
    )
    for label, pair in result["comparisons"].items():
        pair["strict_p_holm"] = round(adjusted[label], 5)
        pair["verdict"] = verdict(pair, alpha=float(args.alpha), adjusted_p=adjusted[label])

    print(json.dumps(result, indent=2))
    print_table(result)
    if args.output is not None:
        args.output.expanduser().resolve().write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    return 0


def print_table(result: Mapping[str, Any]) -> None:
    """A compact human summary after the JSON, for pasting back."""

    def pct(value: Any) -> str:
        return "   -  " if value is None else f"{100 * float(value):5.1f}%"

    print(f"\n=== {result['scenes']} scenes, baseline {result['baseline']} ===")
    print(f"{'checkpoint':<16}{'R':>3}{'strict':>9}{'native':>9}{'grasp':>9}{'lift':>9}"
          f"{'lift|gr':>9}{'str|lift':>9}{'flip':>8}  strict per repeat")
    for label, row in result["checkpoints"].items():
        print(
            f"{label:<16}{row['repeats']:>3}{pct(row['strict']):>9}{pct(row['native']):>9}"
            f"{pct(row['grasped']):>9}{pct(row['lifted']):>9}{pct(row['lifted_given_grasped']):>9}"
            f"{pct(row['strict_given_lifted']):>9}{pct(row['strict_repeat_discordance']):>8}  "
            f"{row['strict_per_repeat']}"
        )
    for label, pair in result["comparisons"].items():
        print(f"\n{label} - {result['baseline']}: verdict {pair['verdict']} (strict Holm p {pair['strict_p_holm']})")
        for name, row in pair["metrics"].items():
            low, high = row["ci95"]
            print(
                f"  {name:<12}{100 * row['difference']:+6.2f} pp  CI [{100 * low:+.2f}, {100 * high:+.2f}]"
                f"  p {row['permutation_p']}"
            )
        for name, row in pair["strict_by_destination"].items():
            low, high = row["ci95"]
            print(
                f"  strict {name:<5}{100 * row['difference']:+6.2f} pp  CI [{100 * low:+.2f}, {100 * high:+.2f}]"
                f"  p {row['permutation_p']}  ({row['scenes']} scenes)"
            )
        print("  per-repeat McNemar p:", [row["p"] for row in pair["per_repeat_mcnemar"]])


if __name__ == "__main__":
    raise SystemExit(main())
