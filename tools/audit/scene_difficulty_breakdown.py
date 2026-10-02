#!/usr/bin/env python3
"""Where do put_into failures live: in particular scenes, or in per-episode chaos?

Repeated matched evaluations give every scene many draws. Pooling them answers a
question no single success rate can: is ~62% failure a property of SOME scenes
(systematic modes worth targeting: object, position, clutter, destination), or
spread thinly over coin-flip scenes (closed-loop chaos, where only robustness
helps)?

What it reports
---------------

* **Spectrum**: per-scene success fraction over all draws, binned; scenes never
  and always solved, against what a world where every scene had the SAME rate
  would produce (binomial expectation).
* **Scene-determined variance (ICC)**: between-scene variance of the true
  success probability, as a share of total outcome variance. Corrected for the
  finite number of draws per scene. 0 = pure chaos, 1 = scenes decide.
* **Reliability**: correlation of per-scene rates between disjoint halves of
  the draws (different checkpoints / repeats), Spearman-Brown corrected.
* **Failure stages by difficulty class**: of the failures in never / hard /
  middle / easy scenes, how many never grasped, grasped but never lifted,
  lifted but did not finish. Shows which stage owns the systematic part.
* **Object x destination** and **geometry** (with ``--scene-manifest``):
  quartile tables and rank correlations of per-scene rates against target
  position, start distance, transport distance, clutter and yaw.
* ``scenes.csv`` (one row per scene) and ``scene_classes.json`` (uids by class)
  for later targeting.

Pooling across checkpoints is only valid when they do not differ; the per-
checkpoint rates and their between-checkpoint scene correlation are printed so
that can be checked rather than assumed.

Usage::

    python3 tools/audit/scene_difficulty_breakdown.py \\
        --eval-root runs/three_stage_put_into_repeats/20260930_110216 \\
        --scene-manifest runs/three_stage/scenes_8192.json \\
        --output runs/three_stage_put_into_repeats/20260930_110216/scene_breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.compare_put_into_repeats import PROTOCOL_KEYS, load_evaluation  # noqa: E402

FLAGS = ("strict", "native", "grasped", "lifted", "carry_slip", "wrong_place", "non_finite")
CLASS_EDGES = (
    ("never", 0.0, 0.0),
    ("hard", 1e-9, 0.25),
    ("middle", 0.25 + 1e-9, 0.75 - 1e-9),
    ("easy", 0.75, 1.0 - 1e-9),
    ("always", 1.0, 1.0),
)


# --------------------------------------------------------------------------- stats


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), as scipy.stats.rankdata."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return None
    rx, ry = rankdata(x[keep]), rankdata(y[keep])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def scene_icc(matrix: np.ndarray) -> dict[str, float]:
    """Share of outcome variance that is between scenes, finite-draw corrected.

    ``matrix`` is (draws, scenes) of 0/1. With n draws per scene,
    E[p_hat(1-p_hat)] = p(1-p)(n-1)/n, so var(p_hat | p) is estimated without
    bias by p_hat(1-p_hat)/(n-1), and the between-scene variance of the true p
    is var(p_hat) minus its mean.
    """

    n = matrix.shape[0]
    p_hat = matrix.mean(axis=0)
    mean = float(p_hat.mean())
    within = float(np.mean(p_hat * (1.0 - p_hat)) / max(1, n - 1))
    between = max(0.0, float(p_hat.var()) - within)
    total = mean * (1.0 - mean)
    return {
        "draws_per_scene": int(n),
        "mean_rate": round(mean, 4),
        "between_scene_variance": round(between, 5),
        "total_variance": round(total, 5),
        "icc": round(between / total, 4) if total > 0 else float("nan"),
        # SD of the true per-scene success probability.
        "true_rate_sd": round(math.sqrt(between), 4),
    }


def split_half_reliability(halves: Sequence[np.ndarray]) -> dict[str, float] | None:
    """Pearson r of per-scene rates between two disjoint draw sets, and the
    Spearman-Brown reliability of their pooled mean."""

    if len(halves) != 2 or min(h.shape[0] for h in halves) == 0:
        return None
    a, b = (h.mean(axis=0) for h in halves)
    if a.std() == 0 or b.std() == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return {"r": round(r, 4), "spearman_brown": round(2 * r / (1 + r), 4) if r > -1 else None}


def binomial_extremes(mean: float, draws: int, scenes: int) -> dict[str, float]:
    """Never/always counts expected if every scene had the pooled rate."""

    return {
        "expected_never": round(scenes * (1.0 - mean) ** draws, 2),
        "expected_always": round(scenes * mean ** draws, 2),
    }


def classify(p: np.ndarray) -> np.ndarray:
    out = np.empty(p.size, dtype=object)
    for name, low, high in CLASS_EDGES:
        out[(p >= low) & (p <= high)] = name
    return out


# --------------------------------------------------------------------------- loading


def discover(eval_root: Path) -> dict[str, list[Path]]:
    """``<root>/<label>/rep*/evaluation.json`` as written by the repeats launcher."""

    groups: dict[str, list[Path]] = {}
    for path in sorted(eval_root.glob("*/rep*/evaluation.json")):
        groups.setdefault(path.parent.parent.name, []).append(path.parent)
    if not groups:
        raise SystemExit(f"No */rep*/evaluation.json under {eval_root}.")
    return groups


def load_groups(groups: Mapping[str, Sequence[Path]]) -> tuple[dict[str, list[dict]], list[str]]:
    loaded = {label: [load_evaluation(Path(d)) for d in dirs] for label, dirs in groups.items()}
    everything = [ev for evs in loaded.values() for ev in evs]
    reference = everything[0]["report"]
    for ev in everything[1:]:
        for key in PROTOCOL_KEYS:
            if ev["report"].get(key) != reference.get(key):
                raise SystemExit(f"Protocols differ on {key}: {ev['dir']}.")
    scenes = set(everything[0]["episodes"])
    for ev in everything[1:]:
        if set(ev["episodes"]) != scenes:
            raise SystemExit(f"{ev['dir']} did not run the same scenes.")
    return loaded, sorted(scenes)


def matrix_for(evals: Sequence[Mapping[str, Any]], scenes: Sequence[str], flag: str) -> np.ndarray:
    return np.asarray(
        [[bool(ev["episodes"][uid].get(flag, False)) for uid in scenes] for ev in evals],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- geometry


def scene_features(manifest_path: Path, scenes: Sequence[str]) -> dict[str, np.ndarray]:
    from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import (
        DESTINATION_SLOT,
        TARGET_SLOT,
        read_manifest,
    )

    all_scenes, manifest = read_manifest(manifest_path.expanduser().resolve(), validate=False)
    by_uid = {scene.scene_uid: scene for scene in all_scenes}
    missing = [uid for uid in scenes if uid not in by_uid]
    if missing:
        raise SystemExit(f"{len(missing)} evaluated scenes are not in the manifest, e.g. {missing[:2]}.")
    geometry = dict(manifest.get("geometry") or {})
    cx = float(np.mean(geometry.get("workspace_x_bounds", (0.0, 0.0))))
    cy = float(np.mean(geometry.get("workspace_y_bounds", (0.0, 0.0))))
    rows: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        rows.setdefault(name, []).append(float(value))

    for uid in scenes:
        scene = by_uid[uid]
        target = scene.target
        receptacle = scene.receptacle
        tx, ty = target.xy
        others = [o for o in scene.objects if o.slot not in (TARGET_SLOT, DESTINATION_SLOT)]
        nearest = min(
            (math.hypot(o.xy[0] - tx, o.xy[1] - ty) for o in others), default=float("nan")
        )
        # Fold yaw to the angle off the gripper's fixed axis, [0, pi/2]: the
        # fixed-yaw pickup sees a 180-degree-symmetric object the same at
        # yaw and yaw + pi.
        folded = abs((float(target.yaw) + math.pi / 2) % math.pi - math.pi / 2)
        add("target_x", tx)
        add("target_y", ty)
        add("target_radius", math.hypot(tx - cx, ty - cy))
        add("receptacle_radius", math.hypot(receptacle.xy[0] - cx, receptacle.xy[1] - cy))
        add("approach_xy_distance", scene.approach_xy_distance)
        add("approach_xyz_distance", scene.approach_xyz_distance)
        add("transport_xy_distance", scene.transport_xy_distance)
        add("start_ee_z", scene.ee_xyz[2])
        add("distractors", len(others))
        add("nearest_distractor_m", nearest)
        add("target_yaw_off_axis", folded)
        add("shade", scene.shade)
    return {name: np.asarray(values, dtype=np.float64) for name, values in rows.items()}


def quartile_table(feature: np.ndarray, outcomes: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    finite = np.isfinite(feature)
    unique = np.unique(feature[finite])
    if unique.size <= 6:  # categorical-ish: one bin per value
        edges = None
        bins = [(float(v), float(v)) for v in unique]
        assign = [feature == v for v in unique]
    else:
        edges = np.quantile(feature[finite], [0, 0.25, 0.5, 0.75, 1.0])
        bins = [(float(edges[i]), float(edges[i + 1])) for i in range(4)]
        assign = [
            finite & (feature >= edges[i]) & ((feature < edges[i + 1]) if i < 3 else (feature <= edges[i + 1]))
            for i in range(4)
        ]
    del edges
    table = []
    for (low, high), mask in zip(bins, assign):
        row: dict[str, Any] = {"low": round(low, 4), "high": round(high, 4), "scenes": int(mask.sum())}
        for name, values in outcomes.items():
            picked = values[mask]
            picked = picked[np.isfinite(picked)]
            row[name] = round(float(picked.mean()), 4) if picked.size else None
        table.append(row)
    return table


# --------------------------------------------------------------------------- main


def analyse(
    loaded: Mapping[str, Sequence[Mapping[str, Any]]],
    scenes: Sequence[str],
    *,
    features: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = list(loaded)
    evals = [ev for label in labels for ev in loaded[label]]
    M = {flag: matrix_for(evals, scenes, flag) for flag in FLAGS}
    draws = M["strict"].shape[0]
    n_scenes = len(scenes)
    first = evals[0]["episodes"]
    destination = np.asarray([first[uid]["destination"] for uid in scenes])
    target = np.asarray([first[uid]["target_catalog"] for uid in scenes])

    grasp_count = M["grasped"].sum(axis=0)
    lift_count = (M["lifted"] * M["grasped"]).sum(axis=0)
    strict_count = M["strict"].sum(axis=0)
    p = {
        "strict": strict_count / draws,
        "grasped": grasp_count / draws,
        "lifted": M["lifted"].mean(axis=0),
        "lift_given_grasp": np.where(grasp_count > 0, lift_count / np.maximum(grasp_count, 1), np.nan),
        "strict_given_lift": np.where(
            lift_count > 0, (M["strict"] * M["lifted"]).sum(axis=0) / np.maximum(lift_count, 1), np.nan
        ),
    }

    result: dict[str, Any] = {
        "scenes": n_scenes,
        "draws_per_scene": draws,
        "checkpoints": {label: len(loaded[label]) for label in labels},
        "protocol": {key: evals[0]["report"].get(key) for key in PROTOCOL_KEYS},
    }

    # Pooling check: per-checkpoint rates and between-checkpoint scene agreement.
    per_ckpt = {label: matrix_for(loaded[label], scenes, "strict") for label in labels}
    result["pooling_check"] = {
        "strict_rate": {label: round(float(m.mean()), 4) for label, m in per_ckpt.items()},
        "scene_rate_correlation": {
            f"{a}~{b}": round(float(np.corrcoef(per_ckpt[a].mean(0), per_ckpt[b].mean(0))[0, 1]), 4)
            for i, a in enumerate(labels)
            for b in labels[i + 1:]
        },
    }

    # Spectrum, ICC, binomial extremes, reliability -- per stage.
    result["stages"] = {}
    for name, matrix in (("strict", M["strict"]), ("grasped", M["grasped"]), ("lifted", M["lifted"])):
        rates = matrix.mean(axis=0)
        hist, _ = np.histogram(rates, bins=[0, 1e-9, 0.25, 0.5, 0.75, 1 - 1e-9, 1.0 + 1e-9])
        odd, even = matrix[0::2], matrix[1::2]
        result["stages"][name] = {
            **scene_icc(matrix),
            "never": int((rates == 0).sum()),
            "always": int((rates == 1).sum()),
            **binomial_extremes(float(rates.mean()), draws, n_scenes),
            "spectrum": dict(zip(["0", "(0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "1"], hist.astype(int).tolist())),
            "split_half_alternate_draws": split_half_reliability([odd, even]),
        }
    # Conditional lift: only scenes with enough grasps to say anything.
    enough = grasp_count >= 4
    lg = p["lift_given_grasp"][enough]
    result["stages"]["lift_given_grasp"] = {
        "scenes_with_4plus_grasps": int(enough.sum()),
        "mean": round(float(np.nanmean(lg)), 4) if lg.size else None,
        "scenes_never_lifting": int((lg == 0).sum()),
        "scenes_always_lifting": int((lg == 1).sum()),
        "spectrum": dict(zip(
            ["0", "(0,.5)", "[.5,.75)", "[.75,1)", "1"],
            np.histogram(lg, bins=[0, 1e-9, 0.5, 0.75, 1 - 1e-9, 1 + 1e-9])[0].astype(int).tolist(),
        )) if lg.size else None,
    }

    # Failure stages by difficulty class.
    classes = classify(p["strict"])
    fail = 1.0 - M["strict"]
    no_grasp = fail * (1.0 - M["grasped"])
    grasp_no_lift = fail * M["grasped"] * (1.0 - M["lifted"])
    lift_no_strict = fail * M["lifted"]
    total_fail = float(fail.sum())
    result["failure_by_class"] = {}
    for name, _, _ in CLASS_EDGES:
        mask = classes == name
        f = float(fail[:, mask].sum())
        result["failure_by_class"][name] = {
            "scenes": int(mask.sum()),
            "share_of_all_failures": round(f / total_fail, 4) if total_fail else None,
            "of_these_failures": {
                "no_grasp": round(float(no_grasp[:, mask].sum()) / f, 4) if f else None,
                "grasp_no_lift": round(float(grasp_no_lift[:, mask].sum()) / f, 4) if f else None,
                "lift_no_strict": round(float(lift_no_strict[:, mask].sum()) / f, 4) if f else None,
            },
            "carry_slip_rate": round(float(M["carry_slip"][:, mask].mean()), 4) if mask.any() else None,
            "wrong_place_rate": round(float(M["wrong_place"][:, mask].mean()), 4) if mask.any() else None,
        }
    result["failure_stage_totals"] = {
        "no_grasp": round(float(no_grasp.sum()) / total_fail, 4),
        "grasp_no_lift": round(float(grasp_no_lift.sum()) / total_fail, 4),
        "lift_no_strict": round(float(lift_no_strict.sum()) / total_fail, 4),
    }

    # Object x destination.
    result["by_cell"] = {}
    for obj in np.unique(target):
        for dest in np.unique(destination):
            mask = (target == obj) & (destination == dest)
            if not mask.any():
                continue
            result["by_cell"][f"{obj}|{dest}"] = {
                "scenes": int(mask.sum()),
                "strict": round(float(p["strict"][mask].mean()), 4),
                "grasped": round(float(p["grasped"][mask].mean()), 4),
                "lift_given_grasp": round(float(np.nanmean(p["lift_given_grasp"][mask])), 4),
                "strict_given_lift": round(float(np.nanmean(p["strict_given_lift"][mask])), 4),
                "never_strict": int((p["strict"][mask] == 0).sum()),
                "icc_strict": scene_icc(M["strict"][:, mask])["icc"],
            }

    # Geometry.
    if features:
        result["geometry"] = {}
        for name, values in features.items():
            entry: dict[str, Any] = {
                "spearman": {
                    stage: (None if (r := spearman(values, p[stage])) is None else round(r, 4))
                    for stage in ("strict", "grasped", "lift_given_grasp", "strict_given_lift")
                },
                "quartiles": quartile_table(values, p),
            }
            for dest in np.unique(destination):
                mask = destination == dest
                r = spearman(values[mask], p["strict"][mask])
                entry["spearman"][f"strict_{dest}"] = None if r is None else round(r, 4)
            result["geometry"][name] = entry

    rows = []
    for i, uid in enumerate(scenes):
        row = {
            "scene_uid": uid,
            "destination": destination[i],
            "target_catalog": target[i],
            "class": classes[i],
            "draws": draws,
            "strict": int(strict_count[i]),
            "grasped": int(grasp_count[i]),
            "lifted": int(lift_count[i]),
            "carry_slip": int(M["carry_slip"][:, i].sum()),
            "wrong_place": int(M["wrong_place"][:, i].sum()),
            "non_finite": int(M["non_finite"][:, i].sum()),
        }
        for name, values in (features or {}).items():
            row[name] = round(float(values[i]), 5)
        rows.append(row)
    return result, rows


def print_summary(result: Mapping[str, Any]) -> None:
    pct = lambda v: "   -  " if v is None else f"{100 * float(v):5.1f}%"
    print(f"\n=== {result['scenes']} scenes x {result['draws_per_scene']} draws ({result['checkpoints']}) ===")
    pc = result["pooling_check"]
    print("pooling check: strict", {k: pct(v) for k, v in pc["strict_rate"].items()},
          "scene-rate r between checkpoints", pc["scene_rate_correlation"])
    print(f"\n{'stage':10s}{'mean':>8}{'ICC':>7}{'trueSD':>8}{'never':>7}{'(binom)':>9}{'always':>8}{'(binom)':>9}{'split-half r':>14}  spectrum")
    for name in ("strict", "grasped", "lifted"):
        s = result["stages"][name]
        sh = s["split_half_alternate_draws"] or {}
        print(f"{name:10s}{pct(s['mean_rate']):>8}{s['icc']:7.2f}{s['true_rate_sd']:8.3f}{s['never']:7d}{s['expected_never']:9.1f}"
              f"{s['always']:8d}{s['expected_always']:9.1f}{sh.get('r', float('nan')):14.3f}  {s['spectrum']}")
    lg = result["stages"]["lift_given_grasp"]
    print(f"lift|grasp over {lg['scenes_with_4plus_grasps']} scenes with >=4 grasps: mean {pct(lg['mean'])}, "
          f"never {lg['scenes_never_lifting']}, always {lg['scenes_always_lifting']}, spectrum {lg['spectrum']}")
    print("\nfailures by strict-difficulty class (share of ALL failures; then what those failures were):")
    for name, row in result["failure_by_class"].items():
        f = row["of_these_failures"]
        print(f"  {name:7s} scenes {row['scenes']:4d}  share {pct(row['share_of_all_failures'])}  "
              f"no_grasp {pct(f['no_grasp'])}  grasp_no_lift {pct(f['grasp_no_lift'])}  lift_no_strict {pct(f['lift_no_strict'])}"
              f"  | slip {pct(row['carry_slip_rate'])} wrong_place {pct(row['wrong_place_rate'])}")
    t = result["failure_stage_totals"]
    print(f"  all     no_grasp {pct(t['no_grasp'])}  grasp_no_lift {pct(t['grasp_no_lift'])}  lift_no_strict {pct(t['lift_no_strict'])}")
    print("\nobject|destination:")
    for name, row in result["by_cell"].items():
        print(f"  {name:28s} n={row['scenes']:3d} strict {pct(row['strict'])} grasp {pct(row['grasped'])} "
              f"lift|gr {pct(row['lift_given_grasp'])} str|lift {pct(row['strict_given_lift'])} never {row['never_strict']:3d} ICC {row['icc_strict']:.2f}")
    if "geometry" in result:
        chance = 2.0 / math.sqrt(max(1, int(result["scenes"])))
        print(f"\ngeometry (Spearman of per-scene rate; |r| > ~{chance:.2f} is beyond chance at {result['scenes']} scenes):")
        print(f"  {'feature':24s}{'strict':>8}{'grasp':>8}{'lift|gr':>9}{'str|lift':>9}{'plate':>8}{'bowl':>8}   strict by quartile")
        for name, entry in result["geometry"].items():
            s = entry["spearman"]
            f = lambda v: "     -" if v is None else f"{v:+.3f}"
            quart = " ".join(pct(q["strict"]).strip() for q in entry["quartiles"])
            print(f"  {name:24s}{f(s['strict']):>8}{f(s['grasped']):>8}{f(s['lift_given_grasp']):>9}{f(s['strict_given_lift']):>9}"
                  f"{f(s.get('strict_plate')):>8}{f(s.get('strict_bowl')):>8}   {quart}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path, default=None,
                        help="A compare_cdpr_three_stage_repeats_remote.sh OUT_ROOT.")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=DIR[,DIR...]")
    parser.add_argument("--only", nargs="+", default=None, help="Labels to pool (default: all found).")
    parser.add_argument("--scene-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    groups: dict[str, list[Path]] = {}
    if args.eval_root is not None:
        groups.update(discover(args.eval_root.expanduser().resolve()))
    for spec in args.checkpoint:
        label, _, dirs = spec.partition("=")
        groups[label] = [Path(d) for d in dirs.split(",") if d]
    if args.only:
        groups = {label: dirs for label, dirs in groups.items() if label in set(args.only)}
    if not groups:
        raise SystemExit("Give --eval-root or --checkpoint.")
    loaded, scenes = load_groups(groups)
    features = scene_features(args.scene_manifest, scenes) if args.scene_manifest else None
    result, rows = analyse(loaded, scenes, features=features)
    print_summary(result)
    if args.output is not None:
        out = args.output.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "breakdown.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
        with (out / "scenes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        classes = {}
        for row in rows:
            classes.setdefault(row["class"], []).append(row["scene_uid"])
        (out / "scene_classes.json").write_text(json.dumps(classes, indent=2) + "\n", "utf-8")
        print(f"\n=== wrote {out}/breakdown.json, scenes.csv, scene_classes.json ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
