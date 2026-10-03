#!/usr/bin/env python3
"""Tables from per-step kinematic traces (episode_kinematic_trace.py).

Questions it answers, each split by target-y quartile (fixed edges from the
2026-10-02 scene breakdown, so the bins match those tables):

1. **What is a "slip"?** At the step contact is lost: was the gripper being
   commanded open (a premature release away from the goal) or not (a passive
   loss of grip)? Where was the hand (height, in or out of the overview frame)
   and where was the object relative to the receptacle?
2. **Does the hand climb out of view after the lift?** Peak EE z, the share of
   post-lift steps above the camera-framing band (0.32 m by default), and the
   share of post-lift steps with the object / EE / receptacle outside the
   overview and wrist frustums -- for strict successes against failures.
3. **Does the controller follow the command?** Post-lift tracking error between
   the commanded target and the measured EE, in xyz and in z alone.
5. **Who commands the vertical motion?** At each policy decision the executed
   action is ``tanh(prior + residual_scale * residual)``. Per phase -- hovering
   over the object before any grasp, and after the lift until any slip -- it
   compares the executed Z command with ``tanh(prior)`` (the frozen SmolVLA
   prior alone) and with the residual's push ``atanh(final) - prior``. Z > 0
   is up. Needs traces written with decision records.
4. **Are grasp misses depth errors?** For episodes that never grasp: the EE -
   object offset at the closest approach, in x and in y. Overview depth error
   predicts a y spread that grows with target y and no matching x spread.

Usage::

    python3 tools/audit/summarize_kinematic_traces.py runs/.../eval_dir/trace \\
        --output runs/.../eval_dir/trace_summary
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_Y_EDGES = (-0.073, 0.002, 0.074)
OUTCOMES = ("strict", "no_grasp", "grasp_no_lift", "slip", "wrong_place", "lifted_other")


def _first(flag: np.ndarray) -> int | None:
    hits = np.flatnonzero(flag)
    return int(hits[0]) if hits.size else None


def episode_rows(trace: Mapping[str, np.ndarray], *, band_top: float, open_window: int) -> list[dict[str, Any]]:
    active = trace["active"].astype(bool)
    steps, worlds = active.shape
    rows = []
    for w in range(worlds):
        live = np.flatnonzero(active[:, w])
        if live.size == 0 or bool(trace["non_finite"][w]):
            continue
        end = int(live[-1]) + 1
        get = lambda key: trace[key][:end, w]  # noqa: E731
        obj, ee, cmd, rec = get("object_xyz"), get("ee_xyz"), get("command_xyz"), get("receptacle_xyz")
        grasp_t, lift_t = _first(get("grasped")), _first(get("lifted"))
        slip_t, wrong_t = _first(get("carry_slip")), _first(get("wrong_place"))
        strict = bool(trace["strict_final"][w])
        if strict:
            outcome = "strict"
        elif grasp_t is None:
            outcome = "no_grasp"
        elif lift_t is None:
            outcome = "grasp_no_lift"
        elif slip_t is not None:
            outcome = "slip"
        elif wrong_t is not None:
            outcome = "wrong_place"
        else:
            outcome = "lifted_other"
        row: dict[str, Any] = {
            "scene_uid": str(trace["scene_uid"][w]),
            "destination": str(trace["destination"][w]),
            "target_catalog": str(trace["target_catalog"][w]),
            "target_y0": float(obj[0, 1]),
            "target_x0": float(obj[0, 0]),
            "receptacle_y": float(rec[0, 1]),
            "outcome": outcome,
            "steps": end,
            "grasp_step": grasp_t,
            "lift_step": lift_t,
            "slip_step": slip_t,
            "wrong_place_step": wrong_t,
        }
        if lift_t is not None:
            stop = slip_t if slip_t is not None else end
            post = slice(lift_t, max(stop, lift_t + 1))
            row["peak_ee_z_after_lift"] = float(ee[post, 2].max())
            row["above_band_frac"] = float((ee[post, 2] > band_top).mean())
            for name in ("object_in_overview", "ee_in_overview", "receptacle_in_overview", "receptacle_in_wrist"):
                row[f"{name}_frac"] = float(get(name)[post].mean())
            error = cmd[post] - ee[post]
            row["track_err_xyz_mean"] = float(np.linalg.norm(error, axis=-1).mean())
            row["track_err_z_mean"] = float(np.abs(error[:, 2]).mean())
            row["track_err_xyz_max"] = float(np.linalg.norm(error, axis=-1).max())
            row["lift_to_receptacle_xy"] = float(np.linalg.norm(obj[lift_t, :2] - rec[lift_t, :2]))
        if slip_t is not None:
            window = slice(max(0, slip_t - open_window), slip_t + 1)
            commanded = bool((get("gripper_command")[window] > 1e-6).any())
            opening = get("opening")
            opened = bool(opening[slip_t] > opening[max(0, slip_t - open_window)] + 1e-3)
            row["slip_type"] = "commanded_open" if commanded else ("opened_uncommanded" if opened else "passive")
            row["lift_to_slip_steps"] = None if lift_t is None else slip_t - lift_t
            row["slip_ee_z"] = float(ee[slip_t, 2])
            row["slip_object_in_overview"] = bool(get("object_in_overview")[slip_t])
            row["slip_ee_in_overview"] = bool(get("ee_in_overview")[slip_t])
            offset = obj[slip_t, :2] - rec[slip_t, :2]
            row["slip_dx"], row["slip_dy"] = float(offset[0]), float(offset[1])
            row["slip_dist_over_radius"] = float(np.linalg.norm(offset) / float(trace["success_radius"][w]))
        if wrong_t is not None:
            offset = obj[wrong_t, :2] - rec[wrong_t, :2]
            row["wrong_dx"], row["wrong_dy"] = float(offset[0]), float(offset[1])
        if outcome == "no_grasp":
            gap = ee[:, :2] - obj[:, :2]
            k = int(np.argmin(np.linalg.norm(gap, axis=-1)))
            row["miss_dx"], row["miss_dy"] = float(gap[k, 0]), float(gap[k, 1])
            row["miss_xy"] = float(np.linalg.norm(gap[k]))
            row["miss_dz"] = float(ee[k, 2] - obj[k, 2])
            row["miss_closed"] = bool((get("gripper_command")[max(0, k - 8): k + 8] < -1e-6).any())
            row["miss_object_in_overview"] = bool(get("object_in_overview")[k])
        rows.append(row)
        row.update(_attribution(trace, w, end=end, grasp_t=grasp_t, lift_t=lift_t, slip_t=slip_t))
        row.update(_attribution_all_dims(trace, w, end=end, grasp_t=grasp_t, lift_t=lift_t, slip_t=slip_t))
    return rows


DIMS = ("x", "y", "z", "yaw", "grip")


def _attribution_all_dims(trace: Mapping[str, np.ndarray], w: int, *, end: int, grasp_t: int | None,
                          lift_t: int | None, slip_t: int | None, hover_xy: float = 0.06,
                          preslip_decisions: int = 3) -> dict[str, Any]:
    """Every action dimension: executed, prior-only, residual push, prior spread
    and residual saturation, per phase (hover, carry, the decisions just before
    a slip). Gripper > 0 is OPEN."""

    if "decision_final" not in trace:
        return {}
    scale = float(trace.get("residual_scale", np.float32(1.0)))
    final = trace["decision_final"][:, w].astype(np.float64)       # (D, per, 5)
    prior = trace["decision_prior"][:, w].astype(np.float64)
    live = trace["decision_active"][:, w].astype(bool)
    step_decision = trace["decision"][:end, w]
    ee, obj = trace["ee_xyz"][:end, w], trace["object_xyz"][:end, w]
    phases: dict[str, list[int]] = {"hover": [], "carry": [], "preslip": []}
    starts = []
    for k, index in enumerate(trace["decision_index"]):
        hits = np.flatnonzero(step_decision == index)
        starts.append(int(hits[0]) if (live[k] and hits.size) else None)
    for k, t0 in enumerate(starts):
        if t0 is None:
            continue
        if (grasp_t is None or t0 < grasp_t) and np.linalg.norm(ee[t0, :2] - obj[t0, :2]) <= hover_xy:
            phases["hover"].append(k)
        if lift_t is not None and t0 >= lift_t and (slip_t is None or t0 < slip_t):
            phases["carry"].append(k)
    if slip_t is not None:
        before = [k for k, t0 in enumerate(starts) if t0 is not None and t0 <= slip_t]
        phases["preslip"] = before[-preslip_decisions:]
    out: dict[str, Any] = {}
    for name, ks in phases.items():
        if not ks:
            continue
        f, pr = final[ks], prior[ks]
        push = np.arctanh(np.clip(f, -0.999999, 0.999999)) - pr
        for d, dim in enumerate(DIMS):
            out[f"{name}_{dim}_final"] = float(f[..., d].mean())
            out[f"{name}_{dim}_prior_only"] = float(np.tanh(pr[..., d]).mean())
            out[f"{name}_{dim}_prior_sd"] = float(np.tanh(pr[..., d]).std())
            out[f"{name}_{dim}_push"] = float(push[..., d].mean())
            out[f"{name}_{dim}_saturated"] = float((np.abs(push[..., d]) > 0.9 * scale).mean())
    return out


def _attribution(trace: Mapping[str, np.ndarray], w: int, *, end: int, grasp_t: int | None,
                 lift_t: int | None, slip_t: int | None, hover_xy: float = 0.06) -> dict[str, Any]:
    """Mean Z command per phase: executed, prior-only, residual push."""

    if "decision_final" not in trace:
        return {}
    final = trace["decision_final"][:, w, :, 2].astype(np.float64)       # (D, per)
    prior = trace["decision_prior"][:, w, :, 2].astype(np.float64)
    live = trace["decision_active"][:, w].astype(bool)
    step_decision = trace["decision"][:end, w]
    ee, obj = trace["ee_xyz"][:end, w], trace["object_xyz"][:end, w]
    out: dict[str, Any] = {}
    phases = {"hover": [], "carry": []}
    for k, index in enumerate(trace["decision_index"]):
        if not live[k]:
            continue
        hits = np.flatnonzero(step_decision == index)
        if hits.size == 0:
            continue
        t0 = int(hits[0])
        before_grasp = grasp_t is None or t0 < grasp_t
        if before_grasp and np.linalg.norm(ee[t0, :2] - obj[t0, :2]) <= hover_xy:
            phases["hover"].append(k)
        if lift_t is not None and t0 >= lift_t and (slip_t is None or t0 < slip_t):
            phases["carry"].append(k)
    for name, ks in phases.items():
        if not ks:
            continue
        f, pr = final[ks], prior[ks]
        out[f"{name}_decisions"] = len(ks)
        out[f"{name}_z_final"] = float(f.mean())
        out[f"{name}_z_prior_only"] = float(np.tanh(pr).mean())
        out[f"{name}_z_residual_push"] = float((np.arctanh(np.clip(f, -0.999999, 0.999999)) - pr).mean())
        out[f"{name}_z_up_share"] = float((f > 0).mean())
    return out


def quartile_of(y: float, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges), y, side="right"))


def summarize(rows: Sequence[Mapping[str, Any]], *, edges: Sequence[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"episodes": len(rows), "y_edges": list(edges)}
    q = np.asarray([quartile_of(r["target_y0"], edges) for r in rows])
    labels = [f"Q{i + 1}" for i in range(len(edges) + 1)]

    def stat(values: Sequence[float]) -> dict[str, Any] | None:
        v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
        if v.size == 0:
            return None
        return {"n": int(v.size), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
                "sd": round(float(v.std()), 4)}

    out["outcomes_by_y"] = {}
    for i, label in enumerate(labels):
        sel = [r for r, qq in zip(rows, q) if qq == i]
        out["outcomes_by_y"][label] = {"episodes": len(sel), **{
            o: round(sum(r["outcome"] == o for r in sel) / max(1, len(sel)), 4) for o in OUTCOMES}}

    slips = [r for r in rows if r["outcome"] == "slip"]
    out["slip_types"] = {}
    for i, label in [(None, "all"), *enumerate(labels)]:
        sel = [r for r, qq in zip(rows, q) if r["outcome"] == "slip" and (i is None or qq == i)]
        types = [r["slip_type"] for r in sel]
        out["slip_types"][label] = {
            "slips": len(sel),
            **{t: round(types.count(t) / max(1, len(sel)), 4) for t in ("commanded_open", "opened_uncommanded", "passive")},
            "lift_to_slip_steps": stat([r["lift_to_slip_steps"] for r in sel]),
            "slip_ee_z": stat([r["slip_ee_z"] for r in sel]),
            "object_out_of_overview_at_slip": round(sum(not r["slip_object_in_overview"] for r in sel) / max(1, len(sel)), 4),
            "dist_over_radius": stat([r["slip_dist_over_radius"] for r in sel]),
            "dx": stat([r["slip_dx"] for r in sel]),
            "dy": stat([r["slip_dy"] for r in sel]),
        }
    del slips

    out["post_lift"] = {}
    for group in ("strict", "slip", "wrong_place", "lifted_other"):
        for i, label in [(None, "all"), *enumerate(labels)]:
            sel = [r for r, qq in zip(rows, q) if r["outcome"] == group and r.get("lift_step") is not None
                   and (i is None or qq == i)]
            if not sel:
                continue
            out["post_lift"][f"{group}|{label}"] = {
                "episodes": len(sel),
                **{k: stat([r[k] for r in sel]) for k in (
                    "peak_ee_z_after_lift", "above_band_frac", "object_in_overview_frac", "ee_in_overview_frac",
                    "receptacle_in_overview_frac", "receptacle_in_wrist_frac", "track_err_xyz_mean",
                    "track_err_z_mean", "lift_to_receptacle_xy")},
            }

    out["z_attribution"] = {}
    for phase, groups in (("hover", ("strict", "no_grasp", "grasp_no_lift", "slip")),
                          ("carry", ("strict", "slip"))):
        for group in groups:
            for i, label in [(None, "all"), *enumerate(labels)]:
                sel = [r for r, qq in zip(rows, q) if r["outcome"] == group
                       and f"{phase}_z_final" in r and (i is None or qq == i)]
                if not sel:
                    continue
                out["z_attribution"][f"{phase}|{group}|{label}"] = {
                    "episodes": len(sel),
                    **{k: stat([r[f"{phase}_{k}"] for r in sel])
                       for k in ("z_final", "z_prior_only", "z_residual_push", "z_up_share")},
                }

    out["dim_attribution"] = {}
    for phase, groups in (("hover", ("strict", "no_grasp")), ("carry", ("strict", "slip")),
                          ("preslip", ("slip",))):
        for group in groups:
            for i, label in [(None, "all"), *enumerate(labels)]:
                sel = [r for r, qq in zip(rows, q) if r["outcome"] == group
                       and f"{phase}_z_final" in r and f"{phase}_grip_final" in r and (i is None or qq == i)]
                if not sel:
                    continue
                out["dim_attribution"][f"{phase}|{group}|{label}"] = {
                    "episodes": len(sel),
                    **{f"{dim}_{m}": round(float(np.mean([r[f"{phase}_{dim}_{m}"] for r in sel])), 4)
                       for dim in DIMS for m in ("final", "prior_only", "prior_sd", "push", "saturated")},
                }

    out["grasp_miss"] = {}
    for i, label in [(None, "all"), *enumerate(labels)]:
        sel = [r for r, qq in zip(rows, q) if r["outcome"] == "no_grasp" and (i is None or qq == i)]
        out["grasp_miss"][label] = {
            "episodes": len(sel),
            "dx": stat([r["miss_dx"] for r in sel]),
            "dy": stat([r["miss_dy"] for r in sel]),
            "xy": stat([r["miss_xy"] for r in sel]),
            "dz": stat([r["miss_dz"] for r in sel]),
            "closed_near_min": round(sum(r["miss_closed"] for r in sel) / max(1, len(sel)), 4),
            "object_out_of_overview": round(sum(not r["miss_object_in_overview"] for r in sel) / max(1, len(sel)), 4),
        }
    return out


def print_summary(s: Mapping[str, Any]) -> None:
    pct = lambda v: f"{100 * float(v):5.1f}%"  # noqa: E731
    f = lambda d, k="median": "    -  " if not d else f"{d[k]:+.3f}"  # noqa: E731
    print(f"\n=== {s['episodes']} episodes; target_y edges {s['y_edges']} ===")
    print("\noutcomes by target_y quartile:")
    print(f"  {'':5s}{'n':>5s}" + "".join(f"{o:>14s}" for o in OUTCOMES))
    for label, row in s["outcomes_by_y"].items():
        print(f"  {label:5s}{row['episodes']:5d}" + "".join(f"{pct(row[o]):>14s}" for o in OUTCOMES))
    print("\nslips: what the gripper was doing when contact was lost")
    print(f"  {'':5s}{'n':>5s}{'cmd_open':>10s}{'uncmd_open':>12s}{'passive':>9s}{'lift->slip':>12s}{'ee_z':>8s}"
          f"{'obj out ovw':>13s}{'dist/R':>8s}{'dx med':>8s}{'dy med':>8s}")
    for label, row in s["slip_types"].items():
        print(f"  {label:5s}{row['slips']:5d}{pct(row['commanded_open']):>10s}{pct(row['opened_uncommanded']):>12s}"
              f"{pct(row['passive']):>9s}{f(row['lift_to_slip_steps']):>12s}{f(row['slip_ee_z']):>8s}"
              f"{pct(row['object_out_of_overview_at_slip']):>13s}{f(row['dist_over_radius']):>8s}"
              f"{f(row['dx']):>8s}{f(row['dy']):>8s}")
    print("\nafter the lift (medians; fractions are of post-lift steps up to any slip):")
    print(f"  {'group|bin':20s}{'n':>5s}{'peak z':>8s}{'>band':>8s}{'obj@ovw':>9s}{'ee@ovw':>8s}{'rec@ovw':>9s}"
          f"{'rec@wrist':>10s}{'trk xyz':>9s}{'trk z':>8s}")
    for key, row in s["post_lift"].items():
        g = lambda k: f(row[k])  # noqa: E731
        print(f"  {key:20s}{row['episodes']:5d}{g('peak_ee_z_after_lift'):>8s}{g('above_band_frac'):>8s}"
              f"{g('object_in_overview_frac'):>9s}{g('ee_in_overview_frac'):>8s}{g('receptacle_in_overview_frac'):>9s}"
              f"{g('receptacle_in_wrist_frac'):>10s}{g('track_err_xyz_mean'):>9s}{g('track_err_z_mean'):>8s}")
    if s.get("z_attribution"):
        print("\nwho commands Z (action units, + = up; means of per-episode means):")
        print("  hover = before any grasp with the EE within 6 cm XY of the object;"
              " carry = after the lift until any slip")
        print(f"  {'phase|outcome|bin':26s}{'n':>5s}{'executed':>10s}{'prior only':>12s}{'residual push':>15s}{'up share':>10s}")
        m = lambda d: "     -  " if not d else f"{d['mean']:+.3f}"  # noqa: E731
        for key, row in s["z_attribution"].items():
            print(f"  {key:26s}{row['episodes']:5d}{m(row['z_final']):>10s}{m(row['z_prior_only']):>12s}"
                  f"{m(row['z_residual_push']):>15s}{m(row['z_up_share']):>10s}")
    if s.get("dim_attribution"):
        print("\nall action dims, per phase (means of per-episode means; grip > 0 = OPEN;"
              " preslip = last 3 decisions before a slip):")
        print("  each cell: executed / prior-only / residual push  [prior sd, push saturated share]")
        for key, row in s["dim_attribution"].items():
            if not key.endswith(("|all", "|Q1", "|Q4")):
                continue
            print(f"  {key} (n={row['episodes']})")
            for dim in DIMS:
                print(f"     {dim:5s} {row[f'{dim}_final']:+.3f} / {row[f'{dim}_prior_only']:+.3f} / "
                      f"{row[f'{dim}_push']:+.3f}   [{row[f'{dim}_prior_sd']:.3f}, {100 * row[f'{dim}_saturated']:.0f}%]")
    print("\ngrasp misses (never grasped): EE - object at closest XY approach (sd shows the spread)")
    print(f"  {'':5s}{'n':>5s}{'dx med':>8s}{'dx sd':>8s}{'dy med':>8s}{'dy sd':>8s}{'xy med':>8s}{'dz med':>8s}"
          f"{'closed':>8s}{'obj out':>9s}")
    for label, row in s["grasp_miss"].items():
        print(f"  {label:5s}{row['episodes']:5d}{f(row['dx']):>8s}{f(row['dx'], 'sd'):>8s}{f(row['dy']):>8s}"
              f"{f(row['dy'], 'sd'):>8s}{f(row['xy']):>8s}{f(row['dz']):>8s}{pct(row['closed_near_min']):>8s}"
              f"{pct(row['object_out_of_overview']):>9s}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dirs", type=Path, nargs="+")
    parser.add_argument("--y-edges", type=float, nargs="+", default=list(DEFAULT_Y_EDGES))
    parser.add_argument("--band-top", type=float, default=0.32,
                        help="Top of the camera-framing EE band (ee_workspace_z_bounds).")
    parser.add_argument("--open-window", type=int, default=4,
                        help="Env steps before contact loss searched for an open command.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for directory in args.trace_dirs:
        paths = sorted(directory.expanduser().resolve().glob("trace_round*.npz"))
        if not paths:
            raise SystemExit(f"No trace_round*.npz under {directory}.")
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                trace = {key: data[key] for key in data.files}
            rows.extend(episode_rows(trace, band_top=float(args.band_top), open_window=int(args.open_window)))
    summary = summarize(rows, edges=args.y_edges)
    print_summary(summary)
    if args.output is not None:
        out = args.output.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
        keys = sorted({k for r in rows for k in r})
        with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n=== wrote {out}/trace_summary.json and episodes.csv ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
