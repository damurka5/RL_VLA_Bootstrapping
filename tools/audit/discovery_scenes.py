#!/usr/bin/env python3
"""Find scenes the deterministic policy fails, and measure what sampling finds.

Self-imitation on the policy's own deterministic successes has nothing to
teach: those actions ARE the policy's mean, and the only target error left is
the prior's per-forward noise. Five fits on two checkpoints never predicted a
held-out success better than the untouched policy (2026-09-23/25/27). New
supervision has to come from scenes where the deterministic mean FAILS but the
sampled GRPO behaviour policy sometimes succeeds.

``select``  reads a deterministic harvest's ``attempts_*.npz`` and classifies
            every failed scene by how far it got. The failure modes the
            campaign cares about each get a name:

            ``no_grasp``     never grasped
            ``failed_lift``  grasped, never held a lift
            ``carry_slip``   lifted, then lost the object
            ``placement``    lifted and carried, but no strict placement
                             (release timing, wrong place, bowl misses)

            It writes a scene list, stratified over mode x destination, for
            ``collect_cdpr_full_put_into.py --scene-uids ... --repeats M
            --stochastic-seed S``.

``yield``   joins the deterministic verdicts with the stochastic attempts and
            reports, per mode, destination and object: attempts, strict
            successes per attempt, and the share of scenes solved at least once.
            That number decides whether a discovered-solution bank is worth
            building, before any is built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILURE_MODES = ("no_grasp", "failed_lift", "carry_slip", "placement")


def failure_mode(row: Mapping[str, Any]) -> str | None:
    """None for a strict success or a diverged episode, else the mode."""

    if bool(row.get("non_finite", False)) or bool(row["strict"]):
        return None
    if not bool(row["grasped"]):
        return "no_grasp"
    if not bool(row["lifted"]):
        return "failed_lift"
    if bool(row["carry_slip"]):
        return "carry_slip"
    return "placement"


def read_attempts(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted({Path(p).expanduser().resolve() for p in paths}):
        with np.load(path, allow_pickle=False) as data:
            columns = {name: np.asarray(data[name]) for name in data.files if name != "schema"}
        count = int(columns["scene_uid"].shape[0])
        for index in range(count):
            row = {name: values[index].item() if values[index].shape == () else values[index]
                   for name, values in columns.items()}
            row.setdefault("repeat_index", 0)
            row.setdefault("rollout_mode", "deterministic")
            row.setdefault("non_finite", False)
            row["source"] = str(path)
            rows.append(row)
    if not rows:
        raise SystemExit("No attempts were read.")
    return rows


def _rank(uid: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}|{uid}".encode("utf-8")).hexdigest()


def select_scenes(
    rows: Sequence[Mapping[str, Any]],
    *,
    modes: Sequence[str],
    max_scenes: int,
    seed: int,
) -> dict[str, Any]:
    """Deterministic failures, stratified as evenly as mode x destination allow."""

    by_scene: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        uid = str(row["scene_uid"])
        if uid in by_scene:
            raise SystemExit(f"Scene {uid} was attempted twice in the deterministic harvest.")
        by_scene[uid] = row
    cells: dict[tuple[str, str], list[str]] = {}
    for uid, row in by_scene.items():
        mode = failure_mode(row)
        if mode is None or mode not in modes:
            continue
        cells.setdefault((mode, str(row["destination"])), []).append(uid)
    for key in cells:
        cells[key].sort(key=lambda uid: _rank(uid, seed))
    available = {f"{mode}/{dest}": len(uids) for (mode, dest), uids in sorted(cells.items())}
    chosen: list[str] = []
    if int(max_scenes) <= 0:
        chosen = [uid for uids in cells.values() for uid in uids]
    else:
        # Round-robin over cells: small cells are taken whole, large ones share
        # what is left. Oversampling the rare failure modes relative to their
        # natural frequency is the point.
        queues = {key: list(uids) for key, uids in sorted(cells.items())}
        while len(chosen) < int(max_scenes) and any(queues.values()):
            for key in list(queues):
                if queues[key] and len(chosen) < int(max_scenes):
                    chosen.append(queues[key].pop(0))
    chosen.sort(key=lambda uid: _rank(uid, seed))
    modes_of = {uid: failure_mode(by_scene[uid]) for uid in chosen}
    selected_counts: dict[str, int] = {}
    for uid in chosen:
        key = f"{modes_of[uid]}/{by_scene[uid]['destination']}"
        selected_counts[key] = selected_counts.get(key, 0) + 1
    return {
        "scene_uids": chosen,
        "failure_mode": modes_of,
        "destination": {uid: str(by_scene[uid]["destination"]) for uid in chosen},
        "target_catalog": {uid: str(by_scene[uid]["target_catalog"]) for uid in chosen},
        "available_by_mode_destination": available,
        "selected_by_mode_destination": dict(sorted(selected_counts.items())),
        "deterministic_scenes": len(by_scene),
        "deterministic_strict": int(sum(bool(r["strict"]) for r in by_scene.values())),
        "modes": list(modes),
        "seed": int(seed),
    }


def discovery_yield(
    selection: Mapping[str, Any], stochastic: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Per failure mode: how often sampling solves what the mean failed."""

    wanted = set(selection["scene_uids"])
    attempts: dict[str, list[Mapping[str, Any]]] = {}
    for row in stochastic:
        if str(row.get("rollout_mode")) != "stochastic":
            raise SystemExit(f"{row['source']} holds a non-stochastic attempt.")
        uid = str(row["scene_uid"])
        if uid not in wanted:
            raise SystemExit(f"Stochastic attempt on unselected scene {uid} ({row['source']}).")
        attempts.setdefault(uid, []).append(row)

    def summarize(uids: Sequence[str]) -> dict[str, Any]:
        tries = [row for uid in uids for row in attempts.get(uid, [])]
        strict = sum(bool(row["strict"]) for row in tries)
        solved = sum(any(bool(row["strict"]) for row in attempts.get(uid, [])) for uid in uids)
        native = sum(bool(row["native"]) for row in tries)
        diverged = sum(bool(row.get("non_finite", False)) for row in tries)
        attempted = sum(1 for uid in uids if attempts.get(uid))
        return {
            "scenes": len(uids),
            "scenes_attempted": attempted,
            "attempts": len(tries),
            "strict_successes": strict,
            "strict_per_attempt": round(strict / len(tries), 4) if tries else None,
            "native_per_attempt": round(native / len(tries), 4) if tries else None,
            "scenes_solved_at_least_once": solved,
            "solved_scene_fraction": round(solved / attempted, 4) if attempted else None,
            "non_finite_attempts": diverged,
        }

    modes = selection["failure_mode"]
    report: dict[str, Any] = {"all": summarize(list(wanted))}
    for column in ("mode", "destination", "target_catalog", "mode_destination"):
        table: dict[str, Any] = {}
        groups: dict[str, list[str]] = {}
        for uid in wanted:
            mode = modes[uid]
            key = {
                "mode": mode,
                "destination": selection["destination"][uid],
                "target_catalog": selection["target_catalog"][uid],
                "mode_destination": f"{mode}/{selection['destination'][uid]}",
            }[column]
            groups.setdefault(key, []).append(uid)
        for key, uids in sorted(groups.items()):
            table[key] = summarize(uids)
        report[f"by_{column}"] = table
    repeats = sorted({int(row["repeat_index"]) for rows in attempts.values() for row in rows})
    report["repeats_seen"] = repeats
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="Classify deterministic failures into a scene list.")
    select.add_argument("--attempts", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--modes", nargs="+", choices=FAILURE_MODES, default=list(FAILURE_MODES))
    select.add_argument("--max-scenes", type=int, default=1024, help="0 keeps every failure.")
    select.add_argument("--seed", type=int, default=20260927)
    measure = sub.add_parser("yield", help="Measure stochastic discovery on the selected scenes.")
    measure.add_argument("--selection", type=Path, required=True)
    measure.add_argument("--attempts", type=Path, nargs="+", required=True)
    measure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "select":
        rows = read_attempts(args.attempts)
        modes_seen = {str(row.get("rollout_mode")) for row in rows}
        if modes_seen != {"deterministic"}:
            raise SystemExit(f"select needs a deterministic harvest; saw {sorted(modes_seen)}.")
        result = select_scenes(rows, modes=args.modes, max_scenes=int(args.max_scenes), seed=int(args.seed))
        result["attempts_files"] = len({row["source"] for row in rows})
        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().resolve().write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"[discovery] {len(result['scene_uids'])} hard scenes of "
            f"{result['deterministic_scenes']} deterministic "
            f"({result['deterministic_strict']} strict); selected "
            f"{result['selected_by_mode_destination']}",
            flush=True,
        )
        return 0

    selection = json.loads(args.selection.expanduser().resolve().read_text(encoding="utf-8"))
    report = discovery_yield(selection, read_attempts(args.attempts))
    args.output.expanduser().resolve().write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[discovery] all: {report['all']}", flush=True)
    for mode, row in report["by_mode"].items():
        print(
            f"[discovery] {mode}: {row['scenes_solved_at_least_once']}/{row['scenes_attempted']} "
            f"scenes solved, {row['strict_successes']}/{row['attempts']} strict attempts",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
