#!/usr/bin/env python3
"""Build an exactly balanced dataset from strict policy-success shards.

Selection is by whole episode, never by row. The default four target objects x
two destinations x 64 successes produces 512 distinct-scene demonstrations.
If any cell is short the build fails and reports the shortage; it never spills
an easier object's surplus into the missing quota.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (  # noqa: E402
    STAGE_MOVE_TO,
    STAGE_PICK_UP,
    STAGE_PLACEMENT,
)
from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import (  # noqa: E402
    SceneGeometryConfig,
)
from tools.audit.build_cdpr_staged_sft_dataset import (  # noqa: E402
    DATASET_SCHEMA,
    dataset_report,
    verify_frame_coverage,
)
from tools.audit.collect_cdpr_full_put_into import SCHEMA as RECORD_SCHEMA  # noqa: E402


DEFAULT_TARGETS = tuple(SceneGeometryConfig().target_catalogs)
DEFAULT_DESTINATIONS = tuple(SceneGeometryConfig().destinations)


def _scalar(data: Any, name: str) -> str:
    return str(np.asarray(data[name]).reshape(()).item())


def _selection_rank(scene_uid: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}|{scene_uid}".encode("utf-8")).hexdigest()


def scan_candidates(
    paths: Sequence[Path],
    *,
    target_catalogs: Sequence[str],
    destinations: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read only per-episode metadata and verify the strict contract."""

    expected_cells = {
        (str(catalog), str(destination))
        for catalog in target_catalogs
        for destination in destinations
    }
    candidates: list[dict[str, Any]] = []
    provenance: dict[str, str] = {}
    seen_episode: dict[str, Path] = {}
    seen_scene: dict[str, Path] = {}

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if _scalar(data, "schema") != RECORD_SCHEMA:
                raise ValueError(
                    f"{path} is not a {RECORD_SCHEMA!r} record shard."
                )
            required = {
                "episode_uid",
                "scene_uid",
                "destination",
                "target_catalog",
                "strict",
                "native",
                "grasped",
                "lifted",
                "released",
                "carry_slip",
                "wrong_place",
                "starts_grasped",
                "checkpoint_sha256",
                "scene_manifest_sha256",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"{path} is missing {missing}.")

            current = {
                "checkpoint_sha256": _scalar(data, "checkpoint_sha256"),
                "scene_manifest_sha256": _scalar(
                    data, "scene_manifest_sha256"
                ),
                "config": _scalar(data, "config"),
                "split": _scalar(data, "split"),
            }
            for key, value in current.items():
                previous = provenance.setdefault(key, value)
                if previous != value:
                    raise ValueError(
                        f"Mixed {key}: {previous!r} and {value!r} ({path})."
                    )
            if current["split"] != "collection":
                raise ValueError(
                    f"{path} comes from split {current['split']!r}; a training "
                    "bank must use the collection split."
                )

            strict = np.asarray(data["strict"], dtype=bool)
            verified = (
                np.asarray(data["native"], dtype=bool)
                & np.asarray(data["grasped"], dtype=bool)
                & np.asarray(data["lifted"], dtype=bool)
                & np.asarray(data["released"], dtype=bool)
                & ~np.asarray(data["carry_slip"], dtype=bool)
                & ~np.asarray(data["wrong_place"], dtype=bool)
            )
            if not bool(strict.all()) or not np.array_equal(strict, verified):
                raise ValueError(
                    f"{path} contains an episode that does not re-derive as "
                    "FullTaskOutcome.strict."
                )
            if bool(np.asarray(data["starts_grasped"], dtype=bool).any()):
                raise ValueError(f"{path} contains a non-empty start.")

            for column, (uid, scene, catalog, destination) in enumerate(
                zip(
                    data["episode_uid"],
                    data["scene_uid"],
                    data["target_catalog"],
                    data["destination"],
                )
            ):
                uid, scene = str(uid), str(scene)
                cell = (str(catalog), str(destination))
                if cell not in expected_cells:
                    raise ValueError(
                        f"{path} episode {uid} belongs to unexpected cell "
                        f"{cell}. Expected {sorted(expected_cells)}."
                    )
                if uid in seen_episode:
                    raise ValueError(
                        f"episode_uid {uid!r} appears in {seen_episode[uid]} "
                        f"and {path}."
                    )
                if scene in seen_scene:
                    raise ValueError(
                        f"scene_uid {scene!r} appears in two successful "
                        f"episodes ({seen_scene[scene]} and {path}); training "
                        "examples must be distinct scenes."
                    )
                seen_episode[uid], seen_scene[scene] = path, path
                candidates.append(
                    {
                        "path": path,
                        "column": int(column),
                        "episode_uid": uid,
                        "scene_uid": scene,
                        "target_catalog": cell[0],
                        "destination": cell[1],
                    }
                )
    if not candidates:
        raise ValueError("No strict-success candidates were found.")
    return candidates, provenance


def select_balanced(
    candidates: Sequence[Mapping[str, Any]],
    *,
    target_catalogs: Sequence[str],
    destinations: Sequence[str],
    successes_per_cell: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stable whole-episode sampling with a hard quota in every cell."""

    if int(successes_per_cell) < 1:
        raise ValueError("--successes-per-cell must be positive.")
    selected: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    shortages: dict[str, int] = {}
    for catalog in target_catalogs:
        for destination in destinations:
            key = f"{catalog}/{destination}"
            available = [
                dict(row)
                for row in candidates
                if row["target_catalog"] == catalog
                and row["destination"] == destination
            ]
            available.sort(
                key=lambda row: _selection_rank(row["scene_uid"], int(seed))
            )
            take = available[: int(successes_per_cell)]
            selected.extend(take)
            cells[key] = {
                "available": len(available),
                "selected": len(take),
            }
            if len(take) < int(successes_per_cell):
                shortages[key] = int(successes_per_cell) - len(take)
    selected.sort(key=lambda row: str(row["episode_uid"]))
    return selected, {
        "successes_per_cell": int(successes_per_cell),
        "selection_seed": int(seed),
        "cells": cells,
        "shortages": shortages,
        "available_total": len(candidates),
        "selected_total": len(selected),
    }


def _stage_for_decision(
    decision: int, *, grasp_step: int, lift_step: int, per: int
) -> tuple[int, int, str]:
    grasp_decision = int(grasp_step) // int(per)
    lift_decision = int(lift_step) // int(per)
    if int(decision) < grasp_decision:
        return STAGE_MOVE_TO, 0, "move_to"
    if int(decision) <= lift_decision:
        return STAGE_PICK_UP, 1, "pick_up"
    return STAGE_PLACEMENT, 2, "placement"


def build_rows(selected: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Materialize selected trajectories as one row per policy decision."""

    columns: dict[str, list[Any]] = {}

    def add(name: str, value: Any) -> None:
        columns.setdefault(name, []).append(value)

    by_path: dict[Path, list[Mapping[str, Any]]] = {}
    for row in selected:
        by_path.setdefault(Path(row["path"]), []).append(row)

    expected_shapes: dict[str, tuple[int, ...]] = {}
    for path, episodes in sorted(by_path.items(), key=lambda item: str(item[0])):
        with np.load(path, allow_pickle=False) as data:
            for episode in episodes:
                world = int(episode["column"])
                active = np.asarray(data["decision_active"][:, world], dtype=bool)
                decisions = np.flatnonzero(active)
                if decisions.size == 0 or not np.array_equal(
                    decisions, np.arange(decisions.size)
                ):
                    raise ValueError(
                        f"{episode['episode_uid']} has non-contiguous active "
                        "policy decisions."
                    )
                action = np.asarray(data["action"][:, world], dtype=np.float32)
                mask = np.asarray(data["action_mask"][:, world], dtype=bool)
                per = int(action.shape[1])
                grasp = int(data["first_grasp_step"][world])
                lift = int(data["first_lift_step"][world])
                release = int(data["first_release_step"][world])
                success = int(data["first_strict_step"][world])
                if not (0 <= grasp <= lift <= release <= success):
                    raise ValueError(
                        f"{episode['episode_uid']} has non-monotone events: "
                        f"grasp={grasp}, lift={lift}, release={release}, "
                        f"strict={success}."
                    )
                event_decisions = [grasp // per, lift // per, success // per]
                for decision in decisions.tolist():
                    if not bool(mask[decision].any()):
                        raise ValueError(
                            f"{episode['episode_uid']} decision {decision} is "
                            "active but supervises no executed action."
                        )
                    stage_raw, stage_id, stage_name = _stage_for_decision(
                        decision, grasp_step=grasp, lift_step=lift, per=per
                    )
                    add(
                        "state",
                        np.asarray(
                            data["state"][decision, world], dtype=np.float32
                        ),
                    )
                    add(
                        "prior",
                        np.asarray(
                            data["prior"][decision, world], dtype=np.float32
                        ),
                    )
                    add("action", action[decision])
                    add("action_mask", mask[decision])
                    add("instruction_id", int(data["instruction_id"][world]))
                    add(
                        "instruction_text",
                        str(data["instruction_text"][world]),
                    )
                    add(
                        "teacher_instruction_text",
                        str(data["instruction_text"][world]),
                    )
                    add("episode_uid", str(data["episode_uid"][world]))
                    add("parent_episode_uid", str(data["episode_uid"][world]))
                    add("decision_index", int(decision))
                    add("scene_uid", str(data["scene_uid"][world]))
                    add(
                        "split",
                        str(np.asarray(data["split"]).reshape(()).item()),
                    )
                    add("rollout_index", 0)
                    add("destination", str(data["destination"][world]))
                    add("target_catalog", str(data["target_catalog"][world]))
                    add("substage_id", int(stage_raw))
                    add("stage_id", int(stage_id))
                    add("stage_name", stage_name)
                    add(
                        "source_checkpoint_sha256",
                        _scalar(data, "checkpoint_sha256"),
                    )
                    add("frame_uid", f"{data['episode_uid'][world]}#{decision}")
                    add(
                        "stage_boundary_distance",
                        min(abs(int(decision) - value) for value in event_decisions),
                    )
                    add("full_chain_success", True)
                    add("source_group", "strict_policy")
                    add("starts_grasped", False)

            for key in ("state", "prior", "action"):
                shape = tuple(np.asarray(data[key]).shape[2:])
                previous = expected_shapes.setdefault(key, shape)
                if previous != shape:
                    raise ValueError(
                        f"Mixed {key} shapes: {previous} and {shape} ({path})."
                    )

    dtypes: dict[str, Any] = {
        "state": np.float32,
        "prior": np.float32,
        "action": np.float32,
        "action_mask": bool,
        "instruction_id": np.int64,
        "instruction_text": "U256",
        "teacher_instruction_text": "U256",
        "episode_uid": "U160",
        "parent_episode_uid": "U160",
        "decision_index": np.int64,
        "scene_uid": "U96",
        "split": "U32",
        "rollout_index": np.int64,
        "destination": "U16",
        "target_catalog": "U64",
        "substage_id": np.int8,
        "stage_id": np.int8,
        "stage_name": "U16",
        "source_checkpoint_sha256": "U128",
        "frame_uid": "U224",
        "stage_boundary_distance": np.int32,
        "full_chain_success": bool,
        "source_group": "U32",
        "starts_grasped": bool,
    }
    return {
        name: np.asarray(values, dtype=dtypes[name])
        for name, values in columns.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--frames", type=Path, nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successes-per-cell", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=20260922)
    parser.add_argument(
        "--target-catalogs", nargs="+", default=list(DEFAULT_TARGETS)
    )
    parser.add_argument(
        "--destinations", nargs="+", default=list(DEFAULT_DESTINATIONS)
    )
    parser.add_argument(
        "--allow-missing-frames",
        action="store_true",
        help="Build an action-only dataset deliberately.",
    )
    args = parser.parse_args(argv)

    paths = sorted({path.expanduser().resolve() for path in args.records})
    if not paths:
        raise SystemExit("--records matched nothing.")
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise SystemExit(f"Record paths do not exist: {missing_paths[:3]}")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        candidates, provenance = scan_candidates(
            paths,
            target_catalogs=args.target_catalogs,
            destinations=args.destinations,
        )
        selected, selection = select_balanced(
            candidates,
            target_catalogs=args.target_catalogs,
            destinations=args.destinations,
            successes_per_cell=int(args.successes_per_cell),
            seed=int(args.selection_seed),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    availability_path = output / "availability.json"
    availability_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    if selection["shortages"]:
        raise SystemExit(
            "Strict-success cells are short: "
            f"{selection['shortages']}. See {availability_path}; collect more "
            "distinct collection scenes before building."
        )

    try:
        dataset = build_rows(selected)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    frame_report: dict[str, Any] | None = None
    frame_paths = sorted(
        {path.expanduser().resolve() for path in args.frames if path.is_file()}
    )
    if frame_paths:
        frame_report = verify_frame_coverage(dataset, frame_paths)
        if (
            frame_report["resolved_fraction"] < 1.0
            and not bool(args.allow_missing_frames)
        ):
            raise SystemExit(
                "Not every selected row has its exact policy frame: "
                f"{frame_report['unresolved_examples']}."
            )
    elif not bool(args.allow_missing_frames):
        raise SystemExit(
            "No --frames were supplied. Pass every frames_*.npz shard, or "
            "use --allow-missing-frames for an action-only audit dataset."
        )

    np.savez_compressed(output / "demonstrations.npz", **dataset)
    episodes = [str(row["episode_uid"]) for row in selected]
    selection_sha256 = hashlib.sha256(
        "\n".join(sorted(episodes)).encode("utf-8")
    ).hexdigest()
    report = {
        "schema": DATASET_SCHEMA,
        "source_schema": RECORD_SCHEMA,
        "priors_stale": False,
        "priors_stale_reason": None,
        "records": [str(path) for path in paths],
        "frames_files": [str(path) for path in frame_paths],
        "provenance": provenance,
        "selection": selection,
        "selected_episode_uid_sha256": selection_sha256,
        "dataset": dataset_report(dataset),
        "frames": frame_report,
        "contract": {
            "empty_start": True,
            "one_final_instruction": True,
            "continuous_single_policy": True,
            "simulator_recovery": False,
            "strict_success_only": True,
            "distinct_scenes": True,
            "balanced_by": ["target_catalog", "destination"],
        },
        "notes": [
            "state and prior were produced by the same checkpoint and final "
            "prompt recorded in each row, so they are not stale for that "
            "checkpoint.",
            "Refresh from the stored frames before training a different "
            "checkpoint or a changed prompt.",
        ],
    }
    (output / "dataset.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "selected_episodes.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    census = report["dataset"]
    print(
        f"[dataset] wrote {output / 'demonstrations.npz'}: "
        f"{census['episodes']} strict episodes, {census['scenes']} distinct "
        f"scenes, {census['rows']} decision rows",
        flush=True,
    )
    print(f"[dataset] cells {selection['cells']}", flush=True)
    if frame_report is not None:
        print(
            f"[dataset] frames resolve {frame_report['resolved']}/"
            f"{frame_report['rows']} rows",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
