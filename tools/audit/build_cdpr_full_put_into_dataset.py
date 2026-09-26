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
from tools.audit.collect_cdpr_full_put_into import (  # noqa: E402
    RETENTION_SCHEMA,
    SCHEMA as RECORD_SCHEMA,
    spread_evenly,
    stage_decisions,
)


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
    grouped_paths = sorted(by_path.items(), key=lambda item: str(item[0]))
    for file_index, (path, episodes) in enumerate(grouped_paths, start=1):
        if file_index == 1 or file_index % 16 == 0 or file_index == len(grouped_paths):
            print(
                f"[dataset] materializing record shard {file_index}/"
                f"{len(grouped_paths)}",
                flush=True,
            )
        with np.load(path, allow_pickle=False) as data:
            # An NPZ member is decompressed on every ``data[name]`` access.
            # Load each member ONCE per file. Indexing the archive in the inner
            # decision loop used to decompress state/prior thousands of times
            # and every appended view retained its full backing array, growing
            # a 512-episode build to tens of gigabytes.
            states = np.asarray(data["state"], dtype=np.float32)
            priors = np.asarray(data["prior"], dtype=np.float32)
            actions = np.asarray(data["action"], dtype=np.float32)
            action_masks = np.asarray(data["action_mask"], dtype=bool)
            decision_active = np.asarray(data["decision_active"], dtype=bool)
            first_grasp = np.asarray(data["first_grasp_step"], dtype=np.int64)
            first_lift = np.asarray(data["first_lift_step"], dtype=np.int64)
            first_release = np.asarray(data["first_release_step"], dtype=np.int64)
            first_strict = np.asarray(data["first_strict_step"], dtype=np.int64)
            instruction_ids = np.asarray(data["instruction_id"], dtype=np.int64)
            instruction_texts = np.asarray(data["instruction_text"])
            episode_uids = np.asarray(data["episode_uid"])
            scene_uids = np.asarray(data["scene_uid"])
            destinations = np.asarray(data["destination"])
            target_catalogs = np.asarray(data["target_catalog"])
            split = _scalar(data, "split")
            checkpoint_sha256 = _scalar(data, "checkpoint_sha256")

            for key, value in (
                ("state", states),
                ("prior", priors),
                ("action", actions),
            ):
                shape = tuple(value.shape[2:])
                previous = expected_shapes.setdefault(key, shape)
                if previous != shape:
                    raise ValueError(
                        f"Mixed {key} shapes: {previous} and {shape} ({path})."
                    )

            for episode in episodes:
                world = int(episode["column"])
                active = decision_active[:, world]
                decisions = np.flatnonzero(active)
                if decisions.size == 0 or not np.array_equal(
                    decisions, np.arange(decisions.size)
                ):
                    raise ValueError(
                        f"{episode['episode_uid']} has non-contiguous active "
                        "policy decisions."
                    )
                action = actions[:, world]
                mask = action_masks[:, world]
                per = int(action.shape[1])
                grasp = int(first_grasp[world])
                lift = int(first_lift[world])
                recorded_release = int(first_release[world])
                success = int(first_strict[world])
                # Collector builds through d679edd timestamped the raw
                # placement predicate's `released` diagnostic. An empty open
                # hand can satisfy that diagnostic at step zero, even though
                # FullTaskOutcome correctly refuses to latch a release until
                # after a grasp. The trajectory verdict is still strict and
                # the executed data are intact; only this event timestamp is
                # early. A strict success proves the latched release no later
                # than its success step, so use that conservative boundary.
                release_repaired = recorded_release < lift
                release = success if release_repaired else recorded_release
                if not (0 <= grasp <= lift <= release <= success):
                    raise ValueError(
                        f"{episode['episode_uid']} has non-monotone events: "
                        f"grasp={grasp}, lift={lift}, "
                        f"release={recorded_release}, "
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
                        np.array(states[decision, world], copy=True),
                    )
                    add(
                        "prior",
                        np.array(priors[decision, world], copy=True),
                    )
                    add("action", np.array(action[decision], copy=True))
                    add("action_mask", np.array(mask[decision], copy=True))
                    add("instruction_id", int(instruction_ids[world]))
                    add(
                        "instruction_text",
                        str(instruction_texts[world]),
                    )
                    add(
                        "teacher_instruction_text",
                        str(instruction_texts[world]),
                    )
                    add("episode_uid", str(episode_uids[world]))
                    add("parent_episode_uid", str(episode_uids[world]))
                    add("decision_index", int(decision))
                    add("scene_uid", str(scene_uids[world]))
                    add("split", split)
                    add("rollout_index", 0)
                    add("destination", str(destinations[world]))
                    add("target_catalog", str(target_catalogs[world]))
                    add("substage_id", int(stage_raw))
                    add("stage_id", int(stage_id))
                    add("stage_name", stage_name)
                    add(
                        "source_checkpoint_sha256",
                        checkpoint_sha256,
                    )
                    add("frame_uid", f"{episode_uids[world]}#{decision}")
                    add(
                        "stage_boundary_distance",
                        min(abs(int(decision) - value) for value in event_decisions),
                    )
                    add("full_chain_success", True)
                    add("release_event_repaired", release_repaired)
                    add("source_group", "strict_policy")
                    add("starts_grasped", False)

    return {
        name: np.asarray(values, dtype=ROW_DTYPES[name])
        for name, values in columns.items()
    }


ROW_DTYPES: dict[str, Any] = {
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
    "release_event_repaired": bool,
    "source_group": "U32",
    "starts_grasped": bool,
}

STAGE_BY_ID = {
    0: (STAGE_MOVE_TO, "move_to"),
    1: (STAGE_PICK_UP, "pick_up"),
    2: (STAGE_PLACEMENT, "placement"),
}


def scan_retention_candidates(
    *,
    retention_paths: Sequence[Path],
    strict_paths: Sequence[Path],
    exclude_episodes: set[str],
    exclude_scenes: set[str],
    rows_per_stage: int,
    target_catalogs: Sequence[str],
    destinations: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, int]]:
    """Every row a retention bank may use, one entry per policy decision.

    Two sources, both recorded by the same checkpoint under the same config:

    ``retention_nonstrict``  the approach (and, if it lifted, the pickup) of
        an episode that grasped but was not a strict success -- exactly the
        decisions the collector marked.
    ``retention_strict_surplus``  up to ``rows_per_stage`` decisions of every
        stage of a strict success the SFT bank did NOT select. This is the only
        source of placement rows.

    Episodes and scenes in the SFT bank are excluded, so the two banks never
    share a scene.
    """

    expected = {(str(c), str(d)) for c in target_catalogs for d in destinations}
    provenance: dict[str, str] = {}
    candidates: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    excluded = {"episodes": 0, "scenes": 0}

    def check_provenance(data: Any, path: Path) -> None:
        current = {
            "checkpoint_sha256": _scalar(data, "checkpoint_sha256"),
            "scene_manifest_sha256": _scalar(data, "scene_manifest_sha256"),
            "config": _scalar(data, "config"),
            "split": _scalar(data, "split"),
        }
        for key, value in current.items():
            previous = provenance.setdefault(key, value)
            if previous != value:
                raise ValueError(f"Mixed {key}: {previous!r} and {value!r} ({path}).")
        if current["split"] != "collection":
            raise ValueError(f"{path} comes from split {current['split']!r}, not collection.")

    def add_rows(path, data, column, decisions, stage_of, group, strict) -> None:
        uid = str(data["episode_uid"][column])
        scene = str(data["scene_uid"][column])
        cell = (str(data["target_catalog"][column]), str(data["destination"][column]))
        if cell not in expected:
            raise ValueError(f"{path} episode {uid} is in unexpected cell {cell}.")
        if uid in seen:
            raise ValueError(f"episode_uid {uid!r} appears in {seen[uid]} and {path}.")
        seen[uid] = path
        for decision in decisions:
            candidates.append(
                {
                    "path": path,
                    "column": int(column),
                    "decision": int(decision),
                    "stage_id": int(stage_of[int(decision)]),
                    "episode_uid": uid,
                    "scene_uid": scene,
                    "target_catalog": cell[0],
                    "destination": cell[1],
                    "source_group": group,
                    "full_chain_success": bool(strict),
                }
            )

    def skip(data, column) -> bool:
        uid = str(data["episode_uid"][column])
        scene = str(data["scene_uid"][column])
        if uid in exclude_episodes:
            excluded["episodes"] += 1
            return True
        if scene in exclude_scenes:
            excluded["scenes"] += 1
            return True
        return False

    for path in retention_paths:
        with np.load(path, allow_pickle=False) as data:
            if _scalar(data, "schema") != RETENTION_SCHEMA:
                raise ValueError(f"{path} is not a {RETENTION_SCHEMA!r} shard.")
            check_provenance(data, path)
            if bool(np.asarray(data["strict"], dtype=bool).any()):
                raise ValueError(f"{path} holds a strict episode; those belong to record shards.")
            mask = np.asarray(data["retention_decision_mask"], dtype=bool)
            active = np.asarray(data["decision_active"], dtype=bool)
            per = int(np.asarray(data["action"]).shape[2])
            for column in range(mask.shape[1]):
                if skip(data, column):
                    continue
                stages = stage_decisions(
                    active_decisions=int(active[:, column].sum()),
                    grasp_step=int(data["first_grasp_step"][column]),
                    lift_step=int(data["first_lift_step"][column]),
                    per=per,
                    include_placement=False,
                )
                stage_of = {int(d): stage for stage, rows in stages.items() for d in rows}
                decisions = np.flatnonzero(mask[:, column])
                missing = [int(d) for d in decisions if int(d) not in stage_of]
                if missing:
                    raise ValueError(
                        f"{path} column {column} marks decisions {missing[:4]} "
                        "outside every completed stage."
                    )
                add_rows(path, data, column, decisions, stage_of, "retention_nonstrict", False)

    for path in strict_paths:
        with np.load(path, allow_pickle=False) as data:
            if _scalar(data, "schema") != RECORD_SCHEMA:
                raise ValueError(f"{path} is not a {RECORD_SCHEMA!r} record shard.")
            check_provenance(data, path)
            active = np.asarray(data["decision_active"], dtype=bool)
            per = int(np.asarray(data["action"]).shape[2])
            for column in range(active.shape[1]):
                if not bool(data["strict"][column]) or skip(data, column):
                    continue
                stages = stage_decisions(
                    active_decisions=int(active[:, column].sum()),
                    grasp_step=int(data["first_grasp_step"][column]),
                    lift_step=int(data["first_lift_step"][column]),
                    per=per,
                    include_placement=True,
                )
                stage_of = {int(d): stage for stage, rows in stages.items() for d in rows}
                decisions = np.concatenate(
                    [spread_evenly(rows, int(rows_per_stage)) for rows in stages.values()]
                    or [np.empty(0, dtype=np.int64)]
                )
                add_rows(path, data, column, np.sort(decisions), stage_of,
                         "retention_strict_surplus", True)
    if not candidates:
        raise ValueError("No retention rows were found.")
    return candidates, provenance, excluded


def select_rows_balanced(
    candidates: Sequence[Mapping[str, Any]],
    *,
    target_catalogs: Sequence[str],
    destinations: Sequence[str],
    rows_per_cell_stage: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exactly ``rows_per_cell_stage`` rows in every object x destination x stage.

    Equal ROWS, so that equal rows can be checked against equal LOSS in the
    SFT report rather than assumed. A short cell fails the build; it is never
    filled from another cell.
    """

    if int(rows_per_cell_stage) < 1:
        raise ValueError("--rows-per-cell-stage must be positive.")
    selected: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    shortages: dict[str, int] = {}
    for catalog in target_catalogs:
        for destination in destinations:
            for stage_id, (_, stage_name) in STAGE_BY_ID.items():
                key = f"{catalog}/{destination}/{stage_name}"
                available = [
                    dict(row)
                    for row in candidates
                    if row["target_catalog"] == catalog
                    and row["destination"] == destination
                    and int(row["stage_id"]) == stage_id
                ]
                available.sort(
                    key=lambda row: _selection_rank(
                        f"{row['episode_uid']}#{row['decision']}", int(seed)
                    )
                )
                take = available[: int(rows_per_cell_stage)]
                selected.extend(take)
                groups: dict[str, int] = {}
                for row in take:
                    groups[row["source_group"]] = groups.get(row["source_group"], 0) + 1
                cells[key] = {
                    "available": len(available),
                    "selected": len(take),
                    "episodes": len({row["episode_uid"] for row in take}),
                    "by_source_group": groups,
                }
                if len(take) < int(rows_per_cell_stage):
                    shortages[key] = int(rows_per_cell_stage) - len(take)
    selected.sort(key=lambda row: (str(row["episode_uid"]), int(row["decision"])))
    return selected, {
        "rows_per_cell_stage": int(rows_per_cell_stage),
        "selection_seed": int(seed),
        "cells": cells,
        "shortages": shortages,
        "available_total": len(candidates),
        "selected_total": len(selected),
    }


def build_retention_rows(selected: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """One row per selected decision, in the strict bank's exact column set."""

    columns: dict[str, list[Any]] = {}

    def add(name: str, value: Any) -> None:
        columns.setdefault(name, []).append(value)

    by_path: dict[Path, list[Mapping[str, Any]]] = {}
    for row in selected:
        by_path.setdefault(Path(row["path"]), []).append(row)
    for path, rows in sorted(by_path.items(), key=lambda item: str(item[0])):
        with np.load(path, allow_pickle=False) as data:
            states = np.asarray(data["state"], dtype=np.float32)
            priors = np.asarray(data["prior"], dtype=np.float32)
            actions = np.asarray(data["action"], dtype=np.float32)
            action_masks = np.asarray(data["action_mask"], dtype=bool)
            instruction_ids = np.asarray(data["instruction_id"], dtype=np.int64)
            instruction_texts = np.asarray(data["instruction_text"])
            split = _scalar(data, "split")
            checkpoint_sha256 = _scalar(data, "checkpoint_sha256")
            for row in rows:
                world, decision = int(row["column"]), int(row["decision"])
                if not bool(action_masks[decision, world].any()):
                    raise ValueError(
                        f"{row['episode_uid']} decision {decision} supervises no "
                        "executed action."
                    )
                stage_raw, stage_name = STAGE_BY_ID[int(row["stage_id"])]
                add("state", np.array(states[decision, world], copy=True))
                add("prior", np.array(priors[decision, world], copy=True))
                add("action", np.array(actions[decision, world], copy=True))
                add("action_mask", np.array(action_masks[decision, world], copy=True))
                add("instruction_id", int(instruction_ids[world]))
                add("instruction_text", str(instruction_texts[world]))
                add("teacher_instruction_text", str(instruction_texts[world]))
                add("episode_uid", str(row["episode_uid"]))
                add("parent_episode_uid", str(row["episode_uid"]))
                add("decision_index", decision)
                add("scene_uid", str(row["scene_uid"]))
                add("split", split)
                add("rollout_index", 0)
                add("destination", str(row["destination"]))
                add("target_catalog", str(row["target_catalog"]))
                add("substage_id", int(stage_raw))
                add("stage_id", int(row["stage_id"]))
                add("stage_name", stage_name)
                add("source_checkpoint_sha256", checkpoint_sha256)
                add("frame_uid", f"{row['episode_uid']}#{decision}")
                add("stage_boundary_distance", 0)
                add("full_chain_success", bool(row["full_chain_success"]))
                add("release_event_repaired", False)
                add("source_group", str(row["source_group"]))
                add("starts_grasped", False)
    return {
        name: np.asarray(values, dtype=ROW_DTYPES[name])
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
    parser.add_argument(
        "--mode",
        choices=("strict", "retention"),
        default="strict",
        help=(
            "strict: the balanced strict-success SFT bank (default). "
            "retention: a retention bank from --retention-records plus the "
            "strict episodes in --records that --exclude-dataset did not use."
        ),
    )
    parser.add_argument("--retention-records", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--exclude-dataset",
        type=Path,
        default=None,
        help="The SFT bank; its episodes and scenes never enter retention.",
    )
    parser.add_argument("--rows-per-cell-stage", type=int, default=128)
    parser.add_argument(
        "--retention-rows-per-stage",
        type=int,
        default=8,
        help="Cap on decisions per stage taken from one surplus strict episode.",
    )
    args = parser.parse_args(argv)
    if args.mode == "retention":
        return build_retention_main(args)

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
        "legacy_trace_repairs": {
            "release_before_lift_episodes": int(
                np.unique(
                    dataset["episode_uid"][dataset["release_event_repaired"]]
                ).size
            ),
            "rule": (
                "For d679edd-era shards whose raw release timestamp precedes "
                "lift, use the strict-success step as the conservative release "
                "boundary. Actions, masks, frames, and strict verdicts are "
                "unchanged."
            ),
        },
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


def build_retention_main(args: argparse.Namespace) -> int:
    """``--mode retention``: see scan_retention_candidates."""

    retention_paths = sorted({path.expanduser().resolve() for path in args.retention_records})
    strict_paths = sorted({path.expanduser().resolve() for path in args.records})
    if not retention_paths:
        raise SystemExit("--mode retention needs --retention-records.")
    if args.exclude_dataset is None:
        raise SystemExit(
            "--mode retention needs --exclude-dataset (the SFT bank), or the "
            "retention bank could share scenes with it."
        )
    with np.load(args.exclude_dataset.expanduser().resolve(), allow_pickle=False) as bank:
        exclude_episodes = {str(value) for value in np.unique(bank["episode_uid"])}
        exclude_scenes = {str(value) for value in np.unique(bank["scene_uid"])}
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        candidates, provenance, excluded = scan_retention_candidates(
            retention_paths=retention_paths,
            strict_paths=strict_paths,
            exclude_episodes=exclude_episodes,
            exclude_scenes=exclude_scenes,
            rows_per_stage=int(args.retention_rows_per_stage),
            target_catalogs=args.target_catalogs,
            destinations=args.destinations,
        )
        selected, selection = select_rows_balanced(
            candidates,
            target_catalogs=args.target_catalogs,
            destinations=args.destinations,
            rows_per_cell_stage=int(args.rows_per_cell_stage),
            seed=int(args.selection_seed),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    selection["excluded_by_sft_bank"] = excluded
    availability_path = output / "availability.json"
    availability_path.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
    if selection["shortages"]:
        raise SystemExit(
            f"Retention cells are short: {selection['shortages']}. See "
            f"{availability_path}; lower --rows-per-cell-stage or collect more."
        )
    dataset = build_retention_rows(selected)
    frame_paths = sorted({path.expanduser().resolve() for path in args.frames if path.is_file()})
    frame_report = None
    if frame_paths:
        frame_report = verify_frame_coverage(dataset, frame_paths)
        if frame_report["resolved_fraction"] < 1.0 and not bool(args.allow_missing_frames):
            raise SystemExit(
                "Not every retention row has its exact policy frame: "
                f"{frame_report['unresolved_examples']}."
            )
    elif not bool(args.allow_missing_frames):
        raise SystemExit("No --frames were supplied for the retention bank.")
    np.savez_compressed(output / "demonstrations.npz", **dataset)
    shared = set(dataset["scene_uid"].tolist()) & exclude_scenes
    if shared:
        raise SystemExit(f"{len(shared)} retention scenes are also in the SFT bank.")
    source_groups = {
        str(name): int((dataset["source_group"] == name).sum())
        for name in np.unique(dataset["source_group"])
    }
    report = {
        "schema": DATASET_SCHEMA,
        "source_schema": [RETENTION_SCHEMA, RECORD_SCHEMA],
        "priors_stale": False,
        "priors_stale_reason": None,
        "retention_records": [str(path) for path in retention_paths],
        "records": [str(path) for path in strict_paths],
        "exclude_dataset": str(args.exclude_dataset),
        "frames_files": [str(path) for path in frame_paths],
        "provenance": provenance,
        "selection": selection,
        "rows_by_source_group": source_groups,
        "dataset": dataset_report(dataset),
        "frames": frame_report,
        "contract": {
            "retention": True,
            "empty_start": True,
            "one_final_instruction": True,
            "continuous_single_policy": True,
            "simulator_recovery": False,
            "same_checkpoint_and_config_as_sft_bank": True,
            "scene_disjoint_from": str(args.exclude_dataset),
            "rows_from_completed_stages_only": True,
            "placement_rows_from_strict_episodes_only": True,
            "balanced_by": ["target_catalog", "destination", "stage_name"],
        },
    }
    (output / "dataset.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[retention] wrote {output / 'demonstrations.npz'}: "
        f"{dataset['state'].shape[0]} rows, "
        f"{np.unique(dataset['episode_uid']).size} episodes, "
        f"{np.unique(dataset['scene_uid']).size} scenes; by source {source_groups}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
