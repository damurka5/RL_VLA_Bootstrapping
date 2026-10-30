#!/usr/bin/env python3
"""Relabel verified skill samples for final-prompt SFT before joint WidowX RL.

Input is sil_record's frame-filtered original-label dataset. Actions, masks,
episode/frame identities and observations stay attached to their real source.
Independent skill episodes are pooled as stage supervision, never presented as
continuous successful full-task demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID  # noqa: E402
from tools.audit.sil_artifacts import atomic_savez  # noqa: E402

KINDS = ("move_to_object", "pick_up", "put_into_plate", "put_into_bowl")
STAGES = ("move_to", "pick_up", "placement")
DEFAULT_OBJECTS = ("apple", "tomato", "orange", "potato", "bell pepper")


def object_text(text: str, kind: str) -> str:
    if kind == "move_to_object" and text.startswith("move to "):
        return text[len("move to "):].strip()
    if kind == "pick_up" and text.startswith("pick up "):
        return text[len("pick up "):].strip()
    destination = kind.removeprefix("put_into_")
    match = re.fullmatch(r"put (.+?) (?:into|on|in) (?:the )?" + destination, text)
    if match:
        return match.group(1).strip()
    raise ValueError(f"Cannot recover the real target object from {kind}: {text!r}")


def build(
    source: dict[str, np.ndarray], *, objects: tuple[str, ...] = DEFAULT_OBJECTS,
    rows_per_cell: int = 2000, retention_fraction: float = .2,
    val_fraction: float = .1, seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not 0 < retention_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("retention_fraction and val_fraction must lie strictly between zero and one")
    required = {"state", "prior", "action", "action_mask", "instruction_id",
                "instruction_text", "episode_uid", "decision_index"}
    if not required <= source.keys():
        raise ValueError(f"Source dataset is missing {sorted(required - source.keys())}")
    size = len(source["state"])
    if any(value.ndim == 0 or len(value) != size for value in source.values()):
        raise ValueError("Every source array must carry one entry per decision")
    rng = np.random.default_rng(seed)
    ids = {INSTRUCTION_TO_ID[name]: name for name in KINDS}
    episodes: dict[str, dict[str, Any]] = {}
    units: dict[str, list[str]] = {}
    dropped: dict[str, int] = {}
    for uid in np.unique(source["episode_uid"]):
        rows = np.flatnonzero(source["episode_uid"] == uid)
        kinds = np.unique(source["instruction_id"][rows])
        texts = np.unique(source["instruction_text"][rows])
        if len(kinds) != 1 or len(texts) != 1:
            raise ValueError(f"Episode {uid} carries inconsistent teacher instructions")
        kind = ids.get(int(kinds[0]))
        if kind is None:
            raise ValueError(f"Unsupported source instruction id {kinds[0]}")
        obj = object_text(str(texts[0]), kind)
        if obj not in objects:
            dropped[obj] = dropped.get(obj, 0) + 1
            continue
        stage = "placement" if kind.startswith("put_into_") else {
            "move_to_object": "move_to", "pick_up": "pick_up"}[kind]
        scene_ids = np.unique(source["scene_uid"][rows]) if "scene_uid" in source else np.array([uid])
        if len(scene_ids) != 1:
            raise ValueError(f"Episode {uid} carries inconsistent scene identities")
        unit = str(scene_ids[0])
        if unit in units:
            previous = episodes[units[unit][0]]
            if (previous["kind"], previous["object"]) != (kind, obj):
                raise ValueError(f"Scene {unit} mixes teacher instructions or targets")
        episodes[str(uid)] = {"rows": rows, "kind": kind, "object": obj, "stage": stage, "unit": unit}
        units.setdefault(unit, []).append(str(uid))

    # Disjoint source episodes for original-label retention and final-prompt
    # supervision. No relabelled copy can leak into the other view's holdout.
    for kind in KINDS:
        group_ids = sorted(unit for unit, uids in units.items() if episodes[uids[0]]["kind"] == kind)
        minimum_main = 4 if kind in KINDS[:2] else 2  # >=2 independent groups per final-prompt cell
        if len(group_ids) < minimum_main + 2:
            raise ValueError(f"{kind}: need at least {minimum_main + 2} filmed successful source groups; "
                             f"found {len(group_ids)}. Recover/replay the retained best skill bank first.")
        group_ids = list(rng.permutation(group_ids))
        retained = min(len(group_ids) - minimum_main, max(2, round(len(group_ids) * retention_fraction)))
        for index, unit in enumerate(group_ids):
            original = index < retained
            if kind.startswith("put_into_"):
                destination = kind.removeprefix("put_into_")
            elif original:
                destination = ""
            else:
                destination = ("plate", "bowl")[(index - retained) % 2]
            for uid in units[unit]:
                episodes[uid].update(view="original_retention" if original else "final_prompt",
                                     destination=destination,
                                     sampling_destination="original_labels" if original else destination)

    cells: dict[tuple[str, str], list[str]] = {}
    for unit, uids in units.items():
        entry = episodes[uids[0]]
        cells.setdefault((entry["sampling_destination"], entry["stage"]), []).append(unit)
    selected = []
    for destination in ("plate", "bowl", "original_labels"):
        for stage in STAGES:
            cell = list(rng.permutation(sorted(cells.get((destination, stage), []))))
            if len(cell) < 2:
                raise ValueError(f"Missing train/validation material for {destination}/{stage}")
            taken, decisions = [], 0
            for unit in cell:
                if len(taken) >= 2 and rows_per_cell > 0 and decisions >= rows_per_cell:
                    break
                taken.append(unit)
                decisions += sum(len(episodes[uid]["rows"]) for uid in units[unit])
            held = min(len(taken) - 1, max(1, round(len(taken) * val_fraction)))
            for index, unit in enumerate(taken):
                for uid in units[unit]:
                    episodes[uid]["validation"] = index < held
                    selected.append(uid)

    indices = np.concatenate([episodes[uid]["rows"] for uid in selected])
    result = {key: value[indices].copy() for key, value in source.items()}
    result["teacher_instruction_id"] = result["instruction_id"].copy()
    result["teacher_instruction_text"] = result["instruction_text"].copy()
    entries = [episodes[uid] for uid in selected for _ in episodes[uid]["rows"]]
    result["stage_name"] = np.array([e["stage"] for e in entries], dtype="U16")
    result["stage_id"] = np.array([STAGES.index(e["stage"]) for e in entries], dtype=np.int8)
    result["target_catalog"] = np.array([e["object"] for e in entries], dtype="U64")
    result["destination"] = np.array([e["destination"] for e in entries], dtype="U8")
    result["sampling_destination"] = np.array([e["sampling_destination"] for e in entries], dtype="U24")
    result["sft_view"] = np.array([e["view"] for e in entries], dtype="U24")
    result["sft_validation"] = np.array([e["validation"] for e in entries], dtype=bool)
    result["sampling_cell_weight"] = np.array([
        retention_fraction / 3 if e["view"] == "original_retention" else (1 - retention_fraction) / 6
        for e in entries], dtype=np.float64)
    result["source_episode_uid"] = result["episode_uid"].copy()
    result["full_chain_success"] = np.zeros(len(indices), dtype=bool)
    for index, entry in enumerate(entries):
        if entry["view"] == "final_prompt":
            result["instruction_id"][index] = INSTRUCTION_TO_ID[f"put_into_{entry['destination']}"]
            # Allocate sufficient text width, even when every teacher prompt was short.
    result["instruction_text"] = np.array([
        f"put {entry['object']} into {entry['destination']}" if entry["view"] == "final_prompt"
        else str(result["teacher_instruction_text"][index]) for index, entry in enumerate(entries)
    ], dtype="U256")

    census = {}
    for destination in ("plate", "bowl", "original_labels"):
        for stage in STAGES:
            mask = (result["sampling_destination"] == destination) & (result["stage_name"] == stage)
            census[f"{destination}/{stage}"] = {
                "decisions": int(mask.sum()), "source_episodes": int(np.unique(result["episode_uid"][mask]).size),
                "train_decisions": int((mask & ~result["sft_validation"]).sum()),
                "validation_decisions": int((mask & result["sft_validation"]).sum()),
                "source_groups": len({e["unit"] for e, selected_row in zip(entries, mask) if selected_row}),
                "objects": {str(obj): int((mask & (result["target_catalog"] == obj)).sum())
                            for obj in np.unique(result["target_catalog"][mask])},
            }
    report = {
        "schema": "widowx200_final_prompt_skill_samples/v2", "priors_stale": True,
        "priors_stale_reason": "Teacher prompts/adapters differ from the student final prompts and initializer",
        "continuous_full_task_demonstrations": False, "full_chain_successes": 0,
        "prefix_destination_scene_context": "as recorded; destination presence is not certified for reach/pick skill scenes",
        "split_group": "recorded scene_uid" if "scene_uid" in source else "legacy source episode identity",
        "cross_teacher_scene_deduplication": False,
        "seed": seed, "retention_exposure": retention_fraction, "val_fraction": val_fraction,
        "rows_per_stage_sampling_destination": rows_per_cell,
        "source_decisions": size, "decisions": len(indices), "source_episodes": len(selected),
        "dropped_non_graspable_target_episodes": dropped, "cells": census,
        "prompts": sorted(str(text) for text in np.unique(result["instruction_text"])),
        "sampler": "balanced by sampling_destination/stage/object with sampling_cell_weight",
    }
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-cell", type=int, default=2000)
    parser.add_argument("--retention-fraction", type=float, default=.2)
    parser.add_argument("--val-fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--require-scene-ids", action="store_true")
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as data:
        source = {key: data[key] for key in data.files}
    if args.require_scene_ids and "scene_uid" not in source:
        raise ValueError("Source pool has no scene_uid. Update sil_record and rebuild the original pool before composition SFT.")
    result, report = build(source, objects=tuple(args.objects), rows_per_cell=args.rows_per_cell,
                           retention_fraction=args.retention_fraction, val_fraction=args.val_fraction, seed=args.seed)
    report["source_dataset"] = str(args.dataset.resolve())
    report["source_dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_savez(args.output / "demonstrations.npz", **result)
    temporary = args.output / "dataset.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    temporary.replace(args.output / "dataset.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
