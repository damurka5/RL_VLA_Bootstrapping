#!/usr/bin/env python3
"""Turn verified three-stage transitions into SFT rows under ONE final instruction.

Three things happen here and nothing else: acceptance is re-verified from the
recordings, real consecutive executed actions are assembled into decision rows,
and every retained row is relabelled to the student's single final prompt.

Two final-prompt views are written. ``demonstrations.npz`` remains the strict
end-to-end view and contains only accepted full chains.
``stage_transitions.npz`` is the larger transition view: a move/alignment slice
is retained only when alignment completed, a pickup slice only when grasp and
lift completed, and a placement slice only when the full chain was accepted.
Failed actions are never promoted to demonstrations.

What relabelling is and is not
------------------------------

The demonstration's object and its ACTUAL destination determine the label. A
bowl trajectory is not a plate demonstration because a plate happens to be in
the scene, and a chain that failed is not a demonstration of the task it failed
at. Both instruction id and text change, for every row of the chain including
its move-to prefix -- that prefix IS part of "put the apple into the bowl", and
labelling it ``move to apple`` is what made the campaign's four families
separate policies in the first place.

The teacher's own wording is kept beside the new one rather than overwritten.
The plate teacher was conditioned on "put apple on the plate" and the student is
asked "put apple into plate"; losing that distinction would make a later prompt
analysis impossible to run.

Rewriting the text is NOT sufficient
------------------------------------

``state`` carries a pooled vision feature and ``prior`` is the SmolVLA chunk the
residual was conditioned on. Both were computed under the TEACHER's prompt and
the teacher's adapter. A dataset whose text says "put apple into plate" while
its prior was drawn under "move to apple" trains the residual to correct a
prediction the student will never make. So this tool writes a bank that is
explicitly marked ``priors_stale``, and ``sil_refresh_priors.py`` must be run
over it with the student initialization and the final prompt before any SFT.
The marker is checked by the SFT entry point rather than left to discipline.

Rows are POLICY DECISIONS
-------------------------

One row is one decision: the state and prior the actor saw, and the four
five-dimensional actions the plant then executed from it. The actor emits eight
and executes four; the unexecuted slots 4-7 are predictions, not future
actions, and are not supervised. Padding after a chain ends carries mask zero.

Usage::

    python tools/audit/build_cdpr_staged_sft_dataset.py \\
        --records runs/three_stage/bank_shard*/staged_*.npz \\
        --frames  runs/three_stage/bank_shard*/frames_*.npz \\
        --output  runs/three_stage/dataset

To extend an existing transition bank without adding more move-to rows, build
the new continuous recordings and merge only their downstream slices::

    python tools/audit/build_cdpr_staged_sft_dataset.py \\
        --records runs/downstream_expansion/bank_shard*/staged_*.npz \\
        --frames runs/three_stage_full/bank_shard*/frames_*.npz \\
                 runs/downstream_expansion/bank_shard*/frames_*.npz \\
        --base-transition-dataset \\
            runs/three_stage_full/dataset/stage_transitions.npz \\
        --additional-stages pick_up placement \\
        --output runs/downstream_expansion/dataset

The robot still executes each chain from the ordinary empty start; only row
admission is stage-selective. This preserves real physical handoffs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (  # noqa: E402
    SEMANTIC_STAGE_OF,
    SOURCE_NAMES,
    STAGE_ALIGN,
    StagedRound,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (  # noqa: E402
    INSTRUCTION_TO_ID,
)

SEMANTIC_STAGES: tuple[str, ...] = ("move_to", "pick_up", "placement")
DATASET_SCHEMA = "cdpr_three_stage_put_into_rows/v1"


def _last_active_decision(record: StagedRound, world: int) -> int:
    live = np.flatnonzero(record.decision_active[:, world])
    return int(live[-1]) if live.size else -1


def _supervised_through(record: StagedRound, world: int, column: str) -> int:
    """The last env step of this chain that is part of the demonstration.

    A stage machine reads its transitions at DECISION boundaries, so the world
    keeps executing the rest of the chunk after the predicate has already
    fired. Those trailing actions are real -- they were executed and they moved
    the plant -- but they happen after the object is already in the receptacle,
    and supervising them teaches whatever the teacher happened to do next.

    Returns the index of the step that produced the success, inclusive.
    Everything after it is masked rather than dropped, so the chunk keeps its
    shape and the action head stays aligned. This is the same rule
    ``sil_record`` applies with ``first_success_step``; there the world is
    frozen after success because the loop consumes ``terminated``, and here it
    is not, so the rule has to be applied explicitly.
    """

    fired = np.flatnonzero(getattr(record, column)[:, world])
    return int(fired[0]) if fired.size else record.actions.shape[0]


def _stage_boundary_distance(
    record: StagedRound, world: int, decision: int
) -> int:
    """Decisions to the nearest teacher handoff, in either direction.

    Carried per row so a sampler can add a measured share of boundary-near
    decisions if the transitions turn out to be the weak point, without having
    to re-open the recordings to find them.
    """

    events = [
        int(record.reach_event[world]),
        int(record.align_event[world]),
        int(record.pickup_event[world]),
    ]
    live = [value for value in events if value >= 0]
    if not live:
        return 10_000
    return int(min(abs(decision - value) for value in live))


def build_rows(
    records: Sequence[tuple[Path, StagedRound]],
    *,
    min_approach_xy: float,
    min_handoff_lift: float,
    include_rejected_pickup_prefix: bool,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Accepted chains, verified stage slices, and partial pickup material."""

    columns: dict[str, list[Any]] = {}
    transitions: dict[str, list[Any]] = {}
    partial: dict[str, list[Any]] = {}

    def emit(
        table: dict[str, list[Any]],
        *,
        record: StagedRound,
        world: int,
        decision: int,
        instruction_id: int,
        instruction_text: str,
        full_chain: bool,
        source_sha: str,
        supervised_through: int,
    ) -> None:
        per = int(record.actions_per_decision)
        start = decision * per
        stop = start + per
        stage_raw = int(record.decision_stage[decision, world])
        # Executed AND part of the demonstration. The second conjunct only
        # bites on the final decision of a chain, where the predicate fired
        # partway through the chunk.
        supervised = record.active[start:stop, world] & (
            np.arange(start, stop) <= int(supervised_through)
        )
        table.setdefault("state", []).append(record.states[decision, world])
        table.setdefault("prior", []).append(record.priors[decision, world])
        table.setdefault("action", []).append(record.actions[start:stop, world])
        table.setdefault("action_mask", []).append(supervised)
        table.setdefault("action_source", []).append(
            record.action_source[start:stop, world]
        )
        table.setdefault("instruction_id", []).append(int(instruction_id))
        table.setdefault("instruction_text", []).append(str(instruction_text))
        table.setdefault("teacher_instruction_text", []).append(
            _teacher_text(record, world, stage_raw)
        )
        table.setdefault("episode_uid", []).append(
            str(record.episode_uid[world])
        )
        table.setdefault("parent_episode_uid", []).append(
            str(record.episode_uid[world])
        )
        table.setdefault("decision_index", []).append(int(decision))
        table.setdefault("scene_uid", []).append(str(record.scene_uid[world]))
        table.setdefault("split", []).append(str(record.split[world]))
        table.setdefault("rollout_index", []).append(
            int(record.rollout_index[world])
        )
        table.setdefault("destination", []).append(
            str(record.destination[world])
        )
        table.setdefault("target_catalog", []).append(
            str(record.target_catalog[world])
        )
        table.setdefault("substage_id", []).append(stage_raw)
        table.setdefault("stage_id", []).append(
            SEMANTIC_STAGES.index(
                SEMANTIC_STAGE_OF.get(stage_raw, "placement")
            )
        )
        table.setdefault("stage_name", []).append(
            SEMANTIC_STAGE_OF.get(stage_raw, "placement")
        )
        table.setdefault("source_checkpoint_sha256", []).append(source_sha)
        table.setdefault("frame_uid", []).append(
            f"{record.episode_uid[world]}#{decision}"
        )
        table.setdefault("stage_boundary_distance", []).append(
            _stage_boundary_distance(record, world, decision)
        )
        table.setdefault("full_chain_success", []).append(bool(full_chain))
        table.setdefault("source_group", []).append(
            f"{record.destination[world]}_{SEMANTIC_STAGE_OF.get(stage_raw, 'placement')}"
        )
        # The historical stratum column every existing tool reads. A full-task
        # chain begins with an empty hand by construction, so it is False on
        # every row here; it is written anyway so a pooled bank of old and new
        # material has one column rather than a missing one.
        table.setdefault("starts_grasped", []).append(False)

    census = {
        "records": len(records),
        "worlds": 0,
        "accepted_chains": 0,
        "transition_success_chains": {
            "move_to": 0,
            "pick_up": 0,
            "placement": 0,
        },
        "partial_pickup_chains": 0,
        "rejection_reasons": {},
    }
    for path, record in records:
        teachers = json.loads(record.teacher_manifest_json)
        accepted, reasons, counts = record.acceptance(
            min_approach_xy=min_approach_xy,
            min_handoff_lift=min_handoff_lift,
        )
        reset_target_xy = record.reset_object_xyz[:, 0, :2]
        reset_receptacle_xy = record.reset_object_xyz[:, 1, :2]
        realized_approach = np.linalg.norm(
            record.reset_ee_xyz[:, :2] - reset_target_xy, axis=-1
        )
        realized_transport = np.linalg.norm(
            reset_target_xy - reset_receptacle_xy, axis=-1
        )
        finite = np.isfinite(record.actions).all(axis=-1) & np.isfinite(
            record.ee_xyz
        ).all(axis=-1)
        transition_valid = (
            ~np.asarray(record.diverged_world_mask, dtype=bool)
            & ~np.asarray(record.physical_grasp[0], dtype=bool)
            & (np.asarray(record.gripper_opening[0]) >= 0.90)
            & (realized_approach >= float(min_approach_xy) - 1e-6)
            & (realized_transport > record.destination_success_radius)
            & (finite | ~record.active).all(axis=0)
        )
        census["worlds"] += int(record.worlds)
        for name, count in counts.items():
            if name == "accepted":
                continue
            census["rejection_reasons"][name] = (
                census["rejection_reasons"].get(name, 0) + int(count)
            )
        source_sha = "|".join(
            f"{role}:{teachers[role]['sha256'][:12]}"
            for role in sorted(teachers)
        )
        for world in range(record.worlds):
            last = _last_active_decision(record, world)
            if last < 0:
                continue
            # Keep only stages whose own terminal event was verified. This is
            # the stage-slice interpretation of the continuous rollout: true
            # upstream handoff states are preserved, but actions from the
            # stage that eventually failed never become positive examples.
            completed = {
                "move_to": bool(transition_valid[world])
                and int(record.align_event[world]) >= 0,
                "pick_up": bool(transition_valid[world])
                and int(record.pickup_event[world]) >= 0,
                "placement": bool(accepted[world]),
            }
            for name, success in completed.items():
                census["transition_success_chains"][name] += int(success)
            cutoffs = {
                # Alignment readiness is sampled at the decision boundary;
                # every active action in that final decision contributed.
                "move_to": record.actions.shape[0],
                "pick_up": _supervised_through(
                    record, world, "pickup_success"
                ),
                "placement": _supervised_through(
                    record, world, "placement_success"
                ),
            }
            for decision in range(last + 1):
                stage_raw = int(record.decision_stage[decision, world])
                semantic = SEMANTIC_STAGE_OF.get(stage_raw)
                if semantic is None or not completed[semantic]:
                    continue
                emit(
                    transitions,
                    record=record,
                    world=world,
                    decision=decision,
                    instruction_id=int(record.instruction_id[world]),
                    instruction_text=str(record.instruction_text[world]),
                    full_chain=bool(accepted[world]),
                    source_sha=source_sha,
                    supervised_through=cutoffs[semantic],
                )
            if accepted[world]:
                census["accepted_chains"] += 1
                cutoff = _supervised_through(
                    record, world, "placement_success"
                )
                for decision in range(last + 1):
                    emit(
                        columns,
                        record=record,
                        world=world,
                        decision=decision,
                        instruction_id=int(record.instruction_id[world]),
                        instruction_text=str(record.instruction_text[world]),
                        full_chain=True,
                        source_sha=source_sha,
                        supervised_through=cutoff,
                    )
                continue
            if not include_rejected_pickup_prefix:
                continue
            # A chain that grasped and lifted but never placed is a valid
            # pick_up demonstration and is NOT a put_into one. Stored apart,
            # under its own label, so a later recovery experiment can ask for
            # it deliberately instead of it leaking into the full-task bank.
            handoff = int(record.pickup_event[world])
            if handoff < 0:
                continue
            census["partial_pickup_chains"] += 1
            label = f"pick up {_object_label(record, world)}"
            cutoff = _supervised_through(record, world, "pickup_success")
            for decision in range(handoff + 1):
                emit(
                    partial,
                    record=record,
                    world=world,
                    decision=decision,
                    instruction_id=int(INSTRUCTION_TO_ID["pick_up"]),
                    instruction_text=label,
                    full_chain=False,
                    source_sha=source_sha,
                    supervised_through=cutoff,
                )
    return (
        _finalize(columns),
        _finalize(transitions),
        _finalize(partial),
        census,
    )


def _teacher_text(record: StagedRound, world: int, stage_raw: int) -> str:
    semantic = SEMANTIC_STAGE_OF.get(int(stage_raw), "placement")
    if semantic == "move_to":
        return str(record.move_teacher_text[world])
    if semantic == "pick_up":
        return str(record.pickup_teacher_text[world])
    return str(record.placement_teacher_text[world])


def _object_label(record: StagedRound, world: int) -> str:
    # "put apple into bowl" -> "apple". Taken from the student text so the two
    # labels cannot disagree about which object the episode is about.
    text = str(record.instruction_text[world])
    parts = text.split()
    return parts[1] if len(parts) > 2 else parts[-1]


_DTYPES: dict[str, Any] = {
    "state": np.float32,
    "prior": np.float32,
    "action": np.float32,
    "action_mask": bool,
    "action_source": np.int8,
    "instruction_id": np.int64,
    "instruction_text": "U256",
    "teacher_instruction_text": "U256",
    "episode_uid": "U128",
    "parent_episode_uid": "U128",
    "decision_index": np.int64,
    "scene_uid": "U64",
    "split": "U24",
    "rollout_index": np.int64,
    "destination": "U8",
    "target_catalog": "U32",
    "substage_id": np.int8,
    "stage_id": np.int8,
    "stage_name": "U16",
    "source_checkpoint_sha256": "U128",
    "frame_uid": "U160",
    "stage_boundary_distance": np.int32,
    "full_chain_success": bool,
    "source_group": "U32",
    "starts_grasped": bool,
}


def _finalize(table: Mapping[str, Sequence[Any]]) -> dict[str, np.ndarray]:
    if not table:
        return {}
    return {
        name: np.asarray(values, dtype=_DTYPES[name])
        for name, values in table.items()
    }


def dataset_report(dataset: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Everything a sampler needs to know BEFORE it is configured.

    The phase-7 sweep ran three composed-fraction arms and every one of them
    realized 0.981, because the pool could not supply the fractions asked for
    and the spill had nowhere to go. That is hours of GPU for one number, and it
    was knowable in advance from a census exactly like this one.
    """

    if not dataset:
        return {"rows": 0}
    rows = int(dataset["state"].shape[0])
    report: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "rows": rows,
        "episodes": int(np.unique(dataset["episode_uid"]).size),
        "scenes": int(np.unique(dataset["scene_uid"]).size),
        "supervised_actions": int(dataset["action_mask"].sum()),
    }
    for column in ("stage_name", "destination", "target_catalog", "instruction_text"):
        table: dict[str, int] = {}
        for name in np.unique(dataset[column]):
            table[str(name)] = int((dataset[column] == name).sum())
        report[f"rows_by_{column}"] = dict(sorted(table.items()))
    cells: dict[str, int] = {}
    for destination in np.unique(dataset["destination"]):
        for stage in np.unique(dataset["stage_name"]):
            mask = (dataset["destination"] == destination) & (
                dataset["stage_name"] == stage
            )
            cells[f"{destination}/{stage}"] = int(mask.sum())
    report["rows_by_destination_stage"] = dict(sorted(cells.items()))
    empty = [name for name, count in cells.items() if count == 0]
    report["empty_strata"] = empty
    # The alignment tail's share of the move-to slice, reported because the tail
    # is a CONTROLLER's output inside a policy's slice and a reader is entitled
    # to know how much of "move_to" is servo.
    move = dataset["stage_name"] == "move_to"
    align = move & (dataset["substage_id"] == STAGE_ALIGN)
    report["alignment_tail_share_of_move_to"] = (
        round(float(align.sum()) / float(move.sum()), 4)
        if int(move.sum()) > 0
        else None
    )
    sources = dataset.get("action_source")
    if sources is not None:
        def source_counts(mask: np.ndarray | None = None) -> dict[str, int]:
            selected = sources if mask is None else sources[np.asarray(mask, dtype=bool)]
            return {
                name: int((selected == index).sum())
                for index, name in enumerate(SOURCE_NAMES)
                if int((selected == index).sum()) > 0
            }

        report["actions_by_source"] = source_counts()
        if "action_mask" in dataset:
            report["supervised_actions_by_source"] = source_counts(
                dataset["action_mask"]
            )
    return report


def select_transition_stages(
    dataset: Mapping[str, np.ndarray], stages: Sequence[str]
) -> dict[str, np.ndarray]:
    """Select complete decision rows for named semantic stages."""

    requested = tuple(dict.fromkeys(str(stage) for stage in stages))
    invalid = sorted(set(requested) - set(SEMANTIC_STAGES))
    if invalid:
        raise SystemExit(
            f"Unknown --additional-stages {invalid}; choose from "
            f"{list(SEMANTIC_STAGES)}."
        )
    if not dataset or not requested:
        return {}
    mask = np.isin(dataset["stage_name"], np.asarray(requested))
    selected = {
        name: np.asarray(values)[mask]
        for name, values in dataset.items()
    }
    missing = [
        stage
        for stage in requested
        if not bool((selected["stage_name"] == stage).any())
    ]
    if missing:
        raise SystemExit(
            "The new recordings produced no verified rows for requested "
            f"stages {missing}. Collect more chains; do not substitute failed "
            "actions as demonstrations."
        )
    return selected


def merge_transition_expansion(
    base: Mapping[str, np.ndarray],
    additional: Mapping[str, np.ndarray],
    *,
    allow_repeated_scenes: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Append verified rows while guarding ids, shapes and scene diversity."""

    if not base:
        raise SystemExit("The base transition dataset is empty.")
    if not additional:
        raise SystemExit("The additional transition selection is empty.")
    base_keys = set(base)
    additional_keys = set(additional)
    if base_keys != additional_keys:
        raise SystemExit(
            "Base and additional transition schemas differ. Missing from "
            f"additional: {sorted(base_keys - additional_keys)}; missing from "
            f"base: {sorted(additional_keys - base_keys)}."
        )
    for name in sorted(base_keys):
        left = np.asarray(base[name])
        right = np.asarray(additional[name])
        if left.ndim != right.ndim or left.shape[1:] != right.shape[1:]:
            raise SystemExit(
                f"Column {name!r} has incompatible row shapes "
                f"{left.shape[1:]} and {right.shape[1:]}."
            )

    base_frames = {str(value) for value in base["frame_uid"]}
    additional_frames = {str(value) for value in additional["frame_uid"]}
    duplicate_frames = sorted(base_frames & additional_frames)
    if duplicate_frames:
        raise SystemExit(
            "Base and additional data share frame_uid values, for example "
            f"{duplicate_frames[:5]}. Use a distinct collection TAG and do "
            "not append the same episodes twice."
        )

    base_scenes = {str(value) for value in base["scene_uid"]}
    additional_scenes = {str(value) for value in additional["scene_uid"]}
    repeated_scenes = sorted(base_scenes & additional_scenes)
    if repeated_scenes and not allow_repeated_scenes:
        raise SystemExit(
            "The expansion reuses scenes already present in the base bank, "
            f"for example {repeated_scenes[:5]}. Use a fresh scene manifest "
            "or advance --scene-offset. Pass --allow-repeated-scenes only for "
            "an explicitly declared stochastic-repeat experiment."
        )

    merged = {
        name: np.concatenate(
            [np.asarray(base[name]), np.asarray(additional[name])], axis=0
        )
        for name in base
    }
    return merged, {
        "base_rows": int(np.asarray(base["state"]).shape[0]),
        "added_rows": int(np.asarray(additional["state"]).shape[0]),
        "combined_rows": int(np.asarray(merged["state"]).shape[0]),
        "base_scenes": len(base_scenes),
        "added_scenes": len(additional_scenes),
        "repeated_scenes": len(repeated_scenes),
    }


def load_stale_transition_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load a raw transition bank and refuse a stale/fresh mixture."""

    resolved = path.expanduser().resolve()
    report_path = resolved.parent / "dataset.json"
    if not report_path.is_file():
        raise SystemExit(
            f"{resolved} has no sibling dataset.json, so its prior provenance "
            "cannot be verified."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != DATASET_SCHEMA:
        raise SystemExit(
            f"{report_path} has schema {report.get('schema')!r}, expected "
            f"{DATASET_SCHEMA!r}."
        )
    if report.get("priors_stale") is not True:
        raise SystemExit(
            "The base transition bank is already refreshed while the new "
            "teacher recordings are stale. Merge the raw pre-refresh bank, "
            "then refresh the combined result once."
        )
    with np.load(resolved, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def verify_frame_coverage(
    dataset: Mapping[str, np.ndarray], frame_paths: Sequence[Path]
) -> dict[str, Any]:
    """Every row must resolve to a picture, by explicit id.

    Not a fraction to be tolerated. A bank whose resolvable rows are a subset is
    a bank selected by "whose replay happened to keep frames", which is a
    selection nobody chose; the design requires 100% and an explicit removal of
    the corresponding view otherwise.
    """

    available: set[str] = set()
    per_file: dict[str, int] = {}
    # Which file each uid came from, so that two banks claiming the same
    # episode cannot be merged silently. The recorder builds the uid from
    # tag/shard/round/world only, so two collection runs that used the same
    # --tag produce byte-identical uids for different physical episodes. The
    # coverage set below is keyed by uid alone, so without this check such a
    # merge reports resolved_fraction 1.0 while pairing rows against whichever
    # run's pictures happened to be loaded -- the silent image/action join the
    # schema's explicit ids exist to prevent.
    owner: dict[str, str] = {}
    for path in frame_paths:
        with np.load(path, allow_pickle=False) as data:
            if "episode_uid" not in data.files:
                raise SystemExit(
                    f"{path} carries no episode_uid. The staged schema joins "
                    "frames by explicit id; a positional frames file from the "
                    "older recorder cannot be used here."
                )
            uids = [str(value) for value in data["episode_uid"]]
            decisions = int(data["decisions"])
        per_file[str(path)] = len(uids)
        for uid in uids:
            previous = owner.get(uid)
            if previous is not None and previous != str(path):
                raise SystemExit(
                    f"episode_uid {uid!r} appears in two frame files:\n"
                    f"  {previous}\n  {path}\n"
                    "These are different physical episodes sharing an id, so "
                    "the frame join would pair rows against the wrong "
                    "pictures. Re-record one of the banks with a distinct "
                    "--tag (TAG= in run_cdpr_three_stage_collection.sh), or "
                    "build the two banks into separate datasets."
                )
            owner[uid] = str(path)
            for decision in range(decisions):
                available.add(f"{uid}#{decision}")
    resolved = np.array(
        [str(value) in available for value in dataset["frame_uid"]], dtype=bool
    )
    return {
        "frame_files": per_file,
        "rows": int(resolved.size),
        "resolved": int(resolved.sum()),
        "resolved_fraction": round(float(resolved.mean()), 6)
        if resolved.size
        else 0.0,
        "unresolved_examples": [
            str(value)
            for value in dataset["frame_uid"][~resolved][:5]
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--frames", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-transition-dataset",
        type=Path,
        default=None,
        help=(
            "Raw, priors-stale stage_transitions.npz to preserve while "
            "appending selected stages from --records."
        ),
    )
    parser.add_argument(
        "--additional-stages",
        nargs="+",
        choices=SEMANTIC_STAGES,
        default=None,
        help=(
            "Stages from the new recordings to append to the base. Requires "
            "--base-transition-dataset. Example: pick_up placement."
        ),
    )
    parser.add_argument(
        "--allow-repeated-scenes",
        action="store_true",
        help=(
            "Permit new rows from scene_uids already in the base. Off by "
            "default because an expansion intended to add diversity must not "
            "silently replay the same starts."
        ),
    )
    parser.add_argument("--min-approach-xy", type=float, default=0.06)
    parser.add_argument("--min-handoff-lift", type=float, default=0.05)
    parser.add_argument(
        "--no-partial-pickup",
        action="store_true",
        help=(
            "Skip the pickup prefixes of chains that failed later. They are "
            "written to their own file and never mixed into the full-task "
            "bank; this only stops them being written at all."
        ),
    )
    parser.add_argument(
        "--allow-missing-frames",
        action="store_true",
        help=(
            "Write the bank even if some rows have no picture. Off by "
            "default: an unresolvable row cannot be refreshed onto the "
            "student, and dropping it silently is a selection nobody chose."
        ),
    )
    args = parser.parse_args(argv)

    paths = sorted({path.expanduser().resolve() for path in args.records})
    if not paths:
        raise SystemExit("--records matched nothing.")
    records = [(path, StagedRound.from_npz(path)) for path in paths]
    # Same guard on the action side. A duplicated uid here does not mispair
    # anything, but np.unique collapses the two episodes into one, so the
    # reported episode count would UNDERSTATE the bank while the row count
    # counted both -- and unique-data counts are what the split and the
    # clustered uncertainty are computed from.
    seen_records: dict[str, Path] = {}
    for path, record in records:
        for uid in (str(value) for value in record.episode_uid):
            previous = seen_records.get(uid)
            if previous is not None:
                raise SystemExit(
                    f"episode_uid {uid!r} appears in two record files:\n"
                    f"  {previous}\n  {path}\n"
                    "Two collection runs that shared a --tag produce "
                    "identical ids for different episodes. Re-record one with "
                    "a distinct --tag (TAG= in "
                    "run_cdpr_three_stage_collection.sh), or build them "
                    "separately."
                )
            seen_records[uid] = path
    print(f"[dataset] {len(records)} staged rounds", flush=True)

    dataset, transitions, partial, census = build_rows(
        records,
        min_approach_xy=float(args.min_approach_xy),
        min_handoff_lift=float(args.min_handoff_lift),
        include_rejected_pickup_prefix=not bool(args.no_partial_pickup),
    )
    if not dataset:
        raise SystemExit(
            "No chain was accepted, so there is nothing to train on. The "
            f"rejection census: {census['rejection_reasons']}"
        )

    expansion_report: dict[str, Any] | None = None
    added_transitions: dict[str, np.ndarray] | None = None
    if args.base_transition_dataset is not None:
        if not args.additional_stages:
            raise SystemExit(
                "--base-transition-dataset requires --additional-stages so "
                "row admission is explicit."
            )
        base_transitions = load_stale_transition_dataset(
            args.base_transition_dataset
        )
        added_transitions = select_transition_stages(
            transitions, args.additional_stages
        )
        transitions, expansion_report = merge_transition_expansion(
            base_transitions,
            added_transitions,
            allow_repeated_scenes=bool(args.allow_repeated_scenes),
        )
        expansion_report.update(
            {
                "base_transition_dataset": str(
                    args.base_transition_dataset.expanduser().resolve()
                ),
                "additional_stages": list(args.additional_stages),
                "added_dataset": dataset_report(added_transitions),
            }
        )
    elif args.additional_stages:
        raise SystemExit(
            "--additional-stages is only meaningful with "
            "--base-transition-dataset."
        )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    frame_report: dict[str, Any] | None = None
    transition_frame_report: dict[str, Any] | None = None
    partial_frame_report: dict[str, Any] | None = None
    if args.frames:
        frame_paths = sorted(
            {path.expanduser().resolve() for path in args.frames}
        )
        frame_report = verify_frame_coverage(dataset, frame_paths)
        print(
            f"[dataset] frames resolve {frame_report['resolved']}/"
            f"{frame_report['rows']} rows "
            f"({frame_report['resolved_fraction']:.4f})",
            flush=True,
        )
        if frame_report["resolved_fraction"] < 1.0 and not args.allow_missing_frames:
            raise SystemExit(
                "Not every row has a picture. Examples: "
                f"{frame_report['unresolved_examples']}. Re-record with "
                "frames enabled, or pass --allow-missing-frames deliberately "
                "and accept that the bank is then selected by which episodes "
                "kept pictures."
            )
        if transitions:
            transition_frame_report = verify_frame_coverage(
                transitions, frame_paths
            )
            print(
                f"[dataset] stage-transition frames resolve "
                f"{transition_frame_report['resolved']}/"
                f"{transition_frame_report['rows']} rows "
                f"({transition_frame_report['resolved_fraction']:.4f})",
                flush=True,
            )
            if (
                transition_frame_report["resolved_fraction"] < 1.0
                and not args.allow_missing_frames
            ):
                raise SystemExit(
                    "Not every verified stage-transition row has a picture. "
                    "Examples: "
                    f"{transition_frame_report['unresolved_examples']}. "
                    "Re-record with the current staged recorder, which keeps "
                    "every alignment-success world, or pass "
                    "--allow-missing-frames only for an action-only audit."
                )
        if partial:
            partial_frame_report = verify_frame_coverage(partial, frame_paths)
            print(
                f"[dataset] partial-pickup frames resolve "
                f"{partial_frame_report['resolved']}/"
                f"{partial_frame_report['rows']} rows "
                f"({partial_frame_report['resolved_fraction']:.4f})",
                flush=True,
            )
            if (
                partial_frame_report["resolved_fraction"] < 1.0
                and not args.allow_missing_frames
            ):
                raise SystemExit(
                    "Not every partial-pickup row has a picture. Examples: "
                    f"{partial_frame_report['unresolved_examples']}. These "
                    "prefixes cannot be refreshed or reused. Re-record with "
                    "the current staged recorder, or pass "
                    "--allow-missing-frames only to preserve the non-image "
                    "audit artifact deliberately."
                )

    np.savez_compressed(output / "demonstrations.npz", **dataset)
    report = {
        "schema": DATASET_SCHEMA,
        # Load-bearing. `state` and `prior` in this file were computed under the
        # TEACHERS' prompts and adapters; the text has been relabelled to the
        # student's and the two no longer agree. sil_refresh_priors.py must run
        # before any SFT, and sil_sft.py refuses a bank still carrying this.
        "priors_stale": True,
        "priors_stale_reason": (
            "state/prior were computed under the teacher prompts and adapters; "
            "instruction_text has been relabelled to the student's final "
            "prompt. Run tools/audit/sil_refresh_priors.py with the student "
            "initialization before training."
        ),
        "records": [str(path) for path in paths],
        "census": census,
        "dataset": dataset_report(dataset),
        "frames": frame_report,
        "stage_transitions": dataset_report(transitions),
        "stage_transition_frames": transition_frame_report,
        "expansion": expansion_report,
        "acceptance": {
            "min_approach_xy": float(args.min_approach_xy),
            "min_handoff_lift": float(args.min_handoff_lift),
        },
    }
    if transitions:
        np.savez_compressed(
            output / "stage_transitions.npz", **transitions
        )
        transition_rows = report["stage_transitions"]
        expansion_note = (
            f"; appended {expansion_report['added_rows']} downstream rows"
            if expansion_report is not None
            else ""
        )
        print(
            f"[dataset] wrote {output / 'stage_transitions.npz'}: "
            f"{transition_rows['rows']} rows from "
            f"{transition_rows['episodes']} chains{expansion_note}; "
            f"new-run verified stage chains "
            f"{census['transition_success_chains']}",
            flush=True,
        )
    if partial:
        np.savez_compressed(output / "partial_pickup.npz", **partial)
        report["partial_pickup"] = dataset_report(partial)
        report["partial_pickup_frames"] = partial_frame_report
        print(
            f"[dataset] wrote {output / 'partial_pickup.npz'} "
            f"({partial['state'].shape[0]} rows from "
            f"{census['partial_pickup_chains']} chains that grasped but never "
            "placed)",
            flush=True,
        )
    (output / "dataset.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    census_rows = report["dataset"]
    print(
        f"[dataset] wrote {output / 'demonstrations.npz'}: "
        f"{census_rows['rows']} rows from {census_rows['episodes']} chains "
        f"over {census_rows['scenes']} scenes",
        flush=True,
    )
    print(
        f"[dataset] rows by stage {census_rows['rows_by_stage_name']}",
        flush=True,
    )
    print(
        f"[dataset] rows by destination {census_rows['rows_by_destination']}",
        flush=True,
    )
    if census_rows["empty_strata"]:
        print(
            "[dataset] EMPTY STRATA: "
            f"{census_rows['empty_strata']} -- a stage-balanced sampler will "
            "refuse this bank rather than quietly substituting another cell.",
            flush=True,
        )
    print(
        "[dataset] priors are STALE by construction; run sil_refresh_priors.py "
        "with the student initialization before any SFT.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
