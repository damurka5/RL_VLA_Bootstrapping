#!/usr/bin/env python3
"""Turn accepted three-stage chains into SFT rows under ONE final instruction.

Three things happen here and nothing else: acceptance is re-verified from the
recordings, real consecutive executed actions are assembled into decision rows,
and every row of a chain is relabelled to the student's single final prompt.

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
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Accepted chains as full-task rows, plus the partial pickup material."""

    columns: dict[str, list[Any]] = {}
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
    ) -> None:
        per = int(record.actions_per_decision)
        start = decision * per
        stop = start + per
        stage_raw = int(record.decision_stage[decision, world])
        table.setdefault("state", []).append(record.states[decision, world])
        table.setdefault("prior", []).append(record.priors[decision, world])
        table.setdefault("action", []).append(record.actions[start:stop, world])
        table.setdefault("action_mask", []).append(
            record.active[start:stop, world]
        )
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
        "partial_pickup_chains": 0,
        "rejection_reasons": {},
    }
    for path, record in records:
        teachers = json.loads(record.teacher_manifest_json)
        accepted, reasons, counts = record.acceptance(
            min_approach_xy=min_approach_xy,
            min_handoff_lift=min_handoff_lift,
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
            if accepted[world]:
                census["accepted_chains"] += 1
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
                )
    return _finalize(columns), _finalize(partial), census


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
        report["actions_by_source"] = {
            name: int((sources == index).sum())
            for index, name in enumerate(
                ("teacher", "yaw_tail", "yaw_hold", "settle_hold")
            )
            if int((sources == index).sum()) > 0
        }
    return report


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
    print(f"[dataset] {len(records)} staged rounds", flush=True)

    dataset, partial, census = build_rows(
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

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    frame_report: dict[str, Any] | None = None
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
        "acceptance": {
            "min_approach_xy": float(args.min_approach_xy),
            "min_handoff_lift": float(args.min_handoff_lift),
        },
    }
    if partial:
        np.savez_compressed(output / "partial_pickup.npz", **partial)
        report["partial_pickup"] = dataset_report(partial)
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
