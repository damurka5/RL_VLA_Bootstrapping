#!/usr/bin/env python3
"""Re-derive a demonstration bank's ``state`` and ``prior`` for a new network.

What this exists to prevent
---------------------------

A retention bank stores what a policy did when it was good at an instruction,
and every SFT that consumes it runs against a network that has since moved. Two
of the columns in ``demonstrations.npz`` are network-dependent and go stale the
moment it does:

``prior``  the SmolVLA chunk the residual was conditioned on, and the anchor the
           SFT's KL term pulls toward.
``state``  proprioception with the pooled vision feature concatenated on, whose
           vision block came out of the recording adapter's connector tokens.

``action`` is not. It is what the plant executed, and it stays true however far
the network drifts. So the durable form of a bank is (frames, actions), and
everything network-dependent is recomputed at the moment of use. That is what
this tool does, and it is what lets one bank stay valid across every future
iteration instead of being re-collected each time.

Why not replay the bank under the new checkpoint
------------------------------------------------

Because that was tried and it destroyed the bank. ``sil_record --mode replay``
pins the recorded actions and re-runs the physics, and its ``patched_action``
tap records whatever network is loaded -- so replaying under a different
checkpoint looks like it should produce exactly these columns. Measured on the
move_to bank at cap 0.19: replayed under its OWN checkpoint it survived
333/333 with 9.6 mm of end-effector drift, and under a placement checkpoint it
survived 30/3072. Something in the reset or the horizon does not survive the
swap.

The deeper objection is that it should never have been a rollout. What is
wanted is a forward pass over stored pictures. Dragging a simulator, a reset
and a termination predicate through it to obtain the same numbers adds three
ways to diverge and no information. There is no physics here, so there is
nothing to diverge.

Usage::

    RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl python tools/audit/sil_refresh_priors.py \\
      --dataset runs/phase4_bank/dataset/demonstrations.npz \\
      --frames runs/phase4_bank/move_to_demos/frames_*.npz \\
      --checkpoint runs/<current>/rl/step_NNNN/smolvla_grpo_adapter.pt \\
      --output runs/phase4_placement/iter_2/bank_refreshed
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

from tools.audit.sil_sft import (  # noqa: E402
    _load_dataset,
    build_runtime_and_trainer,
    check_recomputed_vision,
    load_frame_meta,
    load_frame_meta_by_uid,
    materialize_frames,
    materialize_frames_by_path,
    recompute_state_and_prior,
    resolve_frame_rows,
    resolve_frame_rows_by_uid,
)


def _images(array: np.ndarray, torch: Any, device: str) -> Any:
    """uint8 HWC -> float32 NCHW in [0, 1], as the backend hands the runtime."""

    tensor = torch.as_tensor(array, device=device).to(torch.float32)
    return (tensor / 255.0).permute(0, 3, 1, 2).contiguous()


def select_dataset_view_report(
    report: Mapping[str, Any], dataset_path: Path
) -> dict[str, Any]:
    """Make the carried census describe the NPZ that was actually refreshed."""

    carried = dict(report)
    view = dataset_path.stem
    view_report = carried.get(view)
    if isinstance(view_report, Mapping):
        carried["source_full_chain_dataset"] = carried.get("dataset")
        carried["dataset"] = dict(view_report)
    carried["dataset_view"] = view
    return carried


def group_rows_by_file(
    rows: Sequence[int],
    lookups: Sequence[tuple[str, int, int]],
    budget_position: dict[int, int],
) -> dict[str, list[int]]:
    """Which resolved rows live in which frames file.

    The unit of work is the FILE, not the batch. ``materialize_frames``
    decompresses a file's whole ``overview`` and ``wrist`` arrays to take any
    slice of it, so asking for 256 rows at a time would pay that cost once per
    batch -- seventy-odd times over a two-file bank. One pass per file instead:
    decompress, refresh every row it holds in GPU-sized batches, release.

    Peak host memory is therefore one file's uncompressed frames, which for a
    512-world round at 32 decisions and two cameras is about 4.8 GB. That is
    the same order as the LoRA stage's materialised budget and is the reason
    the loop is not simply "load them all".
    """

    grouped: dict[str, list[int]] = {}
    for row in rows:
        key = lookups[budget_position[int(row)]][0]
        grouped.setdefault(key, []).append(int(row))
    return grouped


def _frames_carry_uids(paths: Sequence[Path]) -> bool:
    """Do ALL the frames files carry episode ids? Half is not enough.

    A mixed set would silently fall back to the positional join for every file,
    including the ones that could have been joined exactly -- so the answer has
    to be about the whole set.
    """

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "episode_uid" not in data.files:
                return False
    return True


def check_final_prompt(
    dataset: Mapping[str, Any], *, prefix: str
) -> dict[str, Any]:
    """One instruction per chain, and it is the student's final one.

    Two distinct failures, both silent afterwards. A chain whose rows carry
    different instructions was never relabelled as a unit, so its move-to
    prefix would be refreshed under "move to apple" while its carry is
    refreshed under "put apple into plate" -- and the SFT would then be fitting
    a single residual to two different conditioning distributions. A chain
    whose instruction still starts with the teacher's wording was not
    relabelled at all.
    """

    texts = dataset.get("instruction_text")
    uids = dataset.get("episode_uid")
    if texts is None or uids is None:
        return {"checked": False}
    violations: list[str] = []
    for uid in np.unique(uids):
        distinct = np.unique(texts[uids == uid])
        if distinct.size != 1:
            violations.append(
                f"{uid}: {distinct.size} distinct instructions "
                f"{list(distinct[:3])}"
            )
        elif prefix and not str(distinct[0]).startswith(prefix):
            violations.append(f"{uid}: {distinct[0]!r} lacks {prefix!r}")
    return {
        "checked": True,
        "prefix": prefix,
        "episodes": int(np.unique(uids).size),
        "distinct_prompts": sorted(
            {str(value) for value in np.unique(texts)}
        )[:8],
        "violations": violations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frames", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "The network to re-derive against -- the one the next SFT will "
            "start from, NOT the one that recorded the bank."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rows per SmolVLA forward. Bounds GPU memory, not host memory.",
    )
    parser.add_argument(
        "--vision-lora",
        action="store_true",
        help=(
            "Attach the vision-tower LoRA when rebuilding the runtime. Must "
            "match how the checkpoint was produced: an SFT checkpoint written "
            "with --train-vision-lora carries vision modules, and rebuilding "
            "without this flag would leave those weights unloaded."
        ),
    )
    parser.add_argument(
        "--final-prompt-prefix",
        default="put ",
        help=(
            "Every row of a relabelled full-task bank must carry the STUDENT's "
            "final prompt, and every row of one chain must carry the same one. "
            "A prefix of '' disables the check, which is right for a retention "
            "bank of original labels and wrong for a put_into bank."
        ),
    )
    parser.add_argument(
        "--min-resolved-fraction",
        type=float,
        default=0.99,
        help=(
            "Abort if fewer than this fraction of rows have a frame. A bank "
            "whose rows cannot all be re-derived is not a bank -- the rows "
            "that survive are the ones whose replay happened to keep pictures, "
            "which is a selection nobody chose. Re-replay at --frame-worlds 0."
        ),
    )
    args = parser.parse_args(argv)

    import torch

    dataset = _load_dataset(args.dataset.expanduser().resolve())
    total = int(dataset["state"].shape[0])
    for column in ("instruction_text", "decision_index", "instruction_id"):
        if column not in dataset:
            raise SystemExit(
                f"{args.dataset} has no {column!r}; it was not written by "
                "sil_record --mode dataset."
            )

    # The FINAL-PROMPT check, before a single forward. This tool exists to make
    # the network-dependent columns agree with the instruction, so a bank whose
    # instruction is still the teacher's -- or whose rows disagree with each
    # other inside one chain -- would be refreshed into a consistent-looking
    # dataset that trains the wrong thing.
    prompt_report = check_final_prompt(
        dataset, prefix=str(args.final_prompt_prefix)
    )
    if prompt_report.get("violations"):
        raise SystemExit(
            "The bank's instruction column is not a single final prompt per "
            f"chain: {prompt_report['violations'][:3]}. Relabel with "
            "build_cdpr_staged_sft_dataset.py, or pass --final-prompt-prefix "
            "'' if this is deliberately an original-label retention bank."
        )

    frame_paths = [path.expanduser().resolve() for path in args.frames]
    # Explicit-id join when both sides can do it. The positional join is kept
    # for banks written by the older recorder, which carry no episode ids in
    # their frames files.
    by_uid = "frame_uid" in dataset and _frames_carry_uids(frame_paths)
    if by_uid:
        frames = load_frame_meta_by_uid(frame_paths)
        found, lookups = resolve_frame_rows_by_uid(
            dataset["episode_uid"], dataset["decision_index"], frames
        )
        gather = materialize_frames_by_path
        required = max(float(args.min_resolved_fraction), 1.0)
    else:
        frames = load_frame_meta(frame_paths)
        found, lookups = resolve_frame_rows(
            dataset["episode_uid"], dataset["decision_index"], frames
        )
        gather = None
        required = float(args.min_resolved_fraction)
    resolved = np.flatnonzero(found)
    fraction = resolved.size / max(total, 1)
    print(
        f"[refresh] {resolved.size}/{total} rows resolved to a frame "
        f"({fraction:.4f}) across {len(frames)} entries, join="
        f"{'episode_uid' if by_uid else 'positional'}",
        flush=True,
    )
    if fraction < required:
        raise SystemExit(
            f"[refresh] only {fraction:.4f} of rows resolved, below "
            f"{required}. Re-replay the bank with "
            "--record-frames --frame-worlds 0 so every kept episode has "
            "pictures, or lower --min-resolved-fraction deliberately."
        )

    payload = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    vision_dim = int(dict(payload["args"]).get("residual_vision_dim", 0))
    proprio_dim = int(dataset["state"].shape[-1]) - vision_dim
    if proprio_dim <= 0:
        raise SystemExit(
            f"state is {dataset['state'].shape[-1]} wide against a "
            f"residual_vision_dim of {vision_dim}; the bank and the checkpoint "
            "disagree about the observation layout."
        )
    runtime, _trainer, _args = build_runtime_and_trainer(
        payload,
        checkpoint=args.checkpoint.expanduser().resolve(),
        device=str(args.device),
        train_vision_lora=bool(args.vision_lora),
    )

    # Written in place over copies, so untouched columns keep their dtypes and
    # every row that survives keeps its action, mask, instruction and uid.
    new_state = np.array(dataset["state"], dtype=np.float32)
    new_prior = np.array(dataset["prior"], dtype=np.float32)

    # resolve_frame_rows returns lookups positionally over the RESOLVED rows,
    # so a row's lookup is at its index within flatnonzero(found).
    budget_position = {int(row): i for i, row in enumerate(resolved)}
    grouped = group_rows_by_file(resolved.tolist(), lookups, budget_position)
    take = (
        (lambda positions: gather(lookups, positions))
        if gather is not None
        else (lambda positions: materialize_frames(frames, lookups, positions))
    )

    integrity: dict[str, Any] | None = None
    done = 0
    for key in sorted(grouped):
        rows = grouped[key]
        overview_all, wrist_all = take(
            [budget_position[row] for row in rows]
        )
        for start in range(0, len(rows), int(args.batch_size)):
            chunk = rows[start : start + int(args.batch_size)]
            index = np.asarray(chunk, dtype=np.int64)
            window = slice(start, start + len(chunk))
            proprio = torch.as_tensor(
                dataset["state"][index, :proprio_dim],
                dtype=torch.float32,
                device=args.device,
            )
            with torch.no_grad():
                prior, state = recompute_state_and_prior(
                    runtime,
                    torch,
                    overview=_images(overview_all[window], torch, args.device),
                    wrist=_images(wrist_all[window], torch, args.device),
                    proprio=proprio,
                    instructions=[
                        str(text)
                        for text in dataset["instruction_text"][index]
                    ],
                    vision_dim=vision_dim,
                    enable_grad=False,
                )
            new_state[index] = state.to(torch.float32).cpu().numpy()
            # The runtime returns the prior in its own chunk layout; the
            # bank stores it flattened per row. Reshaped rather than assumed
            # equal, the same way sil_sft's KL term does reshape_as.
            new_prior[index] = (
                prior.to(torch.float32)
                .cpu()
                .numpy()
                .reshape(new_prior[index].shape)
            )
            if integrity is None:
                # The same rows a second time at batch size 1. Here the
                # headline is EXPECTED to be large -- the point of the tool is
                # that the network moved -- so the control is not a pass/fail
                # bound but the scale the headline is read against: a ratio
                # near 1 says this checkpoint's vision tower is where the
                # recording one's was, and a large ratio says it has moved.
                with torch.no_grad():
                    _, control = recompute_state_and_prior(
                        runtime,
                        torch,
                        overview=_images(
                            overview_all[start : start + 1], torch, args.device
                        ),
                        wrist=_images(
                            wrist_all[start : start + 1], torch, args.device
                        ),
                        proprio=proprio[:1],
                        instructions=[str(dataset["instruction_text"][chunk[0]])],
                        vision_dim=vision_dim,
                        enable_grad=False,
                    )
                integrity = check_recomputed_vision(
                    state[:1],
                    torch.as_tensor(
                        dataset["state"][chunk[:1]],
                        dtype=torch.float32,
                        device=args.device,
                    ),
                    vision_dim=vision_dim,
                    torch=torch,
                    control_state=control,
                )
            done += len(chunk)
        del overview_all, wrist_all
        print(f"[refresh] {key}: {len(rows)} rows ({done}/{resolved.size})",
              flush=True)

    refreshed = {key: value[found] for key, value in dataset.items()}
    refreshed["state"] = new_state[found]
    refreshed["prior"] = new_prior[found]

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "demonstrations.npz", **refreshed)
    report = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "frames": [str(path) for path in args.frames],
        "rows_in": total,
        "rows_resolved": int(resolved.size),
        "rows_out": int(refreshed["state"].shape[0]),
        "resolved_fraction": round(float(fraction), 5),
        "vision_dim": vision_dim,
        "proprio_dim": proprio_dim,
        "vision_lora": bool(args.vision_lora),
        "integrity": integrity,
        "episodes_out": int(len(np.unique(refreshed["episode_uid"]))),
        "by_instruction": {
            str(name): int((refreshed["instruction_text"] == name).sum())
            for name in np.unique(refreshed["instruction_text"])
        }
        if "instruction_text" in refreshed
        else None,
    }
    report["final_prompt"] = prompt_report
    report["frame_join"] = "episode_uid" if by_uid else "positional"
    (output / "refresh.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Carry the source bank's census forward with the stale marker CLEARED, so
    # sil_sft.py accepts the refreshed copy and still refuses the original.
    # Writing a fresh dataset.json rather than editing the source in place: the
    # source is still stale and a later run of this tool must still be able to
    # tell.
    source_report = args.dataset.expanduser().resolve().parent / "dataset.json"
    if source_report.is_file():
        try:
            carried = json.loads(source_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            carried = {}
        carried = select_dataset_view_report(carried, args.dataset)
        carried["priors_stale"] = False
        carried["priors_refreshed_from"] = str(args.dataset)
        carried["priors_refreshed_with"] = str(args.checkpoint)
        carried["priors_refresh_report"] = str(output / "refresh.json")
        (output / "dataset.json").write_text(
            json.dumps(carried, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(
        f"[refresh] wrote {output / 'demonstrations.npz'} "
        f"({report['rows_out']} rows, {report['episodes_out']} episodes)",
        flush=True,
    )
    if integrity is not None:
        print(f"[refresh] vision drift vs recorded: {integrity}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
