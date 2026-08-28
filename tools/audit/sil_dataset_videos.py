#!/usr/bin/env python3
"""Render episodes straight out of the SFT dataset, as video.

Not a rollout. Every other video tool in this repo drives the simulator and
films what comes out, which answers "what does this checkpoint do now". This
one answers a different question -- "what is actually IN the training set" --
and the only honest way to answer it is to show the stored pictures rather than
generate new ones.

That is possible because the bank keeps them. ``sil_record --mode replay
--record-frames`` writes ``frames_<stem>.npz`` holding the uint8 camera tensors
the policy was handed, and ``sil_sft`` consumes exactly those arrays through
``resolve_frame_rows``. This tool walks the same join with the same functions,
so a frame on screen is bit-for-bit a frame the SFT trained on. Re-rendering
from a checkpoint could not promise that: the actions are smoothed, the replay
that produced them is a different rollout from the record, and a re-render would
show a fourth thing.

Only episodes present in the dataset are eligible. That matters because
``--rows-per-instruction`` drops whole episodes at random to balance the mix, so
the recordings hold more episodes than the dataset does, and sampling from the
recordings would show footage that trained nothing.

No simulator, no checkpoint, no GPU. It reads two npz families and shells out to
ffmpeg.

One frame per DECISION, not per env step. At ``replan_every: 4`` an episode of
26 decisions is 104 env steps, so the video is a quarter of the frames the plant
saw -- which is exactly the rate the policy observed at, and the rate the SFT
trained on.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from tools.audit.sil_sft import (  # noqa: E402
    load_frame_meta,
    materialize_frames,
    resolve_frame_rows,
)
from tools.audit.success_episode_videos import _Mp4  # noqa: E402


def group_episodes(
    episode_uid: np.ndarray,
    decision_index: np.ndarray,
    keep: np.ndarray,
) -> dict[str, list[int]]:
    """Dataset rows grouped by episode, each ordered by decision.

    Rows arrive grouped already, but nothing in the format promises it: the
    quota drops whole episodes and a future consumer could shuffle. Sorting by
    decision here costs nothing and removes the assumption -- a video whose
    frames are out of order looks like a physics bug, which is an expensive
    thing to chase.

    Only rows that resolved to a frame are eligible, so an episode whose tail
    lost its pictures is shown as far as it goes rather than skipped.
    """

    episodes: dict[str, list[int]] = defaultdict(list)
    for row in np.flatnonzero(keep):
        episodes[str(episode_uid[row])].append(int(row))
    for uid, rows in episodes.items():
        rows.sort(key=lambda row: int(decision_index[row]))
    return dict(episodes)


def pick_episodes(
    episodes: Mapping[str, Sequence[int]],
    instruction_of: Mapping[str, str],
    *,
    per_instruction: int,
    seed: int,
) -> dict[str, list[str]]:
    """Choose which episodes to film, per instruction, reproducibly.

    Sampled rather than taken from the front. The dataset is ordered by source
    recording, so the first N episodes of an instruction all come from one
    round at one rung -- a systematically easier or harder slice than the mix
    the SFT saw. Ten videos are a qualitative sample and should look like the
    set they came from.

    Sorted before sampling so the choice depends on the seed and not on dict
    iteration order.
    """

    by_instruction: dict[str, list[str]] = defaultdict(list)
    for uid in sorted(episodes):
        by_instruction[instruction_of[uid]].append(uid)
    rng = np.random.default_rng(int(seed))
    chosen: dict[str, list[str]] = {}
    for name, uids in sorted(by_instruction.items()):
        limit = min(int(per_instruction), len(uids))
        index = rng.choice(len(uids), size=limit, replace=False)
        chosen[name] = [uids[int(i)] for i in sorted(index)]
    return chosen


def compose(
    overview: np.ndarray, wrist: np.ndarray, camera: str
) -> np.ndarray:
    """One frame from the two cameras: either alone, or side by side."""

    if camera == "overview":
        return overview
    if camera == "wrist":
        return wrist
    return np.concatenate([overview, wrist], axis=1)


def safe_name(uid: str) -> str:
    """A filename that still identifies the episode it came from."""

    return str(uid).replace("/", "__").replace(" ", "_")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "demonstrations.npz. Either the dataset build's or the refreshed "
            "one -- the refresh re-derives state and prior but leaves "
            "episode_uid and decision_index alone, so both name the same "
            "episodes."
        ),
    )
    parser.add_argument("--frames", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-instruction", type=int, default=10)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--camera",
        choices=("overview", "wrist", "both"),
        default="both",
        help="both puts the overview and wrist side by side in one frame.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from tools.audit.sil_record import _instruction_name

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset.expanduser().resolve(), allow_pickle=False) as d:
        episode_uid = np.asarray(d["episode_uid"])
        decision_index = np.asarray(d["decision_index"])
        instruction_id = np.asarray(d["instruction_id"])
        instruction_text = np.asarray(d["instruction_text"])
        source_group = (
            np.asarray(d["source_group"])
            if "source_group" in d.files
            else np.full(episode_uid.shape, "", dtype="U32")
        )

    meta = load_frame_meta(
        [path.expanduser().resolve() for path in args.frames]
    )
    keep, lookups = resolve_frame_rows(episode_uid, decision_index, meta)
    resolved = int(keep.sum())
    print(
        f"[videos] {resolved}/{episode_uid.shape[0]} dataset rows resolved to "
        f"a stored frame across {len(meta)} frames files",
        flush=True,
    )
    if resolved == 0:
        raise SystemExit(
            "No dataset row resolved to a frame. The --frames glob probably "
            "does not cover the recordings this dataset was built from."
        )

    # resolve_frame_rows returns lookups POSITIONALLY for the kept rows, so the
    # nth lookup belongs to the nth set bit. materialize_frames indexes them by
    # dataset row, which means it needs the inverse map, not the mask.
    lookup_of_row: dict[int, tuple[str, int, int]] = {}
    for position, row in enumerate(np.flatnonzero(keep)):
        lookup_of_row[int(row)] = lookups[position]

    episodes = group_episodes(episode_uid, decision_index, keep)
    instruction_of = {
        uid: _instruction_name(int(instruction_id[rows[0]]))
        for uid, rows in episodes.items()
    }
    chosen = pick_episodes(
        episodes,
        instruction_of,
        per_instruction=int(args.per_instruction),
        seed=int(args.seed),
    )

    available = defaultdict(int)
    for uid in episodes:
        available[instruction_of[uid]] += 1
    for name in sorted(available):
        print(
            f"[videos] {name}: {available[name]} episodes in the dataset, "
            f"filming {len(chosen.get(name, []))}",
            flush=True,
        )

    manifest: list[dict[str, Any]] = []
    for name, uids in sorted(chosen.items()):
        for order, uid in enumerate(uids):
            rows = episodes[uid]
            # materialize_frames wants a lookup list indexed by row id. Build a
            # dense one for just this episode rather than passing the whole
            # dataset's, so peak memory is one episode of pictures.
            local_lookups = [("", 0, 0)] * (max(rows) + 1)
            for row in rows:
                local_lookups[row] = lookup_of_row[row]
            overview, wrist = materialize_frames(meta, local_lookups, rows)
            frames = [
                compose(overview[i], wrist[i], str(args.camera))
                for i in range(len(rows))
            ]
            if not frames:
                continue
            path = (
                output
                / f"{name}_{order:02d}_{safe_name(uid)}.mp4"
            )
            writer = _Mp4(
                path,
                fps=float(args.fps),
                height=int(frames[0].shape[0]),
                width=int(frames[0].shape[1]),
            )
            for frame in frames:
                writer.write(frame)
            writer.close()
            manifest.append(
                {
                    "instruction": name,
                    "instruction_text": str(instruction_text[rows[0]]),
                    "episode_uid": uid,
                    "source_group": str(source_group[rows[0]]),
                    "decisions": len(rows),
                    "video": str(path),
                }
            )
            print(
                f"[videos] {path.name}  {len(rows)} decisions  "
                f"rung {source_group[rows[0]]}",
                flush=True,
            )

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[videos] wrote {len(manifest)} videos and "
        f"{output / 'manifest.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
