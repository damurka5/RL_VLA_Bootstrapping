"""The frame join and the per-file work grouping in sil_refresh_priors.

The join is the part of this pipeline with a history. Its first version
compared the two writers' names raw and matched 0 of 33102 rows, after a whole
harvest had already been paid for -- a silent zero, because a dataset with no
resolved rows trains without complaining. So the test that matters here is not
that the arithmetic is right but that the two spellings of an episode source
still meet: ``sil_record --mode dataset`` keys rows as
``<parent>/<replay stem>/r<round>w<world>`` while the frame tap writes
``frames_<stem>.npz``, and only ``frame_join_key`` knows they are the same
thing.

The grouping is tested for a different reason: ``materialize_frames``
decompresses a file's entire overview and wrist arrays to take any slice, so a
grouping that scattered one file's rows across many calls would re-pay that on
every call. Correctness would survive it; a two-file bank taking seventy
decompressions would not.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.sil_refresh_priors import group_rows_by_file  # noqa: E402
from tools.audit.sil_sft import (  # noqa: E402
    frame_join_key,
    load_frame_meta,
    materialize_frames,
    resolve_frame_rows,
)

DECISIONS, HEIGHT, WIDTH = 4, 8, 10
WORLDS = (2, 5, 9)


def _write_frames(directory: Path, round_index: int) -> Path:
    path = directory / f"frames_move_to_actions_record_0{round_index}.npz"
    shape = (DECISIONS, len(WORLDS), HEIGHT, WIDTH, 3)
    np.savez_compressed(
        path,
        world_index=np.asarray(WORLDS),
        decisions=np.int64(DECISIONS),
        overview=np.zeros(shape, np.uint8),
        wrist=np.zeros(shape, np.uint8),
    )
    return path


def _bank_rows(rounds: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The uids --mode dataset writes, plus one world that kept no frames."""

    uids, decisions = [], []
    for round_index in rounds:
        for world in WORLDS:
            for decision in range(DECISIONS):
                uids.append(
                    "move_to_demos/"
                    f"replay_move_to_actions_record_0{round_index}"
                    f"/r{round_index}w{world}"
                )
                decisions.append(decision)
    # A world outside WORLDS: its replay kept no pictures, so it must be
    # dropped rather than paired with some other episode's frame.
    uids.append("move_to_demos/replay_move_to_actions_record_00/r0w7")
    decisions.append(0)
    return np.asarray(uids, dtype="U128"), np.asarray(decisions, np.int64)


class FrameJoinTests(unittest.TestCase):
    def test_both_writers_reduce_to_the_same_identity(self) -> None:
        self.assertEqual(
            frame_join_key("frames_move_to_actions_record_00"),
            frame_join_key(
                "move_to_demos/replay_move_to_actions_record_00"
            ),
        )

    def test_bank_rows_resolve_against_the_tap(self) -> None:
        with _TempDir() as directory:
            for round_index in (0, 1):
                _write_frames(directory, round_index)
            meta = load_frame_meta(sorted(directory.glob("frames_*.npz")))
            uids, decisions = _bank_rows((0, 1))
            found, lookups = resolve_frame_rows(uids, decisions, meta)

            expected = len(WORLDS) * DECISIONS * 2
            self.assertEqual(int(found.sum()), expected)
            self.assertEqual(len(lookups), expected)
            # The frameless world is the one that was dropped.
            self.assertFalse(bool(found[-1]))

    def test_rows_group_into_one_pass_per_file(self) -> None:
        with _TempDir() as directory:
            for round_index in (0, 1):
                _write_frames(directory, round_index)
            meta = load_frame_meta(sorted(directory.glob("frames_*.npz")))
            uids, decisions = _bank_rows((0, 1))
            found, lookups = resolve_frame_rows(uids, decisions, meta)
            resolved = np.flatnonzero(found)
            position = {int(row): i for i, row in enumerate(resolved)}

            grouped = group_rows_by_file(
                resolved.tolist(), lookups, position
            )
            self.assertEqual(sorted(grouped), sorted(meta))
            self.assertEqual(
                sum(len(rows) for rows in grouped.values()), resolved.size
            )
            for rows in grouped.values():
                self.assertEqual(len(rows), len(WORLDS) * DECISIONS)

    def test_materialized_frames_line_up_with_their_rows(self) -> None:
        with _TempDir() as directory:
            _write_frames(directory, 0)
            meta = load_frame_meta(sorted(directory.glob("frames_*.npz")))
            uids, decisions = _bank_rows((0,))
            found, lookups = resolve_frame_rows(uids, decisions, meta)
            resolved = np.flatnonzero(found)
            position = {int(row): i for i, row in enumerate(resolved)}

            for rows in group_rows_by_file(
                resolved.tolist(), lookups, position
            ).values():
                overview, wrist = materialize_frames(
                    meta, lookups, [position[row] for row in rows]
                )
                # One picture per row, in the order the rows were asked for --
                # the refresh writes results back by that same order.
                self.assertEqual(
                    overview.shape, (len(rows), HEIGHT, WIDTH, 3)
                )
                self.assertEqual(wrist.shape, overview.shape)


class _TempDir:
    def __enter__(self) -> Path:
        import tempfile

        self._handle = tempfile.TemporaryDirectory()
        return Path(self._handle.name)

    def __exit__(self, *exc: object) -> None:
        self._handle.cleanup()


if __name__ == "__main__":
    unittest.main()
