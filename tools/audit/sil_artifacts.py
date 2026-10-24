"""Atomic SIL outputs and cheap checks for a completed replay/frame pair."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def atomic_savez(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def valid_npz(path: Path, required: tuple[str, ...]) -> bool:
    """Check small datasets fully without materializing their arrays."""

    try:
        with zipfile.ZipFile(path) as archive:
            return {f"{key}.npy" for key in required} <= set(archive.namelist()) and archive.testzip() is None
    except (OSError, ValueError, zipfile.BadZipFile, EOFError):
        return False


def replay_marker(replay: Path) -> Path:
    return replay.with_suffix(".complete.json")


def mark_replay_complete(replay: Path, frames: Path, *, frame_successes: int) -> None:
    marker = replay_marker(replay)
    payload = {
        "version": 1,
        "replay_bytes": replay.stat().st_size,
        "frames_bytes": frames.stat().st_size,
        "frame_successes": int(frame_successes),
    }
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(marker)


def completed_replay(replay: Path, frames: Path) -> dict[str, Any] | None:
    """Only atomic pairs with a matching completion record may be reused.

    Do not decompress GB-sized image arrays just to decide whether to resume.
    A crash before either rename or the marker leaves this pair incomplete.
    """

    try:
        marker = json.loads(replay_marker(replay).read_text(encoding="utf-8"))
        if (marker["version"] != 1 or marker["replay_bytes"] != replay.stat().st_size
                or marker["frames_bytes"] != frames.stat().st_size):
            return None
        for path, required in ((replay, {"actions.npy", "success.npy"}),
                               (frames, {"overview.npy", "wrist.npy", "world_index.npy"})):
            with zipfile.ZipFile(path) as archive:
                if not required <= set(archive.namelist()):
                    return None
        return marker
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        return None
