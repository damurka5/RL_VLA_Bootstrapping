"""Stream full-task evaluation episodes to MP4 and keep the ones worth watching.

Used by ``evaluate_cdpr_full_put_into.py --video-dir``. Every evaluated world
streams to its own ffmpeg pipe while the rollout runs, because whether an
episode succeeds is only known at the end, and buffering 64 worlds x ~512
action steps of two 320x240 cameras on the host is ~15 GB. When the rollout
finishes, episodes whose outcome matches the requested filter are renamed into
place with a JSON sidecar; the rest are deleted.

Frames are the policy's own cameras (overview | wrist, side by side), rendered
after every executed action, so the video shows what the policy saw at each
decision plus the motion between decisions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

OUTCOME_FILTERS = ("strict", "native", "failed", "all")


class Mp4Pipe:
    """One ffmpeg process fed raw RGB frames on stdin."""

    def __init__(self, path: Path, *, fps: float, height: int, width: int) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to encode episode MP4s.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{int(width)}x{int(height)}",
                "-r", f"{float(fps):.6f}", "-i", "-",
                # Many pipes run at once: one encoder thread each.
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-threads", "1", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        details = ""
        if self.process.stderr is not None:
            details = self.process.stderr.read().decode("utf-8", "replace")
            self.process.stderr.close()
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited {code} for {self.path}: {details}")


def keep_episode(outcome_filter: str, *, native: bool, strict: bool) -> bool:
    if outcome_filter == "strict":
        return strict
    if outcome_filter == "native":
        return native
    if outcome_filter == "failed":
        return not strict
    if outcome_filter == "all":
        return True
    raise ValueError(f"Unknown outcome filter {outcome_filter!r}; known: {OUTCOME_FILTERS}")


class EpisodeVideoRecorder:
    """Per-world MP4 streams for one evaluation round."""

    def __init__(
        self,
        output_dir: Path,
        *,
        fps: float = 20.0,
        hold_seconds: float = 1.0,
        outcome_filter: str = "strict",
        max_videos: int = 0,
        writer_factory: Callable[..., Any] = Mp4Pipe,
    ) -> None:
        if outcome_filter not in OUTCOME_FILTERS:
            raise ValueError(f"outcome_filter must be one of {OUTCOME_FILTERS}")
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.hold_frames = max(0, int(round(float(hold_seconds) * self.fps)))
        self.outcome_filter = outcome_filter
        self.max_videos = int(max_videos)
        self.writer_factory = writer_factory
        self.kept: list[dict[str, Any]] = []
        self._writers: dict[int, Any] = {}
        self._last: dict[int, np.ndarray] = {}
        self._frames: dict[int, int] = {}
        self._events: dict[int, dict[str, int]] = {}
        self._round = 0

    @property
    def full(self) -> bool:
        return self.max_videos > 0 and len(self.kept) >= self.max_videos

    def start_round(self, round_index: int, worlds: int, *, height: int, width: int) -> None:
        self._round = int(round_index)
        partial = self.output_dir / ".partial"
        for world in range(int(worlds)):
            path = partial / f"round{self._round:02d}_world{world:03d}.mp4"
            self._writers[world] = self.writer_factory(
                path, fps=self.fps, height=int(height), width=2 * int(width)
            )
            self._frames[world] = 0
            self._events[world] = {}

    def write(self, cameras: Any, active: Any) -> None:
        """Append overview|wrist for every active world. ``cameras`` is BCHW in [0,1]."""

        import torch

        rows = torch.nonzero(active.to(dtype=torch.bool), as_tuple=False).reshape(-1)
        if int(rows.numel()) == 0:
            return
        both = torch.cat((cameras.overview[rows], cameras.wrist[rows]), dim=-1)
        frames = (
            (both.clamp(0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
            .permute(0, 2, 3, 1).contiguous().cpu().numpy()
        )
        for frame, world in zip(frames, rows.tolist()):
            self._writers[world].write(frame)
            self._last[world] = frame
            self._frames[world] += 1

    def mark(self, name: str, fired: Any) -> None:
        """Record the frame index at which each world first shows ``fired``."""

        for world in np.flatnonzero(np.asarray(fired.detach().cpu().numpy(), dtype=bool)):
            self._events[int(world)].setdefault(name, self._frames[int(world)])

    def finish_round(
        self,
        *,
        native: Sequence[bool],
        strict: Sequence[bool],
        metadata: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        kept_now = []
        for world, writer in sorted(self._writers.items()):
            last = self._last.get(world)
            if last is not None:
                for _ in range(self.hold_frames):
                    writer.write(last)
            writer.close()
            path = Path(writer.path)
            is_native, is_strict = bool(native[world]), bool(strict[world])
            keep = (
                self._frames[world] > 0
                and not self.full
                and keep_episode(self.outcome_filter, native=is_native, strict=is_strict)
            )
            if not keep:
                path.unlink(missing_ok=True)
                continue
            info = dict(metadata[world])
            label = "strict" if is_strict else ("native" if is_native else "failed")
            stem = (
                f"{label}_{info.get('destination', 'dest')}_{info.get('target_catalog', 'object')}"
                f"_r{self._round:02d}_w{world:03d}"
            )
            final = self.output_dir / f"{stem}.mp4"
            final.parent.mkdir(parents=True, exist_ok=True)
            path.replace(final)
            record = {
                **info,
                "video": final.name,
                "round": self._round,
                "world": world,
                "native": is_native,
                "strict": is_strict,
                "frames": self._frames[world],
                "fps": self.fps,
                "event_frames": dict(self._events[world]),
                "event_seconds": {
                    name: round(index / self.fps, 2)
                    for name, index in self._events[world].items()
                },
            }
            final.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True))
            self.kept.append(record)
            kept_now.append(record)
        self._writers.clear()
        self._last.clear()
        partial = self.output_dir / ".partial"
        if partial.is_dir() and not any(partial.iterdir()):
            partial.rmdir()
        return kept_now

    def write_index(self) -> Path:
        path = self.output_dir / "videos.json"
        path.write_text(json.dumps(self.kept, indent=2, sort_keys=True))
        return path
