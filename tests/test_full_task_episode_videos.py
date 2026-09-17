"""Full-task evaluation video recorder: keep filter, sidecars, event times."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.audit.full_task_episode_videos import EpisodeVideoRecorder, keep_episode


class _FakeWriter:
    def __init__(self, path, *, fps, height, width):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"")
        self.shape = (height, width, 3)
        self.frames = []

    def write(self, frame):
        assert frame.shape == self.shape, (frame.shape, self.shape)
        self.frames.append(frame)

    def close(self):
        self.path.write_bytes(bytes(len(self.frames)))


def _cameras(worlds, value):
    image = torch.full((worlds, 3, 4, 5), float(value))
    return SimpleNamespace(overview=image, wrist=image * 0.5)


class RecorderTests(unittest.TestCase):
    def run_round(self, directory, outcome_filter="strict", max_videos=0):
        recorder = EpisodeVideoRecorder(Path(directory), fps=10, hold_seconds=0.2,
                                        outcome_filter=outcome_filter, max_videos=max_videos,
                                        writer_factory=_FakeWriter)
        recorder.start_round(3, 3, height=4, width=5)
        recorder.write(_cameras(3, 0.1), torch.tensor([True, True, True]))
        # World 1 terminates after the first action; worlds 0 and 2 continue.
        recorder.write(_cameras(3, 0.2), torch.tensor([True, True, True]))
        recorder.mark("grasp", torch.tensor([True, False, False]))
        recorder.write(_cameras(3, 0.3), torch.tensor([True, False, True]))
        recorder.mark("grasp", torch.tensor([True, False, True]))
        kept = recorder.finish_round(
            native=[True, True, False], strict=[True, False, False],
            metadata=[{"destination": "bowl", "target_catalog": "apple", "scene_uid": str(i)}
                      for i in range(3)])
        return recorder, kept

    def test_keeps_only_strict_successes_with_sidecar_and_event_times(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder, kept = self.run_round(directory)
            self.assertEqual([row["world"] for row in kept], [0])
            videos = sorted(p.name for p in Path(directory).glob("*.mp4"))
            self.assertEqual(videos, ["strict_bowl_apple_r03_w000.mp4"])
            sidecar = json.loads(Path(directory, "strict_bowl_apple_r03_w000.json").read_text())
            self.assertEqual(sidecar["frames"], 3)
            # First grasp is marked after the second frame and never moves later.
            self.assertEqual(sidecar["event_frames"], {"grasp": 2})
            self.assertEqual(sidecar["event_seconds"], {"grasp": 0.2})
            # Three frames plus the two-frame terminal hold.
            self.assertEqual(Path(directory, videos[0]).stat().st_size, 5)
            self.assertFalse(Path(directory, ".partial").exists())
            self.assertEqual(json.loads(recorder.write_index().read_text())[0]["video"], videos[0])

    def test_failed_filter_and_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            _, kept = self.run_round(directory, outcome_filter="failed", max_videos=1)
            self.assertEqual([row["world"] for row in kept], [1])
            self.assertEqual(kept[0]["video"], "native_bowl_apple_r03_w001.mp4")

    def test_frame_is_overview_then_wrist(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeVideoRecorder(Path(directory), outcome_filter="all",
                                            writer_factory=_FakeWriter, hold_seconds=0)
            recorder.start_round(0, 1, height=4, width=5)
            writer = recorder._writers[0]
            recorder.write(_cameras(1, 1.0), torch.tensor([True]))
            frame = writer.frames[0]
            self.assertEqual(frame.shape, (4, 10, 3))
            self.assertEqual(int(frame[0, 0, 0]), 255)
            self.assertEqual(int(frame[0, 9, 0]), 128)

    def test_keep_episode_rejects_unknown_filter(self):
        with self.assertRaises(ValueError):
            keep_episode("lucky", native=True, strict=True)

    @unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not installed")
    def test_real_ffmpeg_encode(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeVideoRecorder(Path(directory), fps=10, outcome_filter="all")
            recorder.start_round(0, 2, height=16, width=16)
            for value in torch.linspace(0, 1, 12):
                recorder.write(SimpleNamespace(overview=torch.full((2, 3, 16, 16), float(value)),
                                               wrist=torch.zeros(2, 3, 16, 16)),
                               torch.tensor([True, True]))
            kept = recorder.finish_round(native=[True, False], strict=[True, False],
                                         metadata=[{}, {}])
            self.assertEqual(len(kept), 2)
            for row in kept:
                self.assertGreater(Path(directory, row["video"]).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
