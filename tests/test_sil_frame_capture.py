"""Frames for the LoRA stage: what is tapped, when, and in what units.

The 512-wide vision feature in the demonstrations is a fixed random projection
taken under no_grad, so no gradient reaches the vision tower through it. LoRA
therefore needs the pictures, and the pictures have to be the ones the policy
was actually handed on the SMOOTHED rollout -- which only exists during replay.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch is required for the frame tap",
)
class DecisionFrameTapTests(unittest.TestCase):
    def _backend(self, torch, *, worlds=4, height=3, width=5):
        class Backend:
            def __init__(self) -> None:
                self.calls = 0

            def render_policy_cameras(self):
                self.calls += 1
                # A per-world ramp so a mis-indexed selection is visible, and a
                # per-call offset so a mis-ordered stack is too.
                base = torch.arange(worlds, dtype=torch.float32) / 255.0
                overview = base.view(worlds, 1, 1, 1).expand(
                    worlds, 3, height, width
                ).clone()
                wrist = overview + (self.calls / 255.0)
                return SimpleNamespace(overview=overview, wrist=wrist)

        return Backend()

    def test_it_keeps_only_the_asked_for_worlds_in_uint8(self):
        import torch

        from tools.audit.sil_record import _DecisionFrameTap

        backend = self._backend(torch)
        with _DecisionFrameTap(backend, [1, 3], torch) as tap:
            for _ in range(2):
                backend.render_policy_cameras()
        payload = tap.stack(decisions=2)
        self.assertEqual(payload["overview"].dtype.name, "uint8")
        self.assertEqual(payload["overview"].shape, (2, 2, 3, 5, 3))
        self.assertEqual(list(payload["world_index"]), [1, 3])
        # World 1 and world 3 carry their own ramp values, not world 0's.
        self.assertEqual(int(payload["overview"][0, 0, 0, 0, 0]), 1)
        self.assertEqual(int(payload["overview"][0, 1, 0, 0, 0]), 3)

    def test_the_frame_order_follows_the_decisions(self):
        import torch

        from tools.audit.sil_record import _DecisionFrameTap

        backend = self._backend(torch)
        with _DecisionFrameTap(backend, [0], torch) as tap:
            for _ in range(3):
                backend.render_policy_cameras()
        payload = tap.stack(decisions=3)
        # The wrist ramp carries the call index, so a reordered stack shows up.
        self.assertEqual(
            [int(payload["wrist"][d, 0, 0, 0, 0]) for d in range(3)], [1, 2, 3]
        )

    def test_a_frame_decision_mismatch_raises(self):
        """The failure it guards is silent: every frame paired with the wrong action."""

        import torch

        from tools.audit.sil_record import _DecisionFrameTap

        backend = self._backend(torch)
        with _DecisionFrameTap(backend, [0], torch) as tap:
            for _ in range(3):
                backend.render_policy_cameras()
        with self.assertRaises(RuntimeError) as caught:
            tap.stack(decisions=4)
        self.assertIn("paired with the wrong action", str(caught.exception))

    def test_it_restores_the_render_method(self):
        """Several arms run back to back over one live backend."""

        import torch

        from tools.audit.sil_record import _DecisionFrameTap

        backend = self._backend(torch)
        original = backend.render_policy_cameras
        with _DecisionFrameTap(backend, [0], torch):
            self.assertIsNot(backend.render_policy_cameras, original)
        self.assertNotIn("render_policy_cameras", vars(backend))
        backend.render_policy_cameras()

    def test_it_nests_under_the_video_tap_without_either_losing_frames(self):
        """Both wrap the same method; the inner one must call through."""

        import contextlib

        import torch

        from tools.audit.sil_record import _DecisionFrameTap
        from tools.audit.success_episode_videos import _FrameTap

        backend = self._backend(torch)
        backend.device = torch.device("cpu")
        with contextlib.ExitStack() as stack:
            video = stack.enter_context(_FrameTap(backend, [0], True))
            frames = stack.enter_context(_DecisionFrameTap(backend, [0, 2], torch))
            for _ in range(2):
                backend.render_policy_cameras()
        self.assertEqual(len(video.frames[0]), 2)
        self.assertEqual(frames.stack(decisions=2)["overview"].shape[0], 2)

    def test_no_worlds_means_no_capture_and_no_cost(self):
        import torch

        from tools.audit.sil_record import _DecisionFrameTap

        backend = self._backend(torch)
        with _DecisionFrameTap(backend, [], torch) as tap:
            backend.render_policy_cameras()
        self.assertEqual(tap.stack(decisions=0)["overview"].size, 0)


class FrameCliTests(unittest.TestCase):
    def test_the_flags_exist_and_default_to_off(self):
        """Frames are opt-in: a harvest that does not need them pays nothing."""

        import inspect

        from tools.audit import sil_record

        source = inspect.getsource(sil_record.main)
        self.assertIn('"--record-frames"', source)
        self.assertIn('"--frame-worlds"', source)
        self.assertIn('action="store_true"', source)


if __name__ == "__main__":
    unittest.main()
