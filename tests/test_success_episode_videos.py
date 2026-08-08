"""The seed check has to be able to say "these successes are luck".

The self-imitation loop distils the policy's own successful episodes. That only
works if those episodes contain approach behaviour. At the 0.13 m cap the rollout
budget allows ~0.78 m of travel, so a success from a far start can be a random
walk that ended on the object -- and distilling those teaches luck. The report is
what decides it, so it has to separate the two cases on data where the answer is
known.

The frame tap is tested for the property that makes the video trustworthy: it
tees the tensors the policy was actually handed rather than rendering again, and
it puts the backend back afterwards.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit" / "success_episode_videos.py"

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _load():
    spec = importlib.util.spec_from_file_location("success_episode_videos", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load()


def _rows(*, far_success_cos, far_failure_cos, count=60):
    rng = np.random.RandomState(0)
    rows = []
    for index in range(count):
        rows.append(
            {
                "round": 0,
                "world": index,
                "success": True,
                "ever_grasped": True,
                "start_distance_m": 0.10,
                "far_start": True,
                "cosine_decision0": float(
                    rng.normal(far_success_cos, 0.02)
                ),
                "tracked": False,
            }
        )
        rows.append(
            {
                "round": 0,
                "world": count + index,
                "success": False,
                "ever_grasped": False,
                "start_distance_m": 0.10,
                "far_start": True,
                "cosine_decision0": float(
                    rng.normal(far_failure_cos, 0.02)
                ),
                "tracked": False,
            }
        )
    return rows


class SeedCheckTest(unittest.TestCase):
    def _report(self, rows):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            tool._report(rows, far_start_m=0.06)
        return buffer.getvalue()

    def test_a_real_seed_separates_success_from_failure(self) -> None:
        text = self._report(
            _rows(far_success_cos=0.60, far_failure_cos=0.05)
        )
        self.assertIn("success, far", text)
        self.assertIn("+0.60", text)
        self.assertIn("+0.05", text)

    def test_luck_shows_as_matching_rows(self) -> None:
        """The case the whole check exists for.

        Successes and failures from the same far starts aiming identically
        means the successes are a random walk. The report must make that
        visible rather than showing only the success row.
        """

        text = self._report(
            _rows(far_success_cos=0.055, far_failure_cos=0.055)
        )
        lines = [line for line in text.splitlines() if "far (>=" in line]
        self.assertEqual(len(lines), 2, msg=text)
        values = [float(line.split()[-2]) for line in lines]
        self.assertAlmostEqual(values[0], values[1], delta=0.02)

    def test_empty_input_does_not_crash(self) -> None:
        self.assertIn("no episodes", self._report([]))

    def test_groups_with_no_episodes_report_na_rather_than_a_number(self) -> None:
        rows = _rows(far_success_cos=0.5, far_failure_cos=0.05)
        for row in rows:
            row["far_start"] = True
        text = self._report(rows)
        self.assertIn("n/a", text)


@unittest.skipIf(torch is None, "torch is unavailable")
class FrameTapTest(unittest.TestCase):
    class _Backend:
        def __init__(self):
            self.calls = 0

        def render_policy_cameras(self):
            self.calls += 1
            overview = torch.zeros((4, 3, 2, 2))
            wrist = torch.ones((4, 3, 2, 2))
            # A per-world value so a mixed-up index is visible.
            for world in range(4):
                overview[world] = world / 10.0
            return SimpleNamespace(overview=overview, wrist=wrist)

    def test_it_tees_rather_than_rendering_again(self) -> None:
        """A second render would cost as much as the rollout.

        Worse, it would not be guaranteed to show what the policy acted on.
        """

        backend = self._Backend()
        with tool._FrameTap(backend, [0, 2], both_cameras=False):
            backend.render_policy_cameras()
            backend.render_policy_cameras()
        self.assertEqual(backend.calls, 2)

    def test_frames_are_kept_per_world_and_in_order(self) -> None:
        backend = self._Backend()
        with tool._FrameTap(backend, [0, 2], both_cameras=False) as tap:
            backend.render_policy_cameras()
            backend.render_policy_cameras()
            frames = {w: list(v) for w, v in tap.frames.items()}
        self.assertEqual(sorted(frames), [0, 2])
        self.assertEqual(len(frames[0]), 2)
        # World 2's frames must not be world 0's.
        self.assertFalse(np.array_equal(frames[0][0], frames[2][0]))

    def test_the_wrist_view_is_appended_beside_the_overview(self) -> None:
        backend = self._Backend()
        with tool._FrameTap(backend, [0], both_cameras=True) as tap:
            backend.render_policy_cameras()
            width_both = tap.frames[0][0].shape[1]
        with tool._FrameTap(backend, [0], both_cameras=False) as tap:
            backend.render_policy_cameras()
            width_one = tap.frames[0][0].shape[1]
        self.assertEqual(width_both, 2 * width_one)

    def test_the_backend_is_restored(self) -> None:
        backend = self._Backend()
        with tool._FrameTap(backend, [0], both_cameras=False):
            pass
        self.assertNotIn("render_policy_cameras", vars(backend))
        backend.render_policy_cameras()
        self.assertEqual(backend.calls, 1)

    def test_frames_are_uint8_rgb(self) -> None:
        backend = self._Backend()
        with tool._FrameTap(backend, [0], both_cameras=False) as tap:
            backend.render_policy_cameras()
            frame = tap.frames[0][0]
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(frame.shape[-1], 3)


if __name__ == "__main__":
    unittest.main()
