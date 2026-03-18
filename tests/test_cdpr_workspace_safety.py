from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from robots.cdpr.cdpr_dataset.synthetic_tasks import prepare_cdpr_workspace


class _FakeWorkspaceSim:
    def __init__(self, z: float):
        self.ee = np.array([0.0, 0.0, z], dtype=np.float32)
        self.target = self.ee.copy()
        self.hold_calls: list[int] = []
        self.goto_calls: list[tuple[np.ndarray, int, float]] = []
        self.trajectory_data = [object()]
        self.overview_frames = [object()]
        self.ee_camera_frames = [object()]
        self.frame_capture_timestamps = [0.1]

    def get_end_effector_position(self):
        return self.ee.copy()

    def set_target_position(self, xyz):
        self.target = np.asarray(xyz, dtype=np.float32).copy()

    def goto(self, target, max_steps=120, tol=0.01):
        target_arr = np.asarray(target, dtype=np.float32).copy()
        self.goto_calls.append((target_arr, int(max_steps), float(tol)))
        self.ee = target_arr.copy()
        return True, 1

    def hold_current_pose(self, warm_steps=0):
        self.hold_calls.append(int(warm_steps))


class WorkspaceSafetyTests(unittest.TestCase):
    @mock.patch("robots.cdpr.cdpr_dataset.synthetic_tasks.body_bottom_offset", return_value=0.156)
    @mock.patch("robots.cdpr.cdpr_dataset.synthetic_tasks.infer_workspace_surface_z", return_value=0.0)
    def test_prepare_workspace_lifts_and_clears_buffers(self, _surface_mock, _bottom_mock):
        sim = _FakeWorkspaceSim(z=0.15)

        safety = prepare_cdpr_workspace(
            sim,
            initial_hold_warm_steps=10,
            clear_recordings=True,
        )

        self.assertTrue(safety["lifted_to_spawn_height"])
        self.assertAlmostEqual(float(safety["ee_min_z"]), 0.206, places=6)
        self.assertAlmostEqual(float(safety["ee_spawn_z"]), 0.236, places=6)
        np.testing.assert_allclose(sim.goto_calls[0][0], np.array([0.0, 0.0, 0.236], dtype=np.float32), atol=1e-6)
        self.assertEqual(sim.hold_calls, [10, 6])
        self.assertEqual(sim.trajectory_data, [])
        self.assertEqual(sim.overview_frames, [])
        self.assertEqual(sim.ee_camera_frames, [])
        self.assertEqual(sim.frame_capture_timestamps, [])

    @mock.patch("robots.cdpr.cdpr_dataset.synthetic_tasks.body_bottom_offset", return_value=0.156)
    @mock.patch("robots.cdpr.cdpr_dataset.synthetic_tasks.infer_workspace_surface_z", return_value=0.0)
    def test_prepare_workspace_skips_lift_when_robot_is_already_high_enough(self, _surface_mock, _bottom_mock):
        sim = _FakeWorkspaceSim(z=0.30)

        safety = prepare_cdpr_workspace(
            sim,
            initial_hold_warm_steps=4,
            clear_recordings=False,
        )

        self.assertFalse(safety["lifted_to_spawn_height"])
        self.assertEqual(sim.goto_calls, [])
        self.assertEqual(sim.hold_calls, [4])


if __name__ == "__main__":
    unittest.main()
