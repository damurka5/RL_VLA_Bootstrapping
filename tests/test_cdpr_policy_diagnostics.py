from __future__ import annotations

import unittest

import numpy as np

from rl_vla_bootstrapping.cli.diagnose_cdpr_policy import (
    DiagnosticDemo,
    _build_axis_demos,
    _build_executed_action_sequence,
    _build_random_demos,
    _compute_visible_start_xyz,
    _locked_axes_for_demo,
    _summarize_demo,
)


class PolicyDiagnosticsTests(unittest.TestCase):
    def test_build_axis_demos_matches_chunk_size_and_signs(self):
        demos = _build_axis_demos(8, 0.25, include_negative=True)

        self.assertEqual([demo.name for demo in demos], [
            "axis_x_pos",
            "axis_x_neg",
            "axis_y_pos",
            "axis_y_neg",
            "axis_z_pos",
            "axis_z_neg",
        ])
        for demo in demos:
            self.assertEqual(demo.chunk.shape, (8, 5))
            self.assertTrue(np.all(np.abs(demo.chunk) <= 1.0))

        np.testing.assert_allclose(demos[0].chunk[:, 0], np.full((8,), 0.25, dtype=np.float32))
        np.testing.assert_allclose(demos[1].chunk[:, 0], np.full((8,), -0.25, dtype=np.float32))
        self.assertTrue(np.allclose(demos[0].chunk[:, 1:], 0.0))

    def test_build_random_demos_is_seeded_and_keeps_tool_dims_zero_by_default(self):
        demos_a = _build_random_demos(
            8,
            demo_count=1,
            magnitude=0.5,
            seed=7,
            randomize_yaw=False,
            randomize_gripper=False,
        )
        demos_b = _build_random_demos(
            8,
            demo_count=1,
            magnitude=0.5,
            seed=7,
            randomize_yaw=False,
            randomize_gripper=False,
        )

        self.assertEqual(len(demos_a), 1)
        np.testing.assert_allclose(demos_a[0].chunk, demos_b[0].chunk)
        self.assertTrue(np.all(np.abs(demos_a[0].chunk[:, :3]) <= 0.5))
        self.assertTrue(np.allclose(demos_a[0].chunk[:, 3:], 0.0))

    def test_build_executed_action_sequence_tiles_chunk(self):
        chunk = np.array(
            [
                [0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        executed = _build_executed_action_sequence(chunk, 3)

        self.assertEqual(executed.shape, (6, 5))
        np.testing.assert_allclose(executed[:2], chunk)
        np.testing.assert_allclose(executed[2:4], chunk)
        np.testing.assert_allclose(executed[4:6], chunk)

    def test_compute_visible_start_xyz_lifts_above_spawn_and_respects_limits(self):
        target = _compute_visible_start_xyz(
            np.array([0.01, -0.02, 0.21], dtype=np.float32),
            workspace_safety={"ee_spawn_z": 0.236},
            z_offset=0.05,
            z_limits=(0.08, 0.26),
        )

        np.testing.assert_allclose(target, np.array([0.01, -0.02, 0.26], dtype=np.float32), atol=1e-7)

    def test_locked_axes_for_axis_demo_pins_other_coordinates(self):
        demo = DiagnosticDemo(
            name="axis_x_pos",
            kind="axis",
            description="test",
            chunk=np.tile(np.array([[0.25, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32), (8, 1)),
        )

        self.assertEqual(_locked_axes_for_demo(demo, lock_non_commanded_axes=True), (1, 2))
        self.assertEqual(_locked_axes_for_demo(demo, lock_non_commanded_axes=False), ())

    def test_summarize_demo_aggregates_realized_and_commanded_motion(self):
        demo = DiagnosticDemo(
            name="axis_x_pos",
            kind="axis",
            description="test",
            chunk=np.zeros((2, 5), dtype=np.float32),
        )
        records = [
            {
                "ee_before": [0.0, 0.0, 0.30],
                "ee_after": [0.002, 0.0, 0.30],
                "commanded_xyz_delta_raw": [0.003, 0.0, 0.0],
                "commanded_xyz_delta_effective": [0.003, 0.0, 0.0],
                "realized_xyz_delta": [0.002, 0.0, 0.0],
                "realized_vs_command_gain": 2.0 / 3.0,
                "realized_vs_command_cosine": 1.0,
            },
            {
                "ee_before": [0.002, 0.0, 0.30],
                "ee_after": [0.0035, 0.0, 0.30],
                "commanded_xyz_delta_raw": [0.003, 0.0, 0.0],
                "commanded_xyz_delta_effective": [0.003, 0.0, 0.0],
                "realized_xyz_delta": [0.0015, 0.0, 0.0],
                "realized_vs_command_gain": 0.5,
                "realized_vs_command_cosine": 1.0,
            },
        ]

        summary = _summarize_demo(demo, records)

        np.testing.assert_allclose(summary["commanded_total_xyz_effective"], [0.006, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(summary["realized_total_xyz"], [0.0035, 0.0, 0.0], atol=1e-7)
        self.assertAlmostEqual(summary["commanded_total_distance_mm"], 6.0, places=5)
        self.assertAlmostEqual(summary["realized_total_distance_mm"], 3.5, places=5)
        self.assertAlmostEqual(summary["mean_gain"], (2.0 / 3.0 + 0.5) / 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
