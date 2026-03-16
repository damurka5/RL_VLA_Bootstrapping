from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_mujoco.policy_control import (
    CDPRPolicyControlSpec,
    apply_normalized_cdpr_action,
    policy_action_frequency_hz,
)


class _FakeSim:
    def __init__(self):
        self.ee = np.array([0.0, 0.0, 0.30], dtype=np.float32)
        self.target = self.ee.copy()
        self.yaw = 0.10
        self.gripper = 0.03
        self.capture_history: list[bool] = []
        self.closed = 0
        self.opened = 0

    def get_end_effector_position(self):
        return self.ee.copy()

    def set_target_position(self, xyz):
        self.target = np.asarray(xyz, dtype=np.float32).copy()

    def get_yaw(self):
        return float(self.yaw)

    def set_yaw(self, yaw):
        self.yaw = float(yaw)

    def close_gripper(self):
        self.closed += 1
        self.gripper = 0.0

    def open_gripper(self):
        self.opened += 1
        self.gripper = 0.03

    def get_gripper_opening(self):
        return float(self.gripper)

    def run_simulation_step(self, capture_frame=True):
        self.capture_history.append(bool(capture_frame))
        self.ee = self.target.copy()


class PolicyControlTests(unittest.TestCase):
    def test_apply_normalized_action_matches_training_style_scaling(self):
        sim = _FakeSim()
        spec = CDPRPolicyControlSpec(
            xyz_limits=((-0.8, 0.8), (-0.8, 0.8), (0.08, 1.2)),
            action_step_xyz=0.006,
            action_step_yaw=0.08,
            hold_steps=4,
        )

        result = apply_normalized_cdpr_action(
            sim,
            np.array([1.0, -1.0, 0.5, 1.0, 0.3], dtype=np.float32),
            spec,
            ee_min_z=0.12,
            capture_last_frame=True,
        )

        np.testing.assert_allclose(result["target_xyz"], np.array([0.006, -0.006, 0.303], dtype=np.float32))
        self.assertAlmostEqual(result["target_yaw"], 0.18, places=6)
        self.assertEqual(sim.closed, 1)
        self.assertEqual(sim.opened, 0)
        self.assertEqual(result["sim_steps"], 5)
        self.assertEqual(sim.capture_history, [False, False, False, False, True])

    def test_apply_normalized_action_clamps_workspace_and_respects_min_z(self):
        sim = _FakeSim()
        sim.ee = np.array([0.799, 0.799, 0.09], dtype=np.float32)
        spec = CDPRPolicyControlSpec(
            xyz_limits=((-0.8, 0.8), (-0.8, 0.8), (0.08, 1.2)),
            action_step_xyz=0.006,
            action_step_yaw=0.08,
            hold_steps=0,
        )

        result = apply_normalized_cdpr_action(
            sim,
            np.array([1.0, 1.0, -1.0, 0.0, -0.5], dtype=np.float32),
            spec,
            ee_min_z=0.12,
            capture_last_frame=False,
        )

        np.testing.assert_allclose(result["target_xyz"], np.array([0.8, 0.8, 0.12], dtype=np.float32))
        self.assertEqual(sim.opened, 1)
        self.assertEqual(sim.capture_history, [False])

    def test_policy_frequency_matches_one_plus_hold_steps(self):
        self.assertAlmostEqual(policy_action_frequency_hz(1.0 / 60.0, hold_steps=4), 12.0, places=6)


if __name__ == "__main__":
    unittest.main()
