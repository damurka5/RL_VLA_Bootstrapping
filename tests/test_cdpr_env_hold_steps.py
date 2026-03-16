from __future__ import annotations

import types
import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv


class _FakeSim:
    def __init__(self):
        self.ee = np.array([0.0, 0.0, 0.30], dtype=np.float32)
        self.target = self.ee.copy()
        self.yaw = 0.10
        self.gripper = 0.03
        self.capture_history: list[bool] = []
        self.step_commands: list[tuple[np.ndarray, float, float]] = []
        self.closed = 0
        self.opened = 0

    def get_end_effector_position(self):
        return self.ee.copy()

    def set_target_position(self, xyz):
        self.target = np.asarray(xyz, dtype=np.float32).copy()

    def set_yaw(self, yaw):
        self.yaw = float(yaw)

    def close_gripper(self):
        self.closed += 1
        self.gripper = 0.0

    def open_gripper(self):
        self.opened += 1
        self.gripper = 0.03

    def run_simulation_step(self, capture_frame=True):
        self.capture_history.append(bool(capture_frame))
        self.step_commands.append((self.target.copy(), float(self.yaw), float(self.gripper)))
        self.ee = self.target.copy()


class CDPREnvHoldStepTests(unittest.TestCase):
    def test_apply_action_holds_same_command_for_all_substeps(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        sim = _FakeSim()

        env.sim = sim
        env.action_step_xyz = 0.006
        env.action_step_yaw = 0.08
        env.capture_frames = True
        env.hold_steps = 4
        env._yaw = 0.10
        env._ee_min_z = 0.12
        env._last_gripper_cmd = 0.0
        env._get_ee_position = types.MethodType(lambda self: self.sim.get_end_effector_position(), env)
        env._set_ee_target = types.MethodType(lambda self, xyz: self.sim.set_target_position(xyz), env)

        env._apply_action(np.array([1.0, -1.0, 0.5, 1.0, 0.3], dtype=np.float32))

        np.testing.assert_allclose(sim.target, np.array([0.006, -0.006, 0.303], dtype=np.float32))
        self.assertAlmostEqual(sim.yaw, 0.18, places=6)
        self.assertEqual(sim.closed, 1)
        self.assertEqual(sim.opened, 0)
        self.assertEqual(sim.capture_history, [False, False, False, False, True])
        self.assertEqual(len(sim.step_commands), 5)

        for target_xyz, yaw, gripper in sim.step_commands:
            np.testing.assert_allclose(target_xyz, np.array([0.006, -0.006, 0.303], dtype=np.float32))
            self.assertAlmostEqual(yaw, 0.18, places=6)
            self.assertAlmostEqual(gripper, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
