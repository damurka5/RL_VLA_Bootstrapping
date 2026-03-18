from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv
from robots.cdpr.cdpr_dataset.rl_instruction_tasks import InstructionSpec


class InstructionGoalTests(unittest.TestCase):
    def _env(self) -> CDPRLanguageRLEnv:
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {}
        env.defaults = {}
        env._support_surface_z = 0.15
        env._ee_min_z = 0.18
        return env

    def test_lateral_instruction_uses_workspace_center_waypoint(self):
        env = self._env()
        spec = InstructionSpec(
            instruction_type="move_left",
            text="move left",
            target_object="",
            direction=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
        )

        goal = env._compute_instruction_goal(
            spec=spec,
            initial_ee_pos=np.array([0.12, -0.08, 0.40], dtype=np.float32),
        )

        np.testing.assert_allclose(goal, np.array([-0.40, 0.0, 0.25], dtype=np.float32), atol=1e-7)

    def test_vertical_instruction_keeps_initial_xy(self):
        env = self._env()
        spec = InstructionSpec(
            instruction_type="move_up",
            text="move up",
            target_object="",
            direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
        )
        initial = np.array([0.18, -0.14, 0.40], dtype=np.float32)

        goal = env._compute_instruction_goal(spec=spec, initial_ee_pos=initial)
        direction = env._compute_goal_motion_direction(
            initial_ee_pos=initial,
            goal_pos=goal,
            instruction_direction=spec.direction,
        )

        np.testing.assert_allclose(goal, np.array([0.18, -0.14, 0.50], dtype=np.float32), atol=1e-7)
        np.testing.assert_allclose(direction, np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
