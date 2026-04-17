from __future__ import annotations

import types
import unittest

import numpy as np

import robots.cdpr.cdpr_dataset.rl_cdpr_env as rl_cdpr_env_mod
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

    def test_vertical_instruction_uses_workspace_center_target(self):
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
        np.testing.assert_allclose(goal, np.array([0.0, 0.0, 0.25], dtype=np.float32), atol=1e-7)

    def test_pick_up_instruction_uses_live_target_object_position(self):
        env = self._env()
        env._target_body_name = "apple_body"
        env._get_body_position = lambda body_name: np.array([0.07, -0.03, 0.17], dtype=np.float32)
        spec = InstructionSpec(
            instruction_type="pick_up",
            text="pick up apple",
            target_object="ycb_apple",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
        )

        goal = env._compute_instruction_goal(
            spec=spec,
            initial_ee_pos=np.array([0.12, -0.08, 0.40], dtype=np.float32),
        )

        np.testing.assert_allclose(goal, np.array([0.07, -0.03, 0.17], dtype=np.float32), atol=1e-7)

    def test_move_to_object_instruction_uses_live_target_object_position(self):
        env = self._env()
        env._target_body_name = "mug_body"
        env._get_body_position = lambda body_name: np.array([-0.05, 0.06, 0.19], dtype=np.float32)
        spec = InstructionSpec(
            instruction_type="move_to_object",
            text="move to mug",
            target_object="ycb_mug",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
        )

        goal = env._compute_instruction_goal(
            spec=spec,
            initial_ee_pos=np.array([0.12, -0.08, 0.40], dtype=np.float32),
        )

        np.testing.assert_allclose(goal, np.array([-0.05, 0.06, 0.19], dtype=np.float32), atol=1e-7)

    def test_get_body_position_falls_back_to_xpos_when_body_xpos_is_missing(self):
        env = self._env()
        env.sim = types.SimpleNamespace(
            model=object(),
            data=types.SimpleNamespace(
                xpos=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.11, -0.07, 0.23],
                    ],
                    dtype=np.float32,
                )
            ),
        )

        original_mj = rl_cdpr_env_mod.mj
        rl_cdpr_env_mod.mj = types.SimpleNamespace(
            mj_name2id=lambda model, obj_type, name: 1,
            mjtObj=types.SimpleNamespace(mjOBJ_BODY=0),
        )
        try:
            pos = env._get_body_position("apple_body")
        finally:
            rl_cdpr_env_mod.mj = original_mj

        np.testing.assert_allclose(pos, np.array([0.11, -0.07, 0.23], dtype=np.float32), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
