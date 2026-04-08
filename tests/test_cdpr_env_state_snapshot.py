from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv
from robots.cdpr.cdpr_dataset.rl_instruction_tasks import InstructionSpec, RewardState


class _FakeSim:
    def __init__(self):
        self.state = {"qpos": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        self.language_instruction = ""
        self.restore_calls = 0

    def capture_state(self):
        return {
            "state": self.state["qpos"].copy(),
            "language_instruction": self.language_instruction,
        }

    def restore_state(self, snapshot):
        self.state["qpos"] = np.asarray(snapshot["state"], dtype=np.float64).copy()
        self.language_instruction = str(snapshot["language_instruction"])
        self.restore_calls += 1


class CDPREnvStateSnapshotTests(unittest.TestCase):
    def _make_env(self) -> CDPRLanguageRLEnv:
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.sim = _FakeSim()
        env._step_count = 5
        env._yaw = 0.25
        env._last_gripper_cmd = -0.5
        env._instruction_spec = InstructionSpec(
            instruction_type="move_to_object",
            text="move to apple",
            target_object="ycb_apple",
            direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            target_displacement=0.0,
            lift_target=0.0,
        )
        env._reward_state = RewardState(
            initial_ee_pos=np.array([0.0, 0.0, 0.4], dtype=np.float32),
            initial_obj_pos=np.array([0.1, 0.2, 0.05], dtype=np.float32),
            prev_ee_pos=np.array([0.0, 0.0, 0.4], dtype=np.float32),
            prev_obj_pos=np.array([0.1, 0.2, 0.05], dtype=np.float32),
            prev_distance=0.12,
            prev_camera_align=0.5,
            gripper_closed=False,
            grasped=False,
            step_count=5,
        )
        env._scene_name = "desk"
        env._target_catalog_name = "ycb_apple"
        env._target_body_name = "p0_ycb_apple"
        env._catalog_to_body = {"ycb_apple": "p0_ycb_apple"}
        env._object_body_names = ["p0_ycb_apple", "p1_ycb_bowl"]
        env._scene_catalog_objects = ["ycb_apple", "ycb_bowl"]
        env._desk_texture_name = "oak.png"
        env._current_wrapper_xml = None
        env._inverse_catalog_to_body = {"p0_ycb_apple": "ycb_apple"}
        env._prev_object_positions = {
            "p0_ycb_apple": np.array([0.1, 0.2, 0.05], dtype=np.float32),
        }
        env._prev_ee_for_catch = np.array([0.0, 0.0, 0.4], dtype=np.float32)
        env._last_caught_body = ""
        env._last_caught_catalog = ""
        env._support_surface_z = 0.05
        env._ee_min_z = 0.15
        env._ee_spawn_z = 0.16
        env._locked_target_xyz = np.array([0.0, 0.0, 0.4], dtype=np.float32)
        env._episode_ee_start = np.array([0.0, 0.0, 0.4], dtype=np.float32)
        env._goal_position = np.array([0.1, 0.2, 0.05], dtype=np.float32)
        env._goal_motion_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        env._episode_index = 3
        env._reset_counter = 7
        env._instruction_cycle = ["move_left", "move_to_object"]
        env.np_random = np.random.default_rng(42)
        env.sim.language_instruction = env._instruction_spec.text
        return env

    def test_capture_and_restore_state_round_trip(self):
        env = self._make_env()
        snapshot = env.capture_state()

        env.sim.state["qpos"][0] = 99.0
        env._step_count = 0
        env._yaw = -1.0
        env._last_gripper_cmd = 1.0
        env._scene_name = "mutated"
        env._target_catalog_name = "other"
        env._catalog_to_body["other"] = "body"
        env._object_body_names.append("extra")
        env._prev_object_positions["p0_ycb_apple"][0] = -5.0
        env._locked_target_xyz[2] = -2.0
        env._instruction_cycle.clear()
        env.np_random = np.random.default_rng(999)

        env.restore_state(snapshot)

        np.testing.assert_allclose(env.sim.state["qpos"], np.array([1.0, 2.0, 3.0], dtype=np.float64))
        self.assertEqual(env.sim.restore_calls, 1)
        self.assertEqual(env.sim.language_instruction, "move to apple")
        self.assertEqual(env._step_count, 5)
        self.assertEqual(env._yaw, 0.25)
        self.assertEqual(env._last_gripper_cmd, -0.5)
        self.assertEqual(env._scene_name, "desk")
        self.assertEqual(env._target_catalog_name, "ycb_apple")
        self.assertEqual(env._catalog_to_body, {"ycb_apple": "p0_ycb_apple"})
        self.assertEqual(env._object_body_names, ["p0_ycb_apple", "p1_ycb_bowl"])
        np.testing.assert_allclose(
            env._prev_object_positions["p0_ycb_apple"],
            np.array([0.1, 0.2, 0.05], dtype=np.float32),
        )
        np.testing.assert_allclose(env._locked_target_xyz, np.array([0.0, 0.0, 0.4], dtype=np.float32))
        self.assertEqual(env._instruction_cycle, ["move_left", "move_to_object"])


if __name__ == "__main__":
    unittest.main()
