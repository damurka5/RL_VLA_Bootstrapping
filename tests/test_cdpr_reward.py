from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
    InstructionSpec,
    compute_instruction_validation_success,
    compute_instruction_reward,
    init_reward_state,
)


class RewardDistanceTests(unittest.TestCase):
    def _spec(self, instruction_type: str = "move_left") -> InstructionSpec:
        direction = {
            "move_left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "move_right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "move_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "move_center": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }[instruction_type]
        text = instruction_type.replace("_", " ")
        return InstructionSpec(
            instruction_type=instruction_type,
            text=text,
            target_object="",
            direction=direction,
            target_displacement=0.40,
            lift_target=0.10,
        )

    def test_closer_goal_has_higher_reward(self):
        spec = self._spec("move_left")
        goal = np.array([0.0, 0.0, 0.20], dtype=np.float32)

        reward_far, success_far, info_far = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.20, 0.0, 0.20], dtype=np.float32),
            obj_pos=goal,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.20, 0.0, 0.20], dtype=np.float32),
                initial_obj_pos=goal,
            ),
            camera_alignment=1.0,
            goal_direction=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        )
        reward_near, success_near, info_near = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.05, 0.0, 0.20], dtype=np.float32),
            obj_pos=goal,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.05, 0.0, 0.20], dtype=np.float32),
                initial_obj_pos=goal,
            ),
            camera_alignment=1.0,
            goal_direction=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        )

        self.assertFalse(success_far)
        self.assertFalse(success_near)
        self.assertGreater(reward_near, reward_far)
        self.assertGreater(info_far["distance_to_goal"], info_near["distance_to_goal"])

    def test_camera_alignment_no_longer_drives_reward_or_success(self):
        spec = self._spec("move_up")
        goal = np.array([0.0, 0.0, 0.23], dtype=np.float32)
        ee = np.array([0.0, 0.0, 0.205], dtype=np.float32)

        reward_bad, success_bad, info_bad = compute_instruction_reward(
            spec=spec,
            ee_pos=ee,
            obj_pos=goal,
            reward_state=init_reward_state(initial_ee_pos=ee, initial_obj_pos=goal),
            camera_alignment=0.15,
            goal_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        reward_good, success_good, info_good = compute_instruction_reward(
            spec=spec,
            ee_pos=ee,
            obj_pos=goal,
            reward_state=init_reward_state(initial_ee_pos=ee, initial_obj_pos=goal),
            camera_alignment=0.95,
            goal_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

        self.assertTrue(success_bad)
        self.assertTrue(success_good)
        self.assertAlmostEqual(reward_good, reward_bad, places=6)
        self.assertEqual(info_bad["camera_reward"], 0.0)
        self.assertEqual(info_good["camera_reward"], 0.0)

    def test_near_saturated_actions_receive_penalty(self):
        spec = self._spec("move_left")
        goal = np.array([0.0, 0.0, 0.20], dtype=np.float32)
        ee = np.array([0.05, 0.0, 0.20], dtype=np.float32)
        state = init_reward_state(initial_ee_pos=ee, initial_obj_pos=goal)

        reward_soft, success_soft, info_soft = compute_instruction_reward(
            spec=spec,
            ee_pos=ee,
            obj_pos=goal,
            reward_state=state,
            action=np.array([0.25, -0.10, 0.15, 0.0, -1.0], dtype=np.float32),
            camera_alignment=0.4,
            goal_direction=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        )
        reward_saturated, success_saturated, info_saturated = compute_instruction_reward(
            spec=spec,
            ee_pos=ee,
            obj_pos=goal,
            reward_state=init_reward_state(initial_ee_pos=ee, initial_obj_pos=goal),
            action=np.array([0.99, -0.99, 0.98, 0.97, -1.0], dtype=np.float32),
            camera_alignment=0.4,
            goal_direction=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        )

        self.assertFalse(success_soft)
        self.assertFalse(success_saturated)
        self.assertEqual(info_soft["action_saturation_penalty"], 0.0)
        self.assertGreater(info_saturated["action_saturation_penalty"], 0.0)
        self.assertGreater(info_saturated["action_saturation_rate"], 0.0)
        self.assertLess(reward_saturated, reward_soft)

    def test_directional_validation_success_uses_signed_displacement(self):
        spec = self._spec("move_right")
        initial = np.array([0.10, 0.00, 0.20], dtype=np.float32)

        success, info = compute_instruction_validation_success(
            spec=spec,
            ee_pos=np.array([0.31, 0.00, 0.20], dtype=np.float32),
            reward_state=init_reward_state(initial_ee_pos=initial, initial_obj_pos=initial),
            task_metadata={"directional_success_displacement_threshold": 0.20},
            current_success=False,
        )

        self.assertTrue(success)
        self.assertEqual(info["validation_success_mode"], 1.0)
        self.assertAlmostEqual(info["directional_success_signed_displacement"], 0.21, places=6)
        self.assertAlmostEqual(info["directional_success_threshold"], 0.20, places=6)

    def test_directional_validation_success_requires_threshold(self):
        spec = self._spec("move_right")
        initial = np.array([0.10, 0.00, 0.20], dtype=np.float32)

        success, info = compute_instruction_validation_success(
            spec=spec,
            ee_pos=np.array([0.29, 0.00, 0.20], dtype=np.float32),
            reward_state=init_reward_state(initial_ee_pos=initial, initial_obj_pos=initial),
            task_metadata={"directional_success_displacement_threshold": 0.20},
            current_success=True,
        )

        self.assertFalse(success)
        self.assertEqual(info["validation_success_mode"], 1.0)
        self.assertAlmostEqual(info["directional_success_signed_displacement"], 0.19, places=6)

    def test_directional_validation_success_handles_negative_axis_motion(self):
        spec = self._spec("move_left")
        initial = np.array([0.10, 0.00, 0.20], dtype=np.float32)

        success, info = compute_instruction_validation_success(
            spec=spec,
            ee_pos=np.array([-0.11, 0.00, 0.20], dtype=np.float32),
            reward_state=init_reward_state(initial_ee_pos=initial, initial_obj_pos=initial),
            task_metadata={"directional_success_displacement_threshold": 0.20},
            current_success=False,
        )

        self.assertTrue(success)
        self.assertEqual(info["directional_success_sign"], -1.0)
        self.assertAlmostEqual(info["directional_success_raw_displacement"], -0.21, places=6)
        self.assertAlmostEqual(info["directional_success_signed_displacement"], 0.21, places=6)

    def test_center_validation_success_falls_back_to_point_success(self):
        spec = self._spec("move_center")
        initial = np.array([0.10, 0.00, 0.20], dtype=np.float32)

        success, info = compute_instruction_validation_success(
            spec=spec,
            ee_pos=np.array([0.50, 0.25, 0.55], dtype=np.float32),
            reward_state=init_reward_state(initial_ee_pos=initial, initial_obj_pos=initial),
            task_metadata={"directional_success_displacement_threshold": 0.20},
            current_success=True,
        )

        self.assertTrue(success)
        self.assertEqual(info["validation_success_mode"], 0.0)


if __name__ == "__main__":
    unittest.main()
