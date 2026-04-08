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
            "move_to_object": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "pick_up": np.array([0.0, 0.0, 0.0], dtype=np.float32),
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

    def test_move_to_object_reward_prefers_smaller_xy_distance(self):
        spec = self._spec("move_to_object")
        target = np.array([0.10, -0.10, 0.18], dtype=np.float32)

        reward_far, success_far, info_far = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.25, 0.05, 0.50], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.25, 0.05, 0.50], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata={"move_to_object_xy_tolerance": 0.02},
        )
        reward_near, success_near, info_near = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.11, -0.09, 0.50], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.11, -0.09, 0.50], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata={"move_to_object_xy_tolerance": 0.02},
        )

        self.assertFalse(success_far)
        self.assertTrue(success_near)
        self.assertGreater(reward_near, reward_far)
        self.assertGreater(info_far["move_to_object_xy_distance"], info_near["move_to_object_xy_distance"])
        self.assertGreater(info_near["move_to_object_above_bonus"], 0.0)

    def test_move_to_object_success_uses_xy_tolerance_only(self):
        spec = self._spec("move_to_object")
        target = np.array([0.00, 0.00, 0.16], dtype=np.float32)

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.015, -0.010, 0.48], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.06, 0.06, 0.48], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata={"move_to_object_xy_tolerance": 0.02},
        )

        self.assertTrue(success)
        self.assertGreater(reward, 1.0)
        self.assertAlmostEqual(info["distance_to_goal"], np.linalg.norm([0.015, -0.010]), places=6)
        self.assertAlmostEqual(info["distance_ee_to_object_xyz"], np.linalg.norm([0.015, -0.010, -0.32]), places=6)

    def test_move_to_object_validation_success_uses_reward_result(self):
        spec = self._spec("move_to_object")
        initial = np.array([0.10, 0.00, 0.20], dtype=np.float32)

        success, info = compute_instruction_validation_success(
            spec=spec,
            ee_pos=np.array([0.12, 0.01, 0.45], dtype=np.float32),
            reward_state=init_reward_state(initial_ee_pos=initial, initial_obj_pos=initial),
            task_metadata={"move_to_object_xy_tolerance": 0.02},
            current_success=True,
        )

        self.assertTrue(success)
        self.assertEqual(info["validation_success_mode"], 2.0)
        self.assertEqual(info["move_to_object_validation_success"], 1.0)

    def test_pick_up_reward_prefers_open_centered_pregrasp(self):
        spec = self._spec("pick_up")
        initial_obj = np.array([0.0, 0.0, 0.18], dtype=np.float32)

        reward_far, success_far, info_far = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.18, 0.18, 0.40], dtype=np.float32),
            obj_pos=initial_obj,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.18, 0.18, 0.40], dtype=np.float32),
                initial_obj_pos=initial_obj,
            ),
            gripper_opening=0.03,
            support_surface_z=0.15,
        )
        reward_ready, success_ready, info_ready = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.01, 0.00, 0.26], dtype=np.float32),
            obj_pos=initial_obj,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.01, 0.00, 0.26], dtype=np.float32),
                initial_obj_pos=initial_obj,
            ),
            gripper_opening=0.03,
            support_surface_z=0.15,
        )

        self.assertFalse(success_far)
        self.assertFalse(success_ready)
        self.assertGreater(reward_ready, reward_far)
        self.assertGreater(info_ready["pick_open_reward"], 0.0)

    def test_pick_up_reward_succeeds_when_target_is_caught_and_lifted(self):
        spec = self._spec("pick_up")
        initial_obj = np.array([0.0, 0.0, 0.16], dtype=np.float32)
        state = init_reward_state(
            initial_ee_pos=np.array([0.0, 0.0, 0.28], dtype=np.float32),
            initial_obj_pos=initial_obj,
        )

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.0, 0.0, 0.30], dtype=np.float32),
            obj_pos=np.array([0.0, 0.0, 0.27], dtype=np.float32),
            reward_state=state,
            gripper_opening=0.0,
            support_surface_z=0.15,
            caught_object_is_target=True,
            caught_object_score=0.95,
        )

        self.assertTrue(success)
        self.assertGreater(reward, 2.0)
        self.assertEqual(info["grasped"], 1.0)
        self.assertGreaterEqual(info["pick_target_lift"], 0.10)

    def test_pick_up_reward_penalizes_wrong_object_grasp(self):
        spec = self._spec("pick_up")
        initial_obj = np.array([0.0, 0.0, 0.16], dtype=np.float32)
        target_state = init_reward_state(
            initial_ee_pos=np.array([0.0, 0.0, 0.25], dtype=np.float32),
            initial_obj_pos=initial_obj,
        )
        wrong_state = init_reward_state(
            initial_ee_pos=np.array([0.0, 0.0, 0.25], dtype=np.float32),
            initial_obj_pos=initial_obj,
        )

        reward_target, _, info_target = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.0, 0.0, 0.27], dtype=np.float32),
            obj_pos=np.array([0.0, 0.0, 0.24], dtype=np.float32),
            reward_state=target_state,
            gripper_opening=0.0,
            support_surface_z=0.15,
            caught_object_is_target=True,
            caught_object_score=0.9,
        )
        reward_wrong, _, info_wrong = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.0, 0.0, 0.27], dtype=np.float32),
            obj_pos=np.array([0.0, 0.0, 0.24], dtype=np.float32),
            reward_state=wrong_state,
            gripper_opening=0.0,
            support_surface_z=0.15,
            caught_object_is_target=False,
            caught_object_score=0.9,
            caught_object_catalog="ycb_pear",
        )

        self.assertGreater(info_wrong["pick_wrong_object_penalty"], 0.0)
        self.assertGreater(reward_target, reward_wrong)


if __name__ == "__main__":
    unittest.main()
