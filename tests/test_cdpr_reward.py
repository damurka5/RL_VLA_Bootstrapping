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
            ee_pos=np.array([0.25, 0.05, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.25, 0.05, 0.15], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata={"move_to_object_xy_tolerance": 0.02},
        )
        reward_near, success_near, info_near = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.11, -0.09, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.11, -0.09, 0.15], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata={"move_to_object_xy_tolerance": 0.02},
        )

        self.assertFalse(success_far)
        self.assertTrue(success_near)
        self.assertGreater(reward_near, reward_far)
        self.assertGreater(info_far["move_to_object_xy_distance"], info_near["move_to_object_xy_distance"])
        self.assertAlmostEqual(info_near["move_to_object_above_bonus"], 0.0, places=6)
        self.assertGreater(info_near["move_to_object_distance_reward"], info_far["move_to_object_distance_reward"])

    def test_move_to_object_success_requires_xy_tolerance_and_z_window(self):
        spec = self._spec("move_to_object")
        target = np.array([0.00, 0.00, 0.16], dtype=np.float32)

        reward_good, success_good, info_good = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.015, -0.010, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.06, 0.06, 0.15], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata={"move_to_object_xy_tolerance": 0.02},
        )
        reward_high, success_high, info_high = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.015, -0.010, 0.28], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.06, 0.06, 0.28], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata={"move_to_object_xy_tolerance": 0.02},
        )

        self.assertTrue(success_good)
        self.assertFalse(success_high)
        self.assertGreater(reward_good, reward_high)
        self.assertAlmostEqual(info_good["distance_to_goal"], np.linalg.norm([0.015, -0.010]), places=6)
        self.assertAlmostEqual(info_high["distance_ee_to_object_xyz"], np.linalg.norm([0.015, -0.010, -0.12]), places=6)
        self.assertEqual(info_good["move_to_object_z_in_window"], 1.0)
        self.assertGreater(info_high["move_to_object_z_penalty"], 0.0)

    def test_move_to_object_distance_reward_has_configured_maximum(self):
        spec = self._spec("move_to_object")
        target = np.array([0.10, -0.10, 0.16], dtype=np.float32)

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.10, -0.10, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.10, -0.10, 0.15], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata={"move_to_object_distance_reward_weight": 1.0},
        )

        self.assertTrue(success)
        self.assertAlmostEqual(reward, 1.0, places=6)
        self.assertAlmostEqual(info["move_to_object_distance_reward"], 1.0, places=6)
        self.assertAlmostEqual(info["move_to_object_distance_reward_max"], 1.0, places=6)

    def test_sparse_binary_mode_replaces_move_to_object_shaping(self):
        spec = self._spec("move_to_object")
        target = np.array([0.10, -0.10, 0.16], dtype=np.float32)
        metadata = {
            "reward_mode": "sparse_binary",
            "move_to_object_validation_distance_threshold": 0.03,
        }

        reward_far, success_far, info_far = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.20, -0.10, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.20, -0.10, 0.15], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata=metadata,
        )
        reward_near, success_near, info_near = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.11, -0.10, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.20, -0.10, 0.15], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata=metadata,
        )

        self.assertFalse(success_far)
        self.assertEqual(reward_far, 0.0)
        self.assertTrue(success_near)
        self.assertEqual(reward_near, 1.0)
        self.assertEqual(info_near["sparse_binary_reward"], 1.0)
        self.assertEqual(info_near["distance_reward"], 0.0)

    def test_move_to_object_z_window_penalty_applies_only_outside_band(self):
        spec = self._spec("move_to_object")
        target = np.array([0.00, 0.00, 0.16], dtype=np.float32)

        reward_in_band, _, info_in_band = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.04, 0.00, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.06, 0.00, 0.15], dtype=np.float32),
                initial_obj_pos=target,
            ),
        )
        reward_above_band, _, info_above_band = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.04, 0.00, 0.27], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.06, 0.00, 0.27], dtype=np.float32),
                initial_obj_pos=target,
            ),
        )

        self.assertEqual(info_in_band["move_to_object_z_penalty"], 0.0)
        self.assertGreater(info_above_band["move_to_object_z_penalty"], 0.0)
        self.assertLess(reward_above_band, reward_in_band)

    def test_move_to_object_reward_can_disable_z_penalty_for_xy_only_training(self):
        spec = self._spec("move_to_object")
        target = np.array([0.00, 0.00, 0.16], dtype=np.float32)
        metadata = {
            "move_to_object_xy_tolerance": 0.02,
            "move_to_object_z_penalty_weight": 0.0,
        }

        reward_in_band, _, info_in_band = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.04, 0.00, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.06, 0.00, 0.15], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata=metadata,
        )
        reward_above_band, _, info_above_band = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.04, 0.00, 0.27], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.06, 0.00, 0.27], dtype=np.float32),
                initial_obj_pos=target,
            ),
            task_metadata=metadata,
        )

        self.assertGreater(info_above_band["move_to_object_z_penalty_raw"], 0.0)
        self.assertEqual(info_above_band["move_to_object_z_penalty"], 0.0)
        self.assertAlmostEqual(reward_above_band, reward_in_band, places=6)

    def test_move_to_object_saturation_penalty_is_linear_and_can_include_gripper(self):
        spec = self._spec("move_to_object")
        target = np.array([0.00, 0.00, 0.16], dtype=np.float32)
        metadata = {
            "action_saturation_threshold": 0.70,
            "action_saturation_penalty_weight": 0.20,
            "action_saturation_exponent": 1.0,
            "action_saturation_include_gripper": True,
        }

        reward_threshold, _, info_threshold = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.04, 0.00, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.06, 0.00, 0.15], dtype=np.float32),
                initial_obj_pos=target,
            ),
            action=np.array([0.70, 0.00, 0.00, 0.00, 0.70], dtype=np.float32),
            task_metadata=metadata,
        )
        reward_high, _, info_high = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.04, 0.00, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.06, 0.00, 0.15], dtype=np.float32),
                initial_obj_pos=target,
            ),
            action=np.array([0.90, 0.00, 0.00, 0.00, 0.90], dtype=np.float32),
            task_metadata=metadata,
        )

        self.assertEqual(info_threshold["action_saturation_penalty"], 0.0)
        self.assertGreater(info_high["action_saturation_penalty"], 0.0)
        self.assertGreater(info_high["action_saturation_penalty_raw"], info_threshold["action_saturation_penalty_raw"])
        self.assertLess(reward_high, reward_threshold)

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

    def test_grab_object_sparse_reward_succeeds_when_target_is_caught(self):
        spec = InstructionSpec(
            instruction_type="grab_object",
            text="grab apple",
            target_object="ycb_apple",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
        )
        target = np.array([0.02, 0.00, 0.16], dtype=np.float32)

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.02, 0.00, 0.18], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.02, 0.00, 0.20], dtype=np.float32),
                initial_obj_pos=target,
            ),
            gripper_opening=0.0,
            caught_object_is_target=True,
            caught_object_score=0.9,
        )

        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["sparse_success"], 1.0)
        self.assertEqual(info["grasped"], 1.0)

    def test_grab_object_sparse_reward_requires_caught_target_by_default(self):
        spec = InstructionSpec(
            instruction_type="grab_object",
            text="grab apple",
            target_object="ycb_apple",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
        )
        target = np.array([0.02, 0.00, 0.16], dtype=np.float32)

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.021, 0.00, 0.18], dtype=np.float32),
            obj_pos=target,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.02, 0.00, 0.20], dtype=np.float32),
                initial_obj_pos=target,
            ),
            gripper_opening=0.0,
            caught_object_is_target=False,
            caught_object_score=0.0,
        )

        self.assertFalse(success)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["sparse_success"], 0.0)

    def test_push_right_sparse_reward_uses_object_displacement(self):
        spec = InstructionSpec(
            instruction_type="push_right",
            text="push apple right",
            target_object="ycb_apple",
            direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
        )
        initial = np.array([0.00, 0.00, 0.16], dtype=np.float32)
        moved = np.array([0.09, 0.00, 0.16], dtype=np.float32)

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.08, 0.00, 0.18], dtype=np.float32),
            obj_pos=moved,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.0, 0.0, 0.20], dtype=np.float32),
                initial_obj_pos=initial,
            ),
            task_metadata={"push_success_displacement": 0.08},
        )

        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertAlmostEqual(info["signed_relation_offset"], 0.09, places=6)

    def test_push_forward_sparse_reward_uses_y_displacement(self):
        spec = InstructionSpec(
            instruction_type="push_forward",
            text="push apple forward",
            target_object="ycb_apple",
            direction=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
        )
        initial = np.array([0.00, 0.00, 0.16], dtype=np.float32)
        moved = np.array([0.00, 0.055, 0.16], dtype=np.float32)

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.00, 0.04, 0.18], dtype=np.float32),
            obj_pos=moved,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.0, 0.0, 0.20], dtype=np.float32),
                initial_obj_pos=initial,
            ),
            task_metadata={"push_success_displacement": 0.05},
        )

        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertAlmostEqual(info["signed_relation_offset"], 0.055, places=6)
        self.assertEqual(info["relation_axis"], 1.0)

    def test_left_of_object_sparse_reward_uses_square_success_zone(self):
        class _Env:
            def _get_body_position(self, body_name):
                return {
                    "target_body": np.array([-0.06, 0.01, 0.16], dtype=np.float32),
                    "ref_body": np.array([0.02, 0.00, 0.16], dtype=np.float32),
                }[body_name]

        spec = InstructionSpec(
            instruction_type="move_left_of_object",
            text="move apple to the left of pear",
            target_object="ycb_apple",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
            reference_object="ycb_pear",
        )

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([-0.08, 0.01, 0.18], dtype=np.float32),
            obj_pos=np.array([-0.06, 0.00, 0.16], dtype=np.float32),
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.0, 0.0, 0.20], dtype=np.float32),
                initial_obj_pos=np.array([-0.02, 0.00, 0.16], dtype=np.float32),
            ),
            task_metadata={"relation_left_right_offset": 0.08, "relation_min_target_motion": 0.02},
            env=_Env(),
            target_body_name="target_body",
            reference_body_name="ref_body",
            gripper_opening=0.0,
            caught_object_is_target=True,
        )

        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertAlmostEqual(info["signed_relation_offset"], 0.08, places=6)
        self.assertEqual(info["relation_motion_ok"], 1.0)
        self.assertEqual(info["relation_grasp_ok"], 1.0)
        self.assertAlmostEqual(info["relation_zone_size"], 0.05, places=6)

    def test_front_of_object_sparse_reward_uses_y_relation_and_motion(self):
        class _Env:
            def _get_body_position(self, body_name):
                return {
                    "target_body": np.array([0.01, -0.09, 0.16], dtype=np.float32),
                    "ref_body": np.array([0.00, 0.02, 0.16], dtype=np.float32),
                }[body_name]

        spec = InstructionSpec(
            instruction_type="put_in_front_of_object",
            text="put apple in front of pear",
            target_object="ycb_apple",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
            reference_object="ycb_pear",
        )

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.01, -0.09, 0.18], dtype=np.float32),
            obj_pos=np.array([0.01, -0.09, 0.16], dtype=np.float32),
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.0, 0.0, 0.20], dtype=np.float32),
                initial_obj_pos=np.array([0.01, -0.04, 0.16], dtype=np.float32),
            ),
            task_metadata={"relation_front_behind_offset": 0.08, "relation_min_target_motion": 0.04},
            env=_Env(),
            target_body_name="target_body",
            reference_body_name="ref_body",
            gripper_opening=0.0,
            caught_object_is_target=True,
        )

        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertGreater(info["signed_relation_offset"], 0.08)
        self.assertEqual(info["relation_motion_ok"], 1.0)

    def test_move_front_of_object_uses_square_success_zone(self):
        class _Env:
            def _get_body_position(self, body_name):
                return {
                    "target_body": np.array([0.011, -0.061, 0.16], dtype=np.float32),
                    "ref_body": np.array([0.00, 0.02, 0.16], dtype=np.float32),
                }[body_name]

        spec = InstructionSpec(
            instruction_type="move_in_front_of_object",
            text="move apple in front of pear",
            target_object="ycb_apple",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
            reference_object="ycb_pear",
        )

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.01, -0.06, 0.18], dtype=np.float32),
            obj_pos=np.array([0.01, -0.06, 0.16], dtype=np.float32),
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.0, 0.0, 0.20], dtype=np.float32),
                initial_obj_pos=np.array([0.05, -0.02, 0.16], dtype=np.float32),
            ),
            task_metadata={
                "relation_front_behind_offset": 0.08,
                "move_relation_success_zone_size": 0.05,
            },
            env=_Env(),
            target_body_name="target_body",
            reference_body_name="ref_body",
            gripper_opening=1.0,
            caught_object_is_target=False,
        )

        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertLessEqual(info["relation_axis_error"], 0.025)
        self.assertLessEqual(info["relation_orthogonal_error"], 0.025)
        self.assertEqual(info["relation_grasp_required"], 0.0)

    def test_put_into_plate_can_require_target_grasp_and_motion(self):
        class _Env:
            def _get_body_position(self, body_name):
                return {
                    "target_body": np.array([0.00, 0.00, 0.16], dtype=np.float32),
                    "plate_body": np.array([0.01, 0.00, 0.16], dtype=np.float32),
                }[body_name]

        spec = InstructionSpec(
            instruction_type="put_into_plate",
            text="put apple into plate",
            target_object="ycb_apple",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
            reference_object="plate",
        )
        metadata = {
            "put_min_target_motion": 0.04,
            "put_require_target_grasp": True,
        }
        state = init_reward_state(
            initial_ee_pos=np.array([0.0, 0.0, 0.20], dtype=np.float32),
            initial_obj_pos=np.array([0.06, 0.00, 0.16], dtype=np.float32),
        )

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.0, 0.0, 0.18], dtype=np.float32),
            obj_pos=np.array([0.0, 0.0, 0.16], dtype=np.float32),
            reward_state=state,
            task_metadata=metadata,
            env=_Env(),
            target_body_name="target_body",
            reference_body_name="plate_body",
            gripper_opening=0.0,
            caught_object_is_target=True,
        )

        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["relation_motion_ok"], 1.0)
        self.assertEqual(info["relation_grasp_ok"], 1.0)

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

    def test_sparse_binary_mode_replaces_pick_up_shaping(self):
        spec = self._spec("pick_up")
        initial_obj = np.array([0.0, 0.0, 0.16], dtype=np.float32)

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.0, 0.0, 0.30], dtype=np.float32),
            obj_pos=np.array([0.0, 0.0, 0.24], dtype=np.float32),
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.0, 0.0, 0.28], dtype=np.float32),
                initial_obj_pos=initial_obj,
            ),
            task_metadata={"reward_mode": "sparse_binary", "pick_lift_success_height": 0.05},
            gripper_opening=0.0,
            support_surface_z=0.15,
            caught_object_is_target=True,
            caught_object_score=0.95,
        )

        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["sparse_success"], 1.0)
        self.assertEqual(info["distance_reward"], 0.0)

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
