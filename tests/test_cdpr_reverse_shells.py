from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import (
    SMOLVLA_COMPLEX_PROFILE,
    apply_cdpr_reverse_shell,
    get_cdpr_reverse_shell_specs,
)
from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
    InstructionSpec,
    RewardState,
    compute_instruction_reward,
)
from robots.cdpr.cdpr_dataset.rl_cdpr_env import (
    CDPRLanguageRLEnv,
    _apply_reverse_frontier_policy_gate,
)


class _FakeEnv:
    def __init__(self):
        self._instruction_spec = InstructionSpec(
            instruction_type="put_into_plate",
            text="put apple into plate",
            target_object="ycb_apple",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.0,
            lift_target=0.0,
            reference_object="plate",
        )
        self._target_body_name = "apple_body"
        self._target_catalog_name = "ycb_apple"
        self._reference_body_name = "plate_body"
        self._reference_catalog_name = "plate"
        self._second_reference_body_name = ""
        self._second_reference_catalog_name = ""
        self._support_surface_z = 0.0
        self._ee_min_z = 0.0
        self._episode_ee_start = np.array([0.0, 0.0, 0.20], dtype=np.float32)
        self._locked_target_xyz = self._episode_ee_start.copy()
        self.action_step_xyz = 0.02
        self.hold_steps = 6
        self._task_metadata = {
            "reward_mode": "sparse_binary",
            "sparse_success_reward": 1.0,
            "sparse_failure_reward": 0.0,
            "put_container_xy_tolerance": 0.08,
            "put_container_z_tolerance": 0.10,
            "put_min_target_motion": 0.04,
            "put_require_target_grasp": False,
            "put_require_release": True,
            "put_release_opening_threshold": 0.55,
            "caught_object_start_object_offset": [0.0, 0.0, 0.005],
            "caught_object_start_grip_compression": 0.001,
        }
        self._bodies = {
            "apple_body": np.array([-0.10, -0.05, 0.03], dtype=np.float32),
            "plate_body": np.array([0.12, 0.06, 0.04], dtype=np.float32),
            "pear_body": np.array([0.20, -0.12, 0.03], dtype=np.float32),
        }
        self._ee = np.array([0.0, 0.0, 0.20], dtype=np.float32)
        self._hold_center_rel = np.array([0.0, 0.0, -0.07], dtype=np.float32)
        self._gripper_opening = 1.0
        self._caught_object_start_active = False
        self._caught_object_start_body = ""
        self._caught_object_start_catalog = ""
        self._caught_object_start_position = np.zeros((3,), dtype=np.float32)
        self._caught_object_start_ee_offset = np.zeros((3,), dtype=np.float32)
        self._caught_object_start_hold_offset = np.zeros((3,), dtype=np.float32)
        self._caught_object_start_gripper_opening = 0.0
        self.sim = object()

    def _get_body_position(self, body_name):
        return self._bodies[str(body_name)].copy()

    def _set_body_position(self, body_name, xyz):
        self._bodies[str(body_name)] = np.asarray(xyz, dtype=np.float32).reshape(3).copy()
        return True

    def _get_ee_position(self):
        return self._ee.copy()

    def _set_ee_target(self, xyz):
        self._ee = np.asarray(xyz, dtype=np.float32).reshape(3).copy()

    def _force_gripper_opening(self, opening):
        self._gripper_opening = float(opening)

    def _get_gripper_opening(self):
        return float(self._gripper_opening)

    def _caught_object_start_gripper_opening_for_body(self, body_name):
        del body_name
        return 0.20

    def _caught_object_start_hold_center(self):
        return (self._ee + self._hold_center_rel).astype(np.float32)

    def _reference_object_position(self, second=False):
        del second
        return self._bodies["plate_body"].copy()


class CDPRReverseShellTests(unittest.TestCase):
    def test_grasp_shell_latches_only_after_aligned_policy_close(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "reverse_frontier_dynamic_grasp_latch": True,
            "reverse_frontier_grasp_latch_xy_distance": 0.030,
            "reverse_frontier_grasp_latch_z_distance": 0.060,
            "reverse_frontier_grasp_latch_min_finger_contacts": 1,
            "reverse_frontier_grasp_latch_contactless_xy_distance": 0.028,
            "reverse_frontier_grasp_latch_contactless_z_distance": 0.030,
            "reverse_frontier_grasp_latch_opening_margin": 0.012,
            "reverse_frontier_grasp_latch_closing_command_threshold": 0.05,
            "caught_object_start_object_offset": [0.0, 0.0, 0.005],
        }
        env._curriculum_mode = "reverse_frontier"
        env._curriculum_reset_info = {"curriculum_grasp_required": True}
        env._target_body_name = "target_body"
        env._target_catalog_name = "ycb_apple"
        env._caught_object_start_active = False
        env._reverse_frontier_dynamic_grasp_latched = False
        env._last_gripper_cmd = 0.0
        env._get_gripper_target = lambda: 0.85
        env._caught_object_start_gripper_opening_for_body = lambda body: 0.85
        env._caught_object_start_hold_center = lambda: np.array(
            [0.10, -0.05, 0.07], dtype=np.float32
        )
        env._get_body_position = lambda body: np.array(
            [0.10, -0.05, 0.075], dtype=np.float32
        )
        env._maintain_caught_object_start_pose = lambda: True
        env._reverse_frontier_target_finger_contact_count = lambda body: 1
        ee = np.array([0.10, -0.05, 0.14], dtype=np.float32)

        self.assertFalse(env._maybe_latch_reverse_frontier_grasp(ee))
        self.assertFalse(env._caught_object_start_active)

        env._last_gripper_cmd = -1.0
        self.assertTrue(env._maybe_latch_reverse_frontier_grasp(ee))
        self.assertTrue(env._caught_object_start_active)
        self.assertTrue(env._reverse_frontier_dynamic_grasp_latched)
        self.assertEqual(env._caught_object_start_body, "target_body")
        np.testing.assert_allclose(
            env._caught_object_start_hold_offset,
            np.array([0.0, 0.0, 0.005], dtype=np.float32),
            atol=1e-7,
        )

    def test_policy_gate_blocks_early_sparse_success_until_sampled_target(self):
        reset_info = {"curriculum_shell_target_policy_steps": 3}
        metadata = {"reward_mode": "sparse_binary", "sparse_failure_reward": 0.0}

        early_success, early_reward, early_info = _apply_reverse_frontier_policy_gate(
            success=True,
            reward=1.0,
            reward_info={"sparse_success": 1.0, "success_bonus": 1.0},
            curriculum_mode="reverse_frontier",
            curriculum_reset_info=reset_info,
            policy_step=2,
            task_metadata=metadata,
        )
        self.assertFalse(early_success)
        self.assertEqual(early_reward, 0.0)
        self.assertEqual(early_info["sparse_success"], 0.0)
        self.assertEqual(early_info["success_bonus"], 0.0)
        self.assertEqual(early_info["reverse_frontier_raw_success"], 1.0)
        self.assertEqual(early_info["reverse_frontier_policy_steps_remaining"], 1)

        final_success, final_reward, final_info = _apply_reverse_frontier_policy_gate(
            success=True,
            reward=1.0,
            reward_info={"sparse_success": 1.0, "success_bonus": 1.0},
            curriculum_mode="reverse_frontier",
            curriculum_reset_info=reset_info,
            policy_step=3,
            task_metadata=metadata,
        )
        self.assertTrue(final_success)
        self.assertEqual(final_reward, 1.0)
        self.assertEqual(final_info["sparse_success"], 1.0)
        self.assertEqual(final_info["reverse_frontier_policy_gate_satisfied"], 1.0)

    def test_policy_gate_uses_environment_actions_for_chunked_policy(self):
        reset_info = {
            "curriculum_shell_target_policy_steps": 2,
            "curriculum_shell_target_action_steps": 6,
            "curriculum_actions_per_policy_decision": 4,
        }
        metadata = {"reward_mode": "sparse_binary", "sparse_failure_reward": 0.0}

        early_success, early_reward, early_info = _apply_reverse_frontier_policy_gate(
            success=True,
            reward=1.0,
            reward_info={"sparse_success": 1.0},
            curriculum_mode="reverse_frontier",
            curriculum_reset_info=reset_info,
            policy_step=5,
            task_metadata=metadata,
        )
        self.assertFalse(early_success)
        self.assertEqual(early_reward, 0.0)
        self.assertEqual(early_info["reverse_frontier_action_steps_remaining"], 1)
        self.assertEqual(early_info["reverse_frontier_policy_steps_remaining"], 1)

        final_success, final_reward, final_info = _apply_reverse_frontier_policy_gate(
            success=True,
            reward=1.0,
            reward_info={"sparse_success": 1.0},
            curriculum_mode="reverse_frontier",
            curriculum_reset_info=reset_info,
            policy_step=6,
            task_metadata=metadata,
        )
        self.assertTrue(final_success)
        self.assertEqual(final_reward, 1.0)
        self.assertEqual(final_info["reverse_frontier_action_steps_remaining"], 0)

    def test_policy_gate_can_be_disabled_for_exact_success_termination(self):
        success, reward, info = _apply_reverse_frontier_policy_gate(
            success=True,
            reward=1.0,
            reward_info={"sparse_success": 1.0},
            curriculum_mode="reverse_frontier",
            curriculum_reset_info={"curriculum_shell_target_action_steps": 6},
            policy_step=1,
            task_metadata={
                "reward_mode": "sparse_binary",
                "reverse_frontier_min_action_gate_enabled": False,
            },
        )
        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["reverse_frontier_policy_gate_enabled"], 0.0)
        self.assertEqual(info["reverse_frontier_policy_gate_active"], 0.0)

    def test_put_plate_shell_zero_places_held_object_near_plate(self):
        env = _FakeEnv()
        info = apply_cdpr_reverse_shell(env, shell_id=0, rng=np.random.default_rng(4))

        apple = env._get_body_position("apple_body")
        plate = env._get_body_position("plate_body")

        self.assertEqual(info["curriculum_shell"], 0)
        self.assertTrue(info["curriculum_target_grasped"])
        self.assertTrue(env._caught_object_start_active)
        self.assertLessEqual(float(np.linalg.norm(apple[:2] - plate[:2])), 0.020)
        self.assertGreaterEqual(float(apple[2] - plate[2]), 0.010)
        self.assertLessEqual(float(apple[2] - plate[2]), 0.020)
        self.assertAlmostEqual(info["curriculum_held_gripper_opening"], 0.8519, places=4)
        np.testing.assert_allclose(
            env._caught_object_start_hold_offset,
            np.array([0.0, 0.0, 0.005], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            apple,
            env._caught_object_start_hold_center() + env._caught_object_start_hold_offset,
            atol=1e-6,
        )

    def test_opening_gripper_can_satisfy_put_shell_zero_sparse_success(self):
        env = _FakeEnv()
        info = apply_cdpr_reverse_shell(env, shell_id=0, rng=np.random.default_rng(5))
        env._force_gripper_opening(1.0)
        apple = env._get_body_position("apple_body")
        reward_state = RewardState(
            initial_ee_pos=env._get_ee_position(),
            initial_obj_pos=np.asarray(info["curriculum_reward_initial_obj_pos"], dtype=np.float32),
            prev_ee_pos=env._get_ee_position(),
            prev_obj_pos=apple.copy(),
            gripper_closed=False,
            grasped=True,
        )

        reward, success, reward_info = compute_instruction_reward(
            spec=env._instruction_spec,
            ee_pos=env._get_ee_position(),
            obj_pos=env._get_body_position("plate_body"),
            reward_state=reward_state,
            task_metadata=env._task_metadata,
            env=env,
            target_body_name="apple_body",
            reference_body_name="plate_body",
            gripper_opening=env._get_gripper_opening(),
        )

        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertEqual(reward_info["sparse_success"], 1.0)

    def test_shell_reset_controls_target_relation_but_leaves_distractor_randomized(self):
        env = _FakeEnv()
        pear_before = env._get_body_position("pear_body").copy()

        apply_cdpr_reverse_shell(env, shell_id=2, rng=np.random.default_rng(6))

        np.testing.assert_allclose(env._get_body_position("pear_body"), pear_before)
        self.assertLess(
            float(
                np.linalg.norm(
                    env._get_body_position("apple_body")[:2] - pear_before[:2]
                )
            ),
            0.50,
        )

    def test_complex_shell_four_is_the_shifted_farther_held_continuation(self):
        env = _FakeEnv()
        info = apply_cdpr_reverse_shell(
            env,
            shell_id=4,
            rng=np.random.default_rng(17),
            profile=SMOLVLA_COMPLEX_PROFILE,
        )

        self.assertEqual(
            (info["curriculum_shell_policy_steps_low"], info["curriculum_shell_policy_steps_high"]),
            (5, 6),
        )
        self.assertGreaterEqual(info["curriculum_shell_target_policy_steps"], 5)
        self.assertLessEqual(info["curriculum_shell_target_policy_steps"], 6)
        self.assertTrue(info["curriculum_target_grasped"])
        self.assertTrue(env._caught_object_start_active)
        self.assertEqual(info["curriculum_put_start_stage"], 0)
        self.assertEqual(info["curriculum_shell_relation"], "held_object_near_receptacle")
        self.assertGreater(info["curriculum_shell_distance_m"], 0.08)

    def test_complex_plate_adds_aligned_approach_and_full_grasp_shells(self):
        expected = {
            5: ("aligned_open_gripper_catch_then_finish", (7, 13), 0),
            6: ("near_object_approach_catch_then_finish", (14, 20), 1),
            7: ("full_randomized_approach_catch_then_finish", (21, 32), 2),
        }
        for shell_id, (relation, bounds, phase) in expected.items():
            with self.subTest(shell_id=shell_id):
                env = _FakeEnv()
                ee_before = env._get_ee_position()
                apple_before = env._get_body_position("apple_body")
                info = apply_cdpr_reverse_shell(
                    env,
                    shell_id=shell_id,
                    rng=np.random.default_rng(200 + shell_id),
                    profile=SMOLVLA_COMPLEX_PROFILE,
                )

                self.assertEqual(info["curriculum_shell_count"], 8)
                self.assertEqual(info["curriculum_shell_relation"], relation)
                self.assertEqual(
                    (
                        info["curriculum_shell_policy_steps_low"],
                        info["curriculum_shell_policy_steps_high"],
                    ),
                    bounds,
                )
                self.assertEqual(info["curriculum_grasp_start_phase"], phase)
                self.assertTrue(info["curriculum_grasp_required"])
                self.assertFalse(info["curriculum_target_grasped"])
                self.assertFalse(env._caught_object_start_active)
                self.assertEqual(env._get_gripper_opening(), 1.0)
                if shell_id == 7:
                    np.testing.assert_allclose(env._get_ee_position(), ee_before)
                    np.testing.assert_allclose(
                        env._get_body_position("apple_body"), apple_before
                    )
                    self.assertTrue(
                        np.isnan(info["curriculum_grasp_start_goal_distance_m"])
                    )
                    self.assertTrue(info["curriculum_shell_normal_reset"])
                else:
                    start_distance = info["curriculum_grasp_start_goal_distance_m"]
                    expected_range = (0.120, 0.145) if shell_id == 5 else (0.160, 0.200)
                    self.assertGreaterEqual(start_distance, expected_range[0])
                    self.assertLessEqual(start_distance, expected_range[1])
                    plate = env._get_body_position("plate_body")
                    apple = env._get_body_position("apple_body")
                    self.assertAlmostEqual(
                        float(np.linalg.norm(apple[:2] - plate[:2])),
                        float(start_distance),
                        places=5,
                    )
                    self.assertFalse(info["curriculum_shell_normal_reset"])

    def test_complex_motion_credit_stays_valid_while_approaching_goal(self):
        env = _FakeEnv()
        info = apply_cdpr_reverse_shell(
            env,
            shell_id=0,
            rng=np.random.default_rng(19),
            profile=SMOLVLA_COMPLEX_PROFILE,
        )
        apple = env._get_body_position("apple_body")
        plate = env._get_body_position("plate_body")
        toward_goal = plate[:2] - apple[:2]
        toward_goal /= max(float(np.linalg.norm(toward_goal)), 1e-8)
        moved = apple.copy()
        moved[:2] += toward_goal * 0.01
        env._set_body_position("apple_body", moved)
        env._force_gripper_opening(1.0)
        reward_state = RewardState(
            initial_ee_pos=env._get_ee_position(),
            initial_obj_pos=np.asarray(
                info["curriculum_reward_initial_obj_pos"], dtype=np.float32
            ),
            prev_ee_pos=env._get_ee_position(),
            prev_obj_pos=apple,
            gripper_closed=False,
            grasped=True,
        )

        _, _, reward_info = compute_instruction_reward(
            spec=env._instruction_spec,
            ee_pos=env._get_ee_position(),
            obj_pos=plate,
            reward_state=reward_state,
            task_metadata=env._task_metadata,
            env=env,
            target_body_name="apple_body",
            reference_body_name="plate_body",
            gripper_opening=env._get_gripper_opening(),
        )

        self.assertEqual(reward_info["relation_motion_ok"], 1.0)
        self.assertGreaterEqual(reward_info["target_motion_xy"], 0.04)

    def test_final_shell_is_normal_randomized_reset(self):
        env = _FakeEnv()
        apple_before = env._get_body_position("apple_body").copy()

        info = apply_cdpr_reverse_shell(env, shell_id=5, rng=np.random.default_rng(7))

        self.assertTrue(info["curriculum_shell_normal_reset"])
        np.testing.assert_allclose(env._get_body_position("apple_body"), apple_before)

    def test_catalog_exposes_complex_instruction_specs(self):
        specs = {spec.instruction_id: spec.shell_count for spec in get_cdpr_reverse_shell_specs()}

        self.assertEqual(specs["move_to_object"], 4)
        self.assertEqual(specs["grab_object"], 5)
        self.assertEqual(specs["put_into_plate"], 6)
        self.assertEqual(specs["push_left"], 5)
        self.assertEqual(specs["push_forward"], 5)
        self.assertEqual(specs["move_in_front_of_object"], 5)
        self.assertEqual(specs["move_between_objects"], 5)

    def test_move_to_object_shell_distances_progress_without_large_gap(self):
        env = _FakeEnv()
        env._instruction_spec = InstructionSpec(
            instruction_type="move_to_object",
            text="move to apple",
            target_object="ycb_apple",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.0,
            lift_target=0.0,
        )
        expected_bounds = ((0.010, 0.020), (0.020, 0.050), (0.050, 0.100))

        for shell_id, (lower, upper) in enumerate(expected_bounds):
            with self.subTest(shell_id=shell_id):
                info = apply_cdpr_reverse_shell(
                    env,
                    shell_id=shell_id,
                    rng=np.random.default_rng(100 + shell_id),
                )
                distance = float(info["curriculum_shell_distance_m"])
                self.assertGreaterEqual(distance, lower)
                self.assertLessEqual(distance, upper)


if __name__ == "__main__":
    unittest.main()
