from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import (
    apply_cdpr_reverse_shell,
    get_cdpr_reverse_shell_specs,
)
from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
    InstructionSpec,
    RewardState,
    compute_instruction_reward,
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
        self.assertLess(float(np.linalg.norm(env._get_body_position("apple_body")[:2] - pear_before[:2])), 0.50)

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
        self.assertEqual(specs["move_between_objects"], 5)


if __name__ == "__main__":
    unittest.main()
