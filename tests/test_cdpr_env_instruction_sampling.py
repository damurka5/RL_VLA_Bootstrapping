from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv, SceneSpec


class EnvInstructionSamplingTests(unittest.TestCase):
    def test_uniform_cycle_sampling_covers_each_instruction_once_per_cycle(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.instruction_types = ("move_up", "move_down", "move_left")
        env.instruction_sampling = "uniform_cycle"
        env._instruction_cycle = []
        env.np_random = np.random.default_rng(3)

        first_cycle = [env._sample_instruction_type() for _ in range(3)]
        second_cycle = [env._sample_instruction_type() for _ in range(3)]

        self.assertEqual(set(first_cycle), {"move_up", "move_down", "move_left"})
        self.assertEqual(set(second_cycle), {"move_up", "move_down", "move_left"})

    def test_requested_instruction_type_bypasses_cycle(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.instruction_types = ("move_up", "move_down")
        env.instruction_sampling = "uniform_cycle"
        env._instruction_cycle = ["move_down"]
        env.np_random = np.random.default_rng(0)

        selected = env._sample_instruction_type(options={"instruction_type": "move_up"})

        self.assertEqual(selected, "move_up")
        self.assertEqual(env._instruction_cycle, ["move_down"])

    def test_sample_scene_can_filter_required_objects(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.scenes = [
            SceneSpec(name="desk_a", objects=("ycb_apple", "ycb_mug")),
            SceneSpec(name="desk_b", objects=("ycb_apple", "ycb_plate")),
        ]
        env.np_random = np.random.default_rng(0)

        scene = env._sample_scene(options={"required_objects": ["ycb_plate"]})

        self.assertEqual(scene.name, "desk_b")

    def test_instruction_curriculum_filters_allowed_candidates_by_episode(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.instruction_types = ("grab_object", "push_left", "put_into_plate")
        env._task_metadata = {
            "instruction_curriculum": [
                {"until_episode": 2, "instruction_types": ["grab_object"]},
                {"until_episode": 4, "instruction_types": ["grab_object", "push_left"]},
                {"instruction_types": ["grab_object", "push_left", "put_into_plate"]},
            ],
        }

        env._episode_index = 0
        self.assertEqual(env._allowed_instruction_candidates(), ("grab_object",))

        env._episode_index = 3
        self.assertEqual(env._allowed_instruction_candidates(), ("grab_object", "push_left"))

        env._episode_index = 4
        self.assertEqual(
            env._allowed_instruction_candidates(),
            ("grab_object", "push_left", "put_into_plate"),
        )

    def test_put_instruction_uses_catchable_target_and_container_reference(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "catchable_object_pool": ["ycb_apple", "ycb_pear"],
            "container_object_pool": ["plate", "bowl"],
        }
        env._catalog_to_body = {
            "ycb_apple": "apple_body",
            "plate": "plate_body",
            "bowl": "bowl_body",
        }
        env.np_random = np.random.default_rng(0)
        scene = SceneSpec(name="desk", objects=("ycb_apple", "plate", "bowl"))

        target_catalog, _target_body, reference_catalog, _reference_body, _second_catalog, _second_body = (
            env._select_instruction_objects(scene, instruction_type="put_into_plate")
        )

        self.assertEqual(target_catalog, "ycb_apple")
        self.assertIn(reference_catalog, {"plate", "bowl"})

    def test_relation_instruction_uses_catchable_target(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "catchable_object_pool": ["ycb_apple"],
            "container_object_pool": ["plate", "bowl"],
        }
        env._catalog_to_body = {
            "ycb_apple": "apple_body",
            "plate": "plate_body",
            "bowl": "bowl_body",
        }
        env.np_random = np.random.default_rng(1)
        scene = SceneSpec(name="desk", objects=("plate", "ycb_apple", "bowl"))

        target_catalog, _target_body, reference_catalog, _reference_body, _second_catalog, _second_body = (
            env._select_instruction_objects(scene, instruction_type="put_in_front_of_object")
        )

        self.assertEqual(target_catalog, "ycb_apple")
        self.assertIn(reference_catalog, {"plate", "bowl"})

    def test_caught_object_start_spawns_target_at_end_effector_for_relation_task(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "caught_object_start_probability": 1.0,
            "caught_object_start_object_offset": [0.0, 0.0, -0.04],
            "caught_object_start_xy_jitter": 0.0,
            "caught_object_start_z_jitter": 0.0,
        }
        env.np_random = np.random.default_rng(0)
        env._support_surface_z = 0.15
        env._target_body_name = "apple_body"
        env._target_catalog_name = "ycb_apple"
        env.sim = object()
        env._get_ee_position = lambda: np.array([0.10, -0.05, 0.42], dtype=np.float32)
        env._force_gripper_opening = lambda target: None
        placed: dict[str, np.ndarray] = {}

        def _set_body_position(body_name, xyz):
            placed[str(body_name)] = np.asarray(xyz, dtype=np.float32).copy()
            return True

        env._set_body_position = _set_body_position
        env._reset_caught_object_start_state()

        spawned = env._maybe_spawn_target_caught_at_ee(instruction_type="move_between_objects")

        self.assertTrue(spawned)
        self.assertTrue(env._caught_object_start_active)
        self.assertEqual(env._caught_object_start_body, "apple_body")
        np.testing.assert_allclose(placed["apple_body"], np.array([0.10, -0.05, 0.38], dtype=np.float32))
        np.testing.assert_allclose(
            env._caught_object_start_ee_offset,
            np.array([0.0, 0.0, -0.04], dtype=np.float32),
            atol=1e-6,
        )

    def test_caught_object_start_pose_follows_closed_gripper_until_release(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {"caught_object_start_release_opening_threshold": 0.10}
        env._caught_object_start_active = True
        env._caught_object_start_body = "apple_body"
        env._caught_object_start_catalog = "ycb_apple"
        env._caught_object_start_position = np.array([0.10, -0.05, 0.38], dtype=np.float32)
        env._caught_object_start_ee_offset = np.array([0.0, 0.0, -0.04], dtype=np.float32)
        env._get_ee_position = lambda: np.array([0.20, 0.10, 0.44], dtype=np.float32)
        env._get_gripper_target = lambda: 0.0
        env._get_gripper_opening = lambda: 0.0
        placed: dict[str, np.ndarray] = {}

        def _set_body_position(body_name, xyz):
            placed[str(body_name)] = np.asarray(xyz, dtype=np.float32).copy()
            return True

        env._set_body_position = _set_body_position

        self.assertTrue(env._maintain_caught_object_start_pose())
        np.testing.assert_allclose(placed["apple_body"], np.array([0.20, 0.10, 0.40], dtype=np.float32))

        env._get_gripper_target = lambda: 0.25
        self.assertFalse(env._maintain_caught_object_start_pose())
        self.assertFalse(env._caught_object_start_active)

    def test_caught_object_start_does_not_apply_to_grab_task_by_default(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {"caught_object_start_probability": 1.0}
        env.np_random = np.random.default_rng(0)
        env._target_body_name = "apple_body"

        self.assertFalse(env._should_spawn_target_caught_at_ee(instruction_type="grab_object"))


if __name__ == "__main__":
    unittest.main()
