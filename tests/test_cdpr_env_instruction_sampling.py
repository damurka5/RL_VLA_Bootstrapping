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


if __name__ == "__main__":
    unittest.main()
