from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv


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


if __name__ == "__main__":
    unittest.main()
