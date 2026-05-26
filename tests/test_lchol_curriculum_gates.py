from __future__ import annotations

import unittest

import numpy as np

from rl_vla_bootstrapping.lchol.curriculum import StrictSuccessCurriculum


class LCHOLCurriculumGateTests(unittest.TestCase):
    def test_success_gate_promotes_stage(self):
        curriculum = StrictSuccessCurriculum(min_success_samples=3, window_size=8)
        self.assertEqual(curriculum.stage.name, "approach")

        for _ in range(3):
            curriculum.record({"instruction_type": "move_to_object", "success": True})
            curriculum.record({"instruction_type": "grab_object", "success": True})

        self.assertEqual(curriculum.stage.name, "grasp")
        self.assertIn("pick_up", curriculum.allowed_options())

    def test_sampling_biases_weak_option(self):
        curriculum = StrictSuccessCurriculum(min_success_samples=3, window_size=8)
        for _ in range(4):
            curriculum.record({"instruction_type": "move_to_object", "success": True})
            curriculum.record({"instruction_type": "grab_object", "success": True})
        for _ in range(4):
            curriculum.record({"instruction_type": "pick_up", "success": False})

        draws = [
            curriculum.sample_option(rng=np.random.default_rng(seed), available_options=["grab_object", "pick_up"])
            for seed in range(40)
        ]

        self.assertGreater(draws.count("pick_up"), draws.count("grab_object"))

    def test_relation_motion_flag_alone_does_not_count_as_success(self):
        curriculum = StrictSuccessCurriculum(min_success_samples=3, window_size=8)

        for _ in range(3):
            curriculum.record({"instruction_type": "move_to_object", "success": True})
            curriculum.record({"instruction_type": "grab_object", "relation_motion_ok": True})

        self.assertEqual(curriculum.stage.name, "approach")

    def test_front_and_behind_options_are_available_in_later_stage(self):
        curriculum = StrictSuccessCurriculum(min_success_samples=1, window_size=8)
        for option in ("move_to_object", "grab_object"):
            curriculum.record({"instruction_type": option, "success": True})
        for option in ("grab_object", "pick_up"):
            curriculum.record({"instruction_type": option, "success": True})
        for option in ("push_left", "push_right", "push_forward", "push_backward"):
            curriculum.record({"instruction_type": option, "success": True})
        curriculum.record({"instruction_type": "put_into_plate", "success": True})

        self.assertIn("move_in_front_of_object", curriculum.allowed_options())
        self.assertIn("move_behind_object", curriculum.allowed_options())
        self.assertIn("put_in_front_of_object", curriculum.allowed_options())
        self.assertIn("put_behind_object", curriculum.allowed_options())


if __name__ == "__main__":
    unittest.main()
