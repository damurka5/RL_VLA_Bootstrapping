from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_instruction_tasks import canonical_object_name, sample_instruction


class InstructionTextTests(unittest.TestCase):
    def test_pick_up_instruction_uses_object_text(self):
        spec = sample_instruction(
            target_object="ycb_apple",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["pick_up"],
        )

        self.assertEqual(spec.instruction_type, "pick_up")
        self.assertEqual(spec.text, "pick up apple")
        self.assertEqual(spec.target_object, "ycb_apple")

    def test_move_top_instruction_uses_forward_text(self):
        spec = sample_instruction(
            target_object="ycb_apple",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["move_top"],
        )

        self.assertEqual(spec.instruction_type, "move_top")
        self.assertEqual(spec.text, "move forward")

    def test_move_bottom_instruction_uses_backward_text(self):
        spec = sample_instruction(
            target_object="ycb_apple",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["move_bottom"],
        )

        self.assertEqual(spec.instruction_type, "move_bottom")
        self.assertEqual(spec.text, "move backward")

    def test_canonical_object_name_strips_known_prefixes(self):
        self.assertEqual(canonical_object_name("ycb_mug"), "mug")
        self.assertEqual(canonical_object_name("ycb_b_cups"), "cups")

    def test_sample_instruction_honors_explicit_instruction_type(self):
        spec = sample_instruction(
            target_object="ycb_apple",
            rng=np.random.default_rng(7),
            allowed_instruction_types=["move_left", "move_right"],
            instruction_type="move_right",
        )

        self.assertEqual(spec.instruction_type, "move_right")
        self.assertEqual(spec.text, "move right")


if __name__ == "__main__":
    unittest.main()
