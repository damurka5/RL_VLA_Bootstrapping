from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_instruction_tasks import sample_instruction


class InstructionTextTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
