from __future__ import annotations

import unittest

from robots.cdpr.cdpr_dataset.cdpr_lchol_spec import CDPRLCHOLSpec


class CDPRLCHOLSpecTests(unittest.TestCase):
    def test_push_opposite_direction_is_relabelled(self):
        spec = CDPRLCHOLSpec()
        achieved = spec.achieved_options(
            [
                {
                    "instruction_type": "push_right",
                    "source_instruction": "push apple right",
                    "target_object_catalog": "ycb_apple",
                    "target_motion_x": -0.10,
                }
            ]
        )

        self.assertEqual(achieved[0].option_name, "push_left")
        self.assertEqual(spec.relabel_instruction(achieved[0]), "push apple left")

    def test_relation_score_penalizes_wrong_side(self):
        spec = CDPRLCHOLSpec()
        good = spec.phase_score(
            [
                {
                    "instruction_type": "move_right_of_object",
                    "sparse_success": 0.0,
                    "distance_ee_to_object_xy": 0.04,
                    "relation_error": 0.02,
                    "signed_relation_offset": 0.06,
                    "relation_motion_ok": 1.0,
                }
            ]
        )
        wrong = spec.phase_score(
            [
                {
                    "instruction_type": "move_right_of_object",
                    "sparse_success": 0.0,
                    "distance_ee_to_object_xy": 0.04,
                    "relation_error": 0.12,
                    "signed_relation_offset": -0.06,
                    "relation_motion_ok": 1.0,
                }
            ]
        )

        self.assertGreater(good, wrong)


if __name__ == "__main__":
    unittest.main()
