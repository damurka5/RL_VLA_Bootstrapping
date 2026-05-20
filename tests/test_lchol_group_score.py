from __future__ import annotations

import unittest

import numpy as np

from rl_vla_bootstrapping.lchol.group_score import group_relative_advantages
from robots.cdpr.cdpr_dataset.cdpr_lchol_spec import CDPRLCHOLSpec


class LCHOLGroupScoreTests(unittest.TestCase):
    def test_phase_scores_rank_two_sparse_failures(self):
        spec = CDPRLCHOLSpec()
        weak = {
            "instruction_type": "grab_object",
            "success": False,
            "sparse_success": 0.0,
            "distance_ee_to_object_xy": 0.20,
            "gripper_closed": 0.0,
            "caught_object_is_target": 0.0,
        }
        better = {
            **weak,
            "distance_ee_to_object_xy": 0.03,
            "gripper_closed": 1.0,
            "caught_object_is_target": 1.0,
        }

        scores = np.asarray([spec.phase_score([weak]), spec.phase_score([better])], dtype=np.float32)
        advantages = group_relative_advantages(scores, normalize=True)

        self.assertGreater(scores[1], scores[0])
        self.assertLess(advantages[0], 0.0)
        self.assertGreater(advantages[1], 0.0)

    def test_sparse_success_dominates_phase_progress(self):
        spec = CDPRLCHOLSpec()
        success_low_progress = {
            "instruction_type": "grab_object",
            "success": True,
            "sparse_success": 1.0,
            "distance_ee_to_object_xy": 0.40,
            "gripper_closed": 0.0,
            "caught_object_is_target": 0.0,
        }
        failure_high_progress = {
            "instruction_type": "grab_object",
            "success": False,
            "sparse_success": 0.0,
            "distance_ee_to_object_xy": 0.0,
            "gripper_closed": 1.0,
            "caught_object_is_target": 1.0,
            "pick_target_lift_normalized": 1.0,
        }

        self.assertGreater(spec.phase_score([success_low_progress]), spec.phase_score([failure_high_progress]))

    def test_failure_phase_score_is_bounded(self):
        spec = CDPRLCHOLSpec()
        failure = {
            "instruction_type": "push_right",
            "success": False,
            "sparse_success": 0.0,
            "distance_ee_to_object_xy": 0.0,
            "target_motion_x": 100.0,
            "target_motion_xy": 100.0,
            "push_success_displacement": 0.02,
        }

        score = spec.phase_score([failure])

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
