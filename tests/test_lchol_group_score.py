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


if __name__ == "__main__":
    unittest.main()
