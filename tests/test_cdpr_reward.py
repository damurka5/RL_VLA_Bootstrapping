from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
    InstructionSpec,
    compute_instruction_reward,
    init_reward_state,
)


class RewardDistanceTests(unittest.TestCase):
    def _pick_spec(self) -> InstructionSpec:
        return InstructionSpec(
            instruction_type="pick_up",
            text="pick up apple",
            target_object="ycb_apple",
            direction=np.zeros((2,), dtype=np.float32),
            target_displacement=0.20,
            lift_target=0.10,
        )

    def test_far_reach_uses_xy_distance_only(self):
        spec = self._pick_spec()
        reward_state = init_reward_state(
            initial_ee_pos=np.array([0.12, 0.0, 0.30], dtype=np.float32),
            initial_obj_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

        reward, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.10, 0.0, 0.30], dtype=np.float32),
            obj_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            reward_state=reward_state,
        )

        self.assertFalse(success)
        self.assertGreater(reward, 0.0)
        self.assertAlmostEqual(info["distance_ee_to_object"], 0.10, places=6)
        self.assertAlmostEqual(info["distance_ee_to_object_xy"], 0.10, places=6)
        self.assertGreater(info["distance_ee_to_object_xyz"], info["distance_ee_to_object"])
        self.assertEqual(info["distance_use_xy_only"], 1.0)

    def test_near_reach_switches_back_to_full_xyz_distance(self):
        spec = self._pick_spec()
        reward_state = init_reward_state(
            initial_ee_pos=np.array([0.025, 0.0, 0.05], dtype=np.float32),
            initial_obj_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

        _, success, info = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.020, 0.0, 0.03], dtype=np.float32),
            obj_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            reward_state=reward_state,
        )

        expected_xyz = float(np.linalg.norm(np.array([0.020, 0.0, 0.03], dtype=np.float32)))
        self.assertFalse(success)
        self.assertAlmostEqual(info["distance_ee_to_object"], expected_xyz, places=6)
        self.assertAlmostEqual(info["distance_ee_to_object_xyz"], expected_xyz, places=6)
        self.assertAlmostEqual(info["distance_ee_to_object_xy"], 0.02, places=6)
        self.assertEqual(info["distance_use_xy_only"], 0.0)


if __name__ == "__main__":
    unittest.main()
