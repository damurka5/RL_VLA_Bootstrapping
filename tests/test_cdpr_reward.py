from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
    InstructionSpec,
    compute_instruction_reward,
    init_reward_state,
)


class RewardDistanceTests(unittest.TestCase):
    def _spec(self, instruction_type: str = "move_left") -> InstructionSpec:
        direction = {
            "move_left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "move_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }[instruction_type]
        text = instruction_type.replace("_", " ")
        return InstructionSpec(
            instruction_type=instruction_type,
            text=text,
            target_object="",
            direction=direction,
            target_displacement=0.40,
            lift_target=0.10,
        )

    def test_closer_goal_has_higher_reward(self):
        spec = self._spec("move_left")
        goal = np.array([0.0, 0.0, 0.20], dtype=np.float32)

        reward_far, success_far, info_far = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.20, 0.0, 0.20], dtype=np.float32),
            obj_pos=goal,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.20, 0.0, 0.20], dtype=np.float32),
                initial_obj_pos=goal,
            ),
            camera_alignment=1.0,
            goal_direction=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        )
        reward_near, success_near, info_near = compute_instruction_reward(
            spec=spec,
            ee_pos=np.array([0.05, 0.0, 0.20], dtype=np.float32),
            obj_pos=goal,
            reward_state=init_reward_state(
                initial_ee_pos=np.array([0.05, 0.0, 0.20], dtype=np.float32),
                initial_obj_pos=goal,
            ),
            camera_alignment=1.0,
            goal_direction=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        )

        self.assertFalse(success_far)
        self.assertFalse(success_near)
        self.assertGreater(reward_near, reward_far)
        self.assertGreater(info_far["distance_to_goal"], info_near["distance_to_goal"])

    def test_camera_alignment_changes_reward_and_success(self):
        spec = self._spec("move_up")
        goal = np.array([0.0, 0.0, 0.23], dtype=np.float32)
        ee = np.array([0.0, 0.0, 0.205], dtype=np.float32)

        reward_bad, success_bad, info_bad = compute_instruction_reward(
            spec=spec,
            ee_pos=ee,
            obj_pos=goal,
            reward_state=init_reward_state(initial_ee_pos=ee, initial_obj_pos=goal),
            camera_alignment=0.15,
            goal_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        reward_good, success_good, info_good = compute_instruction_reward(
            spec=spec,
            ee_pos=ee,
            obj_pos=goal,
            reward_state=init_reward_state(initial_ee_pos=ee, initial_obj_pos=goal),
            camera_alignment=0.95,
            goal_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

        self.assertFalse(success_bad)
        self.assertTrue(success_good)
        self.assertGreater(reward_good, reward_bad)
        self.assertGreater(info_good["camera_reward"], info_bad["camera_reward"])


if __name__ == "__main__":
    unittest.main()
