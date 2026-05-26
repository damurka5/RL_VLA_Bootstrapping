from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
    INSTRUCTION_SUCCESS_CRITERIA,
    INSTRUCTION_TYPES,
    InstructionSpec,
    canonical_object_name,
    compute_instruction_validation_success,
    init_reward_state,
    sample_instruction,
)


class InstructionTextTests(unittest.TestCase):
    def test_every_instruction_type_has_success_criteria_text(self):
        self.assertEqual(set(INSTRUCTION_SUCCESS_CRITERIA), set(INSTRUCTION_TYPES))
        for instruction_type, criteria in INSTRUCTION_SUCCESS_CRITERIA.items():
            self.assertIn(instruction_type, INSTRUCTION_TYPES)
            self.assertTrue(criteria.strip())

    def test_move_to_object_instruction_uses_object_text(self):
        spec = sample_instruction(
            target_object="ycb_plate",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["move_to_object"],
        )

        self.assertEqual(spec.instruction_type, "move_to_object")
        self.assertEqual(spec.text, "move to plate")
        self.assertEqual(spec.target_object, "ycb_plate")

    def test_pick_up_instruction_uses_object_text(self):
        spec = sample_instruction(
            target_object="ycb_apple",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["pick_up"],
        )

        self.assertEqual(spec.instruction_type, "pick_up")
        self.assertEqual(spec.text, "pick up apple")
        self.assertEqual(spec.target_object, "ycb_apple")

    def test_complex_manipulation_instructions_use_reference_text(self):
        spec = sample_instruction(
            target_object="ycb_apple",
            reference_object="ycb_plate",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["put_into_plate"],
        )

        self.assertEqual(spec.instruction_type, "put_into_plate")
        self.assertEqual(spec.text, "put apple into plate")
        self.assertEqual(spec.reference_object, "ycb_plate")

        between = sample_instruction(
            target_object="ycb_mug",
            reference_object="ycb_apple",
            second_reference_object="ycb_pear",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["move_between_objects"],
        )

        self.assertEqual(between.text, "move mug between apple and pear")
        self.assertEqual(between.second_reference_object, "ycb_pear")

        front = sample_instruction(
            target_object="ycb_apple",
            reference_object="ycb_pear",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["put_in_front_of_object"],
        )
        behind = sample_instruction(
            target_object="ycb_apple",
            reference_object="ycb_pear",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["put_behind_object"],
        )
        move_front = sample_instruction(
            target_object="ycb_apple",
            reference_object="ycb_pear",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["move_in_front_of_object"],
        )
        move_behind = sample_instruction(
            target_object="ycb_apple",
            reference_object="ycb_pear",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["move_behind_object"],
        )

        self.assertEqual(front.text, "put apple in front of pear")
        self.assertEqual(behind.text, "put apple behind pear")
        self.assertEqual(move_front.text, "move apple in front of pear")
        self.assertEqual(move_behind.text, "move apple behind pear")

    def test_push_forward_instruction_uses_object_text(self):
        spec = sample_instruction(
            target_object="ycb_apple",
            rng=np.random.default_rng(0),
            allowed_instruction_types=["push_forward"],
        )

        self.assertEqual(spec.instruction_type, "push_forward")
        self.assertEqual(spec.text, "push apple forward")

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
        self.assertEqual(canonical_object_name("ycb_b_cups"), "cup")
        self.assertEqual(canonical_object_name("ycb_plate"), "plate")
        self.assertEqual(canonical_object_name("ycb_bowl"), "bowl")

    def test_canonical_object_name_strips_prefixes_and_underscores_generically(self):
        self.assertEqual(canonical_object_name("ycb_demo_object"), "demo object")
        self.assertEqual(canonical_object_name("custom_test_item"), "custom test item")

    def test_sample_instruction_honors_explicit_instruction_type(self):
        spec = sample_instruction(
            target_object="ycb_apple",
            rng=np.random.default_rng(7),
            allowed_instruction_types=["move_left", "move_right"],
            instruction_type="move_right",
        )

        self.assertEqual(spec.instruction_type, "move_right")
        self.assertEqual(spec.text, "move right")

    def test_directional_validation_uses_workspace_center_for_horizontal_moves(self):
        spec = InstructionSpec(
            instruction_type="move_left",
            text="move left",
            target_object="",
            direction=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
        )
        reward_state = init_reward_state(
            initial_ee_pos=np.array([0.16, 0.02, 0.40], dtype=np.float32),
            initial_obj_pos=np.zeros((3,), dtype=np.float32),
        )

        success, info = compute_instruction_validation_success(
            spec=spec,
            ee_pos=np.array([-0.06, 0.02, 0.40], dtype=np.float32),
            reward_state=reward_state,
            task_metadata={
                "goal_center_xy": [0.0, 0.0],
                "directional_success_center_threshold": 0.05,
            },
            current_success=False,
        )

        self.assertTrue(success)
        self.assertAlmostEqual(info["directional_success_signed_displacement"], 0.06, places=6)
        self.assertEqual(info["directional_success_reference_is_workspace_center"], 1.0)

    def test_move_to_object_validation_uses_xy_distance_threshold(self):
        spec = InstructionSpec(
            instruction_type="move_to_object",
            text="move to apple",
            target_object="ycb_apple",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.40,
            lift_target=0.10,
        )
        reward_state = init_reward_state(
            initial_ee_pos=np.array([0.0, 0.0, 0.40], dtype=np.float32),
            initial_obj_pos=np.array([0.03, 0.02, 0.16], dtype=np.float32),
        )

        success, info = compute_instruction_validation_success(
            spec=spec,
            ee_pos=np.array([0.01, 0.02, 0.30], dtype=np.float32),
            reward_state=reward_state,
            task_metadata={"move_to_object_validation_distance_threshold": 0.10},
            current_success=False,
            obj_pos=np.array([0.09, 0.02, 0.16], dtype=np.float32),
        )

        self.assertTrue(success)
        self.assertGreater(info["move_to_object_validation_distance_xyz"], 0.10)
        self.assertGreater(info["move_to_object_validation_distance_xy"], 0.05)
        self.assertLess(info["move_to_object_validation_distance_xy"], 0.10)
        self.assertAlmostEqual(info["move_to_object_validation_distance_threshold"], 0.10, places=7)


if __name__ == "__main__":
    unittest.main()
