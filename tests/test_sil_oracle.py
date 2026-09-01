"""The batched oracle's phase machine, driven against a kinematic stand-in.

The premise of the whole oracle-to-bank path is that the scripted chain can
complete a composed put_into: approach, close, lift, traverse, descend,
release. If it stalls in a phase it produces a demonstration of hovering, and
if it skips one it produces a demonstration missing a motion. Neither raises,
and neither is visible until an SFT has been trained on the result.

The plant here is deliberately crude -- a first-order response with the
measured ~0.5 realised gain, plus a kinematic grasp. It is not a physics check;
it exists to advance the state machine so the chain can be observed end to end.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.audit.sil_oracle import BatchedOracle

GRASP_OFFSET = 0.0075
STEP_XYZ = 0.015
STEP_GRIPPER = 0.05
FITTED = 0.35
PLANT_GAIN = 0.5


def _oracle(instruction: str, *, starts_grasped: bool) -> BatchedOracle:
    return BatchedOracle(
        instruction_types=[instruction],
        starts_grasped=[starts_grasped],
        instruction_texts=[f"put apple into {instruction.rsplit('_', 1)[-1]}"],
        target_slots=[0],
        reference_slots=[1],
        target_catalogs=["robocasa_apple"],
        fitted_openings=[FITTED],
        release_openings=[0.60],
        grasp_height_offset=GRASP_OFFSET,
        release_heights=[0.10],
        support_surface_z=[0.15],
        lift_success_height=0.05,
        action_step_xyz=STEP_XYZ,
        action_step_gripper=STEP_GRIPPER,
    )


def _rollout(oracle: BatchedOracle, *, steps: int, held: bool = False):
    """Advance the chain against a first-order plant. Returns phases seen."""

    ee = np.array([[0.0, 0.0, 0.32]])
    objects = np.zeros((1, 4, 3))
    objects[0, 0] = [0.10, 0.05, 0.185]          # the target on the desk
    objects[0, 1] = [0.16, 0.09, 0.150]          # the receptacle
    if held:
        objects[0, 0] = ee[0] - np.array([0.0, 0.0, GRASP_OFFSET])
    commanded = np.array([0.0 if held else 1.0])
    grasped = bool(held)
    previous = ee.copy()
    seen: list[str] = []
    for _ in range(steps):
        seen.append(oracle.phase_names()[0])
        action = oracle.actions(
            ee=ee,
            ee_velocity=ee - previous,
            ee_yaw=np.zeros(1),
            measured_gripper=commanded.copy(),
            commanded_gripper=commanded,
            object_positions=objects,
            physical_grasp=np.array([grasped]),
            initial_target_z=np.array([0.185]),
        )
        previous = ee.copy()
        ee = ee + action[:, :3] * STEP_XYZ * PLANT_GAIN
        commanded = np.clip(commanded + action[:, 4] * STEP_GRIPPER, 0.0, 1.0)
        close_enough = (
            float(np.linalg.norm(ee[0] - (objects[0, 0] + [0, 0, GRASP_OFFSET])))
            < 0.02
        )
        if not grasped and close_enough and commanded[0] <= FITTED:
            grasped = True
        if grasped and commanded[0] > 0.55:
            grasped = False                       # released
        if grasped:
            objects[0, 0] = ee[0] - np.array([0.0, 0.0, GRASP_OFFSET])
    return seen


class ComposedChainTests(unittest.TestCase):
    def test_the_composed_chain_is_the_grasp_then_the_carry(self) -> None:
        names = [p.name for p in _oracle("put_into_bowl", starts_grasped=False).phases[0]]
        self.assertEqual(
            names,
            [
                "align_above_object",
                "descend_to_grasp_point",
                "close_fingers",
                "raise_to_transit_height",
                "traverse_to_receptacle",
                "descend_to_release_height",
                "open_gripper_release",
                "settle_in_receptacle",
            ],
        )

    def test_a_caught_start_skips_the_grasp(self) -> None:
        names = [p.name for p in _oracle("put_into_plate", starts_grasped=True).phases[0]]
        self.assertNotIn("close_fingers", names)
        self.assertEqual(names[0], "raise_to_transit_height")

    def test_the_chain_advances_past_the_grasp_and_reaches_the_release(self) -> None:
        # The whole premise: a script can do what no policy here can.
        oracle = _oracle("put_into_bowl", starts_grasped=False)
        seen = _rollout(oracle, steps=400)
        for phase in ("align_above_object", "descend_to_grasp_point",
                      "close_fingers", "raise_to_transit_height",
                      "traverse_to_receptacle"):
            self.assertIn(phase, seen, f"never reached {phase}")

    def test_no_phase_swallows_the_whole_episode(self) -> None:
        # A phase whose done() never fires yields a demonstration of hovering.
        oracle = _oracle("put_into_bowl", starts_grasped=False)
        seen = _rollout(oracle, steps=400)
        longest = max(seen.count(name) for name in set(seen))
        self.assertLess(longest, 380, f"one phase held {longest}/400 steps")

    def test_phases_only_move_forward(self) -> None:
        oracle = _oracle("put_into_bowl", starts_grasped=False)
        _rollout(oracle, steps=200)
        chain = [p.name for p in oracle.phases[0]]
        self.assertLessEqual(int(oracle.phase_index[0]), len(chain) - 1)
        self.assertGreater(int(oracle.phase_index[0]), 0)


class BatchBehaviourTests(unittest.TestCase):
    def test_an_unsupported_instruction_gets_a_zero_action(self) -> None:
        # move_to has no oracle. It must not change the batch shape or raise.
        oracle = BatchedOracle(
            instruction_types=["move_to_object", "put_into_bowl"],
            starts_grasped=[False, False],
            instruction_texts=["move to apple", "put apple into bowl"],
            target_slots=[0, 0],
            reference_slots=[-1, 1],
            target_catalogs=["robocasa_apple"] * 2,
            fitted_openings=[FITTED] * 2,
            release_openings=[0.60] * 2,
            grasp_height_offset=GRASP_OFFSET,
            release_heights=[0.10] * 2,
            support_surface_z=[0.15] * 2,
            lift_success_height=0.05,
            action_step_xyz=STEP_XYZ,
            action_step_gripper=STEP_GRIPPER,
        )
        self.assertEqual(oracle.supported.tolist(), [False, True])
        objects = np.zeros((2, 4, 3))
        objects[:, 0] = [0.10, 0.05, 0.185]
        objects[:, 1] = [0.16, 0.09, 0.150]
        action = oracle.actions(
            ee=np.array([[0.0, 0.0, 0.30]] * 2),
            ee_velocity=np.zeros((2, 3)),
            ee_yaw=np.zeros(2),
            measured_gripper=np.ones(2),
            commanded_gripper=np.ones(2),
            object_positions=objects,
            physical_grasp=np.zeros(2, dtype=bool),
            initial_target_z=np.full(2, 0.185),
        )
        self.assertEqual(action.shape, (2, 5))
        self.assertTrue(bool((action[0] == 0).all()))
        self.assertTrue(bool(np.abs(action[1]).sum() > 0))

    def test_actions_stay_inside_the_normalised_range(self) -> None:
        oracle = _oracle("put_into_bowl", starts_grasped=False)
        ee = np.array([[0.4, -0.4, 0.5]])          # deliberately far
        objects = np.zeros((1, 4, 3))
        objects[0, 0] = [-0.2, 0.2, 0.185]
        objects[0, 1] = [0.16, 0.09, 0.150]
        action = oracle.actions(
            ee=ee, ee_velocity=np.zeros((1, 3)), ee_yaw=np.zeros(1),
            measured_gripper=np.ones(1), commanded_gripper=np.ones(1),
            object_positions=objects, physical_grasp=np.zeros(1, dtype=bool),
            initial_target_z=np.array([0.185]),
        )
        self.assertTrue(bool((action >= -1.0).all() and (action <= 1.0).all()))

    def test_worlds_hold_independent_phase_state(self) -> None:
        oracle = BatchedOracle(
            instruction_types=["put_into_bowl", "put_into_bowl"],
            starts_grasped=[False, True],
            instruction_texts=["put apple into bowl"] * 2,
            target_slots=[0, 0], reference_slots=[1, 1],
            target_catalogs=["robocasa_apple"] * 2,
            fitted_openings=[FITTED] * 2, release_openings=[0.60] * 2,
            grasp_height_offset=GRASP_OFFSET, release_heights=[0.10] * 2,
            support_surface_z=[0.15] * 2, lift_success_height=0.05,
            action_step_xyz=STEP_XYZ, action_step_gripper=STEP_GRIPPER,
        )
        self.assertEqual(
            oracle.phase_names(),
            ["align_above_object", "raise_to_transit_height"],
        )


if __name__ == "__main__":
    unittest.main()
