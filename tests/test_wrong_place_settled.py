"""`wrong_place_settled` must test the PLACE, not whether the gripper is open.

THE REGRESSION, AND HOW IT WAS FOUND

`wrong_place_settled` terminates a container episode. It used to be defined
against `~container_ok`, and `container_ok` requires `released` -- the gripper
open past `max(release_threshold, 0.55)`. Meanwhile `state.grasped` is
`caught_target & (opening <= 0.94)`, and `caught_target` is the live
`physical_grasp`, which needs both pads loaded above 0.05 N.

Put those together and a CORRECT placement terminates as a wrong one. A carried
object touches down inside the receptacle while the gripper is still opening;
the surface takes the load; the pads unload; `~state.grasped` goes true;
`target_has_settled` is already true; and `released` is not true YET. Against
`~container_ok` the conjunction fires on the step the object lands.

Measured on `phase7_sparse_joint` step_2017690 with
`tools/audit/grasp_loss_forensics.py`: 85 of composed plate's 140 `no_release`
grasp losses ended exactly this way -- median ZERO env steps between the latch
breaking and the termination, none timed out, none ended holding -- and 63 of
the 70 flagged had the object already at rest INSIDE the 0.091 m plate radius.
The policy was mid-release, commanding a_gripper +0.44 and a_z +0.26 upward,
and at `action_step_gripper` 0.05 needed roughly eleven more steps to cross the
release bar. It got none.

This is §7.8's failure in a third place: a terminal condition sharing a conjunct
with success, firing because that conjunct is not satisfied yet.

WHAT THESE TESTS PIN

Four things, and the fourth is the safety property:

1. an object settled INSIDE the receptacle while still gripped does not
   terminate -- the regression itself;
2. a genuine geometric miss still does, in xy and in z separately, because a
   fix that stopped terminating real wrong drops would let every failed
   placement run to the horizon;
3. success is untouched -- the release is still required to SUCCEED, so nothing
   here makes the task easier to pass;
4. the fix strictly NARROWS termination. `container_ok` implies
   `placement_geometry_ok`, so every episode the old form ended is still ended
   except the ones it should never have. Expressed publicly as
   `wrong_place_drop` implying `not success`.
"""

from __future__ import annotations

import unittest

import torch

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    INSTRUCTION_TO_ID,
    BatchedCatchReleaseDenseReward,
    BatchedTaskState,
    evaluate_active_sparse_tasks,
)

SUPPORT_Z = 0.15
REST_HEIGHT = 0.03
SETTLE_MARGIN = 0.025
# Anything at or below this counts as settled on the support surface.
SETTLED_Z = SUPPORT_Z + REST_HEIGHT + SETTLE_MARGIN - 0.005
CARRY_Z = SUPPORT_Z + 0.15
PLATE_RADIUS = 0.091
BOWL_RADIUS = 0.057

# Below `release_opening` 0.55, so `released` is false: the gripper has begun to
# open and has not got there yet, which is the whole situation under test.
STILL_CLOSING = 0.30
OPEN = 0.90


class _World:
    """One container episode at one instant, in the units the predicate reads.

    `grasped` is passed explicitly rather than derived, because the situation
    being tested is precisely the one where the pads have unloaded (so the live
    `caught_target` is False) while the gripper has not yet opened.
    """

    def __init__(
        self,
        *,
        instruction: str,
        offset_xy: float,
        object_z: float,
        reference_z: float,
        gripper_opening: float,
        caught: bool,
    ) -> None:
        self.instruction = instruction
        self.offset_xy = offset_xy
        self.object_z = object_z
        self.reference_z = reference_z
        self.gripper_opening = gripper_opening
        self.caught = caught


def _evaluate(worlds: list[_World]):
    count = len(worlds)
    objects = torch.zeros((count, 2, 3), dtype=torch.float32)
    for index, world in enumerate(worlds):
        objects[index, 0] = torch.tensor(
            [world.offset_xy, 0.0, world.object_z], dtype=torch.float32
        )
        objects[index, 1] = torch.tensor(
            [0.0, 0.0, world.reference_z], dtype=torch.float32
        )
    # The object started elsewhere, so `minimum_target_motion` cannot be what
    # decides any of these verdicts.
    initial = objects[:, 0].clone()
    initial[:, 0] += 0.20
    initial[:, 2] = CARRY_Z
    state = BatchedTaskState(
        instruction_ids=torch.tensor(
            [INSTRUCTION_TO_ID[world.instruction] for world in worlds],
            dtype=torch.int64,
        ),
        target_slots=torch.zeros((count,), dtype=torch.int64),
        reference_slots=torch.ones((count,), dtype=torch.int64),
        second_reference_slots=torch.full((count,), -1, dtype=torch.int64),
        initial_target_positions=initial,
        ever_grasped=torch.ones((count,), dtype=torch.bool),
        grasped=torch.zeros((count,), dtype=torch.bool),
        step_count=torch.zeros((count,), dtype=torch.int64),
        release_threshold=torch.full((count,), 0.55),
        support_surface_z=torch.full((count,), SUPPORT_Z),
        target_rest_height=torch.full((count,), REST_HEIGHT),
    )
    # The end effector is irrelevant to the container terms and is parked where
    # it cannot satisfy any move_to window by accident.
    ee = torch.zeros((count, 3), dtype=torch.float32)
    ee[:, 0] = 0.5
    ee[:, 2] = CARRY_Z
    return evaluate_active_sparse_tasks(
        state=state,
        ee_position=ee,
        object_positions=objects,
        gripper_opening=torch.tensor(
            [world.gripper_opening for world in worlds], dtype=torch.float32
        ),
        caught_target=torch.tensor(
            [world.caught for world in worlds], dtype=torch.bool
        ),
        active_mask=torch.ones((count,), dtype=torch.bool),
        max_steps=128,
        catch_release_dense_reward=BatchedCatchReleaseDenseReward(),
    )


def _landed(instruction: str, offset_xy: float, opening: float) -> _World:
    """The object is on the support surface and the pads have let go of it."""

    return _World(
        instruction=instruction,
        offset_xy=offset_xy,
        object_z=SETTLED_Z,
        reference_z=SUPPORT_Z + 0.01,
        gripper_opening=opening,
        caught=False,
    )


class CorrectPlacementIsNotTerminatedTest(unittest.TestCase):
    def test_settled_inside_the_plate_while_still_closing_survives(self) -> None:
        """The 63-episode case. Right place, gripper not open yet, keep going."""

        result = _evaluate([_landed("put_into_plate", 0.05, STILL_CLOSING)])
        self.assertFalse(bool(result.diagnostics["wrong_place_drop"][0]))
        self.assertFalse(bool(result.terminated[0]))
        # And it is not a success either: the release is still required to PASS.
        self.assertFalse(bool(result.success[0]))

    def test_settled_inside_the_bowl_while_still_closing_survives(self) -> None:
        result = _evaluate([_landed("put_into_bowl", 0.03, STILL_CLOSING)])
        self.assertFalse(bool(result.diagnostics["wrong_place_drop"][0]))
        self.assertFalse(bool(result.terminated[0]))

    def test_the_radius_is_the_receptacle_s_own(self) -> None:
        """A plate-width offset must still be a miss for the narrower bowl.

        Guards the split being made against one shared radius: 0.08 m is inside
        the plate and outside the bowl, and reading either radius for both would
        pass every other test in this file.
        """

        result = _evaluate(
            [
                _landed("put_into_plate", 0.08, STILL_CLOSING),
                _landed("put_into_bowl", 0.08, STILL_CLOSING),
            ]
        )
        self.assertFalse(bool(result.terminated[0]))
        self.assertTrue(bool(result.terminated[1]))


class GenuineWrongDropsStillTerminateTest(unittest.TestCase):
    def test_settled_outside_the_radius_terminates(self) -> None:
        result = _evaluate(
            [
                _landed("put_into_plate", 0.15, STILL_CLOSING),
                _landed("put_into_bowl", 0.15, STILL_CLOSING),
            ]
        )
        self.assertEqual(
            result.diagnostics["wrong_place_drop"].tolist(), [True, True]
        )
        self.assertEqual(result.terminated.tolist(), [True, True])

    def test_settled_outside_the_z_tolerance_terminates(self) -> None:
        """The z terms must survive the split, not just the xy one.

        Inside the radius in xy, but the receptacle is 0.30 m above the object
        -- past the 0.12 m tolerance. Dropping the z conjuncts from the geometry
        test would let an object that landed on the desk beside a tall
        receptacle run to the horizon.
        """

        world = _landed("put_into_plate", 0.02, STILL_CLOSING)
        world.reference_z = SETTLED_Z + 0.30
        result = _evaluate([world])
        self.assertTrue(bool(result.diagnostics["wrong_place_drop"][0]))
        self.assertTrue(bool(result.terminated[0]))

    def test_an_object_still_held_does_not_terminate(self) -> None:
        """`~state.grasped` is still required: mid-carry is not a placement."""

        world = _World(
            instruction="put_into_plate",
            offset_xy=0.15,
            object_z=CARRY_Z,
            reference_z=SUPPORT_Z + 0.01,
            gripper_opening=STILL_CLOSING,
            caught=True,
        )
        result = _evaluate([world])
        self.assertFalse(bool(result.diagnostics["wrong_place_drop"][0]))
        self.assertFalse(bool(result.terminated[0]))


class SuccessIsUnchangedTest(unittest.TestCase):
    def test_released_inside_the_radius_still_succeeds(self) -> None:
        result = _evaluate([_landed("put_into_plate", 0.05, OPEN)])
        self.assertTrue(bool(result.success[0]))
        self.assertTrue(bool(result.terminated[0]))
        self.assertFalse(bool(result.diagnostics["wrong_place_drop"][0]))

    def test_released_outside_the_radius_is_a_wrong_drop(self) -> None:
        result = _evaluate([_landed("put_into_plate", 0.15, OPEN)])
        self.assertFalse(bool(result.success[0]))
        self.assertTrue(bool(result.diagnostics["wrong_place_drop"][0]))

    def test_the_release_is_still_required_to_pass(self) -> None:
        """The fix must not turn a set-down into a success by itself.

        If it did, composed placement would be scored on arriving rather than on
        letting go, and every number in the campaign would shift for a reason
        that has nothing to do with the policy.
        """

        result = _evaluate(
            [
                _landed("put_into_plate", 0.05, STILL_CLOSING),
                _landed("put_into_plate", 0.05, OPEN),
            ]
        )
        self.assertEqual(result.success.tolist(), [False, True])


class TerminationOnlyNarrowsTest(unittest.TestCase):
    """`container_ok` implies `placement_geometry_ok`, so the fix removes
    terminations and never adds one.

    Checked over a grid rather than a chosen case, because the property is what
    makes this safe to land: no episode that used to run can start dying.
    """

    def test_a_wrong_drop_is_never_also_a_success(self) -> None:
        worlds = [
            _landed(instruction, offset, opening)
            for instruction in ("put_into_plate", "put_into_bowl")
            for offset in (0.0, 0.03, 0.05, 0.08, 0.15, 0.30)
            for opening in (STILL_CLOSING, 0.50, OPEN)
        ]
        result = _evaluate(worlds)
        wrong = result.diagnostics["wrong_place_drop"].bool()
        self.assertTrue(bool((~(wrong & result.success)).all()))
        # The grid must actually exercise both verdicts, or the assertion above
        # passes on a population where nothing happens.
        self.assertTrue(bool(wrong.any()))
        self.assertTrue(bool(result.success.any()))

    def test_every_survivor_is_inside_its_radius(self) -> None:
        """A settled, let-go episode survives only by being correctly placed."""

        for instruction, radius in (
            ("put_into_plate", PLATE_RADIUS),
            ("put_into_bowl", BOWL_RADIUS),
        ):
            for offset in (0.0, 0.03, 0.05, 0.08, 0.15, 0.30):
                with self.subTest(instruction=instruction, offset=offset):
                    result = _evaluate(
                        [_landed(instruction, offset, STILL_CLOSING)]
                    )
                    survived = not bool(
                        result.diagnostics["wrong_place_drop"][0]
                    )
                    self.assertEqual(survived, offset <= radius)


if __name__ == "__main__":
    unittest.main()
