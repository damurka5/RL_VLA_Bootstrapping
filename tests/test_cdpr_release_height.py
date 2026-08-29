"""The release-height gate that separates a placement from a drop.

``container_ok`` already requires the object to have SETTLED in the receptacle,
so a drop and a placement reach the same terminal state and the success test
cannot tell them apart. The discriminator is the object's clearance at the one
step the gripper opened, which is why it is latched rather than recomputed.

These cover the latch, because that is where the failure is silent: a gate that
re-read the clearance every step would deny nothing (the object is low by then)
and a gate that latched at reset would deny everything.
"""

from __future__ import annotations

import unittest

import torch

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    INSTRUCTION_TO_ID,
    BatchedCatchReleaseDenseReward,
    BatchedTaskState,
    BatchedTaskThresholds,
    evaluate_active_sparse_tasks,
)

PLATE_Z = 0.15
REST = 0.02


def make_state(count: int, *, latch: bool = True) -> BatchedTaskState:
    return BatchedTaskState(
        instruction_ids=torch.full(
            (count,), INSTRUCTION_TO_ID["put_into_plate"], dtype=torch.int64
        ),
        target_slots=torch.zeros((count,), dtype=torch.int64),
        reference_slots=torch.ones((count,), dtype=torch.int64),
        second_reference_slots=torch.full((count,), -1, dtype=torch.int64),
        initial_target_positions=torch.tensor(
            [[0.0, 0.0, PLATE_Z + 0.10]] * count
        ),
        ever_grasped=torch.ones((count,), dtype=torch.bool),
        grasped=torch.zeros((count,), dtype=torch.bool),
        step_count=torch.zeros((count,), dtype=torch.int64),
        release_threshold=torch.full((count,), 0.55),
        support_surface_z=torch.full((count,), PLATE_Z),
        target_rest_height=torch.full((count,), REST),
        peak_lift=torch.zeros((count,), dtype=torch.float32),
        release_clearance=(
            torch.full((count,), float("nan"))
            if latch
            else None
        ),
    )


def step(state, *, object_z, gripper, max_height):
    count = int(state.instruction_ids.shape[0])
    objects = torch.zeros((count, 2, 3))
    objects[:, 0, 2] = object_z          # the carried object
    objects[:, 1, 2] = PLATE_Z           # the plate
    ee = objects[:, 0].clone()
    ee[:, 2] += 0.0075
    dense = BatchedCatchReleaseDenseReward(release_max_height=max_height)
    return evaluate_active_sparse_tasks(
        state=state,
        ee_position=ee,
        object_positions=objects,
        gripper_opening=torch.full((count,), float(gripper)),
        caught_target=torch.zeros((count,), dtype=torch.bool),
        active_mask=torch.ones((count,), dtype=torch.bool),
        max_steps=128,
        thresholds=BatchedTaskThresholds(),
        catch_release_dense_reward=dense,
    )


# A drop: released 10 cm above the plate. A placement: released 3 cm above,
# which is still clear of the object's own 2 cm resting height.
DROP_Z = PLATE_Z + 0.10
PLACE_Z = PLATE_Z + 0.03
SETTLED_Z = PLATE_Z + REST


class ReleaseHeightGateTests(unittest.TestCase):
    def test_disabled_by_default_a_drop_still_scores(self) -> None:
        # Every run before this gate existed must behave identically.
        state = make_state(1)
        step(state, object_z=DROP_Z, gripper=0.9, max_height=0.0)
        result = step(state, object_z=SETTLED_Z, gripper=0.9, max_height=0.0)
        self.assertTrue(bool(result.success[0]))

    def test_armed_a_drop_is_denied_after_it_settles(self) -> None:
        # THE regression this exists for. The object ends resting in the plate,
        # so every other condition passes; only the latched release height
        # separates it from a placement.
        state = make_state(1)
        step(state, object_z=DROP_Z, gripper=0.9, max_height=0.05)
        result = step(state, object_z=SETTLED_Z, gripper=0.9, max_height=0.05)
        self.assertFalse(bool(result.success[0]))
        self.assertAlmostEqual(float(state.release_clearance[0]), 0.10, places=5)

    def test_armed_a_placement_scores(self) -> None:
        state = make_state(1)
        step(state, object_z=PLACE_Z, gripper=0.9, max_height=0.05)
        result = step(state, object_z=SETTLED_Z, gripper=0.9, max_height=0.05)
        self.assertTrue(bool(result.success[0]))
        self.assertAlmostEqual(float(state.release_clearance[0]), 0.03, places=5)

    def test_the_latch_is_written_once(self) -> None:
        # A gate that re-read the clearance would deny nothing, because by the
        # time the episode ends the object is always low.
        state = make_state(1)
        step(state, object_z=DROP_Z, gripper=0.9, max_height=0.05)
        first = float(state.release_clearance[0])
        for z in (PLATE_Z + 0.06, SETTLED_Z, SETTLED_Z):
            step(state, object_z=z, gripper=0.9, max_height=0.05)
        self.assertAlmostEqual(float(state.release_clearance[0]), first, places=6)

    def test_nothing_latches_while_the_gripper_is_closed(self) -> None:
        state = make_state(1)
        step(state, object_z=DROP_Z, gripper=0.1, max_height=0.05)
        self.assertTrue(torch.isnan(state.release_clearance[0]))

    def test_a_world_that_never_releases_fails_the_gate(self) -> None:
        # NaN compares false, so this needs no separate branch -- but that is
        # exactly the kind of thing that breaks silently if the comparison is
        # ever rewritten.
        state = make_state(1)
        result = step(state, object_z=SETTLED_Z, gripper=0.1, max_height=0.05)
        self.assertFalse(bool(result.success[0]))

    def test_worlds_latch_independently(self) -> None:
        # Two worlds, one dropping and one placing, in the same batch.
        state = make_state(2)
        objects = torch.zeros((2, 2, 3))
        objects[0, 0, 2] = DROP_Z
        objects[1, 0, 2] = PLACE_Z
        objects[:, 1, 2] = PLATE_Z
        ee = objects[:, 0].clone()
        ee[:, 2] += 0.0075
        evaluate_active_sparse_tasks(
            state=state,
            ee_position=ee,
            object_positions=objects,
            gripper_opening=torch.full((2,), 0.9),
            caught_target=torch.zeros((2,), dtype=torch.bool),
            active_mask=torch.ones((2,), dtype=torch.bool),
            max_steps=128,
            thresholds=BatchedTaskThresholds(),
            catch_release_dense_reward=BatchedCatchReleaseDenseReward(
                release_max_height=0.05
            ),
        )
        result = step(state, object_z=SETTLED_Z, gripper=0.9, max_height=0.05)
        self.assertFalse(bool(result.success[0]))
        self.assertTrue(bool(result.success[1]))

    def test_a_caller_without_the_buffer_keeps_the_old_behaviour(self) -> None:
        # release_clearance=None predates this gate and must stay permissive.
        state = make_state(1, latch=False)
        step(state, object_z=DROP_Z, gripper=0.9, max_height=0.05)
        result = step(state, object_z=SETTLED_Z, gripper=0.9, max_height=0.05)
        self.assertTrue(bool(result.success[0]))


class ConfigPlumbingTests(unittest.TestCase):
    def test_the_knob_reaches_the_dense_config(self) -> None:
        dense = BatchedCatchReleaseDenseReward.from_metadata(
            {"put_release_max_height": 0.06}
        )
        self.assertAlmostEqual(dense.release_max_height, 0.06)

    def test_it_defaults_to_disabled(self) -> None:
        self.assertAlmostEqual(
            BatchedCatchReleaseDenseReward.from_metadata({}).release_max_height,
            0.0,
        )

    def test_a_negative_value_is_clamped_to_disabled(self) -> None:
        dense = BatchedCatchReleaseDenseReward.from_metadata(
            {"put_release_max_height": -0.5}
        )
        self.assertAlmostEqual(dense.release_max_height, 0.0)


if __name__ == "__main__":
    unittest.main()


class PerFamilyThresholdTests(unittest.TestCase):
    """Plate and bowl need different limits; one knob wastes the bowl's room.

    Measured over the bank: an object at rest sits 0.058 m above the plate's
    centre and 0.042 above the bowl's. A shared threshold has to clear the
    plate's tallest object, which leaves the bowl barely constrained.
    """

    def _run(self, instruction, object_z, *, plate_max, bowl_max):
        state = make_state(1)
        state.instruction_ids.fill_(INSTRUCTION_TO_ID[instruction])
        objects = torch.zeros((1, 2, 3))
        objects[:, 1, 2] = PLATE_Z
        dense = BatchedCatchReleaseDenseReward(
            plate_release_max_height=plate_max,
            bowl_release_max_height=bowl_max,
        )

        def one(z):
            objects[:, 0, 2] = z
            ee = objects[:, 0].clone()
            ee[:, 2] += 0.0075
            return evaluate_active_sparse_tasks(
                state=state,
                ee_position=ee,
                object_positions=objects,
                gripper_opening=torch.full((1,), 0.9),
                caught_target=torch.zeros((1,), dtype=torch.bool),
                active_mask=torch.ones((1,), dtype=torch.bool),
                max_steps=128,
                thresholds=BatchedTaskThresholds(),
                catch_release_dense_reward=dense,
            )

        one(object_z)
        return one(SETTLED_Z)

    def test_the_bowl_limit_binds_where_the_plate_limit_would_not(self) -> None:
        # A release at 0.07: allowed for the plate (0.080), denied for the
        # bowl (0.065). One shared threshold could not express this.
        z = PLATE_Z + 0.07
        plate = self._run("put_into_plate", z, plate_max=0.080, bowl_max=0.065)
        bowl = self._run("put_into_bowl", z, plate_max=0.080, bowl_max=0.065)
        self.assertTrue(bool(plate.success[0]))
        self.assertFalse(bool(bowl.success[0]))

    def test_the_measured_release_height_is_denied_for_both(self) -> None:
        # 0.0867 is what phase5_placement_iter3 actually released at.
        z = PLATE_Z + 0.0867
        for name in ("put_into_plate", "put_into_bowl"):
            with self.subTest(name):
                result = self._run(name, z, plate_max=0.080, bowl_max=0.065)
                self.assertFalse(bool(result.success[0]))

    def test_a_settled_object_is_never_denied_by_its_own_height(self) -> None:
        # The failure that would look like the task breaking: a threshold
        # below the resting clearance denies correct placements. Measured
        # maxima are 0.0690 (plate) and 0.0523 (bowl).
        for name, rest_max in (
            ("put_into_plate", 0.0690), ("put_into_bowl", 0.0523)
        ):
            with self.subTest(name):
                result = self._run(
                    name, PLATE_Z + rest_max, plate_max=0.080, bowl_max=0.065
                )
                self.assertTrue(bool(result.success[0]))

    def test_a_family_override_falls_back_to_the_shared_knob(self) -> None:
        dense = BatchedCatchReleaseDenseReward.from_metadata(
            {"put_release_max_height": 0.09}
        )
        self.assertAlmostEqual(dense.release_max_height, 0.09)
        self.assertAlmostEqual(dense.plate_release_max_height, 0.0)

    def test_the_family_knobs_reach_the_config(self) -> None:
        dense = BatchedCatchReleaseDenseReward.from_metadata(
            {
                "put_plate_release_max_height": 0.080,
                "put_bowl_release_max_height": 0.065,
            }
        )
        self.assertAlmostEqual(dense.plate_release_max_height, 0.080)
        self.assertAlmostEqual(dense.bowl_release_max_height, 0.065)
