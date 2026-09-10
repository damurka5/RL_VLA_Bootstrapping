"""The three-stage chain's transition rules, and the traps they exist to avoid.

Four properties, each of which has an already-paid-for failure behind it.

A NATIVE STAGE SUCCESS IS AN EVENT, NOT A VERDICT. ``evaluate_active_sparse_tasks``
returns ``terminated`` on a reach or a pickup, and a collector that consumed
that mask would stop the chain at its first stage. The chain's status lives in
``StageMachine`` and the stage predicates only feed it.

A RELEASE IN PROGRESS IS NOT A DROPPED OBJECT. ``physical_grasp`` includes
``~release_open``, so it goes False several env steps before ``container_ok``
can latch: at ``action_step_gripper`` 0.05 the opening needs ~11 steps to cross
the threshold. This is §7.8 of the campaign report -- a terminal condition
sharing a conjunct with success and firing because the conjunct is not
satisfied YET -- and it cost 63 of composed plate's 228 failures the last time
it appeared. A carry loss must therefore require the hand to still be CLOSED.

THE YAW SERVO MUST NOT DRIVE INTO A JOINT STOP. ``ee_yaw`` is limited to
[-pi, pi]; the shortest-angle error crosses that boundary for a third of all
starts, and a servo that commanded it would push against the stop for the whole
alignment tail while reporting a shrinking error that never shrinks.

STAGE OBSERVERS MUST NOT SHARE STATE. The predicate writes ``ever_grasped``,
``grasped``, ``peak_lift`` and ``release_clearance`` in place; a placement
observer sharing ``ever_grasped`` with the pickup evaluator would have its grasp
history rewritten under it, and ``container_ok`` requires that history.
"""

from __future__ import annotations

import math
import unittest

import torch

from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
    FAILURE_TO_ID,
    SEMANTIC_STAGE_OF,
    STAGE_ALIGN,
    STAGE_COMPLETE,
    STAGE_FAILED,
    STAGE_MOVE_TO,
    STAGE_PICK_UP,
    STAGE_PLACEMENT,
    STAGE_SETTLE,
    PickupReadiness,
    PickupYawCalibration,
    StageBudgets,
    StageMachine,
    YawTailController,
    clone_task_state,
    reachable_yaw_error,
    student_instruction_text,
    teacher_instruction_text,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    INSTRUCTION_TO_ID,
    BatchedTaskState,
)

CPU = torch.device("cpu")


def _calibration(**overrides):
    values = dict(
        target_yaw=0.0,
        tolerance_rad=math.radians(5.0),
        consecutive_decisions=2,
        safe_rotation_z=0.26,
        source="unit-test",
    )
    values.update(overrides)
    return PickupYawCalibration(**values)


def _machine(worlds=1, budgets=None, calibration=None, readiness=None):
    return StageMachine(
        torch=torch,
        device=CPU,
        worlds=worlds,
        budgets=budgets or StageBudgets(8, 8, 8),
        calibration=calibration or _calibration(),
        readiness=readiness or PickupReadiness(),
    )


def _advance(machine, decision, worlds=1, **overrides):
    false = torch.zeros(worlds, dtype=torch.bool)
    arguments = dict(
        reach_success=false.clone(),
        pickup_success=false.clone(),
        placement_success=false.clone(),
        placement_geometry_ok=false.clone(),
        wrong_place_settled=false.clone(),
        physical_grasp=false.clone(),
        released=false.clone(),
        gripper_opening=torch.ones(worlds),
        ee_position=torch.tensor([[0.0, 0.0, 0.20]] * worlds),
        # The grasp point of a resting apple: desk 0.15 + rest 0.0345 + the
        # 0.0075 pad offset. The readiness band is measured against THIS, not
        # against an absolute height -- see PickupReadiness.
        grasp_point_z=torch.full((worlds,), 0.1921),
        target_lift=torch.zeros(worlds),
        yaw_aligned=false.clone(),
        diverged=false.clone(),
    )
    arguments.update(overrides)
    return machine.advance(decision=decision, **arguments)


class StageTransitionTests(unittest.TestCase):
    def test_reach_alone_does_not_promote_without_readiness(self):
        """XY success with a closed hand is not a usable pickup pose."""

        machine = _machine()
        _advance(
            machine,
            0,
            reach_success=torch.tensor([True]),
            gripper_opening=torch.tensor([0.2]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_MOVE_TO)

        machine = _machine()
        _advance(
            machine,
            0,
            reach_success=torch.tensor([True]),
            ee_position=torch.tensor([[0.0, 0.0, 0.45]]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_MOVE_TO)

        machine = _machine()
        _advance(machine, 0, reach_success=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)

    def test_readiness_follows_the_object_not_an_absolute_height(self):
        """The gate that scored 0 of 64 on the first teacher screen.

        The pickup teacher's own aligned start is one centimetre above the
        grasp point, which for these catalogs is 0.195-0.202 m. An absolute
        0.20 m floor rejects three of the four objects at the height the pickup
        teacher was TRAINED to begin from, and under sparse_binary_reward
        nothing in the move-to reward pushes the policy up to compensate.
        """

        # A tall object: grasp point 0.25, gripper hovering one centimetre
        # above it. Correct, and an absolute [0.20, 0.34] band would also have
        # accepted this one -- which is why the bug survived a spot check.
        machine = _machine()
        _advance(
            machine,
            0,
            reach_success=torch.tensor([True]),
            ee_position=torch.tensor([[0.0, 0.0, 0.26]]),
            grasp_point_z=torch.tensor([0.25]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)

        # A short object: grasp point 0.1856 (tomato), gripper one centimetre
        # above it at 0.1956. Also correct, and the absolute floor rejected it.
        machine = _machine()
        _advance(
            machine,
            0,
            reach_success=torch.tensor([True]),
            ee_position=torch.tensor([[0.0, 0.0, 0.1956]]),
            grasp_point_z=torch.tensor([0.1856]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)

        # Far above the object is still refused: that is what the band is for.
        machine = _machine()
        _advance(
            machine,
            0,
            reach_success=torch.tensor([True]),
            ee_position=torch.tensor([[0.0, 0.0, 0.33]]),
            grasp_point_z=torch.tensor([0.1856]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_MOVE_TO)

        # And so is a pose below the controller floor, which is a diverged
        # world rather than a low reach.
        machine = _machine()
        _advance(
            machine,
            0,
            reach_success=torch.tensor([True]),
            ee_position=torch.tensor([[0.0, 0.0, 0.10]]),
            grasp_point_z=torch.tensor([0.105]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_MOVE_TO)

    def test_alignment_requires_consecutive_decisions(self):
        """One decision inside tolerance is a world swinging through it."""

        machine = _machine(calibration=_calibration(consecutive_decisions=2))
        _advance(machine, 0, reach_success=torch.tensor([True]))
        _advance(machine, 1, yaw_aligned=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)
        # A miss resets the streak rather than accumulating toward the bar.
        _advance(machine, 2, yaw_aligned=torch.tensor([False]))
        self.assertEqual(int(machine.aligned_streak[0]), 0)
        _advance(machine, 3, yaw_aligned=torch.tensor([True]))
        _advance(machine, 4, yaw_aligned=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_PICK_UP)

    def test_pickup_handoff_requires_still_holding_at_the_boundary(self):
        machine = _machine()
        _advance(machine, 0, reach_success=torch.tensor([True]))
        _advance(machine, 1, yaw_aligned=torch.tensor([True]))
        _advance(machine, 2, yaw_aligned=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_PICK_UP)
        # The predicate fired somewhere inside the chunk but the object is no
        # longer between the fingers at the boundary: there is no endpoint to
        # hand over, so the placement teacher must not be started empty-handed.
        _advance(
            machine,
            3,
            pickup_success=torch.tensor([True]),
            physical_grasp=torch.tensor([False]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_PICK_UP)
        _advance(
            machine,
            4,
            pickup_success=torch.tensor([True]),
            physical_grasp=torch.tensor([True]),
            gripper_opening=torch.tensor([0.3]),
            target_lift=torch.tensor([0.061]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_PLACEMENT)
        self.assertAlmostEqual(float(machine.handoff_lift[0]), 0.061, places=5)

    def test_release_in_progress_is_not_a_carry_loss(self):
        """The §7.8 trap: a conjunct of success used as a terminal condition."""

        machine = _machine()
        _advance(machine, 0, reach_success=torch.tensor([True]))
        _advance(machine, 1, yaw_aligned=torch.tensor([True]))
        _advance(machine, 2, yaw_aligned=torch.tensor([True]))
        _advance(
            machine,
            3,
            pickup_success=torch.tensor([True]),
            physical_grasp=torch.tensor([True]),
            target_lift=torch.tensor([0.06]),
        )
        # Gripper opening, object no longer "grasped", success not latched yet.
        for decision in range(4, 7):
            _advance(
                machine,
                decision,
                physical_grasp=torch.tensor([False]),
                released=torch.tensor([True]),
                gripper_opening=torch.tensor([0.6]),
            )
            self.assertEqual(int(machine.failure[0]), 0)
            self.assertEqual(int(machine.stage[0]), STAGE_PLACEMENT)

    def test_slip_with_a_closed_hand_is_a_carry_loss(self):
        machine = _machine()
        _advance(machine, 0, reach_success=torch.tensor([True]))
        _advance(machine, 1, yaw_aligned=torch.tensor([True]))
        _advance(machine, 2, yaw_aligned=torch.tensor([True]))
        _advance(
            machine,
            3,
            pickup_success=torch.tensor([True]),
            physical_grasp=torch.tensor([True]),
            target_lift=torch.tensor([0.06]),
        )
        _advance(
            machine,
            4,
            physical_grasp=torch.tensor([False]),
            released=torch.tensor([False]),
            gripper_opening=torch.tensor([0.3]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_FAILED)
        self.assertEqual(
            int(machine.failure[0]), FAILURE_TO_ID["carry_loss"]
        )

    def test_pickup_regrasp_is_counted_not_fatal(self):
        """Regrasp is rejected during TRANSPORT, not during the grasp itself."""

        machine = _machine()
        _advance(machine, 0, reach_success=torch.tensor([True]))
        _advance(machine, 1, yaw_aligned=torch.tensor([True]))
        _advance(machine, 2, yaw_aligned=torch.tensor([True]))
        _advance(machine, 3, physical_grasp=torch.tensor([True]))
        _advance(machine, 4, physical_grasp=torch.tensor([False]))
        self.assertEqual(int(machine.stage[0]), STAGE_PICK_UP)
        self.assertEqual(int(machine.failure[0]), 0)
        self.assertEqual(int(machine.pickup_regrasp[0]), 1)

    def test_premature_grasp_ends_the_chain(self):
        machine = _machine()
        _advance(machine, 0, physical_grasp=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_FAILED)
        self.assertEqual(
            int(machine.failure[0]),
            FAILURE_TO_ID["premature_grasp_before_handoff"],
        )

    def test_move_budget_covers_the_alignment_tail(self):
        """The tail is inside move-to and does not get a fresh budget."""

        machine = _machine(budgets=StageBudgets(3, 8, 8))
        _advance(machine, 0, reach_success=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)
        for decision in range(1, 4):
            _advance(machine, decision)
        self.assertEqual(int(machine.stage[0]), STAGE_FAILED)
        self.assertEqual(
            int(machine.failure[0]), FAILURE_TO_ID["align_budget_exhausted"]
        )

    def test_success_beats_an_exhausted_budget_on_the_same_boundary(self):
        machine = _machine(budgets=StageBudgets(8, 8, 1))
        _advance(machine, 0, reach_success=torch.tensor([True]))
        _advance(machine, 1, yaw_aligned=torch.tensor([True]))
        _advance(machine, 2, yaw_aligned=torch.tensor([True]))
        _advance(
            machine,
            3,
            pickup_success=torch.tensor([True]),
            physical_grasp=torch.tensor([True]),
            target_lift=torch.tensor([0.06]),
        )
        _advance(
            machine,
            4,
            placement_success=torch.tensor([True]),
            placement_geometry_ok=torch.tensor([True]),
            released=torch.tensor([True]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_COMPLETE)
        self.assertEqual(int(machine.failure[0]), 0)

    def test_settle_window_can_reject_a_placement_that_rolls_out(self):
        machine = _machine(budgets=StageBudgets(8, 8, 8, settle_decisions=2))
        _advance(machine, 0, reach_success=torch.tensor([True]))
        _advance(machine, 1, yaw_aligned=torch.tensor([True]))
        _advance(machine, 2, yaw_aligned=torch.tensor([True]))
        _advance(
            machine,
            3,
            pickup_success=torch.tensor([True]),
            physical_grasp=torch.tensor([True]),
            target_lift=torch.tensor([0.06]),
        )
        _advance(
            machine,
            4,
            placement_success=torch.tensor([True]),
            placement_geometry_ok=torch.tensor([True]),
            released=torch.tensor([True]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_SETTLE)
        _advance(machine, 5, placement_geometry_ok=torch.tensor([False]))
        self.assertEqual(int(machine.stage[0]), STAGE_FAILED)
        self.assertEqual(
            int(machine.failure[0]), FAILURE_TO_ID["settle_lost"]
        )

    def test_settle_window_completes_when_the_object_stays(self):
        machine = _machine(budgets=StageBudgets(8, 8, 8, settle_decisions=2))
        _advance(machine, 0, reach_success=torch.tensor([True]))
        _advance(machine, 1, yaw_aligned=torch.tensor([True]))
        _advance(machine, 2, yaw_aligned=torch.tensor([True]))
        _advance(
            machine,
            3,
            pickup_success=torch.tensor([True]),
            physical_grasp=torch.tensor([True]),
            target_lift=torch.tensor([0.06]),
        )
        _advance(
            machine,
            4,
            placement_success=torch.tensor([True]),
            placement_geometry_ok=torch.tensor([True]),
            released=torch.tensor([True]),
        )
        for decision in (5, 6):
            _advance(
                machine,
                decision,
                placement_geometry_ok=torch.tensor([True]),
                released=torch.tensor([True]),
            )
        self.assertEqual(int(machine.stage[0]), STAGE_COMPLETE)

    def test_alignment_is_a_move_to_substage(self):
        self.assertEqual(SEMANTIC_STAGE_OF[STAGE_ALIGN], "move_to")
        self.assertEqual(SEMANTIC_STAGE_OF[STAGE_MOVE_TO], "move_to")
        self.assertEqual(SEMANTIC_STAGE_OF[STAGE_SETTLE], "placement")


class YawTests(unittest.TestCase):
    def test_shortest_path_is_refused_when_it_crosses_the_joint_stop(self):
        current = torch.tensor([3.10])
        error = reachable_yaw_error(
            torch, current, target=-3.10, limits=(-math.pi, math.pi)
        )
        # The short way is +0.083 rad and lands at 3.183, past the +pi stop.
        self.assertLess(float(error[0]), 0.0)
        destination = float(current[0] + error[0])
        self.assertGreaterEqual(destination, -math.pi - 1e-6)
        self.assertLessEqual(destination, math.pi + 1e-6)
        self.assertAlmostEqual(destination, -3.10, places=5)

    def test_ordinary_errors_take_the_short_way(self):
        error = reachable_yaw_error(
            torch,
            torch.tensor([0.5]),
            target=-0.5,
            limits=(-math.pi, math.pi),
        )
        self.assertAlmostEqual(float(error[0]), -1.0, places=6)

    def test_the_tail_holds_xy_and_bridges_a_low_reach(self):
        servo = YawTailController(
            torch=torch,
            calibration=_calibration(safe_rotation_z=0.26),
            action_step_yaw=0.08,
            action_step_xyz=0.015,
        )
        low = servo.actions(
            ee_position=torch.tensor([[0.1, -0.05, 0.20]]),
            ee_yaw=torch.tensor([1.0]),
        )
        # No XY command at all: the reach is finished and the tail must not
        # move the gripper off the object it just approached.
        self.assertEqual(float(low[0, 0]), 0.0)
        self.assertEqual(float(low[0, 1]), 0.0)
        # 6 cm below the safe height saturates the upward bridge.
        self.assertAlmostEqual(float(low[0, 2]), 1.0, places=6)
        # Rotating the wrong way would be worse than not rotating.
        self.assertLess(float(low[0, 3]), 0.0)
        self.assertEqual(float(low[0, 4]), 0.0)

        high = servo.actions(
            ee_position=torch.tensor([[0.1, -0.05, 0.30]]),
            ee_yaw=torch.tensor([0.01]),
        )
        self.assertEqual(float(high[0, 2]), 0.0)
        # Inside a fifth of one yaw step, the command is proportional and small.
        self.assertLess(abs(float(high[0, 3])), 0.2)

    def test_a_calibration_without_provenance_is_refused(self):
        with self.assertRaises(ValueError):
            PickupYawCalibration(target_yaw=0.0).validate()
        with self.assertRaises(ValueError):
            PickupYawCalibration(
                target_yaw=4.0, source="test"
            ).validate()


class ObserverIsolationTests(unittest.TestCase):
    def _state(self, worlds=2):
        return BatchedTaskState(
            instruction_ids=torch.full(
                (worlds,), INSTRUCTION_TO_ID["put_into_plate"], dtype=torch.int64
            ),
            target_slots=torch.zeros(worlds, dtype=torch.int64),
            reference_slots=torch.ones(worlds, dtype=torch.int64),
            second_reference_slots=torch.full(
                (worlds,), -1, dtype=torch.int64
            ),
            initial_target_positions=torch.zeros((worlds, 3)),
            ever_grasped=torch.ones(worlds, dtype=torch.bool),
            grasped=torch.ones(worlds, dtype=torch.bool),
            step_count=torch.full((worlds,), 7, dtype=torch.int64),
            release_threshold=torch.full((worlds,), 0.55),
            support_surface_z=torch.full((worlds,), 0.15),
            target_rest_height=torch.full((worlds,), 0.03),
            peak_lift=torch.full((worlds,), 0.04),
            release_clearance=torch.zeros(worlds),
        )

    def test_clone_does_not_alias_any_mutable_field(self):
        source = self._state()
        clone = clone_task_state(
            torch, source, instruction_id=INSTRUCTION_TO_ID["pick_up"]
        )
        clone.ever_grasped.fill_(False)
        clone.grasped.fill_(False)
        clone.initial_target_positions.fill_(9.0)
        clone.release_threshold.fill_(0.1)

        self.assertTrue(bool(source.ever_grasped.all()))
        self.assertTrue(bool(source.grasped.all()))
        self.assertEqual(float(source.initial_target_positions.max()), 0.0)
        self.assertAlmostEqual(float(source.release_threshold[0]), 0.55, places=6)

    def test_clone_restarts_the_stage_local_counters(self):
        clone = clone_task_state(
            torch, self._state(), instruction_id=INSTRUCTION_TO_ID["pick_up"]
        )
        self.assertEqual(int(clone.step_count.max()), 0)
        self.assertEqual(float(clone.peak_lift.max()), 0.0)
        self.assertTrue(bool(torch.isnan(clone.release_clearance).all()))
        self.assertEqual(
            int(clone.instruction_ids[0]), INSTRUCTION_TO_ID["pick_up"]
        )


class InstructionTextTests(unittest.TestCase):
    def test_the_plate_teacher_keeps_its_own_wording(self):
        self.assertEqual(
            teacher_instruction_text("placement", "apple", "plate"),
            "put apple on the plate",
        )
        self.assertEqual(
            teacher_instruction_text("placement", "apple", "bowl"),
            "put apple into bowl",
        )

    def test_the_student_prompt_is_the_requested_wording(self):
        self.assertEqual(
            student_instruction_text("apple", "plate"), "put apple into plate"
        )
        self.assertEqual(
            student_instruction_text("apple", "bowl"), "put apple into bowl"
        )

    def test_teacher_and_student_plate_wording_differ_on_purpose(self):
        self.assertNotEqual(
            teacher_instruction_text("placement", "apple", "plate"),
            student_instruction_text("apple", "plate"),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
