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
from dataclasses import replace

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
        # Centred over the object, well inside an apple's 0.0130 m slack.
        target_xy_error=torch.full((worlds,), 0.004),
        max_grasp_xy_offset=torch.full((worlds,), 0.0130),
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

    def test_a_single_consecutive_decision_promotes_when_calibrated_to_one(self):
        """The bar is a calibration value, not a constant of the machine.

        The 0.50 descent screen measured 14 of 27 aligning worlds reaching the
        full readiness conjunction at some boundary while only 6 held it for
        two, with max_ready_streak p90 exactly 2.0 -- the requirement sits on
        the edge of the distribution, so it is worth being able to move it.
        Unlike the rejected clearance handoff, this does NOT change the pose
        that is handed over, only how long it must persist.
        """

        machine = _machine(calibration=_calibration(consecutive_decisions=1))
        _advance(machine, 0, reach_success=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)
        _advance(machine, 1, yaw_aligned=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_PICK_UP)

    def test_one_ready_boundary_is_not_enough_at_the_default_bar(self):
        """The same trace under the shipped calibration stays in alignment, so
        the test above is measuring the knob and not a coincidence."""

        machine = _machine(calibration=_calibration(consecutive_decisions=2))
        _advance(machine, 0, reach_success=torch.tensor([True]))
        _advance(machine, 1, yaw_aligned=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)

    def test_the_bar_still_requires_the_rest_of_the_conjunction(self):
        """Lowering it to one must not turn the handoff into a yaw-only test."""

        machine = _machine(calibration=_calibration(consecutive_decisions=1))
        _advance(machine, 0, reach_success=torch.tensor([True]))
        _advance(
            machine,
            1,
            yaw_aligned=torch.tensor([True]),
            # Inside the reach window, outside an apple's lateral slack.
            target_xy_error=torch.tensor([0.017]),
            max_grasp_xy_offset=torch.tensor([0.0130]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)

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

    def test_pickup_handoff_requires_lift_at_boundary_not_just_latched_success(self):
        machine = _machine()
        machine.stage[:] = STAGE_PICK_UP
        _advance(machine, 0, pickup_success=torch.tensor([True]),
                 physical_grasp=torch.tensor([True]), target_lift=torch.tensor([0.049]))
        self.assertEqual(int(machine.stage[0]), STAGE_PICK_UP)
        self.assertEqual(int(machine.pickup_event[0]), -1)
        _advance(machine, 1, pickup_success=torch.tensor([True]),
                 physical_grasp=torch.tensor([True]), target_lift=torch.tensor([0.051]))
        self.assertEqual(int(machine.stage[0]), STAGE_PLACEMENT)
        self.assertAlmostEqual(float(machine.handoff_lift[0]), .051, places=6)

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

    def test_contact_loss_during_goal_opening_can_cross_a_decision_boundary(self):
        from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import release_opening_over_goal
        opening_release = release_opening_over_goal(
            command=torch.tensor([.75347]), opening=torch.tensor([.45603]),
            previous_opening=torch.tensor([.42498]), target_xy=torch.tensor([[.007, 0.]]),
            receptacle_xy=torch.zeros(1, 2), radius=torch.tensor([.057]),
        )
        machine = _machine()
        machine.stage[:] = STAGE_PLACEMENT
        _advance(machine, 0, physical_grasp=torch.tensor([False]),
                 released=torch.tensor([False]), release_in_progress=opening_release)
        self.assertEqual(int(machine.stage[0]), STAGE_PLACEMENT)
        self.assertEqual(int(machine.failure[0]), 0)
        # A discontinued opening is not an indefinite exemption from carry loss.
        _advance(machine, 1, physical_grasp=torch.tensor([False]),
                 released=torch.tensor([False]), release_in_progress=torch.tensor([False]))
        self.assertEqual(int(machine.failure[0]), FAILURE_TO_ID['carry_loss'])

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

    def test_the_loop_budget_covers_the_sum_of_the_stage_caps(self):
        """A loop shorter than the worst case truncates chains invisibly.

        `stage_decisions` resets at every transition, so the alignment tail
        always got a fresh `move_decisions` however long the reach took. With
        the tail uncounted, the worst case was 32 + 32 + 32 + 64 = 160 against
        a 128-decision loop, and a chain that ran past it stopped with no
        failure code and no acceptance -- invisible in every census, because
        nothing decided anything about it. Measured: 6 of 64 worlds per move-to
        screen.
        """

        budgets = StageBudgets(32, 32, 64)
        self.assertEqual(budgets.align_budget, 32)
        self.assertEqual(budgets.total_decisions, 32 + 32 + 32 + 64)

        explicit = StageBudgets(32, 32, 64, align_decisions=12, settle_decisions=4)
        self.assertEqual(explicit.align_budget, 12)
        self.assertEqual(
            explicit.total_decisions, 32 + 12 + 32 + 64 + 4
        )

    def test_the_tail_gets_its_own_cap_not_a_share_of_the_move_budget(self):
        machine = _machine(budgets=StageBudgets(3, 8, 8, align_decisions=6))
        _advance(machine, 0, reach_success=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)
        # Five more decisions of tail: still inside its own six-decision cap,
        # even though the move budget was three.
        for decision in range(1, 6):
            _advance(machine, decision)
            self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)
        _advance(machine, 6)
        self.assertEqual(int(machine.stage[0]), STAGE_FAILED)
        self.assertEqual(
            int(machine.failure[0]), FAILURE_TO_ID["align_budget_exhausted"]
        )

    def test_the_tail_defaults_to_the_move_cap_and_is_counted(self):
        """Default: same cap as move-to, but its OWN counter, and in the total.

        The tail is a substage of move-to for labelling and for the sampler,
        and not for budgeting -- `stage_decisions` resets at the transition.
        Pretending otherwise is what made the loop shorter than the worst case.
        """

        budgets = StageBudgets(3, 8, 8)
        self.assertEqual(budgets.align_budget, 3)
        self.assertEqual(budgets.total_decisions, 3 + 3 + 8 + 8)

        machine = _machine(budgets=budgets)
        _advance(machine, 0, reach_success=torch.tensor([True]))
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)
        for decision in range(1, 4):
            _advance(machine, decision)
            if decision < 3:
                self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)
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


class YawServoDampingTests(unittest.TestCase):
    """Why the tail rings instead of promoting.

    The command is recomputed every ACTION and the plant integrates it:
    `setpoint += a3 * action_step_yaw` with `a3 = error / action_step_yaw`
    means the setpoint absorbs the FULL measured error four times per decision
    while a kp=30 actuator through a damped ball joint is still travelling.

    Measured on the clearance-handoff arm, where centring held 0.0039 m and
    there were zero descent aborts: median yaw error 0.0824 rad against an
    0.0873 rad band -- 94% of tolerance -- with 47% of tail steps outside it
    and 0 of 10 chains promoted.
    """

    def _servo(self, gain):
        return YawTailController(
            torch=torch,
            calibration=_calibration(),
            action_step_yaw=0.08,
            action_step_xyz=0.015,
            yaw_servo_gain=gain,
        )

    def test_damping_scales_the_small_signal_command(self):
        error = torch.tensor([0.04])
        undamped = float(self._servo(1.0).yaw_command(error)[0])
        damped = float(self._servo(0.35).yaw_command(error)[0])
        # Sign is toward the target either way; magnitude is reduced.
        self.assertLess(abs(damped), abs(undamped))
        self.assertAlmostEqual(damped, 0.35 * undamped, places=6)
        self.assertLess(damped, 0.0)

    def test_a_large_error_still_saturates(self):
        """The initial rotation must not be slowed by the damping."""

        far = torch.tensor([1.73])
        self.assertAlmostEqual(
            float(self._servo(0.35).yaw_command(far)[0]), -1.0, places=6
        )

    def test_an_out_of_range_gain_is_refused(self):
        for gain in (0.0, -0.2, 1.5):
            with self.assertRaises(ValueError):
                self._servo(gain)

    def test_unity_gain_reproduces_the_undamped_servo(self):
        error = torch.tensor([0.04])
        self.assertAlmostEqual(
            float(self._servo(1.0).yaw_command(error)[0]),
            float(-0.04 / 0.08),
            places=6,
        )


class LateralReadinessTests(unittest.TestCase):
    """The gate the third screen was missing, and the geometry behind it.

    The production move_to success window is 0.02 m. The open gripper's
    half-aperture is 0.0475 m, so the lateral slack is 0.0130 m for an apple
    and 0.0185 m for the others -- the reach window is WIDER than the grasp
    tolerance. Measured: handoff XY error 0.0166-0.0186 m, the descent stopping
    0.053-0.059 m above the grasp point with the object never moving, and not
    one grasp in 48 chains across three screens.
    """

    def test_the_aperture_matches_the_model(self):
        """A geometry fact must not drift into a constant nobody re-derives."""

        import mujoco

        from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
            FINGER_TIP_DEPTH_M,
            OPEN_GRIPPER_HALF_APERTURE_M,
        )

        model = mujoco.MjModel.from_xml_path(
            "robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml"
        )
        data = mujoco.MjData(model)
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_base")
        for name in ("finger_l", "finger_r"):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint]] = float(
                max(model.jnt_range[joint])
            )
        mujoco.mj_forward(model, data)
        origin = data.xpos[base]

        def inner_face(geom: str) -> float:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            offset = abs(float(data.geom_xpos[index][0] - origin[0]))
            return offset - float(model.geom_size[index][0])

        for geom in ("left_finger_pad", "right_finger_pad"):
            self.assertAlmostEqual(
                inner_face(geom), OPEN_GRIPPER_HALF_APERTURE_M, places=4
            )
        tip = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_tip"
        )
        depth = origin[2] - (
            float(data.geom_xpos[tip][2]) - float(model.geom_size[tip][2])
        )
        self.assertAlmostEqual(depth, FINGER_TIP_DEPTH_M, places=4)

    def test_the_slack_is_narrower_than_the_reach_window(self):
        from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
            max_grasp_xy_offset,
        )

        # The production move_to success window, for comparison.
        reach_window = 0.02
        apple = max_grasp_xy_offset("robocasa_apple", margin=0.0)
        orange = max_grasp_xy_offset("robocasa_orange", margin=0.0)
        self.assertAlmostEqual(apple, 0.0130, places=4)
        self.assertAlmostEqual(orange, 0.0185, places=4)
        self.assertLess(apple, reach_window)
        self.assertLess(orange, reach_window)

    def test_a_catalog_wider_than_the_aperture_reports_negative_slack(self):
        from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
            max_grasp_xy_offset,
        )

        # The geometry fact that removed banana and mug from the target pool,
        # stated as a number instead of a note.
        self.assertLess(max_grasp_xy_offset("robocasa_banana"), 0.0)

    def test_an_off_centre_reach_does_not_promote(self):
        machine = _machine()
        _advance(
            machine,
            0,
            reach_success=torch.tensor([True]),
            # Inside the 0.02 m reach window and outside an apple's slack.
            target_xy_error=torch.tensor([0.017]),
            max_grasp_xy_offset=torch.tensor([0.0130]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_MOVE_TO)

    def test_a_centred_reach_promotes(self):
        machine = _machine()
        _advance(
            machine,
            0,
            reach_success=torch.tensor([True]),
            target_xy_error=torch.tensor([0.009]),
            max_grasp_xy_offset=torch.tensor([0.0130]),
        )
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)


class XYCentringBridgeTests(unittest.TestCase):
    """The bridge that closes the reach's lateral error, and its sequencing.

    Measured on the teacher screen: 57-60 of 64 chains die at
    move_budget_exhausted because the reach lands 0.0166-0.0186 m off against a
    0.0130-0.0185 m lateral tolerance. The bridge exists to close exactly that
    error, and it is off by default because it is the one part of the tail that
    reads privileged geometry.
    """

    def _servo(self, **overrides):
        values = dict(
            torch=torch,
            calibration=_calibration(safe_rotation_z=0.26),
            action_step_yaw=0.08,
            action_step_xyz=0.015,
            action_step_gripper=0.05,
            pickup_height_above_grasp=0.01,
            xy_centring_deadband=0.005,
        )
        values.update(overrides)
        return YawTailController(**values)

    def test_it_translates_toward_the_object_at_the_clearance_height(self):
        servo = self._servo()
        command = servo.actions(
            ee_position=torch.tensor([[0.00, 0.00, 0.26]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.017, 0.0]]),
        )
        # 17 mm of error against a 15 mm step saturates the X channel.
        self.assertAlmostEqual(float(command[0, 0]), 1.0, places=6)
        self.assertEqual(float(command[0, 1]), 0.0)

    def test_it_commands_nothing_inside_the_deadband(self):
        servo = self._servo()
        command = servo.actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.26]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.003, 0.0]]),
        )
        self.assertEqual(float(command[0, 0]), 0.0)
        self.assertEqual(float(command[0, 1]), 0.0)

    def test_it_does_not_translate_while_climbing(self):
        """A low lateral sweep is how a finger is dragged through the object.

        The climb is identified by the yaw NOT yet being aligned: rotation only
        happens at the clearance, so an unaligned wrist below it has not been
        up there yet. A wrist that is low AND aligned is on its way DOWN, and
        there the servo must keep correcting -- see
        test_the_lateral_servo_stays_active_through_the_descent.
        """

        servo = self._servo()
        command = servo.actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.21]]),
            ee_yaw=torch.tensor([2.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.017, 0.0]]),
        )
        self.assertEqual(float(command[0, 0]), 0.0)
        self.assertGreater(float(command[0, 2]), 0.0)

    def test_it_does_not_descend_until_it_is_centred(self):
        """The failure the whole bridge exists to prevent."""

        servo = self._servo()
        off_centre = servo.actions(
            # Yaw already aligned, so only the centring gate can hold it up.
            ee_position=torch.tensor([[0.0, 0.0, 0.26]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.017, 0.0]]),
        )
        self.assertGreaterEqual(float(off_centre[0, 2]), 0.0)
        centred = servo.actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.26]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.001, 0.0]]),
        )
        # Now it descends toward grasp_point_z + 0.01 = 0.20, from 0.26.
        self.assertLess(float(centred[0, 2]), 0.0)

    def test_the_descent_tolerates_drift_and_pauses_for_recentering(self):
        """Hysteresis plus a vertical pause prevents climb/restart chatter.

        Ordinary drift inside the wider band keeps descending. Beyond that
        band, the controller already moves laterally toward the object centre;
        it now holds Z during that correction instead of climbing all the way
        to clearance and restarting the descent.
        """

        servo = self._servo(xy_centring_abort=0.009)
        descending = servo.actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.23]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.007, 0.0]]),
        )
        self.assertLess(float(descending[0, 2]), 0.0)

        recentering = servo.actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.23]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.012, 0.0]]),
        )
        self.assertEqual(float(recentering[0, 2]), 0.0)
        self.assertGreater(float(recentering[0, 0]), 0.0)

        # Entry still needs the tight band: 7 mm at the clearance centres, it
        # does not start the descent.
        entering = servo.actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.26]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.007, 0.0]]),
        )
        self.assertEqual(float(entering[0, 2]), 0.0)
        self.assertGreater(float(entering[0, 0]), 0.0)

    def test_the_lateral_servo_stays_active_through_the_descent(self):
        """A zero XY action is not "hold position", and that was the bug.

        Under the production controller `proposed_target = ee_position + delta`,
        so a zero delta makes the setpoint CHASE the measurement: drift is
        accepted rather than corrected, and on a cable-suspended platform it
        ratchets. Measured over a 48-decision tail with the servo zeroed on the
        way down: lateral error p90 0.0294 m, 128 descent aborts across 8
        worlds, 0 of 10 chains promoted.
        """

        servo = self._servo(xy_centring_abort=0.009)
        descending = servo.actions(
            # Below the clearance with yaw aligned: mid-descent.
            ee_position=torch.tensor([[0.0, 0.0, 0.23]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.007, 0.0]]),
        )
        self.assertLess(float(descending[0, 2]), 0.0)
        # And it is still correcting, toward the centre -- which is away from
        # whichever finger is closest, so it is the safe direction.
        self.assertGreater(float(descending[0, 0]), 0.0)

    def test_descent_gain_slows_only_the_vertical_bridge(self):
        fast = self._servo(descent_gain=1.0).actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.23]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.001, 0.0]]),
        )
        slow = self._servo(descent_gain=0.20).actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.23]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.001, 0.0]]),
        )
        self.assertAlmostEqual(float(fast[0, 2]), -1.0, places=6)
        self.assertAlmostEqual(float(slow[0, 2]), -0.4, places=6)
        self.assertEqual(float(slow[0, 0]), float(fast[0, 0]))
        self.assertEqual(float(slow[0, 3]), float(fast[0, 3]))

    def test_the_initial_climb_still_does_not_translate(self):
        servo = self._servo()
        climbing = servo.actions(
            # Below the clearance and yaw NOT aligned: the first ascent.
            ee_position=torch.tensor([[0.0, 0.0, 0.21]]),
            ee_yaw=torch.tensor([2.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.017, 0.0]]),
        )
        self.assertGreater(float(climbing[0, 2]), 0.0)
        self.assertEqual(float(climbing[0, 0]), 0.0)
        self.assertEqual(float(climbing[0, 3]), 0.0)

    def test_the_clearance_arm_can_actually_promote(self):
        """The gate must not require the descent the arm removes.

        Measured cost of getting this wrong: two full screens in which the yaw
        error was exactly 0.0 rad and the centring 5.6 mm, and every chain
        still died at align_budget_exhausted, because handoff_ready was
        checking a height band around the trained pickup pose that the arm
        deliberately never reaches.
        """

        at_clearance = torch.tensor([[0.0, 0.0, 0.26]])
        grasp_z = torch.tensor([0.19])
        aligned_yaw = torch.tensor([0.0])

        descending = self._servo(handoff_at_clearance=False)
        self.assertFalse(
            bool(
                descending.handoff_ready(
                    ee_yaw=aligned_yaw,
                    ee_position=at_clearance,
                    grasp_point_z=grasp_z,
                )[0]
            )
        )
        clearance = self._servo(handoff_at_clearance=True)
        self.assertTrue(
            bool(
                clearance.handoff_ready(
                    ee_yaw=aligned_yaw,
                    ee_position=at_clearance,
                    grasp_point_z=grasp_z,
                )[0]
            )
        )
        # It still requires the yaw, which is the tail's actual job.
        self.assertFalse(
            bool(
                clearance.handoff_ready(
                    ee_yaw=torch.tensor([1.5]),
                    ee_position=at_clearance,
                    grasp_point_z=grasp_z,
                )[0]
            )
        )

    def test_the_clearance_handoff_arm_never_descends(self):
        servo = self._servo(handoff_at_clearance=True)
        at_clearance = servo.actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.26]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.001, 0.0]]),
        )
        # Aligned and centred, and it holds height rather than diving.
        self.assertEqual(float(at_clearance[0, 2]), 0.0)
        # It still climbs to the clearance and still centres there.
        low = servo.actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.21]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.017, 0.0]]),
        )
        self.assertGreater(float(low[0, 2]), 0.0)
        self.assertEqual(float(low[0, 0]), 0.0)
        centring = servo.actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.26]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.017, 0.0]]),
        )
        self.assertGreater(float(centring[0, 0]), 0.0)

    def test_it_is_absent_unless_asked_for(self):
        servo = self._servo(xy_centring_deadband=None)
        command = servo.actions(
            ee_position=torch.tensor([[0.0, 0.0, 0.26]]),
            ee_yaw=torch.tensor([0.0]),
            grasp_point_z=torch.tensor([0.19]),
            target_xy=torch.tensor([[0.017, 0.0]]),
        )
        self.assertEqual(float(command[0, 0]), 0.0)
        self.assertEqual(float(command[0, 1]), 0.0)

    def test_its_actions_carry_their_own_source_code(self):
        from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
            SOURCE_ALIGN_BRIDGE,
            SOURCE_YAW_TAIL,
            _apply_overrides,
        )

        class _LowDim:
            ee_position = torch.tensor([[0.0, 0.0, 0.26]])
            ee_yaw = torch.zeros(1)
            gripper_opening = torch.ones(1)

        for target_xy, expected in (
            (torch.tensor([[0.017, 0.0]]), SOURCE_ALIGN_BRIDGE),
            (None, SOURCE_YAW_TAIL),
        ):
            _, source = _apply_overrides(
                torch,
                raw=torch.zeros((1, 5)),
                stage=torch.tensor([STAGE_ALIGN]),
                low_dim=_LowDim(),
                servo=self._servo(),
                hold_pickup=False,
                hold_placement=False,
                grasp_point_z=torch.tensor([0.19]),
                target_xy=target_xy,
            )
            self.assertEqual(int(source[0]), expected)

    def test_the_reach_gate_yields_to_the_bridge_but_the_handoff_does_not(self):
        """Gating the reach on centring would stop the bridge ever running."""

        gated = _machine(readiness=PickupReadiness(require_centred_at_reach=True))
        _advance(
            gated,
            0,
            reach_success=torch.tensor([True]),
            target_xy_error=torch.tensor([0.017]),
            max_grasp_xy_offset=torch.tensor([0.0130]),
        )
        self.assertEqual(int(gated.stage[0]), STAGE_MOVE_TO)

        bridged = _machine(
            readiness=PickupReadiness(require_centred_at_reach=False)
        )
        _advance(
            bridged,
            0,
            reach_success=torch.tensor([True]),
            target_xy_error=torch.tensor([0.017]),
            max_grasp_xy_offset=torch.tensor([0.0130]),
        )
        self.assertEqual(int(bridged.stage[0]), STAGE_ALIGN)

        # The handoff still refuses an off-centre pose, however it got here.
        for decision in (1, 2, 3):
            _advance(
                bridged,
                decision,
                yaw_aligned=torch.tensor([True]),
                target_xy_error=torch.tensor([0.017]),
                max_grasp_xy_offset=torch.tensor([0.0130]),
            )
        self.assertNotEqual(int(bridged.stage[0]), STAGE_PICK_UP)
        # Once the bridge has closed the error, it promotes.
        for decision in (4, 5):
            _advance(
                bridged,
                decision,
                yaw_aligned=torch.tensor([True]),
                target_xy_error=torch.tensor([0.004]),
                max_grasp_xy_offset=torch.tensor([0.0130]),
            )
        self.assertEqual(int(bridged.stage[0]), STAGE_PICK_UP)


class CalibrationOverrideTests(unittest.TestCase):
    """What ``--align-consecutive-decisions`` does to a loaded calibration.

    Both CLIs apply it with ``dataclasses.replace`` on the frozen calibration
    and then call ``validate()``, so the override cannot smuggle past the
    checks the calibration file itself is held to, and the value that ends up
    in the result manifest is the one that actually ran.
    """

    def test_the_override_replaces_only_the_bar(self):
        loaded = _calibration(consecutive_decisions=2, target_yaw=0.25)
        overridden = replace(loaded, consecutive_decisions=1)
        overridden.validate()
        self.assertEqual(overridden.consecutive_decisions, 1)
        self.assertEqual(overridden.target_yaw, loaded.target_yaw)
        self.assertEqual(overridden.tolerance_rad, loaded.tolerance_rad)
        self.assertEqual(overridden.source, loaded.source)

    def test_the_overridden_value_is_what_gets_serialized(self):
        """The manifest records the protocol that ran, not the file on disk."""

        overridden = replace(_calibration(), consecutive_decisions=1)
        self.assertEqual(overridden.to_json()["consecutive_decisions"], 1)

    def test_a_bar_below_one_is_still_rejected(self):
        with self.assertRaises(ValueError):
            replace(_calibration(), consecutive_decisions=0).validate()


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
        # No gripper opening supplied: the tail leaves that channel alone.
        self.assertEqual(float(low[0, 4]), 0.0)

        high = servo.actions(
            ee_position=torch.tensor([[0.1, -0.05, 0.30]]),
            ee_yaw=torch.tensor([0.01]),
        )
        self.assertEqual(float(high[0, 2]), 0.0)
        # Inside a fifth of one yaw step, the command is proportional and small.
        self.assertLess(abs(float(high[0, 3])), 0.2)

    def test_the_hand_is_held_open_and_never_squeezed(self):
        """The gate that scored 0 of 64 on the SECOND teacher screen.

        Measured: the reach predicate fired on 29-35 of 64 worlds and 100% of
        those steps arrived with the gripper already closed. Under
        sparse_binary_reward the move-to reward is where(success, 1.0, 0.0)
        with no gripper term, so that channel is unconstrained for move_to --
        and a shared four-instruction policy whose pick_up experience is all
        about closing simply closes. A closed hand cannot be handed to a pickup
        teacher whose aligned start is an open one.
        """

        servo = YawTailController(
            torch=torch,
            calibration=_calibration(),
            action_step_yaw=0.08,
            action_step_xyz=0.015,
            action_step_gripper=0.05,
        )
        # Fully closed: saturate the opening command.
        self.assertAlmostEqual(
            float(servo.open_command(torch.tensor([0.0]))[0]), 1.0, places=6
        )
        # Half a step from open: proportional.
        self.assertAlmostEqual(
            float(servo.open_command(torch.tensor([0.975]))[0]), 0.5, places=6
        )
        # Already open: no command at all. It is a hold, not control.
        self.assertEqual(
            float(servo.open_command(torch.tensor([1.0]))[0]), 0.0
        )
        # It can never squeeze, whatever it is handed.
        self.assertGreaterEqual(
            float(servo.open_command(torch.tensor([1.5]))[0]), 0.0
        )

    def test_the_override_relabels_only_the_channel_it_touched(self):
        from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
            SOURCE_GRIPPER_HOLD,
            SOURCE_TEACHER,
            SOURCE_YAW_TAIL,
            _apply_overrides,
        )

        class _LowDim:
            ee_position = torch.tensor([[0.0, 0.0, 0.27]] * 3)
            ee_yaw = torch.zeros(3)
            gripper_opening = torch.tensor([0.4, 0.4, 0.4])

        servo = YawTailController(
            torch=torch,
            calibration=_calibration(),
            action_step_yaw=0.08,
            action_step_xyz=0.015,
            action_step_gripper=0.05,
        )
        raw = torch.full((3, 5), -0.7)
        stage = torch.tensor([STAGE_MOVE_TO, STAGE_ALIGN, STAGE_PICK_UP])
        applied, source = _apply_overrides(
            torch,
            raw=raw,
            stage=stage,
            low_dim=_LowDim(),
            servo=servo,
            hold_pickup=False,
            hold_placement=False,
            hold_gripper_open=True,
        )
        # Approach: the gripper is opened and the other four channels are the
        # teacher's, untouched.
        self.assertEqual(float(applied[0, 4]), 1.0)
        self.assertAlmostEqual(float(applied[0, 0]), -0.7, places=6)
        self.assertEqual(int(source[0]), SOURCE_GRIPPER_HOLD)
        # Alignment: the whole command is the controller's.
        self.assertEqual(int(source[1]), SOURCE_YAW_TAIL)
        self.assertEqual(float(applied[1, 0]), 0.0)
        self.assertEqual(float(applied[1, 4]), 1.0)
        # Pickup with both holds off: pure teacher.
        self.assertEqual(int(source[2]), SOURCE_TEACHER)
        self.assertAlmostEqual(float(applied[2, 4]), -0.7, places=6)

    def test_the_hold_can_be_switched_off_for_the_ablation(self):
        from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
            SOURCE_TEACHER,
            _apply_overrides,
        )

        class _LowDim:
            ee_position = torch.tensor([[0.0, 0.0, 0.27]])
            ee_yaw = torch.zeros(1)
            gripper_opening = torch.tensor([0.2])

        servo = YawTailController(
            torch=torch,
            calibration=_calibration(),
            action_step_yaw=0.08,
            action_step_xyz=0.015,
        )
        applied, source = _apply_overrides(
            torch,
            raw=torch.full((1, 5), -0.7),
            stage=torch.tensor([STAGE_MOVE_TO]),
            low_dim=_LowDim(),
            servo=servo,
            hold_pickup=False,
            hold_placement=False,
            hold_gripper_open=False,
        )
        self.assertEqual(int(source[0]), SOURCE_TEACHER)
        self.assertAlmostEqual(float(applied[0, 4]), -0.7, places=6)

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
