"""Acceptance, relabelling and row assembly for the three-stage bank.

The traps these cover, all of them named in the design because they have
happened before in this campaign:

RELABELLING IS PER TRAJECTORY, NOT PER SCENE. A bowl chain is not a plate
demonstration because a plate is visible. Every row of an accepted chain --
including its move-to prefix -- takes the chain's own final instruction.

A FAILED CHAIN IS NOT A DEMONSTRATION OF THE TASK IT FAILED. Its pickup prefix
may still be a valid pick_up demonstration, and it is written to a SEPARATE
file under the pick_up label so it can never leak into the full-task bank.

NO WINDOW CROSSES AN EPISODE. Rows are per decision, per world, and the last
row is the decision the chain finished on -- decisions after that ran against a
frozen world and are not demonstrations of anything.

RELABELLED TEXT WITHOUT A REFRESHED PRIOR IS A CORRUPT BANK. ``state`` and
``prior`` were computed under the teachers' prompts; the written report marks
them stale so the SFT entry point can refuse the bank rather than relying on
somebody remembering.
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
    SOURCE_ALIGN_BRIDGE,
    SOURCE_GRIPPER_HOLD,
    SOURCE_YAW_TAIL,
    STAGE_SETTLE,
    STAGE_ALIGN,
    STAGE_COMPLETE,
    STAGE_FAILED,
    STAGE_MOVE_TO,
    STAGE_PICK_UP,
    STAGE_PLACEMENT,
    StagedRound,
)
from tools.audit.build_cdpr_staged_sft_dataset import (
    build_rows,
    dataset_report,
)

PER = 4
DECISIONS = 10
STEPS = DECISIONS * PER


def _chain(
    *,
    destination="plate",
    complete=True,
    live_decisions=8,
    reach=1,
    align=2,
    pickup=4,
    placement=7,
    carry_slip_at=None,
    approach=0.08,
    transport=0.15,
    radius=0.091,
    handoff_lift=0.06,
):
    """One world's worth of arrays, shaped like a real recording."""

    live_steps = live_decisions * PER
    active = np.zeros((STEPS,), bool)
    active[:live_steps] = True
    stage = np.full((STEPS,), STAGE_MOVE_TO, np.int8)
    stage[(reach + 1) * PER :] = STAGE_ALIGN
    stage[(align + 1) * PER :] = STAGE_PICK_UP
    stage[(pickup + 1) * PER :] = STAGE_PLACEMENT
    decision_stage = stage[::PER].copy()

    grasp = np.zeros((STEPS,), bool)
    grasp[(pickup + 1) * PER - 1 :] = True
    released = np.zeros((STEPS,), bool)
    released[placement * PER :] = True
    grasp[placement * PER :] = False
    if carry_slip_at is not None:
        grasp[carry_slip_at:] = False
        released[:] = False

    reset_objects = np.zeros((4, 3), np.float32)
    reset_objects[0, :2] = (0.0, 0.0)
    reset_objects[1, :2] = (transport, 0.0)
    return {
        "active": active,
        "decision_active": active[::PER].copy(),
        "step_stage": stage,
        "decision_stage": decision_stage,
        "physical_grasp": grasp,
        "released": released,
        "reset_object_xyz": reset_objects,
        "reset_ee_xyz": np.array([approach, 0.0, 0.28], np.float32),
        "reach_event": reach,
        "align_event": align,
        "pickup_event": pickup,
        "placement_event": placement if complete else -1,
        "final_stage": STAGE_COMPLETE if complete else STAGE_FAILED,
        "destination": destination,
        "radius": radius,
        "handoff_lift": handoff_lift,
        "approach": approach,
        "transport": transport,
    }


def _round(chains, *, round_index=0):
    worlds = len(chains)
    zeros_step = lambda shape=(): np.zeros(  # noqa: E731
        (STEPS, worlds, *shape), np.float32
    )
    stack = lambda key, dtype: np.stack(  # noqa: E731
        [np.asarray(chain[key], dtype=dtype) for chain in chains], axis=-1
    )

    action_source = np.zeros((STEPS, worlds), np.int8)
    step_stage = stack("step_stage", np.int8)
    action_source[step_stage == STAGE_ALIGN] = SOURCE_YAW_TAIL

    labels = {"plate": "apple", "bowl": "orange"}
    return StagedRound(
        actions=np.linspace(-1.0, 1.0, STEPS * worlds * 5, dtype=np.float32).reshape(
            STEPS, worlds, 5
        ),
        teacher_actions=zeros_step((5,)),
        action_source=action_source,
        active=stack("active", bool),
        step_stage=step_stage,
        ee_xyz=zeros_step((3,)),
        ee_quaternion=zeros_step((4,)),
        ee_yaw=zeros_step(),
        gripper_opening=np.ones((STEPS, worlds), np.float32),
        object_xyz=zeros_step((4, 3)),
        object_quaternion=zeros_step((4, 4)),
        physical_grasp=stack("physical_grasp", bool),
        released=stack("released", bool),
        reach_success=np.zeros((STEPS, worlds), bool),
        pickup_success=np.zeros((STEPS, worlds), bool),
        placement_success=np.zeros((STEPS, worlds), bool),
        wrong_place_settled=np.zeros((STEPS, worlds), bool),
        placement_geometry_ok=np.zeros((STEPS, worlds), bool),
        target_lift=np.zeros((STEPS, worlds), np.float32),
        states=np.zeros((DECISIONS, worlds, 6), np.float32),
        priors=np.zeros((DECISIONS, worlds, 8, 5), np.float32),
        teacher_role=np.zeros((DECISIONS, worlds), np.int8),
        decision_stage=stack("decision_stage", np.int8),
        decision_active=stack("decision_active", bool),
        stage_local_index=np.zeros((DECISIONS, worlds), np.int32),
        episode_uid=np.asarray(
            [f"tag_s0_r{round_index}/r{round_index}w{i}" for i in range(worlds)],
            dtype="U128",
        ),
        rollout_index=np.zeros((worlds,), np.int64),
        scene_uid=np.asarray(
            [f"scene_{i:016x}" for i in range(worlds)], dtype="U64"
        ),
        split=np.asarray(["collection"] * worlds, dtype="U24"),
        destination=np.asarray(
            [chain["destination"] for chain in chains], dtype="U8"
        ),
        target_catalog=np.asarray(
            [
                "robocasa_apple" if chain["destination"] == "plate" else "robocasa_orange"
                for chain in chains
            ],
            dtype="U32",
        ),
        instruction_id=np.asarray(
            [4 if chain["destination"] == "plate" else 3 for chain in chains],
            dtype=np.int64,
        ),
        instruction_text=np.asarray(
            [
                f"put {labels[chain['destination']]} into {chain['destination']}"
                for chain in chains
            ],
            dtype="U256",
        ),
        move_teacher_text=np.asarray(
            [f"move to {labels[chain['destination']]}" for chain in chains],
            dtype="U256",
        ),
        pickup_teacher_text=np.asarray(
            [f"pick up {labels[chain['destination']]}" for chain in chains],
            dtype="U256",
        ),
        placement_teacher_text=np.asarray(
            [
                "put apple on the plate"
                if chain["destination"] == "plate"
                else "put orange into bowl"
                for chain in chains
            ],
            dtype="U256",
        ),
        approach_xy_distance=np.asarray(
            [chain["approach"] for chain in chains], np.float32
        ),
        transport_xy_distance=np.asarray(
            [chain["transport"] for chain in chains], np.float32
        ),
        destination_success_radius=np.asarray(
            [chain["radius"] for chain in chains], np.float32
        ),
        reset_object_xyz=np.stack(
            [chain["reset_object_xyz"] for chain in chains], axis=0
        ),
        reset_ee_xyz=np.stack(
            [chain["reset_ee_xyz"] for chain in chains], axis=0
        ),
        reach_event=np.asarray([chain["reach_event"] for chain in chains], np.int64),
        align_event=np.asarray([chain["align_event"] for chain in chains], np.int64),
        pickup_event=np.asarray(
            [chain["pickup_event"] for chain in chains], np.int64
        ),
        placement_event=np.asarray(
            [chain["placement_event"] for chain in chains], np.int64
        ),
        final_stage=np.asarray(
            [chain["final_stage"] for chain in chains], np.int64
        ),
        failure_code=np.zeros((worlds,), np.int64),
        handoff_lift=np.asarray(
            [chain["handoff_lift"] for chain in chains], np.float32
        ),
        diverged_world_mask=np.zeros((worlds,), bool),
        actions_per_decision=PER,
        round_index=round_index,
        role_switches=3,
        budgets_json=json.dumps({"move_decisions": 3}),
        calibration_json=json.dumps({"target_yaw": 0.0, "source": "test"}),
        teacher_manifest_json=json.dumps(
            {
                role: {"sha256": f"{index:064x}"}
                for index, role in enumerate(("move_to", "pick_up", "placement"))
            }
        ),
        config_json=json.dumps({"state_dim": 6}),
    )


class AcceptanceTests(unittest.TestCase):
    def test_a_complete_chain_is_accepted(self):
        record = _round([_chain()])
        accepted, reasons, _ = record.acceptance()
        self.assertTrue(bool(accepted[0]), msg=str(reasons[0]))

    def test_an_incomplete_chain_is_rejected_once(self):
        record = _round([_chain(complete=False)])
        accepted, reasons, counts = record.acceptance()
        self.assertFalse(bool(accepted[0]))
        self.assertEqual(str(reasons[0]), "chain_did_not_complete")
        # One reason per world; the census must sum to the batch.
        self.assertEqual(sum(counts.values()), record.worlds)

    def test_a_scene_that_starts_inside_the_goal_is_rejected(self):
        record = _round([_chain(transport=0.05, radius=0.091)])
        accepted, reasons, _ = record.acceptance()
        self.assertFalse(bool(accepted[0]))
        self.assertEqual(str(reasons[0]), "started_inside_goal")

    def test_a_short_approach_is_rejected(self):
        record = _round([_chain(approach=0.01)])
        accepted, reasons, _ = record.acceptance()
        self.assertFalse(bool(accepted[0]))
        self.assertEqual(str(reasons[0]), "approach_too_short")

    def test_a_lift_below_the_production_height_is_rejected(self):
        record = _round([_chain(handoff_lift=0.02)])
        accepted, reasons, _ = record.acceptance()
        self.assertFalse(bool(accepted[0]))
        self.assertEqual(str(reasons[0]), "no_lift_at_handoff")

    def test_a_mid_carry_slip_is_rejected_but_the_release_is_not(self):
        clean = _round([_chain()])
        self.assertTrue(bool(clean.acceptance()[0][0]))
        # Same chain, but the object leaves the hand three decisions into the
        # carry rather than at the release.
        slipped = _round([_chain(carry_slip_at=(4 + 2) * PER)])
        accepted, reasons, _ = slipped.acceptance()
        self.assertFalse(bool(accepted[0]))
        self.assertEqual(str(reasons[0]), "carry_interrupted")

    def test_a_diverged_world_is_never_accepted(self):
        record = _round([_chain()])
        record.diverged_world_mask = np.array([True])
        accepted, reasons, _ = record.acceptance()
        self.assertFalse(bool(accepted[0]))
        self.assertEqual(str(reasons[0]), "diverged")


class OpeningReleaseAcceptanceTests(unittest.TestCase):
    def trace(self):
        # Remote scene_44833e357637587f, steps 246..251 mapped to 26..31.
        record = _round([_chain(destination='bowl', radius=.057)])
        record.physical_grasp[20:28, 0] = True
        record.physical_grasp[28:, 0] = False
        record.released[:, 0] = False
        record.released[31, 0] = True
        record.actions[26:32, 0, 4] = [.77253, .72526, .75347, .73081, .77435, .75195]
        record.gripper_opening[25:32, 0] = [.37, .39806, .42498, .45603, .48946, .5255, .56235]
        record.object_xyz[26:32, 0, 0, 0] = [.00636, .00679, .007, .00722, .00749, .00784]
        record.target_lift[26:32, 0] = [.12478, .1231, .12071, .11655, .11048, .10249]
        return record

    def test_recorded_tomato_release_is_accepted_before_threshold_crossing(self):
        from tools.audit.inspect_cdpr_staged_rounds import inspect_round
        record = self.trace()
        self.assertTrue(record.acceptance()[0][0])
        self.assertEqual(record.carry_release_start_steps().tolist(), [28])
        episode = inspect_round(record)['episodes'][0]
        self.assertEqual(episode['carry_loss_step_count'], 3)
        self.assertEqual(episode['unexplained_carry_loss_step_count'], 0)
        self.assertEqual(episode['verified_release_contact_loss_step'], 28)

    def test_opening_away_from_receptacle_is_still_rejected(self):
        record = self.trace()
        record.object_xyz[28, 0, 0, 0] = .10
        self.assertEqual(record.acceptance()[1][0], 'carry_interrupted')

    def test_gripper_command_without_observed_opening_is_not_release(self):
        record = self.trace()
        record.gripper_opening[29, 0] = record.gripper_opening[28, 0]
        self.assertEqual(record.acceptance()[1][0], 'carry_interrupted')

    def test_passive_opening_without_open_command_is_not_release(self):
        record = self.trace()
        record.actions[28, 0, 4] = -.1
        self.assertEqual(record.acceptance()[1][0], 'carry_interrupted')

    def test_later_release_does_not_forgive_an_earlier_transport_slip(self):
        record = self.trace()
        record.physical_grasp[23, 0] = False
        self.assertEqual(record.acceptance()[1][0], 'carry_interrupted')

    def test_opening_that_never_crosses_threshold_is_not_verified_release(self):
        record = self.trace()
        record.released[:, 0] = False
        self.assertEqual(record.acceptance()[1][0], 'carry_interrupted')

    def test_regrasp_during_opening_does_not_qualify_as_continuous_release(self):
        record = self.trace()
        record.physical_grasp[29, 0] = True
        self.assertEqual(record.acceptance()[1][0], 'carry_interrupted')

    def test_the_live_slip_test_forgives_the_same_trace(self):
        """The rule the stage machine and the student evaluation both apply.

        Offline acceptance verifies the whole opening suffix; the two LIVE
        callers cannot -- they see one step at a time. They share this test, and
        it must not fire on the recorded release. `strict` requires
        `~carry_slip`, so a false positive here strikes every successful
        placement from the headline verdict.
        """

        import torch

        from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
            contact_ended_without_release,
            release_opening_over_goal,
        )

        record = self.trace()
        slipped = False
        for step in range(26, 32):
            in_progress = release_opening_over_goal(
                command=torch.tensor([float(record.actions[step, 0, 4])]),
                opening=torch.tensor(
                    [float(record.gripper_opening[step, 0])]
                ),
                previous_opening=torch.tensor(
                    [float(record.gripper_opening[step - 1, 0])]
                ),
                target_xy=torch.tensor(
                    [[float(record.object_xyz[step, 0, 0, 0]), 0.0]]
                ),
                receptacle_xy=torch.zeros(1, 2),
                radius=torch.tensor([0.057]),
            )
            slipped |= bool(
                contact_ended_without_release(
                    physical_grasp=torch.tensor(
                        [bool(record.physical_grasp[step, 0])]
                    ),
                    released=torch.tensor([bool(record.released[step, 0])]),
                    release_in_progress=in_progress,
                )[0]
            )
        self.assertFalse(slipped)

    def test_the_live_slip_test_still_catches_a_closed_hand_losing_it(self):
        import torch

        from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
            contact_ended_without_release,
        )

        self.assertTrue(
            bool(
                contact_ended_without_release(
                    physical_grasp=torch.tensor([False]),
                    released=torch.tensor([False]),
                    release_in_progress=torch.tensor([False]),
                )[0]
            )
        )

    def test_a_one_step_pause_in_the_ramp_is_still_a_release_when_latched(self):
        """Why the live signal is latched over the chunk rather than sampled.

        The stage machine reads at decision boundaries, and an opening ramp is
        not obliged to be increasing on the exact step the boundary lands on.
        Reading the instant would end a correct placement for a one-step pause
        in a release that is plainly under way.
        """

        import torch

        from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
            release_opening_over_goal,
        )

        record = self.trace()
        # Freeze the opening on the step a boundary would land on.
        record.gripper_opening[30, 0] = record.gripper_opening[29, 0]
        chunk = range(28, 32)
        latched = False
        for step in chunk:
            latched |= bool(
                release_opening_over_goal(
                    command=torch.tensor([float(record.actions[step, 0, 4])]),
                    opening=torch.tensor(
                        [float(record.gripper_opening[step, 0])]
                    ),
                    previous_opening=torch.tensor(
                        [float(record.gripper_opening[step - 1, 0])]
                    ),
                    target_xy=torch.tensor(
                        [[float(record.object_xyz[step, 0, 0, 0]), 0.0]]
                    ),
                    receptacle_xy=torch.zeros(1, 2),
                    radius=torch.tensor([0.057]),
                )[0]
            )
        self.assertTrue(latched)
        # Offline acceptance is deliberately stricter and refuses the same
        # chain, because it can see the whole suffix and this one is broken.
        self.assertEqual(record.acceptance()[1][0], "carry_interrupted")


class RowAssemblyTests(unittest.TestCase):
    def _build(self, chains, **kwargs):
        record = _round(chains)
        settings = dict(
            min_approach_xy=0.06,
            min_handoff_lift=0.05,
            include_rejected_pickup_prefix=True,
        )
        settings.update(kwargs)
        return build_rows([(("memory"), record)], **settings)

    def test_every_row_of_a_chain_carries_the_final_instruction(self):
        dataset, _, _ = self._build([_chain(destination="bowl")])
        self.assertEqual(
            sorted(set(dataset["instruction_text"].tolist())),
            ["put orange into bowl"],
        )
        self.assertEqual(sorted(set(dataset["instruction_id"].tolist())), [3])
        # Including the move-to prefix, which is the whole point.
        self.assertIn("move_to", set(dataset["stage_name"].tolist()))

    def test_the_teacher_wording_is_kept_beside_the_new_one(self):
        dataset, _, _ = self._build([_chain(destination="plate")])
        teacher = set(dataset["teacher_instruction_text"].tolist())
        self.assertIn("move to apple", teacher)
        self.assertIn("pick up apple", teacher)
        self.assertIn("put apple on the plate", teacher)
        self.assertNotIn("put apple on the plate", set(
            dataset["instruction_text"].tolist()
        ))

    def test_rows_stop_at_the_last_live_decision(self):
        dataset, _, _ = self._build([_chain(live_decisions=6)])
        self.assertEqual(int(dataset["decision_index"].max()), 5)
        self.assertEqual(int(dataset["state"].shape[0]), 6)

    def test_actions_are_the_executed_chunk_not_the_predicted_one(self):
        record = _round([_chain()])
        dataset, _, _ = build_rows(
            [("memory", record)],
            min_approach_xy=0.06,
            min_handoff_lift=0.05,
            include_rejected_pickup_prefix=False,
        )
        # Four executed actions supervised against eight emitted slots.
        self.assertEqual(dataset["action"].shape[1], PER)
        self.assertEqual(dataset["prior"].shape[1], 8)
        row = int(np.flatnonzero(dataset["decision_index"] == 3)[0])
        np.testing.assert_allclose(
            dataset["action"][row], record.actions[12:16, 0]
        )

    def test_a_failed_chain_supplies_pickup_material_in_a_separate_file(self):
        dataset, partial, census = self._build(
            [_chain(), _chain(destination="bowl", complete=False)]
        )
        self.assertEqual(census["accepted_chains"], 1)
        self.assertEqual(census["partial_pickup_chains"], 1)
        # The full-task bank holds ONLY the accepted chain.
        self.assertEqual(
            sorted(set(dataset["instruction_text"].tolist())),
            ["put apple into plate"],
        )
        self.assertTrue(bool(dataset["full_chain_success"].all()))
        # And the partial material is labelled pick_up, never put_into.
        self.assertEqual(
            sorted(set(partial["instruction_text"].tolist())),
            ["pick up orange"],
        )
        self.assertFalse(bool(partial["full_chain_success"].any()))

    def test_actions_after_the_success_are_masked_not_dropped(self):
        """The chunk keeps its shape; the post-success tail carries mask zero.

        The stage machine reads transitions at DECISION boundaries, so the
        world keeps executing the rest of the chunk after the placement
        predicate has fired. Those actions really happened, but they happen
        after the object is already in the receptacle.
        """

        record = _round([_chain()])
        # The predicate fires on the second action of decision 7.
        record.placement_success[7 * PER + 1, 0] = True
        dataset, _, _ = build_rows(
            [("memory", record)],
            min_approach_xy=0.06,
            min_handoff_lift=0.05,
            include_rejected_pickup_prefix=False,
        )
        final = int(np.flatnonzero(dataset["decision_index"] == 7)[0])
        self.assertEqual(
            dataset["action_mask"][final].tolist(), [True, True, False, False]
        )
        # Shape preserved, so the action head stays aligned.
        self.assertEqual(dataset["action"][final].shape, (PER, 5))
        # Earlier decisions are untouched.
        earlier = int(np.flatnonzero(dataset["decision_index"] == 3)[0])
        self.assertTrue(bool(dataset["action_mask"][earlier].all()))

    def test_a_chain_with_no_latched_success_step_supervises_its_whole_span(self):
        record = _round([_chain()])
        dataset, _, _ = build_rows(
            [("memory", record)],
            min_approach_xy=0.06,
            min_handoff_lift=0.05,
            include_rejected_pickup_prefix=False,
        )
        self.assertTrue(bool(dataset["action_mask"].all()))

    def test_the_boundary_distance_finds_the_nearest_handoff(self):
        dataset, _, _ = self._build([_chain(reach=1, align=2, pickup=4)])
        by_decision = {
            int(row): int(value)
            for row, value in zip(
                dataset["decision_index"], dataset["stage_boundary_distance"]
            )
        }
        self.assertEqual(by_decision[2], 0)
        self.assertEqual(by_decision[3], 1)
        self.assertEqual(by_decision[6], 2)

    def test_the_census_reports_every_destination_stage_cell(self):
        dataset, _, _ = self._build(
            [_chain(destination="plate"), _chain(destination="bowl")]
        )
        report = dataset_report(dataset)
        self.assertEqual(report["empty_strata"], [])
        self.assertEqual(
            sorted(report["rows_by_destination_stage"]),
            [
                "bowl/move_to",
                "bowl/pick_up",
                "bowl/placement",
                "plate/move_to",
                "plate/pick_up",
                "plate/placement",
            ],
        )

    def test_a_missing_stratum_is_named_rather_than_substituted(self):
        dataset, _, _ = self._build([_chain(destination="plate")])
        report = dataset_report(dataset)
        self.assertEqual(report["rows_by_destination"], {"plate": 8})
        self.assertNotIn("bowl", report["rows_by_destination"])

    def test_the_alignment_tail_is_marked_and_counted(self):
        dataset, _, _ = self._build([_chain(reach=1, align=3)])
        align_rows = dataset["substage_id"] == STAGE_ALIGN
        self.assertTrue(bool(align_rows.any()))
        # The tail is a move_to row, not a fourth stage.
        self.assertEqual(
            sorted(set(dataset["stage_name"][align_rows].tolist())), ["move_to"]
        )
        report = dataset_report(dataset)
        self.assertGreater(report["alignment_tail_share_of_move_to"], 0.0)
        self.assertLess(report["alignment_tail_share_of_move_to"], 1.0)
        self.assertIn("yaw_tail", report["actions_by_source"])

    def test_the_census_names_every_current_controller_source(self):
        dataset, _, _ = self._build([_chain(reach=1, align=3)])
        dataset["action_source"][0, 0] = SOURCE_GRIPPER_HOLD
        dataset["action_source"][1, 0] = SOURCE_ALIGN_BRIDGE
        report = dataset_report(dataset)
        self.assertEqual(report["actions_by_source"]["gripper_hold"], 1)
        self.assertEqual(report["actions_by_source"]["align_bridge"], 1)
        self.assertEqual(
            sum(report["actions_by_source"].values()),
            int(dataset["action_source"].size),
        )
        self.assertEqual(
            sum(report["supervised_actions_by_source"].values()),
            int(dataset["action_mask"].sum()),
        )


class ReachDiagnosticTests(unittest.TestCase):
    """The decomposition that turns `reached: 0` into a named cause.

    The first teacher screen scored 0 of 64 chains for every candidate of every
    role. That is a conjunction failing, and a zero says nothing about which
    conjunct: the production XY predicate, an open gripper, no grasp, or the
    height band. The cause was the height band -- an absolute 0.20 m floor
    against a pickup-ready height of 0.195-0.202 m for three of the four target
    objects -- and finding it cost a GPU run that this report would have saved.
    """

    def _round_with_predicate(self, *, readiness, ee_z, target_z):
        record = _round([_chain()])
        record.reach_success[: 4 * PER, 0] = True
        record.ee_xyz[:, 0, 2] = ee_z
        record.object_xyz[:, 0, 0, 2] = target_z
        record.gripper_opening[:, 0] = 1.0
        record.config_json = json.dumps(
            {"pick_grasp_height_offset": 0.0075, "readiness": readiness}
        )
        return record

    def test_it_separates_the_predicate_from_the_readiness_gate(self):
        # The historical band, and a tomato-height object: the reach predicate
        # fires on every step and the gate rejects every one of them.
        record = self._round_with_predicate(
            readiness={
                "min_gripper_opening": 0.90,
                "min_height_above_grasp": -0.005,
                "max_height_above_grasp": 0.12,
                "min_ee_z": 0.20,
                "max_ee_z": 0.34,
            },
            ee_z=0.1956,
            target_z=0.1781,
        )
        report = record.reach_diagnostics()
        self.assertEqual(report["predicate_fired_worlds"], 1)
        self.assertEqual(report["ready_worlds"], 0)
        gates = report["among_predicate_steps"]
        # Named, and named correctly: not the gripper, not a grasp, not the
        # relative band -- the absolute rail.
        self.assertEqual(gates["outside_absolute_rails"], 1.0)
        self.assertEqual(gates["gripper_closed"], 0.0)
        self.assertEqual(gates["already_grasping"], 0.0)
        self.assertEqual(gates["too_low_above_grasp"], 0.0)
        self.assertEqual(gates["too_high_above_grasp"], 0.0)

    def test_the_corrected_rail_accepts_the_same_pose(self):
        record = self._round_with_predicate(
            readiness={
                "min_gripper_opening": 0.90,
                "min_height_above_grasp": -0.005,
                "max_height_above_grasp": 0.12,
                "min_ee_z": 0.18,
                "max_ee_z": 0.40,
            },
            ee_z=0.1956,
            target_z=0.1781,
        )
        report = record.reach_diagnostics()
        self.assertEqual(report["predicate_fired_worlds"], 1)
        self.assertEqual(report["ready_worlds"], 1)
        self.assertEqual(
            report["among_predicate_steps"]["outside_absolute_rails"], 0.0
        )

    def test_a_teacher_that_never_gets_close_reads_differently(self):
        record = _round([_chain()])
        record.ee_xyz[:, 0, :2] = 0.25
        record.config_json = json.dumps({"pick_grasp_height_offset": 0.0075})
        report = record.reach_diagnostics()
        # No predicate steps at all, so there is no gate to blame and the
        # closest-approach distance is the number to read instead.
        self.assertEqual(report["predicate_fired_worlds"], 0)
        self.assertNotIn("among_predicate_steps", report)
        self.assertGreater(report["closest_xy_distance_m"]["median"], 0.02)


class StageDiagnosticTests(unittest.TestCase):
    """The pickup and placement ladders, and the command underneath them."""

    def test_offline_inspector_explains_rejected_native_completion(self):
        from tools.audit.inspect_cdpr_staged_rounds import inspect_round
        record = _round([_chain(carry_slip_at=22)])
        result = inspect_round(record)
        self.assertEqual(result['accepted'], 0)
        episode = result['episodes'][0]
        self.assertEqual(episode['rejection'], 'carry_interrupted')
        self.assertEqual(episode['carry_loss_steps_before_release'][0], 22)
        self.assertEqual(episode['carry_loss_step_count'], 10)
        self.assertEqual(episode['post_action_trace'][2]['step'], 22)
        self.assertFalse(episode['post_action_trace'][2]['grasped'])

    def test_offline_inspector_does_not_call_release_a_carry_loss(self):
        from tools.audit.inspect_cdpr_staged_rounds import inspect_round
        result = inspect_round(_round([_chain()]))
        self.assertEqual(result['accepted'], 1)
        self.assertEqual(result['episodes'][0]['carry_loss_step_count'], 0)
        self.assertEqual(result['episodes'][0]['first_release_step_after_handoff'], 28)

    def test_initial_comparison_separates_reset_and_prior_differences(self):
        from tools.audit.inspect_cdpr_staged_rounds import initial_difference
        first, second = _round([_chain()]), _round([_chain()])
        second.priors[0] += .5
        result = initial_difference(first, second)['max_absolute_difference']
        self.assertEqual(result['reset_ee_m'], 0.)
        self.assertEqual(result['first_state'], 0.)
        self.assertEqual(result['first_prior'], .5)

    def test_offline_inspector_reads_saved_round_on_cpu(self):
        from contextlib import redirect_stdout
        import io
        from pathlib import Path
        import tempfile
        from tools.audit.inspect_cdpr_staged_rounds import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'screen.npz'
            _round([_chain(carry_slip_at=22)]).to_npz(path)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([str(path)]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report['episodes'][0]['rejection'], 'carry_interrupted')

    def _pickup_round(self, *, grasp_at, lift_to, action_z):
        record = _round([_chain()])
        record.config_json = json.dumps({"pick_grasp_height_offset": 0.0075})
        # Object resting; gripper descends into the pickup stage.
        record.object_xyz[:, 0, 0, 2] = 0.1781
        record.ee_xyz[:, 0, 2] = 0.20
        pickup = record.step_stage[:, 0] == STAGE_PICK_UP
        record.physical_grasp[:, 0] = False
        record.physical_grasp[grasp_at:, 0] = True
        record.target_lift[:, 0] = 0.0
        record.target_lift[grasp_at:, 0] = lift_to
        record.pickup_success[:, 0] = pickup & (record.target_lift[:, 0] >= 0.05)
        record.actions[:, 0, 2] = 0.0
        record.actions[pickup, 0, 2] = action_z
        return record

    def test_a_grasp_that_never_rises_reads_apart_from_no_grasp(self):
        """The two failures have opposite fixes, so they must not pool."""

        held = self._pickup_round(grasp_at=13, lift_to=0.008, action_z=0.02)
        report = held.pickup_diagnostics()
        self.assertEqual(report["entered_pickup"], 1)
        self.assertEqual(report["grasped"], 1)
        self.assertEqual(report["lifted"], 0)
        self.assertEqual(report["lift_given_grasp"], 0.0)
        self.assertAlmostEqual(
            report["max_lift_when_grasped_m"]["median"], 0.008, places=4
        )
        # And the COMMAND is what says whether it was even trying.
        self.assertAlmostEqual(
            report["mean_action_z_while_grasped"], 0.02, places=4
        )

        never = self._pickup_round(grasp_at=10**6, lift_to=0.0, action_z=0.0)
        empty = never.pickup_diagnostics()
        self.assertEqual(empty["grasped"], 0)
        self.assertIsNone(empty["lift_given_grasp"])
        self.assertIsNone(empty["mean_action_z_while_grasped"])

    def test_a_grasp_that_lifts_completes_the_ladder(self):
        record = self._pickup_round(
            grasp_at=13, lift_to=0.061, action_z=0.40
        )
        report = record.pickup_diagnostics()
        self.assertEqual(report["grasped"], 1)
        self.assertEqual(report["lifted"], 1)
        self.assertEqual(report["lift_given_grasp"], 1.0)
        self.assertAlmostEqual(
            report["mean_action_z_while_grasped"], 0.40, places=4
        )

    def test_the_reach_table_ignores_later_stages(self):
        """A pickup closing its hand is not a reach failure.

        Unscoped, the pick_up phase's gate table reported gripper_closed 0.38
        and already_grasping 0.15 -- both of them correct behaviour of a stage
        that runs after the reach gate has already been passed.
        """

        record = _round([_chain()])
        record.config_json = json.dumps({"pick_grasp_height_offset": 0.0075})
        record.reach_success[:, 0] = True
        record.gripper_opening[:, 0] = 1.0
        # The pickup stage closes the hand and takes hold, as it should.
        pickup = record.step_stage[:, 0] == STAGE_PICK_UP
        record.gripper_opening[pickup, 0] = 0.3
        record.physical_grasp[pickup, 0] = True
        record.ee_xyz[:, 0, 2] = 0.20
        record.object_xyz[:, 0, 0, 2] = 0.1781
        gates = record.reach_diagnostics()["among_predicate_steps"]
        self.assertEqual(gates["gripper_closed"], 0.0)
        self.assertEqual(gates["already_grasping"], 0.0)

    def test_the_placement_ladder_separates_carry_from_release(self):
        record = _round([_chain()])
        record.config_json = json.dumps({"pick_grasp_height_offset": 0.0075})
        placement = record.step_stage[:, 0] == STAGE_PLACEMENT
        # Carried to the receptacle but never let go.
        record.object_xyz[:, 0, 1, :2] = 0.0
        record.object_xyz[placement, 0, 0, 0] = 0.01
        record.placement_geometry_ok[placement, 0] = True
        record.released[:, 0] = False
        record.placement_success[:, 0] = False
        report = record.placement_diagnostics()
        self.assertEqual(report["entered_placement"], 1)
        self.assertEqual(report["reached_goal_geometry"], 1)
        self.assertEqual(report["released"], 0)
        self.assertEqual(report["placed"], 0)
        self.assertEqual(report["release_given_geometry"], 0.0)
        self.assertLess(
            report["closest_target_receptacle_xy_m"]["median"], 0.02
        )


class RoundTripTests(unittest.TestCase):
    def test_npz_round_trip_preserves_every_column(self):
        import tempfile

        record = _round([_chain(), _chain(destination="bowl")])
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/staged.npz"
            record.to_npz(path)
            restored = StagedRound.from_npz(path)
        np.testing.assert_array_equal(restored.actions, record.actions)
        np.testing.assert_array_equal(restored.episode_uid, record.episode_uid)
        self.assertEqual(restored.actions_per_decision, PER)
        self.assertEqual(
            json.loads(restored.teacher_manifest_json).keys(),
            json.loads(record.teacher_manifest_json).keys(),
        )
        accepted_before = record.acceptance()[0]
        accepted_after = restored.acceptance()[0]
        np.testing.assert_array_equal(accepted_before, accepted_after)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
