"""A lost latch is not a dropped object, and the labels must not blur them.

`grasp_loss_forensics` exists because `caught_target` going False has four
physically distinct causes and the campaign has so far had one word for all of
them. Its whole output is a taxonomy, so the failure mode that matters is a
trajectory that gets the WRONG label -- a confident table about a mechanism
that did not happen. These tests build one trajectory per label, with the
answer fixed by construction rather than read off the output, and check that
the classifier returns it.

Two of them are the ones that would silently corrupt the finding:

  * an object that stays in the hand through a two-step latch gap must be
    `relatched`, not a drop. That case IS the D3 hypothesis, so a classifier
    that missed it would report the null it was written to test.
  * a gap that contains a real gripper opening must NOT be `relatched`. That
    is a release followed by a re-grasp, and counting it as detector chatter
    would manufacture evidence for D3.

The trajectories run through the real `_episode_terms`, so the population and
the `blocking_stage` conditioning are the production ones rather than a
restatement of them here.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.audit.grasp_loss_forensics import (
    _Params,
    _loss_rows,
    _runs_of_true,
    _scene_fingerprint,
    _taxonomy,
)
from tools.audit.placement_failure_decomposition import _Thresholds, _episode_terms
from tools.audit.sil_record import _Recording

STEPS = 40
PLATE_ID = 4
REST = 0.02
CARRY_Z = 0.25
HELD_GAP = 0.01  # object rides 1 cm below the end effector
GRASP_AT = 10
LOSE_AT = 20

METADATA = {
    "put_plate_xy_tolerance": 0.091,
    "put_bowl_xy_tolerance": 0.057,
    "put_container_z_tolerance": 0.12,
    "placement_wrong_drop_settle_margin": 0.025,
}


class _Args:
    """The tool's four judgement calls, at their defaults."""

    relatch_steps = 10
    window_steps = 16
    separation_m = 0.02
    min_steps_after = 4
    slip_bar = 0.008


def _build(worlds: int) -> dict[str, np.ndarray]:
    """A carry that holds from GRASP_AT and loses the latch at LOSE_AT.

    Every world starts identical; each test mutates the one thing its label
    depends on, so a label change can only come from that mutation.
    """

    end_effector = np.zeros((STEPS, worlds, 3), dtype=np.float32)
    objects = np.zeros((STEPS, worlds, 2, 3), dtype=np.float32)
    caught = np.zeros((STEPS, worlds), dtype=bool)
    active = np.ones((STEPS, worlds), dtype=bool)
    # The receptacle, parked where nothing in these trajectories can reach it:
    # every episode must fail, so the recomputed verdict and the recording's
    # own latched success agree at False without needing a success case here.
    objects[:, :, 1, :] = np.array([0.5, 0.0, REST], dtype=np.float32)
    for world in range(worlds):
        end_effector[:, world, 0] = 0.10
        end_effector[:, world, 2] = CARRY_Z
        objects[:, world, 0, 0] = 0.10
        objects[:, world, 0, 2] = CARRY_Z - HELD_GAP
        caught[GRASP_AT:LOSE_AT, world] = True
    return {
        "ee": end_effector,
        "obj": objects,
        "caught": caught,
        "active": active,
        "opening": np.full((STEPS, worlds), 0.30, dtype=np.float32),
    }


def _recording(parts: dict[str, np.ndarray], worlds: int) -> _Recording:
    return _Recording(
        actions=np.zeros((STEPS, worlds, 5), dtype=np.float32),
        active=parts["active"],
        success=np.zeros((STEPS, worlds), dtype=bool),
        terminated=np.zeros((STEPS, worlds), dtype=bool),
        caught_target=parts["caught"],
        ee_xyz=parts["ee"],
        gripper_opening=parts["opening"],
        object_xyz=parts["obj"],
        instruction_ids=np.full((worlds,), PLATE_ID, dtype=np.int64),
        target_slots=np.zeros((worlds,), dtype=np.int64),
        reference_slots=np.ones((worlds,), dtype=np.int64),
        second_reference_slots=np.ones((worlds,), dtype=np.int64),
        horizons=np.full((worlds,), STEPS, dtype=np.int64),
        initial_target_xyz=np.zeros((worlds, 3), dtype=np.float32),
        support_surface_z=np.zeros((worlds,), dtype=np.float32),
        release_threshold=np.full((worlds,), 0.55, dtype=np.float32),
        target_rest_height=np.full((worlds,), REST, dtype=np.float32),
        physical_grasp_at_reset=np.zeros((worlds,), dtype=bool),
        instructions=np.asarray(
            ["put the object into the plate"] * worlds, dtype="U256"
        ),
        actions_per_decision=1,
        round_index=0,
        diverged_worlds=0,
        pick_lift_success_height=0.05,
    )


def _classify(
    parts: dict[str, np.ndarray], worlds: int, *, no_release: bool = True
) -> list[dict]:
    recording = _recording(parts, worlds)
    thresholds = _Thresholds(METADATA)
    base = _episode_terms(recording, thresholds)
    # The population is the production one, including its verdict guard: if the
    # recomputation disagreed, everything downstream would describe a different
    # task and the labels below would be meaningless.
    for row in base:
        assert row["recorded_success"] == row["recomputed_success"], row
        # `no_release` is what makes a False `caught_target` unambiguous, and
        # every trajectory here is built inside it -- except the one that opens
        # the gripper on purpose, which stops being a `no_release` episode by
        # definition and says so rather than asserting its way past it.
        if no_release:
            assert row["blocking_stage"] == "no_release", row
    return _loss_rows(recording, thresholds, _Params(_Args()), base)


class RunsOfTrueTest(unittest.TestCase):
    def test_counts_maximal_runs(self) -> None:
        mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
        self.assertEqual(_runs_of_true(mask), [(1, 3), (5, 6), (7, 10)])

    def test_empty(self) -> None:
        self.assertEqual(_runs_of_true(np.zeros(5, dtype=bool)), [])


class FirstLossClassTest(unittest.TestCase):
    """One trajectory per label, each differing in exactly one thing."""

    def test_relatched_when_the_object_never_leaves_the_hand(self) -> None:
        parts = _build(1)
        parts["caught"][LOSE_AT + 2 :, 0] = True
        row = _classify(parts, 1)[0]
        self.assertEqual(row["first_loss_class"], "relatched")
        self.assertEqual(row["relatch_gap_steps"], 2)
        self.assertEqual(row["episode_outcome"], "ended_holding")

    def test_a_gap_containing_a_real_release_is_not_a_relatch(self) -> None:
        """The guard that keeps a re-grasp from being read as chatter.

        Without it, an episode that opened the gripper, let go, closed again
        and re-acquired would be counted as evidence that the detector
        unlatched on a held object -- inventing support for the hypothesis the
        tool is meant to test.
        """

        parts = _build(1)
        parts["caught"][LOSE_AT + 2 :, 0] = True
        parts["opening"][LOSE_AT + 1, 0] = 0.90  # past the 0.55 release bar
        row = _classify(parts, 1, no_release=False)[0]
        self.assertNotEqual(row["first_loss_class"], "relatched")
        self.assertEqual(row["relatch_gap_steps"], -1)

    def test_separated_and_fell_when_the_object_reaches_rest_height(
        self,
    ) -> None:
        parts = _build(1)
        parts["obj"][LOSE_AT:, 0, 0, 2] = REST
        row = _classify(parts, 1)[0]
        self.assertEqual(row["first_loss_class"], "separated_and_fell")
        self.assertEqual(row["episode_outcome"], "object_at_rest_height")
        self.assertGreater(row["distance_excess_m"], _Args.separation_m)

    def test_separated_no_fall_when_it_leaves_but_stays_up(self) -> None:
        parts = _build(1)
        # Pulled away horizontally, well clear of the pads, but never descends.
        parts["obj"][LOSE_AT:, 0, 0, 0] = 0.10 + 0.08
        row = _classify(parts, 1)[0]
        self.assertEqual(row["first_loss_class"], "separated_no_fall")
        self.assertEqual(row["episode_outcome"], "object_above_rest")

    def test_held_no_relatch_when_nothing_physical_happens(self) -> None:
        parts = _build(1)
        row = _classify(parts, 1)[0]
        self.assertEqual(row["first_loss_class"], "held_no_relatch")
        self.assertEqual(row["distance_excess_m"], 0.0)

    def test_censored_when_the_episode_ends_at_the_loss(self) -> None:
        """`wrong_place_settled` terminates soon after a real drop.

        Those episodes must be named censored rather than folded into
        `held_no_relatch`, which would report "the object stayed in the hand"
        about episodes that were simply not observed long enough to tell.
        """

        parts = _build(1)
        parts["active"][LOSE_AT + 1 :, 0] = False
        row = _classify(parts, 1)[0]
        self.assertEqual(row["first_loss_class"], "censored")
        self.assertEqual(row["steps_live_after_loss"], 0)


class SlipAccountingTest(unittest.TestCase):
    def test_held_steps_over_the_bar_are_counted(self) -> None:
        """The D3 measurement itself: held steps already over the 8 mm bar.

        The object is walked 12 mm per step relative to the end effector while
        the latch is held, which is what a fast carry does to a per-step
        position difference. Every measured held step must land over the bar.
        """

        parts = _build(1)
        drift = 0.012 * np.arange(STEPS, dtype=np.float32)
        parts["obj"][:, 0, 0, 0] = 0.10 + drift
        parts["ee"][:, 0, 0] = 0.10  # the hand does not follow: pure slip
        row = _classify(parts, 1)[0]
        self.assertEqual(row["held_steps_measured"], LOSE_AT - GRASP_AT)
        self.assertEqual(
            row["held_steps_over_slip_bar"], row["held_steps_measured"]
        )
        self.assertAlmostEqual(row["slip_held_p50_m"], 0.012, places=5)

    def test_a_still_carry_is_under_the_bar(self) -> None:
        parts = _build(1)
        row = _classify(parts, 1)[0]
        self.assertEqual(row["held_steps_over_slip_bar"], 0)


class TaxonomyTest(unittest.TestCase):
    def test_fractions_are_over_losses_and_outcomes_over_grasps(self) -> None:
        parts = _build(2)
        parts["caught"][LOSE_AT + 2 :, 0] = True  # relatched
        parts["obj"][LOSE_AT:, 1, 0, 2] = REST  # separated and fell
        summary = _taxonomy(_classify(parts, 2))
        self.assertEqual(summary["grasped_episodes"], 2)
        self.assertEqual(summary["episodes_with_grasp_loss"], 2)
        self.assertEqual(summary["first_loss_class"]["relatched"]["episodes"], 1)
        self.assertEqual(
            summary["first_loss_class"]["separated_and_fell"]["episodes"], 1
        )
        self.assertEqual(
            summary["episode_outcome"]["ended_holding"]["episodes"], 1
        )
        self.assertEqual(
            summary["episode_outcome"]["object_at_rest_height"]["episodes"], 1
        )
        # `slip_at_loss_m` exists only for episodes that lost the latch, and
        # `slip_held_p50_m` for every grasped one. Sharing a denominator
        # between the two would make the oracle contrast unreadable.
        self.assertEqual(summary["slip_at_loss_m"]["n"], 2)
        self.assertEqual(summary["slip_held_p50_m"]["n"], 2)


class CensoredDiagnosisTest(unittest.TestCase):
    """`censored` is not a mechanism, so it has to say why the episode ended.

    A loss three steps from the end can be an episode the predicate stopped --
    the object came to rest somewhere wrong -- or one that simply ran out of
    budget while still holding. Those want opposite work, and without the
    termination evidence carried through, both read as the same class.
    """

    def _terminated(self) -> dict:
        parts = _build(1)
        parts["active"][LOSE_AT + 1 :, 0] = False
        recording_parts = parts
        return recording_parts

    def test_terminated_at_end_is_read_from_the_last_live_step(self) -> None:
        parts = self._terminated()
        recording = _recording(parts, 1)
        recording.terminated[LOSE_AT, 0] = True
        thresholds = _Thresholds(METADATA)
        base = _episode_terms(recording, thresholds)
        row = _loss_rows(recording, thresholds, _Params(_Args()), base)[0]
        self.assertEqual(row["first_loss_class"], "censored")
        self.assertTrue(row["terminated_at_end"])
        self.assertEqual(row["steps_loss_to_end"], 0)

    def test_a_budget_exhausted_episode_is_not_marked_terminated(self) -> None:
        parts = _build(1)
        row = _classify(parts, 1)[0]
        self.assertFalse(row["terminated_at_end"])
        self.assertTrue(row["timed_out"])

    def test_the_diagnosis_counts_only_censored_episodes(self) -> None:
        parts = _build(2)
        parts["active"][LOSE_AT + 1 :, 0] = False  # censored
        parts["obj"][LOSE_AT:, 1, 0, 2] = REST  # separated and fell
        summary = _taxonomy(_classify(parts, 2))
        self.assertEqual(summary["censored_diagnosis"]["episodes"], 1)
        self.assertEqual(
            summary["censored_diagnosis"]["steps_loss_to_end"]["n"], 1
        )


class SceneFingerprintTest(unittest.TestCase):
    """The guard on the paired policy-versus-oracle table.

    Round index and world index matching is what SHOULD imply the same scene,
    because the reset is a pure function of its seed. The fingerprint is the
    physical check on that, and without it two arms run at different caps would
    be paired episode-for-episode and compared as if they were the same starts.
    """

    def test_same_layout_matches(self) -> None:
        parts = _build(1)
        left = _scene_fingerprint(_recording(parts, 1), 0)
        right = _scene_fingerprint(_recording(parts, 1), 0)
        self.assertEqual(left, right)

    def test_a_moved_object_does_not_match(self) -> None:
        parts = _build(1)
        left = _scene_fingerprint(_recording(parts, 1), 0)
        parts["obj"][0, 0, 0, 0] += 0.05
        right = _scene_fingerprint(_recording(parts, 1), 0)
        self.assertNotEqual(left, right)


if __name__ == "__main__":
    unittest.main()
