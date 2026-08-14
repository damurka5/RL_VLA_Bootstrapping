"""The half of ``tools/audit/sil_record.py`` that runs without a GPU.

Worth a test because this arithmetic decides what enters the self-imitation
dataset. An off-by-one in ``first_success_step`` would append post-termination
actions to every demonstration; a sign error in the lift would silently pass or
fail the pick_up relabel. Neither would raise, and both would corrupt a
training run rather than a report.

The recording half needs MJWarp and is exercised by running the tool.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit.sil_record import (
    _Recording,
    _build_dataset,
    _determinism_report,
    _divergence,
    _episode_rows,
    _first_decision_report,
    _flip_report,
    _pick_up_prefix_report,
    _replay_report,
    _reset_identity_report,
    _slice_summary,
)

PLATE, BOWL, PICK = 4, 3, 8


def _synthetic() -> _Recording:
    """Six worlds over eight env steps, with hand-placed outcomes.

    ==========  ============  =========  =====================================
    world       instruction   outcome    note
    ==========  ============  =========  =====================================
    0           plate         ok @ 3     carried, rises 0.02 m
    1           plate         fail       held throughout, never released
    2           bowl          ok @ 5     never grasped in this fixture
    3           pick_up       fail       never grasped
    4           plate         ok @ 2     carried high, rises 0.08 m
    5           plate         fail       held throughout
    ==========  ============  =========  =====================================
    """

    steps, worlds, slots = 8, 6, 3
    instruction_ids = np.array(
        [PLATE, PLATE, BOWL, PICK, PLATE, PLATE], dtype=np.int64
    )
    target_slots = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)

    active = np.ones((steps, worlds), dtype=bool)
    success = np.zeros((steps, worlds), dtype=bool)
    for world, step in {0: 3, 2: 5, 4: 2}.items():
        success[step, world] = True
        # An episode terminates on success, so nothing steps after it.
        active[step + 1 :, world] = False

    initial_target_xyz = np.zeros((worlds, 3), dtype=np.float32)
    initial_target_xyz[:, 2] = 0.20
    object_xyz = np.zeros((steps, worlds, slots, 3), dtype=np.float32)
    object_xyz[..., 2] = 0.20
    for step in range(steps):
        object_xyz[step, 0, target_slots[0], 2] = (
            0.20 + 0.02 * min(step, 3) / 3.0
        )
        object_xyz[step, 4, target_slots[4], 2] = (
            0.20 + 0.08 * min(step, 2) / 2.0
        )

    caught_target = np.zeros((steps, worlds), dtype=bool)
    caught_target[:, [0, 1, 4, 5]] = True  # placement starts already grasped

    return _Recording(
        actions=np.arange(steps * worlds * 5, dtype=np.float32).reshape(
            steps, worlds, 5
        )
        / 1000.0,
        active=active,
        success=success,
        terminated=success.copy(),
        caught_target=caught_target,
        ee_xyz=np.zeros((steps, worlds, 3), dtype=np.float32),
        gripper_opening=np.full((steps, worlds), 0.5, dtype=np.float32),
        object_xyz=object_xyz,
        instruction_ids=instruction_ids,
        target_slots=target_slots,
        reference_slots=np.full(worlds, 2, dtype=np.int64),
        second_reference_slots=np.full(worlds, -1, dtype=np.int64),
        horizons=np.full(worlds, 2, dtype=np.int64),
        initial_target_xyz=initial_target_xyz,
        support_surface_z=np.full(worlds, 0.15, dtype=np.float32),
        release_threshold=np.full(worlds, 0.95, dtype=np.float32),
        target_rest_height=np.full(worlds, 0.02, dtype=np.float32),
        physical_grasp_at_reset=np.array([1, 1, 0, 0, 1, 1], dtype=bool),
        instructions=np.array(["put it in the plate"] * worlds, dtype="U256"),
        actions_per_decision=4,
        round_index=0,
        diverged_worlds=0,
        pick_lift_success_height=0.05,
    )


class EpisodeVerdictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recording = _synthetic()

    def test_success_is_latched_over_env_steps(self) -> None:
        self.assertEqual(
            self.recording.episode_success.tolist(),
            [True, False, True, False, True, False],
        )

    def test_first_success_step_is_minus_one_without_a_success(self) -> None:
        # argmax over an all-False column returns 0, which would read as
        # "succeeded immediately" and truncate the demonstration to nothing.
        self.assertEqual(
            self.recording.first_success_step.tolist(), [3, -1, 5, -1, 2, -1]
        )

    def test_episode_length_counts_only_active_steps(self) -> None:
        self.assertEqual(
            self.recording.episode_length.tolist(), [4, 8, 6, 8, 3, 8]
        )


class EpisodeRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _episode_rows(_synthetic())

    def test_instruction_ids_resolve_to_names(self) -> None:
        self.assertEqual(self.rows[0]["instruction"], "put_into_plate")
        self.assertEqual(self.rows[2]["instruction"], "put_into_bowl")
        self.assertEqual(self.rows[3]["instruction"], "pick_up")

    def test_peak_lift_is_measured_from_the_reset_height(self) -> None:
        self.assertAlmostEqual(self.rows[0]["peak_held_lift_m"], 0.02, places=3)
        self.assertAlmostEqual(self.rows[4]["peak_held_lift_m"], 0.08, places=3)

    def test_a_world_that_never_grasped_reports_zero_not_negative_inf(
        self,
    ) -> None:
        self.assertEqual(self.rows[3]["peak_held_lift_m"], 0.0)
        self.assertFalse(self.rows[3]["ever_grasped"])


class SliceSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = _slice_summary(_episode_rows(_synthetic()))

    def test_denominator_travels_with_the_rate(self) -> None:
        # Selecting on success selects on alignment for any policy, so a slice
        # is not interpretable without the rate it was drawn from.
        plate = self.summary["put_into_plate"]
        self.assertEqual(plate["episodes"], 4)
        self.assertEqual(plate["successes"], 2)
        self.assertEqual(plate["source_success_rate"], 0.5)

    def test_steps_to_success_is_none_when_nothing_succeeded(self) -> None:
        self.assertEqual(
            self.summary["put_into_plate"]["mean_env_steps_to_success"], 2.5
        )
        self.assertIsNone(
            self.summary["pick_up"]["mean_env_steps_to_success"]
        )


class PickUpPrefixTests(unittest.TestCase):
    def test_only_episodes_clearing_the_lift_threshold_would_relabel(
        self,
    ) -> None:
        report = _pick_up_prefix_report(_synthetic())
        self.assertEqual(report["placement_episodes"], 5)
        self.assertEqual(report["successful_placement_episodes"], 3)
        # Of the three successful placements, peak held lift is 0.02 (world 0),
        # 0.0 (world 2, never grasped) and 0.08 (world 4). Threshold is 0.05.
        self.assertEqual(report["would_relabel_as_pick_up"], 1)
        self.assertEqual(report["would_relabel_fraction"], round(1 / 3, 4))


class DeterminismNullTests(unittest.TestCase):
    def test_a_run_against_itself_agrees_exactly(self) -> None:
        recording = _synthetic()
        null = _determinism_report(recording, recording)
        self.assertEqual(null["success_agreement"], 1.0)
        self.assertTrue(null["actions_bitwise_identical"])
        self.assertEqual(null["max_abs_ee_delta_m"], 0.0)


class ReplaySurvivalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _synthetic()

    def test_an_unchanged_replay_survives_completely(self) -> None:
        report = _replay_report(self.source, self.source)
        self.assertEqual(report["recorded_successes"], 3)
        self.assertEqual(report["survival_rate"], 1.0)
        self.assertEqual(report["successes_at_a_different_step"], 0)

    def test_survival_is_scored_against_the_recorded_successes_only(
        self,
    ) -> None:
        # World 0 stops succeeding; world 4 succeeds one step later than it
        # did. The second still counts as survived but must be flagged: it
        # reached the same verdict by a different route.
        broken = copy.deepcopy(self.source)
        broken.success = self.source.success.copy()
        broken.success[3, 0] = False
        broken.success[2, 4] = False
        broken.success[3, 4] = True

        report = _replay_report(self.source, broken)
        self.assertEqual(report["survival_rate"], round(2 / 3, 5))
        self.assertEqual(report["successes_at_a_different_step"], 1)
        self.assertEqual(
            report["by_instruction"]["put_into_plate"]["survival_rate"], 0.5
        )
        self.assertEqual(
            report["by_instruction"]["put_into_bowl"]["survival_rate"], 1.0
        )

    def test_instructions_with_no_recorded_success_are_omitted(self) -> None:
        # pick_up never succeeds in the fixture, so a survival rate for it
        # would be 0/0 and would read as a total loss rather than as no data.
        report = _replay_report(self.source, self.source)
        self.assertNotIn("pick_up", report["by_instruction"])


class FirstDecisionTests(unittest.TestCase):
    """The discriminator: policy nondeterminism versus physics nondeterminism."""

    def test_identical_first_actions_point_at_physics(self) -> None:
        recording = _synthetic()
        report = _first_decision_report(recording, recording)
        self.assertTrue(report["identical"])
        self.assertEqual(report["worlds_differing"], 0)
        self.assertIn("physics diverges first", report["verdict"])

    def test_a_perturbed_first_action_points_upstream_of_physics(self) -> None:
        first = _synthetic()
        second = copy.deepcopy(first)
        second.actions = first.actions.copy()
        second.actions[0, 2, 1] += 1e-7
        report = _first_decision_report(first, second)
        self.assertFalse(report["identical"])
        self.assertEqual(report["worlds_differing"], 1)
        # Not exactly 1e-7: actions are float32, whose resolution near these
        # magnitudes is itself ~1e-7. That is the scale a nondeterministic
        # reduction kernel perturbs the forward by, which is why the report
        # publishes the magnitude rather than just a boolean.
        self.assertAlmostEqual(report["max_abs_delta"], 1e-7, delta=1e-8)
        self.assertIn("upstream of physics", report["verdict"])


class DivergenceMaskingTests(unittest.TestCase):
    def test_frozen_post_termination_tails_are_excluded(self) -> None:
        # World 4 terminates at step 2. Planting a large difference in its
        # frozen tail must not register as trajectory divergence, because the
        # world was not being stepped -- that artefact is exactly what made the
        # first version of this metric unreadable.
        first = _synthetic()
        second = copy.deepcopy(first)
        second.ee_xyz = first.ee_xyz.copy()
        second.ee_xyz[6, 4, 0] += 0.5

        report = _divergence(first, second)
        self.assertEqual(report["max_ee_delta_m_active"], 0.0)
        self.assertEqual(report["max_ee_delta_m_unmasked"], 0.5)

    def test_divergence_inside_the_active_window_is_reported(self) -> None:
        first = _synthetic()
        second = copy.deepcopy(first)
        second.ee_xyz = first.ee_xyz.copy()
        second.ee_xyz[1, 4, 0] += 0.25  # world 4 is still active at step 1

        report = _divergence(first, second)
        self.assertAlmostEqual(report["max_ee_delta_m_active"], 0.25, places=6)


class FlipReportTests(unittest.TestCase):
    def test_flips_are_counted_and_attributed_per_instruction(self) -> None:
        first = _synthetic()
        second = copy.deepcopy(first)
        second.success = first.success.copy()
        second.success[3, 0] = False  # world 0, a plate success, is lost

        report = _flip_report(first, second)
        self.assertEqual(report["flipped_total"], 1)
        self.assertEqual(report["won_in_a_only"], 1)
        self.assertEqual(report["won_in_b_only"], 0)
        plate = report["by_instruction"]["put_into_plate"]
        self.assertEqual(plate["success_a"], 2)
        self.assertEqual(plate["success_b"], 1)
        self.assertEqual(plate["flipped"], 1)

    def test_a_run_against_itself_has_no_flips(self) -> None:
        recording = _synthetic()
        report = _flip_report(recording, recording)
        self.assertEqual(report["flipped_total"], 0)
        self.assertEqual(report["agreement"], 1.0)


class ResetIdentityTests(unittest.TestCase):
    def test_differing_instruction_ids_are_caught(self) -> None:
        first = _synthetic()
        second = copy.deepcopy(first)
        second.instruction_ids = first.instruction_ids.copy()
        second.instruction_ids[0] = PICK

        report = _reset_identity_report(first, second)
        self.assertFalse(report["same_instruction_ids"])
        self.assertTrue(report["same_horizons"])

    def test_identical_resets_report_clean(self) -> None:
        recording = _synthetic()
        report = _reset_identity_report(recording, recording)
        self.assertTrue(report["same_instruction_ids"])
        self.assertTrue(report["same_target_slots"])
        self.assertTrue(report["same_horizons"])
        self.assertTrue(report["same_grasp_at_reset"])
        self.assertEqual(report["max_initial_target_delta_m"], 0.0)


def _with_observations(recording: _Recording) -> _Recording:
    """Attach per-decision states and priors to the fixture.

    Eight env steps at four actions per decision is two decisions, so world 0
    (success at env step 3) spans both and world 4 (success at env step 2) only
    the first.
    """

    steps, worlds = recording.actions.shape[0], recording.worlds
    per_decision = recording.actions_per_decision
    decisions = steps // per_decision
    recording.states = (
        np.arange(decisions * worlds * 3, dtype=np.float32).reshape(
            decisions, worlds, 3
        )
    )
    recording.priors = np.zeros(
        (decisions, worlds, per_decision, 5), dtype=np.float32
    )
    return recording


class DatasetBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recording = _with_observations(_synthetic())

    def test_only_successful_episodes_contribute(self) -> None:
        _, stats = _build_dataset([self.recording])
        # Worlds 0, 2 and 4 succeed; 1, 3 and 5 do not.
        self.assertEqual(stats["episodes_kept"], 3)

    def test_decisions_are_truncated_at_the_success(self) -> None:
        dataset, stats = _build_dataset([self.recording], ["rung"])
        # World 0 succeeds at env step 3 -> decision 0 only (steps 0-3).
        # World 2 succeeds at env step 5 -> decisions 0 and 1.
        # World 4 succeeds at env step 2 -> decision 0 only.
        self.assertEqual(stats["decisions"], 4)
        uids = dataset["episode_uid"].tolist()
        self.assertEqual(uids.count("rung/r0w0"), 1)
        self.assertEqual(uids.count("rung/r0w2"), 2)
        self.assertEqual(uids.count("rung/r0w4"), 1)

    def test_actions_after_termination_are_masked_not_dropped(self) -> None:
        dataset, _ = _build_dataset([self.recording], ["rung"])
        # Every chunk keeps its full width so the action head stays aligned.
        self.assertEqual(
            dataset["action"].shape[1], self.recording.actions_per_decision
        )
        rows = dataset["episode_uid"] == "rung/r0w4"
        # World 4 terminated at env step 2, so steps 3 of that chunk is dead.
        self.assertEqual(dataset["action_mask"][rows][0].tolist(),
                         [True, True, True, False])

    def test_source_success_rate_travels_with_each_slice(self) -> None:
        _, stats = _build_dataset([self.recording])
        plate = stats["by_instruction"]["put_into_plate"]
        self.assertEqual(plate["source_episodes"], 4)
        self.assertEqual(plate["episodes"], 2)
        self.assertEqual(plate["source_success_rate"], 0.5)

    def test_rungs_are_reported_separately_not_pooled(self) -> None:
        # Two rungs whose plate rates are 0.5 and 0.0. The pooled figure is
        # 0.25, which describes neither, so both must survive in by_group.
        easy = _with_observations(_synthetic())
        easy.start_distance_cap = 0.01
        hard = _with_observations(_synthetic())
        hard.start_distance_cap = 0.10
        hard.success = np.zeros_like(hard.success)
        hard.success[5, 2] = True  # one bowl success, no plate successes

        dataset, stats = _build_dataset([easy, hard])
        plate = stats["by_instruction"]["put_into_plate"]
        self.assertEqual(
            plate["by_group"]["cap_0.01"]["source_success_rate"], 0.5
        )
        self.assertEqual(
            plate["by_group"]["cap_0.1"]["source_success_rate"], 0.0
        )
        self.assertEqual(plate["source_success_rate"], 0.25)
        # And the rung is carried per row, so a consumer can drop one.
        self.assertEqual(
            set(dataset["source_group"].tolist()), {"cap_0.01", "cap_0.1"}
        )

    def test_a_recording_without_a_cap_falls_back_to_its_label(self) -> None:
        dataset, _ = _build_dataset(
            [self.recording], ["sil_harvest_0.06"]
        )
        self.assertEqual(
            set(dataset["source_group"].tolist()), {"sil_harvest_0.06"}
        )

    def test_a_verdict_only_recording_is_refused(self) -> None:
        # The recordings written before observation capture existed carry no
        # input side. Building a dataset from them would produce actions with
        # nothing to condition on, which must fail loudly rather than emit an
        # unusable file.
        with self.assertRaises(ValueError) as caught:
            _build_dataset([_synthetic()])
        self.assertIn("observations", str(caught.exception))


class RoundTripTests(unittest.TestCase):
    def test_npz_preserves_every_field_and_dtype(self) -> None:
        recording = _synthetic()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.npz"
            recording.to_npz(path)
            restored = _Recording.from_npz(path)
        for name in recording.__dataclass_fields__:
            original = getattr(recording, name)
            copied = getattr(restored, name)
            with self.subTest(field=name):
                if isinstance(original, np.ndarray):
                    self.assertTrue(np.array_equal(original, copied))
                    self.assertEqual(original.dtype, copied.dtype)
                elif isinstance(original, float) and np.isnan(original):
                    # NaN is the "no cap recorded" sentinel and never equals
                    # itself, so round-tripping it needs an explicit check.
                    self.assertTrue(np.isnan(copied))
                else:
                    self.assertEqual(original, copied)

    def test_a_recorded_cap_survives_the_round_trip(self) -> None:
        recording = _synthetic()
        recording.start_distance_cap = 0.03
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.npz"
            recording.to_npz(path)
            restored = _Recording.from_npz(path)
        self.assertEqual(restored.start_distance_cap, 0.03)


if __name__ == "__main__":
    unittest.main()
