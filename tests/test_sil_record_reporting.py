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
    _merge_shard_summaries,
    plan_device_shards,
    strip_argv_flags,
    _build_dataset,
    _row_quota_mask,
    _determinism_report,
    _divergence,
    _episode_rows,
    _first_decision_report,
    _flip_report,
    _object_mix,
    _pick_up_prefix_report,
    _replay_report,
    _reset_identity_report,
    _slice_summary,
    _instruction_windows,
    _smooth_actions,
    _smoothness,
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


class RowQuotaTests(unittest.TestCase):
    """The quota is in DECISIONS and it never cuts an episode in half."""

    def setUp(self) -> None:
        # Two families with a 4x difference in episode length, which is the
        # asymmetry the quota exists for: measured on the first mixed build,
        # move_to ran 21.2 decisions per episode against 5.2 for placement.
        ids, uids = [], []
        for episode in range(20):
            ids += [PLATE] * 20
            uids += [f"long/{episode}"] * 20
        for episode in range(60):
            ids += [BOWL] * 5
            uids += [f"short/{episode}"] * 5
        self.ids = np.asarray(ids)
        self.uids = np.asarray(uids)

    def _kept(self, mask: np.ndarray) -> dict[int, int]:
        return {
            int(i): int((self.ids[mask] == i).sum())
            for i in np.unique(self.ids[mask])
        }

    def test_budget_is_rows_not_episodes(self) -> None:
        mask = _row_quota_mask(
            self.ids, self.uids, rows_per_instruction=100, seed=0
        )
        kept = self._kept(mask)
        # Equal rows, deliberately unequal episode counts: 5 long episodes
        # against 20 short ones. An episode quota would invert this.
        self.assertEqual(kept[PLATE], 100)
        self.assertEqual(kept[BOWL], 100)
        episodes = {
            int(i): len(np.unique(self.uids[mask][self.ids[mask] == i]))
            for i in (PLATE, BOWL)
        }
        self.assertEqual(episodes[PLATE], 5)
        self.assertEqual(episodes[BOWL], 20)

    def test_no_episode_is_split(self) -> None:
        mask = _row_quota_mask(
            self.ids, self.uids, rows_per_instruction=90, seed=3
        )
        for uid in np.unique(self.uids[mask]):
            rows = self.uids == uid
            self.assertTrue(
                mask[rows].all(),
                f"{uid} was kept only in part; _episode_split would then put "
                "consecutive decisions of one episode on both sides.",
            )

    def test_zero_and_oversized_budgets_keep_everything(self) -> None:
        for budget in (0, -1, 10_000):
            with self.subTest(budget=budget):
                mask = _row_quota_mask(
                    self.ids, self.uids, rows_per_instruction=budget, seed=0
                )
                self.assertTrue(mask.all())

    def test_seed_is_deterministic_and_meaningful(self) -> None:
        first = _row_quota_mask(
            self.ids, self.uids, rows_per_instruction=100, seed=11
        )
        self.assertTrue(
            np.array_equal(
                first,
                _row_quota_mask(
                    self.ids, self.uids, rows_per_instruction=100, seed=11
                ),
            )
        )
        self.assertFalse(
            np.array_equal(
                first,
                _row_quota_mask(
                    self.ids, self.uids, rows_per_instruction=100, seed=12
                ),
            )
        )


class DatasetBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recording = _with_observations(_synthetic())

    def test_only_successful_episodes_contribute(self) -> None:
        _, stats = _build_dataset([self.recording])
        # Worlds 0, 2 and 4 succeed; 1, 3 and 5 do not.
        self.assertEqual(stats["episodes_kept"], 3)

    def test_quota_is_reported_and_reaches_the_arrays(self) -> None:
        full, full_stats = _build_dataset([self.recording], ["rung"], ["src"])
        self.assertIsNone(full_stats["quota"])
        capped, stats = _build_dataset(
            [self.recording], ["rung"], ["src"], rows_per_instruction=1
        )
        self.assertEqual(stats["quota"]["rows_per_instruction"], 1)
        self.assertEqual(
            stats["quota"]["decisions_before"], full_stats["decisions"]
        )
        self.assertEqual(stats["decisions"], capped["state"].shape[0])
        self.assertLessEqual(stats["decisions"], full_stats["decisions"])
        # episodes_kept must describe the FILE, not the recordings, or a
        # quota'd build reports a dataset that was never written.
        self.assertEqual(
            stats["episodes_kept"], len(np.unique(capped["episode_uid"]))
        )
        for entry in stats["by_instruction"].values():
            self.assertLessEqual(entry["episodes_kept"], entry["episodes"])

    def test_decisions_are_truncated_at_the_success(self) -> None:
        dataset, stats = _build_dataset([self.recording], ["rung"], ["src"])
        # World 0 succeeds at env step 3 -> decision 0 only (steps 0-3).
        # World 2 succeeds at env step 5 -> decisions 0 and 1.
        # World 4 succeeds at env step 2 -> decision 0 only.
        self.assertEqual(stats["decisions"], 4)
        uids = dataset["episode_uid"].tolist()
        self.assertEqual(uids.count("src/r0w0"), 1)
        self.assertEqual(uids.count("src/r0w2"), 2)
        self.assertEqual(uids.count("src/r0w4"), 1)

    def test_actions_after_termination_are_masked_not_dropped(self) -> None:
        dataset, _ = _build_dataset([self.recording], ["rung"], ["src"])
        # Every chunk keeps its full width so the action head stays aligned.
        self.assertEqual(
            dataset["action"].shape[1], self.recording.actions_per_decision
        )
        rows = dataset["episode_uid"] == "src/r0w4"
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

    def test_duplicate_input_keys_are_refused(self) -> None:
        # Two files with the same key would merge their episodes under one id,
        # splitting one trajectory across train and validation.
        with self.assertRaises(ValueError) as caught:
            _build_dataset(
                [self.recording, self.recording],
                ["cap_0.03", "cap_0.03"],
                ["same", "same"],
            )
        self.assertIn("not unique", str(caught.exception))

    def test_episode_ids_are_unique_across_families_at_one_cap(self) -> None:
        # sil_harvest_0.03 and sil_pickup_0.03 both label as cap_0.03, so the
        # rung cannot carry the id; the source file must.
        placement = _with_observations(_synthetic())
        placement.start_distance_cap = 0.03
        pickup = _with_observations(_synthetic())
        pickup.start_distance_cap = 0.03
        dataset, stats = _build_dataset(
            [placement, pickup],
            ["cap_0.03", "cap_0.03"],
            ["replay_placement_00", "replay_pickup_00"],
        )
        uids = set(dataset["episode_uid"].tolist())
        self.assertTrue(any(u.startswith("replay_placement_00/") for u in uids))
        self.assertTrue(any(u.startswith("replay_pickup_00/") for u in uids))
        self.assertEqual(stats["episodes_kept"], 6)

    def test_a_recording_without_a_cap_falls_back_to_its_label(self) -> None:
        dataset, _ = _build_dataset(
            [self.recording], ["sil_harvest_0.06"], ["src"]
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


class ObjectMixTests(unittest.TestCase):
    """An object that cannot be grasped contributes attempts and no successes."""

    def test_the_success_filter_narrows_the_object_pool(self) -> None:
        recording = _synthetic()
        # apple=0, banana=1. Worlds 0/2/4 succeed, 1/3/5 do not.
        recording.target_catalog_ids = np.array(
            [0, 0, 0, 1, 0, 1], dtype=np.int64
        )
        mix = _object_mix(recording)
        self.assertEqual(mix["robocasa_apple"]["attempted"], 4)
        self.assertEqual(mix["robocasa_apple"]["kept"], 3)
        # Banana is attempted twice and never kept -- the shape of an object
        # that is wider than the gripper's open gap.
        self.assertEqual(mix["robocasa_banana"]["attempted"], 2)
        self.assertEqual(mix["robocasa_banana"]["kept"], 0)
        self.assertEqual(mix["robocasa_banana"]["rate"], 0.0)

    def test_a_recording_without_catalog_ids_reports_nothing(self) -> None:
        self.assertIsNone(_object_mix(_synthetic()))

    def test_dataset_reports_the_object_mix_per_instruction(self) -> None:
        recording = _with_observations(_synthetic())
        recording.target_catalog_ids = np.array(
            [0, 0, 0, 1, 0, 1], dtype=np.int64
        )
        _, stats = _build_dataset([recording], ["rung"], ["src"])
        plate = stats["by_instruction"]["put_into_plate"]["by_object"]
        # Plate worlds are 0, 1, 4, 5 -> apples 0/1/4, bananas 5.
        self.assertEqual(plate["robocasa_apple"]["kept"], 2)
        self.assertEqual(plate["robocasa_banana"]["kept"], 0)


class SmoothingTests(unittest.TestCase):
    """The filter that feeds part 3's survival number."""

    def setUp(self) -> None:
        self.recording = _synthetic()
        # A jagged xy signal so smoothing has something to remove.
        rng = np.random.default_rng(0)
        self.actions = rng.uniform(
            -1.0, 1.0, size=self.recording.actions.shape
        ).astype(np.float32)
        self.active = self.recording.active

    def test_none_is_an_exact_identity(self) -> None:
        # The control. If this ever changes an action, every survival rate
        # measured against it is uninterpretable.
        out = _smooth_actions(
            self.actions, self.active, method="none", window=5,
            alpha=0.5, channels="xyz",
        )
        self.assertTrue(np.array_equal(out, self.actions))

    def test_the_gripper_channel_is_untouched_by_default(self) -> None:
        out = _smooth_actions(
            self.actions, self.active, method="moving_average", window=5,
            alpha=0.5, channels="xyz",
        )
        self.assertTrue(np.array_equal(out[..., 4], self.actions[..., 4]))
        self.assertTrue(np.array_equal(out[..., 3], self.actions[..., 3]))
        self.assertFalse(np.array_equal(out[..., 0], self.actions[..., 0]))

    def test_all_channels_reaches_the_gripper(self) -> None:
        out = _smooth_actions(
            self.actions, self.active, method="moving_average", window=5,
            alpha=0.5, channels="all",
        )
        self.assertFalse(np.array_equal(out[..., 4], self.actions[..., 4]))

    def test_dead_steps_are_left_exactly_as_recorded(self) -> None:
        # World 4 terminates at env step 2, so steps 3+ ran against a frozen
        # world. Filtering them would pull the live tail toward dead values.
        out = _smooth_actions(
            self.actions, self.active, method="moving_average", window=5,
            alpha=0.5, channels="xyz",
        )
        dead = ~self.active
        self.assertTrue(np.array_equal(out[dead], self.actions[dead]))

    def test_episodes_do_not_bleed_into_each_other(self) -> None:
        # Changing one world's actions must not move any other world's.
        perturbed = self.actions.copy()
        perturbed[:, 0, :] = 0.0
        first = _smooth_actions(
            self.actions, self.active, method="moving_average", window=5,
            alpha=0.5, channels="xyz",
        )
        second = _smooth_actions(
            perturbed, self.active, method="moving_average", window=5,
            alpha=0.5, channels="xyz",
        )
        self.assertTrue(np.array_equal(first[:, 1:], second[:, 1:]))

    def test_filters_stay_inside_the_action_range(self) -> None:
        for method in ("moving_average", "ema", "median"):
            with self.subTest(method=method):
                out = _smooth_actions(
                    self.actions, self.active, method=method, window=5,
                    alpha=0.5, channels="all",
                )
                self.assertLessEqual(float(np.abs(out).max()), 1.0)

    def test_every_filter_reduces_the_step_delta(self) -> None:
        before = _smoothness(self.actions, self.active)
        for method in ("moving_average", "ema", "median"):
            with self.subTest(method=method):
                out = _smooth_actions(
                    self.actions, self.active, method=method, window=5,
                    alpha=0.3, channels="xyz",
                )
                after = _smoothness(out, self.active)
                self.assertLess(
                    after["mean_abs_step_delta"],
                    before["mean_abs_step_delta"],
                )

    def test_a_constant_signal_survives_every_filter_unchanged(self) -> None:
        # Edge padding, not zero padding: a filter that faded the ends would
        # move a constant, and the ends are the approach and the release.
        constant = np.full_like(self.actions, 0.4)
        for method in ("moving_average", "median"):
            with self.subTest(method=method):
                out = _smooth_actions(
                    constant, self.active, method=method, window=5,
                    alpha=0.5, channels="xyz",
                )
                live = self.active
                self.assertTrue(
                    np.allclose(out[live], constant[live], atol=1e-6)
                )

    def test_a_per_world_window_is_applied_per_world(self) -> None:
        # World 0 gets a width-1 window, which is an identity; world 1 gets 5.
        widths = np.ones((self.actions.shape[1],), dtype=np.int64)
        widths[1:] = 5
        out = _smooth_actions(
            self.actions, self.active, method="moving_average", window=5,
            alpha=0.5, channels="xyz", per_world_window=widths,
        )
        self.assertTrue(np.array_equal(out[:, 0], self.actions[:, 0]))
        self.assertFalse(np.array_equal(out[:, 1], self.actions[:, 1]))

    def test_a_wrongly_sized_window_table_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            _smooth_actions(
                self.actions, self.active, method="moving_average", window=5,
                alpha=0.5, channels="xyz",
                per_world_window=np.array([3, 5], dtype=np.int64),
            )

    def test_wider_windows_smooth_harder(self) -> None:
        deltas = []
        for width in (3, 5, 9, 15):
            out = _smooth_actions(
                self.actions, self.active, method="moving_average",
                window=width, alpha=0.5, channels="xyz",
            )
            deltas.append(_smoothness(out, self.active)["mean_abs_step_delta"])
        self.assertEqual(deltas, sorted(deltas, reverse=True))

    def test_an_unknown_method_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            _smooth_actions(
                self.actions, self.active, method="butterworth", window=5,
                alpha=0.5, channels="xyz",
            )


class InstructionWindowTests(unittest.TestCase):
    def test_named_instructions_take_their_override(self) -> None:
        ids = np.array([PLATE, BOWL, PICK, PLATE], dtype=np.int64)
        widths = _instruction_windows(
            ids, default=5, overrides={"put_into_plate": 13, "pick_up": 3}
        )
        self.assertEqual(widths.tolist(), [13, 5, 3, 13])

    def test_an_unknown_instruction_is_refused(self) -> None:
        ids = np.array([PLATE], dtype=np.int64)
        with self.assertRaises(ValueError):
            _instruction_windows(
                ids, default=5, overrides={"put_into_saucepan": 9}
            )


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


class DeviceShardTests(unittest.TestCase):
    """--devices splits a round range; getting it wrong loses rounds silently.

    The sharding is the only new logic behind ``--devices`` -- each shard is
    the ordinary single-device path -- so this is where the correctness lives.
    A dropped round is invisible at harvest time and shows up as a short
    dataset slice days later.
    """

    def test_even_split_covers_every_round_exactly_once(self) -> None:
        shards = plan_device_shards(0, 12, ["cuda:0", "cuda:1"])
        self.assertEqual(
            shards, [("cuda:0", 0, 6), ("cuda:1", 6, 6)]
        )
        covered = [
            index
            for _device, start, count in shards
            for index in range(start, start + count)
        ]
        self.assertEqual(covered, list(range(12)))

    def test_remainder_goes_to_the_earliest_devices(self) -> None:
        shards = plan_device_shards(0, 5, ["cuda:0", "cuda:1"])
        self.assertEqual(shards, [("cuda:0", 0, 3), ("cuda:1", 3, 2)])
        self.assertEqual(sum(count for _d, _s, count in shards), 5)

    def test_ranges_are_contiguous_and_disjoint(self) -> None:
        # Disjointness is what lets every shard write into one directory:
        # record files are named by round index while walking.
        shards = plan_device_shards(4, 7, ["cuda:0", "cuda:1", "cuda:2"])
        covered = [
            index
            for _device, start, count in shards
            for index in range(start, start + count)
        ]
        self.assertEqual(sorted(covered), covered)
        self.assertEqual(len(set(covered)), len(covered))
        self.assertEqual(covered, list(range(4, 11)))

    def test_first_round_offset_is_honoured(self) -> None:
        shards = plan_device_shards(6, 4, ["cuda:0", "cuda:1"])
        self.assertEqual(shards, [("cuda:0", 6, 2), ("cuda:1", 8, 2)])

    def test_more_devices_than_rounds_drops_the_idle_ones(self) -> None:
        # An empty shard would still load the model and build a world.
        shards = plan_device_shards(0, 2, ["cuda:0", "cuda:1", "cuda:2"])
        self.assertEqual(shards, [("cuda:0", 0, 1), ("cuda:1", 1, 1)])

    def test_degenerate_inputs_produce_no_shards(self) -> None:
        self.assertEqual(plan_device_shards(0, 0, ["cuda:0"]), [])
        self.assertEqual(plan_device_shards(0, 4, []), [])
        self.assertEqual(plan_device_shards(0, 4, ["  ", ""]), [])


class ArgvStripTests(unittest.TestCase):
    """Child command lines are rebuilt from the parent's argv."""

    def test_both_spellings_are_removed_with_their_values(self) -> None:
        argv = [
            "--mode", "record",
            "--device", "cuda:0",
            "--devices=cuda:0,cuda:1",
            "--rounds", "12",
            "--worlds", "2048",
        ]
        self.assertEqual(
            strip_argv_flags(argv, ("--device", "--devices", "--rounds")),
            ["--mode", "record", "--worlds", "2048"],
        )

    def test_unrelated_flags_and_their_values_survive(self) -> None:
        # The point of rebuilding from argv is that arguments added to the tool
        # later are inherited without anyone updating the shard code.
        argv = ["--start-distance-cap", "0.20", "--seed-torch", "0"]
        self.assertEqual(strip_argv_flags(argv, ("--device",)), argv)

    def test_a_value_that_looks_like_a_flag_name_is_not_eaten(self) -> None:
        argv = ["--output", "--device", "--rounds", "3"]
        # "--device" here is the VALUE of --output, so stripping --rounds must
        # not resynchronise onto it.
        self.assertEqual(
            strip_argv_flags(argv, ("--rounds",)), ["--output", "--device"]
        )


class ShardSummaryMergeTests(unittest.TestCase):
    """Pooled rates come from summed counts, not averaged rates."""

    def test_unequal_shards_pool_by_count(self) -> None:
        merged = _merge_shard_summaries(
            [
                {"run_00": {"by_instruction": {
                    "put_into_bowl": {"successes": 30, "episodes": 100}}}},
                {"run_01": {"by_instruction": {
                    "put_into_bowl": {"successes": 10, "episodes": 300}}}},
            ]
        )
        # Averaging the rates would give 0.20; the pooled rate is 40/400.
        self.assertEqual(merged["put_into_bowl"]["successes"], 40)
        self.assertEqual(merged["put_into_bowl"]["episodes"], 400)
        self.assertAlmostEqual(
            merged["put_into_bowl"]["source_success_rate"], 0.1
        )

    def test_non_run_keys_and_empty_input_are_ignored(self) -> None:
        merged = _merge_shard_summaries(
            [{"checkpoint": "x", "pick_up_prefix": {"placement_episodes": 3}}]
        )
        self.assertEqual(merged, {})
        self.assertEqual(_merge_shard_summaries([]), {})
