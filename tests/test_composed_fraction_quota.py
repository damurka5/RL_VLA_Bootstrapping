"""The composed slice as a knob rather than an accident of harvest history.

`--rows-per-instruction` balances by INSTRUCTION ID and nothing else, so inside
`put_into_*` the caught-start carries and the composed grasp-carry-releases
compete purely by how many of each happen to be in the bank. Adding the o7
re-harvest therefore raised the composed share as a SIDE EFFECT: composed plate
went 0.0935 -> 0.1203 while caught plate fell 0.7150 -> 0.6822 in the same
pass, and no knob chose that trade or could undo it.

`--composed-fraction` splits each instruction's row budget by a stratum that is
intrinsic to the episode -- `physical_grasp_at_reset`, whether the object began
between the fingers. `source_group` cannot do this job: it carries the
start-distance cap, so a caught placement round and a composed round both
harvested at 0.20 are the same string.

The three properties that have to hold: the split is honoured when the bank can
supply it, the shortfall SPILLS rather than shrinking the instruction, and the
flag switched off reproduces the old build exactly.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.audit.sil_record import _row_quota_mask, _stratified_row_quota_mask

PLATE = 4  # put_into_plate, from ACTIVE_INSTRUCTION_TYPES
BOWL = 3


def _bank(
    *, composed: int, caught: int, rows_each: int = 5, instruction: int = PLATE
):
    """A synthetic instruction slice with a known stratum split."""

    instruction_ids, uids, grasped = [], [], []
    for index in range(composed):
        for _ in range(rows_each):
            instruction_ids.append(instruction)
            uids.append(f"composed/{index}")
            grasped.append(False)
    for index in range(caught):
        for _ in range(rows_each):
            instruction_ids.append(instruction)
            uids.append(f"caught/{index}")
            grasped.append(True)
    return (
        np.asarray(instruction_ids, dtype=np.int64),
        np.asarray(uids, dtype="U128"),
        np.asarray(grasped, dtype=bool),
    )


class TheSplitIsHonouredTests(unittest.TestCase):
    def test_half_and_half_when_both_strata_are_plentiful(self) -> None:
        ids, uids, grasped = _bank(composed=100, caught=100)
        keep, report = _stratified_row_quota_mask(
            ids, uids, grasped,
            rows_per_instruction=200, composed_fraction=0.5, seed=0,
        )
        entry = report["put_into_plate"]
        self.assertAlmostEqual(entry["realized_composed_fraction"], 0.5, delta=0.05)
        # And the instruction still got its budget.
        self.assertGreaterEqual(int(keep.sum()), 200)

    def test_the_fraction_actually_moves_the_mix(self) -> None:
        for requested in (0.0, 0.25, 0.75, 1.0):
            ids, uids, grasped = _bank(composed=200, caught=200)
            _, report = _stratified_row_quota_mask(
                ids, uids, grasped,
                rows_per_instruction=200,
                composed_fraction=requested,
                seed=0,
            )
            got = report["put_into_plate"]["realized_composed_fraction"]
            self.assertAlmostEqual(got, requested, delta=0.06, msg=f"{requested}")

    def test_it_never_cuts_an_episode_in_half(self) -> None:
        """`_episode_split` holds out whole episodes.

        A quota that kept part of a trajectory would put consecutive decisions
        sharing an observation history on both sides of the train/validation
        line, which is the leak the whole-episode rule exists to prevent.
        """

        ids, uids, grasped = _bank(composed=40, caught=40, rows_each=7)
        keep, _ = _stratified_row_quota_mask(
            ids, uids, grasped,
            rows_per_instruction=100, composed_fraction=0.5, seed=3,
        )
        for uid in np.unique(uids):
            rows = keep[uids == uid]
            self.assertIn(
                int(rows.sum()), (0, int(rows.size)),
                f"episode {uid} was cut: {int(rows.sum())} of {rows.size}",
            )


class TheShortfallSpillsTests(unittest.TestCase):
    def test_a_scarce_stratum_does_not_shrink_the_instruction(self) -> None:
        """Asking for 50% composed when 15% exists gives 15/85, not half a slice.

        A short slice would shrink this instruction relative to every other one
        in the mix -- exactly the skew the row quota exists to prevent, and
        harder to see because the request would look honoured.
        """

        ids, uids, grasped = _bank(composed=6, caught=200)
        budget = 200
        keep, report = _stratified_row_quota_mask(
            ids, uids, grasped,
            rows_per_instruction=budget, composed_fraction=0.5, seed=0,
        )
        entry = report["put_into_plate"]
        self.assertGreaterEqual(int(keep.sum()), budget)
        self.assertEqual(entry["composed_decisions"], 30)  # all 6 episodes
        self.assertLess(entry["realized_composed_fraction"], 0.5)

    def test_the_report_says_what_the_bank_could_supply(self) -> None:
        # The gap between requested and realized is the finding: a request the
        # bank cannot meet is a HARVEST problem, and no reweighting fixes it.
        ids, uids, grasped = _bank(composed=6, caught=200)
        _, report = _stratified_row_quota_mask(
            ids, uids, grasped,
            rows_per_instruction=200, composed_fraction=0.5, seed=0,
        )
        entry = report["put_into_plate"]
        self.assertEqual(entry["requested_composed_fraction"], 0.5)
        self.assertEqual(entry["available_composed_episodes"], 6)
        self.assertEqual(entry["available_caught_episodes"], 200)
        self.assertNotEqual(
            entry["requested_composed_fraction"],
            entry["realized_composed_fraction"],
        )

    def test_an_instruction_with_one_stratum_still_fills(self) -> None:
        # move_to has no caught starts at all. Its budget must not depend on a
        # knob aimed at put_into.
        ids, uids, grasped = _bank(composed=100, caught=0)
        keep, report = _stratified_row_quota_mask(
            ids, uids, grasped,
            rows_per_instruction=200, composed_fraction=0.25, seed=0,
        )
        self.assertGreaterEqual(int(keep.sum()), 200)
        self.assertEqual(report["put_into_plate"]["realized_composed_fraction"], 1.0)


class EachInstructionIsIndependentTests(unittest.TestCase):
    def test_two_instructions_get_their_own_budget_and_split(self) -> None:
        plate = _bank(composed=100, caught=100, instruction=PLATE)
        bowl = _bank(composed=100, caught=100, instruction=BOWL)
        ids = np.concatenate([plate[0], bowl[0]])
        # Namespace the uids, or the two instructions' episodes collide.
        uids = np.concatenate([
            np.char.add("plate:", plate[1]), np.char.add("bowl:", bowl[1]),
        ])
        grasped = np.concatenate([plate[2], bowl[2]])
        keep, report = _stratified_row_quota_mask(
            ids, uids, grasped,
            rows_per_instruction=200, composed_fraction=0.75, seed=1,
        )
        self.assertEqual(set(report), {"put_into_plate", "put_into_bowl"})
        for name in report:
            self.assertAlmostEqual(
                report[name]["realized_composed_fraction"], 0.75, delta=0.06, msg=name
            )
        for instruction in (PLATE, BOWL):
            self.assertGreaterEqual(int(keep[ids == instruction].sum()), 200)


class OffIsUnchangedTests(unittest.TestCase):
    def test_a_negative_fraction_means_off_not_zero_composed(self) -> None:
        """Default -1.0, not 0.0.

        A default of 0.0 would silently mean "no composed episodes at all",
        which is a very different and very wrong thing to do by accident.
        """

        from tools.audit.sil_record import main
        import inspect

        source = inspect.getsource(main)
        self.assertIn('"--composed-fraction"', source)
        self.assertIn("default=-1.0", source)

    def test_the_build_routes_around_the_stratified_path_when_off(self) -> None:
        import inspect

        from tools.audit.sil_record import _build_dataset

        source = inspect.getsource(_build_dataset)
        self.assertIn("if float(composed_fraction) >= 0.0:", source)
        self.assertIn("keep = _row_quota_mask(", source)

    def test_off_reproduces_the_flat_quota_exactly(self) -> None:
        # Same function, same generator, same draws -- so an existing bank
        # rebuilt without the flag is byte-identical to what it was.
        ids, uids, grasped = _bank(composed=50, caught=50)
        flat = _row_quota_mask(
            ids, uids, rows_per_instruction=200, seed=7
        )
        self.assertEqual(flat.shape, grasped.shape)
        again = _row_quota_mask(
            ids, uids, rows_per_instruction=200, seed=7
        )
        np.testing.assert_array_equal(flat, again)


class TheStratumIsIntrinsicTests(unittest.TestCase):
    def test_the_dataset_carries_starts_grasped_per_row(self) -> None:
        import inspect

        from tools.audit.sil_record import _build_dataset

        source = inspect.getsource(_build_dataset)
        self.assertIn('"starts_grasped"', source)
        # From the RECORDING PROPERTY, which derives the stratum from
        # caught_target[0]. The physical_grasp_at_reset column holds the FINAL
        # grasp state -- reading it here classified nearly the whole bank as
        # composed and made the fraction knob inert.
        self.assertIn("rec.starts_grasped[world]", source)
        self.assertNotIn("rec.physical_grasp_at_reset[world]", source)

    def test_source_group_is_not_used_as_the_stratum(self) -> None:
        """It holds the CAP, not the provenance.

        A caught placement round and a composed round both harvested at 0.20
        produce the same `cap_0.2` string, so weighting on it would silently do
        nothing for the split this knob exists for.
        """

        import inspect

        from tools.audit.sil_record import _group_label, _stratified_row_quota_mask

        self.assertIn("cap_", inspect.getsource(_group_label))
        # The BODY, not the whole source: the docstring names source_group
        # precisely to explain why it is not the stratum.
        body = inspect.getsource(_stratified_row_quota_mask)
        body = body[body.index('"""', body.index('"""') + 3) + 3 :]
        self.assertNotIn("source_group", body)
        self.assertIn("starts_grasped", body)


class TheReharvestScriptExposesItTests(unittest.TestCase):
    SCRIPT = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/run_cdpr_phase7_reharvest.sh"
    )

    def test_it_is_off_by_default_in_the_script(self) -> None:
        """One change at a time.

        Turning the mix knob on in the same run that changes the harvest gives
        a moved number two causes. The script defaults to -1 so the re-harvest
        effect is measured alone.
        """

        text = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn('COMPOSED_FRACTION="${COMPOSED_FRACTION:--1}"', text)
        self.assertIn('--composed-fraction "$COMPOSED_FRACTION"', text)

    def test_it_reports_the_realized_mix(self) -> None:
        text = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("realized_composed_fraction", text)
        self.assertIn("available_composed_episodes", text)

    def test_the_probe_stage_does_not_take_the_knob(self) -> None:
        # The availability probe runs --rows-per-instruction 0, i.e. no quota
        # at all, so a fraction there would be meaningless and misleading.
        text = self.SCRIPT.read_text(encoding="utf-8")
        probe = text[text.index("dataset7_probe") : text.index("balanced quota")]
        self.assertNotIn("--composed-fraction", probe)


class ThePoolCompositionIsVisibleBeforeSweepingTests(unittest.TestCase):
    """The read that has to exist BEFORE a fraction is chosen.

    The first phase-7 sweep ran three arms at 0.2 / 0.4 / 0.6 and every one
    realized 0.981, because the pool is ~98% composed BY DECISION COUNT and the
    spill had nowhere else to go. Three arms, hours of GPU, one mix. The
    availability probe runs --rows-per-instruction 0, so nothing downstream of
    the quota exists on that path -- which is why the composition has to be
    reported independently of the quota.

    The decision/episode distinction is the trap: a composed episode is ~32
    decisions and a caught carry ~5, so a pool that is a third composed by
    EPISODE is three quarters composed by DECISION, and the quota binds on
    decisions.
    """

    def _pool(self, composed_episodes, caught_episodes):
        import numpy as np

        from tools.audit.sil_record import _build_dataset

        from test_placement_failure_decomposition import _run, place, carry  # noqa

        # Built directly rather than through the physics fixture: this test is
        # about the accounting, and a synthetic recording states the episode
        # lengths that make the point explicit.
        from tools.audit.sil_record import _Recording

        def make(worlds, grasped, steps, per=4):
            decisions = steps // per
            return _Recording(
                actions=np.zeros((steps, worlds, 5), np.float32),
                active=np.ones((steps, worlds), bool),
                success=np.concatenate([
                    np.zeros((steps - 1, worlds), bool),
                    np.ones((1, worlds), bool),
                ]),
                terminated=np.zeros((steps, worlds), bool),
                # THE STRATUM LIVES HERE NOW, at step 0. Setting only
                # physical_grasp_at_reset would encode it in the column that
                # holds the FINAL grasp state and is wrong on every recording
                # already written.
                caught_target=np.concatenate([
                    np.full((1, worlds), grasped, dtype=bool),
                    np.ones((steps - 1, worlds), bool),
                ]),
                ee_xyz=np.zeros((steps, worlds, 3), np.float32),
                gripper_opening=np.zeros((steps, worlds), np.float32),
                object_xyz=np.zeros((steps, worlds, 2, 3), np.float32),
                instruction_ids=np.full((worlds,), PLATE, np.int64),
                target_slots=np.zeros((worlds,), np.int64),
                reference_slots=np.ones((worlds,), np.int64),
                second_reference_slots=np.full((worlds,), -1, np.int64),
                horizons=np.full((worlds,), decisions, np.int64),
                initial_target_xyz=np.zeros((worlds, 3), np.float32),
                support_surface_z=np.zeros((worlds,), np.float32),
                release_threshold=np.full((worlds,), 0.55, np.float32),
                target_rest_height=np.zeros((worlds,), np.float32),
                physical_grasp_at_reset=np.full((worlds,), grasped, bool),
                instructions=np.array(["put the apple into the plate"] * worlds),
                actions_per_decision=per, round_index=0, diverged_worlds=0,
                pick_lift_success_height=0.05,
                states=np.zeros((decisions, worlds, 518), np.float32),
                priors=np.zeros((decisions, worlds, per, 5), np.float32),
                start_distance_cap=0.20,
            )

        _, stats = _build_dataset(
            [make(composed_episodes, False, 128), make(caught_episodes, True, 20)],
            ["o7_demos", "p3_demos"],
            ["o7/replay_00", "p3/replay_00"],
            rows_per_instruction=0,
        )
        return stats["by_instruction"]["put_into_plate"]["pool_strata"]

    def test_the_probe_reports_it_with_no_quota_at_all(self) -> None:
        pool = self._pool(200, 400)
        self.assertIsNotNone(pool)
        self.assertEqual(pool["composed_episodes"], 200)
        self.assertEqual(pool["caught_episodes"], 400)

    def test_the_decision_fraction_is_the_one_that_binds(self) -> None:
        # A third of the EPISODES and three quarters of the DECISIONS. Reading
        # the episode share would predict a mix the quota cannot produce.
        pool = self._pool(200, 400)
        episode_share = pool["composed_episodes"] / (
            pool["composed_episodes"] + pool["caught_episodes"]
        )
        self.assertAlmostEqual(episode_share, 1 / 3, places=2)
        self.assertAlmostEqual(pool["composed_decision_fraction"], 0.7619, places=3)
        self.assertGreater(pool["composed_decision_fraction"], episode_share)

    def test_it_reports_the_length_asymmetry_that_causes_that(self) -> None:
        pool = self._pool(200, 400)
        self.assertAlmostEqual(pool["composed_decisions_per_episode"], 32.0, places=1)
        self.assertAlmostEqual(pool["caught_decisions_per_episode"], 5.0, places=1)

    def test_a_single_stratum_pool_reports_the_absent_side_as_zero(self) -> None:
        pool = self._pool(200, 0)
        self.assertEqual(pool["caught_episodes"], 0)
        self.assertIsNone(pool["caught_decisions_per_episode"])
        self.assertEqual(pool["composed_decision_fraction"], 1.0)


class TheSweepScriptTests(unittest.TestCase):
    SCRIPT = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/run_cdpr_phase7_composed_fraction_sweep.sh"
    )

    def setUp(self) -> None:
        self.text = self.SCRIPT.read_text(encoding="utf-8")

    def test_every_arm_spends_the_same_budget(self) -> None:
        """Otherwise the comparison is between slice SIZES, not mixes.

        The availability read is reused rather than repeated -- the pool has
        not changed, so the balanced quota has not either.
        """

        self.assertIn('--rows-per-instruction "$QUOTA"', self.text)
        self.assertEqual(self.text.count('--rows-per-instruction "$QUOTA"'), 1)

    def test_the_deciding_pair_runs_before_the_retention_legs(self) -> None:
        # An interrupted sweep should still leave a complete trade curve.
        deciding = self.text.index("run_eval \"$tag\" composed")
        retention = self.text.index("run_eval \"$tag\" pick_up")
        self.assertLess(deciding, retention)

    def test_it_evaluates_retention_even_though_those_slices_are_identical(self) -> None:
        # The residual is trained jointly, so a put_into mix can perturb
        # move_to and pick_up through shared weights even when their data is
        # byte-identical across arms.
        for leg in ("pick_up", "move_to", "placement_caught", "composed"):
            self.assertIn(f'run_eval "$tag" {leg}', self.text)

    def test_the_composed_eval_forces_uncaught_starts(self) -> None:
        self.assertIn("placement_caught_object_fraction=0.0", self.text)
        self.assertIn('run_eval "$tag" composed 0.20 "$COMPOSE_CONFIG" "${COMPOSED[@]}"', self.text)

    def test_stages_are_guarded_so_the_sweep_resumes(self) -> None:
        for artefact in (
            'dataset7_$tag/demonstrations.npz',
            'refreshed7_$tag/demonstrations.npz',
            'sft_phase7_$tag/sil_sft_adapter.pt',
        ):
            self.assertIn(f'[[ ! -f "$BANK/{artefact}" ]]', self.text)

    def test_it_reports_realized_and_not_only_requested(self) -> None:
        self.assertIn("realized_composed_fraction", self.text)
        self.assertIn("SHORT", self.text)

    def test_it_refuses_arms_the_pool_cannot_reach(self) -> None:
        """The check the first sweep did not have.

        Three arms at 0.2 / 0.4 / 0.6 all realised 0.981, because put_into's
        pool is ~98% composed by DECISION and the spill had nowhere else to go.
        The reachable floor is (quota - caught_decisions) / quota, and an arm
        below it is the same mix under a different name.
        """

        self.assertIn("reachable composed", self.text)
        self.assertIn("(quota - caught) / quota", self.text)
        self.assertIn("WARNING: arms", self.text)

    def test_the_preflight_runs_before_the_first_gpu_stage(self) -> None:
        pool = self.text.index("pool composition, per instruction")
        first_gpu = self.text.index("sil_refresh_priors.py")
        self.assertLess(pool, first_gpu)
        # And before DRY_RUN exits, so DRY_RUN=1 shows the reachable range.
        dry = self.text.index('DRY_RUN=1, stopping before the first GPU stage')
        self.assertLess(pool, dry)


if __name__ == "__main__":
    unittest.main()
