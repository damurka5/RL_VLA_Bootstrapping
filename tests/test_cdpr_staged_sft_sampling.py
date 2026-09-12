"""Scene-level splits, stage-balanced exposure, retention and the stale guard.

Each of these has a specific failure behind it.

EPISODE SPLITTING IS NOT ENOUGH FOR A SCENE-SHARED BANK. Two rollouts of one
scene differ by the SmolVLA prior's fresh noise draw and almost nothing else.
One in train and one in validation is memorization with extra steps, and the
validation curve looks unusually good while it happens.

BALANCING MUST CHANGE EXPOSURE, NOT LENGTH. A placement carry is roughly three
times a pickup in decisions, so a natural pass gives it three times the
gradient purely from duration. The fix is which rows are DRAWN. Truncating the
carry to the pickup's length, or padding the pickup with repeated frames, would
"balance" the bank by corrupting it -- and the design forbids both by name.

AN EMPTY STRATUM IS AN ERROR. Substituting plate rows for a missing (bowl,
pick_up) cell produces a run that trains on one destination while every log
says two. The phase-7 composed-fraction sweep ran three arms that all realized
0.981 because the pool could not supply what was asked; that is the same bug
one level up.

A RELABELLED BANK WITH STALE PRIORS IS NOT TRAINABLE. Its text says "put apple
into plate" and its prior was drawn under "move to apple". Nothing about the
loss curve says so, so the refusal has to be mechanical.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit.sil_sft import (
    BalancedRowSampler,
    RetentionMixer,
    _scene_split,
    per_group_reachability,
    refuse_stale_priors,
    resolve_frame_rows_by_uid,
)


def _bank(*, scenes=4, rollouts_per_scene=2, stage_lengths=(3, 3, 9)):
    """A bank whose stages have realistic, UNEQUAL lengths."""

    columns: dict[str, list] = {
        "scene_uid": [],
        "episode_uid": [],
        "destination": [],
        "stage_name": [],
        "target_catalog": [],
        "decision_index": [],
        "frame_uid": [],
    }
    names = ("move_to", "pick_up", "placement")
    for scene in range(scenes):
        destination = "plate" if scene % 2 == 0 else "bowl"
        catalog = "robocasa_apple" if scene % 4 < 2 else "robocasa_orange"
        for rollout in range(rollouts_per_scene):
            uid = f"tag/r{rollout}w{scene}"
            decision = 0
            for stage, length in zip(names, stage_lengths):
                for _ in range(length):
                    columns["scene_uid"].append(f"scene_{scene:04d}")
                    columns["episode_uid"].append(uid)
                    columns["destination"].append(destination)
                    columns["stage_name"].append(stage)
                    columns["target_catalog"].append(catalog)
                    columns["decision_index"].append(decision)
                    columns["frame_uid"].append(f"{uid}#{decision}")
                    decision += 1
    return {
        key: np.asarray(values)
        for key, values in columns.items()
    }


class SceneSplitTests(unittest.TestCase):
    def test_a_scene_never_appears_on_both_sides(self):
        bank = _bank(scenes=10)
        train, val = _scene_split(
            bank["scene_uid"], val_fraction=0.3, seed=7
        )
        self.assertTrue((train | val).all())
        self.assertFalse((train & val).any())
        shared = set(bank["scene_uid"][train]) & set(bank["scene_uid"][val])
        self.assertEqual(shared, set())

    def test_both_rollouts_of_a_scene_travel_together(self):
        bank = _bank(scenes=8, rollouts_per_scene=3)
        train, val = _scene_split(
            bank["scene_uid"], val_fraction=0.25, seed=3
        )
        for scene in np.unique(bank["scene_uid"]):
            rows = bank["scene_uid"] == scene
            self.assertIn(
                (bool(train[rows].all()), bool(val[rows].all())),
                [(True, False), (False, True)],
            )

    def test_a_single_scene_bank_is_refused_rather_than_split(self):
        with self.assertRaises(SystemExit):
            _scene_split(
                np.array(["scene_0000"] * 20), val_fraction=0.1, seed=1
            )


class BalancedSamplerTests(unittest.TestCase):
    def test_exposure_is_equal_despite_unequal_stage_lengths(self):
        bank = _bank(scenes=8, stage_lengths=(3, 3, 9))
        rows = np.ones(bank["scene_uid"].shape[0], dtype=bool)
        # The raw bank is dominated by placement, which is the problem.
        natural = {
            stage: float((bank["stage_name"] == stage).mean())
            for stage in ("move_to", "pick_up", "placement")
        }
        self.assertGreater(natural["placement"], 0.55)

        sampler = BalancedRowSampler(bank, rows, seed=13)
        drawn = sampler.draw(30_000)
        for stage in ("move_to", "pick_up", "placement"):
            share = float((bank["stage_name"][drawn] == stage).mean())
            self.assertAlmostEqual(share, 1 / 3, delta=0.02)
        for destination in ("plate", "bowl"):
            share = float((bank["destination"][drawn] == destination).mean())
            self.assertAlmostEqual(share, 0.5, delta=0.02)

    def test_balancing_does_not_change_the_bank(self):
        bank = _bank(scenes=4)
        before = {key: value.copy() for key, value in bank.items()}
        sampler = BalancedRowSampler(
            bank, np.ones(bank["scene_uid"].shape[0], dtype=bool), seed=2
        )
        sampler.draw(500)
        for key, value in before.items():
            np.testing.assert_array_equal(bank[key], value)

    def test_it_draws_only_from_the_rows_it_was_given(self):
        bank = _bank(scenes=6)
        train, _ = _scene_split(bank["scene_uid"], val_fraction=0.5, seed=9)
        sampler = BalancedRowSampler(bank, train, seed=4)
        drawn = sampler.draw(2000)
        self.assertTrue(bool(train[drawn].all()))

    def test_an_empty_stratum_is_named_rather_than_substituted(self):
        bank = _bank(scenes=4)
        # Remove every bowl pick_up row, as a bank whose bowl chains all failed
        # at the grasp would look.
        keep = ~(
            (bank["destination"] == "bowl") & (bank["stage_name"] == "pick_up")
        )
        with self.assertRaises(SystemExit) as caught:
            BalancedRowSampler(bank, keep, seed=1)
        self.assertIn("bowl/pick_up", str(caught.exception))

    def test_a_bank_without_the_columns_is_told_which_ones(self):
        bank = _bank(scenes=2)
        del bank["stage_name"]
        with self.assertRaises(SystemExit) as caught:
            BalancedRowSampler(
                bank, np.ones(bank["scene_uid"].shape[0], bool), seed=1
            )
        self.assertIn("stage_name", str(caught.exception))


class RetentionTests(unittest.TestCase):
    def test_a_declared_share_is_actually_drawn(self):
        retention = {"state": np.zeros((50, 6), np.float32)}
        mixer = RetentionMixer(retention, fraction=0.2, seed=1)
        main, retained = mixer.split_counts(100)
        self.assertEqual((main, retained), (80, 20))
        drawn = mixer.draw(retained)
        self.assertEqual(drawn.size, 20)
        self.assertTrue(bool((drawn < 50).all()))

    def test_retention_cannot_be_asked_for_without_a_bank(self):
        with self.assertRaises(SystemExit):
            RetentionMixer(None, fraction=0.2, seed=1)

    def test_a_bank_without_a_share_is_refused_rather_than_ignored(self):
        with self.assertRaises(SystemExit):
            RetentionMixer({"state": np.zeros((4, 6))}, fraction=0.0, seed=1)

    def test_off_by_default_leaves_the_batch_whole(self):
        mixer = RetentionMixer(None, fraction=0.0, seed=1)
        self.assertFalse(mixer.active)
        self.assertEqual(mixer.split_counts(64), (64, 0))


class StalePriorTests(unittest.TestCase):
    def test_a_stale_bank_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            (Path(directory) / "dataset.json").write_text(
                json.dumps(
                    {
                        "priors_stale": True,
                        "priors_stale_reason": "relabelled, not refreshed",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit) as caught:
                refuse_stale_priors(path, allow=False)
            self.assertIn("relabelled, not refreshed", str(caught.exception))

    def test_the_refusal_can_be_overridden_deliberately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            (Path(directory) / "dataset.json").write_text(
                json.dumps({"priors_stale": True}), encoding="utf-8"
            )
            payload = refuse_stale_priors(path, allow=True)
            self.assertTrue(payload["priors_stale"])

    def test_a_refreshed_bank_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            (Path(directory) / "dataset.json").write_text(
                json.dumps({"priors_stale": False, "rows": 10}),
                encoding="utf-8",
            )
            self.assertEqual(
                refuse_stale_priors(path, allow=False)["rows"], 10
            )

    def test_a_bank_with_no_report_is_not_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                refuse_stale_priors(
                    Path(directory) / "demonstrations.npz", allow=False
                ),
                {},
            )


class FrameJoinTests(unittest.TestCase):
    def test_rows_join_by_explicit_episode_id(self):
        frames = {
            "tag/r0w0": {"path": "a.npz", "column": 0, "decisions": 5},
            "tag/r0w1": {"path": "a.npz", "column": 1, "decisions": 5},
        }
        found, lookups = resolve_frame_rows_by_uid(
            np.array(["tag/r0w1", "tag/r0w0", "tag/r9w9"]),
            np.array([2, 0, 0]),
            frames,
        )
        self.assertEqual(found.tolist(), [True, True, False])
        self.assertEqual(lookups, [("a.npz", 2, 1), ("a.npz", 0, 0)])

    def test_a_decision_past_the_recording_does_not_resolve(self):
        frames = {"tag/r0w0": {"path": "a.npz", "column": 0, "decisions": 3}}
        found, _ = resolve_frame_rows_by_uid(
            np.array(["tag/r0w0"]), np.array([7]), frames
        )
        self.assertFalse(bool(found[0]))


class ReachabilityReportTests(unittest.TestCase):
    def test_reachability_is_reported_per_stage_and_per_axis(self):
        bank = _bank(scenes=2, rollouts_per_scene=1, stage_lengths=(2, 2, 2))
        rows = np.arange(bank["scene_uid"].shape[0])
        bank["prior"] = np.zeros((rows.size, 8, 5), np.float32)
        bank["action"] = np.zeros((rows.size, 4, 5), np.float32)
        bank["action_mask"] = np.ones((rows.size, 4), bool)
        # One axis of the placement rows is pushed outside the residual's
        # bounded correction range: the report has to name that axis.
        placement = bank["stage_name"] == "placement"
        bank["action"][placement, :, 4] = 0.999
        report = per_group_reachability(
            bank, rows, residual_scale=0.5, column="stage_name"
        )
        self.assertLess(
            report["placement"]["reachable_fraction"],
            report["move_to"]["reachable_fraction"],
        )
        self.assertAlmostEqual(
            report["placement"]["by_axis"]["gripper"], 0.0, places=6
        )
        self.assertAlmostEqual(
            report["placement"]["by_axis"]["x"], 1.0, places=6
        )


class ThreeStageRunnerTests(unittest.TestCase):
    """The declared rollout checkpoints must not become a dead variable."""

    def test_check_epochs_drive_independent_training_and_evaluation(self):
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_cdpr_three_stage_sft.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'for checkpoint_epochs in "${ARM_B_EPOCHS[@]}"', runner
        )
        self.assertIn('--epochs "$checkpoint_epochs"', runner)
        self.assertIn(
            'evaluate "$arm_name" "$arm_output/sil_sft_adapter.pt"', runner
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
