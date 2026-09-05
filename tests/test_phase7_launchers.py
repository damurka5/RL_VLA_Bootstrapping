"""The two scripts that spend the next GPU day, and what they refuse.

Every check below corresponds to a failure this campaign has already paid for
once, and each of them looks like a working run until the result is read:

  a "sparse" run that silently kept the dense reward, because dropping the
  reward objects also moves the SUCCESS PREDICATE (plate radius 0.091 -> 0.03);

  a refill loop pinned at one round by arithmetic (grpo_max_groups_per_update
  64 against 64 groups per rank), so dynamic sampling only masks;

  a stop rule reading the caught-dominated metric, which is what spent 2.25M
  steps in phase 6 on a number that described the old task;

  a caught-side harvest run against a config whose caught fraction had been
  annealed away, which would bank composed episodes under a caught name.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TheCaughtHarvestTests(unittest.TestCase):
    SCRIPT = ROOT / "scripts/run_cdpr_phase7_caught_harvest.sh"

    def setUp(self) -> None:
        self.text = self.SCRIPT.read_text(encoding="utf-8")

    def test_it_exists_and_is_executable(self) -> None:
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))

    def test_it_refuses_a_config_that_would_bank_composed_episodes(self) -> None:
        """The one mistake that would waste the whole run silently.

        Harvesting against the composed config -- caught fraction annealed to
        0.0 -- produces composed episodes filed as caught, and nothing would
        show it until the pool read at the very end.
        """

        self.assertIn("placement_caught_object_fraction", self.text)
        self.assertIn("if caught < 0.99:", self.text)
        self.assertIn("placement_caught_curriculum_enabled", self.text)

    def test_it_uses_the_caught_placement_config(self) -> None:
        self.assertIn("cdpr_smolvla_phase4_placement_loop.yaml", self.text)
        self.assertNotIn("cdpr_smolvla_phase5_compose_loop.yaml", self.text)

    def test_it_smooths_because_these_are_policy_actions(self) -> None:
        # The composed harvest used --smooth none: the oracle is a P-D
        # controller and already smooth. These are the policy's own actions.
        self.assertIn('SMOOTH="${SMOOTH:-moving_average}"', self.text)

    def test_it_reports_the_reachable_floor_it_bought(self) -> None:
        # The point of the harvest, measured rather than assumed.
        self.assertIn("(quota - caught) / quota", self.text)
        self.assertIn("STILL ABOVE TARGET", self.text)

    def test_it_guards_frames_not_replays(self) -> None:
        self.assertIn(
            '[[ -f "$BANK/c7_demos/frames_c7_${cap}_${stem}.npz" ]] && continue',
            self.text,
        )


class TheSweepPicksUpTheHarvestTests(unittest.TestCase):
    SCRIPT = ROOT / "scripts/run_cdpr_phase7_composed_fraction_sweep.sh"

    def setUp(self) -> None:
        self.text = self.SCRIPT.read_text(encoding="utf-8")

    def test_the_new_caught_rounds_are_in_the_source_list(self) -> None:
        self.assertIn("c7_demos", self.text)

    def test_absent_sources_are_skipped_rather_than_passed_literally(self) -> None:
        # An unmatched glob would otherwise reach sil_record as a literal path
        # and fail after the availability read had started.
        self.assertIn("compgen -G", self.text)

    def test_the_probe_is_keyed_on_which_sources_went_into_it(self) -> None:
        """A pool that gained the caught harvest must not reuse the old read.

        The reachable-floor preflight trusts that probe; a stale one would
        clear arms the new pool cannot supply and block ones it can.
        """

        self.assertIn("POOL_KEY=", self.text)
        self.assertIn('PROBE="$BANK/dataset7_probe_$POOL_KEY"', self.text)


class TheSparseJointLauncherTests(unittest.TestCase):
    SCRIPT = ROOT / "scripts/train_cdpr_phase7_sparse_joint_remote.sh"

    def setUp(self) -> None:
        self.text = self.SCRIPT.read_text(encoding="utf-8")

    def test_it_exists_and_is_executable(self) -> None:
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))

    def test_it_warm_starts_weights_only_from_the_seed(self) -> None:
        # Under a binary reward a family at zero can never bootstrap: a carry
        # that never opens the gripper scores what one that never moved scores.
        # The SFT seed is the stage-1 cold start.
        self.assertIn("sft_phase7/sil_sft_adapter.pt", self.text)
        self.assertIn("RLVLA_SMOLVLA_WARMSTART_CHECKPOINT", self.text)
        self.assertIn("unset RLVLA_SMOLVLA_RESUME_CHECKPOINT", self.text)

    def test_the_preflight_checks_every_way_the_run_can_look_fine_and_be_wrong(self) -> None:
        for probe in (
            "sparse_binary_reward is not set",
            "shaping weights survived the sparse flag",
            "success radii moved",
            "not all four",
            "grpo_target_informative_groups is 0",
            "the refill cannot take a second round",
            "steer on the caught-dominated metric",
            "does not fit in 32",
        ):
            self.assertIn(probe, self.text, probe)

    def test_the_preflight_refuses_a_sparse_run_with_no_exploration(self) -> None:
        """The check that would have saved 13 hours.

        Phase 7's first attempt ran 2.1M steps at episode_offset_std
        [0,0,0,0,0]: physical_release_rate 0.0771 -> 0.0535, grasp 0.2633 ->
        0.2084, entropy and log_std flat to four decimals, composed bowl to
        0.0000. Under a binary reward nothing pays for keeping the release once
        it drifts.
        """

        for probe in (
            # Matched as the source writes them: these messages are split
            # across adjacent string literals.
            "channel is zero. Under a sparse reward nothing then pays",
            "not on the",
            "GRIPPER channel (index 4), which is the one the release needs",
            'offsets = list(getattr(args, "episode_offset_std", []) or [])',
            "if not any(float(v) > 0.0 for v in offsets):",
        ):
            self.assertIn(probe, self.text, probe)

    def test_the_preflight_refuses_training_a_task_it_does_not_score(self) -> None:
        self.assertIn("placement_caught_object_fraction", self.text)
        self.assertIn("if caught > 0.75:", self.text)
        self.assertIn("approach_gate_uncaught_only is off", self.text)

    def test_the_preflight_refuses_a_ladder_that_parks_below_its_own_floor(self) -> None:
        """P - drop >= D, not a band-width rule.

        Every promotion costs pass rate because the next rung is harder --
        measured over twelve promotions: median 0.091, p90 0.129, max 0.133. A
        family that promotes from P lands near P - drop and stays there: below
        P it cannot promote, above D it will not fall back. Parking is not the
        bug; parking BELOW the level the config itself calls unacceptable is.

        The first version of this check compared the BAND width to the drop and
        had the inequality backwards -- it refused the fix and passed the
        configuration that had just ratcheted. Both directions are asserted
        here for that reason.
        """

        self.assertIn("promote - MEASURED_P90_DROP < demote", self.text)
        self.assertIn("ratchets down", self.text)

    def test_the_ladder_arithmetic_holds_both_ways(self) -> None:
        drop = 0.129

        def refuses(promote, demote):
            return promote - drop < demote - 1e-9

        # What ran, and parked three of four families at ~0.24.
        self.assertTrue(refuses(0.30, 0.20))
        # The replacement.
        self.assertFalse(refuses(0.45, 0.30))
        # Raising promote alone is enough; so is lowering demote alone.
        self.assertFalse(refuses(0.35, 0.20))
        self.assertTrue(refuses(0.45, 0.35))

    def test_the_config_carries_the_corrected_ladder(self) -> None:
        from rl_vla_bootstrapping.core.config import load_project_config

        metadata = dict(
            load_project_config(
                ROOT / "configs/examples/cdpr_smolvla_phase7_sparse_joint.yaml"
            ).task.metadata or {}
        )
        promote = float(
            metadata["random_workspace_start_distance_promote_pass_rate"]
        )
        demote = float(
            metadata["random_workspace_start_distance_demote_pass_rate"]
        )
        self.assertGreaterEqual(promote - 0.129, demote - 1e-9)
        # And the dwell/cooldown that damp the oscillation this trades for are
        # still present, since a tighter band makes them load-bearing.
        self.assertGreater(
            int(metadata["random_workspace_start_distance_promote_dwell_updates"]), 1
        )
        self.assertGreater(
            int(metadata["random_workspace_start_distance_cooldown_updates"]), 1
        )

    def test_the_preflight_runs_before_the_run_directory_is_made(self) -> None:
        preflight = self.text.index("[phase7] preflight clean")
        mkdir = self.text.index('mkdir -p "$RUN_DIR"')
        self.assertLess(preflight, mkdir)

    def test_it_guards_the_run_directory(self) -> None:
        self.assertIn("cdpr_guard_run_dir", self.text)
        self.assertIn("cdpr_compose_run_name", self.text)

    def test_it_says_which_metric_to_steer_on(self) -> None:
        self.assertIn("STEER ON validation_composed/", self.text)
        self.assertIn("usable_groups_collected", self.text)


class ThePreflightAgreesWithTheConfigsTests(unittest.TestCase):
    """Run the launcher's own checks against the real configs."""

    @staticmethod
    def _verdict(config_name: str, worlds: int = 512):
        import yaml

        from rl_vla_bootstrapping.core.commands import append_cli_arg
        from rl_vla_bootstrapping.core.config import load_project_config
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            sparse_binary_reward_requested,
        )

        path = ROOT / "configs/examples" / config_name
        project = load_project_config(path)
        metadata = dict(project.task.metadata or {})
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        section = next(v["args"] for v in raw["training"].values()
                       if isinstance(v, dict) and "args" in v)
        argv = []
        for key, value in section.items():
            append_cli_arg(argv, key, value)
        args = parse_args(argv)
        groups = worlds // int(args.grpo_group_size)
        return {
            "sparse": sparse_binary_reward_requested(metadata),
            "instructions": set(project.task.instruction_types or ()),
            "max_rounds": max(1, -(-int(args.grpo_max_groups_per_update) // groups)),
            "target_groups": int(args.grpo_target_informative_groups),
            "composed_val": int(args.composed_validation_episodes_per_instruction),
            "horizon": int(metadata.get("placement_grasp_horizon_min_decisions", 32)),
        }

    def test_the_config_now_trains_what_it_scores(self) -> None:
        from rl_vla_bootstrapping.core.config import load_project_config

        metadata = dict(
            load_project_config(
                ROOT / "configs/examples/cdpr_smolvla_phase7_sparse_joint.yaml"
            ).task.metadata or {}
        )
        # Half of every container batch is the composed task from step 0,
        # against the first attempt's 10-20%.
        self.assertLessEqual(
            float(metadata["placement_caught_object_fraction"]), 0.75
        )
        # ...and the caught stage is not switched off entirely: it is what
        # holds the carry the composed task ends with.
        self.assertGreater(float(metadata["placement_caught_object_fraction"]), 0.0)
        self.assertTrue(metadata["approach_gate_uncaught_only"])

    def test_exploration_is_on_the_gripper_channel(self) -> None:
        import yaml

        raw = yaml.safe_load(
            (ROOT / "configs/examples/cdpr_smolvla_phase7_sparse_joint.yaml")
            .read_text(encoding="utf-8")
        )
        args = next(v["args"] for v in raw["training"].values()
                    if isinstance(v, dict) and "args" in v)
        offsets = args["episode_offset_std"]
        # Order is [x, y, z, yaw, gripper]; only the release is perturbed.
        self.assertEqual(len(offsets), 5)
        self.assertGreater(float(offsets[4]), 0.0)
        self.assertTrue(all(float(v) == 0.0 for v in offsets[:4]))
        # Below the 0.3 the compose config argues for: that value stalled the
        # approach ladder at the upper rungs, and this run is at them.
        self.assertLess(float(offsets[4]), 0.3)

    def test_the_sparse_joint_config_passes_every_check(self) -> None:
        v = self._verdict("cdpr_smolvla_phase7_sparse_joint.yaml")
        self.assertTrue(v["sparse"])
        self.assertEqual(
            v["instructions"],
            {"move_to_object", "pick_up", "put_into_plate", "put_into_bowl"},
        )
        self.assertGreater(v["max_rounds"], 1)
        self.assertGreater(v["target_groups"], 0)
        self.assertGreater(v["composed_val"], 0)
        self.assertGreaterEqual(v["horizon"], 40)

    def test_the_compose_loop_config_would_be_refused(self) -> None:
        # It is a good config for what it is -- the placement-family RL leg --
        # and a wrong one to launch a sparse joint run with.
        v = self._verdict("cdpr_smolvla_phase5_compose_loop.yaml")
        self.assertFalse(v["sparse"])
        self.assertNotEqual(len(v["instructions"]), 4)
        self.assertEqual(v["max_rounds"], 1)
        self.assertEqual(v["target_groups"], 0)


if __name__ == "__main__":
    unittest.main()
