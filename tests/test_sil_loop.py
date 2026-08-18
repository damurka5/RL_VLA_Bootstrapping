"""The stall trigger, and the ways it must refuse to fire.

Firing wrongly is expensive in both directions: a harvest on a rung RL was
about to clear costs 1.5-2 hours for nothing, and a harvest on a run whose gate
is broken turns a frozen run into a frozen loop.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.sil_loop import (  # noqa: E402
    CurriculumSample,
    StallPolicy,
    evaluate_trigger,
    harvest_ladder,
)


def _history(*rows):
    return [
        CurriculumSample(global_step=step, cap=cap, pass_rate_ema=ema)
        for step, cap, ema in rows
    ]


POLICY = StallPolicy(
    cap_still_steps=600_000,
    min_steps_since_sft=1_000_000,
    promote_pass_rate=0.30,
)


class TriggerTests(unittest.TestCase):
    def test_a_settled_rung_below_the_gate_fires(self):
        history = _history(
            (1_000_000, 0.05, 0.28),
            (1_400_000, 0.05, 0.24),
            (1_800_000, 0.05, 0.22),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            last_sft_step=0,
            cap_promoted_since_sft=True,
        )
        self.assertTrue(decision.fire, decision.blocked_by)
        self.assertEqual(decision.steps_since_cap_change, 800_000)

    def test_a_rising_ema_does_not_fire(self):
        """It will cross on its own; a dataset buys nothing here."""

        history = _history(
            (1_000_000, 0.05, 0.10),
            (1_400_000, 0.05, 0.18),
            (1_800_000, 0.05, 0.27),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            cap_promoted_since_sft=True,
        )
        self.assertFalse(decision.fire)
        self.assertTrue(
            any("still rising" in line for line in decision.blocked_by)
        )

    def test_a_freshly_promoted_cap_does_not_fire(self):
        """The rung has not had time to be a stall yet."""

        history = _history(
            (1_000_000, 0.03, 0.35),
            (1_200_000, 0.05, 0.20),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            cap_promoted_since_sft=True,
        )
        self.assertFalse(decision.fire)
        self.assertTrue(
            any("still for only" in line for line in decision.blocked_by)
        )

    def test_a_cap_that_never_promoted_does_not_fire(self):
        """Nothing new to harvest since the last SFT."""

        history = _history(
            (2_000_000, 0.05, 0.20),
            (2_800_000, 0.05, 0.19),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            last_sft_step=0,
            cap_promoted_since_sft=False,
        )
        self.assertFalse(decision.fire)
        self.assertTrue(
            any("has not promoted" in line for line in decision.blocked_by)
        )

    def test_the_ladder_top_lifts_the_promotion_requirement(self):
        """At the top there is no further promotion to wait for."""

        history = _history(
            (2_000_000, 0.19, 0.20),
            (2_800_000, 0.19, 0.19),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            last_sft_step=0,
            cap_promoted_since_sft=False,
        )
        self.assertTrue(decision.fire, decision.blocked_by)
        self.assertTrue(decision.at_ladder_top)

    def test_too_soon_after_the_last_sft_does_not_fire(self):
        history = _history(
            (2_000_000, 0.05, 0.20),
            (2_700_000, 0.05, 0.19),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            last_sft_step=2_000_000,
            cap_promoted_since_sft=True,
        )
        self.assertFalse(decision.fire)
        self.assertTrue(
            any("since the last SFT" in line for line in decision.blocked_by)
        )

    def test_an_exactly_zero_ema_is_a_dead_gate_and_never_fires(self):
        """The failure that cost ten hours, refused rather than harvested.

        A gate reading a metric the task never emits pins the EMA at exactly
        0.0 forever, which satisfies "below the threshold and not rising"
        perfectly. Harvesting there would turn a frozen run into a frozen loop.
        """

        history = _history(
            (500_000, 0.03, 0.0),
            (1_200_000, 0.03, 0.0),
            (1_900_000, 0.03, 0.0),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            cap_promoted_since_sft=True,
        )
        self.assertFalse(decision.fire)
        self.assertTrue(
            any("dead gate" in line for line in decision.blocked_by)
        )

    def test_a_genuine_zero_that_recovers_is_not_a_dead_gate(self):
        """Early training reads zero and must not be permanently accused."""

        history = _history(
            (500_000, 0.03, 0.0),
            (1_200_000, 0.05, 0.21),
            (1_900_000, 0.05, 0.20),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            cap_promoted_since_sft=True,
        )
        self.assertTrue(decision.fire, decision.blocked_by)

    def test_every_condition_is_reported_either_way(self):
        """watch mode has to say how far off it is, not just no."""

        history = _history((1_000_000, 0.05, 0.20))
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            cap_promoted_since_sft=True,
        )
        self.assertEqual(
            len(decision.reasons) + len(decision.blocked_by), 6
        )

    def test_no_checkpoints_is_not_a_stall(self):
        decision = evaluate_trigger(
            [], policy=POLICY, ladder_top=0.19, cap_promoted_since_sft=True
        )
        self.assertFalse(decision.fire)


class HarvestLadderTests(unittest.TestCase):
    def test_it_never_collects_above_the_cap_reached(self):
        """The brief's rule: no farther than the last cap actually earned."""

        self.assertEqual(
            harvest_ladder(0.09, rungs=[0.03, 0.05, 0.07, 0.09, 0.11]),
            [0.03, 0.05, 0.07, 0.09],
        )

    def test_a_cap_below_every_rung_still_collects_something(self):
        self.assertEqual(harvest_ladder(0.02, rungs=[0.03, 0.05]), [0.02])

    def test_the_top_rung_is_included_at_floating_point_equality(self):
        self.assertIn(0.19, harvest_ladder(0.19, rungs=[0.03, 0.19]))


if __name__ == "__main__":
    unittest.main()


class LadderTopTests(unittest.TestCase):
    """The top rung is a different question, and the first spec got it wrong.

    Below the top, "stalled" means the EMA cannot reach the promote gate. At the
    top there is no promotion to gate, and demanding the EMA sit under 0.30
    would require the policy to get WORSE before the loop would help it.
    Measured on iteration 0: at cap 0.19 the EMA reached 0.587 while success was
    still climbing at +0.097 per 100 updates.
    """

    def test_a_high_ema_at_the_top_does_not_block_a_plateau(self):
        history = _history(
            (4_000_000, 0.19, 0.60),
            (4_600_000, 0.19, 0.61),
            (5_200_000, 0.19, 0.60),
            (5_800_000, 0.19, 0.61),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            last_sft_step=0,
            cap_promoted_since_sft=False,
        )
        self.assertTrue(decision.fire, decision.blocked_by)
        self.assertTrue(decision.at_ladder_top)

    def test_the_real_iteration_zero_window_does_not_fire(self):
        """The run as it actually stands: climbing, so the loop must wait."""

        history = _history(
            (4_000_000, 0.19, 0.41),
            (4_600_000, 0.19, 0.47),
            (5_200_000, 0.19, 0.52),
            (5_800_000, 0.19, 0.59),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            last_sft_step=0,
            cap_promoted_since_sft=False,
        )
        self.assertFalse(decision.fire)
        self.assertTrue(
            any("still rising" in line for line in decision.blocked_by)
        )

    def test_below_the_top_a_high_ema_still_blocks(self):
        """There the EMA crossing 0.30 means RL promotes on its own."""

        history = _history(
            (4_000_000, 0.09, 0.60),
            (4_600_000, 0.09, 0.60),
            (5_200_000, 0.09, 0.60),
            (5_800_000, 0.09, 0.60),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            cap_promoted_since_sft=True,
        )
        self.assertFalse(decision.fire)
        self.assertTrue(
            any("about to promote" in line for line in decision.blocked_by)
        )

    def test_noise_in_one_checkpoint_does_not_read_as_rising(self):
        """Halves, not endpoints: a spike at either end must not decide."""

        history = _history(
            (4_000_000, 0.19, 0.60),
            (4_600_000, 0.19, 0.58),
            (5_200_000, 0.19, 0.59),
            (5_800_000, 0.19, 0.62),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            cap_promoted_since_sft=False,
        )
        self.assertTrue(decision.fire, decision.blocked_by)


class HarvestCommandTests(unittest.TestCase):
    """The subprocess commands, checked without running anything."""

    def _commands(self, tmp, **kwargs):
        from tools.audit import sil_loop

        seen = []
        original = sil_loop._run
        sil_loop._run = lambda command, dry_run: seen.append(list(command))
        try:
            sil_loop.harvest_iteration(
                checkpoint=Path("ckpt.pt"),
                config=Path("cfg.yaml"),
                output=Path(tmp),
                instruction="move_to_object",
                rungs=kwargs.get("rungs", [0.03, 0.05]),
                rounds=kwargs.get("rounds", 2),
                smooth_window=5,
                seed_torch=0,
                frame_worlds=0,
                dry_run=True,
            )
        finally:
            sil_loop._run = original
        return seen

    def test_the_dataset_glob_is_expanded_not_passed_through(self):
        """subprocess does not go through a shell.

        A literal replay_*.npz would arrive at sil_record as one nonexistent
        path and the dataset would be built from nothing.
        """

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            replay = Path(tmp) / "replay"
            replay.mkdir(parents=True)
            for name in ("replay_a.npz", "replay_b.npz"):
                (replay / name).write_bytes(b"")
            commands = self._commands(tmp)
        dataset = [c for c in commands if "dataset" in c][0]
        self.assertNotIn(
            "*", " ".join(dataset), "the glob reached the subprocess"
        )
        self.assertEqual(
            sum(1 for part in dataset if part.endswith(".npz")), 2
        )

    def test_every_recorded_round_is_replayed_at_its_own_rung(self):
        """Replaying a round against another rung's starts silently halves it."""

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            commands = self._commands(tmp, rungs=[0.03, 0.05], rounds=2)
        replays = [c for c in commands if "replay" in c]
        self.assertEqual(len(replays), 4)
        for command in replays:
            actions = command[command.index("--actions") + 1]
            cap = command[command.index("--start-distance-cap") + 1]
            self.assertIn(f"cap_{float(cap):.3f}", actions)

    def test_frames_are_requested_on_the_replay_and_not_the_record(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            commands = self._commands(tmp)
        for command in commands:
            if "record" in command and "--mode" in command:
                mode = command[command.index("--mode") + 1]
                if mode == "record":
                    self.assertNotIn("--record-frames", command)
        replays = [c for c in commands if "--mode" in c and c[c.index("--mode") + 1] == "replay"]
        self.assertTrue(replays)
        for command in replays:
            self.assertIn("--record-frames", command)
