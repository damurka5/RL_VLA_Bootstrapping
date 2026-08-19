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
                lora_epochs=8,
                lora_rows=64,
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

    def test_the_sft_stage_receives_the_frames_and_the_dataset(self):
        """The loop is only sequential if the SFT step gets both halves."""

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            replay = Path(tmp) / "replay"
            replay.mkdir(parents=True)
            for name in ("frames_a.npz", "frames_b.npz", "replay_a.npz"):
                (replay / name).write_bytes(b"")
            commands = self._commands(tmp)
        sft = [c for c in commands if c[1].endswith("sil_sft.py")]
        self.assertEqual(len(sft), 1)
        self.assertIn("--frames", sft[0])
        self.assertEqual(
            sum(1 for part in sft[0] if part.endswith(".npz") and "frames_" in part),
            2,
        )
        self.assertIn("demonstrations.npz", " ".join(sft[0]))

    def test_an_unproven_candidate_does_not_become_the_resume_checkpoint(self):
        """A reject must hand RL back the checkpoint it already had.

        An unresolved difference is "no evidence", not "no effect", and
        resuming from an unproven candidate is how a loop drifts on noise.
        """

        import tempfile

        from tools.audit import sil_loop

        with tempfile.TemporaryDirectory() as tmp:
            replay = Path(tmp) / "replay"
            replay.mkdir(parents=True)
            (replay / "frames_a.npz").write_bytes(b"")
            original = sil_loop._run
            sil_loop._run = lambda command, dry_run: None
            try:
                report = sil_loop.harvest_iteration(
                    checkpoint=Path("pre_sft.pt"), config=Path("cfg.yaml"),
                    output=Path(tmp), instruction="move_to_object",
                    rungs=[0.03], rounds=1, smooth_window=5, seed_torch=0,
                    frame_worlds=0, lora_epochs=1, lora_rows=64,
                    dry_run=True,
                )
            finally:
                sil_loop._run = original
        self.assertFalse(report["verdict"]["accepted"])
        self.assertEqual(report["resume_checkpoint"], "pre_sft.pt")

    def test_the_top_rung_harvest_is_reused_as_the_baseline(self):
        """Same checkpoint, same cap, same seed -- paying twice buys nothing."""

        import tempfile

        from tools.audit import sil_loop

        with tempfile.TemporaryDirectory() as tmp:
            replay = Path(tmp) / "replay"
            replay.mkdir(parents=True)
            (replay / "frames_a.npz").write_bytes(b"")
            original = sil_loop._run
            sil_loop._run = lambda command, dry_run: None
            try:
                report = sil_loop.harvest_iteration(
                    checkpoint=Path("pre_sft.pt"), config=Path("cfg.yaml"),
                    output=Path(tmp), instruction="move_to_object",
                    rungs=[0.03, 0.09], rounds=2, smooth_window=5,
                    seed_torch=0, frame_worlds=0, lora_epochs=1,
                    lora_rows=64, dry_run=True,
                )
            finally:
                sil_loop._run = original
        self.assertTrue(report["baseline_dir"].endswith("harvest/cap_0.090"))

    def test_action_stats_run_on_the_replays_not_the_harvest(self):
        """The dataset is built from the smoothed rollout; score that."""

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            replay = Path(tmp) / "replay"
            replay.mkdir(parents=True)
            for name in ("replay_a.npz", "frames_a.npz"):
                (replay / name).write_bytes(b"")
            commands = self._commands(tmp)
        stats = [c for c in commands if c[1].endswith("sil_action_stats.py")]
        self.assertEqual(len(stats), 1)
        self.assertIn("--successes-only", stats[0])
        joined = " ".join(stats[0])
        self.assertIn("replay_a.npz", joined)
        self.assertNotIn("harvest", joined.split("--output")[0])


class CandidateSelectionTests(unittest.TestCase):
    """Which epoch gets rolled out, decided before any rollout happens."""

    def test_the_candidate_is_pre_registered_by_validation_mse(self):
        """Not the best of three simulated points.

        Taking the maximum of three noisy rates and then testing that maximum
        inflates the result by exactly what the selection gained -- and the
        per-round spread here swings a rate by 0.22 at one cap.
        """

        from tools.audit.sil_loop import select_candidate

        chosen = select_candidate(
            [
                {"epoch": 8, "val_mse": 0.031},
                {"epoch": 16, "val_mse": 0.027},
                {"epoch": 25, "val_mse": 0.029},
            ]
        )
        self.assertEqual(chosen["epoch"], 16)

    def test_no_scored_candidate_returns_none(self):
        from tools.audit.sil_loop import select_candidate

        self.assertIsNone(select_candidate([{"epoch": 1}]))
        self.assertIsNone(
            select_candidate([{"epoch": 1, "val_mse": float("nan")}])
        )


class VerdictTests(unittest.TestCase):
    """accept_or_reject, driven through real record summaries."""

    def _collectable(self, directory, rates, episodes=512):
        """A record summary in the shape sil_eval_table._collect reads."""

        import json

        directory.mkdir(parents=True, exist_ok=True)
        payload = {"mode": "record"}
        for index, rate in enumerate(rates):
            payload[f"run_{index:02d}"] = {
                "by_instruction": {
                    "move_to_object": {
                        "episodes": episodes,
                        "successes": int(round(rate * episodes)),
                        "source_success_rate": float(rate),
                    }
                }
            }
        (directory / "summary_00.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_a_consistent_gain_is_accepted(self):
        import tempfile

        from tools.audit.sil_loop import accept_or_reject

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            cand = Path(tmp) / "cand"
            self._collectable(base, [0.40, 0.44, 0.42, 0.41])
            self._collectable(cand, [0.52, 0.56, 0.54, 0.53])
            out = accept_or_reject(base, cand, instruction="move_to_object")
        self.assertTrue(out["accepted"], out)
        self.assertGreater(out["delta"], 0.0)

    def test_noise_around_zero_is_rejected(self):
        import tempfile

        from tools.audit.sil_loop import accept_or_reject

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            cand = Path(tmp) / "cand"
            self._collectable(base, [0.40, 0.44, 0.42, 0.41])
            self._collectable(cand, [0.44, 0.40, 0.43, 0.40])
            out = accept_or_reject(base, cand, instruction="move_to_object")
        self.assertFalse(out["accepted"])
        self.assertIn("no evidence", out["reason"])

    def test_a_consistent_loss_is_rejected_not_accepted_on_magnitude(self):
        import tempfile

        from tools.audit.sil_loop import accept_or_reject

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            cand = Path(tmp) / "cand"
            self._collectable(base, [0.50, 0.54, 0.52, 0.51])
            self._collectable(cand, [0.30, 0.34, 0.32, 0.31])
            out = accept_or_reject(base, cand, instruction="move_to_object")
        self.assertFalse(out["accepted"])
        self.assertLess(out["delta"], 0.0)

    def test_mismatched_denominators_are_refused(self):
        """Different resets; a delta across them scores the reset distribution."""

        import tempfile

        from tools.audit.sil_loop import accept_or_reject

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            cand = Path(tmp) / "cand"
            self._collectable(base, [0.40, 0.44, 0.42, 0.41], episodes=512)
            self._collectable(cand, [0.52, 0.56, 0.54, 0.53], episodes=256)
            out = accept_or_reject(base, cand, instruction="move_to_object")
        self.assertFalse(out["accepted"])
        self.assertIn("not comparable", out["reason"])


class ActionDriftTests(unittest.TestCase):
    """M4: the loop training on its own collapse."""

    @staticmethod
    def _row(aim, conc):
        return {"aim": aim, "direction_concentration": conc}

    def test_two_consecutive_falls_with_rising_concentration_halt(self):
        from tools.audit.sil_loop import DriftPolicy, check_action_drift

        out = check_action_drift(
            [self._row(0.15, 0.36), self._row(0.12, 0.42), self._row(0.08, 0.51)],
            policy=DriftPolicy(),
        )
        self.assertTrue(out["halt"], out)

    def test_a_falling_aim_alone_does_not_halt(self):
        """Concentration must rise with it; alone it accuses nothing.

        A genuine servo scores 0.82 where the arm sits systematically to one
        side of its goals, which is why it is only read together with aim.
        """

        from tools.audit.sil_loop import DriftPolicy, check_action_drift

        out = check_action_drift(
            [self._row(0.15, 0.40), self._row(0.12, 0.39), self._row(0.08, 0.38)],
            policy=DriftPolicy(),
        )
        self.assertFalse(out["halt"])

    def test_rising_concentration_alone_does_not_halt(self):
        from tools.audit.sil_loop import DriftPolicy, check_action_drift

        out = check_action_drift(
            [self._row(0.12, 0.36), self._row(0.14, 0.44), self._row(0.15, 0.52)],
            policy=DriftPolicy(),
        )
        self.assertFalse(out["halt"])

    def test_one_dip_between_two_good_iterations_does_not_halt(self):
        from tools.audit.sil_loop import DriftPolicy, check_action_drift

        out = check_action_drift(
            [self._row(0.15, 0.36), self._row(0.10, 0.44), self._row(0.16, 0.52)],
            policy=DriftPolicy(),
        )
        self.assertFalse(out["halt"])

    def test_two_iterations_are_not_enough_to_see_two_falls(self):
        from tools.audit.sil_loop import DriftPolicy, check_action_drift

        out = check_action_drift(
            [self._row(0.15, 0.36), self._row(0.08, 0.52)], policy=DriftPolicy()
        )
        self.assertFalse(out["halt"])
        self.assertIn("need three points", out["reason"])

    def test_iterations_without_stats_are_skipped_not_counted(self):
        from tools.audit.sil_loop import DriftPolicy, check_action_drift

        out = check_action_drift(
            [
                self._row(0.15, 0.36),
                {"aim": None, "direction_concentration": None},
                self._row(0.12, 0.44),
                self._row(0.08, 0.52),
            ],
            policy=DriftPolicy(),
        )
        self.assertTrue(out["halt"], out)


class PlateauWindowTests(unittest.TestCase):
    """The still-window only grows; the plateau test must not use all of it.

    Iteration 0 sat at cap 0.19 for 8.2M steps. Across the whole still-window
    the EMA halves differ by +0.078 -- the early climb after the promotion is
    still in there and always will be -- while the last ten checkpoints differ
    by -0.003. Judging the plateau over the full window makes a LONGER stall
    harder to detect, which is backwards.
    """

    @staticmethod
    def _climb_then_flat():
        # 20 checkpoints climbing 0.41 -> 0.59, then 20 flat at ~0.585.
        rows = []
        step = 2_800_000
        for i in range(20):
            rows.append((step, 0.19, 0.41 + 0.009 * i))
            step += 200_000
        for i in range(20):
            rows.append((step, 0.19, 0.585 + (0.004 if i % 2 else -0.004)))
            step += 200_000
        return _history(*rows)

    def test_the_real_shape_fires_on_the_recent_tail(self):
        history = self._climb_then_flat()
        decision = evaluate_trigger(
            history,
            policy=StallPolicy(
                cap_still_steps=600_000,
                min_steps_since_sft=1_000_000,
                promote_pass_rate=0.30,
                plateau_window_steps=2_000_000,
            ),
            ladder_top=0.19,
            last_sft_step=0,
            cap_promoted_since_sft=False,
        )
        self.assertTrue(decision.fire, decision.blocked_by)

    def test_the_whole_window_would_have_blocked_it(self):
        """Pinning the bug, so a future widening of the window is caught."""

        history = self._climb_then_flat()
        decision = evaluate_trigger(
            history,
            policy=StallPolicy(
                cap_still_steps=600_000,
                min_steps_since_sft=1_000_000,
                promote_pass_rate=0.30,
                plateau_window_steps=100_000_000,
            ),
            ladder_top=0.19,
            last_sft_step=0,
            cap_promoted_since_sft=False,
        )
        self.assertFalse(decision.fire)
        self.assertTrue(
            any("still rising" in line for line in decision.blocked_by)
        )

    def test_a_climb_inside_the_recent_window_still_blocks(self):
        """Narrowing the window must not make it fire on anything."""

        rows = []
        step = 2_800_000
        for i in range(20):
            rows.append((step, 0.19, 0.30 + 0.015 * i))
            step += 200_000
        decision = evaluate_trigger(
            _history(*rows),
            policy=StallPolicy(
                cap_still_steps=600_000,
                min_steps_since_sft=1_000_000,
                promote_pass_rate=0.30,
                plateau_window_steps=2_000_000,
            ),
            ladder_top=0.19,
            cap_promoted_since_sft=False,
        )
        self.assertFalse(decision.fire)


class LadderTopToleranceTests(unittest.TestCase):
    """The cap is accumulated, so the top never arrives as the literal value.

    0.03 plus eight increments of 0.02 is 0.18999999999999997. At a 1e-9
    tolerance the top rung is never recognised, and the condition that lets the
    loop fire when there is no further promotion to wait for is silently dead.
    """

    def test_an_accumulated_top_rung_counts_as_the_top(self):
        cap = 0.03
        for _ in range(8):
            cap += 0.02
        self.assertNotEqual(cap, 0.19)
        history = _history(
            (4_000_000, cap, 0.60),
            (4_600_000, cap, 0.61),
            (5_200_000, cap, 0.60),
            (5_800_000, cap, 0.61),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            cap_promoted_since_sft=False,
        )
        self.assertTrue(decision.at_ladder_top)
        self.assertTrue(decision.fire, decision.blocked_by)

    def test_a_rung_below_the_top_is_still_below_it(self):
        """The tolerance must not swallow a whole rung."""

        history = _history(
            (4_000_000, 0.17, 0.60),
            (4_600_000, 0.17, 0.61),
            (5_200_000, 0.17, 0.60),
            (5_800_000, 0.17, 0.61),
        )
        decision = evaluate_trigger(
            history,
            policy=POLICY,
            ladder_top=0.19,
            cap_promoted_since_sft=False,
        )
        self.assertFalse(decision.at_ladder_top)


class VisionLoraWiringTests(unittest.TestCase):
    """The design turns the tower on at the first SFT; the driver must ask."""

    def test_the_driver_requests_the_vision_tower(self):
        import tempfile

        from tools.audit import sil_loop

        seen = []
        original = sil_loop._run
        sil_loop._run = lambda command, dry_run: seen.append(list(command))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "replay").mkdir(parents=True)
                sil_loop.harvest_iteration(
                    checkpoint=Path("ckpt.pt"), config=Path("cfg.yaml"),
                    output=Path(tmp), instruction="move_to_object",
                    rungs=[0.03], rounds=1, smooth_window=5, seed_torch=0,
                    frame_worlds=0, lora_epochs=1, lora_rows=64,
                    dry_run=True,
                )
        finally:
            sil_loop._run = original
        sft = [c for c in seen if c[1].endswith("sil_sft.py")][0]
        self.assertIn("--train-vision-lora", sft)


class SingleRoundVerdictTests(unittest.TestCase):
    """One round is not a weak measurement; it is no measurement.

    _paired cannot form a spread from a single pair, so it returns
    resolved=False whatever the numbers are. A generic REJECT printed beside a
    visible +0.07 delta reads as evidence against the candidate, which it is
    not.
    """

    def _write(self, directory, rates, episodes=512):
        import json

        directory.mkdir(parents=True, exist_ok=True)
        payload = {"mode": "record"}
        for index, rate in enumerate(rates):
            payload[f"run_{index:02d}"] = {
                "by_instruction": {
                    "move_to_object": {
                        "episodes": episodes,
                        "successes": int(round(rate * episodes)),
                        "source_success_rate": float(rate),
                    }
                }
            }
        (directory / "summary_00.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_one_round_says_it_cannot_decide(self):
        import tempfile

        from tools.audit.sil_loop import accept_or_reject

        with tempfile.TemporaryDirectory() as tmp:
            base, cand = Path(tmp) / "b", Path(tmp) / "c"
            self._write(base, [0.613])
            self._write(cand, [0.685])
            out = accept_or_reject(base, cand, instruction="move_to_object")
        self.assertFalse(out["accepted"])
        self.assertIn("--rounds 4", out["reason"])
        self.assertAlmostEqual(out["delta"], 0.072, places=3)

    def test_two_rounds_get_a_real_test(self):
        import tempfile

        from tools.audit.sil_loop import accept_or_reject

        with tempfile.TemporaryDirectory() as tmp:
            base, cand = Path(tmp) / "b", Path(tmp) / "c"
            self._write(base, [0.613, 0.600])
            self._write(cand, [0.685, 0.690])
            out = accept_or_reject(base, cand, instruction="move_to_object")
        self.assertIn("resolved", out)
        self.assertNotIn("--rounds 4", out["reason"])
