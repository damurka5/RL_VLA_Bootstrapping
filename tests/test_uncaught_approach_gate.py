"""The approach gate must read the composed task, not a blend of it and the carry.

A caught placement episode DOES perform an approach -- the gripper carries the
object to the receptacle -- so it sits inside `instruction_successes_normal_start`
by construction. While the caught-fraction curriculum anneals, that pair is a
BLEND whose mixture is the knob's current value.

Measured on phase 7 at caught fraction 0.8: the blended gate read 0.42 for
put_into_plate and promoted the cap 0.02 -> 0.17 in four steps, while the
composed validation of the same policy read 0.013 and its composed bowl reached
exactly 0.000 on 264 episodes. The cap advanced on the carry-only task the run
had already learned. That is the phase-6 error -- a number that moves with the
knob rather than the policy -- in a second place.

The fallback is as load-bearing as the split. At caught fraction 1.0 there are
no uncaught worlds, and a gate reading 0/0 does not fail, it FREEZES: the pass
rate is a clean 0.0, the EMA decays, promotion is never due, and every other
curve looks healthy. That failure has cost this curriculum three separate runs.
"""

from __future__ import annotations

import unittest

import torch

from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    instruction_outcome_counts,
)

PLATE, BOWL, PICK_UP = 4, 3, 8


def _counts(*, caught_flags, successes, instruction=PLATE, prelifted=None):
    """One group per entry; `successes` is per group, per candidate."""

    groups = len(caught_flags)
    task_ids = torch.full((groups,), instruction, dtype=torch.int64)
    return instruction_outcome_counts(
        torch.tensor(successes, dtype=torch.bool),
        task_ids,
        {"put_into_plate": PLATE, "put_into_bowl": BOWL, "pick_up": PICK_UP},
        None if prelifted is None else torch.tensor(prelifted, dtype=torch.bool),
        None,
        torch.tensor(caught_flags, dtype=torch.bool),
    )


class TheBlendIsWhatTheOldGateSawTests(unittest.TestCase):
    def test_the_blended_pair_hides_a_failing_composed_task(self) -> None:
        """Phase 7's numbers, reconstructed.

        Four caught groups that mostly succeed and one uncaught group that does
        not: the blended rate is comfortably over the 0.30 promote gate while
        the composed rate is zero.
        """

        counts = _counts(
            caught_flags=[True, True, True, True, False],
            successes=[[True] * 4] * 4 + [[False] * 4],
        )
        blended = (
            counts["instruction_successes_normal_start/put_into_plate"]
            / counts["instruction_worlds_normal_start/put_into_plate"]
        )
        composed = (
            counts["instruction_successes_uncaught_start/put_into_plate"]
            / counts["instruction_worlds_uncaught_start/put_into_plate"]
        )
        self.assertAlmostEqual(blended, 0.8)
        self.assertAlmostEqual(composed, 0.0)
        self.assertGreater(blended, 0.30)   # promotes
        self.assertLess(composed, 0.30)     # must not

    def test_the_uncaught_pair_counts_only_uncaught_groups(self) -> None:
        counts = _counts(
            caught_flags=[True, False, False],
            successes=[[True] * 4, [True, False, False, False], [False] * 4],
        )
        self.assertEqual(
            counts["instruction_worlds_uncaught_start/put_into_plate"], 8.0
        )
        self.assertEqual(
            counts["instruction_successes_uncaught_start/put_into_plate"], 1.0
        )
        # ...and the blended pair still counts everything, unchanged.
        self.assertEqual(
            counts["instruction_worlds_normal_start/put_into_plate"], 12.0
        )
        self.assertEqual(
            counts["instruction_successes_normal_start/put_into_plate"], 5.0
        )

    def test_prelifted_exclusion_still_applies_on_top(self) -> None:
        # The uncaught split is ANDed with the approach mask, not a replacement
        # for it: a start that skipped the approach must stay excluded however
        # it was seeded.
        counts = _counts(
            caught_flags=[False, False],
            successes=[[True] * 4, [True] * 4],
            prelifted=[True, False],
        )
        self.assertEqual(
            counts["instruction_worlds_uncaught_start/put_into_plate"], 4.0
        )


class TheCountsAreAbsentWhenNotAskedForTests(unittest.TestCase):
    def test_no_caught_mask_means_no_new_keys(self) -> None:
        """Every run that predates this must be byte-identical."""

        counts = instruction_outcome_counts(
            torch.ones((2, 4), dtype=torch.bool),
            torch.full((2,), PLATE, dtype=torch.int64),
            {"put_into_plate": PLATE},
        )
        self.assertNotIn("instruction_successes_uncaught_start/put_into_plate", counts)
        self.assertIn("instruction_successes_normal_start/put_into_plate", counts)


class TheGateRoutingTests(unittest.TestCase):
    """The trainer's selection, exercised as the trainer writes it."""

    @staticmethod
    def _gate(metrics, name, *, uncaught_only, grasp_gated=frozenset({"pick_up"})):
        worlds = metrics.get(f"instruction_worlds_normal_start/{name}", 0.0)
        key = (
            f"instruction_grasps_normal_start/{name}"
            if name in grasp_gated
            else f"instruction_successes_normal_start/{name}"
        )
        if uncaught_only and name not in grasp_gated:
            uncaught_worlds = metrics.get(
                f"instruction_worlds_uncaught_start/{name}", 0.0
            )
            uncaught_key = f"instruction_successes_uncaught_start/{name}"
            if uncaught_worlds > 0.0 and uncaught_key in metrics:
                worlds, key = uncaught_worlds, uncaught_key
        if worlds > 0.0 and key in metrics:
            return metrics.get(key, 0.0) / worlds
        return None

    METRICS = {
        "instruction_worlds_normal_start/put_into_plate": 100.0,
        "instruction_successes_normal_start/put_into_plate": 42.0,
        "instruction_worlds_uncaught_start/put_into_plate": 20.0,
        "instruction_successes_uncaught_start/put_into_plate": 1.0,
    }

    def test_off_reproduces_the_old_blended_rate(self) -> None:
        self.assertAlmostEqual(
            self._gate(self.METRICS, "put_into_plate", uncaught_only=False), 0.42
        )

    def test_on_reads_the_composed_rate(self) -> None:
        self.assertAlmostEqual(
            self._gate(self.METRICS, "put_into_plate", uncaught_only=True), 0.05
        )

    def test_it_falls_back_rather_than_freezing_at_caught_fraction_one(self) -> None:
        """0/0 does not fail, it freezes. This is the important one."""

        metrics = dict(self.METRICS)
        metrics["instruction_worlds_uncaught_start/put_into_plate"] = 0.0
        metrics["instruction_successes_uncaught_start/put_into_plate"] = 0.0
        self.assertAlmostEqual(
            self._gate(metrics, "put_into_plate", uncaught_only=True), 0.42
        )

    def test_pick_up_still_gates_on_its_grasp(self) -> None:
        # pick_up is the one instruction whose approach cannot be scored on
        # success, because success there also requires the lift. The uncaught
        # split must not quietly re-route it.
        metrics = {
            "instruction_worlds_normal_start/pick_up": 100.0,
            "instruction_grasps_normal_start/pick_up": 25.0,
            "instruction_successes_normal_start/pick_up": 5.0,
            "instruction_worlds_uncaught_start/pick_up": 100.0,
            "instruction_successes_uncaught_start/pick_up": 5.0,
        }
        self.assertAlmostEqual(
            self._gate(metrics, "pick_up", uncaught_only=True), 0.25
        )

    def test_the_trainer_uses_exactly_this_selection(self) -> None:
        import inspect

        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as module

        source = inspect.getsource(module.main)
        self.assertIn('"approach_gate_uncaught_only"', source)
        self.assertIn("instruction_worlds_uncaught_start/", source)
        self.assertIn("if uncaught_worlds > 0.0 and uncaught_key in synchronized_metrics:", source)
        self.assertIn("name not in _GRASP_GATED_INSTRUCTIONS", source)


if __name__ == "__main__":
    unittest.main()
