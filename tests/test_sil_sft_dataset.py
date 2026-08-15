"""The GPU-free half of ``tools/audit/sil_sft.py``.

Two of these guard failures that do not raise and do not look like bugs. A
row-level split reports a validation loss that is really a memorization
score, and an unreachable target makes the loss sit at a floor that reads as
underfitting. Both would be debugged as optimizer problems.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.audit.sil_sft import (
    _episode_split,
    _filter_instructions,
    _reachability,
)

PLATE, PICK = 4, 8


class EpisodeSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        # Four episodes, three decisions each.
        self.uids = np.array(
            [f"cap_0.01/r0w{world}" for world in range(4) for _ in range(3)],
            dtype="U32",
        )

    def test_no_episode_appears_on_both_sides(self) -> None:
        train, val = _episode_split(self.uids, val_fraction=0.25, seed=0)
        self.assertTrue(np.array_equal(train, ~val))
        overlap = set(self.uids[train].tolist()) & set(self.uids[val].tolist())
        self.assertEqual(overlap, set())

    def test_every_row_of_a_held_out_episode_is_held_out(self) -> None:
        _, val = _episode_split(self.uids, val_fraction=0.25, seed=0)
        for uid in set(self.uids[val].tolist()):
            self.assertEqual(int((self.uids == uid).sum()), 3)
            self.assertEqual(int(val[self.uids == uid].sum()), 3)

    def test_at_least_one_episode_is_always_held_out(self) -> None:
        # A tiny val_fraction rounds to zero episodes, and a validation set of
        # nothing reports nan rather than failing.
        _, val = _episode_split(self.uids, val_fraction=0.001, seed=0)
        self.assertGreaterEqual(int(val.sum()), 1)

    def test_the_split_is_seeded(self) -> None:
        first = _episode_split(self.uids, val_fraction=0.5, seed=7)[1]
        second = _episode_split(self.uids, val_fraction=0.5, seed=7)[1]
        self.assertTrue(np.array_equal(first, second))


class ReachabilityTests(unittest.TestCase):
    """``action = tanh(prior + scale * u)``, u in [-1, 1]."""

    def test_a_target_inside_the_interval_is_reachable(self) -> None:
        prior = np.zeros((1, 8, 5), dtype=np.float32)
        action = np.full((1, 4, 5), float(np.tanh(0.5)), dtype=np.float32)
        mask = np.ones((1, 4), dtype=bool)
        report = _reachability(prior, action, mask, residual_scale=1.0)
        self.assertEqual(report["reachable_fraction"], 1.0)
        self.assertEqual(report["max_shortfall"], 0.0)

    def test_a_target_beyond_tanh_of_the_scale_is_not(self) -> None:
        prior = np.zeros((1, 8, 5), dtype=np.float32)
        # tanh(1.0) ~ 0.7616 is the ceiling at scale 1.0 from a zero prior.
        action = np.full((1, 4, 5), 0.99, dtype=np.float32)
        mask = np.ones((1, 4), dtype=bool)
        report = _reachability(prior, action, mask, residual_scale=1.0)
        self.assertEqual(report["reachable_fraction"], 0.0)
        self.assertAlmostEqual(
            report["max_shortfall"], 0.99 - float(np.tanh(1.0)), places=5
        )

    def test_the_boundary_is_reachable(self) -> None:
        prior = np.zeros((1, 8, 5), dtype=np.float32)
        action = np.full((1, 4, 5), float(np.tanh(1.0)), dtype=np.float32)
        mask = np.ones((1, 4), dtype=bool)
        report = _reachability(prior, action, mask, residual_scale=1.0)
        self.assertEqual(report["reachable_fraction"], 1.0)

    def test_a_large_prior_shifts_the_reachable_interval(self) -> None:
        # A saturated target IS reachable when the prior already points there,
        # which is why this is computed from the interval and not from atanh.
        prior = np.full((1, 8, 5), 3.0, dtype=np.float32)
        action = np.full((1, 4, 5), 0.99, dtype=np.float32)
        mask = np.ones((1, 4), dtype=bool)
        report = _reachability(prior, action, mask, residual_scale=1.0)
        self.assertEqual(report["reachable_fraction"], 1.0)

    def test_only_the_supervised_slots_are_scored(self) -> None:
        # The prior has 8 slots and only 4 were executed; scoring the unused
        # tail would dilute the fraction with values that have no target.
        prior = np.zeros((1, 8, 5), dtype=np.float32)
        action = np.full((1, 4, 5), 0.99, dtype=np.float32)
        mask = np.ones((1, 4), dtype=bool)
        report = _reachability(prior, action, mask, residual_scale=1.0)
        self.assertEqual(report["supervised_values"], 4 * 5)


class InstructionFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = {
            "state": np.zeros((4, 518), dtype=np.float32),
            "instruction_id": np.array([PLATE, PICK, PLATE, PICK]),
        }

    def test_an_empty_whitelist_keeps_everything(self) -> None:
        mask = _filter_instructions(self.dataset, [])
        self.assertEqual(mask.tolist(), [True, True, True, True])

    def test_a_named_instruction_is_selected(self) -> None:
        mask = _filter_instructions(self.dataset, ["put_into_plate"])
        self.assertEqual(mask.tolist(), [True, False, True, False])

    def test_an_unknown_instruction_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            _filter_instructions(self.dataset, ["put_into_saucepan"])


if __name__ == "__main__":
    unittest.main()
