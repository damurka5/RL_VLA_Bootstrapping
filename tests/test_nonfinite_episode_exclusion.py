"""A world reset for divergence ends its episode and leaves learning.

The backend restores the calibrated base state inside ``step``; before this fix
the collector kept stepping, scoring and recording that world, so a trajectory
that crossed a reset could enter the GRPO update.
"""
from __future__ import annotations

import inspect
import unittest

import numpy as np
import torch

from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    NonFiniteEpisodeGuard,
    RankLocalMJWarpGRPOCollector,
    three_stage_group_credit,
)
from rl_vla_bootstrapping.policy.rank_local_grpo import torch_group_advantages


class _DivergingBackend:
    """Mimics the MJWarp backend's cumulative mask and event counter."""

    def __init__(self, worlds: int, stale_events: int = 0) -> None:
        self.seen = torch.zeros((worlds,), dtype=torch.bool)
        self.events = int(stale_events)

    def diverge(self, *worlds: int) -> None:
        for world in worlds:
            self.seen[world] = True
            self.events += 1

    def nonfinite_world_mask(self):
        return self.seen.clone()

    def pop_nonfinite_world_events(self) -> int:
        count, self.events = self.events, 0
        self.seen.zero_()
        return count


class GuardTests(unittest.TestCase):
    def test_live_divergence_ends_the_episode_and_idle_does_not_count(self):
        backend = _DivergingBackend(4, stale_events=3)
        guard = NonFiniteEpisodeGuard(backend, torch, 4, "cpu")
        self.assertEqual(guard.stale_events, 3)
        self.assertFalse(bool(backend.seen.any()))

        # World 1 is running, world 3 already finished its episode.
        step_active = torch.tensor([True, True, True, False])
        backend.diverge(1, 3)
        after = guard.after_step(step_active)
        self.assertEqual(after.tolist(), [True, False, True, False])
        self.assertEqual(guard.live.tolist(), [False, True, False, False])
        self.assertEqual(guard.idle.tolist(), [False, False, False, True])

        # The cumulative mask does not re-flag world 1 on later steps, and a
        # new divergence is picked up on the step it happens.
        step_active = torch.tensor([True, False, True, False])
        backend.diverge(2)
        after = guard.after_step(step_active)
        self.assertEqual(after.tolist(), [True, False, False, False])
        self.assertEqual(guard.live.tolist(), [False, True, True, False])
        self.assertEqual(guard.finish(), 3 + 3)
        self.assertEqual(backend.events, 0)

    def test_backend_without_a_mask_is_a_no_op(self):
        class CountOnly:
            def pop_nonfinite_world_events(self):
                return 0

        guard = NonFiniteEpisodeGuard(CountOnly(), torch, 2, "cpu")
        active = torch.tensor([True, True])
        self.assertFalse(guard.enabled)
        self.assertIs(guard.after_step(active), active)


class MaskedAdvantageTests(unittest.TestCase):
    def test_invalid_candidate_leaves_the_baseline_and_gets_zero(self):
        outcomes = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]])
        valid = torch.tensor([[True, True, True, True, False]])
        masked = torch_group_advantages(outcomes, valid=valid)
        reference = torch_group_advantages(outcomes[:, :4])
        self.assertTrue(torch.allclose(masked[:, :4], reference, atol=1e-6))
        self.assertEqual(float(masked[0, 4]), 0.0)

    def test_all_valid_matches_the_unmasked_path(self):
        outcomes = torch.rand(6, 8)
        valid = torch.ones_like(outcomes, dtype=torch.bool)
        self.assertTrue(
            torch.allclose(
                torch_group_advantages(outcomes, valid=valid, clip_abs=3.0),
                torch_group_advantages(outcomes, clip_abs=3.0),
                atol=1e-5,
            )
        )

    def test_three_stage_credit_ignores_the_diverged_candidate(self):
        # Group 0: the lone "success" is the diverged candidate, so the valid
        # candidates carry no contrast. Group 1: one valid candidate is left.
        returns = torch.zeros(3, 2, 4)
        returns[:, 0, 3] = 1.0
        returns[:, 1, 0] = 1.0
        valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
        advantage, usable = three_stage_group_credit(
            returns,
            normalize=True,
            clip_abs=0.0,
            dynamic_sampling=True,
            dynamic_min_pass_rate=0.0,
            dynamic_max_pass_rate=1.0,
            min_group_reward_std=0.0,
            valid=valid,
        )
        self.assertFalse(bool(usable.any()))
        self.assertTrue(bool((advantage[:, :, 3] == 0).all()))
        unmasked, unmasked_usable = three_stage_group_credit(
            returns,
            normalize=True,
            clip_abs=0.0,
            dynamic_sampling=True,
            dynamic_min_pass_rate=0.0,
            dynamic_max_pass_rate=1.0,
            min_group_reward_std=0.0,
        )
        self.assertTrue(bool(unmasked_usable.all()))


class WiringTests(unittest.TestCase):
    def test_training_and_validation_rollouts_use_the_guard(self):
        for method in (
            RankLocalMJWarpGRPOCollector.collect_round,
            RankLocalMJWarpGRPOCollector.validate_round,
        ):
            source = inspect.getsource(method)
            self.assertIn("NonFiniteEpisodeGuard(", source)
            self.assertIn("step_active = nonfinite.after_step(step_active)", source)
            self.assertNotIn("pop_nonfinite_world_events()", source)
        source = inspect.getsource(RankLocalMJWarpGRPOCollector.collect_round)
        self.assertIn("~record_diverged", source)
        self.assertEqual(source.count("valid=valid_by_group"), 3)

    def test_evaluator_ends_diverged_episodes(self):
        from tools.audit import evaluate_cdpr_full_put_into as module

        source = inspect.getsource(module.run_unassisted)
        self.assertIn("step_active = nonfinite.after_step(step_active)", source)


class EvaluatorSummaryTests(unittest.TestCase):
    def _row(self, n=4):
        row = {name: np.zeros(n, dtype=bool) for name in (
            "native", "strict", "approached", "grasped", "lifted", "released",
            "carry_slip", "wrong_place", "final_geometry_ok")}
        row.update(
            completion_steps=np.ones(n), min_grasp_distance=np.ones(n),
            peak_lift=np.zeros(n), scene_uid=np.arange(n),
            destination=np.array(["plate", "bowl"] * (n // 2)),
            target_catalog=np.array(["apple"] * n),
        )
        return row

    def test_non_finite_is_reported_beside_the_rates(self):
        from tools.audit.evaluate_cdpr_full_put_into import summarize

        row = self._row()
        row["non_finite"] = np.array([True, False, False, False])
        row["non_finite_idle"] = np.array([False, False, True, True])
        row["grasped"][0] = True
        report = summarize([row])["non_finite"]
        self.assertEqual(report["episodes"], 1)
        self.assertEqual(report["episodes_plate"], 1)
        self.assertEqual(report["episodes_bowl"], 0)
        self.assertEqual(report["after_grasp"], 1)
        self.assertEqual(report["idle_worlds"], 2)

    def test_old_rollouts_without_the_field_still_summarize(self):
        from tools.audit.evaluate_cdpr_full_put_into import summarize

        self.assertIsNone(summarize([self._row()])["non_finite"])


if __name__ == "__main__":
    unittest.main()
