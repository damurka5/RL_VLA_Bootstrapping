"""Two changes that decide what the per_token_random run measures.

**The vision-column reset.** ``residual_vision_pooling`` changes the CONTENT of
the residual's 512 vision inputs without changing their width, so the checkpoint
loads clean and the first layer keeps a mapping learned for the old pooling.
Applied to the new features that mapping is not stale, it is wrong, and the
previous attempt at this swap regressed everything it touched. The reset has to
clear exactly those columns and nothing else -- clearing one column too few
leaves the old mapping partly in place, one too many silently destroys
proprioception, and the failure is invisible either way because the shapes still
match and training still runs.

**The promote dwell.** Every promotion of the 10M run fired the moment the EMA
first touched the threshold, and the rate at the new rung came in well below it
(0.312 -> 0.243, 0.300 -> 0.219). The per-update rate carries about twice the
spread binomial sampling explains, so a threshold the average sits just under is
reachable by two lucky updates. A dwell has to turn that into "stayed above".
"""

from __future__ import annotations

import unittest

from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
    ApproachDistanceCurriculum,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# --------------------------------------------------------------------------
# Vision-column reset
# --------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch is unavailable")
class VisionColumnResetTest(unittest.TestCase):
    STATE_DIM = 70  # 6 proprioception + 64 "vision"
    VISION_DIM = 64
    CHUNK = 2
    ACTION = 5
    HIDDEN = 16

    def _trainer(self):
        """A trainer with the real actor and optimizer, on CPU."""

        import tempfile
        from pathlib import Path

        from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import (
            DistributedContext,
        )
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
            SmolVLAGRPOTrainer,
            parse_args,
        )

        args = parse_args(
            [
                "--device", "cpu",
                "--no-distributed",
                "--hidden-dim", str(self.HIDDEN),
                "--chunk-size", str(self.CHUNK),
                "--action-dim", str(self.ACTION),
            ]
        )
        self._tmp = tempfile.TemporaryDirectory()
        return SmolVLAGRPOTrainer(
            args=args,
            state_dim=self.STATE_DIM,
            action_dim=self.ACTION,
            chunk_size=self.CHUNK,
            run_dir=Path(self._tmp.name),
            device=torch.device("cpu"),
            distributed=DistributedContext(device="cpu"),
        )

    def _first_linear(self, trainer):
        base = trainer._unwrap(trainer.actor)
        return next(
            m for m in base.modules() if isinstance(m, torch.nn.Linear)
        )

    def test_exactly_the_vision_columns_are_cleared(self) -> None:
        trainer = self._trainer()
        first = self._first_linear(trainer)
        with torch.no_grad():
            first.weight.fill_(0.5)
        info = trainer.reset_residual_vision_columns(self.VISION_DIM)

        start = self.STATE_DIM - self.VISION_DIM
        weight = first.weight
        # Vision block: zero.
        self.assertEqual(float(weight[:, start : self.STATE_DIM].abs().sum()), 0.0)
        # Proprioception ahead of it: untouched.
        self.assertTrue(torch.allclose(weight[:, :start], torch.full_like(weight[:, :start], 0.5)))
        # The prior block after it: untouched. This is the one a fencepost error
        # eats, and losing it would silently delete the residual's access to the
        # action it is a residual ON.
        self.assertTrue(
            torch.allclose(
                weight[:, self.STATE_DIM :],
                torch.full_like(weight[:, self.STATE_DIM :], 0.5),
            )
        )
        self.assertEqual(info["vision_reset/columns"], float(self.VISION_DIM))
        self.assertEqual(info["vision_reset/first_column"], float(start))
        self.assertAlmostEqual(
            info["vision_reset/mean_abs_weight_cleared"], 0.5, places=6
        )

    def test_the_prior_block_is_wide_enough_to_notice_a_fencepost(self) -> None:
        """Guards the test above: the prior block must not be empty."""

        trainer = self._trainer()
        first = self._first_linear(trainer)
        self.assertEqual(
            first.in_features, self.STATE_DIM + self.CHUNK * self.ACTION
        )
        self.assertGreater(self.CHUNK * self.ACTION, 0)

    def test_adam_moments_for_those_columns_go_too(self) -> None:
        """A resume restores moments; stale ones take a large first step."""

        trainer = self._trainer()
        first = self._first_linear(trainer)
        # Manufacture optimizer state the way a resume would.
        state = trainer.optimizer.state[first.weight]
        state["exp_avg"] = torch.full_like(first.weight, 0.3)
        state["exp_avg_sq"] = torch.full_like(first.weight, 0.2)

        trainer.reset_residual_vision_columns(self.VISION_DIM)
        start = self.STATE_DIM - self.VISION_DIM
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = trainer.optimizer.state[first.weight][key]
            self.assertEqual(
                float(tensor[:, start : self.STATE_DIM].abs().sum()), 0.0
            )
            self.assertGreater(float(tensor[:, :start].abs().sum()), 0.0)
            self.assertGreater(
                float(tensor[:, self.STATE_DIM :].abs().sum()), 0.0
            )

    def test_the_reset_is_a_no_op_without_vision(self) -> None:
        trainer = self._trainer()
        first = self._first_linear(trainer)
        with torch.no_grad():
            first.weight.fill_(0.5)
        self.assertEqual(trainer.reset_residual_vision_columns(0), {})
        self.assertEqual(float(first.weight.abs().min()), 0.5)

    def test_a_layout_that_does_not_match_is_refused(self) -> None:
        """Silence here would zero arbitrary columns of a working policy."""

        trainer = self._trainer()
        trainer.state_dim = self.STATE_DIM + 7  # pretend the layout moved
        with self.assertRaises(RuntimeError):
            trainer.reset_residual_vision_columns(self.VISION_DIM)

    def test_vision_wider_than_the_state_is_refused(self) -> None:
        trainer = self._trainer()
        with self.assertRaises(ValueError):
            trainer.reset_residual_vision_columns(self.STATE_DIM)

    def test_the_residual_still_runs_and_ignores_vision_afterwards(self) -> None:
        """Zeroed columns mean the vision input cannot move the output."""

        trainer = self._trainer()
        trainer.reset_residual_vision_columns(self.VISION_DIM)
        base = trainer._unwrap(trainer.actor)
        prior = torch.zeros((4, self.CHUNK, self.ACTION))
        state_a = torch.randn((4, self.STATE_DIM))
        state_b = state_a.clone()
        # Change ONLY the vision block.
        state_b[:, self.STATE_DIM - self.VISION_DIM :] = torch.randn(
            (4, self.VISION_DIM)
        )
        with torch.no_grad():
            out_a = base(state_a, prior)
            out_b = base(state_b, prior)
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-6))

    def test_the_vision_path_can_still_learn(self) -> None:
        """Zeroed, not frozen: gradient must reach those columns again."""

        trainer = self._trainer()
        trainer.reset_residual_vision_columns(self.VISION_DIM)
        base = trainer._unwrap(trainer.actor)
        first = self._first_linear(trainer)
        state = torch.randn((8, self.STATE_DIM))
        prior = torch.zeros((8, self.CHUNK, self.ACTION))
        base(state, prior).sum().backward()
        start = self.STATE_DIM - self.VISION_DIM
        self.assertGreater(
            float(first.weight.grad[:, start : self.STATE_DIM].abs().sum()), 0.0
        )


# --------------------------------------------------------------------------
# Promote dwell
# --------------------------------------------------------------------------


def _curriculum(**overrides):
    metadata = {
        "random_workspace_start_distance_curriculum_enabled": True,
        "random_workspace_start_distance_initial": 0.05,
        "random_workspace_start_distance_final": 0.20,
        "random_workspace_start_distance_increment": 0.02,
        "random_workspace_start_distance_promote_pass_rate": 0.30,
        "random_workspace_start_distance_demote_pass_rate": 0.12,
        "random_workspace_start_distance_pass_rate_ema_decay": 0.0,
        "random_workspace_start_distance_cooldown_updates": 1,
    }
    metadata.update(overrides)
    return ApproachDistanceCurriculum(metadata)


class PromoteDwellTest(unittest.TestCase):
    """ema_decay 0 makes the EMA the latest rate, so dwell is what is tested."""

    def test_a_single_spike_no_longer_promotes(self) -> None:
        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=5
        )
        start = item.cap
        for rate in (0.25, 0.25, 0.40, 0.25, 0.25):
            item.observe(rate)
        self.assertEqual(item.cap, start)

    def test_a_sustained_level_still_promotes(self) -> None:
        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=5
        )
        start = item.cap
        for _ in range(5):
            item.observe(0.40)
        self.assertGreater(item.cap, start)

    def test_the_count_restarts_rather_than_accumulating(self) -> None:
        """Four good, one bad, four good must NOT promote at a dwell of 5.

        A tally of qualifying updates would; the claim being made is that the
        level was held, not that it was reached often enough.
        """

        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=5
        )
        start = item.cap
        for rate in (0.4, 0.4, 0.4, 0.4, 0.1, 0.4, 0.4, 0.4, 0.4):
            item.observe(rate)
        self.assertEqual(item.cap, start)

    def test_dwell_one_reproduces_the_old_single_crossing(self) -> None:
        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=1
        )
        start = item.cap
        item.observe(0.40)
        self.assertGreater(item.cap, start)

    def test_demotion_is_not_delayed_by_the_dwell(self) -> None:
        """The dwell guards promotion. Falling through the floor is urgent."""

        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=5,
            random_workspace_start_distance_initial=0.03,
        )
        item.cap = 0.09
        item.observe(0.01)
        self.assertLess(item.cap, 0.09)

    def test_a_promotion_clears_the_count(self) -> None:
        """Otherwise the next rung promotes on carried-over credit."""

        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=2,
            random_workspace_start_distance_cooldown_updates=1,
        )
        item.observe(0.40)
        item.observe(0.40)
        promoted = item.cap
        item.observe(0.40)  # consumed by the cooldown
        item.observe(0.40)  # dwell 1 of 2
        self.assertEqual(item.cap, promoted)
        item.observe(0.40)  # dwell 2 of 2
        self.assertGreater(item.cap, promoted)

    def test_the_dwell_survives_a_checkpoint_round_trip(self) -> None:
        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=5
        )
        item.observe(0.40)
        item.observe(0.40)
        restored = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=5
        )
        restored.load_state_dict(item.state_dict())
        self.assertEqual(restored._dwell, item._dwell)

    def test_an_old_checkpoint_without_the_field_restores_at_zero(self) -> None:
        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=5
        )
        state = item.state_dict()
        del state["dwell"]
        item.load_state_dict(state)
        self.assertEqual(item._dwell, 0)

    def test_a_restart_clears_the_count(self) -> None:
        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=5,
            random_workspace_start_distance_initial=0.03,
        )
        item.cap = 0.09
        item.observe(0.40)
        item.observe(0.40)
        self.assertTrue(item.restart())
        self.assertEqual(item._dwell, 0)


class ObservedPromotionsTest(unittest.TestCase):
    """The 10M run's own EMA trace, replayed through both settings.

    These are the readings around the 0.10 -> 0.13 promotion at update 85. The
    old gate fired on them; the realized rate at 0.13 was 0.219, below the
    threshold it had just cleared. The dwell must not fire on the same trace.
    """

    TRACE = (0.293, 0.293, 0.291, 0.293, 0.289, 0.296, 0.300)

    def test_the_old_gate_fires_on_this_trace(self) -> None:
        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=1
        )
        start = item.cap
        for value in self.TRACE:
            item.observe(value)
        self.assertGreater(item.cap, start)

    def test_the_dwell_does_not(self) -> None:
        item = _curriculum(
            random_workspace_start_distance_promote_dwell_updates=5
        )
        start = item.cap
        for value in self.TRACE:
            item.observe(value)
        self.assertEqual(item.cap, start)


if __name__ == "__main__":
    unittest.main()
