"""Per-episode exploration offsets must price the policy that actually acted.

The offset shifts the BEHAVIOUR mean the rollout sampled from, so every place
that recomputes a log-prob has to add it back. If one of them forgets, the
importance ratio is taken against a distribution no action was ever drawn from,
and the surrogate silently optimizes the wrong thing -- there is no crash and no
NaN to notice, only a slow wrong answer. The load-bearing assertion here is that
an update replayed at unchanged parameters gives ratio == 1.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import DistributedContext
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
    SmolVLAGRPOTrainer,
    _normal_log_prob,
    parse_args,
    torch,
)


def _args(*extra: str):
    return parse_args(
        [
            "--device", "cpu",
            "--no-distributed",
            "--hidden-dim", "16",
            "--chunk-size", "2",
            "--action-dim", "5",
            *extra,
        ]
    )


def _trainer(args, root: Path) -> SmolVLAGRPOTrainer:
    return SmolVLAGRPOTrainer(
        args=args,
        state_dim=6,
        action_dim=5,
        chunk_size=2,
        run_dir=root,
        device=torch.device("cpu"),
        distributed=DistributedContext(device="cpu"),
    )


@unittest.skipIf(torch is None, "torch is not installed")
class EpisodeOffsetExplorationTests(unittest.TestCase):
    def test_parser_broadcasts_and_validates(self):
        self.assertEqual(_args().episode_offset_std, [0.0])
        self.assertEqual(
            _args("--episode-offset-std", "0", "0", "0.25", "0", "0")
            .episode_offset_std,
            [0.0, 0.0, 0.25, 0.0, 0.0],
        )
        with self.assertRaises(SystemExit):
            _args("--episode-offset-std", "-0.1")
        with self.assertRaises(SystemExit):
            # Two values against action_dim 5 is a silent per-dimension bug
            # waiting to happen, so it must be rejected outright.
            _args("--episode-offset-std", "0.1", "0.2")

    def test_disabled_by_default_returns_no_offset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = _trainer(_args(), Path(temp_dir))
            self.assertFalse(trainer.episode_offset_enabled)
            self.assertIsNone(trainer.sample_episode_offsets(8))

    def test_offset_is_per_world_and_respects_per_dimension_std(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = _trainer(
                _args("--episode-offset-std", "0", "0", "0.25", "0", "0"),
                Path(temp_dir),
            )
            offsets = trainer.sample_episode_offsets(512)
            self.assertEqual(tuple(offsets.shape), (512, 5))
            # Zero-std dimensions must stay exactly zero: a per-dimension vector
            # is how a run asks for z-only exploration.
            for dim in (0, 1, 3, 4):
                self.assertTrue(bool(torch.all(offsets[:, dim] == 0.0)))
            self.assertGreater(float(offsets[:, 2].std()), 0.15)
            self.assertLess(float(offsets[:, 2].std()), 0.35)
            # Different worlds get different offsets -- that is what makes a
            # GRPO group a finite-difference probe rather than eight copies.
            self.assertGreater(
                float((offsets[0, 2] - offsets[1, 2]).abs()), 0.0
            )

    def test_sampler_shifts_the_behaviour_mean_not_the_reported_mean(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = _trainer(
                _args("--episode-offset-std", "0.25"), Path(temp_dir)
            )
            states = torch.randn(4, 6)
            priors = torch.zeros(4, 2, 5)
            offsets = trainer.sample_episode_offsets(4)

            generator = torch.Generator().manual_seed(11)
            actions, log_probs, means = trainer.sample_action_chunks_tensor(
                states=states,
                priors=priors,
                action_count=2,
                generator=generator,
                mean_offset=offsets,
            )
            # The third return value is the UNPERTURBED policy mean: the
            # residual telemetry built on it measures what the policy learned.
            expected_mean = trainer._unwrap(trainer.actor)(states, priors)[:, :2]
            self.assertTrue(torch.allclose(means, expected_mean, atol=1e-6))

            log_std = trainer._unwrap(trainer.actor).clamped_log_std()[
                :2
            ].unsqueeze(0)
            self.assertTrue(
                torch.allclose(
                    log_probs,
                    _normal_log_prob(
                        actions, means + offsets.unsqueeze(1), log_std
                    ),
                    atol=1e-5,
                )
            )
            # And emphatically NOT against the unperturbed mean.
            self.assertFalse(
                torch.allclose(
                    log_probs,
                    _normal_log_prob(actions, means, log_std),
                    atol=1e-3,
                )
            )

    def test_update_replays_the_behaviour_distribution(self):
        """ratio == 1 at unchanged parameters, only if the offset is restored."""

        from rl_vla_bootstrapping.policy.rank_local_grpo import EqualDDPSchedule

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = _trainer(
                _args("--episode-offset-std", "0.25"), Path(temp_dir)
            )
            worlds = 8
            states = torch.randn(worlds, 6)
            priors = torch.zeros(worlds, 2, 5)
            offsets = trainer.sample_episode_offsets(worlds)
            generator = torch.Generator().manual_seed(5)
            actions, log_probs, _ = trainer.sample_action_chunks_tensor(
                states=states,
                priors=priors,
                action_count=2,
                generator=generator,
                mean_offset=offsets,
            )

            records = {
                "state": states,
                "prior": priors,
                "action": actions[:, 0],
                "action_index": torch.zeros(worlds, dtype=torch.long),
                "old_log_prob": log_probs[:, 0],
                "advantage": torch.zeros(worlds),
                "mean_offset": offsets,
            }
            policy_mean, log_std = trainer._mean_and_log_std(
                records["state"],
                records["prior"],
                records["action_index"],
            )
            replayed = _normal_log_prob(
                records["action"],
                trainer._apply_mean_offset(policy_mean, records["mean_offset"]),
                log_std,
            )
            ratio = torch.exp(replayed - records["old_log_prob"])
            self.assertTrue(
                torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4),
                f"ratio should be 1 at unchanged parameters, got {ratio}",
            )

            # Dropping the offset is the exact bug this guards: the ratio then
            # departs from 1 even though nothing about the policy changed.
            naive = torch.exp(
                _normal_log_prob(records["action"], policy_mean, log_std)
                - records["old_log_prob"]
            )
            self.assertFalse(
                torch.allclose(naive, torch.ones_like(naive), atol=1e-2)
            )

            # The real update path must run clean with the extra key present.
            metrics = trainer.update_tensor_records(
                records,
                loss_mask=torch.ones(worlds),
                schedule=EqualDDPSchedule(
                    records_per_minibatch=worlds,
                    ppo_epochs=1,
                    global_max_records=worlds,
                ),
            )
            self.assertIn("clip_fraction_mean", metrics)

    def test_records_without_the_key_take_the_original_path(self):
        """An older rollout buffer must replay untouched."""

        from rl_vla_bootstrapping.policy.rank_local_grpo import EqualDDPSchedule

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = _trainer(_args(), Path(temp_dir))
            worlds = 4
            states = torch.randn(worlds, 6)
            priors = torch.zeros(worlds, 2, 5)
            generator = torch.Generator().manual_seed(3)
            actions, log_probs, _ = trainer.sample_action_chunks_tensor(
                states=states,
                priors=priors,
                action_count=2,
                generator=generator,
            )
            metrics = trainer.update_tensor_records(
                {
                    "state": states,
                    "prior": priors,
                    "action": actions[:, 0],
                    "action_index": torch.zeros(worlds, dtype=torch.long),
                    "old_log_prob": log_probs[:, 0],
                    "advantage": torch.zeros(worlds),
                },
                loss_mask=torch.ones(worlds),
                schedule=EqualDDPSchedule(
                    records_per_minibatch=worlds,
                    ppo_epochs=1,
                    global_max_records=worlds,
                ),
            )
            self.assertIn("clip_fraction_mean", metrics)


if __name__ == "__main__":
    unittest.main()
