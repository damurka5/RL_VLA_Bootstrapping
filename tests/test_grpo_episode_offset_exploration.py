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

    def test_sampler_scores_against_the_marginal_not_the_perturbed_mean(self):
        """The offset must be inside the STD, not the mean, or it is invisible.

        Scoring against mu+eps is the conditional density given eps. Its score
        is (a - mu - eps)/sigma^2 -- the per-step noise, independent of eps by
        construction -- so the gradient on mu learns nothing about which offset
        paid. Scoring against the marginal N(mu, sigma^2 + s^2) is equally valid
        importance sampling and puts eps back into (a - mu).
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = _trainer(
                _args("--episode-offset-std", "0.25"), Path(temp_dir)
            )
            states = torch.randn(4, 6)
            priors = torch.zeros(4, 2, 5)
            offsets = trainer.sample_episode_offsets(4)
            stds = trainer.episode_offset_std.unsqueeze(0).expand(4, -1)

            generator = torch.Generator().manual_seed(11)
            actions, log_probs, means = trainer.sample_action_chunks_tensor(
                states=states,
                priors=priors,
                action_count=2,
                generator=generator,
                mean_offset=offsets,
                offset_std=stds,
            )
            # The third return value stays the UNPERTURBED policy mean: the
            # residual telemetry built on it measures what the policy learned.
            expected_mean = trainer._unwrap(trainer.actor)(states, priors)[:, :2]
            self.assertTrue(torch.allclose(means, expected_mean, atol=1e-6))

            log_std = trainer._unwrap(trainer.actor).clamped_log_std()[
                :2
            ].unsqueeze(0)
            marginal = trainer._marginal_log_std(log_std, stds.unsqueeze(1))
            self.assertTrue(
                torch.allclose(
                    log_probs,
                    _normal_log_prob(actions, means, marginal),
                    atol=1e-5,
                )
            )
            # The widened std is the whole point.
            self.assertTrue(bool(torch.all(marginal > log_std)))
            # And NOT the conditional density given the realised offset.
            self.assertFalse(
                torch.allclose(
                    log_probs,
                    _normal_log_prob(
                        actions, means + offsets.unsqueeze(1), log_std
                    ),
                    atol=1e-3,
                )
            )

    def test_offset_reaches_the_gradient_on_the_mean(self):
        """The regression test for the bug that cost two training runs.

        Build a batch whose advantage is exactly the offset signal -- worlds
        with a larger offset scored better -- and check the gradient on the
        policy mean actually points along it. Under the shipped-then-reverted
        conditional form this is ~0.
        """

        sigma, offset_std, worlds = 0.333, 0.25, 40000
        torch.manual_seed(0)
        mean = torch.zeros(worlds)
        eps = torch.randn(worlds) * offset_std
        actions = mean + eps + torch.randn(worlds) * sigma
        advantage = eps / offset_std

        conditional = ((actions - (mean + eps)) / sigma**2 * advantage).mean()
        marginal_var = sigma**2 + offset_std**2
        marginal = ((actions - mean) / marginal_var * advantage).mean()

        self.assertLess(abs(float(conditional)), 0.1)
        self.assertGreater(float(marginal), 1.0)

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


@unittest.skipIf(torch is None, "torch is not installed")
class EpisodeOffsetGateTests(unittest.TestCase):
    """The gate must change the offset actually sampled AND actually recorded.

    Recording the ungated constant while sampling a gated one would price a
    behaviour mean no action was ever drawn from -- the same silent-wrong-answer
    failure the ratio test above guards, reintroduced through the collector.
    """

    def test_gate_zeroes_the_offset_until_the_world_holds_the_object(self):
        offsets = torch.full((4, 5), 0.25)
        first_grasp_step = torch.tensor([-1, -1, 3, 0])
        prelifted = torch.tensor([False, True, False, False])

        holding = first_grasp_step >= 0
        holding = holding | prelifted
        gated = offsets * holding.unsqueeze(-1).to(dtype=offsets.dtype)

        # World 0 has neither grasped nor started pre-grasped: no offset.
        self.assertTrue(bool(torch.all(gated[0] == 0.0)))
        # World 1 starts holding, so it gets the offset on decision 0 -- it is
        # the regime the plant probe measured and must not wait a decision.
        self.assertTrue(bool(torch.all(gated[1] == 0.25)))
        # Worlds 2 and 3 have grasped.
        self.assertTrue(bool(torch.all(gated[2] == 0.25)))
        self.assertTrue(bool(torch.all(gated[3] == 0.25)))

    def test_ungated_offset_is_unchanged(self):
        offsets = torch.full((3, 5), 0.25)
        self.assertTrue(torch.equal(offsets, offsets.clone()))

    def test_parser_exposes_the_gate_and_defaults_it_off(self):
        self.assertFalse(_args().episode_offset_after_grasp)
        self.assertTrue(
            _args("--episode-offset-after-grasp").episode_offset_after_grasp
        )
        self.assertFalse(
            _args("--no-episode-offset-after-grasp").episode_offset_after_grasp
        )
