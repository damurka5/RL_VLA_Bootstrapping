"""The latent-Gaussian conditional-offset likelihood (spec section 3, tests 7B).

The estimator: sample ``u = mu + b + sigma*eps``, execute ``clip(u)``, score
``log N(u; mu + b, sigma^2)`` with ``b`` the realized, recorded episode offset.
The offset is exogenous with a parameter-free density, so it cancels in every
ratio; clipping is a deterministic map applied after the policy's sample.

The load-bearing tests are the gradient checks against a stochastic control
problem whose returns depend on the actions (with clipping, and a persistent
offset that changes later states): the score estimator must agree with finite
differences of the expected return. Two historical alternatives are measured
on the same problem and must NOT agree -- scoring the clipped action, and
scoring each step against an independent widened marginal.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import torch

if torch is not None:
    import numpy as np

    from rl_vla_bootstrapping.policy.latent_correction_policy import (
        POLICY_ARCHITECTURE_CORRECTION,
        POLICY_ARCHITECTURE_LEGACY,
        latent_gaussian_log_prob,
    )
    from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
        ThreeStageMilestones,
        concatenate_collector_rounds,
        gate_episode_offsets,
        latent_slot_records,
    )
    from rl_vla_bootstrapping.policy.rank_local_grpo import EqualDDPSchedule, pad_tensor_records
    from test_latent_correction_policy import (
        ACTION_DIM,
        CHUNK,
        EXECUTED,
        converted,
        inputs,
        latent_args,
        latent_records,
        legacy_args,
        make_trainer,
        write_legacy_checkpoint,
    )


def _numpy_mean(trainer, states, priors):
    """Independent float64 NumPy forward of the correction actor."""

    actor = trainer._unwrap(trainer.actor).actor

    def mlp(module, x):
        layers = [m for m in module.net if isinstance(m, torch.nn.Linear)]
        for index, layer in enumerate(layers):
            x = x @ layer.weight.detach().double().numpy().T + layer.bias.detach().double().numpy()
            if index < len(layers) - 1:
                x = np.maximum(x, 0.0)
        return x

    s = states.double().numpy()
    p = priors.double().numpy()
    x = np.concatenate([s, p.reshape(p.shape[0], -1)], axis=-1)
    reference = p + actor.residual_scale * np.tanh(mlp(actor.reference_net, x)).reshape(p.shape)
    correction = mlp(actor.correction_net, x).reshape(p.shape)
    return np.tanh(reference + correction)


@unittest.skipIf(torch is None, "torch is not installed")
class SamplingAndDensityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.legacy_path, self.legacy = write_legacy_checkpoint(self.root, with_lora=False)
        self.candidate, _ = converted(self.root, POLICY_ARCHITECTURE_CORRECTION, self.legacy_path, with_lora=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_same_noise_and_offsets_give_identical_controller_actions(self):
        states, priors = inputs(64, 1)
        offsets = torch.zeros(64, ACTION_DIM)
        offsets[:, 4] = torch.linspace(-0.4, 0.4, 64)
        old, _, _ = self.legacy.sample_action_chunks_tensor(
            states=states, priors=priors, action_count=EXECUTED,
            generator=torch.Generator().manual_seed(7), mean_offset=offsets,
            offset_std=torch.full((64, ACTION_DIM), 0.15),
        )
        new = self.candidate.sample_latent_action_chunks_tensor(
            states=states, priors=priors, action_count=EXECUTED,
            generator=torch.Generator().manual_seed(7), mean_offset=offsets,
        )
        self.assertTrue(torch.equal(old, new["executed_action"]))
        # Without an offset, too, and with externally supplied noise.
        noise = torch.randn(64, EXECUTED, ACTION_DIM, generator=torch.Generator().manual_seed(3))
        a = self.candidate.sample_latent_action_chunks_tensor(
            states=states, priors=priors, action_count=EXECUTED, noise=noise)
        b, _, _ = self.legacy.sample_action_chunks_tensor(
            states=states, priors=priors, action_count=EXECUTED,
            generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(a["executed_action"], b))

    def test_latents_beyond_both_bounds_are_stored_and_scored(self):
        states, priors = inputs(256, 2)
        offsets = torch.zeros(256, ACTION_DIM)
        offsets[:, 4] = torch.linspace(-0.6, 0.6, 256)
        sample = self.candidate.sample_latent_action_chunks_tensor(
            states=states, priors=priors, action_count=EXECUTED,
            generator=torch.Generator().manual_seed(4), mean_offset=offsets)
        u, a = sample["policy_sample"], sample["executed_action"]
        self.assertTrue(bool((u > 1.0).any()) and bool((u < -1.0).any()))
        outside = u.abs() > 1.0
        self.assertTrue(bool((u[outside] != a[outside]).all()))
        self.assertTrue(torch.equal(u[~outside], a[~outside]))
        log_std = self.candidate._unwrap(self.candidate.actor).clamped_log_std()[:EXECUTED]
        expected = torch.distributions.Normal(
            sample["mean"] + offsets.unsqueeze(1), log_std.exp()).log_prob(u).sum(-1)
        self.assertTrue(torch.allclose(sample["old_log_prob"], expected, atol=1e-5))
        on_executed = torch.distributions.Normal(
            sample["mean"] + offsets.unsqueeze(1), log_std.exp()).log_prob(a).sum(-1)
        self.assertFalse(torch.allclose(sample["old_log_prob"], on_executed, atol=1e-3))

    def test_log_prob_matches_torch_normal(self):
        generator = torch.Generator().manual_seed(5)
        for offset_scale in (0.0, 0.3):
            mean = torch.randn(32, EXECUTED, ACTION_DIM, generator=generator, dtype=torch.float64)
            log_std = torch.linspace(-1.7, -0.2, EXECUTED * ACTION_DIM, dtype=torch.float64).reshape(1, EXECUTED, ACTION_DIM)
            offset = offset_scale * torch.randn(32, 1, ACTION_DIM, generator=generator, dtype=torch.float64)
            sample = mean + offset + 2.0 * torch.randn(32, EXECUTED, ACTION_DIM, generator=generator, dtype=torch.float64)
            ours = latent_gaussian_log_prob(sample, mean, offset, log_std)
            theirs = torch.distributions.Normal(mean + offset, log_std.exp()).log_prob(sample).sum(-1)
            self.assertLess(float((ours - theirs).abs().max()), 1e-12)

    def test_ratio_is_one_at_unchanged_parameters_and_analytic_after_a_change(self):
        states, priors = inputs(64, 6)
        records = latent_records(self.candidate, states, priors, seed=9, advantage_seed=1)
        self.assertTrue(bool((records["policy_sample"] != records["executed_action"]).any()))
        mean, log_std = self.candidate._mean_and_log_std(records["state"], records["prior"], records["action_index"])
        new = latent_gaussian_log_prob(records["policy_sample"], mean, records["behavior_mean_offset"], log_std)
        self.assertLess(float((new - records["old_log_prob"]).detach().abs().max()), 1e-5)
        # Change the parameters and compare with an independent float64 ratio.
        policy = self.candidate._unwrap(self.candidate.actor)
        with torch.no_grad():
            policy.actor.correction_output_layer().bias.normal_(0.0, 0.2, generator=torch.Generator().manual_seed(2))
            policy.actor.correction_output_layer().weight.normal_(0.0, 0.01, generator=torch.Generator().manual_seed(3))
            mean, log_std = self.candidate._mean_and_log_std(records["state"], records["prior"], records["action_index"])
            new = latent_gaussian_log_prob(records["policy_sample"], mean, records["behavior_mean_offset"], log_std)
        ratio = torch.exp(new - records["old_log_prob"]).double().numpy()
        old_mean = _numpy_mean(self.legacy_like(), states, priors)
        new_mean = _numpy_mean(self.candidate, states, priors)
        sigma = log_std.detach().double().exp().numpy()
        idx = records["action_index"].numpy()
        rows = np.arange(idx.size) % states.shape[0]
        u = records["policy_sample"].double().numpy()
        b = records["behavior_mean_offset"].double().numpy()
        mu_old, mu_new = old_mean[rows, idx], new_mean[rows, idx]
        # Same sigma before and after, so the log-ratio is the difference of
        # the two squared standardized residuals.
        analytic = np.exp(
            ((((u - mu_old - b) ** 2) - ((u - mu_new - b) ** 2)) / (2 * sigma ** 2)).sum(-1)
        )
        self.assertGreater(float(np.abs(analytic - 1.0).max()), 0.05, "the change must move ratios")
        self.assertLess(float(np.abs(np.log(ratio) - np.log(analytic)).max()), 2e-4)

    def legacy_like(self):
        """A correction trainer with the zero-init (pre-change) weights."""

        trainer, _ = converted(self.root, POLICY_ARCHITECTURE_CORRECTION, self.legacy_path, with_lora=False)
        return trainer

    def test_update_reports_ratio_one_through_padding_and_masks(self):
        states, priors = inputs(16, 7)
        records = latent_records(self.candidate, states, priors, seed=1, advantage_seed=2)
        n = int(records["advantage"].shape[0])
        mask = torch.ones(n)
        mask[::5] = 0.0  # filtered rows
        torch.manual_seed(0)
        metrics = self.candidate.update_tensor_records(
            records, loss_mask=mask,
            schedule=EqualDDPSchedule(records_per_minibatch=96, ppo_epochs=1, global_max_records=96),
        )
        # One optimizer step at the starting parameters: every ratio is 1.
        self.assertEqual(metrics["optimizer_steps"], 1.0)
        self.assertLess(abs(metrics["approx_kl_mean"]), 1e-6)
        self.assertEqual(metrics["clip_fraction_mean"], 0.0)
        self.assertEqual(metrics["padded_records"], 96.0)
        self.assertGreater(metrics["latent/update_clip_fraction_gripper"]
                           + metrics["latent/update_clip_fraction_z"], 0.0)
        self.assertIn("latent/gaussian_entropy_mean", metrics)


@unittest.skipIf(torch is None, "torch is not installed")
class ScoreGradientTests(unittest.TestCase):
    """Score-function gradients against the true gradient of expected return."""

    def test_single_step_with_clipping_matches_integrated_gradient(self):
        dtype = torch.float64
        mu, sigma, s, c = 0.85, 0.3, 0.25, 0.9
        R = lambda a: -(a - c) ** 2  # noqa: E731
        # True dJ/dmu by central differences of J integrated on a fine grid.
        v = sigma ** 2 + s ** 2
        z = torch.linspace(-12 * math.sqrt(v), 12 * math.sqrt(v), 400001, dtype=dtype)
        w = torch.exp(-z ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v) * (z[1] - z[0])
        J = lambda m: float((R((m + z).clamp(-1, 1)) * w).sum())  # noqa: E731
        true = (J(mu + 1e-5) - J(mu - 1e-5)) / 2e-5
        # Expectations of the estimators, by 2-D integration over (b, eps).
        b = torch.linspace(-10 * s, 10 * s, 3001, dtype=dtype)
        e = torch.linspace(-10.0, 10.0, 3001, dtype=dtype)
        B, E = torch.meshgrid(b, e, indexing="ij")
        W = (torch.exp(-B ** 2 / (2 * s * s)) / math.sqrt(2 * math.pi * s * s)
             * torch.exp(-E ** 2 / 2) / math.sqrt(2 * math.pi) * (b[1] - b[0]) * (e[1] - e[0]))
        U = mu + B + sigma * E
        A = U.clamp(-1, 1)
        self.assertGreater(float(((U.abs() > 1).to(dtype) * W).sum()), 0.3, "clipping must matter")
        latent = float((R(A) * (U - mu - B) / sigma ** 2 * W).sum())
        clipped_marginal = float((R(A) * (A - mu) / v * W).sum())
        print(f"\n[B5 single-step] true {true:.7f}  latent-conditional {latent:.7f}  "
              f"historical clipped/marginal {clipped_marginal:.7f}")
        self.assertLess(abs(latent - true), 1e-5)
        self.assertGreater(abs(clipped_marginal - true), 1e-3)
        # The code path: autograd through latent_gaussian_log_prob, Monte Carlo.
        n = 400_000
        g = torch.Generator().manual_seed(11)
        offset = torch.randn(n, 1, generator=g, dtype=dtype) * s
        u = mu + offset + sigma * torch.randn(n, 1, generator=g, dtype=dtype)
        reward = R(u.clamp(-1, 1))[:, 0]
        theta = torch.tensor([mu], dtype=dtype, requires_grad=True)
        logp = latent_gaussian_log_prob(u, theta.expand(n, 1), offset, torch.tensor([math.log(sigma)], dtype=dtype))
        per = ((reward - reward.mean()) * ((u - mu - offset) / sigma ** 2)[:, 0]).detach()
        estimate = float(torch.autograd.grad(((reward - reward.mean()).detach() * logp).mean(), theta)[0])
        se = float(per.std() / math.sqrt(n))
        self.assertAlmostEqual(estimate, float(per.mean()), places=10)
        self.assertLess(abs(estimate - true), 5.0 * se)

    def test_multi_step_persistent_offset_matches_finite_differences(self):
        """x_{t+1} = x_t + k*clip(u_t); the offset shifts every later state.

        The policy mean depends on the state, so the realized offset changes
        later means; the conditional score must still be unbiased. Checked
        against central finite differences with common random numbers, at a
        5-standard-error tolerance on the per-sample difference (fixed seed, so
        the test is deterministic; 5 SE is a two-sided p ~ 6e-7 band).
        """

        dtype = torch.float64
        T, sigma, s, target, k, n = 3, 0.3, 0.25, 1.6, 0.6, 400_000
        g = torch.Generator().manual_seed(0)
        offset = torch.randn(n, 1, generator=g, dtype=dtype) * s
        eps = torch.randn(n, T, 1, generator=g, dtype=dtype)
        log_std = torch.tensor([math.log(sigma)], dtype=dtype)
        theta0 = torch.tensor([0.7, 0.4], dtype=dtype)

        def rollout(theta):
            x = torch.zeros(n, 1, dtype=dtype)
            latents = []
            for t in range(T):
                u = theta[0] + theta[1] * x + offset + sigma * eps[:, t]
                latents.append(u)
                x = x + k * u.clamp(-1, 1)
            return -(x[:, 0] - target) ** 2, torch.stack(latents, 1)

        reward, latents = rollout(theta0)
        self.assertGreater(float((latents.abs() > 1).to(dtype).mean()), 0.3, "clipping must matter")
        advantage = (reward - reward.mean()).detach()

        def per_sample_scores(score_fn):
            x = torch.zeros(n, 1, dtype=dtype)
            d = torch.zeros(n, 2, dtype=dtype)
            for t in range(T):
                mu = theta0[0] + theta0[1] * x
                score = score_fn(latents[:, t], mu)[:, 0]
                d[:, 0] += score
                d[:, 1] += score * x[:, 0]
                x = x + k * latents[:, t].clamp(-1, 1)
            return advantage[:, None] * d

        h = 1e-4
        fd = torch.stack([
            (rollout(theta0 + h * torch.eye(2, dtype=dtype)[i])[0]
             - rollout(theta0 - h * torch.eye(2, dtype=dtype)[i])[0]) / (2 * h)
            for i in range(2)
        ], dim=1)

        # The code path: autograd through latent_gaussian_log_prob.
        theta = theta0.clone().requires_grad_(True)
        x = torch.zeros(n, 1, dtype=dtype)
        logp = torch.zeros(n, dtype=dtype)
        for t in range(T):
            logp = logp + latent_gaussian_log_prob(latents[:, t], theta[0] + theta[1] * x, offset, log_std)
            x = x + k * latents[:, t].clamp(-1, 1)
        estimate = torch.autograd.grad((advantage * logp).mean(), theta)[0]

        latent = per_sample_scores(lambda u, mu: (u - mu - offset) / sigma ** 2)
        self.assertLess(float((latent.mean(0) - estimate).abs().max()), 1e-9)
        v = sigma ** 2 + s ** 2
        clipped_marginal = per_sample_scores(lambda u, mu: (u.clamp(-1, 1) - mu) / v)
        independent_marginal = per_sample_scores(lambda u, mu: (u - mu) / v)

        def z_scores(per):
            diff = per - fd
            return (diff.mean(0) / (diff.std(0) / math.sqrt(n))).tolist()

        print(f"\n[B5 multi-step] FD {fd.mean(0).tolist()}  latent {latent.mean(0).tolist()} "
              f"z={z_scores(latent)}; clipped-marginal z={z_scores(clipped_marginal)}; "
              f"independent-marginal z={z_scores(independent_marginal)}")
        self.assertTrue(all(abs(z) < 5.0 for z in z_scores(latent)))
        self.assertTrue(any(abs(z) > 20.0 for z in z_scores(clipped_marginal)))
        self.assertTrue(any(abs(z) > 20.0 for z in z_scores(independent_marginal)))


@unittest.skipIf(torch is None, "torch is not installed")
class RecordPlumbingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.legacy_path, self.legacy = write_legacy_checkpoint(self.root, with_lora=False)
        self.candidate, _ = converted(self.root, POLICY_ARCHITECTURE_CORRECTION, self.legacy_path, with_lora=False)

    def tearDown(self):
        self._tmp.cleanup()

    def _round(self, *, holding_by_decision, episode_offsets, seed):
        """Two decisions of four slots, gate read once per decision."""

        worlds = int(episode_offsets.shape[0])
        states, priors = inputs(worlds, seed)
        lists: dict[str, list] = {}
        for decision, holding in enumerate(holding_by_decision):
            step_offsets = gate_episode_offsets(episode_offsets, holding)
            sample = self.candidate.sample_latent_action_chunks_tensor(
                states=states, priors=priors, action_count=EXECUTED,
                generator=torch.Generator().manual_seed(seed * 10 + decision), mean_offset=step_offsets)
            for slot in range(EXECUTED):
                row = latent_slot_records(torch, sample, slot, 1)
                row.update(state=states, prior=priors,
                           action_index=torch.full((worlds,), slot, dtype=torch.long),
                           expected_offset=step_offsets.clone())
                for key, value in row.items():
                    lists.setdefault(key, []).append(value)
            # Mutating the episode draw afterwards must not reach stored rows.
            episode_offsets.mul_(1.0)
        return {key: torch.cat(value, dim=0) for key, value in lists.items()}

    def test_gate_transition_records_the_offset_of_the_sampling_decision(self):
        worlds = 8
        offsets = torch.zeros(worlds, ACTION_DIM)
        offsets[:, 4] = torch.linspace(-0.3, 0.3, worlds)
        holding = [torch.zeros(worlds, dtype=torch.bool), torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)]
        rows = self._round(holding_by_decision=holding, episode_offsets=offsets, seed=1)
        self.assertTrue(torch.equal(rows["behavior_mean_offset"], rows["expected_offset"]))
        first = rows["behavior_mean_offset"][: worlds * EXECUTED]
        second = rows["behavior_mean_offset"][worlds * EXECUTED:].view(EXECUTED, worlds, ACTION_DIM)
        self.assertEqual(float(first.abs().sum()), 0.0)
        for slot in range(EXECUTED):
            self.assertTrue(torch.equal(second[slot][holding[1]], offsets[holding[1]]))
            self.assertEqual(float(second[slot][~holding[1]].abs().sum()), 0.0)

    def test_a_new_episode_draws_fresh_offsets_and_starts_ungated(self):
        trainer = self.candidate
        g = torch.Generator()
        g.manual_seed(100)
        first = trainer.sample_episode_offsets(8, generator=g)
        g.manual_seed(200)
        second = trainer.sample_episode_offsets(8, generator=g)
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(float(first[:, :4].abs().sum()), 0.0)  # gripper-only offset
        fresh = ThreeStageMilestones.zeros(torch, 8, torch.device("cpu"))
        self.assertEqual(float(gate_episode_offsets(second, fresh.picked_up).abs().sum()), 0.0)

    def test_fields_survive_concatenation_padding_and_slot_indexing(self):
        worlds = 8
        offsets = torch.zeros(worlds, ACTION_DIM)
        offsets[:, 4] = torch.linspace(-0.3, 0.3, worlds)
        rounds = []
        for seed in (1, 2):
            rows = self._round(
                holding_by_decision=[torch.ones(worlds, dtype=torch.bool)] * 2,
                episode_offsets=offsets.clone(), seed=seed)
            rows.pop("expected_offset")
            n = int(rows["state"].shape[0])
            rows["advantage"] = torch.randn(n, generator=torch.Generator().manual_seed(seed))
            rows["credit_stage"] = torch.full((n,), 2, dtype=torch.long)
            rows["candidate_id"] = torch.arange(n) % worlds
            rounds.append(SimpleNamespace(
                records=rows, loss_mask=torch.ones(n, dtype=torch.bool),
                candidate_rewards=torch.zeros(worlds), candidate_success=torch.zeros(1, worlds, dtype=torch.bool),
                candidate_ever_grasped=None, group_instruction_ids=torch.zeros(1, dtype=torch.long),
                group_shell_ids=torch.zeros(1, dtype=torch.long), group_prelifted=None,
                group_caught_start=None, group_skips_approach=None, metrics={},
            ))
        records = concatenate_collector_rounds(rounds)[0]
        for key in ("policy_sample", "executed_action", "behavior_mean_offset", "old_log_prob"):
            self.assertTrue(torch.equal(records[key], torch.cat([r.records[key] for r in rounds])), key)
        padded, mask = pad_tensor_records(records, target_records=200)
        n = int(records["state"].shape[0])
        for key in ("policy_sample", "behavior_mean_offset", "action_index", "likelihood_version"):
            self.assertTrue(torch.equal(padded[key][:n], records[key]), key)
        self.assertEqual(float(mask[n:].sum()), 0.0)
        mean, log_std = self.candidate._mean_and_log_std(padded["state"][:n], padded["prior"][:n], padded["action_index"][:n])
        new = latent_gaussian_log_prob(padded["policy_sample"][:n], mean, padded["behavior_mean_offset"][:n], log_std)
        self.assertLess(float((new - padded["old_log_prob"][:n]).detach().abs().max()), 1e-5)


@unittest.skipIf(torch is None, "torch is not installed")
class FailClosedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.legacy_path, self.legacy = write_legacy_checkpoint(self.root, with_lora=False)
        self.candidate, _ = converted(self.root, POLICY_ARCHITECTURE_CORRECTION, self.legacy_path, with_lora=False)
        states, priors = inputs(8, 3)
        self.records = latent_records(self.candidate, states, priors, seed=0, advantage_seed=0)

    def tearDown(self):
        self._tmp.cleanup()

    def _update(self, trainer, records):
        n = int(next(iter(records.values())).shape[0])
        return trainer.update_tensor_records(
            records, loss_mask=torch.ones(n),
            schedule=EqualDDPSchedule(records_per_minibatch=32, ppo_epochs=1, global_max_records=n))

    def test_missing_fields_fail(self):
        for field in ("policy_sample", "behavior_mean_offset", "likelihood_version", "executed_action"):
            broken = dict(self.records)
            broken.pop(field)
            with self.assertRaises(KeyError, msg=field):
                self._update(self.candidate, broken)

    def test_mismatched_likelihood_version_fails(self):
        broken = dict(self.records)
        broken["likelihood_version"] = torch.zeros_like(broken["likelihood_version"])
        with self.assertRaises(ValueError):
            self._update(self.candidate, broken)

    def test_legacy_records_are_not_on_policy_for_the_latent_likelihood(self):
        legacy_style = dict(self.records)
        legacy_style["action"] = legacy_style["executed_action"]
        with self.assertRaises(KeyError):
            self._update(self.candidate, legacy_style)
        with_std = dict(self.records)
        with_std["offset_std"] = torch.zeros_like(with_std["behavior_mean_offset"])
        with self.assertRaises(KeyError):
            self._update(self.candidate, with_std)
        # And latent records cannot be fed to a legacy trainer.
        with self.assertRaises(KeyError):
            self._update(self.legacy, dict(self.records))

    def test_unsupported_paths_fail_before_training(self):
        states, priors = inputs(2, 1)
        with self.assertRaises(RuntimeError):
            self.candidate.sample_action_chunks_batch(states=states.numpy(), priors=priors.numpy(), action_count=4)
        with self.assertRaises(RuntimeError):
            self.candidate.sample_action_group(state=states[0].numpy(), prior=priors[0].numpy(), action_index=0, group_size=2)
        with self.assertRaises(RuntimeError):
            self.candidate.update([{"state": states[0].numpy()}])
        with self.assertRaises(RuntimeError):
            self.candidate.update_vla_lora({})
        with self.assertRaises(ValueError):
            self.candidate.sample_action_chunks_tensor(
                states=states, priors=priors, action_count=4,
                mean_offset=torch.zeros(2, ACTION_DIM), offset_std=torch.zeros(2, ACTION_DIM))
        control = make_trainer(latent_args(POLICY_ARCHITECTURE_LEGACY), self.root / "control")
        self.assertTrue(control.latent_likelihood)
        legacy = make_trainer(legacy_args(), self.root / "legacy2")
        self.assertFalse(legacy.latent_likelihood)


if __name__ == "__main__":
    unittest.main()
