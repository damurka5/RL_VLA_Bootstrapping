"""Zero-initialized correction over a frozen reference: initialization,
learnability, checkpoints and version-aware loading.

Spec: docs/reports/campaign/CDPR_ZERO_INIT_CORRECTION_IMPLEMENTATION.md,
sections 2, 5 and 7A/7C. All checks use a synthetic legacy checkpoint with a
nonzero prior, a non-unit reference scale and saturated residuals; the real
step_56072006 checks are the GPU preflight's job.
"""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import DistributedContext
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
    SmolVLAGRPOTrainer,
    parse_args,
    torch,
)

if torch is not None:
    from torch import nn

    from rl_vla_bootstrapping.policy.latent_correction_policy import (
        ACTION_LIKELIHOOD_LATENT,
        POLICY_ARCHITECTURE_CORRECTION,
        POLICY_ARCHITECTURE_LEGACY,
        checkpoint_policy_architecture,
        convert_legacy_policy_state,
        require_legacy_policy_checkpoint,
    )
    from rl_vla_bootstrapping.policy.rank_local_grpo import EqualDDPSchedule

STATE_DIM, CHUNK, ACTION_DIM, HIDDEN, SCALE = 6, 8, 5, 16, 0.7
EXECUTED = 4


def _argv(*extra: str) -> list[str]:
    return [
        "--device", "cpu", "--no-distributed",
        "--hidden-dim", str(HIDDEN), "--chunk-size", str(CHUNK), "--action-dim", str(ACTION_DIM),
        "--residual-scale", str(SCALE),
        "--min-log-std", "-1.72", "--max-log-std", "-1.10", "--init-log-std", "-1.2",
        "--microbatch-size", "16", "--minibatch-size", "16",
        "--entropy-coef", "0.0002", "--max-grad-norm", "1.0",
        "--clip-range-low", "0.2", "--clip-range-high", "0.28",
        "--episode-offset-std", "0", "0", "0", "0", "0.15",
        *extra,
    ]


def legacy_args(*extra: str):
    return parse_args(_argv(*extra))


def latent_args(architecture: str, *extra: str):
    return parse_args(_argv(
        "--policy-architecture", architecture,
        "--action-likelihood", ACTION_LIKELIHOOD_LATENT,
        "--optimizer-lr-override", "1e-3",
        *extra,
    ))


def make_trainer(args, root: Path) -> "SmolVLAGRPOTrainer":
    return SmolVLAGRPOTrainer(
        args=args, state_dim=STATE_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK,
        run_dir=root, device=torch.device("cpu"), distributed=DistributedContext(device="cpu"),
    )


class FakeLoRARuntime:
    """Stands in for the SmolVLA runtime: a module holding lora_* tensors."""

    def __init__(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.policy = nn.Module()
        self.policy.block = nn.Module()
        self.policy.block.lora_A = nn.Linear(3, 2, bias=False)
        self.policy.block.lora_B = nn.Linear(2, 3, bias=False)
        self.policy.block.base = nn.Linear(3, 3)


def saturate(trainer: "SmolVLAGRPOTrainer", gain: float = 120.0) -> None:
    """Scale the legacy residual's output layer so its inner tanh saturates."""

    base = trainer._unwrap(trainer.actor)
    with torch.no_grad():
        base.actor.net.net[-1].weight.mul_(gain)
        base.actor.net.net[-1].bias.uniform_(-3.0, 3.0)
        base.log_std.copy_(torch.linspace(-1.7, -1.15, CHUNK * ACTION_DIM).reshape(CHUNK, ACTION_DIM))


def inputs(batch: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(batch, STATE_DIM, generator=generator) * 3.0
    # A nonzero, nearly constant prior like the real one (+ a little spread).
    offset = torch.tensor([-0.31, 0.79, 0.99, 0.2, -0.52])
    priors = offset + 0.05 * torch.randn(batch, CHUNK, ACTION_DIM, generator=generator)
    return states, priors


def write_legacy_checkpoint(root: Path, *, with_lora: bool = True, seed: int = 5) -> tuple[Path, "SmolVLAGRPOTrainer"]:
    torch.manual_seed(seed)
    trainer = make_trainer(legacy_args(), root / "legacy")
    saturate(trainer)
    if with_lora:
        trainer.vla_runtime = FakeLoRARuntime(seed + 1)
    path = trainer.save(global_step=56072006, args=trainer.args)
    return path, trainer


def converted(root: Path, architecture: str, legacy_path: Path, *, with_lora: bool = True):
    trainer = make_trainer(latent_args(architecture), root / architecture)
    if with_lora:
        trainer.vla_runtime = FakeLoRARuntime(999)  # different weights; must be overwritten
    info = trainer.initialize_from_legacy_checkpoint(legacy_path)
    return trainer, info


def latent_records(trainer, states, priors, *, seed: int, advantage_seed: int):
    """Latent records for the four executed slots of one decision per world."""

    offsets = torch.zeros(states.shape[0], ACTION_DIM)
    offsets[::2, 4] = torch.linspace(-0.3, 0.3, offsets[::2].shape[0])
    sample = trainer.sample_latent_action_chunks_tensor(
        states=states, priors=priors, action_count=EXECUTED,
        generator=torch.Generator().manual_seed(seed), mean_offset=offsets,
    )
    rows = {k: [] for k in ("state", "prior", "executed_action", "policy_sample",
                            "behavior_mean_offset", "action_index", "old_log_prob",
                            "likelihood_version", "advantage")}
    adv = torch.randn(states.shape[0], generator=torch.Generator().manual_seed(advantage_seed))
    for slot in range(EXECUTED):
        rows["state"].append(states)
        rows["prior"].append(priors)
        rows["executed_action"].append(sample["executed_action"][:, slot])
        rows["policy_sample"].append(sample["policy_sample"][:, slot])
        rows["behavior_mean_offset"].append(sample["behavior_mean_offset"])
        rows["action_index"].append(torch.full((states.shape[0],), slot, dtype=torch.long))
        rows["old_log_prob"].append(sample["old_log_prob"][:, slot])
        rows["likelihood_version"].append(torch.ones(states.shape[0], dtype=torch.int8))
        rows["advantage"].append(adv)
    return {k: torch.cat(v, dim=0) for k, v in rows.items()}


def run_update(trainer, records, *, seed: int, minibatch: int = 16):
    torch.manual_seed(seed)  # the update's minibatch permutation
    n = int(records["advantage"].shape[0])
    return trainer.update_tensor_records(
        records, loss_mask=torch.ones(n),
        schedule=EqualDDPSchedule(records_per_minibatch=minibatch, ppo_epochs=1, global_max_records=n),
    )


def flat(module) -> "torch.Tensor":
    return torch.cat([p.detach().flatten() for p in module.parameters()])


@unittest.skipIf(torch is None, "torch is not installed")
class InitializationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.legacy_path, self.legacy = write_legacy_checkpoint(self.root)
        self.candidate, self.lineage = converted(self.root, POLICY_ARCHITECTURE_CORRECTION, self.legacy_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_converted_means_equal_legacy_means_exactly(self):
        legacy_actor = self.legacy._unwrap(self.legacy.actor)
        candidate_actor = self.candidate._unwrap(self.candidate.actor)
        worst = 0.0
        saturated = []
        for seed in range(5):
            states, priors = inputs(64, seed)
            with torch.no_grad():
                old = legacy_actor(states, priors)
                new = candidate_actor(states, priors)
                residual, _ = legacy_actor.actor.reference_terms(states, priors)
            saturated.append(float((residual.abs() > 0.99).float().mean()))
            worst = max(worst, float((old - new).abs().max()))
            # float32 means, every slot and dimension: bitwise identical.
            self.assertTrue(torch.equal(old, new))
        print(f"\n[A1] max |mean_new - mean_old| over 320 inputs x 8 slots x 5 dims = {worst:.3e}; "
              f"inner-tanh saturated share {min(saturated):.2f}-{max(saturated):.2f}")
        self.assertGreater(min(saturated), 0.5, "the synthetic residual must be saturated")
        self.assertNotEqual(SCALE, 1.0)

    def test_correction_is_zero_and_storage_is_disjoint(self):
        actor = self.candidate._unwrap(self.candidate.actor).actor
        states, priors = inputs(32, 11)
        with torch.no_grad():
            parts = actor.components(states, priors)
        self.assertTrue(torch.equal(parts["correction"], torch.zeros_like(parts["correction"])))
        reference = {p.data_ptr() for p in actor.reference_net.parameters()}
        correction = {p.data_ptr() for p in actor.correction_net.parameters()}
        source = {v.data_ptr() for v in torch.load(self.legacy_path, weights_only=False)["policy"].values()}
        self.assertFalse(reference & correction)
        self.assertFalse((reference | correction) & source)
        legacy_state = self.legacy._unwrap(self.legacy.actor).actor.net.state_dict()
        for key, value in actor.reference_net.state_dict().items():
            self.assertTrue(torch.equal(value, legacy_state[key]), key)
        # Hidden layers copied, final layer zeroed.
        corr = actor.correction_net.state_dict()
        self.assertTrue(torch.equal(corr["net.0.weight"], legacy_state["net.0.weight"]))
        self.assertTrue(torch.equal(corr["net.2.weight"], legacy_state["net.2.weight"]))
        self.assertEqual(float(corr["net.4.weight"].abs().sum()), 0.0)
        self.assertEqual(float(corr["net.4.bias"].abs().sum()), 0.0)
        # log_std copied exactly.
        self.assertTrue(torch.equal(
            self.candidate._unwrap(self.candidate.actor).log_std,
            self.legacy._unwrap(self.legacy.actor).log_std))

    def test_lora_restored_exactly_and_frozen(self):
        restored = {k: v for k, v in self.candidate.vla_runtime.policy.state_dict().items() if "lora_" in k}
        source = torch.load(self.legacy_path, weights_only=False)["vla_lora"]
        self.assertEqual(set(restored), set(source))
        for key, value in source.items():
            self.assertTrue(torch.equal(restored[key], value), key)
        for name, param in self.candidate.vla_runtime.policy.named_parameters():
            if "lora_" in name:
                self.assertFalse(param.requires_grad, name)
        self.assertEqual(self.candidate.lora_max_abs_change(), 0.0)

    def test_mode_changes_never_unfreeze_the_reference(self):
        policy = self.candidate._unwrap(self.candidate.actor)
        for mode in (True, False, True):
            policy.train(mode)
            self.assertFalse(policy.actor.reference_net.training)
            self.assertFalse(any(p.requires_grad for p in policy.actor.reference_net.parameters()))
            self.assertTrue(all(p.requires_grad for p in policy.actor.correction_net.parameters()))

    def test_optimizer_holds_exactly_correction_and_log_std(self):
        policy = self.candidate._unwrap(self.candidate.actor)
        expected = {id(p) for p in policy.actor.correction_net.parameters()} | {id(policy.log_std)}
        held = {id(p) for group in self.candidate.optimizer.param_groups for p in group["params"]}
        self.assertEqual(held, expected)
        self.assertEqual(len(self.candidate.optimizer.state), 0, "fresh optimizer: no moments")
        self.assertEqual(self.candidate.trainable_parameter_count(),
                         sum(p.numel() for p in policy.actor.correction_net.parameters()) + CHUNK * ACTION_DIM)
        self.assertEqual(self.candidate.optimizer_lr(), 1e-3)
        # The control arm keeps the legacy partition: everything trains.
        control, _ = converted(self.root, POLICY_ARCHITECTURE_LEGACY, self.legacy_path)
        all_params = {id(p) for p in control._unwrap(control.actor).parameters()}
        held = {id(p) for group in control.optimizer.param_groups for p in group["params"]}
        self.assertEqual(held, all_params)

    def test_lineage_is_recorded(self):
        self.assertEqual(self.lineage["source_global_step"], 56072006)
        self.assertEqual(self.lineage["init_mode"], "legacy_conversion")
        self.assertEqual(len(self.lineage["source_sha256"]), 64)
        self.assertEqual(self.candidate.gradient_step, 0)

    def test_construction_consumes_the_same_rng_as_legacy(self):
        torch.manual_seed(123)
        make_trainer(legacy_args(), self.root / "rng_a")
        after_legacy = torch.random.get_rng_state()
        torch.manual_seed(123)
        make_trainer(latent_args(POLICY_ARCHITECTURE_CORRECTION), self.root / "rng_b")
        after_candidate = torch.random.get_rng_state()
        self.assertTrue(torch.equal(after_legacy, after_candidate))

    def test_conversion_copies_without_aliasing(self):
        state = torch.load(self.legacy_path, weights_only=False)["policy"]
        new = convert_legacy_policy_state(state)
        self.assertEqual(set(k for k in new if k.startswith("actor.reference_net.")),
                         {"actor.reference_net." + k[len("actor.net."):] for k in state if k.startswith("actor.net.")})
        new["actor.reference_net.net.0.weight"].add_(1.0)
        self.assertFalse(torch.equal(new["actor.reference_net.net.0.weight"], state["actor.net.net.0.weight"]))


@unittest.skipIf(torch is None, "torch is not installed")
class LearnabilityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.legacy_path, self.legacy = write_legacy_checkpoint(self.root, with_lora=False)
        self.candidate, _ = converted(self.root, POLICY_ARCHITECTURE_CORRECTION, self.legacy_path, with_lora=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_first_step_gradient_reaches_only_the_output_layer(self):
        policy = self.candidate._unwrap(self.candidate.actor)
        states, priors = inputs(32, 3)
        weights = torch.randn(32, CHUNK, ACTION_DIM, generator=torch.Generator().manual_seed(1))
        (policy(states, priors) * weights).sum().backward()
        final = policy.actor.correction_output_layer()
        self.assertGreater(float(final.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(final.bias.grad.abs().sum()), 0.0)
        for layer in (policy.actor.correction_net.net[0], policy.actor.correction_net.net[2]):
            self.assertEqual(float(layer.weight.grad.abs().sum()), 0.0)
        for param in policy.actor.reference_net.parameters():
            self.assertIsNone(param.grad)
        # Once the output weights move, the hidden layers get gradient.
        policy.zero_grad(set_to_none=True)
        with torch.no_grad():
            final.weight.normal_(0.0, 0.01)
        (policy(states, priors) * weights).sum().backward()
        self.assertGreater(float(policy.actor.correction_net.net[0].weight.grad.abs().sum()), 0.0)

    def test_updates_move_the_correction_and_never_the_reference(self):
        policy = self.candidate._unwrap(self.candidate.actor)
        reference_before = {k: v.clone() for k, v in policy.actor.reference_net.state_dict().items()}
        correction_before = flat(policy.actor.correction_net)
        states, priors = inputs(16, 4)
        for update in range(3):
            records = latent_records(self.candidate, states, priors, seed=update, advantage_seed=10 + update)
            # Update 0 is one optimizer step (four micro-batch backwards), so
            # its hidden-layer gradient is the first step's: exactly zero.
            metrics = run_update(self.candidate, records, seed=update,
                                 minibatch=64 if update == 0 else 16)
            self.assertEqual(metrics["correction/reference_max_abs_change"], 0.0)
            if update == 0:
                self.assertEqual(metrics["optimizer_steps"], 1.0)
                self.assertEqual(metrics["correction/grad_norm_hidden_mean"], 0.0)
                self.assertGreater(metrics["correction/grad_norm_final_mean"], 0.0)
        for key, value in policy.actor.reference_net.state_dict().items():
            self.assertTrue(torch.equal(value, reference_before[key]), key)
        self.assertGreater(float((flat(policy.actor.correction_net) - correction_before).abs().max()), 0.0)
        self.assertGreater(metrics["correction/grad_norm_hidden_mean"], 0.0)

    def test_negative_correction_exceeds_legacy_authority(self):
        """Mathematical authority only: not a claim of learned improvement."""

        legacy_actor = self.legacy._unwrap(self.legacy.actor).actor
        policy = self.candidate._unwrap(self.candidate.actor)
        states, priors = inputs(64, 9)
        # Legacy attainable lower bound for this exact input: inner tanh = -1.
        bound = torch.tanh(priors - legacy_actor.residual_scale)
        with torch.no_grad():
            bias = policy.actor.correction_output_layer().bias.view(CHUNK, ACTION_DIM)
            bias[:, 1] = -4.0  # y
            bias[:, 2] = -4.0  # z
            new = policy(states, priors)
        for dim in (1, 2):
            self.assertTrue(bool((new[..., dim] < bound[..., dim]).all()), dim)

    def test_unexecuted_slots_receive_no_action_loss(self):
        policy = self.candidate._unwrap(self.candidate.actor)
        states, priors = inputs(16, 5)
        records = latent_records(self.candidate, states, priors, seed=0, advantage_seed=1)
        mean, log_std = self.candidate._mean_and_log_std(
            records["state"], records["prior"], records["action_index"])
        from rl_vla_bootstrapping.policy.latent_correction_policy import latent_gaussian_log_prob

        logp = latent_gaussian_log_prob(records["policy_sample"], mean, records["behavior_mean_offset"], log_std)
        (logp * records["advantage"]).sum().backward()
        final = policy.actor.correction_output_layer()
        rows = final.weight.grad.view(CHUNK, ACTION_DIM, -1)
        self.assertGreater(float(rows[:EXECUTED].abs().sum()), 0.0)
        self.assertEqual(float(rows[EXECUTED:].abs().sum()), 0.0)
        self.assertEqual(float(policy.log_std.grad[EXECUTED:].abs().sum()), 0.0)


@unittest.skipIf(torch is None, "torch is not installed")
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.legacy_path, self.legacy = write_legacy_checkpoint(self.root, with_lora=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_legacy_checkpoints_keep_historical_outputs(self):
        payload = torch.load(self.legacy_path, weights_only=False)
        self.assertEqual(checkpoint_policy_architecture(payload), POLICY_ARCHITECTURE_LEGACY)
        # An untagged (pre-existing) legacy payload is read as legacy.
        for key in ("policy_architecture", "action_likelihood"):
            payload.pop(key)
        untagged = self.root / "untagged.pt"
        torch.save(payload, untagged)
        restored = make_trainer(legacy_args(), self.root / "restored")
        restored.load(untagged)
        states, priors = inputs(32, 2)
        actor = restored._unwrap(restored.actor).actor
        with torch.no_grad():
            features = torch.cat([states, priors.reshape(32, -1)], dim=-1)
            historical = torch.tanh(priors + SCALE * torch.tanh(actor.net(features)).reshape_as(priors))
            self.assertTrue(torch.equal(restored._unwrap(restored.actor)(states, priors), historical))

    def test_resume_preserves_correction_and_continuation_matches(self):
        states, priors = inputs(16, 6)
        batches = [latent_records(make_trainer(latent_args(POLICY_ARCHITECTURE_CORRECTION), self.root / "rec"),
                                  states, priors, seed=s, advantage_seed=20 + s) for s in range(4)]

        def fresh():
            trainer, _ = converted(self.root, POLICY_ARCHITECTURE_CORRECTION, self.legacy_path, with_lora=False)
            return trainer

        uninterrupted = fresh()
        for index, batch in enumerate(batches):
            run_update(uninterrupted, batch, seed=index)
        first = fresh()
        for index, batch in enumerate(batches[:2]):
            run_update(first, batch, seed=index)
        path = first.save(global_step=1234, args=first.args)
        payload = torch.load(path, weights_only=False)
        self.assertEqual(payload["policy_architecture"], POLICY_ARCHITECTURE_CORRECTION)
        self.assertEqual(payload["pilot_global_step"], 1234)
        self.assertEqual(payload["source_global_step"], 56072006)
        resumed = make_trainer(latent_args(POLICY_ARCHITECTURE_CORRECTION), self.root / "resumed")
        step = resumed.load(path)
        self.assertEqual(step, 1234)
        self.assertEqual(resumed.lineage["source_global_step"], 56072006)
        self.assertTrue(torch.equal(flat(resumed._unwrap(resumed.actor)), flat(first._unwrap(first.actor))))
        out_a = resumed._unwrap(resumed.actor)(states, priors)
        out_b = first._unwrap(first.actor)(states, priors)
        self.assertTrue(torch.equal(out_a, out_b))
        self.assertGreater(float(resumed._unwrap(resumed.actor).actor.correction_output_layer().weight.detach().abs().sum()), 0.0)
        for index, batch in enumerate(batches[2:], start=2):
            run_update(resumed, batch, seed=index)
        error = float((flat(resumed._unwrap(resumed.actor)) - flat(uninterrupted._unwrap(uninterrupted.actor))).abs().max())
        print(f"\n[C2] resumed vs uninterrupted continuation, max |param diff| = {error:.3e} (tolerance 0)")
        self.assertEqual(error, 0.0)

    def test_evaluator_style_loading_uses_the_saved_correction(self):
        trainer, _ = converted(self.root, POLICY_ARCHITECTURE_CORRECTION, self.legacy_path, with_lora=False)
        policy = trainer._unwrap(trainer.actor)
        with torch.no_grad():
            policy.actor.correction_output_layer().bias.fill_(-0.5)
        path = trainer.save(global_step=10, args=trainer.args)
        payload = torch.load(path, weights_only=False)
        # tools/audit/xy_approach_probe._build_world: args from the checkpoint,
        # a trainer built from them, the policy state loaded strictly.
        args = Namespace(**payload["args"])
        evaluator = SmolVLAGRPOTrainer(
            args=args, state_dim=int(payload["state_dim"]), action_dim=int(payload["action_dim"]),
            chunk_size=int(payload["chunk_size"]), run_dir=self.root / "eval", device=torch.device("cpu"))
        evaluator._unwrap(evaluator.actor).load_state_dict(payload["policy"])
        states, priors = inputs(16, 8)
        chunk = evaluator.deterministic_action_chunks_tensor(states=states, priors=priors, action_count=EXECUTED)
        reference_only = self.legacy._unwrap(self.legacy.actor)(states, priors)[:, :EXECUTED]
        parts = evaluator.action_components_tensor(states=states, priors=priors, action_count=EXECUTED)
        self.assertTrue(torch.equal(chunk, torch.tanh(parts["reference_logit"] + parts["correction"])))
        self.assertGreater(float((chunk - reference_only).detach().abs().max()), 0.05)
        # Legacy-only utilities refuse it outright.
        with self.assertRaises(RuntimeError):
            require_legacy_policy_checkpoint(payload, "test")
        from tools.audit.reference_anchor_drift import build_residual

        with self.assertRaises(RuntimeError):
            build_residual(payload, torch)
        from rl_vla_bootstrapping.cli.validate_cdpr_smolvla_policy import (
            _checkpoint_actor_state,
            _checkpoint_state_dim,
        )

        payload.pop("state_dim")
        self.assertEqual(_checkpoint_state_dim(payload), STATE_DIM)
        self.assertIn("correction_net.net.4.bias", _checkpoint_actor_state(payload))

    def test_ambiguous_initializers_are_rejected(self):
        candidate, _ = converted(self.root, POLICY_ARCHITECTURE_CORRECTION, self.legacy_path, with_lora=False)
        pilot_path = candidate.save(global_step=5, args=candidate.args)
        # Resume a legacy checkpoint into a latent run.
        with self.assertRaises(RuntimeError):
            make_trainer(latent_args(POLICY_ARCHITECTURE_CORRECTION), self.root / "a").load(self.legacy_path)
        # Resume a candidate checkpoint into the control architecture.
        with self.assertRaises(RuntimeError):
            make_trainer(latent_args(POLICY_ARCHITECTURE_LEGACY), self.root / "b").load(pilot_path)
        # Resume a candidate checkpoint into a legacy run.
        with self.assertRaises(RuntimeError):
            make_trainer(legacy_args(), self.root / "c").load(pilot_path)
        # Convert a checkpoint that is not legacy.
        with self.assertRaises(RuntimeError):
            make_trainer(latent_args(POLICY_ARCHITECTURE_CORRECTION), self.root / "d").initialize_from_legacy_checkpoint(pilot_path)
        # Weights-only warm start is ambiguous in the latent mode.
        with self.assertRaises(RuntimeError):
            make_trainer(latent_args(POLICY_ARCHITECTURE_CORRECTION), self.root / "e").load_weights_only(self.legacy_path)
        # A different reference scale is a different source.
        mismatched = parse_args(_argv(
            "--policy-architecture", POLICY_ARCHITECTURE_CORRECTION,
            "--action-likelihood", ACTION_LIKELIHOOD_LATENT, "--residual-scale", "1.0"))
        with self.assertRaises(RuntimeError):
            make_trainer(mismatched, self.root / "f").initialize_from_legacy_checkpoint(self.legacy_path)
        # Untagged state holding correction weights is not guessed at.
        payload = torch.load(pilot_path, weights_only=False)
        payload.pop("policy_architecture")
        with self.assertRaises(RuntimeError):
            checkpoint_policy_architecture(payload)



@unittest.skipIf(torch is None, "torch is not installed")
class RolloutTelemetryTests(unittest.TestCase):
    def test_component_clip_and_offset_summaries(self):
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import LatentRolloutTelemetry

        with tempfile.TemporaryDirectory() as tmp:
            legacy_path, _ = write_legacy_checkpoint(Path(tmp), with_lora=False)
            candidate, _ = converted(Path(tmp), POLICY_ARCHITECTURE_CORRECTION, legacy_path, with_lora=False)
            states, priors = inputs(16, 1)
            offsets = torch.zeros(16, ACTION_DIM)
            offsets[:8, 4] = 0.5
            sample = candidate.sample_latent_action_chunks_tensor(
                states=states, priors=priors, action_count=EXECUTED,
                generator=torch.Generator().manual_seed(0), mean_offset=offsets, with_components=True)
            live = torch.ones(16, dtype=torch.bool)
            live[-4:] = False
            telemetry = LatentRolloutTelemetry(torch, ACTION_DIM)
            telemetry.record_decision(sample["components"], live)
            for slot in range(EXECUTED):
                telemetry.record_slot(sample, slot, live)
            metrics = telemetry.metrics()
        self.assertEqual(metrics["policy_components/correction_logit_z_mean"], 0.0)
        self.assertEqual(metrics["policy_components/abs_correction_gripper_q95"], 0.0)
        self.assertAlmostEqual(metrics["latent/offset_gate_occupancy_fraction"], 8 / 12)
        self.assertAlmostEqual(metrics["latent/realized_offset_abs_gripper_mean"], 0.5, places=6)
        u, a = sample["policy_sample"][live], sample["executed_action"][live]
        self.assertAlmostEqual(metrics["latent/clip_fraction_z"], float((u[..., 2] != a[..., 2]).float().mean()), places=6)
        self.assertIn("policy_components/frozen_reference_inner_tanh_saturated_z_fraction", metrics)

    def test_target_y_quartile_diagnostics(self):
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import target_y_quartile_metrics

        target_y = torch.tensor([-0.1] * 4 + [0.05] * 4)       # Q1 group, Q3 group
        instruction_ids = torch.tensor([7] * 4 + [9] * 4)
        stage_usable = torch.tensor([[True, False], [False, False], [True, True]])
        record_world = torch.arange(8).repeat(2)
        record_stage = torch.tensor([0] * 8 + [2] * 8)
        loss_mask = torch.ones(16, dtype=torch.bool)
        loss_mask[8:12] = False
        advantage = torch.ones(16)
        out = target_y_quartile_metrics(
            torch, target_y=target_y, instruction_ids=instruction_ids, group_size=4,
            stage_usable=stage_usable, record_world=record_world, record_stage=record_stage,
            loss_mask=loss_mask, advantage=advantage, plate_id=7, bowl_id=9)
        self.assertEqual(out["stage_y_quartile/q1_usable_groups"], 1.0)
        self.assertEqual(out["stage_y_quartile/q3_bowl_usable_groups"], 1.0)
        self.assertEqual(out["stage_y_quartile/q1_approach_selected_records"], 4.0)
        self.assertEqual(out["stage_y_quartile/q1_placement_selected_records"], 0.0)
        self.assertEqual(out["stage_y_quartile/q3_placement_selected_records"], 4.0)
        self.assertEqual(out["stage_y_quartile/q3_bowl_placement_selected_records"], 4.0)
        self.assertEqual(out["stage_y_quartile/q2_usable_groups"], 0.0)


if __name__ == "__main__":
    unittest.main()
