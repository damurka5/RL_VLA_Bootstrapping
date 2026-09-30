"""The frozen-reference anchor: exact zero at the reference, a pull once moved."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_vla_bootstrapping.policy.rank_local_grpo import EqualDDPSchedule
from rl_vla_bootstrapping.policy.reference_anchor import ReferenceAnchor, load_anchor_rows
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import torch
from test_grpo_episode_offset_exploration import _args, _trainer


def write_bank(path: Path, policy, *, rows: int = 24, seed: int = 0) -> None:
    """A bank recorded by ``policy`` itself: actions are its deterministic mean."""

    rng = np.random.default_rng(seed)
    state = rng.normal(size=(rows, 6)).astype(np.float32)
    prior = rng.normal(scale=0.1, size=(rows, 2, 5)).astype(np.float32)
    with torch.no_grad():
        action = policy(torch.as_tensor(state), torch.as_tensor(prior))[:, :2].numpy()
    stages = np.asarray(["move_to", "pick_up", "placement"] * (rows // 3))
    np.savez(
        path,
        state=state, prior=prior, action=action,
        action_mask=np.ones((rows, 2), dtype=bool),
        stage_name=stages,
        destination=np.asarray(["plate", "bowl"] * (rows // 2)),
        target_catalog=np.asarray(["apple"] * rows),
    )


def perturb(module, scale: float = 0.05, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=generator))


def build(trainer, root: Path, *, coef: float = 1.0, stages=("move_to", "pick_up"), **kwargs):
    base = trainer._unwrap(trainer.actor)
    reference = root / "reference.pt"
    if not reference.exists():
        torch.save({"policy": base.state_dict()}, reference)
    bank = root / "bank.npz"
    if not bank.exists():
        write_bank(bank, base)
    return ReferenceAnchor.build(
        torch=torch, actor=base, bank_path=bank, reference_checkpoint=reference,
        stages=list(stages), coef=coef, batch_size=8, seed=0,
        device=torch.device("cpu"), **kwargs,
    )


def zero_advantage_records(trainer, n: int = 8):
    states = torch.randn(n, 6)
    priors = torch.zeros(n, 2, 5)
    actions, log_probs, _ = trainer.sample_action_chunks_tensor(
        states=states, priors=priors, action_count=2, generator=torch.Generator().manual_seed(4))
    return {
        "state": states, "prior": priors, "action": actions[:, 0],
        "action_index": torch.zeros(n, dtype=torch.long), "old_log_prob": log_probs[:, 0],
        "advantage": torch.zeros(n), "credit_stage": torch.ones(n, dtype=torch.long),
    }


@unittest.skipIf(torch is None, "torch is not installed")
class ReferenceAnchorTests(unittest.TestCase):
    def test_zero_at_the_reference_and_positive_once_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = _trainer(_args(), root)
            anchor = build(trainer, root)
            base = trainer._unwrap(trainer.actor)
            self.assertEqual(anchor.full_kl(base), 0.0)
            # The bank was recorded by the reference, so its actions ARE the mean.
            self.assertLess(anchor.census["recorded_action_mse_vs_reference_mean"], 1e-12)
            self.assertEqual(anchor.census["by_stage"], {"move_to": 8, "pick_up": 8})
            perturb(base)
            self.assertGreater(anchor.full_kl(base), 0.0)

    def test_reference_is_the_checkpoint_not_the_live_actor(self):
        """A resumed pilot must stay anchored to where it started."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = _trainer(_args(), root)
            base = trainer._unwrap(trainer.actor)
            torch.save({"policy": base.state_dict()}, root / "reference.pt")
            write_bank(root / "bank.npz", base)
            perturb(base)
            anchor = build(trainer, root)
            self.assertGreater(anchor.full_kl(base), 0.0)

    def test_stage_filter_and_unknown_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = _trainer(_args(), root)
            write_bank(root / "bank.npz", trainer._unwrap(trainer.actor))
            rows = load_anchor_rows(root / "bank.npz", stages=["placement"])
            self.assertEqual(set(rows["stage_name"]), {"placement"})
            with self.assertRaises(ValueError):
                load_anchor_rows(root / "bank.npz", stages=["pickup"])

    def test_lora_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = _trainer(_args(), root)
            base = trainer._unwrap(trainer.actor)
            lora = {"expert.lora_A": torch.ones(3)}
            torch.save({"policy": base.state_dict(), "vla_lora": lora}, root / "reference.pt")
            write_bank(root / "bank.npz", base)
            build(trainer, root, runtime_lora_state={"expert.lora_A": torch.ones(3)})
            with self.assertRaises(ValueError):
                build(trainer, root, runtime_lora_state={"expert.lora_A": torch.zeros(3)})

    def test_update_pulls_the_residual_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = _trainer(_args("--entropy-coef", "0", "--action-l2", "0", "--microbatch-size", "4",
                                     "--learning-rate", "1e-2"), root)
            trainer.reference_anchor = build(trainer, root, coef=1.0)
            base = trainer._unwrap(trainer.actor)
            perturb(base, scale=0.1)
            before = trainer.reference_anchor.full_kl(base)
            schedule = EqualDDPSchedule(records_per_minibatch=8, ppo_epochs=1, global_max_records=8)
            for _ in range(5):
                metrics = trainer.update_tensor_records(
                    zero_advantage_records(trainer), loss_mask=torch.ones(8), schedule=schedule)
            self.assertLess(metrics["anchor/kl_bank"], before)
            self.assertGreater(metrics["anchor/kl_batch_mean"], 0.0)
            self.assertGreater(metrics["anchor/grad_norm_first"], 0.0)

    def test_coef_zero_only_measures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = _trainer(_args("--entropy-coef", "0", "--action-l2", "0", "--microbatch-size", "4"), root)
            trainer.reference_anchor = build(trainer, root, coef=0.0)
            base = trainer._unwrap(trainer.actor)
            perturb(base)
            start = {key: value.clone() for key, value in base.state_dict().items()}
            metrics = trainer.update_tensor_records(
                zero_advantage_records(trainer), loss_mask=torch.ones(8),
                schedule=EqualDDPSchedule(records_per_minibatch=8, ppo_epochs=1, global_max_records=8))
            # Zero advantages and no anchor pull: the weights do not move.
            for key, value in base.state_dict().items():
                self.assertTrue(torch.allclose(value, start[key]), key)
            self.assertGreater(metrics["anchor/kl_bank"], 0.0)
            self.assertEqual(metrics["anchor/kl_batch_mean"], 0.0)


class AnchorMetricSyncTests(unittest.TestCase):
    def test_anchor_metrics_are_rank_means(self):
        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import _RANK_MEAN_UPDATE_METRICS

        for key in ("anchor/coef", "anchor/kl_bank", "anchor/grad_norm_first", "anchor/rows"):
            self.assertIn(key, _RANK_MEAN_UPDATE_METRICS)


class AnchorEnvKnobTests(unittest.TestCase):
    def test_env_knobs_reach_the_training_argv_and_parse(self):
        import os
        from unittest import mock

        from rl_vla_bootstrapping.core.config import load_project_config
        from rl_vla_bootstrapping.policy.smolvla import build_smolvla_rl_plan
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args

        root = Path(__file__).resolve().parents[1]
        config = load_project_config(
            root / "configs" / "examples" / "cdpr_smolvla_three_stage_put_into.yaml"
        )
        env = {
            "RLVLA_SMOLVLA_REFERENCE_ANCHOR_BANK": "/bank.npz",
            "RLVLA_SMOLVLA_REFERENCE_ANCHOR_CHECKPOINT": "/ref.pt",
            "RLVLA_SMOLVLA_REFERENCE_ANCHOR_COEF": "0.3",
            "RLVLA_SMOLVLA_REFERENCE_ANCHOR_BATCH": "128",
            "RLVLA_SMOLVLA_REFERENCE_ANCHOR_STAGES": "pick_up",
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env):
            plan = build_smolvla_rl_plan(config, Path(tmp))
        argv = [str(item) for item in plan.command]
        flags = [
            "--reference-anchor-bank", "--reference-anchor-checkpoint",
            "--reference-anchor-coef", "--reference-anchor-batch-size",
            "--reference-anchor-stages",
        ]
        passed = [item for flag in flags for item in (flag, argv[argv.index(flag) + 1])]
        args = parse_args(passed)
        self.assertEqual(args.reference_anchor_bank, "/bank.npz")
        self.assertEqual(args.reference_anchor_checkpoint, "/ref.pt")
        self.assertAlmostEqual(args.reference_anchor_coef, 0.3)
        self.assertEqual(args.reference_anchor_batch_size, 128)
        self.assertEqual(args.reference_anchor_stages, "pick_up")


if __name__ == "__main__":
    unittest.main()
