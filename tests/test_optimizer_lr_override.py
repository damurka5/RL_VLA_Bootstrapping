"""A resume keeps the saved learning rate unless it is explicitly overridden."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from test_grpo_episode_offset_exploration import _args, _trainer


class OptimizerLrOverrideTests(unittest.TestCase):
    def _saved(self, root: Path, lr: str) -> Path:
        args = _args("--learning-rate", lr)
        trainer = _trainer(args, root / "saved")
        return trainer.save(global_step=7, args=args)

    def test_resume_ignores_the_new_yaml_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._saved(Path(tmp), "1e-4")
            trainer = _trainer(_args("--learning-rate", "3e-5"), Path(tmp) / "resumed")
            self.assertEqual(trainer.load(checkpoint), 7)
            # The trap the override exists for: the YAML says 3e-5, 1e-4 runs.
            self.assertAlmostEqual(trainer.optimizer_lr(), 1e-4)
            self.assertAlmostEqual(trainer.checkpoint_optimizer_lr, 1e-4)

    def test_override_applies_after_the_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._saved(Path(tmp), "1e-4")
            trainer = _trainer(
                _args("--learning-rate", "1e-4", "--optimizer-lr-override", "3e-5"),
                Path(tmp) / "resumed",
            )
            self.assertAlmostEqual(trainer.optimizer_lr(), 3e-5)
            trainer.load(checkpoint)
            self.assertAlmostEqual(trainer.checkpoint_optimizer_lr, 1e-4)
            self.assertAlmostEqual(trainer.optimizer_lr(), 3e-5)
            # Adam moments still come from the checkpoint.
            self.assertTrue(
                all(float(group["lr"]) == 3e-5 for group in trainer.optimizer.param_groups)
            )

    def test_override_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _trainer(_args("--optimizer-lr-override", "0"), Path(tmp))


class PilotEnvironmentTests(unittest.TestCase):
    def test_env_knobs_reach_the_training_argv(self):
        from rl_vla_bootstrapping.core.config import load_project_config
        from rl_vla_bootstrapping.policy.smolvla import build_smolvla_rl_plan

        root = Path(__file__).resolve().parents[1]
        config = load_project_config(
            root / "configs" / "examples" / "cdpr_smolvla_three_stage_put_into.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"RLVLA_SMOLVLA_PPO_EPOCHS": "1", "RLVLA_SMOLVLA_OPTIMIZER_LR_OVERRIDE": "3e-05"},
        ):
            plan = build_smolvla_rl_plan(config, Path(tmp))
        argv = [str(item) for item in plan.command]
        # The override follows the config's own value, and argparse keeps the last.
        index = max(i for i, item in enumerate(argv) if item == "--ppo-epochs")
        self.assertEqual(argv[index + 1], "1")
        index = argv.index("--optimizer-lr-override")
        self.assertAlmostEqual(float(argv[index + 1]), 3e-5)


if __name__ == "__main__":
    unittest.main()
