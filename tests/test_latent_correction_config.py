"""The pilot config differs from the historical one only where declared.

Spec section 4 keeps the task, staged sparse reward, scenes, controller and
evaluation protocol unchanged. The only training.rl.args additions allowed are
the architecture and likelihood tags; the pilot knobs (PPO epochs, LR, update
cap, step budget) are launcher overrides recorded as deviations, not YAML edits.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "configs" / "examples" / "cdpr_smolvla_three_stage_put_into.yaml"
PILOT = ROOT / "configs" / "examples" / "cdpr_smolvla_three_stage_put_into_latent_correction.yaml"
DECLARED_ARGS = {
    "policy_architecture": "frozen_reference_logit_correction_v1",
    "action_likelihood": "latent_gaussian_conditional_offset_v1",
}


def _diff(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        out = []
        for key in sorted(set(left) | set(right), key=str):
            out += _diff(left.get(key, "<absent>"), right.get(key, "<absent>"), f"{path}.{key}")
        return out
    return [] if left == right else [path]


class PilotConfigDifferenceTests(unittest.TestCase):
    def test_only_declared_fields_differ(self):
        historical = yaml.safe_load(HISTORICAL.read_text(encoding="utf-8"))
        pilot = yaml.safe_load(PILOT.read_text(encoding="utf-8"))
        differences = set(_diff(historical, pilot))
        allowed = {".project.name", ".remote.notes"} | {
            f".training.rl.args.{key}" for key in DECLARED_ARGS
        }
        self.assertEqual(differences - allowed, set())
        for key, value in DECLARED_ARGS.items():
            self.assertNotIn(key, historical["training"]["rl"]["args"])
            self.assertEqual(pilot["training"]["rl"]["args"][key], value)
        # The added note is prepended; every historical note survives verbatim.
        self.assertEqual(pilot["remote"]["notes"][1:], historical["remote"]["notes"])
        # Pilot knobs stay at their historical YAML values.
        for key in ("ppo_epochs", "learning_rate", "mjwarp_max_updates", "max_train_steps",
                    "episode_offset_std", "three_stage_stage_loss_weights"):
            self.assertEqual(pilot["training"]["rl"]["args"][key], historical["training"]["rl"]["args"][key])

    def test_pilot_config_parses_into_the_candidate(self):
        from rl_vla_bootstrapping.core.commands import append_cli_arg
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args

        section = yaml.safe_load(PILOT.read_text(encoding="utf-8"))["training"]["rl"]["args"]
        argv: list[str] = []
        for key, value in section.items():
            append_cli_arg(argv, key, value)
        args = parse_args(argv)
        self.assertEqual(args.policy_architecture, DECLARED_ARGS["policy_architecture"])
        self.assertEqual(args.action_likelihood, DECLARED_ARGS["action_likelihood"])
        self.assertEqual(args.episode_offset_std, [0.0, 0.0, 0.0, 0.0, 0.15])
        self.assertTrue(args.train_vla_lora)
        self.assertFalse(args.vla_lora_updates_enabled)
        # The control arm overrides only the architecture.
        append_cli_arg(argv, "policy_architecture", "bounded_residual_v0")
        self.assertEqual(parse_args(argv).policy_architecture, "bounded_residual_v0")

    def test_env_knobs_reach_the_training_argv(self):
        from rl_vla_bootstrapping.core.config import load_project_config
        from rl_vla_bootstrapping.policy.smolvla import build_smolvla_rl_plan

        config = load_project_config(PILOT)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
            "RLVLA_SMOLVLA_LEGACY_INIT_CHECKPOINT": "/x/step_56072006/smolvla_grpo_adapter.pt",
            "RLVLA_SMOLVLA_POLICY_ARCHITECTURE": "bounded_residual_v0",
        }):
            argv = [str(item) for item in build_smolvla_rl_plan(config, Path(tmp)).command]
        index = argv.index("--legacy-init-checkpoint")
        self.assertEqual(argv[index + 1], "/x/step_56072006/smolvla_grpo_adapter.pt")
        index = max(i for i, item in enumerate(argv) if item == "--policy-architecture")
        self.assertEqual(argv[index + 1], "bounded_residual_v0")


class ArgumentValidationTests(unittest.TestCase):
    def _parse(self, *extra):
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args

        return parse_args(["--device", "cpu", "--no-distributed", *extra])

    def test_unsupported_combinations_are_refused(self):
        latent = ("--action-likelihood", "latent_gaussian_conditional_offset_v1")
        with self.assertRaises(SystemExit):  # correction needs the latent likelihood
            self._parse("--policy-architecture", "frozen_reference_logit_correction_v1")
        with self.assertRaises(SystemExit):  # no LoRA update path
            self._parse(*latent, "--train-vla-lora", "--vla-lora-updates-enabled")
        with self.assertRaises(SystemExit):  # conversion and resume together
            self._parse(*latent, "--legacy-init-checkpoint", "a.pt", "--resume-checkpoint", "b.pt")
        with self.assertRaises(SystemExit):  # conversion is the latent pilot's
            self._parse("--legacy-init-checkpoint", "a.pt")
        with self.assertRaises(SystemExit):  # no anchor in the pilot
            self._parse(*latent, "--reference-anchor-bank", "bank.npz")
        defaults = self._parse()
        self.assertEqual(defaults.policy_architecture, "bounded_residual_v0")
        self.assertEqual(defaults.action_likelihood, "clipped_action_v0")


if __name__ == "__main__":
    unittest.main()
