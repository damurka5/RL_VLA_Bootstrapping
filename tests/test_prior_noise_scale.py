"""The SmolVLA prior's flow-matching start noise can be scaled, and it is recorded."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from rl_vla_bootstrapping.policy.smolvla_cdpr import (
    PRIOR_NOISE_SCALE_ENV,
    apply_prior_noise_scale,
    prior_noise_scale_from_env,
)


def _policy():
    model = SimpleNamespace(sample_noise=lambda shape, device: torch.ones(shape))
    return SimpleNamespace(model=model)


class PriorNoiseScaleTests(unittest.TestCase):
    def test_unset_or_one_leaves_lerobot_untouched(self):
        policy = _policy()
        original = policy.model.sample_noise
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(PRIOR_NOISE_SCALE_ENV, None)
            self.assertIsNone(apply_prior_noise_scale(policy))
        self.assertIsNone(apply_prior_noise_scale(policy, 1.0))
        self.assertIs(policy.model.sample_noise, original)

    def test_scale_and_zero(self):
        policy = _policy()
        self.assertEqual(apply_prior_noise_scale(policy, 0.5), 0.5)
        self.assertTrue(torch.equal(policy.model.sample_noise((2, 3), "cpu"), torch.full((2, 3), 0.5)))
        # A second application does not compound.
        self.assertEqual(apply_prior_noise_scale(policy, 0.5), 0.5)
        self.assertTrue(torch.equal(policy.model.sample_noise((1,), "cpu"), torch.full((1,), 0.5)))
        zero = _policy()
        apply_prior_noise_scale(zero, 0.0)
        self.assertTrue(torch.equal(zero.model.sample_noise((4,), "cpu"), torch.zeros(4)))

    def test_env_is_validated_and_a_missing_hook_is_refused(self):
        with mock.patch.dict(os.environ, {PRIOR_NOISE_SCALE_ENV: "-1"}):
            with self.assertRaises(ValueError):
                prior_noise_scale_from_env()
        with mock.patch.dict(os.environ, {PRIOR_NOISE_SCALE_ENV: "0"}):
            self.assertEqual(prior_noise_scale_from_env(), 0.0)
            with self.assertRaises(RuntimeError):
                apply_prior_noise_scale(SimpleNamespace(model=SimpleNamespace()))

    def test_comparison_refuses_mixed_scales_unless_asked(self):
        from tools.audit.compare_put_into_evaluations import main

        with tempfile.TemporaryDirectory() as tmp:
            dirs = []
            for name, scale in (("a", None), ("b", 0.0)):
                path = Path(tmp) / name
                path.mkdir()
                (path / "evaluation.json").write_text(json.dumps({
                    "checkpoint": name, "scene_manifest_sha256": "x", "split": "v", "worlds": 2,
                    "rounds": 1, "distinct_scene_rounds": True, "decisions": 128,
                    "prior_noise_scale": scale,
                    "results": {"episodes": [{"scene_uid": "s", "strict": True, "native": True}],
                                "chains": 1, "scenes": 1, "strict": {"rate": 1.0}},
                }))
                dirs.append(str(path))
            with self.assertRaises(SystemExit):
                main(dirs)
            self.assertEqual(main([*dirs, "--allow-prior-noise-difference"]), 0)


if __name__ == "__main__":
    unittest.main()
