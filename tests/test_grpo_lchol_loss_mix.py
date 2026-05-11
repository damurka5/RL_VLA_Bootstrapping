from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

_INSERTED_TORCH_STUB = False

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch_stub.set_float32_matmul_precision = lambda *args, **kwargs: None
    sys.modules["torch"] = torch_stub
    _INSERTED_TORCH_STUB = True

if "PIL" not in sys.modules or "PIL.Image" not in sys.modules:
    pil_stub = types.ModuleType("PIL")
    image_stub = types.ModuleType("PIL.Image")
    image_stub.fromarray = lambda arr: object()
    pil_stub.Image = image_stub
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = image_stub

from rl_vla_bootstrapping.policy.grpo_finetune_cdpr_fast import (
    _split_wrapper_argv,
    _transform_external_grpo_source_for_lchol,
)

if _INSERTED_TORCH_STUB:
    sys.modules.pop("torch", None)


class GRPOLCHOLPatchTests(unittest.TestCase):
    def test_wrapper_strips_lchol_args_before_external_parser(self):
        _external, forwarded, fast_args = _split_wrapper_argv(
            [
                "--lchol_enabled",
                "--lchol_hindsight_bc_coef",
                "0.35",
                "--rollout_steps",
                "4",
            ]
        )

        self.assertEqual(forwarded, ["--rollout_steps", "4"])
        self.assertTrue(fast_args.lchol.enabled)
        self.assertAlmostEqual(fast_args.lchol.hindsight_bc_coef, 0.35)

    def test_source_transform_adds_phase_score_and_bc_loss(self):
        external = Path("/Users/damirnurtdinov/Desktop/My Courses/Диплом/openvla-oft/vla-scripts/grpo_finetune_cdpr.py")
        if not external.exists():
            self.skipTest("Local OpenVLA-OFT GRPO script is not available.")

        patched = _transform_external_grpo_source_for_lchol(external.read_text(encoding="utf-8"))

        self.assertIn("candidate_rewards.append(float(candidate_group_score))", patched)
        self.assertIn("lchol_bc_loss = _rlvla_lchol_bc_loss", patched)
        self.assertIn("loss_lchol_bc_mean", patched)


if __name__ == "__main__":
    unittest.main()
