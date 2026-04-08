from __future__ import annotations

import sys
import tempfile
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

    class _DummyImage:
        def convert(self, mode: str):
            return self

    image_stub.fromarray = lambda arr: _DummyImage()
    pil_stub.Image = image_stub
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = image_stub

from rl_vla_bootstrapping.policy.grpo_finetune_cdpr_fast import (
    _infer_resume_artifacts,
    _patch_desk_texture_prepare,
    _split_wrapper_argv,
)

if _INSERTED_TORCH_STUB:
    sys.modules.pop("torch", None)


class FastGRPOWrapperTests(unittest.TestCase):
    def test_patch_desk_texture_prepare_uses_single_writer(self):
        calls: list[tuple[str, int]] = []

        def original_prepare(src_dir, run_dir, is_main, rank, max_textures):
            calls.append(("prepare", int(rank)))
            return f"prepared:{rank}"

        def broadcast_object(obj, rank):
            calls.append(("broadcast", int(rank)))
            return "broadcasted"

        module = types.SimpleNamespace(
            _prepare_desk_textures_dir=original_prepare,
            _broadcast_object=broadcast_object,
        )

        _patch_desk_texture_prepare(module)

        self.assertEqual(module._prepare_desk_textures_dir("/tmp/src", Path("/tmp/run"), False, 1, 5), "broadcasted")
        self.assertEqual(module._prepare_desk_textures_dir("/tmp/src", Path("/tmp/run"), True, 0, 5), "prepared:0")
        self.assertEqual(calls, [("broadcast", 1), ("prepare", 0)])

    def test_split_wrapper_argv_strips_wrapper_only_options(self):
        external_script, forwarded, fast_args = _split_wrapper_argv(
            [
                "--external_grpo_script",
                "/tmp/external_grpo.py",
                "--tensorboard_rollout_every_global_steps",
                "100",
                "--no-resume_actor_stats",
                "--rollout_steps",
                "170",
            ]
        )

        self.assertEqual(external_script, Path("/tmp/external_grpo.py").resolve())
        self.assertEqual(forwarded, ["--rollout_steps", "170"])
        self.assertEqual(fast_args.tensorboard_rollout_every_global_steps, 100)
        self.assertFalse(fast_args.resume_actor_stats)

    def test_infer_resume_artifacts_prefers_grpo_actor_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "step_0122400"
            adapter_dir = checkpoint_dir / "vla_cdpr_adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "grpo_actor_stats.pt").write_text("grpo", encoding="utf-8")
            (checkpoint_dir / "ppo_actor_stats.pt").write_text("ppo", encoding="utf-8")

            artifacts = _infer_resume_artifacts(["--adapter_path", str(checkpoint_dir)])

            self.assertEqual(artifacts.checkpoint_dir, checkpoint_dir.resolve())
            self.assertEqual(artifacts.actor_stats_path, (checkpoint_dir / "grpo_actor_stats.pt").resolve())

    def test_infer_resume_artifacts_falls_back_to_ppo_actor_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "step_0122400"
            adapter_dir = checkpoint_dir / "vla_cdpr_adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "ppo_actor_stats.pt").write_text("ppo", encoding="utf-8")

            artifacts = _infer_resume_artifacts(["--adapter_path", str(adapter_dir)])

            self.assertEqual(artifacts.checkpoint_dir, checkpoint_dir.resolve())
            self.assertEqual(artifacts.actor_stats_path, (checkpoint_dir / "ppo_actor_stats.pt").resolve())


if __name__ == "__main__":
    unittest.main()
