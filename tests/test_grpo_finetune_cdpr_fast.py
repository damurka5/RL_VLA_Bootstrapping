from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest import mock
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
    _RolloutTensorboardLogger,
    _infer_resume_artifacts,
    _patch_distributed_timeout,
    _patch_desk_texture_prepare,
    _split_wrapper_argv,
    _transform_external_grpo_source_for_ddp_sync,
)

if _INSERTED_TORCH_STUB:
    sys.modules.pop("torch", None)


class FastGRPOWrapperTests(unittest.TestCase):
    def test_rollout_tensorboard_logger_emits_sparse_manipulation_metrics(self):
        class FakeSummaryWriter:
            scalars: list[tuple[str, float, int]] = []

            def __init__(self, log_dir: str, flush_secs: int = 10):
                self.log_dir = log_dir
                self.flush_secs = flush_secs

            def add_scalar(self, tag: str, value: float, step: int):
                self.scalars.append((tag, float(value), int(step)))

            def flush(self):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            FakeSummaryWriter.scalars.clear()
            logger = _RolloutTensorboardLogger(FakeSummaryWriter, every_global_steps=2)
            logger.set_run_dir(Path(tmp))

            for idx in range(2):
                logger.capture_reward(
                    env_reward=1.0,
                    shaped_reward=1.0,
                    closer_bonus=0.0,
                    farther_penalty=0.0,
                    distance_delta_raw=0.0,
                )
                logger.finalize_step(
                    {
                        "success": idx == 1,
                        "sparse_success": idx == 1,
                        "env_done": idx == 1,
                        "distance_ee_to_object_xy": 0.03 + 0.01 * idx,
                        "target_motion_xy": 0.05 * idx,
                        "relation_error": 0.08 - 0.02 * idx,
                        "gripper_closed": 1.0,
                        "caught_object_is_target": idx == 1,
                    },
                    {},
                )

            tags = {tag for tag, _, _ in FakeSummaryWriter.scalars}
            self.assertIn("rollout_step/sparse_success_mean", tags)
            self.assertIn("rollout_step/success_rate_mean", tags)
            self.assertIn("rollout_step/episode_success_rate_mean", tags)
            self.assertIn("rollout_step/distance_ee_to_object_xy_mean", tags)
            self.assertIn("rollout_step/relation_error_mean", tags)
            self.assertIn("rollout_step/target_motion_xy_mean", tags)

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
                "--ddp_timeout_seconds",
                "14400",
                "--rollout_steps",
                "170",
            ]
        )

        self.assertEqual(external_script, Path("/tmp/external_grpo.py").resolve())
        self.assertEqual(forwarded, ["--rollout_steps", "170"])
        self.assertEqual(fast_args.tensorboard_rollout_every_global_steps, 100)
        self.assertFalse(fast_args.resume_actor_stats)
        self.assertEqual(fast_args.ddp_timeout_seconds, 14400)

    def test_patch_distributed_timeout_overrides_external_ppo_init(self):
        calls: list[tuple[str, float]] = []

        class FakeDist:
            @staticmethod
            def is_initialized():
                return False

            @staticmethod
            def init_process_group(*, backend, timeout):
                calls.append((backend, float(timeout.total_seconds())))

        fake_ppo = types.SimpleNamespace(
            dist=FakeDist,
            torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
        )
        module = types.SimpleNamespace(ppo=fake_ppo)

        _patch_distributed_timeout(module, timeout_seconds=14400)

        with mock.patch.dict(
            "os.environ",
            {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "1"},
            clear=False,
        ):
            self.assertEqual(fake_ppo._init_distributed(), (0, 1, 2))

        self.assertEqual(calls, [("nccl", 14400.0)])

    def test_ddp_sync_transform_adds_rank_barriers_before_update_and_train(self):
        source = (
            "        for update in range(1, args.total_updates + 1):\n"
            "            policy.eval()\n"
            "            do_rollout()\n"
            "            policy.train()\n"
        )

        patched = _transform_external_grpo_source_for_ddp_sync(source)

        self.assertIn('_rlvla_ddp_sync("pre_update", update=update)', patched)
        self.assertIn('_rlvla_ddp_sync("pre_train", update=update)', patched)

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
