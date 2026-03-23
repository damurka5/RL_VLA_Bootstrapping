from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

_INSERTED_TORCH_STUB = False

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
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

from rl_vla_bootstrapping.policy.ppo_finetune_cdpr_fast import (
    _RolloutTensorboardLogger,
    _split_wrapper_argv,
)

if _INSERTED_TORCH_STUB:
    sys.modules.pop("torch", None)


class _FakeSummaryWriter:
    instances: list["_FakeSummaryWriter"] = []

    def __init__(self, log_dir: str, flush_secs: int):
        self.log_dir = log_dir
        self.flush_secs = flush_secs
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_calls = 0
        self.closed = False
        self.__class__.instances.append(self)

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        self.scalars.append((str(tag), float(value), int(global_step)))

    def flush(self) -> None:
        self.flush_calls += 1

    def close(self) -> None:
        self.closed = True


class FastPPOWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSummaryWriter.instances.clear()

    def test_split_wrapper_argv_strips_wrapper_only_options(self):
        external_script, forwarded, fast_args = _split_wrapper_argv(
            [
                "--external_ppo_script",
                "/tmp/external_ppo.py",
                "--tensorboard_rollout_every_global_steps",
                "100",
                "--rollout_steps",
                "170",
            ]
        )

        self.assertEqual(external_script, Path("/tmp/external_ppo.py").resolve())
        self.assertEqual(forwarded, ["--rollout_steps", "170"])
        self.assertEqual(fast_args.tensorboard_rollout_every_global_steps, 100)

    def test_rollout_tensorboard_logger_emits_window_means_on_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _RolloutTensorboardLogger(_FakeSummaryWriter, every_global_steps=3)
            logger.set_run_dir(Path(tmp))

            for reward_env, reward_shaped, success in (
                (1.0, 1.5, False),
                (2.0, 2.5, True),
                (3.0, 3.5, False),
            ):
                logger.capture_reward(
                    env_reward=reward_env,
                    shaped_reward=reward_shaped,
                    closer_bonus=0.2,
                    farther_penalty=0.0,
                    distance_delta_raw=0.1,
                )
                logger.finalize_step(
                    {
                        "success": success,
                        "target_grasped": success,
                        "unstable_transition": False,
                        "reward_env_clipped": False,
                        "reward_env_non_finite": False,
                    },
                    {
                        "r_xyz": 0.4,
                        "r_orient": 0.3,
                        "r_obj": 0.2,
                        "r_success": 1.0 if success else 0.0,
                    },
                )

            self.assertEqual(logger.global_step, 3)
            self.assertEqual(len(_FakeSummaryWriter.instances), 1)

            writer = _FakeSummaryWriter.instances[0]
            self.assertTrue(writer.log_dir.endswith("tensorboard"))
            scalars = {tag: (value, step) for tag, value, step in writer.scalars}

            self.assertEqual(scalars["rollout_step/reward_env_mean"], (2.0, 3))
            self.assertEqual(scalars["rollout_step/reward_shaped_mean"], (2.5, 3))
            self.assertEqual(scalars["rollout_step/success_rate_mean"], (1.0 / 3.0, 3))
            self.assertEqual(scalars["rollout_step/window_size"], (3.0, 3))
            self.assertGreaterEqual(writer.flush_calls, 1)


if __name__ == "__main__":
    unittest.main()
