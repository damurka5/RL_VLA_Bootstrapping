from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import numpy as np

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline
from rl_vla_bootstrapping.policy.smolvla_cdpr import (
    DEFAULT_SMOLVLA_CHECKPOINT,
    SmolVLAActionAdapterSpec,
    SmolVLAObservationSpec,
    SmolVLARuntime,
    adapt_cdpr_observations_to_smolvla_batch,
    adapt_smolvla_actions_to_cdpr,
    cdpr_state_from_observation,
    torch,
)


ROOT = Path(__file__).resolve().parents[1]


def _fake_cdpr_obs() -> dict[str, np.ndarray]:
    return {
        "ee_position": np.array([0.1, 0.2, 0.4], dtype=np.float32),
        "target_object_position": np.array([0.0, 0.0, 0.2], dtype=np.float32),
        "all_object_positions": np.zeros((2, 3), dtype=np.float32),
        "object_position_mask": np.array([1.0, 0.0], dtype=np.float32),
        "instruction_onehot": np.eye(3, dtype=np.float32)[1],
        "goal_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


class SmolVLACDPRTests(unittest.TestCase):
    def test_adapter_imports_without_lerobot_runtime(self):
        self.assertEqual(DEFAULT_SMOLVLA_CHECKPOINT, "lerobot/smolvla_base")

    def test_cdpr_state_adapter_outputs_configured_width(self):
        obs = _fake_cdpr_obs()
        state = cdpr_state_from_observation(obs, {"ee_yaw": 0.5, "gripper_opening": 0.75}, state_dim=6)

        self.assertEqual(state.shape, (6,))
        np.testing.assert_allclose(state[:5], [0.1, 0.2, 0.4, 0.5, 0.75], rtol=1e-6)
        self.assertGreaterEqual(float(state[5]), 0.0)

    def test_observation_adapter_builds_numpy_smolvla_batch(self):
        obs = _fake_cdpr_obs()
        primary = np.zeros((8, 8, 3), dtype=np.uint8)
        wrist = np.ones((8, 8, 3), dtype=np.uint8) * 255
        spec = SmolVLAObservationSpec(image_size=8, state_dim=6)

        batch = adapt_cdpr_observations_to_smolvla_batch(
            primary_images=[primary],
            wrist_images=[wrist],
            observations=[obs],
            infos=[{"language_instruction": "move left"}],
            instructions=["move left"],
            spec=spec,
            device=None,
        )

        self.assertEqual(batch["observation.state"].shape, (1, 6))
        self.assertEqual(batch["observation.images.camera1"].shape, (1, 3, 8, 8))
        self.assertEqual(batch["observation.images.camera2"].shape, (1, 3, 8, 8))
        self.assertEqual(batch["observation.images.camera3"].shape, (1, 3, 8, 8))
        self.assertEqual(batch["task"], ["move left\n"])
        self.assertEqual(float(batch["observation.images.camera2"].max()), 1.0)

    def test_action_adapter_maps_6d_smolvla_actions_to_5d_cdpr_chunks(self):
        raw = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 9.0, -0.5],
                [0.6, 0.7, 0.8, 0.9, 8.0, -1.0],
            ],
            dtype=np.float32,
        )

        chunk = adapt_smolvla_actions_to_cdpr(
            raw,
            spec=SmolVLAActionAdapterSpec(chunk_size=4, normalization="clip"),
        )

        self.assertEqual(chunk.shape, (4, 5))
        np.testing.assert_allclose(chunk[0], [0.1, 0.2, 0.3, 0.4, -0.5])
        np.testing.assert_allclose(chunk[1], [0.6, 0.7, 0.8, 0.9, -1.0])
        np.testing.assert_allclose(chunk[2], chunk[1])
        np.testing.assert_allclose(chunk[3], chunk[1])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_tokenizer_attention_mask_is_bool_for_lerobot_attention(self):
        class _Tokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.ones((len(texts), 4), dtype=torch.long),
                    "attention_mask": torch.ones((len(texts), 4), dtype=torch.long),
                }

        class _Policy:
            class config:
                tokenizer_max_length = 4

        runtime = SmolVLARuntime(
            policy=_Policy(),
            checkpoint="unit",
            device=torch.device("cpu"),
            dtype=torch.float32,
            obs_spec=SmolVLAObservationSpec(),
            action_spec=SmolVLAActionAdapterSpec(),
            tokenizer=_Tokenizer(),
        )

        input_ids, attention_mask = runtime._tokenize(["move left"])

        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(attention_mask.dtype, torch.bool)

    def test_smolvla_config_loads_and_builds_torchrun_plan(self):
        config_path = ROOT / "configs" / "examples" / "cdpr_smolvla_dense_2gpu.yaml"
        config = load_project_config(config_path)
        pipeline = BootstrapPipeline(config)
        plan = pipeline.build_stage_plans(ROOT / "runs" / "smolvla_unit", ["rl"])[0]

        self.assertEqual(config.policy.type, "smolvla_cdpr")
        self.assertEqual(config.policy.base_checkpoint, "lerobot/smolvla_base")
        self.assertIn("move_to_object", config.task.instruction_types)
        self.assertEqual(config.policy.action_codec.chunk_size, 8)
        self.assertEqual(plan.command[0], "torchrun")
        self.assertIn("-m", plan.command)
        self.assertIn("rl_vla_bootstrapping.policy.smolvla_finetune_cdpr", plan.command)
        self.assertIn("--mixed-precision", plan.command)
        self.assertIn("bf16", plan.command)
        self.assertIn("--num-envs-per-rank", plan.command)
        self.assertIn("8", plan.command)
        self.assertIn("--hidden-dim", plan.command)
        self.assertIn("12288", plan.command)
        self.assertIn("--batch-size", plan.command)
        self.assertIn("8192", plan.command)
        self.assertIn("--materialize-optimizer-state", plan.command)
        self.assertIn("RLVLA_TASK_METADATA_JSON", plan.env)

    def test_new_remote_scripts_pass_bash_syntax(self):
        scripts = [
            ROOT / "scripts" / "setup_smolvla_remote.sh",
            ROOT / "scripts" / "train_cdpr_smolvla_dense_2gpu_remote.sh",
            ROOT / "scripts" / "evaluate_cdpr_smolvla_dense_remote.sh",
        ]
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(["bash", "-n", str(script)], check=False, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
