from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rl_vla_bootstrapping.cli.validate_cdpr_smolvla_policy import (
    _checkpoint_state_dim,
    _configure_checkpoint_compatible_object_slots,
    _max_objects_from_state_dim,
)
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline
from rl_vla_bootstrapping.policy.octo_finetune_cdpr import _format_step_progress
from rl_vla_bootstrapping.policy.smolvla_cdpr import (
    DEFAULT_SMOLVLA_CHECKPOINT,
    SmolVLAActionAdapterSpec,
    SmolVLAObservationSpec,
    SmolVLARuntime,
    _resolve_torch_device,
    adapt_cdpr_observations_to_smolvla_batch,
    adapt_smolvla_actions_to_cdpr,
    cdpr_state_from_observation,
    torch,
)
from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import (
    DenseRolloutTelemetry,
    _exploration_noise,
    _task_aware_exploration_scales,
    parse_args as parse_dense_args,
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
    def test_plain_cuda_device_resolves_to_current_index(self):
        original_is_available = torch.cuda.is_available
        original_current_device = torch.cuda.current_device
        try:
            torch.cuda.is_available = lambda: True
            torch.cuda.current_device = lambda: 1
            device = _resolve_torch_device("cuda")
        finally:
            torch.cuda.is_available = original_is_available
            torch.cuda.current_device = original_current_device

        self.assertEqual(str(device), "cuda:1")

    def test_validation_infers_checkpoint_object_slots_from_actor_shape(self):
        payload = {
            "chunk_size": 8,
            "action_dim": 5,
            "actor": {"net.net.0.weight": np.zeros((1024, 97), dtype=np.float32)},
        }
        args = SimpleNamespace(max_objects=None)

        self.assertEqual(_checkpoint_state_dim(payload), 57)
        self.assertEqual(_max_objects_from_state_dim(57), 4)

        _configure_checkpoint_compatible_object_slots(args, payload)

        self.assertEqual(args.max_objects, 4)

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
        self.assertIn("4", plan.command)
        self.assertIn("--hidden-dim", plan.command)
        self.assertIn("1024", plan.command)
        self.assertIn("--batch-size", plan.command)
        self.assertIn("1024", plan.command)
        self.assertNotIn("--materialize-optimizer-state", plan.command)
        self.assertIn("RLVLA_TASK_METADATA_JSON", plan.env)

    def test_resume_progress_formatter_reports_run_window_eta(self):
        text = _format_step_progress(
            global_step=3_600_000,
            max_train_steps=5_500_000,
            start_step=3_500_000,
            elapsed_seconds=100.0,
        )

        self.assertIn("progress=3600000/5500000", text)
        self.assertIn("run=100000/2000000", text)
        self.assertIn("rate=1000.00 step/s", text)
        self.assertIn("eta=31m40s", text)

    def test_strict_dense_bridge_matches_sparse_tasks_and_runs_one_million_steps(self):
        config = load_project_config(
            ROOT / "configs" / "examples" / "cdpr_smolvla_strict_dense_bridge.yaml"
        )
        plan = BootstrapPipeline(config).build_stage_plans(
            ROOT / "runs" / "strict_dense_unit", ["rl"]
        )[0]
        command = plan.command

        self.assertEqual(config.task.metadata["reward_mode"], "dense")
        self.assertTrue(config.task.metadata["manipulation_dense_reward_enabled"])
        self.assertEqual(config.task.metadata["move_to_object_xy_window_high"], 0.02)
        self.assertTrue(config.task.metadata["put_require_release"])
        self.assertEqual(len(config.task.instruction_types), 8)
        self.assertEqual(command[command.index("--max-train-steps") + 1], "7700000")
        self.assertEqual(command[command.index("--noise-schedule-start-step") + 1], "6700000")
        self.assertIn("--task-aware-exploration", command)
        self.assertEqual(command[command.index("--nproc-per-node") + 1], "1")

    def test_task_aware_exploration_restarts_and_opens_only_near_release(self):
        args = parse_dense_args(
            [
                "--task-aware-exploration",
                "--exploration-noise",
                "0.12",
                "--min-exploration-noise",
                "0.035",
                "--noise-decay-steps",
                "800000",
                "--noise-schedule-start-step",
                "6700000",
            ]
        )
        self.assertAlmostEqual(_exploration_noise(args, 6_700_000), 0.12)
        self.assertAlmostEqual(_exploration_noise(args, 7_500_000), 0.035)

        far_scales, far_release = _task_aware_exploration_scales(
            args,
            instruction_type="put_into_plate",
            info={"relation_error": 0.10, "relation_motion_ok": 1.0},
            action_dim=5,
        )
        near_scales, near_release = _task_aware_exploration_scales(
            args,
            instruction_type="put_into_plate",
            info={
                "relation_error": 0.01,
                "relation_motion_ok": 1.0,
                "relation_grasp_history_ok": 1.0,
                "put_container_z_error": 0.02,
                "put_container_z_tolerance": 0.12,
            },
            action_dim=5,
        )
        self.assertFalse(far_release)
        self.assertTrue(near_release)
        self.assertLess(far_scales[4], near_scales[4])
        self.assertGreater(near_scales[2], near_scales[0])

    def test_dense_telemetry_aggregates_predicates_by_instruction(self):
        telemetry = DenseRolloutTelemetry(step_window=8, episode_window=4, action_dim=5)
        telemetry.record_step(
            instruction="push_left",
            reward=0.5,
            action=np.array([-0.8, 0.1, 0.0, 0.0, 0.0], dtype=np.float32),
            info={"push_support_ok": 1.0, "push_orthogonal_drift": 0.01},
            noise_scales=np.array([1.0, 0.25, 0.1, 0.1, 0.04], dtype=np.float32),
            release_exploration_active=False,
        )
        telemetry.record_episode(
            instruction="push_left",
            reward=3.0,
            length=12,
            success=True,
        )
        metrics = telemetry.snapshot()
        self.assertEqual(metrics["dense_telemetry/push_left/push_support_ok_mean"], 1.0)
        self.assertEqual(metrics["dense_telemetry/push_left/success_rate"], 1.0)
        self.assertAlmostEqual(
            metrics["dense_telemetry/push_left/action_abs_x_mean"],
            0.8,
            places=6,
        )

    def test_new_remote_scripts_pass_bash_syntax(self):
        scripts = [
            ROOT / "scripts" / "setup_smolvla_remote.sh",
            ROOT / "scripts" / "train_cdpr_smolvla_dense_2gpu_remote.sh",
            ROOT / "scripts" / "train_cdpr_smolvla_stage2_dual_remote.sh",
            ROOT / "scripts" / "train_cdpr_smolvla_strict_dense_bridge_remote.sh",
            ROOT / "scripts" / "train_cdpr_smolvla_complex_grpo_dual_remote.sh",
            ROOT / "scripts" / "evaluate_cdpr_smolvla_dense_remote.sh",
        ]
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(["bash", "-n", str(script)], check=False, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
