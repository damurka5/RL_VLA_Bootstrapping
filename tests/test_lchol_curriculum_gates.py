from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from rl_vla_bootstrapping.lchol.curriculum import StrictSuccessCurriculum
from rl_vla_bootstrapping.lchol.grpo_runtime import LCHOLGRPOConfig, LCHOLGRPORuntime


class LCHOLCurriculumGateTests(unittest.TestCase):
    def test_success_gate_promotes_stage(self):
        curriculum = StrictSuccessCurriculum(min_success_samples=3, window_size=8)
        self.assertEqual(curriculum.stage.name, "approach")

        for _ in range(3):
            curriculum.record({"instruction_type": "move_to_object", "success": True})
            curriculum.record({"instruction_type": "grab_object", "success": True})

        self.assertEqual(curriculum.stage.name, "grasp")
        self.assertIn("pick_up", curriculum.allowed_options())

    def test_sampling_biases_weak_option(self):
        curriculum = StrictSuccessCurriculum(min_success_samples=3, window_size=8)
        for _ in range(4):
            curriculum.record({"instruction_type": "move_to_object", "success": True})
            curriculum.record({"instruction_type": "grab_object", "success": True})
        for _ in range(4):
            curriculum.record({"instruction_type": "pick_up", "success": False})

        draws = [
            curriculum.sample_option(rng=np.random.default_rng(seed), available_options=["grab_object", "pick_up"])
            for seed in range(40)
        ]

        self.assertGreater(draws.count("pick_up"), draws.count("grab_object"))

    def test_relation_motion_flag_alone_does_not_count_as_success(self):
        curriculum = StrictSuccessCurriculum(min_success_samples=3, window_size=8)

        for _ in range(3):
            curriculum.record({"instruction_type": "move_to_object", "success": True})
            curriculum.record({"instruction_type": "grab_object", "relation_motion_ok": True})

        self.assertEqual(curriculum.stage.name, "approach")

    def test_front_and_behind_options_are_available_in_later_stage(self):
        curriculum = StrictSuccessCurriculum(min_success_samples=1, window_size=8)
        for option in ("move_to_object", "grab_object"):
            curriculum.record({"instruction_type": option, "success": True})
        for option in ("grab_object", "pick_up"):
            curriculum.record({"instruction_type": option, "success": True})
        for option in ("push_left", "push_right", "push_forward", "push_backward"):
            curriculum.record({"instruction_type": option, "success": True})
        curriculum.record({"instruction_type": "put_into_plate", "success": True})

        self.assertIn("move_in_front_of_object", curriculum.allowed_options())
        self.assertIn("move_behind_object", curriculum.allowed_options())
        self.assertIn("put_in_front_of_object", curriculum.allowed_options())
        self.assertIn("put_behind_object", curriculum.allowed_options())

    def test_runtime_dense_gate_arms_sparse_stage_after_mean_success_threshold(self):
        metadata = {
            "dense_to_sparse_success_threshold": 0.70,
            "dense_to_sparse_min_success_samples": 2,
            "dense_stage_instruction_types": ["catch_object", "release_object"],
            "dense_stage_metadata": {"reward_mode": "dense"},
            "sparse_stage_metadata": {"reward_mode": "sparse_binary"},
            "reward_mode": "sparse_binary",
        }

        with mock.patch.dict("os.environ", {"RLVLA_TASK_METADATA_JSON": json.dumps(metadata)}):
            runtime = LCHOLGRPORuntime(
                config=LCHOLGRPOConfig(enabled=True),
                spec=object(),
                available_options=("move_to_object", "grab_object"),
                seed=0,
            )

        self.assertTrue(runtime.dense_gate_active())
        self.assertIn(runtime.sample_reset_options()["instruction_type"], {"catch_object", "release_object"})
        self.assertEqual(runtime.current_task_metadata()["reward_mode"], "dense")

        runtime.record_dense_validation(
            [
                {"instruction_id": "catch_object", "success_rate": 0.60, "rollouts": 2},
                {"instruction_id": "release_object", "success_rate": 0.80, "rollouts": 2},
            ]
        )
        self.assertTrue(runtime.dense_gate_active())

        runtime.record_dense_validation(
            [
                {"instruction_id": "catch_object", "success_rate": 0.80, "rollouts": 2},
                {"instruction_id": "release_object", "success_rate": 0.80, "rollouts": 2},
            ]
        )
        self.assertFalse(runtime.dense_gate_active())
        self.assertEqual(runtime.current_task_metadata()["reward_mode"], "sparse_binary")
        self.assertAlmostEqual(runtime.metrics()["dense_stage/mean_success"], 0.80, places=7)
        self.assertAlmostEqual(runtime.metrics()["dense_stage/success_rate/catch_object"], 0.80, places=7)
        self.assertTrue(runtime.consume_grpo_stats_reset_request())
        self.assertFalse(runtime.consume_grpo_stats_reset_request())

    def test_runtime_opens_dense_gate_at_dense_update_limit_then_stops_after_sparse_limit(self):
        metadata = {
            "dense_to_sparse_success_threshold": 0.70,
            "dense_to_sparse_min_success_samples": 2,
            "dense_stage_max_updates": 2,
            "sparse_stage_max_updates": 3,
            "dense_stage_instruction_types": ["catch_object"],
            "sparse_stage_instruction_types": ["move_to_object"],
            "dense_stage_metadata": {"reward_mode": "dense"},
            "sparse_stage_metadata": {"reward_mode": "sparse_binary"},
        }

        with mock.patch.dict("os.environ", {"RLVLA_TASK_METADATA_JSON": json.dumps(metadata)}):
            runtime = LCHOLGRPORuntime(
                config=LCHOLGRPOConfig(enabled=True),
                spec=object(),
                available_options=("move_to_object", "catch_object"),
                seed=0,
            )

        for update in (1, 2):
            runtime.before_update(update=update)
            self.assertTrue(runtime.dense_gate_active())
            runtime.after_update(update=update)

        self.assertFalse(runtime.dense_gate_active())
        self.assertTrue(runtime.consume_grpo_stats_reset_request())
        self.assertFalse(runtime.should_stop_training())

        for update in (3, 4, 5):
            runtime.before_update(update=update)
            runtime.after_update(update=update)

        self.assertEqual(runtime.dense_updates_completed, 2)
        self.assertEqual(runtime.sparse_updates_completed, 3)
        self.assertTrue(runtime.should_stop_training())

    def test_runtime_configures_env_instruction_set_for_dense_gate(self):
        metadata = {
            "dense_to_sparse_success_threshold": 0.70,
            "dense_stage_instruction_types": ["catch_object"],
            "sparse_stage_instruction_types": ["move_to_object"],
            "dense_stage_metadata": {"reward_mode": "dense"},
            "reward_mode": "sparse_binary",
        }

        with mock.patch.dict("os.environ", {"RLVLA_TASK_METADATA_JSON": json.dumps(metadata)}):
            runtime = LCHOLGRPORuntime(
                config=LCHOLGRPOConfig(enabled=True),
                spec=object(),
                available_options=("move_to_object", "grab_object"),
                seed=0,
            )

        env = type("Env", (), {"_task_metadata": {}, "instruction_types": ("move_to_object",)})()
        runtime.configure_env_for_current_stage(env)
        self.assertEqual(env.instruction_types, ("catch_object",))
        self.assertEqual(env._task_metadata["reward_mode"], "dense")

        runtime.record_dense_validation([{"instruction_id": "catch_object", "success_rate": 1.0, "rollouts": 1}])
        runtime.configure_env_for_current_stage(env)
        self.assertEqual(env.instruction_types, ("move_to_object",))
        self.assertEqual(env._task_metadata["reward_mode"], "sparse_binary")

    def test_runtime_syncs_dense_gate_state_and_requests_stat_reset(self):
        metadata = {
            "dense_to_sparse_success_threshold": 0.70,
            "dense_stage_instruction_types": ["catch_object"],
            "sparse_stage_instruction_types": ["move_to_object"],
            "dense_stage_metadata": {"reward_mode": "dense"},
            "reward_mode": "sparse_binary",
        }

        with mock.patch.dict("os.environ", {"RLVLA_TASK_METADATA_JSON": json.dumps(metadata)}):
            writer_runtime = LCHOLGRPORuntime(
                config=LCHOLGRPOConfig(enabled=True),
                spec=object(),
                available_options=("move_to_object", "catch_object"),
                seed=0,
            )
            reader_runtime = LCHOLGRPORuntime(
                config=LCHOLGRPOConfig(enabled=True),
                spec=object(),
                available_options=("move_to_object", "catch_object"),
                seed=1,
            )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            writer_runtime.record_dense_validation(
                [{"instruction_id": "catch_object", "success_rate": 1.0, "rollouts": 1}],
                run_dir=run_dir,
                update=3,
            )

            self.assertTrue(reader_runtime.dense_gate_active())
            reader_runtime.sync_dense_gate_state(run_dir=run_dir)

        self.assertFalse(reader_runtime.dense_gate_active())
        self.assertEqual(reader_runtime.dense_gate_success["catch_object"], 1.0)
        self.assertEqual(reader_runtime.dense_gate_rollouts["catch_object"], 1)
        self.assertTrue(reader_runtime.consume_grpo_stats_reset_request())

    def test_runtime_logs_dense_warmup_success_under_stage_namespace(self):
        metadata = {
            "dense_to_sparse_success_threshold": 0.70,
            "dense_stage_instruction_types": ["catch_object"],
            "dense_stage_metadata": {"reward_mode": "dense"},
        }

        with mock.patch.dict("os.environ", {"RLVLA_TASK_METADATA_JSON": json.dumps(metadata)}):
            runtime = LCHOLGRPORuntime(
                config=LCHOLGRPOConfig(enabled=True),
                spec=object(),
                available_options=("move_to_object", "catch_object"),
                seed=0,
            )
        runtime.record_dense_validation(
            [{"instruction_id": "catch_object", "success_rate": 0.5, "rollouts": 1, "mean_reward": 2.5}]
        )

        class FakeWriter:
            def __init__(self):
                self.scalars: list[tuple[str, float, int]] = []

            def add_scalar(self, tag, value, step):
                self.scalars.append((str(tag), float(value), int(step)))

            def flush(self):
                pass

        writer = FakeWriter()
        runtime.log_update(update=1, global_step=10, tb_writer=writer, is_main=True)

        tags = {tag for tag, _value, _step in writer.scalars}
        self.assertIn("stage/dense/mean_success", tags)
        self.assertIn("stage/dense/mean_reward", tags)
        self.assertIn("stage/dense/success_rate/catch_object", tags)
        self.assertIn("stage/dense/reward/catch_object", tags)
        self.assertNotIn("lchol/dense_stage/mean_success", tags)


if __name__ == "__main__":
    unittest.main()
