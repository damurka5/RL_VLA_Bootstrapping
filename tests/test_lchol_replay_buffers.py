from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rl_vla_bootstrapping.lchol.replay_buffers import PerOptionReplayBuffer
from rl_vla_bootstrapping.lchol.grpo_runtime import LCHOLGRPOConfig, LCHOLGRPORuntime
from robots.cdpr.cdpr_dataset.cdpr_lchol_spec import CDPRLCHOLSpec


class LCHOLReplayBufferTests(unittest.TestCase):
    def test_sampling_respects_allowed_options_and_capacity(self):
        replay = PerOptionReplayBuffer(capacity_per_option=2)
        replay.add("grab_object", "grab-old")
        replay.add("grab_object", "grab-new")
        replay.add("grab_object", "grab-newest")
        replay.add("push_left", "push")

        self.assertEqual(replay.sizes()["grab_object"], 2)

        samples = replay.sample_balanced(
            batch_size=8,
            rng=np.random.default_rng(0),
            allowed_options=["push_left"],
        )

        self.assertEqual(samples, ["push"] * 8)

    def test_runtime_metrics_include_replay_episode_counts(self):
        runtime = LCHOLGRPORuntime(
            config=LCHOLGRPOConfig(enabled=True),
            spec=CDPRLCHOLSpec(),
            available_options=("grab_object",),
            seed=0,
        )

        runtime.capture_candidate(
            obs={"instruction": "put apple into plate"},
            step_info={
                "env_instance_id": 0,
                "episode_index": 7,
                "instruction_type": "put_into_plate",
                "target_object_catalog": "ycb_apple",
                "source_instruction": "put apple into plate",
                "distance_ee_to_object_xy": 0.02,
                "gripper_closed": 1.0,
                "caught_object_is_target": 1.0,
            },
            sampled_action=np.zeros((4,), dtype=np.float32),
            group_score=1.0,
            update=1,
            global_step=1,
        )

        metrics = runtime.metrics()

        self.assertEqual(metrics["replay/total_records"], 1.0)
        self.assertEqual(metrics["replay/episodes_total"], 1.0)
        self.assertEqual(metrics["replay/episodes/grab_object"], 1.0)

    def test_runtime_metrics_include_reverse_shell_success_rates(self):
        runtime = LCHOLGRPORuntime(
            config=LCHOLGRPOConfig(enabled=True, curriculum="reverse_frontier"),
            spec=CDPRLCHOLSpec(),
            available_options=("put_into_plate",),
            seed=0,
        )

        runtime.record_reverse_validation(
            [
                {
                    "instruction_id": "put_into_plate",
                    "shell_id": 2,
                    "success_rate": 0.42,
                    "rollouts": 50,
                    "action_saturation_rate": 0.10,
                }
            ]
        )

        metrics = runtime.metrics()

        self.assertAlmostEqual(
            metrics["reverse_frontier/shell_success_rate/put_into_plate/shell_02"],
            0.42,
        )

    def test_sparse_start_requests_stage_specific_grpo_stats_initialization(self):
        with mock.patch.dict(
            os.environ,
            {
                "RLVLA_TASK_METADATA_JSON": (
                    '{"lchol_start_stage":"sparse",'
                    '"sparse_stage_instruction_types":["move_to_object"]}'
                )
            },
        ):
            runtime = LCHOLGRPORuntime(
                config=LCHOLGRPOConfig(enabled=True),
                spec=CDPRLCHOLSpec(),
                available_options=("move_to_object",),
                seed=0,
            )

        self.assertTrue(runtime.consume_grpo_stats_reset_request())
        self.assertFalse(runtime.consume_grpo_stats_reset_request())

    def test_sparse_episode_outcomes_write_binary_reward_csv_and_summary(self):
        with mock.patch.dict(
            os.environ,
            {
                "RLVLA_TASK_METADATA_JSON": (
                    '{"lchol_start_stage":"sparse",'
                    '"sparse_stage_instruction_types":["grab_object"]}'
                )
            },
        ):
            runtime = LCHOLGRPORuntime(
                config=LCHOLGRPOConfig(enabled=True),
                spec=CDPRLCHOLSpec(),
                available_options=("grab_object",),
                seed=0,
                rank=2,
            )

        success_info = {
            "env_instance_id": 0,
            "episode_index": 7,
            "instruction_type": "put_into_plate",
            "curriculum_shell": 1,
            "target_object_catalog": "ycb_apple",
            "source_instruction": "put apple into plate",
            "distance_ee_to_object_xy": 0.02,
            "gripper_closed": 1.0,
            "caught_object_is_target": 1.0,
            "success": True,
            "terminated": True,
            "env_done": True,
        }
        capture_info = dict(success_info)
        capture_info.pop("success")
        capture_info.pop("terminated")
        capture_info.pop("env_done")
        runtime.capture_candidate(
            obs={"instruction": "put apple into plate"},
            step_info=capture_info,
            sampled_action=np.zeros((4,), dtype=np.float32),
            group_score=1.0,
            update=1,
            global_step=1,
        )
        runtime.record_selected_step(
            step_info=success_info,
            env_reward=1.0,
            shaped_reward=1.0,
            done=True,
            env_done=True,
            forced_scene_refresh=False,
            forced_unstable_reset=False,
            update=1,
            global_step=1,
        )
        runtime.record_selected_step(
            step_info={
                "env_instance_id": 0,
                "episode_index": 8,
                "instruction_type": "grab_object",
                "curriculum_shell": 1,
                "target_object_catalog": "ycb_apple",
                "success": False,
                "truncated": True,
                "env_done": True,
                "episode_timeout": True,
            },
            env_reward=0.0,
            shaped_reward=-0.25,
            done=True,
            env_done=True,
            forced_scene_refresh=False,
            forced_unstable_reset=False,
            update=1,
            global_step=2,
        )

        metrics = runtime.metrics()
        self.assertEqual(
            metrics["sparse_stage/buffer_episode_outcomes/rank_local/cumulative/episodes_total"],
            2.0,
        )
        self.assertEqual(
            metrics["sparse_stage/buffer_episode_outcomes/rank_local/cumulative/reward_1_ratio"],
            0.5,
        )

        with tempfile.TemporaryDirectory() as tmp:
            runtime.after_update(update=1, global_step=2, run_dir=tmp, is_main=True)
            stats_dir = Path(tmp) / "lchol_episode_stats"
            with (stats_dir / "sparse_episode_outcomes.csv").open(
                newline="",
                encoding="utf-8",
            ) as fp:
                rows = list(csv.DictReader(fp))
            with (stats_dir / "sparse_episode_outcome_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as fp:
                summary = list(csv.DictReader(fp))
            scalar_tags: list[str] = []

            class Writer:
                def add_scalar(self, tag, value, global_step):
                    del value, global_step
                    scalar_tags.append(str(tag))

                def flush(self):
                    return None

            runtime.log_persisted_sparse_outcomes(
                run_dir=tmp,
                tb_writer=Writer(),
                global_step=2,
            )

        self.assertEqual([row["binary_reward"] for row in rows], ["1", "0"])
        self.assertEqual([row["stored_in_replay"] for row in rows], ["1", "0"])
        self.assertEqual({row["rank"] for row in rows}, {"2"})
        overall = next(row for row in summary if row["scope"] == "all")
        self.assertEqual(overall["episodes"], "2")
        self.assertEqual(overall["reward_1_count"], "1")
        self.assertEqual(overall["reward_0_count"], "1")
        self.assertEqual(float(overall["reward_1_ratio"]), 0.5)
        self.assertIn(
            "stage/sparse/buffer_episode_outcomes/global/cumulative/reward_1_ratio",
            scalar_tags,
        )


if __name__ == "__main__":
    unittest.main()
