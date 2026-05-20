from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_vla_bootstrapping.cli.validate_cdpr_policy import (
    _default_max_steps,
    _instruction_validation_task_metadata,
    _save_episode_video,
    _validation_buckets,
    _parse_instruction_types,
    _resolve_policy_artifacts,
    _summarize_instruction_results,
    _summarize_instruction_text_results,
    _validation_env_vars,
    EpisodeResult,
)


class ValidateCDPRPolicyTests(unittest.TestCase):
    def test_default_max_steps_prefers_validation_horizon(self):
        config = type(
            "_Config",
            (),
            {
                "training": type(
                    "_Training",
                    (),
                    {
                        "rl": type(
                            "_RL",
                            (),
                            {
                                "args": {
                                    "validation_max_steps": 32,
                                    "max_env_steps": 64,
                                }
                            },
                        )()
                    },
                )()
            },
        )()
        args = type("_Args", (), {"max_steps": None})()

        self.assertEqual(_default_max_steps(config, args), 32)

    def test_resolve_policy_artifacts_prefers_checkpoint_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "step_0081600"
            adapter_dir = checkpoint_dir / "vla_cdpr_adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "action_head.pt").write_text("weights", encoding="utf-8")

            args = type(
                "_Args",
                (),
                {
                    "checkpoint_dir": str(checkpoint_dir),
                    "adapter_path": None,
                    "action_head_path": None,
                },
            )()
            config = type(
                "_Config",
                (),
                {
                    "training": type(
                        "_Training",
                        (),
                        {
                            "rl": type(
                                "_RL",
                                (),
                                {
                                    "args": {},
                                },
                            )()
                        },
                    )()
                },
            )()

            artifacts = _resolve_policy_artifacts(args, config)

            self.assertEqual(artifacts.checkpoint_dir, checkpoint_dir.resolve())
            self.assertEqual(artifacts.adapter_path, adapter_dir.resolve())
            self.assertEqual(artifacts.action_head_path, (checkpoint_dir / "action_head.pt").resolve())

    def test_summarize_instruction_results_computes_success_rate(self):
        episode_results = [
            EpisodeResult(
                episode_index=0,
                seed=1,
                instruction_type="move_up",
                instruction_text="move up",
                success=True,
                terminated=True,
                truncated=False,
                steps=12,
                reward_total=1.5,
                scene="desk",
                goal_position=[0.0, 0.0, 0.1],
                ee_start=[0.0, 0.0, 0.4],
            ),
            EpisodeResult(
                episode_index=1,
                seed=2,
                instruction_type="move_up",
                instruction_text="move up",
                success=False,
                terminated=False,
                truncated=True,
                steps=32,
                reward_total=0.5,
                scene="desk",
                goal_position=[0.0, 0.0, 0.1],
                ee_start=[0.0, 0.0, 0.4],
            ),
        ]

        summary = _summarize_instruction_results(
            instruction_type="move_up",
            episode_results=episode_results,
            video_path="/tmp/move_up.mp4",
        )

        self.assertEqual(summary.successes, 1)
        self.assertEqual(summary.episodes, 2)
        self.assertAlmostEqual(summary.success_rate, 0.5, places=7)
        self.assertAlmostEqual(summary.mean_reward, 1.0, places=7)
        self.assertAlmostEqual(summary.mean_steps, 22.0, places=7)
        self.assertEqual(summary.video_path, "/tmp/move_up.mp4")
        self.assertEqual(summary.success_video_path, "/tmp/move_up.mp4")
        self.assertIsNone(summary.failure_video_path)

    def test_summarize_instruction_results_keeps_failure_video(self):
        episode_results = [
            EpisodeResult(
                episode_index=0,
                seed=1,
                instruction_type="grab_object",
                instruction_text="grab apple",
                success=False,
                terminated=False,
                truncated=True,
                steps=120,
                reward_total=0.0,
                scene="desk",
                goal_position=[0.0, 0.0, 0.1],
                ee_start=[0.0, 0.0, 0.4],
            )
        ]

        summary = _summarize_instruction_results(
            instruction_type="grab_object",
            episode_results=episode_results,
            video_path="/tmp/grab_failure.mp4",
            failure_video_path="/tmp/grab_failure.mp4",
        )

        self.assertEqual(summary.successes, 0)
        self.assertEqual(summary.video_path, "/tmp/grab_failure.mp4")
        self.assertIsNone(summary.success_video_path)
        self.assertEqual(summary.failure_video_path, "/tmp/grab_failure.mp4")

    def test_save_episode_video_names_success_and_failure_artifacts(self):
        class FakeSim:
            overview_frames = ["frame"]

            @staticmethod
            def _estimate_video_fps():
                return 20.0

            @staticmethod
            def save_video(frames, output_path: str, fps: float):
                Path(output_path).write_text(f"{len(frames)}:{fps}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            episode = EpisodeResult(
                episode_index=3,
                seed=1,
                instruction_type="move_to_object",
                instruction_text="move to apple",
                success=False,
                terminated=False,
                truncated=True,
                steps=120,
                reward_total=0.0,
                scene="desk",
                goal_position=[0.0, 0.0, 0.1],
                ee_start=[0.0, 0.0, 0.4],
                target_object_catalog="ycb_apple",
            )

            path = _save_episode_video(
                sim=FakeSim(),
                output_dir=output_dir,
                instruction_type="move_to_object",
                episode_result=episode,
                outcome="failure",
            )

            self.assertIsNotNone(path)
            self.assertTrue(Path(path).name.startswith("move_to_object_ycb_apple_failure_episode_003"))
            self.assertTrue(Path(path).exists())
            self.assertEqual(Path(path).read_text(encoding="utf-8"), "120:10.0")
            summary_path = output_dir / "move_to_object_ycb_apple_failure_episode_003_summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["video_frame_count"], 120)
            self.assertAlmostEqual(summary["video_fps"], 10.0, places=7)
            self.assertAlmostEqual(summary["video_duration_sec"], 12.0, places=7)

    def test_parse_instruction_types_accepts_human_friendly_aliases(self):
        instruction_types = _parse_instruction_types(
            ["left", "right", "forward", "backward", "move to object"]
        )

        self.assertEqual(
            instruction_types,
            ("move_left", "move_right", "move_top", "move_bottom", "move_to_object"),
        )

    def test_summarize_instruction_text_results_groups_object_prompts_separately(self):
        episode_results = [
            EpisodeResult(
                episode_index=0,
                seed=1,
                instruction_type="move_left",
                instruction_text="move left",
                success=True,
                terminated=True,
                truncated=False,
                steps=10,
                reward_total=1.0,
                scene="desk_a",
                goal_position=[0.0, 0.0, 0.1],
                ee_start=[0.0, 0.0, 0.4],
            ),
            EpisodeResult(
                episode_index=1,
                seed=2,
                instruction_type="move_to_object",
                instruction_text="move to apple",
                success=True,
                terminated=True,
                truncated=False,
                steps=14,
                reward_total=2.0,
                scene="desk_b",
                goal_position=[0.1, 0.1, 0.1],
                ee_start=[0.0, 0.0, 0.4],
                target_object_catalog="ycb_apple",
            ),
            EpisodeResult(
                episode_index=2,
                seed=3,
                instruction_type="move_to_object",
                instruction_text="move to apple",
                success=False,
                terminated=False,
                truncated=True,
                steps=32,
                reward_total=0.5,
                scene="desk_c",
                goal_position=[0.1, 0.1, 0.1],
                ee_start=[0.0, 0.0, 0.4],
                target_object_catalog="ycb_apple",
            ),
            EpisodeResult(
                episode_index=3,
                seed=4,
                instruction_type="move_to_object",
                instruction_text="move to pear",
                success=True,
                terminated=True,
                truncated=False,
                steps=18,
                reward_total=1.5,
                scene="desk_d",
                goal_position=[0.1, -0.1, 0.1],
                ee_start=[0.0, 0.0, 0.4],
                target_object_catalog="ycb_pear",
            ),
        ]

        summaries = _summarize_instruction_text_results(episode_results)
        by_text = {summary.instruction_text: summary for summary in summaries}

        self.assertEqual(set(by_text), {"move left", "move to apple", "move to pear"})
        self.assertEqual(by_text["move left"].instruction_types, ("move_left",))
        self.assertEqual(by_text["move to apple"].target_object_catalogs, ("ycb_apple",))
        self.assertEqual(by_text["move to apple"].episodes, 2)
        self.assertAlmostEqual(by_text["move to apple"].success_rate, 0.5, places=7)
        self.assertEqual(by_text["move to pear"].episodes, 1)
        self.assertAlmostEqual(by_text["move to pear"].success_rate, 1.0, places=7)

    def test_validation_env_vars_override_success_behavior(self):
        config = type(
            "_Config",
            (),
            {
                "project": type("_Project", (), {"env": {}})(),
                "remote": type("_Remote", (), {"env_vars": {}})(),
                "task": type(
                    "_Task",
                    (),
                    {
                        "metadata": {"success_distance": 0.03},
                        "reward": None,
                        "success_predicate": None,
                        "goal_region": {},
                        "goal_relation": None,
                        "dense_reward_terms": {},
                    },
                )(),
                "training": type("_Training", (), {"rl": type("_RL", (), {"args": {}})()})(),
            },
        )()
        args = type(
            "_Args",
            (),
            {
                "success_distance": 0.05,
                "directional_displacement_threshold": 0.20,
            },
        )()

        env = _validation_env_vars(config, args)

        self.assertIn("RLVLA_TASK_METADATA_JSON", env)
        self.assertIn('"success_distance": 0.05', env["RLVLA_TASK_METADATA_JSON"])
        self.assertIn('"directional_success_displacement_threshold": 0.2', env["RLVLA_TASK_METADATA_JSON"])
        self.assertEqual(env["RLVLA_TASK_SUCCESS_ATTRIBUTE"], "compute_instruction_validation_success")
        self.assertIn("rl_instruction_tasks.py", env["RLVLA_TASK_SUCCESS_FILE"])

    def test_move_to_object_validation_metadata_forces_single_target_scenes(self):
        config = type(
            "_Config",
            (),
            {
                "project": type("_Project", (), {"env": {}})(),
                "remote": type("_Remote", (), {"env_vars": {}})(),
                "task": type(
                    "_Task",
                    (),
                    {
                        "metadata": {
                            "target_object_pool": ["ycb_apple", "ycb_pear"],
                            "distractor_object_pool": ["ycb_plate"],
                            "min_scene_objects": 1,
                            "max_scene_objects": 3,
                        },
                        "target_objects": ["ycb_apple", "ycb_pear"],
                        "reward": None,
                        "success_predicate": None,
                        "goal_region": {},
                        "goal_relation": None,
                        "dense_reward_terms": {},
                    },
                )(),
                "training": type("_Training", (), {"rl": type("_RL", (), {"args": {}})()})(),
            },
        )()
        args = type(
            "_Args",
            (),
            {
                "success_distance": 0.05,
                "directional_displacement_threshold": 0.05,
            },
        )()

        metadata = _instruction_validation_task_metadata(config, args, instruction_type="move_to_object")
        env = _validation_env_vars(config, args, instruction_type="move_to_object")

        self.assertEqual(metadata["target_object_pool"], ["ycb_apple", "ycb_pear"])
        self.assertEqual(metadata["distractor_object_pool"], [])
        self.assertEqual(metadata["min_scene_objects"], 1)
        self.assertEqual(metadata["max_scene_objects"], 1)
        self.assertAlmostEqual(metadata["move_to_object_validation_distance_threshold"], 0.10, places=7)
        self.assertIn('"distractor_object_pool": []', env["RLVLA_TASK_METADATA_JSON"])
        self.assertIn('"max_scene_objects": 1', env["RLVLA_TASK_METADATA_JSON"])
        self.assertIn('"move_to_object_validation_distance_threshold": 0.1', env["RLVLA_TASK_METADATA_JSON"])

    def test_move_to_object_validation_uses_minimum_episode_budget_per_target(self):
        config = type(
            "_Config",
            (),
            {
                "project": type("_Project", (), {"env": {}})(),
                "remote": type("_Remote", (), {"env_vars": {}})(),
                "task": type(
                    "_Task",
                    (),
                    {
                        "metadata": {
                            "target_object_pool": ["ycb_apple", "ycb_pear", "ycb_peach"],
                        },
                        "target_objects": ["ycb_apple", "ycb_pear", "ycb_peach"],
                        "reward": None,
                        "success_predicate": None,
                        "goal_region": {},
                        "goal_relation": None,
                        "dense_reward_terms": {},
                    },
                )(),
                "training": type("_Training", (), {"rl": type("_RL", (), {"args": {}})()})(),
            },
        )()
        args = type(
            "_Args",
            (),
            {
                "episodes_per_instruction": 100,
                "move_to_object_episodes_per_target": 50,
                "success_distance": 0.05,
                "directional_displacement_threshold": 0.05,
            },
        )()

        buckets = _validation_buckets(config, args, instruction_type="move_to_object")

        self.assertEqual(len(buckets), 3)
        self.assertEqual([bucket.target_object for bucket in buckets], ["ycb_apple", "ycb_pear", "ycb_peach"])
        self.assertTrue(all(bucket.episodes == 50 for bucket in buckets))
        self.assertTrue(all('"max_scene_objects": 1' in bucket.env_vars["RLVLA_TASK_METADATA_JSON"] for bucket in buckets))

    def test_move_to_object_validation_scales_up_when_base_budget_is_larger(self):
        config = type(
            "_Config",
            (),
            {
                "project": type("_Project", (), {"env": {}})(),
                "remote": type("_Remote", (), {"env_vars": {}})(),
                "task": type(
                    "_Task",
                    (),
                    {
                        "metadata": {
                            "target_object_pool": ["ycb_apple", "ycb_pear"],
                        },
                        "target_objects": ["ycb_apple", "ycb_pear"],
                        "reward": None,
                        "success_predicate": None,
                        "goal_region": {},
                        "goal_relation": None,
                        "dense_reward_terms": {},
                    },
                )(),
                "training": type("_Training", (), {"rl": type("_RL", (), {"args": {}})()})(),
            },
        )()
        args = type(
            "_Args",
            (),
            {
                "episodes_per_instruction": 130,
                "move_to_object_episodes_per_target": 50,
                "success_distance": 0.05,
                "directional_displacement_threshold": 0.05,
            },
        )()

        buckets = _validation_buckets(config, args, instruction_type="move_to_object")

        self.assertEqual(len(buckets), 2)
        self.assertTrue(all(bucket.episodes == 65 for bucket in buckets))


if __name__ == "__main__":
    unittest.main()
