from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rl_vla_bootstrapping.cli.validate_cdpr_policy import (
    _default_max_steps,
    _resolve_policy_artifacts,
    _summarize_instruction_results,
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

    def test_validation_env_vars_override_success_distance(self):
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
        args = type("_Args", (), {"success_distance": 0.05})()

        env = _validation_env_vars(config, args)

        self.assertIn("RLVLA_TASK_METADATA_JSON", env)
        self.assertIn('"success_distance": 0.05', env["RLVLA_TASK_METADATA_JSON"])


if __name__ == "__main__":
    unittest.main()
