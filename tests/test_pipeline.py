from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_vla_bootstrapping.assets import stage_asset_bundles
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class PipelineTests(unittest.TestCase):
    def test_action_codec_manifest_and_stage_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robot_root = root / "robot"
            repo_root = root / "policy_repo"
            dataset_root = root / "dataset"

            _write(robot_root / "robot.xml", "<mujoco/>")
            _write(
                robot_root / "controller.py",
                "class DummyController:\n"
                "    def __init__(self, xml_path=None, output_dir=None):\n"
                "        self.xml_path = xml_path\n"
                "        self.output_dir = output_dir\n"
                "        self.overview_frames = []\n"
                "        self.ee_camera_frames = []\n"
                "    def initialize(self):\n"
                "        pass\n"
                "    def run_simulation_step(self, capture_frame=True):\n"
                "        self.overview_frames.append([[[0, 0, 0]]])\n"
                "        self.ee_camera_frames.append([[[255, 255, 255]]])\n"
                "    def cleanup(self):\n"
                "        pass\n",
            )
            _write(repo_root / "ppo.py", "print('ppo')\n")
            _write(repo_root / "finetune.py", "print('sft')\n")
            (dataset_root / "textures").mkdir(parents=True, exist_ok=True)
            _write(dataset_root / "catalog.yaml", "defaults: {}\nscenes: []\n")
            _write(dataset_root / "reward_hook.py", "def reward_fn(spec=None, **kwargs):\n    return 1.0, False, {'custom_reward': 1.0}\n")
            _write(dataset_root / "success_hook.py", "def success_fn(current_success=False, **kwargs):\n    return current_success, {'checked_success': 1.0}\n")

            config = {
                "project": {"name": "test_project", "output_root": "runs"},
                "repos": {
                    "openvla_oft": "policy_repo",
                    "dataset_repo": "dataset",
                    "embodiment_repo": "robot",
                },
                "assets": {
                    "bundles": [
                        {
                            "name": "dummy_assets",
                            "source_path": "dataset/textures",
                            "target_path": "assets/externals/dummy_assets",
                            "required": True,
                        }
                    ]
                },
                "embodiment": {
                    "name": "dummy_bot",
                    "kind": "mujoco",
                    "robot_root": "robot",
                    "xml_path": "robot/robot.xml",
                    "dof": 5,
                    "controller": {
                        "file": "robot/controller.py",
                        "class_name": "DummyController",
                        "frame_buffers": {"overview": "overview_frames", "wrist": "ee_camera_frames"},
                    },
                    "action_adapter": {
                        "common_action_keys": ["x", "y", "z", "yaw", "gripper"],
                        "controller_limits": {
                            "x": [-1, 1],
                            "y": [-1, 1],
                            "z": [-1, 1],
                            "yaw": [-1, 1],
                            "gripper": [0, 1],
                        },
                    },
                },
                "task": {
                    "target_objects": ["ycb_apple"],
                    "reward": {"file": "dataset/reward_hook.py", "function": "reward_fn"},
                    "success_predicate": {
                        "file": "dataset/success_hook.py",
                        "function": "success_fn",
                    },
                    "goal_region": {"center": [0.0, 0.0, 0.2]},
                    "goal_relation": "inside_region",
                    "dense_reward_terms": {"reach": 1.0},
                    "metadata": {
                        "mode": "zero_demo",
                        "scene_object_pool": ["ycb_apple", "ycb_pear", "plate", "ycb_spoon"],
                    },
                },
                "simulation": {
                    "catalog_path": "dataset/catalog.yaml",
                    "desk_textures_dir": "dataset/missing_cache",
                    "desk_textures_fallback_dir": "dataset/textures",
                },
                "policy": {
                    "type": "openvla_oft",
                    "repo_path": "policy_repo",
                    "base_checkpoint": "openvla/openvla-7b",
                    "rl_script": "policy_repo/ppo.py",
                    "sft_script": "policy_repo/finetune.py",
                },
                "training": {
                    "preview_before_rl": True,
                    "rl": {
                        "enabled": True,
                        "args": {
                            "total_updates": 1,
                            "wrapper_cleanup": False,
                            "lock_non_commanded_axes": True,
                            "lock_non_commanded_axes_threshold": 0.05,
                            "randomize_ee_start": True,
                            "ee_start_x_bounds": [-0.12, 0.12],
                            "ee_start_y_bounds": [-0.08, 0.10],
                        },
                    },
                    "sft": {"enabled": True, "args": {"resume_from_rl": False}},
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            project_config = load_project_config(config_path)
            pipeline = BootstrapPipeline(project_config)

            self.assertEqual(pipeline.validate(), [])

            statuses = stage_asset_bundles(project_config, mode="symlink", force=True)
            self.assertTrue(statuses[0].exists)
            self.assertTrue((root / "assets" / "externals" / "dummy_assets").exists())

            run_dir = pipeline.make_run_dir("unit")
            manifests = pipeline.export_manifests(run_dir)
            self.assertTrue(manifests["action_codec"].exists())

            plans = pipeline.build_stage_plans(run_dir, ["all"])
            self.assertEqual([plan.name for plan in plans], ["preview", "rl", "sft"])
            self.assertIn("--total_updates", plans[1].command)
            self.assertIn("--no-wrapper_cleanup", plans[1].command)
            self.assertNotIn("--lock_non_commanded_axes", plans[1].command)
            self.assertNotIn("--lock_non_commanded_axes_threshold", plans[1].command)
            self.assertNotIn("--randomize_ee_start", plans[1].command)
            self.assertNotIn("--ee_start_x_bounds", plans[1].command)
            self.assertNotIn("--ee_start_y_bounds", plans[1].command)
            desk_textures_idx = plans[1].command.index("--desk_textures_dir") + 1
            self.assertTrue(Path(plans[1].command[desk_textures_idx]).samefile(dataset_root / "textures"))
            self.assertIn("--run_root_dir", plans[1].command)
            allowed_objects_idx = plans[1].command.index("--allowed_objects") + 1
            self.assertEqual(
                plans[1].command[allowed_objects_idx : allowed_objects_idx + 4],
                ["ycb_apple", "ycb_pear", "plate", "ycb_spoon"],
            )
            self.assertTrue(Path(plans[1].env["RLVLA_TASK_REWARD_FILE"]).samefile(dataset_root / "reward_hook.py"))
            self.assertEqual(plans[1].env["RLVLA_TASK_REWARD_ATTRIBUTE"], "reward_fn")
            self.assertTrue(Path(plans[1].env["RLVLA_TASK_SUCCESS_FILE"]).samefile(dataset_root / "success_hook.py"))
            self.assertEqual(plans[1].env["RLVLA_TASK_SUCCESS_ATTRIBUTE"], "success_fn")
            self.assertEqual(plans[1].env["RLVLA_TASK_GOAL_RELATION"], "inside_region")
            self.assertEqual(plans[1].env["RLVLA_CDPR_LOCK_NON_COMMANDED_AXES"], "1")
            self.assertEqual(plans[1].env["RLVLA_CDPR_LOCK_NON_COMMANDED_AXES_THRESHOLD"], "0.05")
            self.assertEqual(plans[1].env["RLVLA_CDPR_RANDOMIZE_EE_START"], "1")
            self.assertEqual(plans[1].env["RLVLA_CDPR_EE_START_X_BOUNDS"], "[-0.12, 0.12]")
            self.assertEqual(plans[1].env["RLVLA_CDPR_EE_START_Y_BOUNDS"], "[-0.08, 0.1]")
            self.assertIn("--run_root_dir", plans[2].command)

            preview_only = pipeline.build_stage_plans(run_dir, ["preview"])
            self.assertEqual([plan.name for plan in preview_only], ["preview"])


if __name__ == "__main__":
    unittest.main()
