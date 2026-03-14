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
                "simulation": {
                    "catalog_path": "dataset/catalog.yaml",
                    "desk_textures_dir": "dataset/textures",
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
                    "rl": {"enabled": True, "args": {"total_updates": 1}},
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
            self.assertIn("--total-updates", plans[1].command)

            preview_only = pipeline.build_stage_plans(run_dir, ["preview"])
            self.assertEqual([plan.name for plan in preview_only], ["preview"])


if __name__ == "__main__":
    unittest.main()
