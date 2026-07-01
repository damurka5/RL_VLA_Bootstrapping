from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline
from rl_vla_bootstrapping.policy.octo_cdpr_adapter import (
    CDPROctoObservationAdapter,
    CDPRStateLayout,
    OctoActionAdapterSpec,
    OctoObservationSpec,
    _prepare_octo_import_path,
    adapt_octo_actions_to_cdpr,
    flatten_cdpr_observation,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fake_cdpr_obs() -> dict[str, np.ndarray]:
    return {
        "ee_position": np.array([0.1, 0.2, 0.4], dtype=np.float32),
        "target_object_position": np.array([0.0, 0.0, 0.2], dtype=np.float32),
        "all_object_positions": np.zeros((2, 3), dtype=np.float32),
        "object_position_mask": np.array([1.0, 0.0], dtype=np.float32),
        "instruction_onehot": np.eye(3, dtype=np.float32)[1],
        "goal_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


class OctoCDPRAdapterTests(unittest.TestCase):
    def test_observation_adapter_builds_octo_history_batch(self):
        adapter = CDPROctoObservationAdapter(OctoObservationSpec(image_size=8, history=1))
        primary = np.zeros((8, 8, 3), dtype=np.uint8)
        wrist = np.ones((8, 8, 3), dtype=np.uint8) * 255

        out = adapter.from_images(
            primary_image=primary,
            wrist_image=wrist,
            proprio=np.arange(5, dtype=np.float32),
        )

        self.assertEqual(out["image_primary"].shape, (1, 1, 8, 8, 3))
        self.assertEqual(out["image_wrist"].shape, (1, 1, 8, 8, 3))
        self.assertEqual(out["proprio"].shape, (1, 1, 5))
        self.assertTrue(out["timestep_pad_mask"].all())
        self.assertTrue(out["pad_mask_dict"]["image_primary"].all())

    def test_observation_adapter_matches_checkpoint_example_schema(self):
        example_observation = {
            "image_primary": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
            "image_wrist": np.zeros((1, 2, 128, 128, 3), dtype=np.uint8),
            "timestep": np.zeros((1, 2), dtype=np.int32),
            "task_completed": np.zeros((1, 2), dtype=bool),
            "timestep_pad_mask": np.zeros((1, 2), dtype=bool),
            "pad_mask_dict": {
                "image_primary": np.zeros((1, 1), dtype=bool),
                "image_wrist": np.zeros((1, 2), dtype=bool),
                "timestep": np.zeros((1, 2), dtype=bool),
            },
        }
        adapter = CDPROctoObservationAdapter(
            OctoObservationSpec(image_size=8, history=1, include_proprio=True),
            example_observation=example_observation,
        )
        out = adapter.from_images(
            primary_image=np.zeros((64, 64, 3), dtype=np.uint8),
            wrist_image=np.ones((64, 64, 3), dtype=np.uint8),
            proprio=np.ones(5, dtype=np.float32),
        )

        self.assertEqual(set(out), set(example_observation))
        self.assertNotIn("proprio", out)
        for key, value in example_observation.items():
            if key == "pad_mask_dict":
                continue
            self.assertEqual(out[key].shape, value.shape)
        for key, value in example_observation["pad_mask_dict"].items():
            self.assertEqual(out["pad_mask_dict"][key].shape, value.shape)
            self.assertTrue(out["pad_mask_dict"][key].all())
        self.assertEqual(out["image_wrist"].shape, (1, 2, 128, 128, 3))
        self.assertTrue(out["timestep_pad_mask"].all())

    def test_action_adapter_maps_7d_octo_actions_to_5d_cdpr_chunks(self):
        raw = np.array(
            [
                [0.1, 0.2, 0.3, 9.0, 8.0, 0.4, -0.5],
                [0.6, 0.7, 0.8, 7.0, 6.0, 0.9, -1.0],
            ],
            dtype=np.float32,
        )

        chunk = adapt_octo_actions_to_cdpr(
            raw,
            spec=OctoActionAdapterSpec(chunk_size=4, normalization="clip"),
        )

        self.assertEqual(chunk.shape, (4, 5))
        np.testing.assert_allclose(chunk[0], [0.1, 0.2, 0.3, 0.4, -0.5])
        np.testing.assert_allclose(chunk[1], [0.6, 0.7, 0.8, 0.9, -1.0])
        np.testing.assert_allclose(chunk[2], chunk[1])
        np.testing.assert_allclose(chunk[3], chunk[1])

    def test_cdpr_state_layout_flatten(self):
        obs = _fake_cdpr_obs()
        layout = CDPRStateLayout.from_observation(obs)
        flat = flatten_cdpr_observation(obs)

        self.assertEqual(layout.state_dim, flat.size)
        np.testing.assert_allclose(layout.flatten(obs), flat)
        np.testing.assert_allclose(flat[layout.ee_slice], obs["ee_position"])

    def test_octo_pipeline_plan_uses_lazy_runtime_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robot_root = root / "robot"
            scripts_root = root / "rl_vla_bootstrapping" / "policy"
            dataset_root = root / "dataset"
            config_dir = root / "configs" / "examples"

            _write(robot_root / "robot.xml", "<mujoco/>")
            _write(robot_root / "controller.py", "class DummyController: pass\n")
            _write(scripts_root / "octo_finetune_cdpr.py", "print('octo')\n")
            _write(dataset_root / "catalog.yaml", "defaults: {}\nscenes: []\n")
            _write(dataset_root / "reward.py", "def reward_fn(**kwargs): return 0.0, False, {}\n")

            config = {
                "project": {"name": "octo_unit", "output_root": "../../runs"},
                "repos": {"dataset_repo": "../../dataset", "embodiment_repo": "../../robot"},
                "embodiment": {
                    "name": "dummy_cdpr",
                    "kind": "mujoco",
                    "robot_root": "../../robot",
                    "xml_path": "../../robot/robot.xml",
                    "dof": 5,
                    "controller": {"file": "../../robot/controller.py", "class_name": "DummyController"},
                    "action_adapter": {
                        "common_action_keys": ["x", "y", "z", "yaw", "gripper"],
                        "controller_scales": {"x": 0.01, "y": 0.01, "z": 0.01, "yaw": 0.1, "gripper": 0.05},
                        "controller_limits": {
                            "x": [-1, 1],
                            "y": [-1, 1],
                            "z": [0, 1],
                            "yaw": [-3.14, 3.14],
                            "gripper": [0, 1],
                        },
                    },
                },
                "task": {
                    "instruction_types": ["move_left", "move_to_object"],
                    "target_objects": ["ycb_apple"],
                    "reward": {"file": "../../dataset/reward.py", "function": "reward_fn"},
                    "metadata": {"scene_object_pool": ["ycb_apple"]},
                },
                "simulation": {"catalog_path": "../../dataset/catalog.yaml"},
                "policy": {
                    "type": "octo_small_cdpr",
                    "base_checkpoint": "hf://rail-berkeley/octo-small-1.5",
                    "rl_script": "../../rl_vla_bootstrapping/policy/octo_finetune_cdpr.py",
                    "action_codec": {"chunk_size": 4, "quantization": {"enabled": False}},
                },
                "training": {
                    "preview_before_rl": False,
                    "rl": {"enabled": True, "args": {"max_train_steps": 10}},
                },
            }
            config_path = config_dir / "octo.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            project_config = load_project_config(config_path)
            pipeline = BootstrapPipeline(project_config)
            run_dir = root / "runs" / "unit"
            plans = pipeline.build_stage_plans(run_dir, ["rl"])

            self.assertEqual([plan.name for plan in plans], ["rl"])
            command = plans[0].command or []
            self.assertIn("-m", command)
            self.assertIn("rl_vla_bootstrapping.policy.octo_finetune_cdpr", command)
            self.assertIn("--base-checkpoint", command)
            self.assertIn("hf://rail-berkeley/octo-small-1.5", command)
            self.assertIn("--instruction-types", command)
            self.assertIn("move_to_object", command)
            self.assertIn("RLVLA_TASK_METADATA_JSON", plans[0].env)

    def test_prepare_octo_import_path_prioritizes_repo_over_local_shadow(self):
        old_path = list(sys.path)
        old_env = os.environ.get("OCTO_REPO_PATH")
        old_modules = {name: module for name, module in sys.modules.items() if name == "octo" or name.startswith("octo.")}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                repo = root / "berkeley_octo"
                shadow = root / "shadow"
                _write(repo / "octo" / "__init__.py", "")
                _write(repo / "octo" / "model" / "__init__.py", "")
                _write(repo / "octo" / "model" / "octo_model.py", "class OctoModel: pass\n")
                _write(shadow / "octo.py", "MARKER = 'shadow'\n")

                for name in list(sys.modules):
                    if name == "octo" or name.startswith("octo."):
                        sys.modules.pop(name, None)
                sys.path.insert(0, shadow.as_posix())
                sys.path.append(repo.as_posix())
                importlib.import_module("octo")
                self.assertEqual(Path(sys.modules["octo"].__file__).resolve(), (shadow / "octo.py").resolve())

                os.environ["OCTO_REPO_PATH"] = repo.as_posix()
                _prepare_octo_import_path()
                module = importlib.import_module("octo.model.octo_model")

                self.assertEqual(Path(module.__file__).resolve(), (repo / "octo" / "model" / "octo_model.py").resolve())
                self.assertEqual(sys.path[0], repo.resolve().as_posix())
        finally:
            sys.path[:] = old_path
            if old_env is None:
                os.environ.pop("OCTO_REPO_PATH", None)
            else:
                os.environ["OCTO_REPO_PATH"] = old_env
            for name in list(sys.modules):
                if name == "octo" or name.startswith("octo."):
                    sys.modules.pop(name, None)
            sys.modules.update(old_modules)


if __name__ == "__main__":
    unittest.main()
