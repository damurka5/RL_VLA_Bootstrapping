from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

_INSERTED_TORCH_STUB = False

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *args, **kwargs: None,
    )
    torch_stub.manual_seed = lambda *args, **kwargs: None
    torch_stub.device = lambda name: name
    torch_stub.as_tensor = lambda *args, **kwargs: None
    torch_stub.randn_like = lambda tensor: tensor
    torch_stub.cat = lambda tensors, dim=-1: None
    torch_stub.min = lambda *args, **kwargs: None
    torch_stub.save = lambda *args, **kwargs: None
    torch_stub.distributions = types.SimpleNamespace(Normal=object)

    nn_stub = types.ModuleType("torch.nn")

    class _Module:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

    class _Linear(_Module):
        pass

    class _Sequential(_Module):
        pass

    class _ModuleList(list):
        pass

    nn_stub.Module = _Module
    nn_stub.Linear = _Linear
    nn_stub.Sequential = _Sequential
    nn_stub.ModuleList = _ModuleList

    functional_stub = types.ModuleType("torch.nn.functional")
    functional_stub.softplus = lambda x: x
    functional_stub.relu = lambda x: x
    functional_stub.mse_loss = lambda a, b: 0.0

    torch_stub.nn = nn_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.nn.functional"] = functional_stub
    _INSERTED_TORCH_STUB = True

from rl_vla_bootstrapping.policy.blac_finetune_cdpr import (
    BarrierConfig,
    ObservationLayout,
    _backup_project_action,
    _goal_distance_sq_np,
    _make_run_dir,
    parse_args,
)

if _INSERTED_TORCH_STUB:
    sys.modules.pop("torch.nn.functional", None)
    sys.modules.pop("torch.nn", None)
    sys.modules.pop("torch", None)


class BLACFinetuneCDPRTests(unittest.TestCase):
    def test_parse_args_accepts_stage_compatibility_flags(self):
        args, unknown = parse_args(
            [
                "--algorithm",
                "redq",
                "--allowed_objects",
                "ycb_apple",
                "plate",
                "--instruction_types",
                "lift",
                "move_left",
                "--no-wrapper_cleanup",
                "--external_ppo_script",
                "/tmp/external.py",
            ]
        )

        self.assertEqual(args.algorithm, "redq")
        self.assertEqual(args.allowed_objects, ["ycb_apple", "plate"])
        self.assertEqual(args.instruction_types, ["lift", "move_left"])
        self.assertFalse(args.wrapper_cleanup)
        self.assertEqual(unknown, ["--external_ppo_script", "/tmp/external.py"])

    def test_layout_and_goal_distance_follow_cdpr_observation_schema(self):
        obs = {
            "ee_position": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "target_object_position": np.array([0.4, 0.2, 0.1], dtype=np.float32),
            "all_object_positions": np.zeros((2, 3), dtype=np.float32),
            "object_position_mask": np.array([1.0, 0.0], dtype=np.float32),
            "instruction_onehot": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "goal_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

        layout = ObservationLayout.from_observation(obs)
        flat = layout.flatten(obs)

        self.assertEqual(layout.state_dim, flat.size)
        self.assertEqual(layout.max_objects, 2)
        self.assertAlmostEqual(_goal_distance_sq_np(flat, layout), 0.13, places=6)

    def test_backup_projection_clips_xyz_motion_inside_workspace(self):
        obs = {
            "ee_position": np.array([0.89, 0.0, 0.10], dtype=np.float32),
            "target_object_position": np.array([0.5, 0.1, 0.2], dtype=np.float32),
            "all_object_positions": np.zeros((1, 3), dtype=np.float32),
            "object_position_mask": np.array([0.0], dtype=np.float32),
            "instruction_onehot": np.array([1.0], dtype=np.float32),
            "goal_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
        layout = ObservationLayout.from_observation(obs)
        state = layout.flatten(obs)
        config = BarrierConfig(
            x_low=-0.90,
            x_high=0.90,
            y_low=-0.90,
            y_high=0.90,
            z_low=0.05,
            z_high=0.60,
            eta=0.25,
            beta=0.10,
        )

        safe_action = _backup_project_action(
            state,
            np.array([1.0, 0.0, 0.0, 0.4, -0.8], dtype=np.float32),
            layout,
            config,
            action_step_xyz=0.05,
            goal_gain=0.0,
        )

        next_x = float(state[layout.ee_slice][0] + safe_action[0] * 0.05)
        self.assertLessEqual(next_x, 0.90 + 1e-6)
        self.assertAlmostEqual(float(safe_action[3]), 0.4, places=6)
        self.assertAlmostEqual(float(safe_action[4]), -0.8, places=6)

    def test_make_run_dir_creates_algorithm_specific_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp, "unit", "blac")
            self.assertTrue(run_dir.is_dir())
            self.assertIn("unit_blac_", run_dir.name)
            self.assertTrue(Path(tmp).resolve() in run_dir.parents)


if __name__ == "__main__":
    unittest.main()
