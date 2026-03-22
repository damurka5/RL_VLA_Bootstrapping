from __future__ import annotations

import unittest

from rl_vla_bootstrapping.policy.openvla_blac_finetune_cdpr import (
    BarrierConfig,
    ObservationLayout,
    _make_run_dir,
    _safety_cost_np,
    parse_args,
)


class OpenVLABLACFinetuneTests(unittest.TestCase):
    def test_parse_args_keeps_unknown_stage_flags_out_of_namespace(self):
        args, unknown = parse_args(
            [
                "--config",
                "cfg.yaml",
                "--adapter-path",
                "/tmp/adapter",
                "--action-head-path",
                "/tmp/head.pt",
                "--no-freeze_vla_backbone",
                "--external_ppo_script",
                "ignored.py",
            ]
        )

        self.assertEqual(args.config, "cfg.yaml")
        self.assertFalse(args.freeze_vla_backbone)
        self.assertEqual(unknown, ["--external_ppo_script", "ignored.py"])

    def test_safety_cost_uses_workspace_bounds(self):
        obs = {
            "ee_position": [-1.1, 0.0, 0.01],
            "target_object_position": [0.0, 0.0, 0.1],
            "all_object_positions": [[0.0, 0.0, 0.0]],
            "object_position_mask": [0.0],
            "instruction_onehot": [1.0],
            "goal_direction": [1.0, 0.0, 0.0],
        }
        layout = ObservationLayout.from_observation(obs)
        state = layout.flatten(obs)
        cfg = BarrierConfig(
            x_low=-0.9,
            x_high=0.9,
            y_low=-0.9,
            y_high=0.9,
            z_low=0.05,
            z_high=0.60,
            eta=0.25,
            beta=0.1,
        )

        self.assertGreater(_safety_cost_np(state, layout, cfg), 0.0)

    def test_make_run_dir_builds_timestamped_folder(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp, "openvla_blac")
            self.assertTrue(run_dir.is_dir())
            self.assertIn("openvla_blac_", run_dir.name)


if __name__ == "__main__":
    unittest.main()
