from pathlib import Path
import unittest

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
    ApproachDistanceCurriculum,
    PerInstructionApproachCurriculum,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/examples/cdpr_smolvla_release_recovery_pilot.yaml"


class FixedApproachTests(unittest.TestCase):
    def test_single_rung_survives_success_failure_restart_and_resume(self):
        c = ApproachDistanceCurriculum({
            "random_workspace_start_distance_curriculum_enabled": True,
            "random_workspace_start_distance_ladder": [0.08],
            "random_workspace_start_distance_initial": 0.01,
            "random_workspace_start_distance_final": 0.34,
            "random_workspace_start_distance_pass_rate_ema_decay": 0.0,
            "random_workspace_start_distance_cooldown_updates": 1,
        })
        for saved_cap in (0.01, 0.34):
            c.load_state_dict({"cap": saved_cap})
            for rate in [1.0] * 20 + [0.0] * 20:
                c.observe(rate)
                self.assertEqual(c.current_cap(), 0.08)
            c.restart()
            self.assertEqual(c.current_cap(), 0.08)

    def test_fixed_and_adaptive_families_can_coexist(self):
        c = PerInstructionApproachCurriculum({
            "random_workspace_start_distance_curriculum_enabled": True,
            "random_workspace_start_distance_ladder": [0.03, 0.20],
            "random_workspace_start_distance_ladder_by_instruction": {
                "pick_up": [0.06], "move_to_object": [0.08, 0.14],
            },
            "random_workspace_start_distance_pass_rate_ema_decay": 0.0,
            "random_workspace_start_distance_cooldown_updates": 1,
        }, instruction_types=("pick_up", "move_to_object", "put_into_plate"))
        for _ in range(20):
            c.observe({name: 1.0 for name in c.instruction_names})
        caps = c.caps_by_instruction_id()
        self.assertEqual(caps[INSTRUCTION_TO_ID["pick_up"]], 0.06)
        self.assertEqual(caps[INSTRUCTION_TO_ID["move_to_object"]], 0.14)
        self.assertEqual(caps[INSTRUCTION_TO_ID["put_into_plate"]], 0.20)

    def test_disabled_still_means_uncapped(self):
        c = ApproachDistanceCurriculum({
            "random_workspace_start_distance_curriculum_enabled": False,
            "random_workspace_start_distance_ladder": [0.08],
        })
        self.assertEqual(c.current_cap(), float("inf"))

    def test_empty_ladder_keeps_incremental_curriculum(self):
        c = ApproachDistanceCurriculum({
            "random_workspace_start_distance_curriculum_enabled": True,
            "random_workspace_start_distance_ladder": [],
            "random_workspace_start_distance_initial": 0.03,
            "random_workspace_start_distance_final": 0.05,
            "random_workspace_start_distance_increment": 0.02,
            "random_workspace_start_distance_pass_rate_ema_decay": 0.0,
        })
        c.observe(1.0)
        self.assertEqual(c.current_cap(), 0.05)

    def test_pilot_caps_override_old_checkpoint_without_changing_horizon_scale(self):
        cfg = load_project_config(CONFIG)
        c = PerInstructionApproachCurriculum(
            cfg.task.metadata, instruction_types=cfg.task.instruction_types,
        )
        c.load_state_dict({name: {"cap": 0.13} for name in c.instruction_names})
        expected = {"move_to_object": 0.08, "pick_up": 0.06,
                    "put_into_plate": 0.20, "put_into_bowl": 0.20}
        for _ in range(30):
            c.observe({name: 1.0 for name in c.instruction_names})
        self.assertEqual(c.caps_by_instruction_id(),
                         {INSTRUCTION_TO_ID[k]: v for k, v in expected.items()})
        self.assertEqual(cfg.task.metadata["random_workspace_start_distance_final"], 0.34)

    def test_pilot_keeps_success_geometry_and_has_no_assisted_starts(self):
        base = load_project_config(ROOT / "configs/examples/cdpr_smolvla_phase7_sparse_joint.yaml")
        cfg = load_project_config(CONFIG)
        for key in ("put_plate_xy_tolerance", "put_bowl_xy_tolerance",
                    "put_container_z_tolerance", "placement_wrong_drop_settle_margin",
                    "pick_lift_success_height", "put_require_release",
                    "put_require_target_grasp_history", "placement_grasp_horizon_min_decisions"):
            self.assertEqual(cfg.task.metadata[key], base.task.metadata[key], key)
        self.assertTrue(cfg.task.metadata["sparse_binary_reward"])
        self.assertEqual(cfg.task.metadata["placement_caught_object_fraction"], 0)
        self.assertFalse(cfg.task.metadata["placement_caught_curriculum_enabled"])
        self.assertEqual(cfg.task.metadata["pick_up_prelifted_group_fraction"], 0)
        self.assertEqual(cfg.task.metadata["pick_up_aligned_group_fraction"], 0)
        self.assertFalse(cfg.training.sft.enabled)

    def test_pilot_arguments_reach_the_training_parser(self):
        from rl_vla_bootstrapping.core.commands import append_cli_arg
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args

        cfg = load_project_config(CONFIG)
        argv = []
        for key, value in cfg.training.rl.args.items():
            append_cli_arg(argv, key, value)
        args = parse_args(argv)
        self.assertEqual(args.max_train_steps, 500000)
        self.assertEqual(args.validation_every_steps, 100000)
        self.assertEqual(args.episode_offset_std, [0.0, 0.0, 0.0, 0.0, 0.15])
        self.assertFalse(args.episode_offset_after_grasp)
        self.assertFalse(args.split_credit_at_grasp)


if __name__ == "__main__":
    unittest.main()
