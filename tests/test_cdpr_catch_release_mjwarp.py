from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "examples"
    / "cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml"
)
CHECKPOINT = (
    "/root/repo/RL_VLA_Bootstrapping/runs/"
    "cdpr_smolvla_move_to_scratch_mjwarp_w512_20260719_081705/"
    "rl/step_5000081/smolvla_grpo_adapter.pt"
)


class CDPRCatchReleaseConfigTests(unittest.TestCase):
    def test_resume_profile_uses_random_workspace_and_15m_additional_steps(self):
        config = load_project_config(CONFIG)
        plan = BootstrapPipeline(config).build_stage_plans(
            ROOT / "runs" / "catch_release_unit", ["rl"]
        )[0]
        command = plan.command
        self.assertEqual(
            config.task.instruction_types,
            ("put_into_plate", "put_into_bowl", "pick_up"),
        )
        self.assertTrue(
            config.task.metadata["random_workspace_gripper_start"]
        )
        self.assertTrue(
            config.task.metadata["placement_start_with_caught_object"]
        )
        self.assertEqual(config.simulator.worlds_per_rank, 512)
        self.assertEqual(config.simulator.groups_per_rank, 64)
        self.assertEqual(
            config.task.metadata["ee_workspace_x_bounds"], [-0.28, 0.28]
        )
        self.assertEqual(
            config.task.metadata["random_workspace_min_goal_xy_distance"],
            0.10,
        )
        self.assertEqual(
            command[command.index("--resume-checkpoint") + 1], CHECKPOINT
        )
        self.assertEqual(
            command[command.index("--max-train-steps") + 1], "20000081"
        )
        self.assertEqual(
            command[command.index("--complex-training-approach") + 1],
            "none",
        )
        self.assertEqual(
            config.training.rl.algorithm, "smolvla_residual_grpo_mjwarp"
        )
        self.assertNotIn(
            "reverse_frontier_profile", config.task.metadata
        )

    def test_launcher_defaults_to_exact_checkpoint_and_15m_continuation(self):
        source = (
            ROOT
            / "scripts"
            / "train_cdpr_smolvla_catch_release_grpo_mjlab_dual_remote.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(f'CHECKPOINT="${{CHECKPOINT:-{CHECKPOINT}}}"', source)
        self.assertIn(
            'ADDITIONAL_TRAIN_STEPS="${ADDITIONAL_TRAIN_STEPS:-15000000}"',
            source,
        )
        self.assertIn(
            'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-$((START_STEP + ADDITIONAL_TRAIN_STEPS))}"',
            source,
        )
        self.assertIn('WORLDS_PER_RANK="${WORLDS_PER_RANK:-512}"', source)

    def test_two_stage_launcher_restarts_grpo_with_discovered_checkpoint(self):
        source = (
            ROOT
            / "scripts"
            / "train_cdpr_smolvla_move_to_then_catch_release_grpo_mjlab_dual_remote.sh"
        ).read_text(encoding="utf-8")
        phase_1 = source.index('bash "$MOVE_TO_LAUNCHER"')
        discovery = source.index(
            'MOVE_TO_CHECKPOINT="$(latest_move_to_checkpoint)"'
        )
        phase_2 = source.rindex('bash "$CATCH_RELEASE_LAUNCHER"')
        self.assertLess(phase_1, discovery)
        self.assertLess(discovery, phase_2)
        self.assertIn(
            'START_STEP="$(checkpoint_step "$MOVE_TO_CHECKPOINT")"',
            source,
        )
        self.assertIn(
            'MAX_TRAIN_STEPS="$((START_STEP + CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS))"',
            source,
        )
        self.assertIn(
            'CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS="${CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS:-15000000}"',
            source,
        )
        self.assertIn(
            'curriculum=none lchol=disabled checkpoint_handoff=full_grpo_state',
            source,
        )
        self.assertIn(
            'assert_plain_grpo_config "$MOVE_TO_CONFIG"', source
        )
        self.assertIn(
            'assert_plain_grpo_config "$CATCH_RELEASE_CONFIG"', source
        )

    def test_both_pipeline_configs_are_plain_grpo_without_shells_or_lchol(self):
        move_to = load_project_config(
            ROOT
            / "configs"
            / "examples"
            / "cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml"
        )
        catch_release = load_project_config(CONFIG)
        for config in (move_to, catch_release):
            with self.subTest(config=config.project.name):
                self.assertEqual(
                    config.training.rl.algorithm,
                    "smolvla_residual_grpo_mjwarp",
                )
                self.assertEqual(
                    config.training.rl.args["complex_training_approach"],
                    "none",
                )
                self.assertNotIn(
                    "reverse_frontier_profile", config.task.metadata
                )


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch is required for tensorized catch/release rewards",
)
class CDPRCatchReleaseRewardTests(unittest.TestCase):
    @staticmethod
    def _fake_backend(torch):
        class FakeBackend:
            def __init__(self) -> None:
                self.torch = torch
                self.device = torch.device("cpu")
                self.object_body_ids = torch.arange(4, dtype=torch.int64)
                self.object_positions = None
                self.object_quaternions = None
                self.ee_positions = None
                self.ee_yaw = None

            def reset_worlds(self, _worlds):
                return None

            def set_object_catalogs(self, _catalogs):
                return None

            def set_free_body_poses(
                self, _body_ids, positions, quaternions
            ):
                self.object_positions = positions.clone()
                self.object_quaternions = quaternions.clone()

            def set_end_effector_poses(self, positions, yaw):
                self.ee_positions = positions.clone()
                self.ee_yaw = yaw.clone()

            def set_gripper_openings(self, _openings):
                return None

            def set_visual_variants(self, *_args):
                return None

            def broadcast_group_state(self, _base_worlds):
                return None

            def low_dim_observations(self):
                ee_quaternion = torch.zeros(
                    (self.ee_positions.shape[0], 4), dtype=torch.float32
                )
                ee_quaternion[:, 0] = torch.cos(0.5 * self.ee_yaw)
                ee_quaternion[:, 3] = torch.sin(0.5 * self.ee_yaw)
                return SimpleNamespace(
                    ee_position=self.ee_positions,
                    ee_quaternion=ee_quaternion,
                    object_positions=self.object_positions,
                    object_quaternions=self.object_quaternions,
                )

        return FakeBackend()

    def test_random_workspace_resets_keep_placement_caught_and_pick_on_desk(
        self,
    ):
        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
            RankLocalCurriculum,
        )
        from rl_vla_bootstrapping.policy.rank_local_grpo import (
            RankLocalGroupLayout,
        )

        layout = RankLocalGroupLayout(
            worlds_per_rank=8, groups_per_rank=1, group_size=8
        )
        metadata = {
            "random_workspace_gripper_start": True,
            "placement_start_with_caught_object": True,
            "random_workspace_min_goal_xy_distance": 0.10,
            "random_workspace_horizon_low": 21,
            "random_workspace_horizon_high": 32,
            "ee_workspace_x_bounds": [-0.28, 0.28],
            "ee_workspace_y_bounds": [-0.28, 0.28],
            "ee_workspace_z_bounds": [0.30, 0.48],
        }
        objects = (
            "robocasa_apple",
            "robocasa_banana",
            "robocasa_plate",
            "robocasa_bowl",
        )
        for instruction in (
            "put_into_plate",
            "put_into_bowl",
            "pick_up",
        ):
            with self.subTest(instruction=instruction):
                backend = self._fake_backend(torch)
                resetter = BatchedReverseFrontierResetter(
                    backend=backend,
                    layout=layout,
                    curriculum=RankLocalCurriculum(device=backend.device),
                    rank=0,
                    base_seed=17,
                    instruction_types=(instruction,),
                    allowed_objects=objects,
                    task_metadata=metadata,
                )
                reset = resetter.reset(update_index=0, round_index=0)
                ee = backend.ee_positions
                target = backend.object_positions[:, 0]
                if instruction == "pick_up":
                    self.assertFalse(
                        bool(reset.task_state.ever_grasped.any().item())
                    )
                    self.assertTrue(
                        torch.allclose(
                            target[:, 2],
                            reset.task_state.support_surface_z
                            + reset.task_state.target_rest_height,
                        )
                    )
                    self.assertTrue(
                        reset.instructions[0].startswith("pick up ")
                    )
                    goal = target
                else:
                    self.assertTrue(
                        bool(reset.task_state.ever_grasped.all().item())
                    )
                    self.assertTrue(
                        torch.allclose(
                            target[:, 2], ee[:, 2] - 0.08, atol=1.0e-6
                        )
                    )
                    reference = backend.object_positions[:, 1]
                    goal = reference
                xy_distance = torch.linalg.vector_norm(
                    ee[:, :2] - goal[:, :2], dim=-1
                )
                self.assertTrue(bool((xy_distance >= 0.10).all().item()))
                self.assertTrue(bool((reset.horizons >= 21).all().item()))

    def test_release_radius_wrong_drop_and_grasp_lift_rewards(self):
        import torch

        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
            BatchedCatchReleaseDenseReward,
            BatchedTaskState,
            evaluate_active_sparse_tasks,
        )

        objects = torch.zeros((4, 4, 3), dtype=torch.float32)
        # Correct plate release.
        objects[0, 0] = torch.tensor([0.08, 0.00, 0.18])
        objects[0, 1] = torch.tensor([0.00, 0.00, 0.16])
        # Wrong bowl drop, outside the 0.057 m radius and settled on support.
        objects[1, 0] = torch.tensor([0.10, 0.00, 0.18])
        objects[1, 1] = torch.tensor([0.00, 0.00, 0.176])
        # Grasped but not lifted.
        objects[2, 0] = torch.tensor([0.00, 0.00, 0.18])
        # Grasped and lifted 0.06 m.
        objects[3, 0] = torch.tensor([0.00, 0.00, 0.24])

        initial = objects[:, 0].clone()
        initial[0] = torch.tensor([0.20, 0.00, 0.32])
        initial[1] = torch.tensor([0.20, 0.00, 0.32])
        initial[3, 2] = 0.18
        instruction_ids = torch.tensor(
            [
                INSTRUCTION_TO_ID["put_into_plate"],
                INSTRUCTION_TO_ID["put_into_bowl"],
                INSTRUCTION_TO_ID["pick_up"],
                INSTRUCTION_TO_ID["pick_up"],
            ],
            dtype=torch.int64,
        )
        state = BatchedTaskState(
            instruction_ids=instruction_ids,
            target_slots=torch.zeros((4,), dtype=torch.int64),
            reference_slots=torch.tensor([1, 1, -1, -1]),
            second_reference_slots=torch.full((4,), -1, dtype=torch.int64),
            initial_target_positions=initial,
            ever_grasped=torch.tensor([True, True, False, False]),
            grasped=torch.tensor([False, False, False, False]),
            step_count=torch.zeros((4,), dtype=torch.int64),
            release_threshold=torch.full((4,), 0.55),
            support_surface_z=torch.full((4,), 0.15),
            target_rest_height=torch.full((4,), 0.03),
        )
        ee = torch.tensor(
            [
                [0.08, 0.00, 0.26],
                [0.10, 0.00, 0.26],
                [0.00, 0.00, 0.26],
                [0.00, 0.00, 0.32],
            ],
            dtype=torch.float32,
        )
        result = evaluate_active_sparse_tasks(
            state=state,
            ee_position=ee,
            object_positions=objects,
            gripper_opening=torch.tensor([0.90, 0.90, 0.50, 0.50]),
            caught_target=torch.tensor([False, False, True, True]),
            active_mask=torch.ones((4,), dtype=torch.bool),
            max_steps=128,
            catch_release_dense_reward=BatchedCatchReleaseDenseReward(),
        )

        self.assertEqual(
            result.success.tolist(), [True, False, False, True]
        )
        self.assertEqual(
            result.terminated.tolist(), [True, True, False, True]
        )
        self.assertTrue(
            bool(result.diagnostics["wrong_place_drop"][1].item())
        )
        self.assertAlmostEqual(float(result.rewards[0].item()), 3.0, places=5)
        self.assertAlmostEqual(float(result.rewards[2].item()), 2.0, places=5)
        self.assertAlmostEqual(float(result.rewards[3].item()), 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
