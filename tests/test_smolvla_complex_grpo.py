from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.lchol.smolvla_complex import SmolVLAComplexRuntime
from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline
from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import DistributedContext, EnvSlot
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
    SmolVLAGRPOTrainer,
    _dynamic_frontier_retry_options,
    _evaluate_trajectory_group,
    _resolve_checkpoint,
    _trajectory_group_is_informative,
    _trajectory_group_metric_scalars,
    parse_args,
    torch,
)
from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import (
    SMOLVLA_COMPLEX_PROFILE,
    apply_cdpr_reverse_shell,
    get_cdpr_reverse_shell_specs,
)
from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
    InstructionSpec,
    compute_instruction_reward,
    init_reward_state,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    "/root/repo/RL_VLA_Bootstrapping/runs/"
    "cdpr_smolvla_stage3_object_dense_complex_resume_step_5000000_to_10000000_20260710_193100/"
    "rl/step_6700000"
)


def _move_spec() -> InstructionSpec:
    return InstructionSpec(
        instruction_type="move_to_object",
        text="move to apple",
        target_object="ycb_apple",
        direction=np.zeros((3,), dtype=np.float32),
        target_displacement=0.0,
        lift_target=0.0,
    )


class _MoveShellEnv:
    def __init__(self) -> None:
        self._instruction_spec = _move_spec()
        self._target_body_name = "apple_body"
        self._task_metadata = {"reverse_frontier_profile": SMOLVLA_COMPLEX_PROFILE}
        self._bodies = {"apple_body": np.array([0.0, 0.0, 0.03], dtype=np.float32)}
        self._ee = np.array([0.0, 0.0, 0.20], dtype=np.float32)
        self.action_step_xyz = 0.015
        self.hold_steps = 6

    def _get_body_position(self, body_name):
        return self._bodies[str(body_name)].copy()

    def _get_ee_position(self):
        return self._ee.copy()

    def _set_ee_target(self, xyz):
        self._ee = np.asarray(xyz, dtype=np.float32).reshape(3).copy()


class _TrajectoryTestEnv:
    def __init__(self) -> None:
        self.step_count = 0

    def capture_state(self):
        return {"step_count": int(self.step_count)}

    def restore_state(self, snapshot):
        self.step_count = int(snapshot["step_count"])

    def step(self, action):
        self.step_count += 1
        success = bool(float(np.asarray(action).reshape(-1)[0]) > 0.0 and self.step_count >= 2)
        truncated = bool(self.step_count >= 3 and not success)
        obs = {"state": np.full((6,), float(self.step_count), dtype=np.float32)}
        info = {
            "success": success,
            "instruction_type": "move_to_object",
            "language_instruction": "move to apple",
            "step": int(self.step_count),
        }
        return obs, float(success), success, truncated, info


class _TrajectoryTestLayout:
    @staticmethod
    def flatten(obs):
        return np.asarray(obs["state"], dtype=np.float32).copy()


class _TrajectoryTestRuntime:
    @staticmethod
    def sample_cdpr_chunks_from_envs(*, envs, observations, infos, instructions):
        return np.zeros((len(envs), 2, 5), dtype=np.float32)


class _TrajectoryTestTrainer:
    def __init__(self) -> None:
        self.calls = 0

    def sample_action_group(self, *, state, prior, action_index, group_size):
        # Candidate 0 succeeds in two decisions; candidate 1 fails after three.
        sign = 1.0 if self.calls < 2 else -1.0
        self.calls += 1
        action = np.full((1, 5), sign, dtype=np.float32)
        return action, np.zeros((1,), dtype=np.float32), action[0].copy()


class SmolVLAComplexGRPOTests(unittest.TestCase):
    def test_comparison_configs_build_distinct_grpo_plans(self):
        expected = {
            "cdpr_smolvla_complex_reverse_frontier_grpo.yaml": "reverse_frontier",
            "cdpr_smolvla_complex_lchol_hindsight_grpo.yaml": "lchol_hindsight",
        }
        for filename, approach in expected.items():
            with self.subTest(filename=filename):
                config = load_project_config(ROOT / "configs" / "examples" / filename)
                plan = BootstrapPipeline(config).build_stage_plans(
                    ROOT / "runs" / f"unit_{approach}", ["rl"]
                )[0]
                command = plan.command
                self.assertIn("rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr", command)
                self.assertEqual(command[command.index("--complex-training-approach") + 1], approach)
                self.assertEqual(command[command.index("--batch-size") + 1], "4096")
                self.assertEqual(command[command.index("--minibatch-size") + 1], "4096")
                self.assertEqual(command[command.index("--microbatch-size") + 1], "2048")
                self.assertEqual(command[command.index("--resume-checkpoint") + 1], CHECKPOINT)
                self.assertEqual(config.task.metadata["reward_mode"], "sparse_binary")
                self.assertFalse(config.task.metadata["reward_output_normalization_enabled"])
                if approach == "reverse_frontier":
                    self.assertEqual(command[command.index("--grpo-group-size") + 1], "8")
                    self.assertIn("--grpo-trajectory-groups", command)
                    self.assertIn("--grpo-dynamic-sampling", command)
                    self.assertEqual(
                        command[command.index("--grpo-trajectory-max-decisions") + 1],
                        "128",
                    )
                    self.assertEqual(
                        command[command.index("--grpo-target-records-per-update") + 1],
                        "4096",
                    )
                    self.assertEqual(command[command.index("--clip-range-low") + 1], "0.2")
                    self.assertEqual(command[command.index("--clip-range-high") + 1], "0.28")
                    self.assertEqual(command[command.index("--entropy-coef") + 1], "0.001")
                    self.assertEqual(command[command.index("--ppo-epochs") + 1], "1")
                    self.assertEqual(config.task.metadata["move_to_object_xy_tolerance"], 0.02)
                    self.assertEqual(config.task.metadata["put_container_xy_tolerance"], 0.03)
                    self.assertEqual(config.task.metadata["move_relation_success_zone_size"], 0.03)
                    self.assertEqual(config.task.metadata["between_xy_tolerance"], 0.03)
                    self.assertTrue(config.task.metadata["push_enforce_orthogonal_tolerance"])
                    self.assertTrue(config.task.metadata["push_enforce_max_overshoot"])
                    self.assertEqual(
                        config.task.metadata["reverse_frontier_policy_decision_bounds"],
                        [
                            [4, 6],
                            [7, 10],
                            [11, 16],
                            [17, 24],
                            [25, 52],
                            [53, 80],
                            [81, 128],
                        ],
                    )
                    self.assertTrue(config.task.metadata["put_require_target_grasp_history"])
                    self.assertTrue(
                        config.task.metadata["move_relation_require_target_grasp_history"]
                    )
                    self.assertTrue(
                        config.task.metadata["relation_require_target_grasp_history"]
                    )
                    self.assertTrue(
                        config.task.metadata["reverse_frontier_dynamic_grasp_latch"]
                    )
                    self.assertEqual(
                        config.task.metadata["reverse_frontier_grasp_latch_xy_distance"],
                        0.030,
                    )
                    self.assertEqual(
                        config.task.metadata["reverse_frontier_grasp_latch_z_distance"],
                        0.060,
                    )
                    self.assertEqual(
                        config.task.metadata[
                            "reverse_frontier_grasp_latch_min_finger_contacts"
                        ],
                        1,
                    )
                    self.assertEqual(
                        command[command.index("--max-env-steps") + 1],
                        "128",
                    )

    def test_reverse_frontier_profile_has_longer_base_and_grasp_shells(self):
        specs = get_cdpr_reverse_shell_specs(profile=SMOLVLA_COMPLEX_PROFILE)
        self.assertEqual(len(specs), 8)
        shell_counts = {spec.instruction_id: spec.shell_count for spec in specs}
        self.assertEqual(shell_counts["move_to_object"], 4)
        self.assertEqual(shell_counts["push_left"], 4)
        self.assertEqual(shell_counts["push_right"], 4)
        self.assertEqual(shell_counts["put_into_bowl"], 4)
        for instruction in (
            "put_into_plate",
            "move_left_of_object",
            "move_right_of_object",
            "move_between_objects",
        ):
            self.assertEqual(shell_counts[instruction], 7)

        expected = ((4, 6), (7, 10), (11, 16), (17, 24))
        for shell_id, bounds in enumerate(expected):
            env = _MoveShellEnv()
            info = apply_cdpr_reverse_shell(
                env, shell_id=shell_id, rng=np.random.default_rng(100 + shell_id)
            )
            self.assertEqual(
                (
                    info["curriculum_shell_policy_steps_low"],
                    info["curriculum_shell_policy_steps_high"],
                ),
                bounds,
            )
            self.assertGreaterEqual(info["curriculum_shell_target_policy_steps"], bounds[0])
            self.assertLessEqual(info["curriculum_shell_target_policy_steps"], bounds[1])
            self.assertEqual(
                info["curriculum_shell_target_sim_steps"],
                7 * info["curriculum_shell_target_policy_steps"],
            )
            self.assertEqual(
                (
                    info["curriculum_shell_sim_steps_low"],
                    info["curriculum_shell_sim_steps_high"],
                ),
                (7 * bounds[0], 7 * bounds[1]),
            )
            expected_distance = 0.03 + (
                info["curriculum_shell_target_policy_steps"] - 0.5
            ) * env.action_step_xyz * 0.44
            self.assertAlmostEqual(info["curriculum_shell_distance_m"], expected_distance)
            self.assertEqual(info["curriculum_shell_profile"], SMOLVLA_COMPLEX_PROFILE)

        easiest_env = _MoveShellEnv()
        easiest = apply_cdpr_reverse_shell(
            easiest_env,
            shell_id=0,
            rng=np.random.default_rng(999),
            target_policy_steps=4,
        )
        self.assertEqual(easiest["curriculum_shell_target_policy_steps"], 4)

    def test_dapo_dynamic_sampling_and_reverse_frontier_retries(self):
        args = parse_args(
            [
                "--no-distributed",
                "--grpo-group-size",
                "8",
                "--grpo-trajectory-groups",
                "--grpo-dynamic-sampling",
                "--grpo-dynamic-min-pass-rate",
                "0.1",
                "--grpo-dynamic-max-pass-rate",
                "0.9",
            ]
        )
        self.assertFalse(_trajectory_group_is_informative(np.zeros((8,)), args))
        self.assertFalse(_trajectory_group_is_informative(np.ones((8,)), args))
        mixed = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(_trajectory_group_is_informative(mixed, args))

        base_stats = {
            "group_size": 8,
            "instruction_type": "put_into_plate",
            "curriculum_shell_count": 7,
            "informative_group": 0.0,
            "candidate_success_count": 0.0,
            "curriculum_shell_policy_steps_low": 4,
        }
        easier, retry_kind = _dynamic_frontier_retry_options(
            {**base_stats, "curriculum_shell": 3}, args
        )
        self.assertEqual(retry_kind, "easier_shell")
        self.assertEqual(easier["curriculum_shell"], 2)

        shell0, retry_kind = _dynamic_frontier_retry_options(
            {**base_stats, "curriculum_shell": 0}, args
        )
        self.assertEqual(retry_kind, "shell0_easiest")
        self.assertEqual(shell0["curriculum_shell"], 0)
        self.assertEqual(shell0["curriculum_target_policy_steps"], 4)

        harder, retry_kind = _dynamic_frontier_retry_options(
            {
                **base_stats,
                "curriculum_shell": 3,
                "candidate_success_count": 8.0,
            },
            args,
        )
        self.assertEqual(retry_kind, "harder_shell")
        self.assertEqual(harder["curriculum_shell"], 4)

    def test_trajectory_group_metrics_include_histogram_lengths_and_shell_pass_rate(self):
        stats = [
            {
                "candidate_success_count": 1.0,
                "candidate_reward_mean": 0.125,
                "informative_group": 1.0,
                "all_fail_group": 0.0,
                "all_success_group": 0.0,
                "sampled_policy_decisions": 37.0,
                "accepted_policy_records": 37.0,
                "trajectory_length_mean": 4.625,
                "trajectory_length_min": 3.0,
                "trajectory_length_max": 7.0,
                "instruction_type": "put_into_plate",
                "curriculum_shell": 0,
            },
            {
                "candidate_success_count": 0.0,
                "candidate_reward_mean": 0.0,
                "informative_group": 0.0,
                "all_fail_group": 1.0,
                "all_success_group": 0.0,
                "sampled_policy_decisions": 64.0,
                "accepted_policy_records": 0.0,
                "trajectory_length_mean": 8.0,
                "trajectory_length_min": 8.0,
                "trajectory_length_max": 8.0,
                "shell0_easiest_retry": 1.0,
                "instruction_type": "put_into_plate",
                "curriculum_shell": 0,
            },
        ]
        metrics = _trajectory_group_metric_scalars(stats, group_size=8)
        self.assertEqual(metrics["rollout/grpo_group_success_count_0_rate"], 0.5)
        self.assertEqual(metrics["rollout/grpo_group_success_count_1_rate"], 0.5)
        self.assertEqual(metrics["rollout/grpo_informative_group_rate"], 0.5)
        self.assertEqual(metrics["rollout/grpo_trajectory_length_min"], 3.0)
        self.assertEqual(metrics["rollout/grpo_trajectory_length_max"], 8.0)
        self.assertAlmostEqual(
            metrics["rollout/grpo_group_pass_rate/put_into_plate/shell_00"],
            0.0625,
        )

    def test_trajectory_group_accepts_different_candidate_lengths(self):
        args = parse_args(
            [
                "--no-distributed",
                "--grpo-group-size",
                "2",
                "--grpo-trajectory-groups",
                "--grpo-dynamic-sampling",
                "--grpo-trajectory-max-decisions",
                "3",
                "--action-dim",
                "5",
            ]
        )
        env = _TrajectoryTestEnv()
        obs = {"state": np.zeros((6,), dtype=np.float32)}
        info = {
            "instruction_type": "move_to_object",
            "language_instruction": "move to apple",
            "curriculum_shell": 0,
            "curriculum_shell_count": 4,
            "curriculum_shell_policy_steps_low": 2,
        }
        slot = EnvSlot(
            env=env,
            obs=obs,
            info=info,
            state=np.zeros((6,), dtype=np.float32),
            instruction="move to apple",
        )
        records, _selected, stats, selected_state = _evaluate_trajectory_group(
            trainer=_TrajectoryTestTrainer(),
            runtime=_TrajectoryTestRuntime(),
            slot=slot,
            layout=_TrajectoryTestLayout(),
            args=args,
            progress_only=True,
        )
        self.assertEqual(len(records), 5)
        self.assertEqual(stats["candidate_success_count"], 1.0)
        self.assertEqual(stats["informative_group"], 1.0)
        self.assertEqual(stats["trajectory_length_min"], 2.0)
        self.assertEqual(stats["trajectory_length_max"], 3.0)
        trajectory_advantages = {
            int(record["trajectory_index"]): float(record["advantage"])
            for record in records
        }
        self.assertGreater(trajectory_advantages[0], 0.0)
        self.assertLess(trajectory_advantages[1], 0.0)
        self.assertIn(int(selected_state["length"]), {2, 3})

    def test_lchol_put_stage_promotes_only_after_eighty_percent(self):
        args = SimpleNamespace(
            complex_training_approach="lchol_hindsight",
            instruction_types=["put_into_plate"],
            lchol_hindsight_replay_capacity=100,
            metrics_window_episodes=20,
            put_stage_history_episodes=5,
            put_stage_min_episodes=5,
            put_stage_promotion_success=0.80,
        )
        runtime = SmolVLAComplexRuntime(args=args, seed=3)
        self.assertEqual(runtime.reset_options()["put_start_stage"], 0)
        for success in (True, True, True, False):
            self.assertFalse(
                runtime.record_episode(
                    instruction_type="put_into_plate", success=success, episode_put_stage=0
                )
            )
        self.assertTrue(
            runtime.record_episode(
                instruction_type="put_into_plate", success=True, episode_put_stage=0
            )
        )
        options = runtime.reset_options()
        self.assertEqual(options["put_start_stage"], 1)
        self.assertFalse(options["start_with_caught_object"])

    def test_sparse_move_reward_is_exact_binary_and_blocks_vertical_detour(self):
        target = np.array([0.0, 0.0, 0.15], dtype=np.float32)
        state = init_reward_state(
            initial_ee_pos=np.array([0.10, 0.0, 0.15], dtype=np.float32),
            initial_obj_pos=target,
        )
        metadata = {
            "reward_mode": "sparse_binary",
            "sparse_success_reward": 1.0,
            "sparse_failure_reward": 0.0,
            "move_to_object_validation_distance_threshold": 0.03,
            "move_to_object_max_z_excursion": 0.015,
            "action_saturation_penalty_weight": 100.0,
            "reward_output_normalization_enabled": True,
            "reward_output_normalization": "scale",
            "reward_output_scale": 99.0,
        }
        reward, success, _ = compute_instruction_reward(
            spec=_move_spec(),
            ee_pos=np.array([0.05, 0.0, 0.20], dtype=np.float32),
            obj_pos=target,
            reward_state=state,
            action=np.ones((5,), dtype=np.float32),
            task_metadata=metadata,
        )
        self.assertFalse(success)
        self.assertEqual(reward, 0.0)

        reward, success, info = compute_instruction_reward(
            spec=_move_spec(),
            ee_pos=np.array([0.01, 0.0, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=state,
            action=np.ones((5,), dtype=np.float32),
            task_metadata=metadata,
        )
        self.assertFalse(success)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["move_to_object_z_excursion_ok"], 0.0)

        clean_state = init_reward_state(
            initial_ee_pos=np.array([0.04, 0.0, 0.15], dtype=np.float32),
            initial_obj_pos=target,
        )
        reward, success, _ = compute_instruction_reward(
            spec=_move_spec(),
            ee_pos=np.array([0.01, 0.0, 0.15], dtype=np.float32),
            obj_pos=target,
            reward_state=clean_state,
            action=np.ones((5,), dtype=np.float32),
            task_metadata=metadata,
        )
        self.assertTrue(success)
        self.assertEqual(reward, 1.0)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_grpo_bootstraps_from_td3_actor_checkpoint_directory(self):
        args = parse_args(
            [
                "--device",
                "cpu",
                "--no-distributed",
                "--hidden-dim",
                "16",
                "--chunk-size",
                "2",
                "--action-dim",
                "5",
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = SmolVLAGRPOTrainer(
                args=args,
                state_dim=6,
                action_dim=5,
                chunk_size=2,
                run_dir=root / "source",
                device=torch.device("cpu"),
                distributed=DistributedContext(device="cpu"),
            )
            step_dir = root / "step_6700000"
            step_dir.mkdir()
            checkpoint = step_dir / "smolvla_cdpr_adapter.pt"
            torch.save(
                {"actor": source._unwrap(source.actor).actor.state_dict(), "global_step": 6_700_000},
                checkpoint,
            )
            target = SmolVLAGRPOTrainer(
                args=args,
                state_dim=6,
                action_dim=5,
                chunk_size=2,
                run_dir=root / "target",
                device=torch.device("cpu"),
                distributed=DistributedContext(device="cpu"),
            )

            self.assertEqual(_resolve_checkpoint(step_dir), checkpoint)
            self.assertEqual(target.load(checkpoint), 6_700_000)
            self.assertEqual(target.bootstrap_source, "smolvla_td3_actor")


if __name__ == "__main__":
    unittest.main()
