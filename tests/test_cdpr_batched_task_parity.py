from __future__ import annotations

import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch is required for batched reward parity",
)
class CDPRBatchedTaskParityTests(unittest.TestCase):
    def test_dense_move_to_distance_reward_matches_existing_cpu_hook(self):
        import torch

        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            BatchedMoveToDistanceReward,
            BatchedTaskState,
            BatchedTaskThresholds,
            evaluate_active_sparse_tasks,
        )
        from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
            InstructionSpec,
            compute_instruction_reward,
            init_reward_state,
        )

        metadata = {
            "reward_mode": "dense",
            "distance_reward_exponent": 2.0,
            "success_distance": 0.02,
            "success_bonus": 0.0,
            "move_to_object_xy_reward_scale": 0.08,
            "move_to_object_distance_reward_weight": 1.0,
            "move_to_object_xy_window_low": 0.0,
            "move_to_object_xy_window_high": 0.02,
            "move_to_object_excess_distance_penalty_weight": 0.0,
            "move_to_object_too_close_penalty_weight": 0.0,
            "move_to_object_success_bonus": 0.0,
            "move_to_object_require_z_window": False,
            "move_to_object_z_penalty_weight": 0.0,
            "action_saturation_penalty_weight": 0.0,
        }
        distances = np.asarray([0.0, 0.02, 0.10, 0.18], dtype=np.float32)
        count = int(distances.size)
        objects = torch.zeros((count, 4, 3), dtype=torch.float32)
        objects[:, 0, 2] = 0.18
        ee = torch.zeros((count, 3), dtype=torch.float32)
        ee[:, 0] = torch.from_numpy(distances)
        ee[:, 2] = 0.40
        state = BatchedTaskState(
            instruction_ids=torch.zeros((count,), dtype=torch.int64),
            target_slots=torch.zeros((count,), dtype=torch.int64),
            reference_slots=torch.full((count,), -1, dtype=torch.int64),
            second_reference_slots=torch.full((count,), -1, dtype=torch.int64),
            initial_target_positions=objects[:, 0].clone(),
            ever_grasped=torch.zeros((count,), dtype=torch.bool),
            grasped=torch.zeros((count,), dtype=torch.bool),
            step_count=torch.zeros((count,), dtype=torch.int64),
            release_threshold=torch.full((count,), 0.55),
            support_surface_z=torch.full((count,), 0.15),
        )
        reward_config = BatchedMoveToDistanceReward.from_metadata(metadata)
        result = evaluate_active_sparse_tasks(
            state=state,
            ee_position=ee,
            object_positions=objects,
            gripper_opening=torch.ones((count,), dtype=torch.float32),
            caught_target=torch.zeros((count,), dtype=torch.bool),
            active_mask=torch.ones((count,), dtype=torch.bool),
            max_steps=128,
            thresholds=BatchedTaskThresholds(
                move_to_xy_low=reward_config.xy_window_low,
                move_to_xy=reward_config.xy_window_high,
            ),
            move_to_distance_reward=reward_config,
        )
        spec = InstructionSpec(
            instruction_type="move_to_object",
            text="move to apple",
            target_object="target",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.0,
            lift_target=0.0,
        )
        cpu_rewards = []
        cpu_success = []
        for index in range(count):
            ee_value = ee[index].numpy()
            object_value = objects[index, 0].numpy()
            reward_state = init_reward_state(ee_value, object_value)
            reward, success, _info = compute_instruction_reward(
                spec=spec,
                ee_pos=ee_value,
                obj_pos=object_value,
                reward_state=reward_state,
                action=np.zeros((5,), dtype=np.float32),
                task_metadata=metadata,
            )
            cpu_rewards.append(reward)
            cpu_success.append(success)
        np.testing.assert_allclose(
            result.rewards.numpy(),
            np.asarray(cpu_rewards, dtype=np.float32),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        self.assertEqual(result.success.tolist(), cpu_success)

    def test_dense_move_to_z_reward_and_success_match_cpu_hook(self):
        import torch

        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            BatchedMoveToDistanceReward,
            BatchedTaskState,
            BatchedTaskThresholds,
            evaluate_active_sparse_tasks,
        )
        from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
            InstructionSpec,
            compute_instruction_reward,
            init_reward_state,
        )

        metadata = {
            "reward_mode": "dense",
            "success_distance": 0.02,
            "success_bonus": 0.0,
            "move_to_object_xy_window_low": 0.0,
            "move_to_object_xy_window_high": 0.02,
            "move_to_object_xy_reward_scale": 0.08,
            "move_to_object_distance_reward_weight": 1.0,
            "move_to_object_success_bonus": 0.0,
            "move_to_object_require_z_window": True,
            "move_to_object_z_window_low": 0.265,
            "move_to_object_z_window_high": 0.275,
            "move_to_object_z_penalty_scale": 0.03,
            "move_to_object_z_penalty_weight": 1.0,
            "action_saturation_penalty_weight": 0.0,
        }
        heights = torch.tensor([0.27, 0.25, 0.29], dtype=torch.float32)
        count = int(heights.numel())
        objects = torch.zeros((count, 4, 3), dtype=torch.float32)
        objects[:, 0, 2] = 0.18
        ee = torch.zeros((count, 3), dtype=torch.float32)
        ee[:, 0] = 0.01
        ee[:, 2] = heights
        state = BatchedTaskState(
            instruction_ids=torch.zeros((count,), dtype=torch.int64),
            target_slots=torch.zeros((count,), dtype=torch.int64),
            reference_slots=torch.full((count,), -1, dtype=torch.int64),
            second_reference_slots=torch.full(
                (count,), -1, dtype=torch.int64
            ),
            initial_target_positions=objects[:, 0].clone(),
            ever_grasped=torch.zeros((count,), dtype=torch.bool),
            grasped=torch.zeros((count,), dtype=torch.bool),
            step_count=torch.zeros((count,), dtype=torch.int64),
            release_threshold=torch.full((count,), 0.55),
            support_surface_z=torch.full((count,), 0.15),
        )
        reward_config = BatchedMoveToDistanceReward.from_metadata(metadata)
        result = evaluate_active_sparse_tasks(
            state=state,
            ee_position=ee,
            object_positions=objects,
            gripper_opening=torch.ones((count,), dtype=torch.float32),
            caught_target=torch.zeros((count,), dtype=torch.bool),
            active_mask=torch.ones((count,), dtype=torch.bool),
            max_steps=128,
            thresholds=BatchedTaskThresholds(
                move_to_xy_low=reward_config.xy_window_low,
                move_to_xy=reward_config.xy_window_high,
            ),
            move_to_distance_reward=reward_config,
        )
        spec = InstructionSpec(
            instruction_type="move_to_object",
            text="move to apple",
            target_object="target",
            direction=np.zeros((3,), dtype=np.float32),
            target_displacement=0.0,
            lift_target=0.0,
        )
        cpu_rewards = []
        cpu_success = []
        for index in range(count):
            ee_value = ee[index].numpy()
            object_value = objects[index, 0].numpy()
            reward, success, _ = compute_instruction_reward(
                spec=spec,
                ee_pos=ee_value,
                obj_pos=object_value,
                reward_state=init_reward_state(ee_value, object_value),
                action=np.zeros((5,), dtype=np.float32),
                task_metadata=metadata,
            )
            cpu_rewards.append(reward)
            cpu_success.append(success)

        np.testing.assert_allclose(
            result.rewards.numpy(),
            np.asarray(cpu_rewards, dtype=np.float32),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        self.assertEqual(result.success.tolist(), cpu_success)
        self.assertEqual(result.success.tolist(), [True, False, False])

    def test_active_success_predicates_match_cpu_reference_fixtures(self):
        import torch

        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            ACTIVE_INSTRUCTION_TYPES,
            BatchedTaskState,
            evaluate_active_sparse_tasks,
        )
        from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
            InstructionSpec,
            compute_instruction_validation_success,
            init_reward_state,
        )

        device = torch.device("cpu")
        objects = torch.zeros((8, 4, 3), dtype=torch.float32, device=device)
        initial = torch.zeros((8, 3), dtype=torch.float32, device=device)
        ee = torch.zeros((8, 3), dtype=torch.float32, device=device)
        ee[:, 2] = 0.40
        target_slot = torch.zeros((8,), dtype=torch.int64, device=device)
        reference_slot = torch.full((8,), -1, dtype=torch.int64, device=device)
        second_slot = torch.full((8,), -1, dtype=torch.int64, device=device)

        objects[0, 0] = torch.tensor([0.0, 0.0, 0.188])
        initial[0] = objects[0, 0]
        ee[0, :2] = torch.tensor([0.01, 0.0])

        # Stay one millimeter inside the threshold so float32 boundary
        # rounding cannot turn the CPU reference into a false negative.
        objects[1, 0] = torch.tensor([0.019, 0.0, 0.188])
        initial[1] = torch.tensor([0.10, 0.0, 0.188])
        objects[2, 0] = torch.tensor([-0.019, 0.0, 0.188])
        initial[2] = torch.tensor([-0.10, 0.0, 0.188])

        reference_slot[3:8] = 1
        objects[3, 0] = torch.tensor([0.01, 0.0, 0.20])
        objects[3, 1] = torch.tensor([0.0, 0.0, 0.174])
        initial[3] = torch.tensor([0.15, 0.0, 0.30])
        objects[4, 0] = torch.tensor([0.10, 0.10, 0.19])
        objects[4, 1] = torch.tensor([0.10, 0.10, 0.162])
        initial[4] = torch.tensor([0.18, 0.10, 0.30])

        objects[5, 0] = torch.tensor([-0.10, 0.0, 0.23])
        objects[5, 1] = torch.tensor([0.0, 0.0, 0.20])
        initial[5] = torch.tensor([-0.05, 0.0, 0.23])
        objects[6, 0] = torch.tensor([0.10, 0.0, 0.23])
        objects[6, 1] = torch.tensor([0.0, 0.0, 0.20])
        initial[6] = torch.tensor([0.05, 0.0, 0.23])

        second_slot[7] = 2
        objects[7, 0] = torch.tensor([0.0, 0.0, 0.23])
        objects[7, 1] = torch.tensor([-0.10, 0.0, 0.20])
        objects[7, 2] = torch.tensor([0.10, 0.0, 0.20])
        initial[7] = torch.tensor([0.0, 0.05, 0.23])

        history = torch.zeros((8,), dtype=torch.bool, device=device)
        history[3:] = True
        state = BatchedTaskState(
            instruction_ids=torch.arange(8, dtype=torch.int64, device=device),
            target_slots=target_slot,
            reference_slots=reference_slot,
            second_reference_slots=second_slot,
            initial_target_positions=initial.clone(),
            ever_grasped=history.clone(),
            grasped=torch.zeros_like(history),
            step_count=torch.zeros((8,), dtype=torch.int64, device=device),
            release_threshold=torch.full((8,), 0.55, device=device),
            support_surface_z=torch.full((8,), 0.15, device=device),
        )
        result = evaluate_active_sparse_tasks(
            state=state,
            ee_position=ee,
            object_positions=objects,
            gripper_opening=torch.full((8,), 0.95, device=device),
            caught_target=torch.zeros((8,), dtype=torch.bool, device=device),
            active_mask=torch.ones((8,), dtype=torch.bool, device=device),
            max_steps=128,
        )
        self.assertEqual(result.success.tolist(), [True] * 8)

        metadata = {
            "reward_mode": "sparse_binary",
            "move_to_object_validation_distance_threshold": 0.02,
            "move_to_object_require_z_window": False,
            "push_position_only_reward": True,
            "push_success_displacement": 0.08,
            "push_enforce_orthogonal_tolerance": True,
            "push_orthogonal_tolerance": 0.02,
            "push_enforce_max_overshoot": True,
            "push_max_overshoot": 0.025,
            "push_require_object_on_support": True,
            "push_support_min_clearance": 0.005,
            "push_support_vertical_tolerance": 0.04,
            "put_container_xy_tolerance": 0.03,
            "put_container_z_tolerance": 0.12,
            "put_require_release": True,
            "put_require_target_grasp_history": True,
            "put_min_target_motion": 0.04,
            "relation_left_right_offset": 0.10,
            "move_relation_success_zone_size": 0.03,
            "move_relation_min_target_motion": 0.04,
            "move_relation_require_target_grasp_history": True,
            "move_relation_require_release": True,
            "relation_min_target_motion": 0.04,
            "relation_require_target_grasp": False,
            "relation_require_target_grasp_history": True,
            "relation_require_release": True,
            "between_xy_tolerance": 0.03,
        }

        class FakeEnv:
            def __init__(self, positions):
                self.positions = positions

            def _get_body_position(self, name):
                return np.asarray(self.positions[name], dtype=np.float32).copy()

        def cpu_evaluate(objects_value, initial_value, ee_value):
            cpu_success = []
            for index, instruction_type in enumerate(ACTIVE_INSTRUCTION_TYPES):
                positions = {
                    "target": objects_value[index, 0].numpy(),
                    "reference": objects_value[index, 1].numpy(),
                    "second": objects_value[index, 2].numpy(),
                }
                env = FakeEnv(positions)
                reward_state = init_reward_state(
                    ee_value[index].numpy(), initial_value[index].numpy()
                )
                reward_state.ever_grasped = bool(index >= 3)
                reward_state.grasped = False
                spec = InstructionSpec(
                    instruction_type=instruction_type,
                    text=instruction_type,
                    target_object="target",
                    direction=np.zeros((3,), dtype=np.float32),
                    target_displacement=0.0,
                    lift_target=0.0,
                    reference_object="reference",
                    second_reference_object="second",
                )
                success, _ = compute_instruction_validation_success(
                    spec=spec,
                    ee_pos=ee_value[index].numpy(),
                    reward_state=reward_state,
                    task_metadata=metadata,
                    obj_pos=positions["target"],
                    goal_pos=positions["target"],
                    env=env,
                    target_body_name="target",
                    reference_body_name="reference",
                    second_reference_body_name="second",
                    gripper_opening=0.95,
                    support_surface_z=0.15,
                    caught_object_is_target=False,
                )
                cpu_success.append(bool(success))
            return cpu_success

        cpu_success = cpu_evaluate(objects, initial, ee)
        self.assertEqual(cpu_success, result.success.tolist())

        failing_objects = objects.clone()
        failing_ee = ee.clone()
        failing_ee[0, 0] = 0.05
        failing_objects[1, 0, 0] = 0.05
        failing_objects[2, 0, 0] = -0.05
        failing_objects[3, 0, 0] = 0.08
        failing_objects[4, 0, 0] = 0.15
        failing_objects[5, 0, 0] = -0.05
        failing_objects[6, 0, 0] = 0.05
        failing_objects[7, 0, 1] = 0.08
        failing_state = BatchedTaskState(
            instruction_ids=torch.arange(8, dtype=torch.int64, device=device),
            target_slots=target_slot,
            reference_slots=reference_slot,
            second_reference_slots=second_slot,
            initial_target_positions=initial.clone(),
            ever_grasped=history.clone(),
            grasped=torch.zeros_like(history),
            step_count=torch.zeros((8,), dtype=torch.int64, device=device),
            release_threshold=torch.full((8,), 0.55, device=device),
            support_surface_z=torch.full((8,), 0.15, device=device),
        )
        failing_result = evaluate_active_sparse_tasks(
            state=failing_state,
            ee_position=failing_ee,
            object_positions=failing_objects,
            gripper_opening=torch.full((8,), 0.95, device=device),
            caught_target=torch.zeros((8,), dtype=torch.bool, device=device),
            active_mask=torch.ones((8,), dtype=torch.bool, device=device),
            max_steps=128,
        )
        self.assertEqual(failing_result.success.tolist(), [False] * 8)
        self.assertEqual(
            cpu_evaluate(failing_objects, initial, failing_ee),
            failing_result.success.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
