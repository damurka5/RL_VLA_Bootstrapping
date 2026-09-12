"""Three independent milestone returns for end-to-end put_into GRPO."""

from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace

from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    GroupedFullTaskSceneResetter,
    ThreeStageMilestones,
    advance_three_stage_milestones,
    stage_balanced_record_weights,
    three_stage_group_credit,
)
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
    parse_args,
    torch,
)


def _args(*extra: str):
    return parse_args(["--device", "cpu", "--no-distributed", *extra])


@unittest.skipIf(torch is None, "torch is not installed")
class ThreeStageSparseCreditTests(unittest.TestCase):
    def test_parser_is_opt_in_and_rejects_the_legacy_split_together(self):
        self.assertFalse(_args().three_stage_sparse_credit)
        self.assertTrue(
            _args("--three-stage-sparse-credit").three_stage_sparse_credit
        )
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            _args("--three-stage-sparse-credit", "--split-credit-at-grasp")

    def test_each_stage_has_its_own_group_filter_and_advantage(self):
        returns = torch.tensor(
            [
                [[0, 0, 1, 1], [1, 1, 1, 1]],
                [[0, 0, 0, 0], [0, 1, 0, 1]],
                [[0, 1, 0, 1], [0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        )
        advantage, usable = three_stage_group_credit(
            returns,
            normalize=True,
            clip_abs=6.0,
            dynamic_sampling=True,
            dynamic_min_pass_rate=0.1,
            dynamic_max_pass_rate=0.9,
            min_group_reward_std=0.05,
        )
        self.assertEqual(usable.tolist(), [[True, False], [False, True], [True, False]])
        self.assertGreater(float(advantage[0, 0, 2]), 0.0)
        self.assertGreater(float(advantage[1, 1, 1]), 0.0)
        self.assertGreater(float(advantage[2, 0, 1]), 0.0)

    def test_loss_mass_is_equal_even_when_stage_lengths_are_not(self):
        stage = torch.tensor([0, 0, 0, 0, 1, 1, 2])
        valid = torch.ones(7, dtype=torch.bool)
        weights = stage_balanced_record_weights(stage, valid)
        totals = [float(weights[stage == value].sum()) for value in range(3)]
        self.assertAlmostEqual(totals[0], totals[1], places=6)
        self.assertAlmostEqual(totals[1], totals[2], places=6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_manifest_sampler_selects_one_scene_per_group(self):
        resetter = object.__new__(GroupedFullTaskSceneResetter)
        resetter.layout = SimpleNamespace(groups_per_rank=4)
        resetter.scenes = tuple(
            SimpleNamespace(scene_index=index) for index in range(20)
        )
        resetter.rank = 1
        resetter.base_seed = 7
        selected = resetter._group_scenes(update_index=2, round_index=3)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({scene.scene_index for scene in selected}), 4)

    def test_pickup_cannot_be_credited_before_open_hand_approach(self):
        worlds = 1
        reset = SimpleNamespace(
            task_state=SimpleNamespace(
                target_slots=torch.tensor([0]),
                reference_slots=torch.tensor([1]),
                initial_target_positions=torch.tensor([[0.0, 0.0, 0.15]]),
            )
        )
        low_dim = SimpleNamespace(
            object_positions=torch.tensor([[[0.0, 0.0, 0.20], [0.1, 0.0, 0.15]]]),
            gripper_opening=torch.tensor([0.5]),
        )
        result = SimpleNamespace(
            success=torch.tensor([False]),
            diagnostics={
                "pick_grasp_distance": torch.tensor([0.01]),
                "ever_grasped": torch.tensor([True]),
                "credited_lift": torch.tensor([0.05]),
                "released": torch.tensor([False]),
                "container_xy_radius": torch.tensor([0.091]),
                "wrong_place_drop": torch.tensor([False]),
            },
        )
        state = advance_three_stage_milestones(
            ThreeStageMilestones.zeros(torch, worlds, torch.device("cpu")),
            reset=reset,
            low_dim=low_dim,
            result=result,
            physical_grasp=torch.tensor([True]),
            gripper_command=torch.tensor([-1.0]),
            previous_opening=torch.tensor([0.9]),
            active_mask=torch.tensor([True]),
        )
        self.assertFalse(bool(state.approached[0]))
        self.assertFalse(bool(state.picked_up[0]))

        # The same geometry with an open hand and an unmoved object earns M1;
        # M2 remains ordered after it.
        low_dim.gripper_opening = torch.tensor([0.95])
        low_dim.object_positions[0, 0, 2] = 0.15
        result.diagnostics["credited_lift"] = torch.tensor([0.0])
        state = advance_three_stage_milestones(
            state,
            reset=reset,
            low_dim=low_dim,
            result=result,
            physical_grasp=torch.tensor([False]),
            gripper_command=torch.tensor([0.0]),
            previous_opening=torch.tensor([0.95]),
            active_mask=torch.tensor([True]),
        )
        self.assertTrue(bool(state.approached[0]))
        self.assertFalse(bool(state.picked_up[0]))

    def test_intentional_release_is_not_marked_as_carry_slip(self):
        reset = SimpleNamespace(
            task_state=SimpleNamespace(
                target_slots=torch.tensor([0]),
                reference_slots=torch.tensor([1]),
                initial_target_positions=torch.tensor([[0.0, 0.0, 0.15]]),
            )
        )
        low_dim = SimpleNamespace(
            object_positions=torch.tensor([[[0.1, 0.0, 0.20], [0.1, 0.0, 0.15]]]),
            gripper_opening=torch.tensor([0.50]),
        )
        diagnostics = {
            "pick_grasp_distance": torch.tensor([0.20]),
            "ever_grasped": torch.tensor([True]),
            "credited_lift": torch.tensor([0.05]),
            "released": torch.tensor([False]),
            "container_xy_radius": torch.tensor([0.091]),
            "wrong_place_drop": torch.tensor([False]),
        }
        state = ThreeStageMilestones(
            approached=torch.tensor([True]),
            picked_up=torch.tensor([True]),
            placed=torch.tensor([False]),
            released=torch.tensor([False]),
            carry_slip=torch.tensor([False]),
            wrong_place=torch.tensor([False]),
        )
        state = advance_three_stage_milestones(
            state,
            reset=reset,
            low_dim=low_dim,
            result=SimpleNamespace(success=torch.tensor([False]), diagnostics=diagnostics),
            physical_grasp=torch.tensor([False]),
            gripper_command=torch.tensor([1.0]),
            previous_opening=torch.tensor([0.45]),
            active_mask=torch.tensor([True]),
        )
        self.assertFalse(bool(state.carry_slip[0]))

        diagnostics["released"] = torch.tensor([True])
        low_dim.gripper_opening = torch.tensor([0.60])
        state = advance_three_stage_milestones(
            state,
            reset=reset,
            low_dim=low_dim,
            result=SimpleNamespace(success=torch.tensor([True]), diagnostics=diagnostics),
            physical_grasp=torch.tensor([False]),
            gripper_command=torch.tensor([1.0]),
            previous_opening=torch.tensor([0.50]),
            active_mask=torch.tensor([True]),
        )
        self.assertTrue(bool(state.placed[0]))
        self.assertFalse(bool(state.carry_slip[0]))


if __name__ == "__main__":
    unittest.main()
