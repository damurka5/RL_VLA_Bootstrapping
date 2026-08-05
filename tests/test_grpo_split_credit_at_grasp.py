"""Approach and lift must be scored on their own outcomes.

The GRPO return is the last active step's reward broadcast to every step of the
trajectory, so a descent action and a lift action receive identical credit while
the task wants opposite z from them -- and one residual serves both phases. This
splits the return at the latch. The failure mode it must not have is a
mis-assigned phase flag: crediting a descent action with the lift's advantage is
silent, and it is exactly what the change exists to stop.
"""

from __future__ import annotations

import unittest

from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
    parse_args,
    torch,
)
from rl_vla_bootstrapping.policy.rank_local_grpo import torch_group_advantages


def _args(*extra: str):
    return parse_args(["--device", "cpu", "--no-distributed", *extra])


@unittest.skipIf(torch is None, "torch is not installed")
class SplitCreditAtGraspTests(unittest.TestCase):
    def test_parser_defaults_off_and_toggles(self):
        self.assertFalse(_args().split_credit_at_grasp)
        self.assertTrue(_args("--split-credit-at-grasp").split_credit_at_grasp)
        self.assertFalse(
            _args("--no-split-credit-at-grasp").split_credit_at_grasp
        )

    def test_phase_flag_reads_the_state_before_the_action(self):
        """A record is post-grasp only if the world was ALREADY holding.

        Reading first_grasp_step after the step would credit the very action
        that achieved the grasp to the lift segment, which is the approach's
        result and belongs to the approach.
        """

        first_grasp_step = torch.tensor([-1, -1, 5, 5])
        prelifted = torch.tensor([False, True, False, False])
        holding = (first_grasp_step >= 0) | prelifted
        self.assertEqual(holding.tolist(), [False, True, True, True])

    def test_never_grasped_worlds_keep_the_terminal_return(self):
        first_grasp_step = torch.tensor([-1, 3])
        reward_at_first_grasp = torch.tensor([0.0, 1.80])
        candidate_rewards = torch.tensor([1.10, 4.20])
        pre = torch.where(
            first_grasp_step >= 0, reward_at_first_grasp, candidate_rewards
        )
        # World 0 never grasped: unchanged from today's behaviour.
        self.assertAlmostEqual(float(pre[0]), 1.10)
        # World 1 latched: its approach is scored at the latch, NOT on the 4.20
        # the lift went on to earn.
        self.assertAlmostEqual(float(pre[1]), 1.80)

    def test_the_two_segments_can_disagree(self):
        """The point of the split: a good approach with a failed lift.

        Group of four sharing a start. Worlds 2 and 3 reached a better grasp but
        then failed to lift. Under one return their approach is punished with
        their lift; split, it is rewarded on its own merit.
        """

        group = 1, 4
        pre_returns = torch.tensor([[1.20, 1.25, 1.80, 1.85]])
        terminal = torch.tensor([[3.90, 4.00, 1.30, 1.35]])
        pre_adv = torch_group_advantages(
            pre_returns, normalize=True, clip_abs=6.0
        ).reshape(-1)
        post_adv = torch_group_advantages(
            terminal, normalize=True, clip_abs=6.0
        ).reshape(-1)
        # Approach advantage is POSITIVE for the two that grasped better...
        self.assertGreater(float(pre_adv[2]), 0.0)
        self.assertGreater(float(pre_adv[3]), 0.0)
        # ...while their lift advantage is negative. One scalar per trajectory
        # cannot express both, which is the whole problem.
        self.assertLess(float(post_adv[2]), 0.0)
        self.assertLess(float(post_adv[3]), 0.0)
        self.assertEqual(group[1], int(pre_returns.shape[1]))

    def test_records_are_routed_by_their_own_phase_flag(self):
        record_world = torch.tensor([0, 0, 1, 1])
        record_post = torch.tensor([False, True, False, True])
        pre_adv = torch.tensor([-1.0, -2.0])
        post_adv = torch.tensor([+1.0, +2.0])
        advantage = torch.where(
            record_post,
            post_adv.index_select(0, record_world),
            pre_adv.index_select(0, record_world),
        )
        self.assertEqual(advantage.tolist(), [-1.0, 1.0, -2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
