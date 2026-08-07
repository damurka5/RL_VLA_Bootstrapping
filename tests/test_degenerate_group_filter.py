"""Groups that separated nothing must not vote on the gradient.

The GRPO advantage is the centred reward divided by the group's own std, floored
at 1e-6, then clipped at ``grpo_clip_advantage_abs`` (6.0). So a group whose
eight candidates all did the same thing does not contribute a small gradient --
it contributes a FULL-MAGNITUDE one, made of rollout noise. Measured over the
10M pick_up run: 41 of 128 groups per update had a reward std under 0.05, 31 of
them pre-grasped groups where no candidate lifted and the eight rewards are
near-identical by construction.

This is DAPO's dynamic sampling translated to a dense reward. The filter that
was already in the collector -- ``reward_span > 1e-6`` -- is the literal
all-correct-or-all-wrong test and never fires on a dense reward:
``informative_groups`` equalled ``groups_collected`` on every update of every
run in the campaign.

The arithmetic that makes this worth a filter rather than a shrug, checked
directly below: a group whose rewards differ by 0.001 produces the same
advantage magnitude as one whose rewards differ by 3.0.
"""

from __future__ import annotations

import unittest

from rl_vla_bootstrapping.policy.rank_local_grpo import torch_group_advantages

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class WhyTheFilterIsNeededTest(unittest.TestCase):
    def test_noise_and_signal_produce_the_same_advantage_magnitude(self) -> None:
        """The premise of the whole change, asserted rather than asserted-to."""

        signal = torch.tensor([[0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0]])
        noise = torch.tensor(
            [[1.0, 1.001, 1.0, 1.001, 1.0, 1.001, 1.0, 1.001]]
        )
        a_signal = torch_group_advantages(signal, normalize=True, clip_abs=6.0)
        a_noise = torch_group_advantages(noise, normalize=True, clip_abs=6.0)
        self.assertAlmostEqual(
            float(a_signal.abs().mean()), float(a_noise.abs().mean()), places=4
        )
        # Both are order 1, not order 0.001.
        self.assertGreater(float(a_noise.abs().mean()), 0.9)

    def test_the_std_floor_is_what_does_it(self) -> None:
        """A group with literally identical rewards divides by the 1e-6 floor."""

        flat = torch.full((1, 8), 2.0)
        out = torch_group_advantages(flat, normalize=True, clip_abs=6.0)
        # Centred to exactly zero here, so the floor shows up as 0/1e-6 = 0.
        # The danger is the NEARLY-flat case above, which is what the campaign
        # actually produces -- rollout noise, not bit-identical rewards.
        self.assertEqual(float(out.abs().max()), 0.0)


@unittest.skipIf(torch is None, "torch is unavailable")
class FilterSelectionTest(unittest.TestCase):
    """The masking arithmetic, on the shapes collect_round uses.

    Reproduces the collector's expressions rather than importing the rollout,
    which needs MJWarp and a GPU. What is being checked is which records survive
    -- the part a fencepost or a stream mix-up silently gets wrong.
    """

    GROUPS = 4
    SIZE = 8

    def _rewards(self):
        # Groups 0 and 2 separate something; 1 and 3 are noise.
        rewards = torch.zeros((self.GROUPS, self.SIZE))
        rewards[0] = torch.linspace(0.0, 3.0, self.SIZE)
        rewards[1] = 1.0 + torch.linspace(0.0, 0.001, self.SIZE)
        rewards[2] = torch.linspace(0.0, 2.0, self.SIZE)
        rewards[3] = 2.0 + torch.linspace(0.0, 0.002, self.SIZE)
        return rewards

    def test_only_the_flat_groups_are_dropped(self) -> None:
        rewards = self._rewards()
        degenerate = rewards.std(dim=1, unbiased=False) < 0.05
        self.assertEqual(degenerate.tolist(), [False, True, False, True])

    def test_a_zero_threshold_keeps_everything(self) -> None:
        rewards = self._rewards()
        degenerate = rewards.std(dim=1, unbiased=False) < 0.0
        self.assertEqual(degenerate.sum().item(), 0)

    def test_the_mask_expands_to_every_candidate_of_a_dropped_group(self) -> None:
        rewards = self._rewards()
        degenerate = rewards.std(dim=1, unbiased=False) < 0.05
        world = ~degenerate.repeat_interleave(self.SIZE)
        self.assertEqual(int(world.numel()), self.GROUPS * self.SIZE)
        # Groups 1 and 3 gone entirely, groups 0 and 2 kept entirely.
        self.assertEqual(world[: self.SIZE].sum().item(), self.SIZE)
        self.assertEqual(world[self.SIZE : 2 * self.SIZE].sum().item(), 0)
        self.assertEqual(
            world[2 * self.SIZE : 3 * self.SIZE].sum().item(), self.SIZE
        )
        self.assertEqual(world[3 * self.SIZE :].sum().item(), 0)

    def test_the_two_return_streams_are_filtered_independently(self) -> None:
        """The case pre-grasped groups actually produce.

        A group can separate the approach (did it reach a good grasp?) while
        separating nothing on the lift, or the reverse. Masking both on one test
        throws away usable gradient; masking neither keeps the noise. Records
        must be filtered by the stream their advantage came from.
        """

        terminal = torch.zeros((2, self.SIZE))
        terminal[0] = 5.0  # lift: nothing separated
        terminal[1] = torch.linspace(0.0, 4.0, self.SIZE)
        pre = torch.zeros((2, self.SIZE))
        pre[0] = torch.linspace(0.0, 3.0, self.SIZE)  # approach: separated
        pre[1] = 2.0  # approach: nothing separated

        degenerate_terminal = terminal.std(dim=1, unbiased=False) < 0.05
        degenerate_pre = pre.std(dim=1, unbiased=False) < 0.05
        self.assertEqual(degenerate_terminal.tolist(), [True, False])
        self.assertEqual(degenerate_pre.tolist(), [False, True])

        usable_terminal = ~degenerate_terminal.repeat_interleave(self.SIZE)
        usable_pre = ~degenerate_pre.repeat_interleave(self.SIZE)
        record_world = torch.arange(2 * self.SIZE)
        # Half the records are post-grasp (terminal stream), half pre.
        record_post = torch.zeros((2 * self.SIZE,), dtype=torch.bool)
        record_post[::2] = True

        usable = torch.where(
            record_post,
            usable_terminal.index_select(0, record_world),
            usable_pre.index_select(0, record_world),
        )
        # Group 0: post-grasp records dropped, pre-grasp records kept.
        self.assertEqual(usable[: self.SIZE][::2].sum().item(), 0)
        self.assertEqual(
            usable[: self.SIZE][1::2].sum().item(), self.SIZE // 2
        )
        # Group 1: the reverse.
        self.assertEqual(
            usable[self.SIZE :][::2].sum().item(), self.SIZE // 2
        )
        self.assertEqual(usable[self.SIZE :][1::2].sum().item(), 0)

    def test_a_single_stream_mask_would_lose_usable_gradient(self) -> None:
        """Why the per-stream split earns its complexity.

        Filtering both streams on the terminal test alone would drop group 0's
        approach records, which did separate something.
        """

        terminal = torch.zeros((2, self.SIZE))
        terminal[0] = 5.0
        terminal[1] = torch.linspace(0.0, 4.0, self.SIZE)
        naive = (~(terminal.std(dim=1, unbiased=False) < 0.05)).repeat_interleave(
            self.SIZE
        )
        self.assertEqual(naive[: self.SIZE].sum().item(), 0)


class CollectorWiringTest(unittest.TestCase):
    """The knob has to reach the collector, and default to off."""

    def test_the_collector_accepts_the_threshold(self) -> None:
        import inspect

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            RankLocalMJWarpGRPOCollector,
        )

        signature = inspect.signature(
            RankLocalMJWarpGRPOCollector.__init__
        )
        parameter = signature.parameters.get("min_group_reward_std")
        self.assertIsNotNone(parameter)
        self.assertEqual(parameter.default, 0.0)

    def test_the_training_argument_exists_and_defaults_to_off(self) -> None:
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
            parse_args,
        )

        args = parse_args(["--device", "cpu", "--no-distributed"])
        self.assertEqual(args.grpo_min_group_reward_std, 0.0)

    def test_the_entrypoint_passes_it_through(self) -> None:
        import inspect

        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as mod

        source = inspect.getsource(mod)
        self.assertIn("min_group_reward_std=float(", source)

    def test_the_pick_up_config_enables_it(self) -> None:
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[1]
        raw = yaml.safe_load(
            (
                root
                / "configs"
                / "examples"
                / "cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml"
            ).read_text()
        )

        def find(node, key):
            if isinstance(node, dict):
                for name, value in node.items():
                    if name == key:
                        return value
                    found = find(value, key)
                    if found is not None:
                        return found
            return None

        self.assertEqual(find(raw, "grpo_min_group_reward_std"), 0.05)


if __name__ == "__main__":
    unittest.main()
