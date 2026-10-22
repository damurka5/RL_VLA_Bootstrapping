"""Re-heading SmolVLA's action expert to a robot's own action width.

Exercised against a stand-in with SmolVLA's attribute layout rather than the
real policy, so these run on a host with no LeRobot and no GPU. That is also
the point of the module being duck-typed: the surgery is two `nn.Linear`
replacements and a config update, and none of it needs the 450M-parameter model
to be correct.
"""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    _TORCH = False

from rl_vla_bootstrapping.policy.smolvla_action_expert import (
    reinitialize_action_expert,
    trainable_action_expert_parameters,
)

requires_torch = unittest.skipUnless(_TORCH, "PyTorch is not installed.")


if _TORCH:

    class _FakeConfig:
        def __init__(self) -> None:
            self.max_action_dim = 32
            self.max_state_dim = 32
            self.chunk_size = 8

    class _FakeFlowModel(nn.Module):
        """Same attribute names and widths as LeRobot's VLAFlowMatching."""

        def __init__(self, expert_hidden: int = 16, text_hidden: int = 24) -> None:
            super().__init__()
            self.state_proj = nn.Linear(32, text_hidden)
            self.action_in_proj = nn.Linear(32, expert_hidden)
            self.action_out_proj = nn.Linear(expert_hidden, 32)
            self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
            self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
            # Stands in for the expert transformer: pretrained, width-agnostic,
            # and must survive the surgery untouched.
            self.vlm_with_expert = nn.Linear(expert_hidden, expert_hidden)

    class _FakePolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _FakeConfig()
            self.model = _FakeFlowModel()


@requires_torch
class ReinitializeActionExpertTests(unittest.TestCase):
    def test_projections_take_the_robot_action_width(self) -> None:
        policy = _FakePolicy()
        report = reinitialize_action_expert(policy, action_dim=5)
        self.assertEqual(policy.model.action_in_proj.in_features, 5)
        self.assertEqual(policy.model.action_out_proj.out_features, 5)
        # The hidden side is the expert's, and must not move.
        self.assertEqual(policy.model.action_in_proj.out_features, 16)
        self.assertEqual(policy.model.action_out_proj.in_features, 16)
        self.assertEqual(report.action_dim, 5)
        self.assertEqual(report.previous_action_dim, 32)

    def test_config_moves_with_the_layers(self) -> None:
        """`prepare_action` pads from the config; a stale 32 feeds a 5-wide Linear."""

        policy = _FakePolicy()
        reinitialize_action_expert(policy, action_dim=5)
        self.assertEqual(policy.config.max_action_dim, 5)
        self.assertEqual(policy.config.max_state_dim, 32, "state was not asked for")

    def test_state_projection_is_opt_in(self) -> None:
        policy = _FakePolicy()
        self.assertEqual(policy.model.state_proj.in_features, 32)
        reinitialize_action_expert(policy, action_dim=5, state_dim=6)
        self.assertEqual(policy.model.state_proj.in_features, 6)
        self.assertEqual(policy.config.max_state_dim, 6)

    def test_the_expert_transformer_is_not_touched(self) -> None:
        """Re-heading, not retraining: the pretrained body must survive."""

        policy = _FakePolicy()
        before = policy.model.vlm_with_expert.weight.detach().clone()
        reinitialize_action_expert(policy, action_dim=5)
        torch.testing.assert_close(policy.model.vlm_with_expert.weight, before)

    def test_only_the_new_head_trains(self) -> None:
        policy = _FakePolicy()
        reinitialize_action_expert(policy, action_dim=5)
        trainable = {
            name
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(all(n.startswith("model.action_") for n in trainable), trainable)
        self.assertFalse(
            any("vlm_with_expert" in name for name in trainable),
            "the pretrained expert must stay frozen unless LoRA is attached",
        )

    def test_freeze_rest_can_be_declined_for_lora(self) -> None:
        policy = _FakePolicy()
        reinitialize_action_expert(policy, action_dim=5, freeze_rest=False)
        self.assertTrue(policy.model.vlm_with_expert.weight.requires_grad)

    def test_report_says_the_prior_is_gone(self) -> None:
        """Replacing `action_out_proj` destroys the prior; callers must know.

        A KL leash to the prior, or a residual head initialized as a small
        correction on top of it, is meaningless after this and would train
        against a decoder that outputs noise.
        """

        policy = _FakePolicy()
        report = reinitialize_action_expert(policy, action_dim=5)
        self.assertFalse(report.prior_preserved)
        metadata = report.to_metadata()
        self.assertEqual(metadata["action_expert/action_dim"], 5)
        self.assertEqual(metadata["action_expert/previous_action_dim"], 32)
        self.assertFalse(metadata["action_expert/prior_preserved"])

    def test_output_is_the_robot_action_width_end_to_end(self) -> None:
        policy = _FakePolicy()
        reinitialize_action_expert(policy, action_dim=5)
        hidden = torch.zeros(3, policy.model.action_out_proj.in_features)
        self.assertEqual(tuple(policy.model.action_out_proj(hidden).shape), (3, 5))
        action = torch.zeros(3, 5)
        self.assertEqual(
            tuple(policy.model.action_in_proj(action).shape),
            (3, policy.model.action_in_proj.out_features),
        )

    def test_trainable_parameter_list_is_stable_and_complete(self) -> None:
        policy = _FakePolicy()
        report = reinitialize_action_expert(policy, action_dim=5, state_dim=6)
        params = trainable_action_expert_parameters(policy)
        self.assertEqual(
            sum(p.numel() for p in params), report.trainable_parameters
        )
        # Stable order, so two ranks build identical optimizer state.
        again = trainable_action_expert_parameters(policy)
        self.assertEqual([id(p) for p in params], [id(p) for p in again])

    def test_rejects_a_model_without_the_expected_layout(self) -> None:
        class _Wrong(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Linear(4, 4)
                self.config = _FakeConfig()

        with self.assertRaises(AttributeError):
            reinitialize_action_expert(_Wrong(), action_dim=5)

    def test_rejects_a_nonsense_width(self) -> None:
        with self.assertRaises(ValueError):
            reinitialize_action_expert(_FakePolicy(), action_dim=0)
        with self.assertRaises(ValueError):
            reinitialize_action_expert(_FakePolicy(), action_dim=5, state_dim=-1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
