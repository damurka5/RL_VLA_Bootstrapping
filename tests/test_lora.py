import unittest

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch-less CI skips.
    torch = None
    nn = None

if torch is not None:
    from rl_vla_bootstrapping.policy.lora import (
        LoRALinear,
        attach_lora,
        count_trainable,
        freeze_all_but_lora,
        lora_parameters,
    )


@unittest.skipIf(torch is None, "PyTorch is required for LoRA tests.")
class LoRATests(unittest.TestCase):
    def _model(self):
        # Two "attention" blocks with q/k/v/o proj, one under an "expert"
        # prefix and one under a "vlm" prefix, to test selective targeting.
        return nn.ModuleDict(
            {
                "expert": nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(8, 8),
                        "k_proj": nn.Linear(8, 8),
                        "v_proj": nn.Linear(8, 8),
                        "o_proj": nn.Linear(8, 8),
                        "mlp": nn.Linear(8, 16),
                    }
                ),
                "vlm": nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(8, 8),
                        "v_proj": nn.Linear(8, 8),
                    }
                ),
            }
        )

    def test_adapter_is_identity_at_init(self):
        base = nn.Linear(8, 8)
        wrapped = LoRALinear(base, rank=4, alpha=8.0)
        x = torch.randn(3, 8)
        torch.testing.assert_close(wrapped(x), base(x))

    def test_exposes_base_linear_attributes(self):
        # LeRobot's smolvlm_with_expert reads q_proj.weight.dtype to cast
        # activations, so the wrapper must look like an nn.Linear.
        base = nn.Linear(8, 16)
        wrapped = LoRALinear(base, rank=4, alpha=8.0)
        self.assertIs(wrapped.weight, base.weight)
        self.assertIs(wrapped.bias, base.bias)
        self.assertEqual(wrapped.in_features, 8)
        self.assertEqual(wrapped.out_features, 16)
        self.assertEqual(wrapped.weight.dtype, base.weight.dtype)

    def test_base_parameters_are_not_double_counted(self):
        model = self._model()
        attach_lora(
            model,
            target_leaf_names=("q_proj",),
            name_contains=("expert",),
            rank=4,
            alpha=8.0,
        )
        names = [name for name, _ in model.named_parameters()]
        # The forwarded .weight property must not register an extra parameter.
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("expert.q_proj.base.weight", names)
        self.assertNotIn("expert.q_proj.weight", names)

    def test_targets_only_matching_expert_projections(self):
        model = self._model()
        replaced = attach_lora(
            model,
            target_leaf_names=("q_proj", "k_proj", "v_proj", "o_proj"),
            name_contains=("expert",),
            rank=4,
            alpha=8.0,
        )
        self.assertEqual(
            sorted(replaced),
            ["expert.k_proj", "expert.o_proj", "expert.q_proj", "expert.v_proj"],
        )
        # The vlm projections and the expert mlp are untouched.
        self.assertIsInstance(model["vlm"]["q_proj"], nn.Linear)
        self.assertNotIsInstance(model["expert"]["mlp"], LoRALinear)
        self.assertIsInstance(model["expert"]["q_proj"], LoRALinear)

    def test_only_lora_params_are_trainable(self):
        model = self._model()
        attach_lora(
            model,
            target_leaf_names=("q_proj", "v_proj"),
            name_contains=("expert",),
            rank=4,
            alpha=8.0,
        )
        freeze_all_but_lora(model)
        trainable = lora_parameters(model)
        # 2 wrapped linears * (lora_a + lora_b) = 4 tensors.
        self.assertEqual(len(trainable), 4)
        # rank 4 on 8->8 linears: 4*(8+8) per linear, two linears.
        self.assertEqual(count_trainable(model), 2 * 4 * (8 + 8))
        for name, param in model.named_parameters():
            self.assertEqual(param.requires_grad, "lora_" in name)

    def test_gradient_flows_only_into_lora(self):
        model = self._model()
        attach_lora(
            model,
            target_leaf_names=("q_proj",),
            name_contains=("expert",),
            rank=2,
            alpha=4.0,
        )
        freeze_all_but_lora(model)
        x = torch.randn(5, 8)
        out = model["expert"]["q_proj"](x)
        out.sum().backward()
        wrapped = model["expert"]["q_proj"]
        self.assertIsNotNone(wrapped.lora_b.grad)
        self.assertIsNone(wrapped.base.weight.grad)


if __name__ == "__main__":
    unittest.main()
