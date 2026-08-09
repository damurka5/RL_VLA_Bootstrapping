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


class VisionTowerLoRATest(unittest.TestCase):
    """LoRA on the VLM vision tower, and the failure that would be silent.

    Every measurement points at the encoder: the frozen connector decodes the
    gripper->object direction at ~0.07 cosine, the un-projected 30720-d version
    is no better, and the task needs localization to ~2 cm. Nothing downstream
    can create information the encoder never produced.

    The trap is the leaf names. A SigLIP-style vision tower uses ``out_proj``
    rather than ``o_proj`` and ``fc1``/``fc2`` rather than
    ``gate_proj``/``up_proj``/``down_proj``, so reusing the action expert's list
    wraps almost nothing -- and a run would then train an empty adapter for days
    while reporting a healthy module count from the expert half.
    """

    def _tower(self):
        import torch.nn as nn

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)
                self.out_proj = nn.Linear(8, 8)

        class Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.o_proj = nn.Linear(8, 8)

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = Attention()
                self.lm_expert = Expert()

        return Root()

    def test_the_vision_tower_leaf_names_are_wrapped(self):
        from rl_vla_bootstrapping.policy.lora import attach_lora

        root = self._tower()
        replaced = attach_lora(
            root,
            target_leaf_names=("q_proj", "k_proj", "v_proj", "out_proj"),
            name_contains=("vision_model",),
            rank=2,
            alpha=4.0,
        )
        self.assertEqual(len(replaced), 4)
        self.assertTrue(all("vision_model" in name for name in replaced))

    def test_the_expert_leaf_names_miss_the_vision_tower(self):
        """The silent failure this configuration exists to avoid."""

        from rl_vla_bootstrapping.policy.lora import attach_lora

        root = self._tower()
        replaced = attach_lora(
            root,
            # The action expert's list, applied to the vision tower.
            target_leaf_names=("q_proj", "k_proj", "v_proj", "o_proj"),
            name_contains=("vision_model",),
            rank=2,
            alpha=4.0,
        )
        # q/k/v match by coincidence; out_proj -- the one that mixes heads --
        # does not, and o_proj matches nothing at all.
        self.assertNotIn(
            "vision_model.out_proj", [name for name in replaced]
        )

    def test_the_two_groups_do_not_overlap(self):
        from rl_vla_bootstrapping.policy.lora import attach_lora

        root = self._tower()
        expert = attach_lora(
            root,
            target_leaf_names=("q_proj", "o_proj"),
            name_contains=("lm_expert",),
            rank=2,
            alpha=4.0,
        )
        vision = attach_lora(
            root,
            target_leaf_names=("q_proj", "out_proj"),
            name_contains=("vision_model",),
            rank=2,
            alpha=4.0,
        )
        self.assertTrue(set(expert).isdisjoint(set(vision)))
        self.assertTrue(all("lm_expert" in name for name in expert))
        self.assertTrue(all("vision_model" in name for name in vision))

    def test_the_flag_and_its_filters_exist_and_default_off(self):
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
            parse_args,
        )

        args = parse_args(["--device", "cpu", "--no-distributed"])
        self.assertFalse(args.train_vla_vision_lora)
        self.assertEqual(args.lora_vision_name_contains, "vision_model")
        self.assertIn("out_proj", args.lora_vision_leaf_names)
        self.assertNotIn("o_proj,", args.lora_vision_leaf_names)

    def test_the_pick_up_config_actually_enables_the_vision_tower(self):
        """A 3.6M-step run was spent believing it was on when it was not.

        The instruction was "add these two lines to the config by hand", the
        lines never arrived, and the only symptom was vla_lora/vision_modules
        sitting at 0 in a metric nobody thought to check. The flag belongs in
        the config, and this is what keeps it there.
        """

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

        self.assertIs(find(raw, "train_vla_vision_lora"), True)
        # The vision backward is the expensive part; 16 is the expert-only size.
        self.assertLessEqual(int(find(raw, "vla_microbatch_size")), 8)

    def test_the_startup_log_states_the_vision_status_either_way(self):
        """Silence read as "it attached" once already."""

        import inspect

        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as mod

        source = inspect.getsource(mod)
        self.assertIn("vision tower ADAPTED", source)
        self.assertIn("vision tower NOT adapted", source)
