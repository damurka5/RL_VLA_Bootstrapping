from __future__ import annotations

import types
import unittest
from unittest import mock

from rl_vla_bootstrapping.policy.openvla_actor_critic import (
    iter_model_candidates,
    load_generate_config,
    prepare_prompt,
    resolve_llm_dim,
    set_num_images_in_input,
)


class OpenVLAActorCriticHelpersTests(unittest.TestCase):
    def test_prepare_prompt_matches_openvla_instruction_format(self):
        prompt = prepare_prompt("Pick Up Apple")
        self.assertEqual(prompt, "In: What action should the robot take to pick up apple?\nOut:")

    def test_iter_model_candidates_walks_wrapped_models_once(self):
        leaf = types.SimpleNamespace(name="leaf")
        wrapped = types.SimpleNamespace(model=leaf)
        root = types.SimpleNamespace(base_model=wrapped, module=wrapped)

        names = [getattr(item, "name", type(item).__name__) for item in iter_model_candidates(root)]

        self.assertEqual(names.count("leaf"), 1)
        self.assertGreaterEqual(len(names), 3)

    def test_resolve_llm_dim_finds_text_config_hidden_size(self):
        vla = types.SimpleNamespace(
            base_model=types.SimpleNamespace(
                config=types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=4096))
            )
        )
        self.assertEqual(resolve_llm_dim(vla), 4096)

    def test_set_num_images_in_input_updates_backbone_when_available(self):
        backbone = types.SimpleNamespace(num_images_in_input=1)
        wrapped = types.SimpleNamespace(base_model=types.SimpleNamespace(vision_backbone=backbone))

        updated = set_num_images_in_input(wrapped, 2)

        self.assertEqual(updated, 2)
        self.assertEqual(backbone.num_images_in_input, 2)

    def test_load_generate_config_has_fallback_without_libero(self):
        with mock.patch(
            "rl_vla_bootstrapping.policy.openvla_actor_critic.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'libero'"),
        ):
            config_cls, note = load_generate_config()

        self.assertIsNotNone(config_cls)
        self.assertIsNotNone(note)
        self.assertIn("libero", note.lower())


if __name__ == "__main__":
    unittest.main()
