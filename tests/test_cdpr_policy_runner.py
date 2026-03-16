from __future__ import annotations

import types
import unittest
from unittest import mock

from rl_vla_bootstrapping.cli.run_cdpr_policy import (
    _FallbackGenerateConfig,
    _load_generate_config,
    _resolve_llm_dim,
    _set_num_images_in_input,
)


class PolicyRunnerConfigTests(unittest.TestCase):
    def test_load_generate_config_falls_back_without_libero(self):
        with mock.patch(
            "rl_vla_bootstrapping.cli.run_cdpr_policy.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'libero'"),
        ):
            config_cls, note = _load_generate_config()

        self.assertIs(config_cls, _FallbackGenerateConfig)
        self.assertIsNotNone(note)
        self.assertIn("libero", note)

    def test_load_generate_config_prefers_openvla_definition_when_available(self):
        fake_module = types.SimpleNamespace(GenerateConfig=object)
        with mock.patch(
            "rl_vla_bootstrapping.cli.run_cdpr_policy.importlib.import_module",
            return_value=fake_module,
        ):
            config_cls, note = _load_generate_config()

        self.assertIs(config_cls, object)
        self.assertIsNone(note)

    def test_resolve_llm_dim_from_wrapped_text_config(self):
        leaf = types.SimpleNamespace(
            config=types.SimpleNamespace(
                text_config=types.SimpleNamespace(hidden_size=4096),
            )
        )
        wrapped = types.SimpleNamespace(base_model=leaf)

        self.assertEqual(_resolve_llm_dim(wrapped), 4096)

    def test_set_num_images_in_input_updates_wrapped_backbone(self):
        backbone = types.SimpleNamespace(num_images_in_input=1)
        wrapped = types.SimpleNamespace(base_model=types.SimpleNamespace(vision_backbone=backbone))

        updated = _set_num_images_in_input(wrapped, 2)

        self.assertEqual(updated, 2)
        self.assertEqual(backbone.num_images_in_input, 2)

    def test_set_num_images_in_input_falls_back_to_one_without_api(self):
        updated = _set_num_images_in_input(types.SimpleNamespace(), 2)
        self.assertEqual(updated, 1)


if __name__ == "__main__":
    unittest.main()
