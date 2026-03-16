from __future__ import annotations

import types
import unittest
from unittest import mock

from rl_vla_bootstrapping.cli.run_cdpr_policy import _FallbackGenerateConfig, _load_generate_config


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


if __name__ == "__main__":
    unittest.main()
