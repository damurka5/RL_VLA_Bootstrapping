from __future__ import annotations

import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


class CDPRCompiledModelCacheTests(unittest.TestCase):
    def _load_module(self):
        try:
            from robots.cdpr.cdpr_mujoco import model_cache
        except Exception as exc:
            self.skipTest(f"MuJoCo model cache module is unavailable: {exc}")
        return model_cache

    def test_compiled_model_cache_honors_lru_max_size(self):
        model_cache = self._load_module()
        model_cache.clear_cache()

        class FakeModel:
            def __init__(self, path: str):
                self.path = str(path)
                self.opt = types.SimpleNamespace(timestep=0.0)

        def fake_from_xml_path(path: str):
            return FakeModel(path)

        with TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "scene.xml"
            xml_path.write_text("<mujoco/>", encoding="utf-8")

            with mock.patch.dict("os.environ", {"RLVLA_CDPR_COMPILED_MODEL_CACHE_MAX_SIZE": "2"}), mock.patch.object(
                model_cache.mj,
                "MjModel",
                types.SimpleNamespace(from_xml_path=fake_from_xml_path),
            ):
                for variant in range(3):
                    model_cache.get_compiled_model(
                        xml_path,
                        enabled=True,
                        timestep=1.0 / 60.0,
                        offscreen_width=64,
                        offscreen_height=64,
                        offscreen_samples="1",
                        semantic_key={"variant": variant},
                    )

                stats = model_cache.cache_stats()
                self.assertEqual(stats["size"], 2)
                self.assertEqual(stats["max_size"], 2)
                self.assertEqual(stats["evictions"], 1)

                _model, event = model_cache.get_compiled_model(
                    xml_path,
                    enabled=True,
                    timestep=1.0 / 60.0,
                    offscreen_width=64,
                    offscreen_height=64,
                    offscreen_samples="1",
                    semantic_key={"variant": 2},
                )

                self.assertTrue(event.hit)
                self.assertEqual(model_cache.cache_stats()["hits"], 1)

        model_cache.clear_cache()

    def test_disabled_compiled_model_cache_does_not_store_models(self):
        model_cache = self._load_module()
        model_cache.clear_cache()

        class FakeModel:
            def __init__(self, path: str):
                self.path = str(path)
                self.opt = types.SimpleNamespace(timestep=0.0)

        with TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "scene.xml"
            xml_path.write_text("<mujoco/>", encoding="utf-8")

            with mock.patch.object(
                model_cache.mj,
                "MjModel",
                types.SimpleNamespace(from_xml_path=lambda path: FakeModel(path)),
            ):
                _model, event = model_cache.get_compiled_model(
                    xml_path,
                    enabled=False,
                    timestep=1.0 / 60.0,
                    offscreen_width=64,
                    offscreen_height=64,
                    offscreen_samples="1",
                    semantic_key={"variant": "disabled"},
                )

        self.assertFalse(event.enabled)
        self.assertEqual(event.reason, "disabled")
        self.assertEqual(model_cache.cache_stats()["size"], 0)
        model_cache.clear_cache()


if __name__ == "__main__":
    unittest.main()
