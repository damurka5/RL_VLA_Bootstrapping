from __future__ import annotations

import importlib
import sys
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock


def _load_generate_module():
    dummy_pkg = sys.modules.setdefault("cdpr_mujoco", types.ModuleType("cdpr_mujoco"))
    dummy_headless = types.ModuleType("cdpr_mujoco.headless_cdpr_egl")

    class DummySimulation:
        pass

    dummy_headless.HeadlessCDPRSimulation = DummySimulation
    sys.modules["cdpr_mujoco.headless_cdpr_egl"] = dummy_headless
    setattr(dummy_pkg, "headless_cdpr_egl", dummy_headless)
    return importlib.import_module("robots.cdpr.cdpr_dataset.generate_cdpr_dataset")


def _load_rl_env_module():
    _load_generate_module()
    return importlib.import_module("robots.cdpr.cdpr_dataset.rl_cdpr_env")


class WrapperBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_generate_module()
        cls.rl_env_mod = _load_rl_env_module()

    def test_list_wrapper_bundle_paths_tracks_local_helper_xmls(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrapper = root / "desk__apple_wrapper.xml"
            helper_a = root / "desk_zshift.xml"
            helper_b = root / "cdpr_ee_override.xml"
            helper_c = root / "placed_0_apple.xml"

            helper_a.write_text("<mujoco/>", encoding="utf-8")
            helper_b.write_text("<mujoco/>", encoding="utf-8")
            helper_c.write_text("<mujoco/>", encoding="utf-8")
            wrapper.write_text(
                (
                    "<mujoco>"
                    '<include file="desk_zshift.xml"/>'
                    '<include file="cdpr_ee_override.xml"/>'
                    '<include file="placed_0_apple.xml"/>'
                    "</mujoco>"
                ),
                encoding="utf-8",
            )

            bundle_paths = self.mod.list_wrapper_bundle_paths(wrapper)

            self.assertEqual(
                bundle_paths,
                [wrapper.resolve(), helper_a.resolve(), helper_b.resolve(), helper_c.resolve()],
            )

    def test_isolate_wrapper_bundle_prefixes_helper_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrapper = root / "desk__apple_wrapper.xml"
            helper_a = root / "desk_zshift.xml"
            helper_b = root / "cdpr_ee_override.xml"

            helper_a.write_text("<mujoco/>", encoding="utf-8")
            helper_b.write_text("<mujoco/>", encoding="utf-8")
            wrapper.write_text(
                (
                    "<mujoco>"
                    '<include file="desk_zshift.xml"/>'
                    '<include file="cdpr_ee_override.xml"/>'
                    "</mujoco>"
                ),
                encoding="utf-8",
            )

            created = self.mod._isolate_wrapper_bundle(wrapper)

            self.assertEqual(
                [path.name for path in created],
                [
                    "desk__apple_wrapper__desk_zshift.xml",
                    "desk__apple_wrapper__cdpr_ee_override.xml",
                ],
            )
            for path in created:
                self.assertTrue(path.exists(), path)

            include_files = [
                inc.get("file")
                for inc in ET.parse(wrapper).getroot().iter("include")
            ]
            self.assertEqual(
                include_files,
                [
                    "desk__apple_wrapper__desk_zshift.xml",
                    "desk__apple_wrapper__cdpr_ee_override.xml",
                ],
            )
            self.assertTrue(self.mod._wrapper_bundle_isolated(wrapper))

    def test_build_wrapper_uses_scene_switcher_script_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrapper = root / "preview_scene.xml"
            issued_cmd: list[str] = []

            def fake_run(cmd, *args, **kwargs):
                issued_cmd[:] = list(cmd)
                wrapper.write_text("<mujoco/>", encoding="utf-8")
                return SimpleNamespace(returncode=0)

            with mock.patch.object(self.mod.subprocess, "run", side_effect=fake_run):
                out = self.mod.build_wrapper_if_needed(
                    scene_name="desk",
                    object_names=["ycb_apple"],
                    wrapper_out=wrapper,
                    use_cache=False,
                )

            self.assertEqual(out, wrapper.resolve())
            self.assertEqual(issued_cmd[0], sys.executable)
            self.assertEqual(Path(issued_cmd[1]).resolve(), self.mod.SCENE_SWITCHER.resolve())
            self.assertNotIn("-m", issued_cmd)
            self.assertTrue(wrapper.exists())

    def test_import_wrapper_builder_supports_call_and_unpack(self):
        handle = self.rl_env_mod._import_wrapper_builder()

        self.assertTrue(callable(handle))

        build_wrapper_if_needed, list_wrapper_bundle_paths = handle
        self.assertIs(build_wrapper_if_needed, self.mod.build_wrapper_if_needed)
        self.assertIs(list_wrapper_bundle_paths, self.mod.list_wrapper_bundle_paths)


if __name__ == "__main__":
    unittest.main()
