from __future__ import annotations

import importlib
import sys
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_generate_module():
    dummy_pkg = sys.modules.setdefault("cdpr_mujoco", types.ModuleType("cdpr_mujoco"))
    dummy_headless = types.ModuleType("cdpr_mujoco.headless_cdpr_egl")

    class DummySimulation:
        pass

    dummy_headless.HeadlessCDPRSimulation = DummySimulation
    sys.modules["cdpr_mujoco.headless_cdpr_egl"] = dummy_headless
    setattr(dummy_pkg, "headless_cdpr_egl", dummy_headless)
    return importlib.import_module("robots.cdpr.cdpr_dataset.generate_cdpr_dataset")


class WrapperBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_generate_module()

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


if __name__ == "__main__":
    unittest.main()
