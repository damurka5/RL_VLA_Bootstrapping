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

import numpy as np


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

    def test_scene_switcher_command_embeds_negative_ee_start_in_same_arg(self):
        cmd = self.mod._scene_switcher_command(
            scene_name="desk",
            scene_z=-0.85,
            ee_start=np.array([-0.12, 0.04, 0.40], dtype=float),
            table_z=0.15,
            settle_time=0.0,
            wrapper_path=Path("/tmp/preview_scene.xml"),
        )

        self.assertIn("--ee_start=-0.12,0.04,0.4", cmd)

    def test_import_wrapper_builder_supports_call_and_unpack(self):
        handle = self.rl_env_mod._import_wrapper_builder()

        self.assertTrue(callable(handle))

        build_wrapper_if_needed, list_wrapper_bundle_paths = handle
        self.assertIs(build_wrapper_if_needed, self.mod.build_wrapper_if_needed)
        self.assertIs(list_wrapper_bundle_paths, self.mod.list_wrapper_bundle_paths)

    def test_rl_env_build_wrapper_uses_unique_temp_bundle_for_randomized_ee_start(self):
        scene = self.rl_env_mod.SceneSpec(name="desk", objects=("ycb_apple",))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            issued: dict[str, object] = {}

            def fake_builder(**kwargs):
                issued.update(kwargs)
                wrapper_path = Path(kwargs["wrapper_out"]).resolve()
                wrapper_path.write_text("<mujoco/>", encoding="utf-8")
                return wrapper_path

            env = self.rl_env_mod.CDPRLanguageRLEnv.__new__(self.rl_env_mod.CDPRLanguageRLEnv)
            env.defaults = {
                "scene_z": -0.85,
                "table_z": 0.15,
                "settle_time": 0.0,
                "ee_start": [0.0, 0.0, 0.40],
            }
            env.randomize_ee_start = True
            env.ee_start_x_bounds = (-0.10, 0.10)
            env.ee_start_y_bounds = (-0.08, 0.05)
            env.ee_start_z = None
            env.use_wrapper_cache = True
            env.wrapper_cleanup = False
            env.wrapper_dir = root
            env.desk_texture_files = []
            env._cleanup_paths = []
            env._cleanup_path_set = set()
            env.np_random = np.random.default_rng(11)

            with mock.patch.object(
                self.rl_env_mod,
                "_import_wrapper_builder",
                return_value=self.rl_env_mod.WrapperBuilderHandle(
                    build_wrapper_if_needed=fake_builder,
                    list_wrapper_bundle_paths=lambda path: [Path(path).resolve()],
                ),
            ):
                ee_start = env._sample_episode_ee_start()
                wrapper_xml = env._build_wrapper(scene=scene, ee_start=ee_start)

            self.assertTrue(wrapper_xml.exists())
            self.assertFalse(bool(issued["use_cache"]))
            self.assertEqual(Path(issued["wrapper_out"]).parent, root)
            self.assertIn(wrapper_xml.resolve(), env._cleanup_paths)
            sampled = np.asarray(issued["ee_start"], dtype=np.float32)
            self.assertGreaterEqual(float(sampled[0]), -0.10)
            self.assertLessEqual(float(sampled[0]), 0.10)
            self.assertGreaterEqual(float(sampled[1]), -0.08)
            self.assertLessEqual(float(sampled[1]), 0.05)
            self.assertAlmostEqual(float(sampled[2]), 0.40, places=6)

    def test_rl_env_can_override_episode_ee_start_via_reset_options(self):
        env = self.rl_env_mod.CDPRLanguageRLEnv.__new__(self.rl_env_mod.CDPRLanguageRLEnv)
        env.defaults = {"ee_start": [0.0, 0.0, 0.40]}
        env.randomize_ee_start = False
        env.ee_start_x_bounds = (-0.12, 0.12)
        env.ee_start_y_bounds = (-0.12, 0.12)
        env.ee_start_z = None
        env.np_random = np.random.default_rng(0)

        ee_start = env._sample_episode_ee_start(options={"ee_start": [0.07, -0.03, 0.18]})

        np.testing.assert_allclose(ee_start, np.array([0.07, -0.03, 0.40], dtype=np.float64), atol=1e-9)

    def test_rl_env_same_reset_seed_advances_episode_rng(self):
        env = self.rl_env_mod.CDPRLanguageRLEnv.__new__(self.rl_env_mod.CDPRLanguageRLEnv)
        env.np_random = np.random.default_rng(0)
        env._reset_counter = 0
        env._episode_index = -1

        env._prepare_episode_rng(123)
        first = float(env.np_random.uniform())
        self.assertEqual(env._episode_index, 0)

        env._prepare_episode_rng(123)
        second = float(env.np_random.uniform())
        self.assertEqual(env._episode_index, 1)

        self.assertNotEqual(first, second)

    def test_rl_env_build_episode_wrapper_falls_back_when_cached_wrapper_lacks_ee_start_kwarg(self):
        env = self.rl_env_mod.CDPRLanguageRLEnv.__new__(self.rl_env_mod.CDPRLanguageRLEnv)
        scene = self.rl_env_mod.SceneSpec(name="desk", objects=("ycb_apple",))

        def cached_wrapper(this, scene):
            return Path("/tmp/cached_scene.xml")

        env._build_wrapper = types.MethodType(cached_wrapper, env)

        out = env._build_episode_wrapper(scene=scene, ee_start=np.array([0.1, -0.1, 0.4], dtype=np.float32))

        self.assertEqual(out, Path("/tmp/cached_scene.xml").resolve())

    def test_rl_env_move_episode_start_applies_sampled_xy_even_without_wrapper_override(self):
        env = self.rl_env_mod.CDPRLanguageRLEnv.__new__(self.rl_env_mod.CDPRLanguageRLEnv)

        class FakeSim:
            def __init__(self):
                self.target = None
                self.goto_calls = []
                self.hold_calls = []

            def set_target_position(self, xyz):
                self.target = np.asarray(xyz, dtype=np.float32).copy()

            def goto(self, target, max_steps=120, tol=0.01):
                self.goto_calls.append((np.asarray(target, dtype=np.float32).copy(), int(max_steps), float(tol)))

            def hold_current_pose(self, warm_steps=0):
                self.hold_calls.append(int(warm_steps))

        env.sim = FakeSim()
        env._episode_ee_start = np.array([0.17, -0.11, 0.40], dtype=np.float32)
        env._ee_spawn_z = 0.236
        env._ee_min_z = 0.12
        env._locked_target_xyz = np.zeros((3,), dtype=np.float32)

        env._move_ee_to_episode_start()

        np.testing.assert_allclose(env.sim.target, np.array([0.17, -0.11, 0.40], dtype=np.float32), atol=1e-7)
        np.testing.assert_allclose(env.sim.goto_calls[0][0], np.array([0.17, -0.11, 0.40], dtype=np.float32), atol=1e-7)
        self.assertEqual(env.sim.hold_calls, [6])
        np.testing.assert_allclose(env._locked_target_xyz, np.array([0.17, -0.11, 0.40], dtype=np.float32), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
