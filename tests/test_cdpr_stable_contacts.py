from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco as mj
except Exception as exc:  # pragma: no cover - optional runtime dependency
    mj = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None

from robots.cdpr.cdpr_mujoco import cdpr_scene_switcher as switcher
from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv


ROOT = Path(__file__).resolve().parents[1]
CDPR_XML = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr.xml"
STABLE_OBJECTS = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "stable_objects"


def _require_mujoco(testcase: unittest.TestCase) -> None:
    if mj is None:
        testcase.skipTest(f"MuJoCo unavailable: {MUJOCO_IMPORT_ERROR}")


def _geom_id(model: mj.MjModel, name: str) -> int:
    geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
    if geom_id == -1:
        raise AssertionError(f"Missing geom: {name}")
    return int(geom_id)


def _pair_names(model: mj.MjModel) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for idx in range(int(model.npair)):
        geom1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[idx]))
        geom2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[idx]))
        out.add((str(geom1), str(geom2)))
    return out


class CDPRStableContactTests(unittest.TestCase):
    def test_apple_and_pear_assets_load_with_visual_collision_split(self):
        _require_mujoco(self)

        expected_collision_counts = {"ycb_apple": 1, "ycb_pear": 3}
        for object_name, expected_count in expected_collision_counts.items():
            with self.subTest(object_name=object_name):
                model = mj.MjModel.from_xml_path(str(STABLE_OBJECTS / f"{object_name}.xml"))
                collision_names = [
                    mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
                    for gid in range(model.ngeom)
                    if "collision" in str(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid))
                ]
                visual_names = [
                    mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
                    for gid in range(model.ngeom)
                    if "visual" in str(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid))
                ]

                self.assertEqual(len(collision_names), expected_count)
                self.assertGreaterEqual(len(visual_names), 1)
                self.assertGreater(float(model.body_mass[1]), 0.0)
                self.assertTrue(np.all(np.asarray(model.body_inertia[1]) > 0.0))

                for name in collision_names:
                    gid = _geom_id(model, str(name))
                    self.assertNotEqual(int(model.geom_contype[gid]), 0)
                    self.assertNotEqual(int(model.geom_conaffinity[gid]), 0)
                    self.assertEqual(int(model.geom_group[gid]), 3)
                for name in visual_names:
                    gid = _geom_id(model, str(name))
                    self.assertEqual(int(model.geom_contype[gid]), 0)
                    self.assertEqual(int(model.geom_conaffinity[gid]), 0)
                    self.assertEqual(int(model.geom_group[gid]), 1)

    def test_fingertip_pad_geoms_exist(self):
        _require_mujoco(self)

        model = mj.MjModel.from_xml_path(str(CDPR_XML))
        for name in ("left_finger_pad", "right_finger_pad"):
            gid = _geom_id(model, name)
            self.assertEqual(int(model.geom_group[gid]), 3)
            self.assertEqual(int(model.geom_condim[gid]), 4)
            self.assertGreater(float(model.geom_size[gid][0]), 0.0)

    def test_caught_object_grasp_center_uses_collision_pads_not_visual_tips(self):
        _require_mujoco(self)

        model = mj.MjModel.from_xml_path(str(CDPR_XML))
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.sim = type("Sim", (), {"model": model, "data": data})()

        geometry = env._finger_pair_geometry()

        self.assertIsNotNone(geometry)
        left_pad = _geom_id(model, "left_finger_pad")
        right_pad = _geom_id(model, "right_finger_pad")
        expected_center = 0.5 * (
            np.asarray(data.geom_xpos[left_pad], dtype=np.float32)
            + np.asarray(data.geom_xpos[right_pad], dtype=np.float32)
        )
        np.testing.assert_allclose(geometry["center"], expected_center, atol=1e-7)

        left_tip = _geom_id(model, "finger_l_tip")
        right_tip = _geom_id(model, "finger_r_tip")
        visual_tip_center = 0.5 * (
            np.asarray(data.geom_xpos[left_tip], dtype=np.float32)
            + np.asarray(data.geom_xpos[right_tip], dtype=np.float32)
        )
        self.assertGreater(float(np.linalg.norm(expected_center - visual_tip_center)), 0.01)

    def test_pinned_caught_start_is_reported_closed_until_release_target(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "caught_object_start_pin_object": True,
            "caught_object_start_release_opening_margin": 0.04,
        }
        env._caught_object_start_active = True
        env._caught_object_start_gripper_opening = 0.852
        env.sim = type(
            "Sim",
            (),
            {
                "gripper_min": 0.0,
                "gripper_max": 1.0,
                "get_gripper_target": lambda self: 0.852,
            },
        )()

        # The physical opening can lag the fitted target without invalidating
        # the environment's still-active pin constraint.
        self.assertTrue(env._is_gripper_closed(0.909))

        env.sim.get_gripper_target = lambda: 0.95
        self.assertFalse(env._is_gripper_closed(0.95))

    def test_pinned_caught_start_is_authoritative_for_catch_detection(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {"caught_object_start_pin_object": True}
        env._caught_object_start_active = True
        env._caught_object_start_body = "p1_ycb_apple"
        env._caught_object_start_catalog = "ycb_apple"
        env._caught_object_start_gripper_opening = 0.85
        env._object_body_names = ["p1_ycb_apple"]
        env._target_catalog_name = "ycb_apple"
        env._inverse_catalog_to_body = {"p1_ycb_apple": "ycb_apple"}
        env._reward_state = type("RewardState", (), {"gripper_closed": True})()
        env.sim = type("Sim", (), {"get_gripper_target": lambda self: 0.85})()

        body, catalog, score, is_target = env._detect_caught_object(np.zeros(3, dtype=np.float32))

        self.assertEqual(body, "p1_ycb_apple")
        self.assertEqual(catalog, "ycb_apple")
        self.assertEqual(score, 1.0)
        self.assertTrue(is_target)

    def test_stable_contact_wrapper_inserts_expected_pairs(self):
        _require_mujoco(self)

        with tempfile.TemporaryDirectory(prefix="cdpr_contact_test_") as tmp:
            tmpdir = Path(tmp)
            scene_xml = tmpdir / "table_scene.xml"
            scene_xml.write_text(
                """<mujoco model="table_scene">
  <worldbody>
    <body name="unit_table" pos="0 0 -0.015">
      <geom name="unit_table_top" class="table_collision" type="box" size="0.45 0.35 0.015"/>
    </body>
  </worldbody>
</mujoco>
""",
                encoding="utf-8",
            )
            placed_xmls = []
            for idx, object_name in enumerate(("ycb_apple", "ycb_pear")):
                placed_xml = tmpdir / f"placed_{object_name}.xml"
                switcher.make_placed_object_xml(
                    STABLE_OBJECTS / f"{object_name}.xml",
                    placed_xml,
                    prefix=f"p{idx}",
                    pos=np.array([0.05 * idx, 0.0, 0.10], dtype=float),
                    quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
                    force_dynamic=True,
                    logical_name=object_name,
                )
                placed_xmls.append(placed_xml)

            wrapper_xml = tmpdir / "wrapper.xml"
            switcher.build_wrapper_mjcf(
                scene_xml,
                CDPR_XML,
                placed_xmls,
                wrapper_xml,
                contact_preset="stable_contact",
                table_geom_names=("unit_table_top",),
            )
            model = mj.MjModel.from_xml_path(str(wrapper_xml))
            pairs = _pair_names(model)

            self.assertIn(("left_finger_pad", "p0_ycb_apple_collision"), pairs)
            self.assertIn(("right_finger_pad", "p0_ycb_apple_collision"), pairs)
            self.assertIn(("unit_table_top", "p0_ycb_apple_collision"), pairs)
            self.assertIn(("left_finger_pad", "p1_ycb_pear_collision_lower"), pairs)
            self.assertIn(("right_finger_pad", "p1_ycb_pear_collision_mid"), pairs)
            self.assertIn(("unit_table_top", "p1_ycb_pear_collision_neck"), pairs)
            self.assertEqual(float(model.opt.timestep), 0.002)
            self.assertEqual(int(model.opt.iterations), 100)

    def test_short_stable_contact_simulation_has_no_nans(self):
        _require_mujoco(self)

        with tempfile.TemporaryDirectory(prefix="cdpr_contact_smoke_") as tmp:
            tmpdir = Path(tmp)
            scene_xml = tmpdir / "table_scene.xml"
            scene_xml.write_text(
                """<mujoco model="table_scene">
  <worldbody>
    <body name="unit_table" pos="0 0 -0.015">
      <geom name="unit_table_top" class="table_collision" type="box" size="0.45 0.35 0.015"/>
    </body>
  </worldbody>
</mujoco>
""",
                encoding="utf-8",
            )
            placed_xml = tmpdir / "placed_ycb_apple.xml"
            switcher.make_placed_object_xml(
                STABLE_OBJECTS / "ycb_apple.xml",
                placed_xml,
                prefix="p0",
                pos=np.array([0.0, 0.0, 0.08], dtype=float),
                quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
                force_dynamic=True,
                logical_name="ycb_apple",
            )
            wrapper_xml = tmpdir / "wrapper.xml"
            switcher.build_wrapper_mjcf(
                scene_xml,
                CDPR_XML,
                [placed_xml],
                wrapper_xml,
                contact_preset="stable_contact",
                table_geom_names=("unit_table_top",),
            )
            model = mj.MjModel.from_xml_path(str(wrapper_xml))
            data = mj.MjData(model)
            for _ in range(50):
                mj.mj_step(model, data)
                self.assertTrue(np.all(np.isfinite(data.qpos)))
                self.assertTrue(np.all(np.isfinite(data.qvel)))


if __name__ == "__main__":
    unittest.main()
