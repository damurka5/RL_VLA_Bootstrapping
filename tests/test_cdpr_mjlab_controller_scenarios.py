from __future__ import annotations

import importlib.util
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.render_cdpr_mjlab_camera_videos import (
    CAUGHT_GRIPPER_OPENINGS,
    CAUGHT_OBJECT_OFFSET_Z,
    DEFAULT_XML,
    SCENARIOS,
    MuJoCoReferenceRunner,
    _phase_complete,
    _policy_action,
    _scenario_metrics,
    _telemetry_lines,
)


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_XML = (
    ROOT
    / "robots"
    / "cdpr"
    / "cdpr_mujoco"
    / "cdpr_mjwarp_smoke.xml"
)


class _NullRenderer:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class CDPRMJLabControllerScenarioTests(unittest.TestCase):
    def test_mjwarp_scene_has_dark_skybox_textured_plane_and_no_wall(self):
        root = ET.parse(WRAPPER_XML).getroot()
        skybox = root.find("./asset/texture[@name='mjwarp_skybox']")
        self.assertIsNotNone(skybox)
        self.assertEqual(skybox.get("type"), "skybox")
        self.assertEqual(skybox.get("builtin"), "gradient")

        visual = root.find(
            "./worldbody/geom[@name='mjwarp_desk_surface_visual']"
        )
        physical = root.find(
            "./worldbody/geom[@name='mjwarp_desk_surface']"
        )
        self.assertIsNotNone(visual)
        self.assertIsNotNone(physical)
        self.assertEqual(visual.get("type"), "plane")
        self.assertEqual(visual.get("group"), "2")
        self.assertEqual(physical.get("type"), "box")
        self.assertEqual(physical.get("group"), "3")
        self.assertIsNone(
            root.find("./worldbody/geom[@name='mjwarp_background']")
        )
        for index in range(1, 5):
            cable = root.find(
                f"./worldbody/geom[@name='mjwarp_cable_visual_{index}']"
            )
            self.assertIsNotNone(cable)
            self.assertEqual(cable.get("type"), "capsule")
            self.assertEqual(cable.get("group"), "4")
            self.assertAlmostEqual(float(cable.get("size")), 0.002)

        for index in range(7):
            material = root.find(
                f"./asset/material[@name='mjwarp_desk_mat_{index}']"
            )
            self.assertIsNotNone(material)
            self.assertEqual(material.get("texuniform"), "true")

        backend_source = (
            ROOT
            / "rl_vla_bootstrapping"
            / "simulation"
            / "mjlab_mjwarp_backend.py"
        ).read_text(encoding="utf-8")
        self.assertIn("render_skybox=True", backend_source)
        self.assertIn(
            "background_color=(0.035, 0.050, 0.075, 1.0)",
            backend_source,
        )
        self.assertIn("enabled_geom_groups=[0, 1, 2, 4]", backend_source)

    @unittest.skipUnless(
        importlib.util.find_spec("mujoco") is not None,
        "MuJoCo is required to inspect compiled actuator parameters.",
    )
    def test_compiled_controller_gains_and_cable_width_match_2ms_timestep(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(WRAPPER_XML))
        self.assertAlmostEqual(float(np.min(model.tendon_width)), 0.002)
        for name in (
            "slider_1_pos",
            "slider_2_pos",
            "slider_3_pos",
            "slider_4_pos",
        ):
            actuator = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            self.assertAlmostEqual(
                float(model.actuator_gainprm[actuator, 0]), 6000.0
            )
            self.assertAlmostEqual(
                float(model.actuator_biasprm[actuator, 2]), -100.0
            )
        gripper = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_gripper"
        )
        self.assertAlmostEqual(
            float(model.actuator_gainprm[gripper, 0]), 5.0
        )
        self.assertAlmostEqual(
            float(model.actuator_biasprm[gripper, 2]), -0.1
        )

    def test_training_resets_center_caught_apple_and_use_fine_release_actions(self):
        self.assertAlmostEqual(CAUGHT_GRIPPER_OPENINGS["robocasa_apple"], 0.46)
        for name in (
            "training_put_into_bowl",
            "training_put_on_plate",
        ):
            scenario = SCENARIOS[name]
            initial_object = np.asarray(
                scenario.object_positions[0], dtype=np.float64
            )
            ee = np.asarray(scenario.ee_start, dtype=np.float64)
            np.testing.assert_allclose(initial_object[:2], ee[:2])
            self.assertAlmostEqual(
                float(initial_object[2] - ee[2]),
                CAUGHT_OBJECT_OFFSET_Z,
            )
            self.assertAlmostEqual(scenario.gripper_opening, 0.46)
            held_phases = [
                phase for phase in scenario.phases if phase.name != "release"
            ][:3]
            self.assertTrue(
                all(phase.target_gripper == 0.46 for phase in held_phases)
            )
            release = next(
                phase
                for phase in scenario.phases
                if phase.name == "release"
            )
            self.assertLessEqual(release.translation_action_limit, 0.10)
            self.assertGreaterEqual(release.minimum_steps, 18)

    def test_video_telemetry_includes_vla_action_controller_and_cables(self):
        scenario = SCENARIOS["training_put_into_bowl"]
        state = {
            "ee_position": np.array([0.10, 0.08, 0.35]),
            "target_position": np.array([0.12, 0.09, 0.36]),
            "object_position": np.array([0.10, 0.08, 0.34]),
            "reference_position": np.array([0.12, 0.09, 0.15]),
            "tendon_lengths": np.full(4, 4.23),
            "gripper_opening": 0.46,
            "gripper_target": 0.46,
        }
        text = "\n".join(
            _telemetry_lines(
                camera_label="overview",
                scenario=scenario,
                phase="carry_at_safe_height",
                step=7,
                action=np.array([0.5, -0.2, 0.1, 0.0, -0.3]),
                state=state,
            )
        )
        self.assertIn("VLA-like normalized action", text)
        self.assertIn("executed delta", text)
        self.assertIn("controller_target", text)
        self.assertIn("gripper=", text)
        self.assertIn("receptacle_xy_error", text)
        self.assertIn("cable_lengths_m", text)

    @unittest.skipUnless(
        importlib.util.find_spec("mujoco") is not None,
        "MuJoCo is required for the deterministic controller regression.",
    )
    def test_controller_reaches_and_completes_bowl_and_plate_scenarios(self):
        import mujoco

        with mock.patch.object(mujoco, "Renderer", _NullRenderer):
            runner = MuJoCoReferenceRunner(
                xml_path=DEFAULT_XML,
                width=32,
                height=32,
            )
            try:
                for name in (
                    "training_put_into_bowl",
                    "training_put_on_plate",
                ):
                    with self.subTest(scenario=name):
                        scenario = SCENARIOS[name]
                        runner.reset(scenario)
                        state = runner.observe()
                        initial_offset = (
                            np.asarray(
                                state["object_position"], dtype=np.float64
                            )
                            - np.asarray(
                                state["ee_position"], dtype=np.float64
                            )
                        )
                        held_openings: list[float] = []
                        held_slips: list[float] = []
                        for phase in scenario.phases:
                            completed = False
                            for used in range(
                                1, int(phase.max_steps) + 1
                            ):
                                state = runner.step(
                                    _policy_action(state, phase)
                                )
                                if phase.target_gripper < 0.55:
                                    held_openings.append(
                                        float(state["gripper_opening"])
                                    )
                                    held_slips.append(
                                        float(
                                            np.linalg.norm(
                                                np.asarray(
                                                    state[
                                                        "object_position"
                                                    ],
                                                    dtype=np.float64,
                                                )
                                                - np.asarray(
                                                    state["ee_position"],
                                                    dtype=np.float64,
                                                )
                                                - initial_offset
                                            )
                                        )
                                    )
                                if _phase_complete(state, phase, used):
                                    completed = True
                                    break
                            self.assertTrue(
                                completed,
                                f"{name}/{phase.name} failed to reach target",
                            )

                        metrics = _scenario_metrics(scenario, state)
                        self.assertTrue(metrics["success"], metrics)
                        self.assertLessEqual(
                            metrics["xy_error"], 0.03
                        )
                        self.assertLessEqual(max(held_slips), 0.03)
                        settled = np.asarray(
                            held_openings[5:], dtype=np.float64
                        )
                        self.assertLessEqual(
                            float(np.max(np.abs(np.diff(settled)))),
                            0.02,
                        )
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
