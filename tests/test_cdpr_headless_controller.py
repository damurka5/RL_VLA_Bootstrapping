from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import (
    HeadlessCDPRSimulation,
    HeadlessCDPRController,
    _solve_slider_preload_targets,
)


class HeadlessCDPRControllerTests(unittest.TestCase):
    def test_solve_slider_preload_targets_uses_tendon_upper_limit_residual(self):
        targets = _solve_slider_preload_targets(
            current_slider_qpos=np.array([0.0, -0.1, 0.2, -0.3], dtype=float),
            current_tendon_lengths=np.array([4.6, 4.2, 3.9, 4.1], dtype=float),
            tendon_upper_limits=np.array([4.235, 4.235, 4.235, 4.235], dtype=float),
            dlength_dq=np.array([1.0, 1.0, 2.0, -0.5], dtype=float),
        )

        np.testing.assert_allclose(
            targets,
            np.array([-0.365, -0.065, 0.3675, -0.57], dtype=float),
            atol=1e-9,
        )

    def test_inverse_kinematics_uses_attach_point_offset(self):
        frame_points = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (4, 1))
        controller = HeadlessCDPRController(
            frame_points,
            initial_pos=np.array([0.0, 0.0, 0.0], dtype=float),
            attach_point_offset=np.array([0.0, 0.0, 0.25], dtype=float),
        )

        lengths = controller.inverse_kinematics(np.array([0.0, 0.0, 0.0], dtype=float))

        np.testing.assert_allclose(lengths, np.full(4, 0.75, dtype=float))

    def test_compute_control_keeps_current_slider_target_when_pose_matches(self):
        frame_points = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (4, 1))
        controller = HeadlessCDPRController(frame_points, attach_point_offset=np.zeros(3, dtype=float))
        controller.configure_tendon_model(np.full(4, 2.0, dtype=float))

        current_q = np.array([0.1, -0.2, 0.3, -0.4], dtype=float)
        pose = np.array([0.0, 0.0, 0.0], dtype=float)

        control = controller.compute_control(
            pose,
            pose,
            current_slider_qpos=current_q,
            current_tendon_lengths=np.ones(4, dtype=float),
        )

        np.testing.assert_allclose(control, current_q)

    def test_compute_control_uses_geometric_delta_mapping(self):
        frame_points = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (4, 1))
        controller = HeadlessCDPRController(frame_points, attach_point_offset=np.zeros(3, dtype=float))
        controller.configure_tendon_model(np.full(4, 2.0, dtype=float))

        current_q = np.full(4, 0.1, dtype=float)
        current_pose = np.array([0.0, 0.0, 0.0], dtype=float)
        target_pose = np.array([0.0, 0.0, 0.2], dtype=float)

        control = controller.compute_control(
            target_pose,
            current_pose,
            current_slider_qpos=current_q,
            current_tendon_lengths=np.ones(4, dtype=float),
        )

        expected_delta = (1.0 - 0.8) / 2.0
        np.testing.assert_allclose(control, current_q + expected_delta)

    def test_simulation_set_yaw_clips_to_joint_limits(self):
        sim = HeadlessCDPRSimulation.__new__(HeadlessCDPRSimulation)
        sim.act_yaw = 2
        sim.yaw_min = -1.5
        sim.yaw_max = 1.5
        sim.data = type("Data", (), {"ctrl": np.zeros(3, dtype=float)})()

        sim.set_yaw(3.0)
        self.assertAlmostEqual(float(sim.data.ctrl[2]), 1.5, places=9)

        sim.set_yaw(-4.0)
        self.assertAlmostEqual(float(sim.data.ctrl[2]), -1.5, places=9)

    def test_set_slider_targets_clips_each_slider_to_actuator_range(self):
        sim = HeadlessCDPRSimulation.__new__(HeadlessCDPRSimulation)
        sim.act_sliders = [0, 1, 2, 3]
        sim.slider_qadr = [0, 1, 2, 3]
        sim.slider_joint_ids = [0, 1, 2, 3]
        sim.model = type(
            "Model",
            (),
            {
                "actuator_ctrllimited": np.array([1, 1, 1, 1], dtype=np.int32),
                "actuator_ctrlrange": np.array(
                    [[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0], [-4.0, 4.0]],
                    dtype=float,
                ),
                "jnt_dofadr": np.array([0, 1, 2, 3], dtype=np.int32),
                "nv": 4,
            },
        )()
        sim.data = type(
            "Data",
            (),
            {
                "ctrl": np.zeros(4, dtype=float),
                "qpos": np.zeros(4, dtype=float),
                "qvel": np.zeros(4, dtype=float),
            },
        )()

        clipped = sim._set_slider_targets(np.array([5.0, -5.0, 0.5, -0.5], dtype=float))

        np.testing.assert_allclose(clipped, np.array([1.0, -2.0, 0.5, -0.5], dtype=float), atol=1e-9)
        np.testing.assert_allclose(sim.data.ctrl, clipped, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
