"""WidowX-200 model, kinematics, controller, and scene layout.

The point of these tests is not that the code runs. It is that the numbers the
rest of the stack depends on are the ones the MODEL actually has. On the CDPR,
a grasp-height offset that was written down rather than measured cost `pick_up`
10M steps of training against a reward whose optimum the gripper could not
occupy (see `cdpr_gripper_geometry`). Every constant here is therefore checked
against MuJoCo's own forward kinematics and contact solver on the compiled MJCF,
so an edit to the XML, the gains, or the link geometry fails a test instead of
quietly producing a controller that tracks the wrong point.

The MuJoCo-dependent tests skip cleanly where MuJoCo is unavailable; the pure
algebra does not.
"""

from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

import numpy as np

from robots.widowx200.widowx200_mujoco import kinematics as K
from robots.widowx200.widowx200_mujoco import workspace as W
from robots.widowx200.widowx200_mujoco.controller import (
    WidowX200ControlSpec,
    WidowX200MountPose,
    WidowX200TaskSpaceController,
    integrate_task_targets,
    joint_targets_from_task_targets,
)

ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots/widowx200/widowx200_mujoco"
ROBOT_XML = ROBOT_DIR / "wx200.xml"
SCENE_XML = ROBOT_DIR / "wx200_scene.xml"

try:  # pragma: no cover - availability, not behaviour
    import mujoco as mj

    _MUJOCO = True
except Exception:  # pragma: no cover
    mj = None
    _MUJOCO = False

requires_mujoco = unittest.skipUnless(_MUJOCO, "MuJoCo is not installed.")


def _sample_targets(count: int, seed: int = 0) -> list[tuple[np.ndarray, float]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        out.append(
            (
                np.array(
                    [
                        rng.uniform(0.12, 0.36),
                        rng.uniform(-0.22, 0.22),
                        rng.uniform(0.02, 0.28),
                    ]
                ),
                float(rng.uniform(-math.pi, math.pi)),
            )
        )
    return out


class KinematicsAlgebraTests(unittest.TestCase):
    """The closed form, checked against itself and against the URDF numbers."""

    def test_link_geometry_matches_the_interbotix_urdf(self) -> None:
        # Transcribed from wx200.urdf.xacro. If a link length here drifts, the
        # IK still returns a pose -- just one for a different robot.
        self.assertAlmostEqual(K.WX200.shoulder_height, 0.11065, places=6)
        self.assertAlmostEqual(K.WX200.upper_arm_length, math.hypot(0.05, 0.20), places=9)
        self.assertAlmostEqual(K.WX200.forearm_length, 0.20, places=9)
        self.assertAlmostEqual(K.WX200.wrist_length, 0.158575, places=9)
        self.assertAlmostEqual(K.WX200.max_planar_reach, 0.4061553, places=6)

    def test_forward_kinematics_inverts_the_ik(self) -> None:
        for position, yaw in _sample_targets(300, seed=1):
            solution = K.top_down_ik(position, yaw, clamp_to_reach=False)
            if not bool(solution.reachable):
                continue
            pose = K.forward_kinematics(solution.q)
            np.testing.assert_allclose(pose["position"], position, atol=1e-9)
            self.assertAlmostEqual(
                math.atan2(
                    math.sin(float(pose["yaw"]) - yaw),
                    math.cos(float(pose["yaw"]) - yaw),
                ),
                0.0,
                places=9,
            )
            # The whole point of the top-down constraint.
            self.assertAlmostEqual(float(pose["pitch"]), K.TOP_DOWN_PITCH, places=9)

    def test_ik_is_total_and_reports_failure_rather_than_raising(self) -> None:
        # One unreachable world must never poison a batch of 128, so the
        # batched path needs a function that always returns finite joints.
        far = np.array([[2.0, 0.0, 0.1], [0.0, 0.0, 5.0], [0.0, 0.0, -3.0]])
        solution = K.top_down_ik(far, np.zeros(3))
        self.assertTrue(np.all(np.isfinite(np.asarray(solution.q))))
        self.assertFalse(bool(np.any(np.asarray(solution.reachable))))

    def test_joint_limits_are_respected(self) -> None:
        solution = K.top_down_ik(
            np.array([[0.30, 0.0, 0.05], [0.18, 0.10, 0.20]]), np.zeros(2)
        )
        q = np.asarray(solution.q)
        for index, name in enumerate(K.JOINT_NAMES):
            low, high = K.JOINT_LIMITS[name]
            self.assertTrue(np.all(q[:, index] >= low - 1e-9), name)
            self.assertTrue(np.all(q[:, index] <= high + 1e-9), name)

    def test_batched_and_scalar_solutions_agree(self) -> None:
        samples = _sample_targets(64, seed=2)
        positions = np.stack([p for p, _ in samples])
        yaws = np.array([y for _, y in samples])
        batched = np.asarray(K.top_down_ik(positions, yaws).q)
        for index, (position, yaw) in enumerate(samples):
            single = np.asarray(K.top_down_ik(position, yaw).q)
            np.testing.assert_allclose(batched[index], single, atol=1e-12)


class TorchParityTests(unittest.TestCase):
    """The batched controller runs the Torch branch; it must not diverge."""

    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:  # pragma: no cover
            self.skipTest("PyTorch is not installed.")

    def test_torch_matches_numpy(self) -> None:
        import torch

        samples = _sample_targets(48, seed=3)
        positions = np.stack([p for p, _ in samples])
        yaws = np.array([y for _, y in samples])
        reference = np.asarray(K.top_down_ik(positions, yaws).q)
        got = K.top_down_ik(
            torch.as_tensor(positions, dtype=torch.float32),
            torch.as_tensor(yaws, dtype=torch.float32),
        )
        np.testing.assert_allclose(
            np.asarray(got.q.detach().cpu()), reference, atol=2e-5
        )


class WorkspaceTests(unittest.TestCase):
    def test_layout_stays_inside_the_measured_reach(self) -> None:
        layout = W.DEFAULT_LAYOUT
        # Every height in the working band, from an object at rest on the desk
        # through carrying it, must reach past the outer spawn radius.
        for world_z in (
            layout.workspace_z[0], 0.19, 0.22, 0.24, layout.workspace_z[1]
        ):
            self.assertGreater(
                W.usable_radius(world_z),
                layout.spawn_radius[1],
                f"spawn sector is not reachable at z={world_z}",
            )

    def test_spawn_sector_is_inside_the_controller_workspace(self) -> None:
        layout = W.DEFAULT_LAYOUT
        (x_lo, x_hi), (y_lo, y_hi) = layout.sample_bounds()
        self.assertGreaterEqual(x_lo, layout.workspace_x[0] - 1e-9)
        self.assertLessEqual(x_hi, layout.workspace_x[1] + 1e-9)
        self.assertGreaterEqual(y_lo, layout.workspace_y[0] - 1e-9)
        self.assertLessEqual(y_hi, layout.workspace_y[1] + 1e-9)

    def test_sample_bounds_enclose_the_sector(self) -> None:
        layout = W.DEFAULT_LAYOUT
        (x_lo, x_hi), (y_lo, y_hi) = layout.sample_bounds()
        rng = np.random.default_rng(4)
        found = 0
        for _ in range(4000):
            x = rng.uniform(layout.workspace_x[0], layout.workspace_x[1])
            y = rng.uniform(layout.workspace_y[0], layout.workspace_y[1])
            if layout.contains(x, y):
                found += 1
                self.assertTrue(x_lo - 1e-9 <= x <= x_hi + 1e-9)
                self.assertTrue(y_lo - 1e-9 <= y <= y_hi + 1e-9)
        self.assertGreater(found, 100, "sector sampling found almost nothing")

    def test_keep_out_radius_excludes_the_arm_base(self) -> None:
        # The waist is singular on its own axis and the base plate occupies the
        # first ~5.5 cm; spawning inside either is a bug, not a hard task.
        self.assertGreater(W.DEFAULT_LAYOUT.spawn_radius[0], 0.10)
        self.assertGreaterEqual(
            W.DEFAULT_LAYOUT.spawn_radius[0], W.DEFAULT_LAYOUT.min_reach_radius
        )


class ControllerAlgebraTests(unittest.TestCase):
    def _spec(self, **kwargs) -> WidowX200ControlSpec:
        layout = W.DEFAULT_LAYOUT
        defaults = dict(
            mount=WidowX200MountPose(layout.base_position, layout.base_yaw),
            workspace_x=layout.workspace_x,
            workspace_y=layout.workspace_y,
            workspace_z=layout.workspace_z,
            min_reach_radius=layout.min_reach_radius,
        )
        defaults.update(kwargs)
        return WidowX200ControlSpec(**defaults)

    def test_mount_transform_round_trips(self) -> None:
        spec = self._spec()
        ops = K._ops_for(np.zeros(3))
        point = np.array([0.07, -0.03, 0.22])
        base = spec.mount.world_to_base(ops, point)
        np.testing.assert_allclose(spec.mount.base_to_world(ops, base), point, atol=1e-12)

    def test_target_is_clamped_to_the_world_workspace_first(self) -> None:
        spec = self._spec()
        result = joint_targets_from_task_targets(
            np.array([0.0, 0.0, 5.0]), np.float64(0.0), np.float64(1.0), spec
        )
        self.assertLessEqual(
            float(np.asarray(result["achieved_target"])[2]),
            spec.workspace_z[1] + 1e-9,
        )

    def test_leash_bounds_how_far_the_target_runs_ahead(self) -> None:
        spec = self._spec(target_leash=0.03)
        ee = np.array([0.0, 0.0, 0.25])
        target = ee.copy()
        for _ in range(50):
            target = np.asarray(
                integrate_task_targets(
                    np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                    ee,
                    target,
                    np.float64(0.0),
                    np.float64(1.0),
                    spec,
                )["target_position"]
            )
        # Without the leash this walks to 0.75 m ahead and a reversal would then
        # do nothing for dozens of steps.
        self.assertLessEqual(float(np.linalg.norm(target - ee)), 0.03 + 1e-9)

    def test_gripper_channel_maps_onto_the_finger_travel(self) -> None:
        spec = self._spec()
        low, high = K.FINGER_LIMITS
        for opening, expected in ((0.0, low), (1.0, high), (0.5, 0.5 * (low + high))):
            result = joint_targets_from_task_targets(
                np.array([0.0, -0.05, 0.22]),
                np.float64(0.0),
                np.float64(opening),
                spec,
            )
            self.assertAlmostEqual(float(result["gripper_ctrl"]), expected, places=9)


@requires_mujoco
class CompiledModelTests(unittest.TestCase):
    """Everything that only the compiled model can answer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mj.MjModel.from_xml_path(str(ROBOT_XML))
        cls.data = mj.MjData(cls.model)
        cls.qadr = {
            name: int(
                cls.model.jnt_qposadr[
                    mj.mj_name2id(cls.model, mj.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in K.JOINT_NAMES
        }
        cls.ee_body = mj.mj_name2id(cls.model, mj.mjtObj.mjOBJ_BODY, "ee_base")

    def _pose(self, q: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        for name, value in zip(K.JOINT_NAMES, np.asarray(q).reshape(5)):
            self.data.qpos[self.qadr[name]] = value
        mj.mj_forward(self.model, self.data)

    def test_ik_agrees_with_mujoco_forward_kinematics(self) -> None:
        """The load-bearing test: our algebra and MuJoCo's must be the same arm."""

        checked = 0
        for position, yaw in _sample_targets(400, seed=5):
            solution = K.top_down_ik(position, yaw, clamp_to_reach=False)
            if not bool(solution.reachable):
                continue
            checked += 1
            self._pose(np.asarray(solution.q))
            np.testing.assert_allclose(
                self.data.xpos[self.ee_body], position, atol=1e-9
            )
            rotation = self.data.xmat[self.ee_body].reshape(3, 3)
            # The tool's approach axis (local +x) must be straight down.
            np.testing.assert_allclose(rotation[:, 0], [0.0, 0.0, -1.0], atol=1e-9)
            # ...and the finger-opening axis (local +y) must carry the yaw.
            measured = math.atan2(rotation[1, 1], rotation[0, 1])
            self.assertAlmostEqual(
                math.atan2(math.sin(measured - yaw), math.cos(measured - yaw)),
                0.0,
                places=9,
            )
        self.assertGreater(checked, 200, "too few reachable samples to be meaningful")

    def test_yaw_convention_matches_the_cdpr(self) -> None:
        """yaw = 0 puts the finger axis on world +x, as on the CDPR.

        `ee_yaw` is an absolute value in the SmolVLA state vector, so a
        different convention would shift the observation distribution under a
        warm start from a CDPR checkpoint.
        """

        solution = K.top_down_ik(np.array([0.28, 0.0, 0.22]), 0.0, clamp_to_reach=False)
        self._pose(np.asarray(solution.q))
        rotation = self.data.xmat[self.ee_body].reshape(3, 3)
        np.testing.assert_allclose(rotation[:, 1], [1.0, 0.0, 0.0], atol=1e-9)

    def test_tracked_body_sits_between_the_pads(self) -> None:
        """`ee_base` must BE the grasp point, so grasp offsets stay near zero.

        Tracking the arm flange instead would put every `*_grasp_height_offset`
        5 cm out and reproduce the CDPR pick_up failure exactly.
        """

        self._pose(np.zeros(5))
        for name in ("left_finger", "right_finger"):
            self.data.qpos[
                self.model.jnt_qposadr[
                    mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
                ]
            ] = K.FINGER_LIMITS[1]
        mj.mj_forward(self.model, self.data)
        ee = self.data.xpos[self.ee_body]
        left = self.data.geom_xpos[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "left_finger_pad")
        ]
        right = self.data.geom_xpos[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "right_finger_pad")
        ]
        midpoint = 0.5 * (left + right)
        self.assertLess(
            float(np.linalg.norm(midpoint - ee)),
            0.002,
            "ee_base is not the point between the finger pads",
        )

    def test_gripper_span(self) -> None:
        spans = []
        for opening in K.FINGER_LIMITS:
            self.data.qpos[:] = 0.0
            for name in ("left_finger", "right_finger"):
                self.data.qpos[
                    self.model.jnt_qposadr[
                        mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
                    ]
                ] = opening
            mj.mj_forward(self.model, self.data)
            left = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "left_finger_pad")
            right = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "right_finger_pad")
            spans.append(
                abs(self.data.geom_xpos[left][1] - self.data.geom_xpos[right][1])
                - 2.0 * self.model.geom_size[left][1]
            )
        # 8 mm closed, 52 mm open. This is the number that decides which
        # objects the catalog may contain at all -- the CDPR's 69 mm apple does
        # not fit between these fingers at any opening.
        self.assertAlmostEqual(spans[0], 0.008, places=3)
        self.assertAlmostEqual(spans[1], 0.052, places=3)

    def test_wrist_camera_sees_the_grasp_point(self) -> None:
        solution = K.top_down_ik(np.array([0.28, 0.0, 0.22]), 0.0, clamp_to_reach=False)
        self._pose(np.asarray(solution.q))
        camera = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
        look = -self.data.cam_xmat[camera].reshape(3, 3)[:, 2]
        wanted = self.data.xpos[self.ee_body] - self.data.cam_xpos[camera]
        wanted = wanted / np.linalg.norm(wanted)
        offset = math.degrees(math.acos(float(np.clip(look @ wanted, -1.0, 1.0))))
        fovy = float(self.model.cam_fovy[camera])
        self.assertLess(offset, 0.5 * fovy, "grasp point is outside the wrist frustum")


@requires_mujoco
class SceneAndControlTests(unittest.TestCase):
    """End-to-end: the scene, the mount, and a grasp driven by 5-channel actions."""

    def _build(self, object_radius: float = 0.020, xy=(0.10, -0.06)):
        scene = SCENE_XML.read_text().replace(
            "</mujoco>",
            f"""
  <worldbody>
    <body name="probe_object" pos="{xy[0]} {xy[1]} 0.24">
      <freejoint name="probe_free"/>
      <inertial pos="0 0 0" mass="0.10" diaginertia="4e-5 4e-5 4e-5"/>
      <geom name="probe_geom" type="sphere" size="{object_radius}"
            contype="1" conaffinity="6" condim="4" friction="1.2 0.02 0.002"
            solref="0.006 1" solimp="0.95 0.99 0.001" margin="0.001"
            rgba="0.85 0.25 0.2 1"/>
    </body>
  </worldbody>
</mujoco>""",
        )
        path = ROBOT_DIR / "_test_scene.xml"
        path.write_text(scene)
        try:
            model = mj.MjModel.from_xml_path(str(path))
        finally:
            os.remove(path)
        W.mount_widowx200(model, W.DEFAULT_LAYOUT, mujoco=mj)
        layout = W.DEFAULT_LAYOUT
        spec = WidowX200ControlSpec(
            mount=WidowX200MountPose(layout.base_position, layout.base_yaw),
            workspace_x=layout.workspace_x,
            workspace_y=layout.workspace_y,
            workspace_z=layout.workspace_z,
            min_reach_radius=layout.min_reach_radius,
        )
        data = mj.MjData(model)
        return model, data, WidowX200TaskSpaceController(
            model, data, spec=spec, mujoco=mj
        ), spec

    def test_scene_compiles_and_the_mount_matches_the_layout(self) -> None:
        model = mj.MjModel.from_xml_path(str(SCENE_XML))
        W.mount_widowx200(model, W.DEFAULT_LAYOUT, mujoco=mj)
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wx200_mount")
        np.testing.assert_allclose(
            model.body_pos[body], W.DEFAULT_LAYOUT.base_position, atol=1e-12
        )
        # The desk surface the whole task layer is written against.
        desk = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "mjwarp_desk_surface_visual")
        self.assertAlmostEqual(float(model.geom_pos[desk][2]), W.DESK_SURFACE_Z, places=6)

    def test_reset_places_the_tool_where_asked(self) -> None:
        _, _, controller, _ = self._build()
        for target in ([0.0, 0.0, 0.26], [0.12, -0.05, 0.22], [-0.10, 0.04, 0.25]):
            controller.reset_to_pose(target, yaw=0.3, gripper=1.0)
            np.testing.assert_allclose(
                controller.get_end_effector_position(), target, atol=1e-6
            )

    def test_servo_holds_position(self) -> None:
        _, _, controller, spec = self._build()
        controller.reset_to_pose([0.0, 0.0, 0.24], 0.0, 1.0)
        target = controller.get_end_effector_position().copy()
        for _ in range(300):
            controller.run_simulation_step()
        error = float(np.linalg.norm(controller.get_end_effector_position() - target))
        # Measured 1.4 mm. A regression here means the gains or the gravity
        # load moved, and every reward reads this position.
        self.assertLess(error, 0.004)

    def test_action_channels_move_the_expected_axes(self) -> None:
        _, _, controller, spec = self._build()
        for axis, index in (("x", 0), ("y", 1), ("z", 2)):
            controller.reset_to_pose([0.0, 0.0, 0.22], 0.0, 1.0)
            start = controller.get_end_effector_position().copy()
            action = np.zeros(5)
            action[index] = 1.0
            for _ in range(8):
                controller.apply_normalized_action(action)
            delta = controller.get_end_effector_position() - start
            self.assertGreater(delta[index], 0.01, f"{axis} did not advance")
            for other in range(3):
                if other != index:
                    self.assertLess(abs(delta[other]), 0.01, f"{axis} leaked into axis {other}")

    def test_grasp_and_lift_through_normalized_actions_only(self) -> None:
        """The migration's actual claim: the CDPR action contract drives this arm."""

        model, data, controller, spec = self._build(object_radius=0.020)
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "probe_object")
        controller.reset_to_pose([-0.14, 0.06, 0.26], 0.0, 1.0)
        for _ in range(400):
            mj.mj_step(model, data)
        resting = data.xpos[body].copy()

        def servo(target, steps, gripper):
            for _ in range(steps):
                error = target - controller.get_end_effector_position()
                controller.apply_normalized_action(
                    np.clip(
                        np.concatenate(
                            [error / spec.action_step_xyz, [0.0, gripper]]
                        ),
                        -1.0,
                        1.0,
                    )
                )

        servo(resting + np.array([0.0, 0.0, 0.08]), 30, 0.0)
        servo(resting, 30, 0.0)
        servo(resting, 30, -1.0)

        left = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "left_finger_pad")
        right = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "right_finger_pad")
        target_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "probe_geom")
        pads = {
            frozenset((int(data.contact[i].geom1), int(data.contact[i].geom2)))
            for i in range(data.ncon)
        }
        self.assertIn(frozenset((left, target_geom)), pads, "no left pad contact")
        self.assertIn(frozenset((right, target_geom)), pads, "no right pad contact")

        before = float(data.xpos[body][2])
        servo(resting + np.array([0.0, 0.0, 0.10]), 40, -1.0)
        self.assertGreater(
            float(data.xpos[body][2]) - before,
            0.06,
            "the object was not lifted with the gripper",
        )

    def test_graspable_size_envelope(self) -> None:
        """Which objects this gripper can hold at all -- a catalog constraint.

        Measured band is 30-56 mm. The current RoboCasa catalog's apple (69 mm),
        bell pepper (74 mm), tomato, orange, and potato (58 mm) are all outside
        it: they need rescaling before any grasp task can use them.
        """

        def lifts(radius: float) -> float:
            model, data, controller, spec = self._build(object_radius=radius)
            body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "probe_object")
            controller.reset_to_pose([-0.14, 0.06, 0.26], 0.0, 1.0)
            for _ in range(400):
                mj.mj_step(model, data)
            resting = data.xpos[body].copy()

            def servo(target, steps, gripper):
                for _ in range(steps):
                    error = target - controller.get_end_effector_position()
                    controller.apply_normalized_action(
                        np.clip(
                            np.concatenate(
                                [error / spec.action_step_xyz, [0.0, gripper]]
                            ),
                            -1.0,
                            1.0,
                        )
                    )

            servo(resting + np.array([0.0, 0.0, 0.08]), 30, 0.0)
            servo(resting, 30, 0.0)
            servo(resting, 30, -1.0)
            before = float(data.xpos[body][2])
            servo(resting + np.array([0.0, 0.0, 0.10]), 40, -1.0)
            return float(data.xpos[body][2]) - before

        self.assertGreater(lifts(0.020), 0.06, "40 mm object should be graspable")
        self.assertLess(lifts(0.0345), 0.02, "the 69 mm CDPR apple must NOT fit")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


@requires_mujoco
class BatchedControllerParityTests(unittest.TestCase):
    """The GPU controller and the host controller must be the same controller.

    The CDPR paid for a divergence between its CPU and MJWarp controllers with a
    whole parity-report gate (`docs/cdpr_mjwarp_compatibility.md`). Here both
    paths call the same `joint_targets_from_task_targets`, so parity is
    structural rather than maintained -- this test is what keeps it that way.
    """

    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:  # pragma: no cover
            self.skipTest("PyTorch is not installed.")
        from robots.widowx200.widowx200_mujoco.batched_controller import (
            WidowX200BatchedController,
            WidowX200ModelIndices,
        )

        self.BatchedController = WidowX200BatchedController
        self.ModelIndices = WidowX200ModelIndices
        self.model = mj.MjModel.from_xml_path(str(ROBOT_XML))
        layout = W.DEFAULT_LAYOUT
        self.spec = WidowX200ControlSpec(
            mount=WidowX200MountPose(layout.base_position, layout.base_yaw),
            workspace_x=layout.workspace_x,
            workspace_y=layout.workspace_y,
            workspace_z=layout.workspace_z,
            min_reach_radius=layout.min_reach_radius,
        )

    def _controller(self, nworld: int):
        import torch

        return self.BatchedController(
            torch=torch,
            device=torch.device("cpu"),
            nworld=nworld,
            spec=self.spec,
            indices=self.ModelIndices.resolve(self.model, mj),
        )

    def test_model_indices_resolve(self) -> None:
        indices = self.ModelIndices.resolve(self.model, mj)
        self.assertEqual(len(indices.joint_qadr), 5)
        self.assertEqual(len(indices.actuator_ids), 5)
        self.assertNotIn(indices.gripper_actuator_id, indices.actuator_ids)

    def test_write_controls_matches_the_numpy_solution(self) -> None:
        import torch

        nworld = 12
        controller = self._controller(nworld)
        rng = np.random.default_rng(11)
        positions = np.stack(
            [
                rng.uniform(-0.2, 0.2, nworld),
                rng.uniform(-0.1, 0.2, nworld),
                rng.uniform(0.18, 0.32, nworld),
            ],
            axis=-1,
        )
        yaws = rng.uniform(-1.0, 1.0, nworld)
        openings = rng.uniform(0.0, 1.0, nworld)
        controller.reset_worlds(np.arange(nworld), positions, yaws, openings)

        ctrl = torch.zeros((nworld, int(self.model.nu)), dtype=torch.float32)
        controller.write_controls(ctrl)

        indices = self.ModelIndices.resolve(self.model, mj)
        for world in range(nworld):
            expected = joint_targets_from_task_targets(
                positions[world],
                np.float64(yaws[world]),
                np.float64(openings[world]),
                self.spec,
            )
            got = ctrl[world, list(indices.actuator_ids)].numpy()
            np.testing.assert_allclose(
                got, np.asarray(expected["q"]), atol=2e-5, err_msg=f"world {world}"
            )
            self.assertAlmostEqual(
                float(ctrl[world, indices.gripper_actuator_id]),
                float(expected["gripper_ctrl"]),
                places=5,
            )

    def test_masked_worlds_do_not_move(self) -> None:
        import torch

        nworld = 8
        controller = self._controller(nworld)
        start = np.tile(np.array([0.0, 0.0, 0.26]), (nworld, 1))
        controller.reset_worlds(np.arange(nworld), start, np.zeros(nworld), np.ones(nworld))
        before = controller.target_position.clone()

        active = torch.zeros(nworld, dtype=torch.bool)
        active[::2] = True
        controller.integrate_actions(
            torch.ones((nworld, 5), dtype=torch.float32),
            active,
            torch.as_tensor(start, dtype=torch.float32),
        )
        moved = (controller.target_position - before).abs().sum(dim=-1) > 1e-6
        np.testing.assert_array_equal(moved.numpy(), active.numpy())

    def test_group_broadcast_makes_candidates_identical(self) -> None:
        # GRPO compares candidates that must share one initial condition;
        # controller state is part of that condition, not just qpos.
        nworld, group = 8, 4
        controller = self._controller(nworld)
        rng = np.random.default_rng(12)
        controller.reset_worlds(
            np.arange(nworld),
            rng.uniform(-0.1, 0.1, (nworld, 3)) + np.array([0.0, 0.0, 0.26]),
            rng.uniform(-1.0, 1.0, nworld),
            rng.uniform(0.0, 1.0, nworld),
        )
        controller.broadcast_group_state(np.array([0, 4]), group)
        targets = controller.target_position.numpy()
        np.testing.assert_allclose(targets[0:4], np.tile(targets[0], (4, 1)), atol=0)
        np.testing.assert_allclose(targets[4:8], np.tile(targets[4], (4, 1)), atol=0)

    def test_teleport_writes_a_pose_the_servo_already_satisfies(self) -> None:
        """qpos and ctrl must agree, or every episode starts with a lurch."""

        import torch

        nworld = 4
        controller = self._controller(nworld)
        indices = self.ModelIndices.resolve(self.model, mj)
        qpos = torch.zeros((nworld, int(self.model.nq)), dtype=torch.float32)
        qvel = torch.zeros((nworld, int(self.model.nv)), dtype=torch.float32)
        ctrl = torch.zeros((nworld, int(self.model.nu)), dtype=torch.float32)
        positions = np.tile(np.array([0.05, -0.02, 0.26]), (nworld, 1))
        controller.set_end_effector_poses(
            qpos, ctrl, qvel, positions, np.zeros(nworld), np.ones(nworld)
        )
        for column, (qadr, actuator) in enumerate(
            zip(indices.joint_qadr, indices.actuator_ids)
        ):
            np.testing.assert_allclose(
                qpos[:, qadr].numpy(), ctrl[:, actuator].numpy(), atol=0
            )

        # ...and MuJoCo must agree the pose is the one that was asked for.
        data = mj.MjData(self.model)
        data.qpos[:] = qpos[0].numpy()
        mj.mj_forward(self.model, data)
        world_ee = data.xpos[indices.ee_body_id]
        ops = K._ops_for(np.zeros(3))
        np.testing.assert_allclose(
            self.spec.mount.base_to_world(ops, world_ee), positions[0], atol=1e-6
        )

    def test_metadata_records_the_control_contract(self) -> None:
        metadata = self._controller(2).metadata()
        for key in (
            "action_step_xyz",
            "hold_steps",
            "target_leash",
            "pitch",
            "mount_position",
            "mount_yaw",
        ):
            self.assertIn(key, metadata, f"{key} must be checkpointed")


class SectorLatticeTests(unittest.TestCase):
    """The object-placement lattice, which replaces the CDPR's 3x3 grid."""

    def test_every_cell_is_inside_the_sector(self) -> None:
        layout = W.DEFAULT_LAYOUT
        for x, y in W.sector_lattice():
            self.assertTrue(layout.contains(x, y), f"cell ({x:.3f}, {y:.3f}) unreachable")

    def test_separation_clears_the_collector_requirement(self) -> None:
        # 0.16 m is the CDPR collector's separation, set by the widest
        # realistic object pair. Two objects closer than that can spawn
        # touching, which scores the spawn rather than the policy.
        cells = W.sector_lattice()
        self.assertGreaterEqual(len(cells), 4, "fewer cells than object slots")
        worst = min(
            math.dist(a, b)
            for index, a in enumerate(cells)
            for b in cells[index + 1 :]
        )
        self.assertGreaterEqual(worst, 0.16)

    def test_cells_stay_reachable_through_a_lift(self) -> None:
        layout = W.DEFAULT_LAYOUT
        for world_z in (layout.workspace_z[0], 0.19, 0.24, layout.workspace_z[1]):
            self.assertGreaterEqual(
                W.usable_radius(world_z),
                layout.spawn_radius[1],
                f"outer lattice ring is out of reach at z={world_z}",
            )
