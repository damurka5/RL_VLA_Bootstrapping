"""The oracle harness must be able to run on the engine training actually uses.

The harness drives the production reset, reward, success predicate and grasp
detector, so it is the check to run after touching reset shaping or the reward
ladder. It used to hard-code ``backend="mujoco_cpu"``, which meant a run on the
GPU box verified a CPU relative of the training physics and said so nowhere in
its output. These tests cover the two halves of fixing that: choosing the
backend, and the two methods MJWarp was missing that the harness calls.

None of this needs a GPU. The MJWarp methods are exercised unbound against a
stub, which is exactly where a shape or a unit conversion goes wrong.
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace


class ResolveSimulatorBackendTests(unittest.TestCase):
    """Silently downgrading a requested backend is the failure to avoid."""

    @staticmethod
    def _resolve(*args, **kwargs):
        from rl_vla_bootstrapping.simulation.cdpr_backend import (
            resolve_simulator_backend,
        )

        return resolve_simulator_backend(*args, **kwargs)

    def test_auto_takes_mjwarp_when_the_runtime_is_there(self):
        backend, reason = self._resolve("auto", cuda_available=True)
        self.assertEqual(backend, "mjlab_mjwarp")
        self.assertIn("auto", reason)

    def test_auto_falls_back_without_cuda(self):
        backend, reason = self._resolve("auto", cuda_available=False)
        self.assertEqual(backend, "mujoco_cpu")
        self.assertIn("no CUDA", reason)

    def test_auto_falls_back_when_a_package_is_missing(self):
        backend, reason = self._resolve(
            "auto", cuda_available=True, missing_dependencies=("mujoco_warp",)
        )
        self.assertEqual(backend, "mujoco_cpu")
        self.assertIn("mujoco_warp", reason)

    def test_asking_for_mjwarp_without_cuda_is_an_error_not_a_downgrade(self):
        """The whole point: "verified on MJWarp" must never mean CPU MuJoCo."""

        from rl_vla_bootstrapping.simulation.cdpr_backend import (
            SimulatorDependencyError,
        )

        with self.assertRaises(SimulatorDependencyError):
            self._resolve("mjlab_mjwarp", cuda_available=False)
        with self.assertRaises(SimulatorDependencyError) as caught:
            self._resolve(
                "mjlab_mjwarp",
                cuda_available=True,
                missing_dependencies=("warp", "mujoco_warp"),
            )
        self.assertIn("warp", str(caught.exception))

    def test_asking_for_cpu_is_honoured_on_a_gpu_box(self):
        backend, reason = self._resolve("mujoco_cpu", cuda_available=True)
        self.assertEqual(backend, "mujoco_cpu")
        self.assertIn("requested", reason)

    def test_an_unknown_name_is_rejected(self):
        with self.assertRaises(ValueError):
            self._resolve("isaac", cuda_available=True)

    def test_every_reason_is_reportable(self):
        """The reason lands in the manifest, so it must never be empty."""

        for requested, cuda, missing in (
            ("auto", True, ()),
            ("auto", False, ()),
            ("auto", True, ("warp",)),
            ("mujoco_cpu", False, ()),
            ("mjlab_mjwarp", True, ()),
        ):
            with self.subTest(requested=requested, cuda=cuda):
                _, reason = self._resolve(
                    requested,
                    cuda_available=cuda,
                    missing_dependencies=missing,
                )
                self.assertTrue(reason.strip())

    def test_the_dependency_probe_does_not_raise(self):
        """It runs before any backend exists, so it must not be able to crash."""

        from rl_vla_bootstrapping.simulation.cdpr_backend import (
            MJWARP_RUNTIME_PACKAGES,
            missing_mjwarp_dependencies,
        )

        missing = missing_mjwarp_dependencies()
        self.assertIsInstance(missing, tuple)
        self.assertTrue(set(missing).issubset(set(MJWARP_RUNTIME_PACKAGES)))


class BackendSurfaceParityTests(unittest.TestCase):
    """Both backends must answer everything the harness asks of them.

    Checked on the classes, so it runs without CUDA -- which is the point: the
    MJWarp backend was missing controller_state and render_world, and nothing
    would have surfaced that until someone ran the harness on the GPU box.
    """

    # Every backend attribute the harness touches.
    REQUIRED = (
        "close",
        "controller_state",
        "device",
        "finger_object_contact_metrics",
        "low_dim_observations",
        "metadata",
        "object_body_ids",
        "render_world",
        "set_free_body_poses",
        "set_gripper_openings",
        "step",
        "worlds_per_rank",
    )

    def test_the_mjwarp_backend_covers_the_harness_surface(self):
        from rl_vla_bootstrapping.simulation.mjlab_mjwarp_backend import (
            MJLabMJWarpCDPRBackend,
        )

        # object_body_ids is set in __init__, not declared on the class.
        instance_attributes = {"object_body_ids"}
        for name in self.REQUIRED:
            if name in instance_attributes:
                continue
            with self.subTest(attribute=name):
                self.assertTrue(
                    hasattr(MJLabMJWarpCDPRBackend, name),
                    f"MJLabMJWarpCDPRBackend is missing {name}, which the "
                    "oracle harness calls",
                )
        self.assertIn(
            "self.object_body_ids",
            _mjwarp_source(),
            "object_body_ids must still be assigned in __init__",
        )

    def test_the_cpu_reference_backend_covers_the_same_surface(self):
        from rl_vla_bootstrapping.simulation.mujoco_reference_batched_backend import (
            MujocoReferenceBatchedBackend,
        )

        for name in self.REQUIRED:
            if name == "object_body_ids":
                continue
            with self.subTest(attribute=name):
                self.assertTrue(hasattr(MujocoReferenceBatchedBackend, name))


def _mjwarp_source() -> str:
    import inspect

    from rl_vla_bootstrapping.simulation import mjlab_mjwarp_backend

    return inspect.getsource(mjlab_mjwarp_backend)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "the MJWarp adapters operate on tensors",
)
class MJWarpHarnessAdapterTests(unittest.TestCase):
    """controller_state and render_world, called unbound against a stub.

    MJWarp holds its controller set-points as CUDA tensors and its camera
    output as normalized float32 BCHW; the harness expects host numpy and HWC
    uint8, the way the CPU reference backend gives them. That conversion is
    where a silent shape or scale mistake lives, and it does not need a GPU to
    test.
    """

    def test_controller_state_matches_the_cpu_backend_contract(self):
        import numpy as np
        import torch

        from rl_vla_bootstrapping.simulation.mjlab_mjwarp_backend import (
            MJLabMJWarpCDPRBackend,
        )

        worlds = 3
        target = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=torch.float32,
        )
        yaw = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
        gripper = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
        stub = SimpleNamespace(
            _controller_target=target,
            _controller_yaw=yaw,
            _controller_gripper=gripper,
        )
        state = MJLabMJWarpCDPRBackend.controller_state(stub)

        self.assertEqual(set(state), {"target", "yaw", "gripper"})
        for key in state:
            with self.subTest(key=key):
                self.assertIsInstance(state[key], np.ndarray)
        self.assertEqual(state["target"].shape, (worlds, 3))
        self.assertEqual(state["yaw"].shape, (worlds,))
        self.assertEqual(state["gripper"].shape, (worlds,))
        # The harness indexes per world and calls float() on the result; that
        # has to give the commanded value back, not a view of something else.
        self.assertAlmostEqual(float(state["gripper"][1]), 0.5, places=6)
        np.testing.assert_allclose(state["target"], target.numpy(), atol=1e-6)

        # A copy, not a live view: the harness reads it once per action and
        # would otherwise see the set-points move under it.
        gripper[1] = 0.0
        self.assertAlmostEqual(float(state["gripper"][1]), 0.5, places=6)

    def test_render_world_returns_hwc_uint8_for_the_asked_world(self):
        import torch

        from rl_vla_bootstrapping.simulation.cdpr_backend import CDPRRenderBatch
        from rl_vla_bootstrapping.simulation.mjlab_mjwarp_backend import (
            MJLabMJWarpCDPRBackend,
        )

        worlds, height, width = 2, 4, 6
        overview = torch.zeros((worlds, 3, height, width), dtype=torch.float32)
        wrist = torch.zeros((worlds, 3, height, width), dtype=torch.float32)
        # World 1 is white on the overview and mid-grey on the wrist, so the
        # world index and the camera mapping are both observable.
        overview[1] = 1.0
        wrist[1] = 0.5
        stub = SimpleNamespace(
            worlds_per_rank=worlds,
            render_policy_cameras=lambda: CDPRRenderBatch(
                overview=overview, wrist=wrist
            ),
        )

        frames = MJLabMJWarpCDPRBackend.render_world(stub, 1)
        self.assertEqual(set(frames), {"overview", "ee_camera"})
        for name, frame in frames.items():
            with self.subTest(camera=name):
                self.assertEqual(frame.shape, (height, width, 3))
                self.assertEqual(frame.dtype.name, "uint8")
        self.assertEqual(int(frames["overview"].max()), 255)
        # 0.5 -> 128 (round-half-even on .5 would give 128 here either way);
        # the check that matters is that it is scaled by 255, not clipped to 1.
        self.assertEqual(int(frames["ee_camera"].max()), 128)

        zero = MJLabMJWarpCDPRBackend.render_world(stub, 0)
        self.assertEqual(int(zero["overview"].max()), 0)

    def test_render_world_rejects_a_world_outside_the_batch(self):
        import torch

        from rl_vla_bootstrapping.simulation.cdpr_backend import CDPRRenderBatch
        from rl_vla_bootstrapping.simulation.mjlab_mjwarp_backend import (
            MJLabMJWarpCDPRBackend,
        )

        empty = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
        stub = SimpleNamespace(
            worlds_per_rank=2,
            render_policy_cameras=lambda: CDPRRenderBatch(
                overview=empty, wrist=empty
            ),
        )
        with self.assertRaises(IndexError):
            MJLabMJWarpCDPRBackend.render_world(stub, 2)
        with self.assertRaises(IndexError):
            MJLabMJWarpCDPRBackend.render_world(stub, -1)


if __name__ == "__main__":
    unittest.main()
