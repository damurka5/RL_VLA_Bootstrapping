from __future__ import annotations

from typing import Any, Mapping, Sequence

from .cdpr_backend import (
    CDPRBackendConfig,
    CDPRFingerContactBatch,
    CDPRLowDimBatch,
    CDPRRenderBatch,
    CDPRSimulatorBackend,
)


class MujocoCPUReferenceBackend(CDPRSimulatorBackend):
    """Adapter for the existing CPU environment.

    The production CPU trainer continues to use its established environment
    directly.  This adapter exists for parity fixtures and backend-neutral
    validation; it deliberately makes no claim that CPU rollout is batched.
    """

    def __init__(self, *, config: CDPRBackendConfig, env: Any) -> None:
        if int(config.worlds_per_rank) != 1:
            raise ValueError("The CPU reference backend adapter owns exactly one world.")
        self.config = config
        self.env = env

    @property
    def device(self) -> str:
        return "cpu"

    def reset_worlds(
        self,
        world_indices: Any,
        *,
        qpos: Any | None = None,
        qvel: Any | None = None,
        controller_state: Mapping[str, Any] | None = None,
    ) -> None:
        del world_indices, qpos, qvel, controller_state
        self.env.reset()

    def broadcast_group_state(self, base_world_indices: Any) -> None:
        del base_world_indices

    def step(self, actions: Any, active_mask: Any) -> CDPRLowDimBatch:
        import numpy as np

        if bool(np.asarray(active_mask).reshape(-1)[0]):
            self.env.step(np.asarray(actions).reshape(-1, 5)[0])
        return self.low_dim_observations()

    def low_dim_observations(self) -> CDPRLowDimBatch:
        import numpy as np

        env = self.env
        obs = env._get_obs()
        ee_pose = env.get_ee_pose()
        gripper = env.get_gripper_state()
        objects = np.asarray(obs.get("all_object_positions", np.zeros((4, 3))), dtype=np.float32)
        if objects.shape != (4, 3):
            padded = np.zeros((4, 3), dtype=np.float32)
            padded[: min(4, len(objects))] = objects[:4]
            objects = padded
        return CDPRLowDimBatch(
            ee_position=np.asarray(ee_pose["position"], dtype=np.float32)[None],
            ee_quaternion=np.asarray(ee_pose["quaternion"], dtype=np.float32)[None],
            ee_yaw=np.asarray([env._read_current_yaw()], dtype=np.float32),
            gripper_opening=np.asarray([gripper["opening"]], dtype=np.float32),
            target_position=np.asarray(env.sim.target_pos, dtype=np.float32)[None],
            tendon_lengths=np.asarray(env.sim.get_cable_lengths(), dtype=np.float32)[None],
            object_positions=objects[None],
            object_quaternions=np.zeros((1, 4, 4), dtype=np.float32),
        )

    def render_policy_cameras(self) -> CDPRRenderBatch:
        import numpy as np

        frames = self.env.render(("overview", "ee_camera"))

        def to_bchw(value: Any) -> Any:
            arr = np.asarray(value, dtype=np.float32) / 255.0
            return np.transpose(arr[None], (0, 3, 1, 2))

        return CDPRRenderBatch(
            overview=to_bchw(frames["overview"]),
            wrist=to_bchw(frames["ee_camera"]),
        )

    def body_pose(self, body_names: Sequence[str]) -> tuple[Any, Any]:
        import numpy as np

        poses = [self.env.get_body_pose(name) for name in body_names]
        return (
            np.asarray([item["position"] for item in poses], dtype=np.float32)[None],
            np.asarray([item["quaternion"] for item in poses], dtype=np.float32)[None],
        )

    def body_velocity(self, body_names: Sequence[str]) -> tuple[Any, Any]:
        import numpy as np

        values = [self.env.get_body_velocity(name) for name in body_names]
        return (
            np.asarray([item["linear"] for item in values], dtype=np.float32)[None],
            np.asarray([item["angular"] for item in values], dtype=np.float32)[None],
        )

    def contact_mask(self, geom_a_ids: Any, geom_b_ids: Any) -> Any:
        del geom_a_ids, geom_b_ids
        raise NotImplementedError("Use the CPU environment contact summary in parity fixtures.")

    def finger_object_contact_metrics(
        self, target_slots: Any
    ) -> CDPRFingerContactBatch:
        del target_slots
        raise NotImplementedError("Use the CPU environment contact summary in parity fixtures.")

    def set_object_catalogs(self, catalog_ids: Any) -> None:
        del catalog_ids
        raise NotImplementedError("CPU scene topology is selected by the existing scene builder.")

    def set_visual_variants(
        self,
        texture_variant_ids: Any,
        background_rgba: Any,
        gripper_shade: Any,
    ) -> None:
        del texture_variant_ids, background_rgba, gripper_shade
        raise NotImplementedError(
            "CPU visual randomization remains owned by CDPRLanguageRLEnv."
        )

    def set_end_effector_poses(
        self,
        positions: Any,
        yaws: Any,
        *,
        zero_velocity: bool = True,
    ) -> None:
        del positions, yaws, zero_velocity
        raise NotImplementedError(
            "CPU reset pose changes remain owned by CDPRLanguageRLEnv."
        )

    def set_gripper_openings(self, openings: Any) -> None:
        del openings
        raise NotImplementedError(
            "CPU gripper reset state remains owned by CDPRLanguageRLEnv."
        )

    def set_free_body_poses(
        self,
        body_ids: Any,
        positions: Any,
        quaternions: Any | None = None,
        *,
        zero_velocity: bool = True,
    ) -> None:
        del body_ids, positions, quaternions, zero_velocity
        raise NotImplementedError("Use CDPRLanguageRLEnv.set_object_pose for the CPU backend.")

    def export_worlds(self, world_indices: Sequence[int]) -> list[dict[str, Any]]:
        if tuple(world_indices) not in {(0,), ()}:
            raise IndexError("The CPU reference backend has only world zero.")
        state = self.low_dim_observations()
        return [
            {
                "ee_position": state.ee_position[0].copy(),
                "target_position": state.target_position[0].copy(),
                "object_positions": state.object_positions[0].copy(),
                "tendon_lengths": state.tendon_lengths[0].copy(),
            }
        ]

    def metadata(self) -> dict[str, Any]:
        import mujoco

        return {
            "backend": "mujoco_cpu",
            "mujoco_version": str(mujoco.__version__),
            "worlds_per_rank": 1,
            "physics_dtype": "float64",
        }

    def close(self) -> None:
        self.env.close()
