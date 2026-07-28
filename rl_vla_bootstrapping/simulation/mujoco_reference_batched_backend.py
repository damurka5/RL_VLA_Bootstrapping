"""Batched CPU MuJoCo backend that satisfies the production backend contract.

The MJ-Lab/MJWarp backend is CUDA-only, so it cannot run on a laptop.  This
adapter owns ``worlds_per_rank`` independent CPU ``HeadlessCDPRSimulation``
instances built from the *same* fixed-topology MJWarp MJCF and exposes the same
tensor-shaped API (``CDPRSimulatorBackend``).  That lets the real reset code
(``BatchedReverseFrontierResetter``), the real reward
(``evaluate_active_sparse_tasks``) and the real physical-grasp detector run
unchanged against CPU physics.

It is a *reference* backend, not the production one: the solver is MuJoCo's CPU
pipeline at float64 instead of MJWarp's GPU pipeline at float32, so contact
forces and therefore grasp latch timings can differ in the last digits.  Every
consumer must report ``metadata()['exact_production_backend'] is False``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cdpr_backend import (
    CDPRBackendConfig,
    CDPRFingerContactBatch,
    CDPRLowDimBatch,
    CDPRRenderBatch,
    CDPRSimulatorBackend,
)
from .cdpr_object_catalog import (
    ACTIVE_CDPR_CATALOGS,
    GEOM_SLOT_NAMES,
    INACTIVE_CATALOG_ID,
    OBJECT_VARIANTS,
    compile_catalog_variant_models,
    slot_geom_name,
)


CAMERA_NAMES = ("overview", "ee_camera")
_GRIPPER_SURFACE_GEOMS = (
    "palm",
    "finger_l_shoulder",
    "finger_l_link",
    "finger_l_tip",
    "left_finger_pad",
    "finger_r_shoulder",
    "finger_r_link",
    "finger_r_tip",
    "right_finger_pad",
)
# MJWarp treats a contact as touching when its signed distance is <= 2 mm.
_CONTACT_DISTANCE_TOLERANCE = 0.002


def _name_id(mj: Any, model: Any, objtype: Any, name: str) -> int:
    value = int(mj.mj_name2id(model, objtype, name))
    if value < 0:
        raise RuntimeError(f"Missing MJCF element {name!r}.")
    return value


class MujocoReferenceBatchedBackend(CDPRSimulatorBackend):
    """CPU stand-in for ``MJLabMJWarpCDPRBackend`` with identical tensor API."""

    def __init__(self, *, config: CDPRBackendConfig, xml_path: Path | None = None) -> None:
        import mujoco as mj
        import torch

        from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import (
            HeadlessCDPRSimulation,
        )

        self.config = config
        self.mj = mj
        self.torch = torch
        self._device = "cpu"
        self._nonfinite_world_events = 0
        resolved = Path(xml_path or config.xml_path or "").expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"CDPR MJCF does not exist: {resolved}")
        self.xml_path = resolved

        self.sims: list[Any] = []
        for _ in range(int(config.worlds_per_rank)):
            sim = HeadlessCDPRSimulation(
                str(resolved),
                record_trajectory=False,
                use_model_cache=False,
                timestep=0.002,
                render_enabled=False,
            )
            sim.initialize()
            self.sims.append(sim)

        model = self.sims[0].model
        self.ee_body_id = _name_id(mj, model, mj.mjtObj.mjOBJ_BODY, "ee_base")
        self.object_body_ids = tuple(
            _name_id(mj, model, mj.mjtObj.mjOBJ_BODY, f"mjwarp_object_slot_{slot}")
            for slot in range(4)
        )
        self._object_qadr = tuple(
            int(model.jnt_qposadr[
                _name_id(
                    mj,
                    model,
                    mj.mjtObj.mjOBJ_JOINT,
                    f"mjwarp_object_slot_{slot}_free",
                )
            ])
            for slot in range(4)
        )
        self._object_dofadr = tuple(
            int(model.jnt_dofadr[
                _name_id(
                    mj,
                    model,
                    mj.mjtObj.mjOBJ_JOINT,
                    f"mjwarp_object_slot_{slot}_free",
                )
            ])
            for slot in range(4)
        )
        ee_joint = _name_id(mj, model, mj.mjtObj.mjOBJ_JOINT, "ee_free")
        self._ee_qadr = int(model.jnt_qposadr[ee_joint])
        self._ee_dofadr = int(model.jnt_dofadr[ee_joint])
        yaw_joint = _name_id(mj, model, mj.mjtObj.mjOBJ_JOINT, "ee_yaw")
        self._yaw_qadr = int(model.jnt_qposadr[yaw_joint])
        self._yaw_dofadr = int(model.jnt_dofadr[yaw_joint])
        self._yaw_limits = (
            (float(model.jnt_range[yaw_joint][0]), float(model.jnt_range[yaw_joint][1]))
            if bool(model.jnt_limited[yaw_joint])
            else (-math.pi, math.pi)
        )
        self._slot_geom_ids = tuple(
            tuple(
                _name_id(mj, model, mj.mjtObj.mjOBJ_GEOM, slot_geom_name(slot, name))
                for name in GEOM_SLOT_NAMES
            )
            for slot in range(4)
        )
        # The collider used for grasp evidence is the same primitive MJWarp
        # matches finger pads against: every geom belonging to the slot body.
        self._pad_geom_ids = (
            _name_id(mj, model, mj.mjtObj.mjOBJ_GEOM, "left_finger_pad"),
            _name_id(mj, model, mj.mjtObj.mjOBJ_GEOM, "right_finger_pad"),
        )
        self._gripper_surface_geom_ids = tuple(
            _name_id(mj, model, mj.mjtObj.mjOBJ_GEOM, name)
            for name in _GRIPPER_SURFACE_GEOMS
        )
        self._desk_visual_geom_id = _name_id(
            mj, model, mj.mjtObj.mjOBJ_GEOM, "mjwarp_desk_surface_visual"
        )
        self._desk_material_ids = tuple(
            _name_id(mj, model, mj.mjtObj.mjOBJ_MATERIAL, f"mjwarp_desk_mat_{index}")
            for index in range(7)
        )
        self._tendon_ids = tuple(
            _name_id(mj, model, mj.mjtObj.mjOBJ_TENDON, name)
            for name in self._tendon_names(model)
        )

        # MuJoCo's compiler marks a geom whose local pose is the identity with
        # geom_sameframe=1 and then skips the local transform in mj_kinematics.
        # Catalog switching rewrites geom_pos/geom_quat at runtime, so any slot
        # geom that happened to compile at the body origin would silently ignore
        # its new offset -- including the "park at z=10" pose that disables an
        # unused primitive, which would leave a stray collider at the object
        # centre. Clear the flag once for every swappable geom.
        # MuJoCo also builds a per-body bounding-volume hierarchy at compile time
        # and uses it as the collision mid-phase. That BVH is a model constant:
        # it is not rebuilt when catalog switching moves or resizes a slot geom,
        # so a swapped-in collider gets culled before narrowphase and the object
        # falls through the desk. Mid-phase is an acceleration structure only --
        # disabling it changes cost, not contact physics.
        for sim in self.sims:
            for slot_geoms in self._slot_geom_ids:
                for geom in slot_geoms:
                    sim.model.geom_sameframe[geom] = 0
            sim.model.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_MIDPHASE)

        self._catalog_models = compile_catalog_variant_models(mj, resolved)
        self._catalog_ids = np.full((self.worlds_per_rank, 4), INACTIVE_CATALOG_ID, dtype=np.int64)
        self._object_rest_height = np.zeros((self.worlds_per_rank, 4), dtype=np.float32)

        self._controller_target = np.zeros((self.worlds_per_rank, 3), dtype=np.float64)
        self._controller_yaw = np.zeros((self.worlds_per_rank,), dtype=np.float64)
        self._controller_gripper = np.ones((self.worlds_per_rank,), dtype=np.float64)
        for world, sim in enumerate(self.sims):
            self._controller_target[world] = sim.get_end_effector_position()
            self._controller_yaw[world] = sim.get_yaw()
            self._controller_gripper[world] = sim.get_gripper_opening()

        self._workspace_min = np.array(
            (min(config.workspace_x), min(config.workspace_y), min(config.workspace_z)),
            dtype=np.float64,
        )
        self._workspace_max = np.array(
            (max(config.workspace_x), max(config.workspace_y), max(config.workspace_z)),
            dtype=np.float64,
        )
        self._renderers: dict[int, Any] = {}
        self._scene_option = mj.MjvOption()
        mj.mjv_defaultOption(self._scene_option)
        self._scene_option.geomgroup[:] = 0
        # Group 3 is collision-only; excluding it matches the MJWarp
        # policy-camera contract and keeps black proxy boxes out of frame.
        self._scene_option.geomgroup[:3] = 1

    @staticmethod
    def _tendon_names(model: Any) -> tuple[str, ...]:
        import mujoco as mj

        names = []
        for index in range(int(model.ntendon)):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_TENDON, index)
            if name:
                names.append(name)
        return tuple(names[:4])

    # ------------------------------------------------------------------ misc

    @property
    def device(self) -> Any:
        return self._device

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "mujoco_reference_batched",
            "exact_production_backend": False,
            "production_backend": "mjlab_mjwarp",
            "physics_dtype": "float64",
            "worlds_per_rank": int(self.worlds_per_rank),
            "physics_substeps": int(self.config.physics_substeps),
            "xml_path": str(self.xml_path),
        }

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        for sim in self.sims:
            sim.cleanup()

    def controller_state(self) -> dict[str, np.ndarray]:
        """Commanded controller set-points, as the policy's actions accumulate them."""

        return {
            "target": self._controller_target.copy(),
            "yaw": self._controller_yaw.copy(),
            "gripper": self._controller_gripper.copy(),
        }

    def pop_nonfinite_world_events(self) -> int:
        count = int(self._nonfinite_world_events)
        self._nonfinite_world_events = 0
        return count

    def _tensor(self, value: Any, dtype: Any = None) -> Any:
        return self.torch.as_tensor(
            np.asarray(value), dtype=dtype or self.torch.float32, device=self._device
        )

    # ----------------------------------------------------------------- reset

    def reset_worlds(
        self,
        world_indices: Any,
        *,
        qpos: Any | None = None,
        qvel: Any | None = None,
        controller_state: Mapping[str, Any] | None = None,
    ) -> None:
        del qpos, qvel, controller_state
        indices = np.asarray(
            self.torch.as_tensor(world_indices).detach().cpu().numpy()
        ).reshape(-1)
        for world in indices.tolist():
            sim = self.sims[int(world)]
            sim.reset_data_state()
            self.mj.mj_forward(sim.model, sim.data)
            self._controller_target[int(world)] = sim.get_end_effector_position()
            self._controller_yaw[int(world)] = sim.get_yaw()
            self._controller_gripper[int(world)] = sim.get_gripper_opening()

    def broadcast_group_state(self, base_world_indices: Any) -> None:
        """Copy each group's base world onto its remaining members.

        The resetter already wrote identical per-group poses into every world,
        so this only has to make the *derived* state (slider preload, contacts,
        controller bookkeeping) bit-identical, exactly as MJWarp does.
        """

        bases = np.asarray(
            self.torch.as_tensor(base_world_indices).detach().cpu().numpy()
        ).reshape(-1).tolist()
        group_size = max(1, int(self.config.grpo_group_size))
        for base in bases:
            base = int(base)
            source = self.sims[base]
            for offset in range(1, group_size):
                world = base + offset
                if world >= self.worlds_per_rank:
                    break
                target = self.sims[world]
                target.data.qpos[:] = source.data.qpos
                target.data.qvel[:] = source.data.qvel
                target.data.ctrl[:] = source.data.ctrl
                target.model.geom_dataid[:] = source.model.geom_dataid
                target.model.geom_size[:] = source.model.geom_size
                target.model.geom_pos[:] = source.model.geom_pos
                target.model.geom_quat[:] = source.model.geom_quat
                target.model.geom_rgba[:] = source.model.geom_rgba
                target.model.geom_matid[:] = source.model.geom_matid
                target.model.geom_aabb[:] = source.model.geom_aabb
                target.model.geom_rbound[:] = source.model.geom_rbound
                target.model.body_mass[:] = source.model.body_mass
                target.model.body_inertia[:] = source.model.body_inertia
                self.mj.mj_setConst(target.model, target.data)
                self.mj.mj_forward(target.model, target.data)
                target.target_pos = np.asarray(source.target_pos, dtype=float).copy()
                self._controller_target[world] = self._controller_target[base]
                self._controller_yaw[world] = self._controller_yaw[base]
                self._controller_gripper[world] = self._controller_gripper[base]
                self._catalog_ids[world] = self._catalog_ids[base]
                self._object_rest_height[world] = self._object_rest_height[base]

    def set_object_catalogs(self, catalog_ids: Any) -> None:
        ids = np.asarray(
            self.torch.as_tensor(catalog_ids).detach().cpu().numpy(), dtype=np.int64
        )
        if ids.shape != (self.worlds_per_rank, 4):
            raise ValueError(
                f"catalog_ids must have shape ({self.worlds_per_rank}, 4), got {ids.shape}."
            )
        for world, sim in enumerate(self.sims):
            for slot in range(4):
                catalog_id = int(ids[world, slot])
                self._apply_slot_catalog(sim, slot, catalog_id)
                self._object_rest_height[world, slot] = (
                    0.0
                    if catalog_id == INACTIVE_CATALOG_ID
                    else float(OBJECT_VARIANTS[ACTIVE_CDPR_CATALOGS[catalog_id]].rest_height)
                )
            self.mj.mj_setConst(sim.model, sim.data)
            self.mj.mj_forward(sim.model, sim.data)
        self._catalog_ids[:] = ids

    def _apply_slot_catalog(self, sim: Any, slot: int, catalog_id: int) -> None:
        mj = self.mj
        # An inactive slot keeps its geometry (CPU MuJoCo cannot carry a mesh
        # geom with dataid -1) but is made fully transparent.  The resetter
        # additionally parks the body metres outside every camera frustum, so
        # the world matches MJWarp's "slot absent" state visually and
        # physically.
        reference_catalog = (
            ACTIVE_CDPR_CATALOGS[0]
            if catalog_id == INACTIVE_CATALOG_ID
            else ACTIVE_CDPR_CATALOGS[catalog_id]
        )
        reference = self._catalog_models[reference_catalog]
        for index, geom_slot in enumerate(GEOM_SLOT_NAMES):
            geom = self._slot_geom_ids[slot][index]
            reference_geom = _name_id(
                mj, reference, mj.mjtObj.mjOBJ_GEOM, slot_geom_name(slot, geom_slot)
            )
            sim.model.geom_dataid[geom] = reference.geom_dataid[reference_geom]
            sim.model.geom_size[geom] = reference.geom_size[reference_geom]
            sim.model.geom_pos[geom] = reference.geom_pos[reference_geom]
            sim.model.geom_quat[geom] = reference.geom_quat[reference_geom]
            sim.model.geom_aabb[geom] = reference.geom_aabb[reference_geom]
            sim.model.geom_rbound[geom] = reference.geom_rbound[reference_geom]
            if catalog_id == INACTIVE_CATALOG_ID:
                sim.model.geom_matid[geom] = -1
                sim.model.geom_rgba[geom] = (0.0, 0.0, 0.0, 0.0)
            else:
                sim.model.geom_matid[geom] = reference.geom_matid[reference_geom]
                sim.model.geom_rgba[geom] = reference.geom_rgba[reference_geom]
        body = self.object_body_ids[slot]
        if catalog_id == INACTIVE_CATALOG_ID:
            sim.model.body_mass[body] = 1.0e-4
            sim.model.body_inertia[body] = (1.0e-8, 1.0e-8, 1.0e-8)
        else:
            variant = OBJECT_VARIANTS[ACTIVE_CDPR_CATALOGS[catalog_id]]
            sim.model.body_mass[body] = float(variant.mass)
            sim.model.body_inertia[body] = np.asarray(variant.inertia, dtype=np.float64)

    def set_visual_variants(
        self,
        texture_variant_ids: Any,
        background_rgba: Any,
        gripper_shade: Any,
    ) -> None:
        variants = np.asarray(
            self.torch.as_tensor(texture_variant_ids).detach().cpu().numpy()
        ).reshape(-1)
        shade = np.asarray(
            self.torch.as_tensor(gripper_shade).detach().cpu().numpy(), dtype=np.float64
        ).reshape(-1)
        background = np.asarray(
            self.torch.as_tensor(background_rgba).detach().cpu().numpy(), dtype=np.float64
        )
        if variants.shape != (self.worlds_per_rank,):
            raise ValueError("One desk texture variant is required per world.")
        if background.shape != (self.worlds_per_rank, 4):
            raise ValueError("background_rgba must have shape [world, 4].")
        if shade.shape != (self.worlds_per_rank,):
            raise ValueError("One gripper shade is required per world.")
        for world, sim in enumerate(self.sims):
            variant = int(variants[world])
            if not 0 <= variant < len(self._desk_material_ids):
                raise ValueError("Desk texture variants must be in [0, 7).")
            sim.model.geom_matid[self._desk_visual_geom_id] = self._desk_material_ids[variant]
            value = float(np.clip(shade[world], 0.0, 1.0))
            for geom in self._gripper_surface_geom_ids:
                sim.model.geom_rgba[geom, :3] = value

    def set_end_effector_poses(
        self,
        positions: Any,
        yaws: Any,
        *,
        zero_velocity: bool = True,
    ) -> None:
        position = np.asarray(
            self.torch.as_tensor(positions).detach().cpu().numpy(), dtype=np.float64
        )
        yaw = np.asarray(
            self.torch.as_tensor(yaws).detach().cpu().numpy(), dtype=np.float64
        ).reshape(-1)
        if position.shape != (self.worlds_per_rank, 3):
            raise ValueError("End-effector reset positions must have shape [world, 3].")
        if yaw.shape != (self.worlds_per_rank,):
            raise ValueError("End-effector reset yaws must have shape [world].")
        position = np.minimum(np.maximum(position, self._workspace_min), self._workspace_max)
        yaw = np.clip(yaw, self._yaw_limits[0], self._yaw_limits[1])
        for world, sim in enumerate(self.sims):
            sim.data.qpos[self._ee_qadr : self._ee_qadr + 3] = position[world]
            sim.data.qpos[self._yaw_qadr] = yaw[world]
            if zero_velocity:
                sim.data.qvel[self._ee_dofadr : self._ee_dofadr + 6] = 0.0
                sim.data.qvel[self._yaw_dofadr] = 0.0
            self.mj.mj_forward(sim.model, sim.data)
            # CPU equivalent of the MJWarp slider-preload loop: drive the four
            # prismatic winches until the tendon lengths match the teleported
            # platform pose, so the cables start neither slack nor stretched.
            sim._sync_controller_geometry_from_state()
            sim._match_sliders_to_ee_lengths(max_iter=12, tol=1.0e-6)
            sim.target_pos = sim.get_end_effector_position().copy()
            sim.set_yaw(float(yaw[world]))
            self.mj.mj_forward(sim.model, sim.data)
            self._controller_target[world] = sim.get_end_effector_position()
            self._controller_yaw[world] = float(sim.data.qpos[self._yaw_qadr])

    def set_gripper_openings(self, openings: Any) -> None:
        opening = np.asarray(
            self.torch.as_tensor(openings).detach().cpu().numpy(), dtype=np.float64
        ).reshape(-1)
        if opening.shape != (self.worlds_per_rank,):
            raise ValueError("One normalized gripper reset opening is required per world.")
        opening = np.clip(opening, 0.0, 1.0)
        for world, sim in enumerate(self.sims):
            value = float(opening[world])
            if sim.jnt_finger_l_qadr is not None:
                sim.data.qpos[sim.jnt_finger_l_qadr] = sim.gripper_joint_min + value * (
                    sim.gripper_joint_max - sim.gripper_joint_min
                )
            sim.set_gripper(value)
            self.mj.mj_forward(sim.model, sim.data)
            self._controller_gripper[world] = value

    def set_free_body_poses(
        self,
        body_ids: Any,
        positions: Any,
        quaternions: Any | None = None,
        *,
        zero_velocity: bool = True,
    ) -> None:
        ids = tuple(int(value) for value in body_ids)
        pos = np.asarray(
            self.torch.as_tensor(positions).detach().cpu().numpy(), dtype=np.float64
        )
        if pos.shape != (self.worlds_per_rank, len(ids), 3):
            raise ValueError(
                f"positions must have shape ({self.worlds_per_rank}, {len(ids)}, 3)."
            )
        if quaternions is None:
            quat = np.zeros((self.worlds_per_rank, len(ids), 4), dtype=np.float64)
            quat[..., 0] = 1.0
        else:
            quat = np.asarray(
                self.torch.as_tensor(quaternions).detach().cpu().numpy(), dtype=np.float64
            )
            if quat.shape != (self.worlds_per_rank, len(ids), 4):
                raise ValueError(
                    f"quaternions must have shape ({self.worlds_per_rank}, {len(ids)}, 4)."
                )
            norm = np.linalg.norm(quat, axis=-1, keepdims=True)
            quat = quat / np.maximum(norm, 1.0e-8)
        for column, body_id in enumerate(ids):
            if body_id not in self.object_body_ids:
                raise ValueError(f"Body id {body_id} is not a fixed object slot.")
            slot = self.object_body_ids.index(body_id)
            qadr = self._object_qadr[slot]
            dofadr = self._object_dofadr[slot]
            for world, sim in enumerate(self.sims):
                sim.data.qpos[qadr : qadr + 3] = pos[world, column]
                sim.data.qpos[qadr + 3 : qadr + 7] = quat[world, column]
                if zero_velocity:
                    sim.data.qvel[dofadr : dofadr + 6] = 0.0
        for sim in self.sims:
            self.mj.mj_forward(sim.model, sim.data)

    # ------------------------------------------------------------------ step

    def step(self, actions: Any, active_mask: Any) -> CDPRLowDimBatch:
        action = np.asarray(
            self.torch.as_tensor(actions).detach().cpu().numpy(), dtype=np.float64
        )
        active = np.asarray(
            self.torch.as_tensor(active_mask).detach().cpu().numpy()
        ).reshape(-1).astype(bool)
        if action.shape != (self.worlds_per_rank, 5):
            raise ValueError(f"actions must have shape ({self.worlds_per_rank}, 5).")
        if active.shape != (self.worlds_per_rank,):
            raise ValueError(f"active_mask must have shape ({self.worlds_per_rank},).")
        action = np.clip(action, -1.0, 1.0)
        action = np.where(active[:, None], action, 0.0)

        for world, sim in enumerate(self.sims):
            ee = np.asarray(sim.get_end_effector_position(), dtype=np.float64)
            delta = action[world, :3] * float(self.config.action_step_xyz)
            if self.config.lock_non_commanded_axes:
                moving = np.abs(action[world, :3]) > float(
                    self.config.lock_non_commanded_axes_threshold
                )
                proposed = np.where(moving, self._controller_target[world] + delta, self._controller_target[world])
            else:
                proposed = ee + delta
            proposed = np.minimum(np.maximum(proposed, self._workspace_min), self._workspace_max)
            if active[world]:
                self._controller_target[world] = proposed
                self._controller_yaw[world] = float(
                    np.clip(
                        self._controller_yaw[world]
                        + action[world, 3] * float(self.config.action_step_yaw),
                        self._yaw_limits[0],
                        self._yaw_limits[1],
                    )
                )
                self._controller_gripper[world] = float(
                    np.clip(
                        self._controller_gripper[world]
                        + action[world, 4] * float(self.config.action_step_gripper),
                        0.0,
                        1.0,
                    )
                )
            sim.target_pos = self._controller_target[world].copy()
            sim.set_yaw(float(self._controller_yaw[world]))
            sim.set_gripper(float(self._controller_gripper[world]))
            for _ in range(int(self.config.physics_substeps)):
                sim.run_simulation_step(capture_frame=False)
            self.mj.mj_forward(sim.model, sim.data)
            if not (
                np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel))
            ):
                self._nonfinite_world_events += 1
                self.reset_worlds(self.torch.tensor([world]))
        return self.low_dim_observations()

    def low_dim_observations(self) -> CDPRLowDimBatch:
        worlds = self.worlds_per_rank
        ee_position = np.zeros((worlds, 3), dtype=np.float32)
        ee_quaternion = np.zeros((worlds, 4), dtype=np.float32)
        ee_yaw = np.zeros((worlds,), dtype=np.float32)
        gripper = np.zeros((worlds,), dtype=np.float32)
        tendons = np.zeros((worlds, 4), dtype=np.float32)
        object_positions = np.zeros((worlds, 4, 3), dtype=np.float32)
        object_quaternions = np.zeros((worlds, 4, 4), dtype=np.float32)
        for world, sim in enumerate(self.sims):
            ee_position[world] = sim.data.xpos[self.ee_body_id]
            ee_quaternion[world] = sim.data.xquat[self.ee_body_id]
            ee_yaw[world] = sim.data.qpos[self._yaw_qadr]
            gripper[world] = sim.get_gripper_opening()
            tendons[world] = np.asarray(sim.get_cable_lengths(), dtype=np.float32)[:4]
            for slot, body in enumerate(self.object_body_ids):
                object_positions[world, slot] = sim.data.xpos[body]
                object_quaternions[world, slot] = sim.data.xquat[body]
        return CDPRLowDimBatch(
            ee_position=self._tensor(ee_position),
            ee_quaternion=self._tensor(ee_quaternion),
            ee_yaw=self._tensor(ee_yaw),
            gripper_opening=self._tensor(gripper),
            target_position=self._tensor(self._controller_target),
            tendon_lengths=self._tensor(tendons),
            object_positions=self._tensor(object_positions),
            object_quaternions=self._tensor(object_quaternions),
        )

    # --------------------------------------------------------------- contact

    def finger_object_contact_metrics(self, target_slots: Any) -> CDPRFingerContactBatch:
        slots = np.asarray(
            self.torch.as_tensor(target_slots).detach().cpu().numpy(), dtype=np.int64
        ).reshape(-1)
        if slots.shape != (self.worlds_per_rank,):
            raise ValueError("One target object slot is required per world.")
        left_pad, right_pad = self._pad_geom_ids
        left_force = np.zeros((self.worlds_per_rank,), dtype=np.float32)
        right_force = np.zeros((self.worlds_per_rank,), dtype=np.float32)
        left_count = np.zeros((self.worlds_per_rank,), dtype=np.int64)
        right_count = np.zeros((self.worlds_per_rank,), dtype=np.int64)
        wrench = np.zeros(6, dtype=np.float64)
        for world, sim in enumerate(self.sims):
            target_geoms = set(self._slot_geom_ids[int(slots[world])])
            for index in range(int(sim.data.ncon)):
                contact = sim.data.contact[index]
                if float(contact.dist) > _CONTACT_DISTANCE_TOLERANCE:
                    continue
                first = int(contact.geom1)
                second = int(contact.geom2)
                first_is_target = first in target_geoms
                second_is_target = second in target_geoms
                hits_left = (first_is_target and second == left_pad) or (
                    second_is_target and first == left_pad
                )
                hits_right = (first_is_target and second == right_pad) or (
                    second_is_target and first == right_pad
                )
                if not (hits_left or hits_right):
                    continue
                self.mj.mj_contactForce(sim.model, sim.data, index, wrench)
                normal = abs(float(wrench[0]))
                if hits_left:
                    left_force[world] += normal
                    left_count[world] += 1
                if hits_right:
                    right_force[world] += normal
                    right_count[world] += 1
        bool_dtype = self.torch.bool
        return CDPRFingerContactBatch(
            left_contact=self._tensor(left_count > 0, dtype=bool_dtype),
            right_contact=self._tensor(right_count > 0, dtype=bool_dtype),
            left_normal_force=self._tensor(left_force),
            right_normal_force=self._tensor(right_force),
        )

    def contact_mask(self, geom_a_ids: Any, geom_b_ids: Any) -> Any:
        a_ids = set(
            int(v) for v in np.asarray(self.torch.as_tensor(geom_a_ids).cpu().numpy()).reshape(-1)
        )
        b_ids = set(
            int(v) for v in np.asarray(self.torch.as_tensor(geom_b_ids).cpu().numpy()).reshape(-1)
        )
        mask = np.zeros((self.worlds_per_rank,), dtype=bool)
        for world, sim in enumerate(self.sims):
            for index in range(int(sim.data.ncon)):
                contact = sim.data.contact[index]
                first = int(contact.geom1)
                second = int(contact.geom2)
                if (first in a_ids and second in b_ids) or (second in a_ids and first in b_ids):
                    mask[world] = True
                    break
        return self._tensor(mask, dtype=self.torch.bool)

    def body_pose(self, body_names: Sequence[str]) -> tuple[Any, Any]:
        ids = [
            _name_id(self.mj, self.sims[0].model, self.mj.mjtObj.mjOBJ_BODY, str(name))
            for name in body_names
        ]
        positions = np.zeros((self.worlds_per_rank, len(ids), 3), dtype=np.float32)
        quaternions = np.zeros((self.worlds_per_rank, len(ids), 4), dtype=np.float32)
        for world, sim in enumerate(self.sims):
            for column, body in enumerate(ids):
                positions[world, column] = sim.data.xpos[body]
                quaternions[world, column] = sim.data.xquat[body]
        return self._tensor(positions), self._tensor(quaternions)

    def body_velocity(self, body_names: Sequence[str]) -> tuple[Any, Any]:
        ids = [
            _name_id(self.mj, self.sims[0].model, self.mj.mjtObj.mjOBJ_BODY, str(name))
            for name in body_names
        ]
        linear = np.zeros((self.worlds_per_rank, len(ids), 3), dtype=np.float32)
        angular = np.zeros((self.worlds_per_rank, len(ids), 3), dtype=np.float32)
        velocity = np.zeros(6, dtype=np.float64)
        for world, sim in enumerate(self.sims):
            for column, body in enumerate(ids):
                self.mj.mj_objectVelocity(
                    sim.model, sim.data, self.mj.mjtObj.mjOBJ_BODY, int(body), velocity, 0
                )
                angular[world, column] = velocity[:3]
                linear[world, column] = velocity[3:]
        return self._tensor(linear), self._tensor(angular)

    # ---------------------------------------------------------------- render

    def render_world(self, world: int) -> dict[str, np.ndarray]:
        """RGB frames for one world's two policy cameras, HWC uint8."""

        sim = self.sims[int(world)]
        renderer = self._renderers.get(int(world))
        if renderer is None:
            renderer = self.mj.Renderer(
                sim.model,
                height=int(self.config.render_height),
                width=int(self.config.render_width),
            )
            self._renderers[int(world)] = renderer
        frames: dict[str, np.ndarray] = {}
        for camera in CAMERA_NAMES:
            renderer.update_scene(sim.data, camera=camera, scene_option=self._scene_option)
            renderer.scene.flags[self.mj.mjtRndFlag.mjRND_SKYBOX] = 1
            frames[camera] = np.asarray(renderer.render(), dtype=np.uint8).copy()
        return frames

    def render_policy_cameras(self) -> CDPRRenderBatch:
        overview = []
        wrist = []
        for world in range(self.worlds_per_rank):
            frames = self.render_world(world)
            overview.append(frames["overview"])
            wrist.append(frames["ee_camera"])
        stack = np.stack(overview, axis=0).astype(np.float32) / 255.0
        wrist_stack = np.stack(wrist, axis=0).astype(np.float32) / 255.0
        return CDPRRenderBatch(
            overview=self._tensor(stack.transpose(0, 3, 1, 2)),
            wrist=self._tensor(wrist_stack.transpose(0, 3, 1, 2)),
        )

    def export_worlds(self, world_indices: Sequence[int]) -> list[dict[str, Any]]:
        exported: list[dict[str, Any]] = []
        for world in world_indices:
            sim = self.sims[int(world)]
            exported.append(
                {
                    "world": int(world),
                    "qpos": np.asarray(sim.data.qpos, dtype=np.float64).copy(),
                    "qvel": np.asarray(sim.data.qvel, dtype=np.float64).copy(),
                    "catalog_ids": self._catalog_ids[int(world)].copy(),
                }
            )
        return exported
