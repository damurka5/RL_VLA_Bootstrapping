"""Batched MJWarp backend for the Interbotix WidowX-200.

The CDPR backend's world allocation, partial reset, GRPO group broadcast,
rendering, contacts, per-world object catalogs, visual randomization and
export are embodiment-independent and are inherited unchanged. What this class
replaces is exactly the robot's block (see the hooks at the top of
``MJLabMJWarpCDPRBackend``):

* **no cable calibration.** The base state is the arm standing at a home pose
  solved by closed-form IK, with ``qpos`` and ``ctrl`` equal so the servos
  start satisfied. There is no tendon preload to restore after a reset.
* **the controller** is ``WidowX200BatchedController``: the action is
  integrated onto the previous TARGET and leashed to the measured tool, then
  turned into five joint angles by top-down IK. The CDPR builds each target
  from the measured pose, which on this arm silently attenuates every action
  (robots/widowx200/README.md, "What is different").
* **observations** keep the CDPR contract: ``ee_yaw`` is the measured
  finger-axis angle (from the ``ee_base`` frame; this arm has no yaw joint),
  and the 4-wide ``tendon_lengths`` slot carries the five joint angles, which
  no task code reads.

The action means the same thing as on the CDPR: ``[x, y, z, yaw, gripper]`` in
``[-1, 1]``, world-frame end-effector deltas. One action is
``1 + hold_steps`` physics steps; the WidowX configs set ``hold_steps: 24``
(50 ms, 20 Hz) -- see ``WidowX200ControlSpec``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from robots.widowx200.widowx200_mujoco import workspace as widowx_workspace
from robots.widowx200.widowx200_mujoco.batched_controller import (
    WidowX200BatchedController,
    WidowX200ModelIndices,
)
from robots.widowx200.widowx200_mujoco.controller import (
    WidowX200ControlSpec,
    WidowX200MountPose,
    joint_targets_from_task_targets,
)
from robots.widowx200.widowx200_mujoco.kinematics import FINGER_LIMITS

from .cdpr_backend import CDPRBackendConfig, CDPRLowDimBatch
from .mjlab_mjwarp_backend import MJLabMJWarpCDPRBackend, _host_name_id

__all__ = ["WidowX200MJWarpBackend", "WIDOWX200_HOME_POSE"]


# Where every world's arm stands after a reset that carries no pose: above the
# middle of the object sector, open, fingers along world +x. The collector
# teleports to the episode's own start right after, so this only has to be
# reachable and clear of the objects.
WIDOWX200_HOME_POSE = {"position": (0.0, 0.02, 0.24), "yaw": 0.0, "gripper": 1.0}

# A parallel gripper is symmetric under a half turn, so +-pi/2 covers every
# grasp. Bounding the integrated yaw there keeps wrist_rotate clear of its
# +-pi wrap, where a small yaw step would ask for a full turn of the wrist.
WIDOWX200_YAW_LIMITS = (-0.5 * math.pi, 0.5 * math.pi)

# CDPRBackendConfig's own default ceiling (0.25, 0.60) is a CDPR number: the
# arm's top-down reach ends near 0.41 and its outer sector at 0.27. A config
# that leaves the bounds at that default gets the arm's layout instead.
_CDPR_DEFAULT_WORKSPACE_Z = (0.25, 0.60)


def _gripper_surface_names() -> tuple[str, ...]:
    return (
        "wx200_gripper_visual",
        "wx200_gripper_bar_visual",
        "left_finger_visual",
        "right_finger_visual",
    )


class WidowX200MJWarpBackend(MJLabMJWarpCDPRBackend):
    """``nworld`` WidowX-200 arms on GPU-resident MJWarp worlds."""

    embodiment = "widowx200"

    # -- configuration -----------------------------------------------------

    def _control_spec(self) -> WidowX200ControlSpec:
        layout = widowx_workspace.DEFAULT_LAYOUT
        z_bounds = tuple(float(value) for value in self.config.workspace_z)
        if z_bounds == _CDPR_DEFAULT_WORKSPACE_Z:
            z_bounds = tuple(layout.workspace_z)
        # A config may be stricter than the arm, never looser.
        z_low = max(float(z_bounds[0]), float(layout.workspace_z[0]))
        z_high = min(float(z_bounds[1]), float(layout.workspace_z[1]))
        if z_low >= z_high:
            raise ValueError(
                f"Controller z bounds {z_bounds} do not overlap the WidowX-200 "
                f"layout's {layout.workspace_z}."
            )
        return WidowX200ControlSpec(
            mount=WidowX200MountPose(layout.base_position, layout.base_yaw),
            action_step_xyz=float(self.config.action_step_xyz),
            action_step_yaw=float(self.config.action_step_yaw),
            action_step_gripper=float(self.config.action_step_gripper),
            hold_steps=int(self.config.hold_steps),
            workspace_x=tuple(layout.workspace_x),
            workspace_y=tuple(layout.workspace_y),
            workspace_z=(z_low, z_high),
            min_reach_radius=float(layout.min_reach_radius),
        )

    # -- embodiment hooks ---------------------------------------------------

    def _prepare_host_models(self) -> None:
        if int(
            self.mujoco.mj_name2id(
                self.host_model, self.mujoco.mjtObj.mjOBJ_BODY, "wx200_mount"
            )
        ) < 0:
            raise RuntimeError(
                f"{self.config.xml_path} has no `wx200_mount` body; the WidowX "
                "backend needs robots/widowx200/widowx200_mujoco/"
                "wx200_mjwarp_scene.xml."
            )
        # Every compiled variant, not only the host: the per-world collider
        # tables are read from all of them, and a variant left at the origin
        # would carry a different body_pos for the mount.
        for model in self._catalog_reference_models.values():
            model.opt.timestep = self.host_model.opt.timestep
            widowx_workspace.mount_widowx200(
                model, widowx_workspace.DEFAULT_LAYOUT, mujoco=self.mujoco
            )
        self._spec = self._control_spec()

    def _calibrate_host(self) -> dict[str, Any]:
        import numpy as np

        mj = self.mujoco
        model = self.host_model
        data = mj.MjData(model)
        indices = WidowX200ModelIndices.resolve(model, mj)
        home = WIDOWX200_HOME_POSE
        solution = joint_targets_from_task_targets(
            np.asarray(home["position"], dtype=np.float64),
            np.float64(home["yaw"]),
            np.float64(home["gripper"]),
            self._spec,
        )
        if not bool(np.asarray(solution["reachable"])):
            raise RuntimeError(f"WidowX-200 home pose {home} is unreachable.")
        q = np.asarray(solution["q"], dtype=np.float64).reshape(5)
        for qadr, actuator, value in zip(
            indices.joint_qadr, indices.actuator_ids, q
        ):
            data.qpos[qadr] = value
            data.ctrl[actuator] = value
        finger = float(solution["gripper_ctrl"])
        data.qpos[indices.left_finger_qadr] = finger
        data.qpos[indices.right_finger_qadr] = finger
        data.ctrl[indices.gripper_actuator_id] = finger
        data.qvel[:] = 0.0
        mj.mj_forward(model, data)
        return {
            "indices": indices,
            "base_qpos": data.qpos.astype(np.float32).copy(),
            "base_qvel": data.qvel.astype(np.float32).copy(),
            "base_ctrl": data.ctrl.astype(np.float32).copy(),
        }

    def _resolve_embodiment_ids(self) -> None:
        mj = self.mujoco
        model = self.host_model
        indices: WidowX200ModelIndices = self._calibration["indices"]
        self._indices = indices
        self.finger_joint_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_JOINT, "left_finger"
        )
        self.right_finger_joint_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_JOINT, "right_finger"
        )
        self.finger_qadr = int(model.jnt_qposadr[self.finger_joint_id])
        self.finger_dofadr = int(model.jnt_dofadr[self.finger_joint_id])
        self.right_finger_qadr = int(
            model.jnt_qposadr[self.right_finger_joint_id]
        )
        self.right_finger_dofadr = int(
            model.jnt_dofadr[self.right_finger_joint_id]
        )
        self.gripper_actuator_id = int(indices.gripper_actuator_id)
        self.gripper_surface_geom_ids = tuple(
            _host_name_id(mj, model, mj.mjtObj.mjOBJ_GEOM, name)
            for name in _gripper_surface_names()
        )

    def _bind_embodiment_views(self) -> None:
        torch = self.torch
        self._joint_qadr_tensor = torch.tensor(
            self._indices.joint_qadr, dtype=torch.int64, device=self._device
        )
        self._joint_dofadr_tensor = torch.tensor(
            self._indices.joint_dofadr, dtype=torch.int64, device=self._device
        )

    def _initialize_controller_state(self) -> None:
        torch = self.torch
        nworld = self.worlds_per_rank
        self._controller = WidowX200BatchedController(
            torch=torch,
            device=self._device,
            nworld=nworld,
            spec=self._spec,
            indices=self._indices,
        )
        self._finger_limits = torch.tensor(
            FINGER_LIMITS, dtype=torch.float32, device=self._device
        )
        self._gripper_ctrl_limits = self._finger_limits.clone()
        self._yaw_limits = torch.tensor(
            WIDOWX200_YAW_LIMITS, dtype=torch.float32, device=self._device
        )
        self._workspace_min = torch.tensor(
            (
                self._spec.workspace_x[0],
                self._spec.workspace_y[0],
                self._spec.workspace_z[0],
            ),
            dtype=torch.float32,
            device=self._device,
        )
        self._workspace_max = torch.tensor(
            (
                self._spec.workspace_x[1],
                self._spec.workspace_y[1],
                self._spec.workspace_z[1],
            ),
            dtype=torch.float32,
            device=self._device,
        )
        self._last_actions = torch.zeros(
            (nworld, 5), dtype=torch.float32, device=self._device
        )
        from .cdpr_object_catalog import INACTIVE_CATALOG_ID

        self._catalog_ids = torch.full(
            (nworld, 4),
            INACTIVE_CATALOG_ID,
            dtype=torch.int64,
            device=self._device,
        )
        self._object_rest_height = torch.zeros(
            (nworld, 4), dtype=torch.float32, device=self._device
        )
        self._controller.reset_worlds(
            torch.arange(nworld, dtype=torch.int64, device=self._device),
            self._xpos[:, self.ee_body_id],
            self._measured_yaw(),
            self._normalized_gripper_opening(),
        )
        self._controller.write_controls(self._ctrl)

    # The CDPR backend's controller set-points under their old names, for
    # tooling that reads them directly (the scripted oracle in
    # tools/audit/xy_approach_probe.py reads the commanded gripper).
    @property
    def _controller_target(self) -> Any:
        return self._controller.target_position

    @property
    def _controller_yaw(self) -> Any:
        return self._controller.target_yaw

    @property
    def _controller_gripper(self) -> Any:
        return self._controller.gripper_opening

    def _controller_state_tensors(self) -> tuple[Any, ...]:
        controller = self._controller
        return (
            controller._target,  # noqa: SLF001 - broadcast copies in place
            controller._yaw,  # noqa: SLF001
            controller._gripper,  # noqa: SLF001
            self._last_actions,
        )

    def _reset_controller_state(
        self, indices: Any, controller_state: Mapping[str, Any] | None
    ) -> None:
        state = dict(controller_state or {})
        target = state.get("target_position", self._xpos[:, self.ee_body_id])
        yaw = state.get("target_yaw", self._measured_yaw())
        gripper = state.get("target_gripper", self._normalized_gripper_opening())

        def subset(value: Any, width: int | None) -> Any:
            tensor = self.torch.as_tensor(
                value, dtype=self.torch.float32, device=self._device
            )
            if int(tensor.shape[0]) == self.worlds_per_rank:
                tensor = tensor.index_select(0, indices)
            return tensor if width is None else tensor.reshape(-1, width)

        self._controller.reset_worlds(
            indices, subset(target, 3), subset(yaw, None), subset(gripper, None)
        )
        self._last_actions.index_fill_(0, indices, 0.0)
        # A reset world must hold its pose from the first substep, not chase
        # whatever the previous episode last commanded.
        self._controller.write_controls(self._ctrl)

    # -- observations -------------------------------------------------------

    def _measured_yaw(self) -> Any:
        """World angle of the finger-opening axis (``ee_base`` local +y).

        From the quaternion, so no ``xmat`` view is needed:
        ``R01 = 2(xy - wz)``, ``R11 = 1 - 2(x^2 + z^2)``. Matches
        ``WidowX200BatchedController.measured_yaw`` and the CDPR convention
        (yaw 0 puts the fingers on world +x).
        """

        quaternion = self._xquat[:, self.ee_body_id]
        w, x, y, z = quaternion.unbind(dim=-1)
        r01 = 2.0 * (x * y - w * z)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        return self.torch.atan2(r11, r01)

    def low_dim_observations(self) -> CDPRLowDimBatch:
        object_positions = self._xpos.index_select(
            1, self._object_body_ids_tensor
        )
        object_quaternions = self._xquat.index_select(
            1, self._object_body_ids_tensor
        )
        return CDPRLowDimBatch(
            ee_position=self._xpos[:, self.ee_body_id],
            ee_quaternion=self._xquat[:, self.ee_body_id],
            ee_yaw=self._measured_yaw(),
            gripper_opening=self._normalized_gripper_opening(),
            target_position=self._controller.target_position,
            # No cables. The slot is kept so export and the dataclass contract
            # hold; it carries the five joint angles instead.
            tendon_lengths=self._qpos.index_select(1, self._joint_qadr_tensor),
            object_positions=object_positions,
            object_quaternions=object_quaternions,
        )

    def controller_state(self) -> dict[str, Any]:
        import numpy as np

        def host(value: Any) -> Any:
            return np.asarray(value.detach().cpu().numpy(), dtype=np.float64).copy()

        return {
            "target": host(self._controller.target_position),
            "yaw": host(self._controller.target_yaw),
            "gripper": host(self._controller.gripper_opening),
            "reachable": host(self._controller.reachable.to(self.torch.float32)),
        }

    # -- stepping -----------------------------------------------------------

    def step(self, actions: Any, active_mask: Any) -> CDPRLowDimBatch:
        torch = self.torch
        action_tensor = torch.as_tensor(
            actions, dtype=torch.float32, device=self._device
        )
        active = torch.as_tensor(
            active_mask, dtype=torch.bool, device=self._device
        ).reshape(-1)
        if tuple(action_tensor.shape) != (self.worlds_per_rank, 5):
            raise ValueError(
                f"MJWarp actions must have shape ({self.worlds_per_rank}, 5), "
                f"got {tuple(action_tensor.shape)}."
            )
        if tuple(active.shape) != (self.worlds_per_rank,):
            raise ValueError(
                f"active_mask must have shape ({self.worlds_per_rank},), "
                f"got {tuple(active.shape)}."
            )
        action_tensor = action_tensor.clamp(-1.0, 1.0)
        masked_action = torch.where(
            active[:, None], action_tensor, torch.zeros_like(action_tensor)
        )
        controller = self._controller
        controller.integrate_actions(
            masked_action, active, self._xpos[:, self.ee_body_id]
        )
        # Clamp the STORED target to the workspace box as well as the solved
        # one. The leash already keeps it within 3 cm of the tool, but a target
        # parked below the floor would make the next upward commands spend
        # steps climbing back before the arm moves.
        controller._target.copy_(  # noqa: SLF001 - in-place, shape-stable
            torch.maximum(
                torch.minimum(controller._target, self._workspace_max),  # noqa: SLF001
                self._workspace_min,
            )
        )
        controller._yaw.clamp_(self._yaw_limits[0], self._yaw_limits[1])  # noqa: SLF001
        self._last_actions.copy_(masked_action)
        # IK depends only on the target, which is fixed for the whole action,
        # and MJWarp never writes ctrl -- so one solve per action is the same
        # servo setpoint the CDPR rewrites every substep.
        controller.write_controls(self._ctrl)

        with self.wp.ScopedDevice(str(self._device)):
            for _ in range(self.config.physics_substeps):
                self.mjw.step(self.model, self.data)
        self._contain_nonfinite_worlds()
        return self.low_dim_observations()

    def _update_cable_visuals(self) -> None:
        return None

    # -- teleports ----------------------------------------------------------

    def set_end_effector_poses(
        self,
        positions: Any,
        yaws: Any,
        *,
        zero_velocity: bool = True,
    ) -> None:
        """Solve the pose once and write it into ``qpos`` and ``ctrl`` alike.

        The CDPR teleports a free body and then re-solves cable preload; here
        the IK solution IS the pose, so the servos start satisfied and the arm
        does not lurch on the first step. Joint velocities are always zeroed:
        a teleported arm has no meaningful velocity to keep.
        """

        torch = self.torch
        position = torch.as_tensor(
            positions, dtype=torch.float32, device=self._device
        )
        yaw = torch.as_tensor(
            yaws, dtype=torch.float32, device=self._device
        ).reshape(-1)
        if tuple(position.shape) != (self.worlds_per_rank, 3):
            raise ValueError("End-effector reset positions must have shape [world, 3].")
        if tuple(yaw.shape) != (self.worlds_per_rank,):
            raise ValueError("End-effector reset yaws must have shape [world].")
        yaw = yaw.clamp(self._yaw_limits[0], self._yaw_limits[1])
        self._controller.set_end_effector_poses(
            self._qpos,
            self._ctrl,
            self._qvel,
            position,
            yaw,
            self._normalized_gripper_opening(),
        )
        self._qvel[:, self.right_finger_dofadr] = 0.0
        self._qvel[:, self.finger_dofadr] = 0.0
        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.forward(self.model, self.data)

    def set_gripper_openings(self, openings: Any) -> None:
        opening = self.torch.as_tensor(
            openings, dtype=self.torch.float32, device=self._device
        ).reshape(-1)
        if tuple(opening.shape) != (self.worlds_per_rank,):
            raise ValueError("One normalized gripper reset opening is required per world.")
        opening = opening.clamp(0.0, 1.0)
        low, high = self._finger_limits
        finger = low + opening * (high - low)
        # The two fingers are coupled by a joint equality; writing only one
        # would start every reset with the equality violated.
        self._qpos[:, self.finger_qadr] = finger
        self._qpos[:, self.right_finger_qadr] = finger
        self._qvel[:, self.finger_dofadr] = 0.0
        self._qvel[:, self.right_finger_dofadr] = 0.0
        self._ctrl[:, self.gripper_actuator_id] = finger
        self._controller._gripper.copy_(opening)  # noqa: SLF001
        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.forward(self.model, self.data)

    # -- provenance -----------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload.update(
            {
                "embodiment": self.embodiment,
                "controller_implementation": "batched_torch_widowx200_top_down_ik_v1",
                "widowx200_controller": self._controller.metadata(),
                "widowx200_yaw_limits": list(WIDOWX200_YAW_LIMITS),
                "widowx200_home_pose": {
                    "position": list(WIDOWX200_HOME_POSE["position"]),
                    "yaw": float(WIDOWX200_HOME_POSE["yaw"]),
                    "gripper": float(WIDOWX200_HOME_POSE["gripper"]),
                },
                "rendered_geom_groups": [0, 1, 2, 4],
                "camera_order": ["overview", "ee_camera", "ee_camera"],
            }
        )
        payload.pop("cable_visual_geom_group", None)
        return payload


def widowx200_backend_config_check(config: CDPRBackendConfig) -> None:
    """Refuse the CDPR control period on this arm.

    ``hold_steps`` 6 (14 ms) asks the WidowX for ~1 m/s at the tool, past its
    joint-rate limit; the arm is driven at 20 Hz (``hold_steps`` 24).
    """

    if int(config.hold_steps) < 12:
        raise ValueError(
            f"hold_steps={config.hold_steps} is the CDPR control period; the "
            "WidowX-200 runs hold_steps=24 (50 ms). Set it in the config's "
            "training.rl.args."
        )
