"""GPU-resident WidowX-200 controller for the batched MJWarp rollout.

This is the arm's replacement for the CDPR-specific block inside
``MJLabMJWarpCDPRBackend``: ``_calibrate_host_cdpr``, ``_initialize_controller_state``,
``_write_controller_controls``, the target/yaw/gripper integration in ``step``,
and ``set_end_effector_poses``. Everything else in that backend -- world
allocation, partial reset, group broadcast, rendering, contacts, object
catalogs, visual randomization, export, metadata -- is embodiment-independent
and is meant to be reused as it stands.

Two properties the batched path requires, and which this class is written to
keep:

* **No host synchronisation.** Every method is pure tensor arithmetic. Nothing
  branches on a tensor value, nothing calls ``.item()``, nothing indexes with a
  Python conditional. A single ``if tensor > x`` here would serialise 128
  worlds through the host on every substep.

* **Total, never raising.** An unreachable request is clamped and flagged in
  ``reachable``. The CDPR could diverge one world at a time and the backend
  contains it (``_contain_nonfinite_worlds``); an IK that raised would take the
  whole rank down instead.

The calibration story is much shorter than the CDPR's. That backend has to
finite-difference four tendon lengths against four slider coordinates at
startup because the cable map has no closed form. This arm's map does, so
startup only has to resolve names to indices -- there is no numerical
calibration to drift, and no preload to restore after a reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .controller import (
    WidowX200ControlSpec,
    integrate_task_targets,
    joint_targets_from_task_targets,
)
from .kinematics import FINGER_LIMITS, JOINT_NAMES

__all__ = ["WidowX200ModelIndices", "WidowX200BatchedController"]


@dataclass(frozen=True)
class WidowX200ModelIndices:
    """Name-to-index resolution, done once on the host model.

    Every lookup raises if the name is missing. That is deliberate: a model
    edit that renames a joint should stop the run at startup, not produce a
    controller that writes into the wrong actuator slot and trains for a week.
    """

    joint_qadr: tuple[int, ...]
    joint_dofadr: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    gripper_actuator_id: int
    left_finger_qadr: int
    right_finger_qadr: int
    ee_body_id: int

    @classmethod
    def resolve(cls, model: Any, mujoco: Any) -> "WidowX200ModelIndices":
        def joint(name: str) -> int:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if index == -1:
                raise KeyError(f"WidowX-200 model has no joint {name!r}.")
            return int(index)

        def actuator(name: str) -> int:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if index == -1:
                raise KeyError(f"WidowX-200 model has no actuator {name!r}.")
            return int(index)

        def body(name: str) -> int:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if index == -1:
                raise KeyError(f"WidowX-200 model has no body {name!r}.")
            return int(index)

        joints = [joint(name) for name in JOINT_NAMES]
        return cls(
            joint_qadr=tuple(int(model.jnt_qposadr[index]) for index in joints),
            joint_dofadr=tuple(int(model.jnt_dofadr[index]) for index in joints),
            actuator_ids=tuple(actuator(f"{name}_pos") for name in JOINT_NAMES),
            gripper_actuator_id=actuator("act_gripper"),
            left_finger_qadr=int(model.jnt_qposadr[joint("left_finger")]),
            right_finger_qadr=int(model.jnt_qposadr[joint("right_finger")]),
            ee_body_id=body("ee_base"),
        )


class WidowX200BatchedController:
    """Per-world Cartesian target, yaw, and gripper opening for ``nworld`` arms."""

    def __init__(
        self,
        *,
        torch: Any,
        device: Any,
        nworld: int,
        spec: WidowX200ControlSpec,
        indices: WidowX200ModelIndices,
    ) -> None:
        self.torch = torch
        self.device = device
        self.nworld = int(nworld)
        self.spec = spec
        self.indices = indices

        zeros = torch.zeros((self.nworld, 3), dtype=torch.float32, device=device)
        self._target = zeros.clone()
        self._yaw = torch.zeros(self.nworld, dtype=torch.float32, device=device)
        self._gripper = torch.ones(self.nworld, dtype=torch.float32, device=device)
        self._reachable = torch.ones(
            self.nworld, dtype=torch.bool, device=device
        )
        self._joint_actuator_index = torch.as_tensor(
            list(indices.actuator_ids), dtype=torch.int64, device=device
        )

    # -- state ------------------------------------------------------------

    @property
    def target_position(self) -> Any:
        return self._target

    @property
    def target_yaw(self) -> Any:
        return self._yaw

    @property
    def gripper_opening(self) -> Any:
        return self._gripper

    @property
    def reachable(self) -> Any:
        """Per-world flag: was the LAST commanded target inside the arm's reach.

        Worth logging. A rising fraction means the policy is pressing into the
        workspace wall, which on the CDPR would just have been clamped silently
        and looked like a policy that had stopped improving.
        """

        return self._reachable

    def state_dict(self) -> dict[str, Any]:
        return {
            "target_position": self._target.clone(),
            "target_yaw": self._yaw.clone(),
            "gripper_opening": self._gripper.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._target.copy_(state["target_position"])
        self._yaw.copy_(state["target_yaw"])
        self._gripper.copy_(state["gripper_opening"])

    # -- reset and broadcast ---------------------------------------------

    def reset_worlds(
        self,
        world_indices: Any,
        positions: Any,
        yaws: Any,
        openings: Any,
    ) -> None:
        """Seed controller state for a subset of worlds, as the backend resets."""

        torch = self.torch
        world_indices = torch.as_tensor(
            world_indices, dtype=torch.int64, device=self.device
        ).reshape(-1)
        self._target.index_copy_(
            0,
            world_indices,
            torch.as_tensor(positions, dtype=torch.float32, device=self.device).reshape(
                -1, 3
            ),
        )
        self._yaw.index_copy_(
            0,
            world_indices,
            torch.as_tensor(yaws, dtype=torch.float32, device=self.device).reshape(-1),
        )
        self._gripper.index_copy_(
            0,
            world_indices,
            torch.as_tensor(
                openings, dtype=torch.float32, device=self.device
            ).reshape(-1),
        )

    def broadcast_group_state(self, base_world_indices: Any, group_size: int) -> None:
        """Copy one base world's controller state to every candidate in its group.

        GRPO requires the candidates in a group to start identical; the CDPR
        backend does the same for its cable state. Controller state is part of
        the initial condition, so it has to be broadcast alongside ``qpos``.
        """

        torch = self.torch
        base = torch.as_tensor(
            base_world_indices, dtype=torch.int64, device=self.device
        ).reshape(-1)
        repeated = base.repeat_interleave(int(group_size))
        self._target.copy_(self._target.index_select(0, repeated))
        self._yaw.copy_(self._yaw.index_select(0, repeated))
        self._gripper.copy_(self._gripper.index_select(0, repeated))

    # -- per-action integration -------------------------------------------

    def integrate_actions(
        self, actions: Any, active_mask: Any, ee_position: Any
    ) -> None:
        """Advance the controller target by one masked normalized action.

        ``actions`` is ``(nworld, 5)``; completed worlds are held frozen by
        ``active_mask`` rather than removed, exactly as the CDPR collector does
        -- fixed shapes are the whole reason the batched path is fast.
        """

        torch = self.torch
        actions = torch.as_tensor(
            actions, dtype=torch.float32, device=self.device
        ).reshape(self.nworld, 5)
        active = torch.as_tensor(
            active_mask, dtype=torch.bool, device=self.device
        ).reshape(self.nworld)
        masked = torch.where(active[:, None], actions, torch.zeros_like(actions))

        updated = integrate_task_targets(
            masked,
            ee_position,
            self._target,
            self._yaw,
            self._gripper,
            self.spec,
        )
        self._target.copy_(
            torch.where(active[:, None], updated["target_position"], self._target)
        )
        self._yaw.copy_(torch.where(active, updated["target_yaw"], self._yaw))
        self._gripper.copy_(
            torch.where(active, updated["gripper_opening"], self._gripper)
        )

    # -- per-substep control write ----------------------------------------

    def write_controls(self, ctrl: Any) -> Any:
        """Solve IK for the current targets and write the actuator commands.

        Called once per physics substep, like the CDPR's
        ``_write_controller_controls``, so the servo chases a fixed setpoint
        for the whole ``1 + hold_steps`` window rather than being re-aimed
        mid-flight.
        """

        solution = joint_targets_from_task_targets(
            self._target, self._yaw, self._gripper, self.spec
        )
        self._reachable = solution["reachable"]
        ctrl[:, self._joint_actuator_index] = solution["q"]
        ctrl[:, self.indices.gripper_actuator_id] = solution["gripper_ctrl"]
        return ctrl

    # -- teleport ----------------------------------------------------------

    def set_end_effector_poses(
        self,
        qpos: Any,
        ctrl: Any,
        qvel: Any,
        positions: Any,
        yaws: Any,
        openings: Any,
        world_indices: Any | None = None,
    ) -> Any:
        """Write a batched reset pose straight into ``qpos``/``ctrl``.

        The CDPR equivalent has to recompute tendon preload after a teleport;
        here the IK solution IS the pose, so ``qpos`` and ``ctrl`` are set to
        the same joint vector and the servo starts already satisfied. Without
        that the arm lurches on an episode's first step, which is a transient
        the policy would see in every single rollout.
        """

        torch = self.torch
        solution = joint_targets_from_task_targets(
            torch.as_tensor(positions, dtype=torch.float32, device=self.device).reshape(
                -1, 3
            ),
            torch.as_tensor(yaws, dtype=torch.float32, device=self.device).reshape(-1),
            torch.as_tensor(
                openings, dtype=torch.float32, device=self.device
            ).reshape(-1),
            self.spec,
        )
        q = solution["q"]
        gripper = solution["gripper_ctrl"]

        rows = (
            torch.arange(self.nworld, dtype=torch.int64, device=self.device)
            if world_indices is None
            else torch.as_tensor(
                world_indices, dtype=torch.int64, device=self.device
            ).reshape(-1)
        )
        for column, qadr in enumerate(self.indices.joint_qadr):
            qpos[rows, qadr] = q[:, column]
        for column, actuator in enumerate(self.indices.actuator_ids):
            ctrl[rows, actuator] = q[:, column]
        qpos[rows, self.indices.left_finger_qadr] = gripper
        qpos[rows, self.indices.right_finger_qadr] = gripper
        ctrl[rows, self.indices.gripper_actuator_id] = gripper
        for dofadr in self.indices.joint_dofadr:
            qvel[rows, dofadr] = 0.0

        self.reset_worlds(rows, solution["achieved_target"], yaws, openings)
        return solution

    # -- observations ------------------------------------------------------

    def normalized_gripper_opening(self, qpos: Any) -> Any:
        low, high = FINGER_LIMITS
        return ((qpos[:, self.indices.left_finger_qadr] - low) / (high - low)).clamp(
            0.0, 1.0
        )

    def measured_yaw(self, xmat: Any) -> Any:
        """World yaw of the finger-opening axis, from ``ee_base``'s rotation.

        The CDPR reads its yaw straight off a hinge joint. This arm has no such
        joint -- yaw is distributed across waist and wrist_rotate -- so the
        observation is derived from the frame. It must agree with what
        ``top_down_ik`` was asked for, which
        ``tests/test_widowx200_embodiment.py`` checks against MuJoCo.
        """

        rotation = xmat[:, self.indices.ee_body_id].reshape(-1, 3, 3)
        return self.torch.atan2(rotation[:, 1, 1], rotation[:, 0, 1])

    def metadata(self) -> dict[str, Any]:
        """Checkpoint fields, in the spirit of the CDPR backend's ``metadata``.

        A checkpoint that does not record the control contract cannot be safely
        resumed: these five numbers change what an action MEANS, so a resume
        that silently changed any of them would continue training a different
        problem.
        """

        return {
            "embodiment": "widowx200_5dof",
            "controller": "top_down_task_space_ik",
            "action_step_xyz": float(self.spec.action_step_xyz),
            "action_step_yaw": float(self.spec.action_step_yaw),
            "action_step_gripper": float(self.spec.action_step_gripper),
            "hold_steps": int(self.spec.hold_steps),
            "target_leash": float(self.spec.target_leash),
            "pitch": float(self.spec.pitch),
            "mount_position": list(self.spec.mount.position),
            "mount_yaw": float(self.spec.mount.yaw),
            "workspace_x": list(self.spec.workspace_x),
            "workspace_y": list(self.spec.workspace_y),
            "workspace_z": list(self.spec.workspace_z),
            "min_reach_radius": float(self.spec.min_reach_radius),
            "reach_margin": float(self.spec.reach_margin),
            "joint_names": list(JOINT_NAMES),
            "finger_limits": list(FINGER_LIMITS),
        }
