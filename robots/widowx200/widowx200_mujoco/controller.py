"""Task-space controller for the WidowX-200, in the CDPR's action contract.

The CDPR's controller turns a Cartesian target into four cable lengths and then
into four slider positions. This one turns the same Cartesian target into five
joint angles. Everything above the controller -- the normalized five-channel
action, the workspace clamp, the yaw integration, the normalized gripper, the
`hold_steps` substepping -- is identical, which is the whole point: the policy
cannot tell which arm is underneath.

Two consumers, one algebra:

* the host path (``WidowX200TaskSpaceController``) drives a single CPU MuJoCo
  model and duck-types the same methods ``policy_control`` already calls on the
  CDPR simulator, so ``apply_normalized_cdpr_action`` runs on this arm
  unchanged;
* the batched path (``joint_targets_from_task_targets``) is a pure array
  function over ``(nworld, ...)`` tensors with no host synchronisation and no
  Python branching, so the MJWarp backend can call it in place of
  ``_write_controller_controls``.

Both go through ``kinematics.top_down_ik``, so there is one place where the arm
geometry lives and one place a mistake in it can hide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .kinematics import (
    FINGER_LIMITS,
    JOINT_LIMITS,
    JOINT_NAMES,
    TOP_DOWN_PITCH,
    WX200,
    WidowX200Geometry,
    clamp_target_to_reach,
    top_down_ik,
)

__all__ = [
    "WidowX200MountPose",
    "WidowX200ControlSpec",
    "integrate_task_targets",
    "joint_targets_from_task_targets",
    "WidowX200TaskSpaceController",
]


@dataclass(frozen=True)
class WidowX200MountPose:
    """Where the arm's base_link sits in world coordinates.

    The scene owns this, not the robot MJCF -- ``wx200.xml`` describes an arm
    standing at the origin facing +x, and the scene wrapper repositions
    ``wx200_mount``. Keeping the mount here as well lets the controller convert
    world-frame targets (which is what the policy and every reward speak) into
    the base frame the IK needs.
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0

    def world_to_base(self, ops: Any, point: Any) -> Any:
        cos_y = math.cos(-self.yaw)
        sin_y = math.sin(-self.yaw)
        dx = point[..., 0] - self.position[0]
        dy = point[..., 1] - self.position[1]
        dz = point[..., 2] - self.position[2]
        return ops.stack([dx * cos_y - dy * sin_y, dx * sin_y + dy * cos_y, dz])

    def base_to_world(self, ops: Any, point: Any) -> Any:
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        x = point[..., 0]
        y = point[..., 1]
        z = point[..., 2]
        return ops.stack(
            [
                x * cos_y - y * sin_y + self.position[0],
                x * sin_y + y * cos_y + self.position[1],
                z + self.position[2],
            ]
        )


@dataclass(frozen=True)
class WidowX200ControlSpec:
    """Everything the controller needs that is not link geometry.

    The scales default to the CDPR's production values. They are not arbitrary
    -- 0.015 m per unit action at ``hold_steps=6`` is the step size the current
    curricula, shaping windows, and success radii were all tuned against -- so
    changing them is a retraining decision, not a tuning knob.
    """

    mount: WidowX200MountPose = field(default_factory=WidowX200MountPose)
    geometry: WidowX200Geometry = WX200
    action_step_xyz: float = 0.015
    action_step_yaw: float = 0.08
    action_step_gripper: float = 0.05
    # 24, not the CDPR's 6. One action is 1 + hold_steps physics steps, so at
    # the 2 ms MJWarp timestep this is a 50 ms / 20 Hz control period against
    # the CDPR's 14 ms / 71 Hz. 71 Hz asks this arm for ~1.07 m/s at every
    # step, which is past its joint-rate limit; 20 Hz is also the rate a real
    # WidowX is actually driven at.
    hold_steps: int = 24
    pitch: float = TOP_DOWN_PITCH
    elbow_up: bool = True
    # World-frame clamp on the controller target, same meaning as
    # CDPRBackendConfig.workspace_*: the box the policy may drive inside. It is
    # applied BEFORE the reach clamp, so a config can be stricter than the arm
    # but never looser.
    workspace_x: tuple[float, float] = (-0.30, 0.30)
    workspace_y: tuple[float, float] = (-0.30, 0.30)
    workspace_z: tuple[float, float] = (0.16, 0.42)
    # Kinematic safety margin held back from full extension. At the very edge
    # the arm is singular in the radial direction: a millimetre of commanded
    # motion needs an unbounded joint rate, the position servo lags, and the
    # measured end-effector stops tracking the target the reward is written
    # against.
    reach_margin: float = 0.02
    # Keep-out cylinder around the waist axis. The arm can physically fold
    # closer, but on its own axis the waist is singular: a centimetre of
    # commanded XY needs a large, fast waist swing. The policy commands XY
    # deltas, so this region turns small actions into violent ones -- and the
    # arm's own base occupies most of it anyway.
    min_reach_radius: float = 0.15
    # How far the commanded target may run ahead of the MEASURED end effector.
    #
    # The CDPR builds each target from the measured pose (`ee + delta`), which
    # is fine for a light platform on stiff cables: it tracks a 15 mm step
    # inside one 14 ms action window, so measured and commanded coincide. This
    # arm does not. Measured here: at 14 ms it recovers 12% of a 15 mm step, at
    # 50 ms 37% -- so `ee + delta` silently ATTENUATES every action, and the
    # policy's effective step size becomes a function of how fast the arm
    # happens to be moving.
    #
    # Integrating the target instead (`target + delta`) keeps the commanded
    # step exact and turns the servo lag into a constant trailing distance.
    # Uncapped, that distance winds up to ~0.18 m and a reversal then does
    # nothing for a dozen steps. The leash caps it, so the trade is explicit:
    #
    #   leash    effective step   coast after commands cease
    #   0.020    7.2 mm           12.7 mm
    #   0.030    10.4 mm          19.1 mm
    #   0.045    13.1 mm          26.5 mm
    #
    # (hold_steps=24; measured by tests/test_widowx200_controller.py, which
    # pins these so a gain or damping edit cannot move them unnoticed.)
    #
    # 0.030 is the default: it keeps the effective step at 69% of the CDPR's
    # 15 mm while the coast stays inside the 20 mm move_to success radius, so a
    # policy can still stop on target. Raising it buys speed and spends
    # stopping precision.
    target_leash: float = 0.030

    @property
    def sim_steps_per_policy_action(self) -> int:
        return max(1, int(self.hold_steps) + 1)


def _ops_for(sample: Any) -> Any:
    from . import kinematics

    return kinematics._ops_for(sample)  # noqa: SLF001 - one shared array shim


def integrate_task_targets(
    action: Any,
    ee_position: Any,
    target_position: Any,
    target_yaw: Any,
    gripper_opening: Any,
    spec: WidowX200ControlSpec,
) -> dict[str, Any]:
    """Advance the controller target by one normalized action.

    ``action`` is ``(..., 5)`` in ``[-1, 1]``: XYZ, yaw, gripper. This is the
    exact counterpart of the CDPR backend's target integration in
    ``MJLabMJWarpCDPRBackend.step``, with one difference -- the XYZ target is
    integrated from the previous TARGET rather than from the measured pose, and
    then leashed to it. See ``WidowX200ControlSpec.target_leash`` for why.

    Pure array ops on ``(nworld, ...)``: no branching, no host reads, so the
    batched backend can call it inside its substep loop.
    """

    ops = _ops_for(action)
    action = ops.clip(ops.asarray(action), -1.0, 1.0)
    ee_position = ops.asarray(ee_position, like=action)
    target = ops.asarray(target_position, like=action)

    proposed = target + action[..., :3] * float(spec.action_step_xyz)
    offset = proposed - ee_position
    distance = ops.sqrt(
        offset[..., 0] ** 2 + offset[..., 1] ** 2 + offset[..., 2] ** 2
    )
    scale = ops.minimum(
        float(spec.target_leash) / ops.maximum(distance, 1e-9), 1.0
    )
    leashed = ops.stack(
        [
            ee_position[..., 0] + offset[..., 0] * scale,
            ee_position[..., 1] + offset[..., 1] * scale,
            ee_position[..., 2] + offset[..., 2] * scale,
        ]
    )

    yaw = ops.asarray(target_yaw, like=action) + action[..., 3] * float(
        spec.action_step_yaw
    )
    gripper = ops.clip(
        ops.asarray(gripper_opening, like=action)
        + action[..., 4] * float(spec.action_step_gripper),
        0.0,
        1.0,
    )
    return {"target_position": leashed, "target_yaw": yaw, "gripper_opening": gripper}


def joint_targets_from_task_targets(
    target_position: Any,
    target_yaw: Any,
    gripper_opening: Any,
    spec: WidowX200ControlSpec,
) -> dict[str, Any]:
    """World task targets -> joint angles and the gripper actuator command.

    Shapes are ``(..., 3)``, ``(...)``, ``(...)`` in, ``(..., 5)`` and ``(...)``
    out. Total by construction: an unreachable request is clamped and reported
    in ``reachable`` rather than raised, because in the batched path one bad
    world must not stop 128 good ones.
    """

    ops = _ops_for(target_position)
    position = ops.asarray(target_position)
    yaw = ops.asarray(target_yaw, like=position)
    opening = ops.asarray(gripper_opening, like=position)

    # World box first, so a config's workspace stays authoritative.
    world = ops.stack(
        [
            ops.clip(position[..., 0], *spec.workspace_x),
            ops.clip(position[..., 1], *spec.workspace_y),
            ops.clip(position[..., 2], *spec.workspace_z),
        ]
    )
    base_point = spec.mount.world_to_base(ops, world)
    base_yaw = yaw - spec.mount.yaw

    # Push the target out of the waist keep-out along its own bearing, so the
    # commanded direction survives and only the radius moves.
    radius = ops.sqrt(base_point[..., 0] ** 2 + base_point[..., 1] ** 2)
    push = ops.maximum(spec.min_reach_radius / ops.maximum(radius, 1e-6), 1.0)
    base_point = ops.stack(
        [base_point[..., 0] * push, base_point[..., 1] * push, base_point[..., 2]]
    )

    # `clamp_to_reach` returns the joints for the clamped point but reports
    # reachability of what was asked for, which is what a caller needs to know.
    clamped_base = clamp_target_to_reach(
        base_point,
        geometry=spec.geometry,
        gamma=spec.pitch,
        margin=spec.reach_margin,
    )
    requested = top_down_ik(
        base_point,
        base_yaw,
        geometry=spec.geometry,
        gamma=spec.pitch,
        elbow_up=spec.elbow_up,
        clamp_to_reach=False,
    )
    solution = top_down_ik(
        clamped_base,
        base_yaw,
        geometry=spec.geometry,
        gamma=spec.pitch,
        elbow_up=spec.elbow_up,
        clamp_to_reach=False,
    )

    low, high = FINGER_LIMITS
    gripper_ctrl = low + ops.clip(opening, 0.0, 1.0) * (high - low)
    return {
        "q": solution.q,
        "gripper_ctrl": gripper_ctrl,
        "reachable": requested.reachable,
        "achieved_target": spec.mount.base_to_world(ops, clamped_base),
    }


class WidowX200TaskSpaceController:
    """Single-model host controller with the CDPR simulator's method names.

    ``robots/cdpr/cdpr_mujoco/policy_control.apply_normalized_cdpr_action``
    talks to its simulator through ``get_end_effector_position`` /
    ``set_end_effector_target`` / ``get_yaw`` / ``set_yaw`` /
    ``get_gripper_target`` / ``set_gripper`` / ``run_simulation_step``. Every
    one of those is implemented here against the arm, so the existing action
    codec, diagnostics, and reference-episode tooling run on this embodiment
    with no branch on robot type.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        spec: WidowX200ControlSpec | None = None,
        mujoco: Any = None,
    ) -> None:
        if mujoco is None:
            import mujoco as mujoco_module

            mujoco = mujoco_module
        self._mj = mujoco
        self.model = model
        self.data = data
        self.spec = spec or WidowX200ControlSpec()

        self._joint_qadr = [
            int(model.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in JOINT_NAMES
        ]
        self._actuator_ids = [
            self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
            for name in JOINT_NAMES
        ]
        self._gripper_actuator = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
        self._finger_qadr = int(
            model.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, "left_finger")]
        )
        self._ee_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "ee_base")

        self._target_yaw = 0.0
        self._gripper_target = 1.0
        self._controller_target = self.get_end_effector_position()

    def _id(self, objtype: Any, name: str) -> int:
        index = self._mj.mj_name2id(self.model, objtype, name)
        if index == -1:
            raise KeyError(f"WidowX-200 model is missing {name!r}.")
        return int(index)

    # -- state ------------------------------------------------------------

    def get_end_effector_position(self) -> Any:
        import numpy as np

        return np.asarray(self.data.xpos[self._ee_body], dtype=np.float64).copy()

    def get_yaw(self) -> float:
        return float(self._target_yaw)

    def get_gripper_target(self) -> float:
        return float(self._gripper_target)

    def get_gripper_opening(self) -> float:
        low, high = FINGER_LIMITS
        value = (float(self.data.qpos[self._finger_qadr]) - low) / (high - low)
        return float(min(max(value, 0.0), 1.0))

    def get_joint_positions(self) -> Any:
        import numpy as np

        return np.asarray(
            [float(self.data.qpos[adr]) for adr in self._joint_qadr], dtype=np.float64
        )

    # -- commands ---------------------------------------------------------

    def set_end_effector_target(self, xyz: Any) -> dict[str, Any]:
        import numpy as np

        result = joint_targets_from_task_targets(
            np.asarray(xyz, dtype=np.float64).reshape(3),
            np.float64(self._target_yaw),
            np.float64(self._gripper_target),
            self.spec,
        )
        q = np.asarray(result["q"], dtype=np.float64).reshape(5)
        for actuator, value in zip(self._actuator_ids, q):
            self.data.ctrl[actuator] = value
        self.data.ctrl[self._gripper_actuator] = float(result["gripper_ctrl"])
        self._last_solution = result
        return result

    def set_yaw(self, yaw: float) -> None:
        self._target_yaw = float(yaw)
        self.set_end_effector_target(self._controller_target)

    def set_gripper(self, opening: float) -> None:
        self._gripper_target = float(min(max(float(opening), 0.0), 1.0))
        low, high = FINGER_LIMITS
        self.data.ctrl[self._gripper_actuator] = low + self._gripper_target * (
            high - low
        )

    def apply_normalized_action(self, action: Any) -> dict[str, Any]:
        """One policy action: integrate, solve, and run the substeps.

        This is the arm's equivalent of
        ``policy_control.apply_normalized_cdpr_action`` and takes the identical
        five-channel normalized vector. It is the preferred entry point --
        ``set_end_effector_target`` alone is the absolute-target primitive and
        does not apply the leash.
        """

        import numpy as np

        action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0
        )
        if action.size != 5:
            raise ValueError(f"Expected a 5-channel action, got {action.shape}.")

        integrated = integrate_task_targets(
            action,
            self.get_end_effector_position(),
            self._controller_target,
            np.float64(self._target_yaw),
            np.float64(self._gripper_target),
            self.spec,
        )
        self._controller_target = np.asarray(
            integrated["target_position"], dtype=np.float64
        ).reshape(3)
        self._target_yaw = float(integrated["target_yaw"])
        self._gripper_target = float(integrated["gripper_opening"])

        result = self.set_end_effector_target(self._controller_target)
        for _ in range(self.spec.sim_steps_per_policy_action):
            self.run_simulation_step()

        result = dict(result)
        result.update(
            {
                "commanded_action": action,
                "target_xyz": self._controller_target.copy(),
                "target_yaw": self._target_yaw,
                "gripper_target": self._gripper_target,
                "ee_position": self.get_end_effector_position(),
                "gripper_opening": self.get_gripper_opening(),
                "sim_steps": self.spec.sim_steps_per_policy_action,
            }
        )
        return result

    # -- stepping ---------------------------------------------------------

    def run_simulation_step(self, capture_frame: bool = False) -> None:
        self._mj.mj_step(self.model, self.data)

    def reset_to_pose(
        self,
        position: Sequence[float],
        yaw: float = 0.0,
        gripper: float = 1.0,
    ) -> dict[str, Any]:
        """Teleport to a task-space pose, the arm analogue of a CDPR reset.

        The CDPR reset writes a free-joint pose and recomputes tendon preload.
        Here the equivalent is solving the pose once and writing both ``qpos``
        and ``ctrl``, so the servo starts already satisfied and the arm does not
        lurch on the first step of an episode.
        """

        import numpy as np

        self._target_yaw = float(yaw)
        self._gripper_target = float(min(max(float(gripper), 0.0), 1.0))
        result = joint_targets_from_task_targets(
            np.asarray(position, dtype=np.float64).reshape(3),
            np.float64(self._target_yaw),
            np.float64(self._gripper_target),
            self.spec,
        )
        q = np.asarray(result["q"], dtype=np.float64).reshape(5)
        for adr, actuator, value in zip(
            self._joint_qadr, self._actuator_ids, q
        ):
            self.data.qpos[adr] = value
            self.data.ctrl[actuator] = value
        low, high = FINGER_LIMITS
        finger = low + self._gripper_target * (high - low)
        self.data.qpos[self._finger_qadr] = finger
        right = int(
            self.model.jnt_qposadr[
                self._id(self._mj.mjtObj.mjOBJ_JOINT, "right_finger")
            ]
        )
        self.data.qpos[right] = finger
        self.data.ctrl[self._gripper_actuator] = finger
        self.data.qvel[:] = 0.0
        self._controller_target = np.asarray(
            result["achieved_target"], dtype=np.float64
        ).reshape(3)
        self._mj.mj_forward(self.model, self.data)
        self._last_solution = result
        return result
