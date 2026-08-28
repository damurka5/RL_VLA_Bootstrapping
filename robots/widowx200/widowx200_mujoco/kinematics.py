"""Closed-form kinematics for the Interbotix WidowX-200, in the CDPR action space.

Why closed form, and why "top-down".
------------------------------------

The whole training stack -- rewards, curricula, success predicates, the SmolVLA
residual, and every checkpoint from phase 1 through phase 5 -- is written
against a five-channel normalized action ``[x, y, z, yaw, gripper]`` whose XYZ
entries are *world-frame end-effector deltas*.  That contract is not a CDPR
detail; it is the interface the policy learned.  A serial arm can honour it
exactly, provided the controller turns a Cartesian target back into joint
angles every step.

The WidowX-200 has five revolute joints:

    waist (z)  shoulder (y)  elbow (y)  wrist_angle (y)  wrist_rotate (x)

Three of them (shoulder, elbow, wrist_angle) share the +y axis, so they move the
tool inside one vertical plane and control exactly three quantities there: the
in-plane position (two) and the tool pitch (one).  The waist selects the plane
and the wrist_rotate spins the tool about its own approach axis.  Five joints,
six task quantities -- one has to be given up, and the natural choice is pitch.

Pinning pitch to "straight down" (the tool's approach axis anti-parallel to
world +z) makes the remaining map ``(x, y, z, yaw) -> (q1..q5)`` square,
decoupled, and solvable in closed form with no iteration, no Jacobian, and no
null space.  It also matches how these arms are driven in practice -- BridgeData
and the Interbotix ``set_ee_pose_components`` path both command Cartesian poses
with a fixed approach -- so the simulated interface stays deployment-faithful.

The cost is real and worth stating: with pitch pinned there are no side or
angled approaches.  Tasks needing them (reaching into a mug, say) require
promoting pitch to a sixth action channel, which is a checkpoint-incompatible
change to the action space.  ``TOP_DOWN_PITCH`` is therefore a parameter here
rather than a constant baked into the algebra: ``top_down_ik`` accepts any
``gamma`` and the six-channel variant is a config change, not a rewrite.

Geometry source
---------------

Every constant below is transcribed from the official Interbotix URDF,
``interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/wx200.urdf.xacro``
(BSD-3, Trossen Robotics).  ``tests/test_widowx200_kinematics.py`` checks this
module against MuJoCo's own forward kinematics on the compiled MJCF, so a
transcription error fails a test rather than silently producing a controller
that tracks the wrong point -- the failure mode that cost ``pick_up`` 10M steps
on the CDPR (see ``cdpr_gripper_geometry``).

Angle convention
----------------

Both IK and FK work in the vertical arm plane, in coordinates ``(r, z)`` where
``r`` is distance along the waist direction.  In-plane angles are measured
**from +z toward +r**, which makes each segment a clean ``L * (sin, cos)``:

    alpha = q_shoulder + UPPER_ARM_SKEW          upper-arm direction
    beta  = q_shoulder + q_elbow + pi/2          forearm direction
    gamma = alpha_sum  + q_wrist_angle + pi/2    tool approach direction

``gamma = pi`` is straight down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "JOINT_NAMES",
    "JOINT_LIMITS",
    "FINGER_LIMITS",
    "TOP_DOWN_PITCH",
    "WidowX200Geometry",
    "WX200",
    "IKSolution",
    "top_down_ik",
    "forward_kinematics",
    "clamp_target_to_reach",
    "reach_envelope",
]


JOINT_NAMES: tuple[str, ...] = (
    "waist",
    "shoulder",
    "elbow",
    "wrist_angle",
    "wrist_rotate",
)

# Radians, straight from the URDF <limit> tags. The URDF writes the waist and
# wrist_rotate limits as +/-(pi - pi_offset) with pi_offset = 0.0; they are kept
# a hair inside +/-pi here so a wrapped command never lands exactly on the stop.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "waist": (-math.pi + 1e-3, math.pi - 1e-3),
    "shoulder": (math.radians(-108.0), math.radians(113.0)),
    "elbow": (math.radians(-108.0), math.radians(93.0)),
    "wrist_angle": (math.radians(-100.0), math.radians(123.0)),
    "wrist_rotate": (-math.pi + 1e-3, math.pi - 1e-3),
}

# left_finger travel, metres. right_finger mirrors it (URDF `mimic`, realised as
# a MuJoCo joint equality). 0.015 is closed, 0.037 is fully open.
FINGER_LIMITS: tuple[float, float] = (0.015, 0.037)

# gamma for a straight-down approach.
TOP_DOWN_PITCH: float = math.pi


@dataclass(frozen=True)
class WidowX200Geometry:
    """Link geometry of the WidowX-200, reduced to the planar IK parameters."""

    # Height of the shoulder joint above base_link: waist 0.072 + shoulder
    # 0.03865.
    shoulder_height: float = 0.072 + 0.03865
    # shoulder -> elbow is (0.05, 0, 0.2): a skewed link, not a straight one.
    upper_arm_dx: float = 0.05
    upper_arm_dz: float = 0.20
    # elbow -> wrist_angle.
    forearm_length: float = 0.20
    # wrist_angle -> ee_gripper_link, summed along the wrist x axis:
    # wrist_rotate 0.065 + ee_arm 0.043 + ee_bar 0.023 + ee_gripper 0.027575.
    wrist_length: float = 0.065 + 0.043 + 0.023 + 0.027575

    @property
    def upper_arm_length(self) -> float:
        return math.hypot(self.upper_arm_dx, self.upper_arm_dz)

    @property
    def upper_arm_skew(self) -> float:
        """Built-in lean of the upper arm, measured from +z toward +r."""

        return math.atan2(self.upper_arm_dx, self.upper_arm_dz)

    @property
    def max_planar_reach(self) -> float:
        """Shoulder-to-wrist distance at full extension."""

        return self.upper_arm_length + self.forearm_length

    @property
    def min_planar_reach(self) -> float:
        """Shoulder-to-wrist distance with the elbow fully folded."""

        return abs(self.upper_arm_length - self.forearm_length)


WX200 = WidowX200Geometry()
UPPER_ARM_SKEW = WX200.upper_arm_skew


class _NumpyOps:
    """Array-module shim so one algebra serves NumPy hosts and Torch batches."""

    def __init__(self) -> None:
        import numpy as np

        self.np = np

    def asarray(self, value: Any, like: Any = None) -> Any:
        return self.np.asarray(value, dtype=self.np.float64)

    def sqrt(self, value: Any) -> Any:
        return self.np.sqrt(value)

    def atan2(self, y: Any, x: Any) -> Any:
        return self.np.arctan2(y, x)

    def clip(self, value: Any, low: float, high: float) -> Any:
        return self.np.clip(value, low, high)

    def minimum(self, a: Any, b: Any) -> Any:
        return self.np.minimum(a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        return self.np.maximum(a, b)

    def acos(self, value: Any) -> Any:
        return self.np.arccos(value)

    def sin(self, value: Any) -> Any:
        return self.np.sin(value)

    def cos(self, value: Any) -> Any:
        return self.np.cos(value)

    def where(self, mask: Any, a: Any, b: Any) -> Any:
        return self.np.where(mask, a, b)

    def stack(self, values: list[Any]) -> Any:
        return self.np.stack(values, axis=-1)

    def full_like(self, value: Any, fill: float) -> Any:
        return self.np.full_like(value, fill)


class _TorchOps:
    """Same algebra over torch tensors, kept free of host synchronisation."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch

    def asarray(self, value: Any, like: Any = None) -> Any:
        if isinstance(value, self.torch.Tensor):
            return value
        kwargs: dict[str, Any] = {"dtype": self.torch.float32}
        if like is not None:
            kwargs["device"] = like.device
        return self.torch.as_tensor(value, **kwargs)

    def sqrt(self, value: Any) -> Any:
        return self.torch.sqrt(value)

    def atan2(self, y: Any, x: Any) -> Any:
        return self.torch.atan2(y, x)

    def clip(self, value: Any, low: float, high: float) -> Any:
        return value.clamp(low, high)

    def minimum(self, a: Any, b: Any) -> Any:
        return self.torch.minimum(a, self.torch.as_tensor(b, dtype=a.dtype, device=a.device))

    def maximum(self, a: Any, b: Any) -> Any:
        return self.torch.maximum(a, self.torch.as_tensor(b, dtype=a.dtype, device=a.device))

    def acos(self, value: Any) -> Any:
        return self.torch.acos(value)

    def sin(self, value: Any) -> Any:
        return self.torch.sin(value)

    def cos(self, value: Any) -> Any:
        return self.torch.cos(value)

    def where(self, mask: Any, a: Any, b: Any) -> Any:
        return self.torch.where(mask, a, b)

    def stack(self, values: list[Any]) -> Any:
        return self.torch.stack(values, dim=-1)

    def full_like(self, value: Any, fill: float) -> Any:
        return self.torch.full_like(value, fill)


def _ops_for(sample: Any) -> Any:
    """Pick the array module from the value itself, not from a flag."""

    module = type(sample).__module__.split(".")[0]
    if module == "torch":
        import torch

        return _TorchOps(torch)
    return _NumpyOps()


@dataclass(frozen=True)
class IKSolution:
    """Joint angles plus the evidence needed to trust them.

    ``reachable`` is False where the requested point lay outside the annulus or
    the joint box, in which case ``q`` holds the closest honoured pose rather
    than NaN. The batched controller needs a total function -- one unreachable
    world must not poison a batch of 128 -- so failure is reported in a mask and
    never raised.
    """

    q: Any
    reachable: Any
    clamped_target: Any


def _planar_ik(
    ops: Any,
    radius: Any,
    height: Any,
    gamma: Any,
    geometry: WidowX200Geometry,
    elbow_up: bool,
) -> tuple[Any, Any, Any, Any]:
    """Solve shoulder/elbow/wrist_angle for one in-plane point and pitch.

    ``radius``/``height`` are the tool point in the arm plane, measured from
    base_link. Returns ``(q_shoulder, q_elbow, q_wrist_angle, reachable)``.
    """

    l1 = geometry.upper_arm_length
    l2 = geometry.forearm_length
    l3 = geometry.wrist_length

    # Back off along the approach axis to the wrist_angle joint, then solve the
    # remaining two-link problem from the shoulder.
    u = radius - l3 * ops.sin(gamma)
    v = (height - geometry.shoulder_height) - l3 * ops.cos(gamma)

    distance = ops.sqrt(u * u + v * v)
    reach_max = l1 + l2
    reach_min = abs(l1 - l2)
    reachable = (distance <= reach_max) & (distance >= reach_min)

    # Guard the divisions and the acos domain. Values are already clamped where
    # `reachable` is False, so the returned pose is the closest legal one.
    safe_distance = ops.clip(distance, max(reach_min, 1e-6), reach_max)

    cos_delta = (safe_distance * safe_distance + l1 * l1 - l2 * l2) / (
        2.0 * l1 * safe_distance
    )
    cos_spread = (safe_distance * safe_distance - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    delta = ops.acos(ops.clip(cos_delta, -1.0, 1.0))
    spread = ops.acos(ops.clip(cos_spread, -1.0, 1.0))

    # Angle of the shoulder->wrist chord, measured from +z toward +r.
    chord = ops.atan2(u, v)
    sign = 1.0 if elbow_up else -1.0
    alpha = chord - sign * delta
    beta = alpha + sign * spread

    q_shoulder = alpha - geometry.upper_arm_skew
    q_elbow = beta - (0.5 * math.pi) - q_shoulder
    q_wrist = gamma - (0.5 * math.pi) - q_shoulder - q_elbow
    return q_shoulder, q_elbow, q_wrist, reachable


def _wrap(ops: Any, angle: Any) -> Any:
    """Wrap to (-pi, pi] for either array module."""

    if isinstance(ops, _TorchOps):
        two_pi = 2.0 * math.pi
        return angle - two_pi * ops.torch.floor((angle + math.pi) / two_pi)
    two_pi = 2.0 * math.pi
    return angle - two_pi * ops.np.floor((angle + math.pi) / two_pi)


def clamp_target_to_reach(
    position: Any,
    *,
    geometry: WidowX200Geometry = WX200,
    gamma: float = TOP_DOWN_PITCH,
    margin: float = 0.005,
) -> Any:
    """Pull an out-of-reach target back onto the reachable sphere.

    Clamping the *target* rather than the joints keeps the controller's contract
    intact: the tool still goes where the controller says it went. Clamping the
    joints instead would leave the tool somewhere the reward never asked for,
    which is how a policy learns to servo a point it cannot occupy.
    """

    ops = _ops_for(position)
    x = position[..., 0]
    y = position[..., 1]
    z = position[..., 2]
    radius = ops.sqrt(x * x + y * y)

    l3 = geometry.wrist_length
    sin_gamma = math.sin(gamma)
    cos_gamma = math.cos(gamma)
    u = radius - l3 * sin_gamma
    v = (z - geometry.shoulder_height) - l3 * cos_gamma
    distance = ops.sqrt(u * u + v * v)

    limit = geometry.max_planar_reach - float(margin)
    scale = ops.minimum(limit / ops.maximum(distance, 1e-6), 1.0)
    u_c = u * scale
    v_c = v * scale

    radius_c = u_c + l3 * sin_gamma
    z_c = v_c + l3 * cos_gamma + geometry.shoulder_height
    # Preserve the waist direction exactly; only the in-plane radius moves.
    ratio = radius_c / ops.maximum(radius, 1e-6)
    return ops.stack([x * ratio, y * ratio, z_c])


def top_down_ik(
    position: Any,
    yaw: Any,
    *,
    geometry: WidowX200Geometry = WX200,
    gamma: float = TOP_DOWN_PITCH,
    elbow_up: bool = True,
    clamp_to_reach: bool = True,
    enforce_joint_limits: bool = True,
) -> IKSolution:
    """Map a world tool pose to WidowX-200 joint angles.

    ``position`` is ``(..., 3)`` in base_link coordinates and ``yaw`` is
    ``(...)``, the world-frame rotation of the finger-opening axis. Works
    identically on NumPy arrays (host tooling, tests) and Torch tensors (the
    batched MJWarp controller); the array module is inferred from the input.
    """

    ops = _ops_for(position)
    position = ops.asarray(position)
    yaw = ops.asarray(yaw, like=position)

    if clamp_to_reach:
        # Solve the REQUEST first, purely to learn whether it was reachable,
        # then solve the clamped point for the joints actually commanded.
        # Reporting reachability of the clamped point instead would make the
        # flag a tautology -- it is always reachable, by construction -- and
        # the batched path would lose its only way to notice that the policy
        # is driving into a wall.
        requested = top_down_ik(
            position,
            yaw,
            geometry=geometry,
            gamma=gamma,
            elbow_up=elbow_up,
            clamp_to_reach=False,
            enforce_joint_limits=enforce_joint_limits,
        )
        clamped = clamp_target_to_reach(position, geometry=geometry, gamma=gamma)
        solved = top_down_ik(
            clamped,
            yaw,
            geometry=geometry,
            gamma=gamma,
            elbow_up=elbow_up,
            clamp_to_reach=False,
            enforce_joint_limits=enforce_joint_limits,
        )
        return IKSolution(
            q=solved.q, reachable=requested.reachable, clamped_target=clamped
        )

    x = position[..., 0]
    y = position[..., 1]
    z = position[..., 2]

    q_waist = ops.atan2(y, x)
    radius = ops.sqrt(x * x + y * y)
    gamma_arr = ops.full_like(radius, float(gamma))

    q_shoulder, q_elbow, q_wrist, reachable = _planar_ik(
        ops, radius, z, gamma_arr, geometry, elbow_up
    )

    # The finger axis is perpendicular to the arm plane, so a world-frame yaw
    # command becomes a wrist_rotate measured relative to the waist. The sign
    # and offset are fixed by WRIST_ROTATE_YAW_SIGN/OFFSET, which
    # tests/test_widowx200_kinematics.py pins against MuJoCo's own FK.
    q_wrist_rotate = _wrap(
        ops, WRIST_ROTATE_YAW_SIGN * (yaw - q_waist) + WRIST_ROTATE_YAW_OFFSET
    )

    joints = [q_waist, q_shoulder, q_elbow, q_wrist, q_wrist_rotate]
    if enforce_joint_limits:
        limited: list[Any] = []
        for name, value in zip(JOINT_NAMES, joints):
            low, high = JOINT_LIMITS[name]
            clamped = ops.clip(value, low, high)
            reachable = reachable & (
                (value >= low - 1e-6) & (value <= high + 1e-6)
            )
            limited.append(clamped)
        joints = limited

    return IKSolution(
        q=ops.stack(joints),
        reachable=reachable,
        clamped_target=position,
    )


# Sign and offset relating wrist_rotate to a world-frame tool yaw.
#
# The sign is -1 because wrist_rotate spins about the arm's local +x, which
# points straight DOWN once the approach is vertical, so a positive joint angle
# runs the world yaw backwards. The offset is pi/2 because at wrist_rotate = 0
# the finger-opening axis lies along the arm plane's normal rather than along
# it. Both were measured against MuJoCo's own forward kinematics -- the residual
# is zero to machine precision over a random pose sweep -- and
# tests/test_widowx200_kinematics.py re-measures them so a model edit that
# rotates the gripper cannot pass silently.
#
# The resulting convention matches the CDPR's `ee_yaw`: yaw = 0 puts the
# finger-opening axis along world +x. That is deliberate. `ee_yaw` is an
# absolute value in the SmolVLA state vector, so a mismatched convention would
# shift the observation distribution under a warm start from a CDPR checkpoint.
WRIST_ROTATE_YAW_SIGN: float = -1.0
WRIST_ROTATE_YAW_OFFSET: float = 0.5 * math.pi


def forward_kinematics(
    q: Any,
    *,
    geometry: WidowX200Geometry = WX200,
) -> dict[str, Any]:
    """Tool pose from joint angles: the analytic inverse of ``top_down_ik``.

    Returns ``position`` (``ee_gripper_link`` in base_link coordinates),
    ``yaw`` (world rotation of the finger axis) and ``pitch`` (the in-plane
    approach angle ``gamma``; ``pi`` is straight down).
    """

    ops = _ops_for(q)
    q = ops.asarray(q)
    q_waist = q[..., 0]
    q_shoulder = q[..., 1]
    q_elbow = q[..., 2]
    q_wrist = q[..., 3]
    q_wrist_rotate = q[..., 4]

    alpha = q_shoulder + geometry.upper_arm_skew
    beta = q_shoulder + q_elbow + 0.5 * math.pi
    gamma = beta + q_wrist

    radius = (
        geometry.upper_arm_length * ops.sin(alpha)
        + geometry.forearm_length * ops.sin(beta)
        + geometry.wrist_length * ops.sin(gamma)
    )
    height = (
        geometry.shoulder_height
        + geometry.upper_arm_length * ops.cos(alpha)
        + geometry.forearm_length * ops.cos(beta)
        + geometry.wrist_length * ops.cos(gamma)
    )
    position = ops.stack(
        [radius * ops.cos(q_waist), radius * ops.sin(q_waist), height]
    )
    yaw = _wrap(
        ops,
        q_waist + (q_wrist_rotate - WRIST_ROTATE_YAW_OFFSET) / WRIST_ROTATE_YAW_SIGN,
    )
    return {"position": position, "yaw": yaw, "pitch": gamma}


def reach_envelope(
    height: float,
    *,
    geometry: WidowX200Geometry = WX200,
    gamma: float = TOP_DOWN_PITCH,
    samples: int = 2048,
) -> tuple[float, float]:
    """Min/max horizontal radius the tool can hold at ``height``, top-down.

    This is what the scene layout must respect -- it decides where the arm base
    goes and how large the object-spawn envelope may be. It is measured from
    the kinematics and the joint box rather than assumed from the datasheet's
    "550 mm reach", which is the fully-extended figure and unreachable with the
    wrist folded to vertical.
    """

    import numpy as np

    radii = np.linspace(0.0, geometry.max_planar_reach + geometry.wrist_length, samples)
    positions = np.stack(
        [radii, np.zeros_like(radii), np.full_like(radii, float(height))], axis=-1
    )
    solution = top_down_ik(
        positions,
        np.zeros_like(radii),
        geometry=geometry,
        gamma=gamma,
        clamp_to_reach=False,
    )
    ok = np.asarray(solution.reachable, dtype=bool)
    if not ok.any():
        return (0.0, 0.0)
    return (float(radii[ok].min()), float(radii[ok].max()))
