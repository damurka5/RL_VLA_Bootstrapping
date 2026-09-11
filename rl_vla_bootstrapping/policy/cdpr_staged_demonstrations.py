"""One continuous ``move_to -> pick_up -> put_into`` episode, per world.

The unit of collection here is a CHAIN, not a task. A world keeps one scene,
one physical state and one global action clock from the empty-gripper start to
the release, and the teacher driving it changes underneath. That is the whole
point: three separately-trained policies produce one physically continuous
trajectory that a single-prompt student can be asked to reproduce.

What must not happen at a handoff, and what this module does instead
--------------------------------------------------------------------

**No pose writes.** ``set_end_effector_poses`` between stages would change the
state without an action, and the concatenated demonstration would contain a
discontinuity no policy could ever produce. Every change of state in a chain is
caused by a recorded five-dimensional action. The yaw alignment is therefore a
recorded CONTROLLER, driven through the ordinary action interface, and not a
setpoint write.

**No reset.** A pickup does not start from a fresh "similar" scene; it starts
from the exact live state the move-to left behind, including the object's
settled pose, the contact history and the controller integrator.

**No aliased task state.** ``evaluate_active_sparse_tasks`` mutates the
``BatchedTaskState`` it is handed -- ``ever_grasped``, ``grasped``,
``step_count``, ``peak_lift``, ``release_clearance`` are all written in place.
Three observers therefore need three independently allocated states, or the
pickup evaluator would silently consume the placement observer's grasp history.
The placement observer is the PERSISTENT one and runs from env step zero, so
its ``ever_grasped`` and its lift datum survive both handoffs.

**No treating a stage event as the episode's verdict.** ``terminated`` from the
move-to evaluator means "this world satisfied the reach predicate", which is an
event inside the chain. The chain's own status is tracked separately, in
``StageMachine``, and a world stays live through it.

Why the alignment tail exists
-----------------------------

The reach teacher was trained to put the gripper above the object; nothing in
its reward mentions wrist yaw, and the reset samples yaw uniformly over the
joint's full range. A pickup teacher that learned to grasp from one presentation
will not reliably grasp from a yaw 2 rad away from it. The tail rotates the
wrist to a calibrated fixed world yaw, holding the XYZ setpoint, through
recorded actions -- so the student can learn the rotation from the data instead
of being handed a servo at evaluation time.

The provenance this creates is deliberate and must travel with the bank: these
are policy-PLUS-controller demonstrations. An evaluation that re-applies the
same servo measures an assisted system and is reported as its own arm.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    INSTRUCTION_TO_ID,
    BatchedTaskState,
)
from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import (
    # The gripper aperture and the two centring-slack functions live with the
    # object hulls they are measured against, so that scene generation can
    # reject an ungraspable presentation without importing torch. Re-exported
    # here because this module is where every caller of them already looks.
    OPEN_GRIPPER_HALF_APERTURE_M,
    catalog_xy_radius,
    max_grasp_xy_offset,
    projected_grasp_xy_offset,
    scene_object_quaternion,
)
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import OBJECT_VARIANTS


# How far the finger tips reach below ee_base, measured from the MJCF. It is
# why a descent from a lateral offset stops early: the tips arrive level with
# the object's upper surface before the pads are anywhere near bracketing it.
FINGER_TIP_DEPTH_M = 0.0390


# --------------------------------------------------------------------------
# Stages, roles and failure reasons
# --------------------------------------------------------------------------

STAGE_MOVE_TO = 0
STAGE_ALIGN = 1
STAGE_PICK_UP = 2
STAGE_PLACEMENT = 3
STAGE_COMPLETE = 4
STAGE_FAILED = 5
# A recorded HOLD after the placement predicate fires. Optional, and off by
# default, because it changes what "success" means: a chain that satisfies the
# predicate and then watches the object roll out of the plate is not a
# placement, and only a settle window can tell the two apart. When it is on,
# the same window is applied in student evaluation, so the two numbers keep
# measuring the same event.
STAGE_SETTLE = 6

STAGE_NAMES: tuple[str, ...] = (
    "move_to",
    "align",
    "pick_up",
    "placement",
    "complete",
    "failed",
    "settle",
)

# The semantic stage a dataset row belongs to. ``align`` is a SUBSTAGE of
# move-to, not a fourth stage: the design keeps the alignment tail inside the
# move-to slice and reports its share separately, so a stage-balanced sampler
# draws "one third move" rather than "one quarter move".
SEMANTIC_STAGE_OF: dict[int, str] = {
    STAGE_MOVE_TO: "move_to",
    STAGE_ALIGN: "move_to",
    STAGE_PICK_UP: "pick_up",
    STAGE_PLACEMENT: "placement",
    STAGE_SETTLE: "placement",
}

TEACHER_ROLES: tuple[str, ...] = ("move_to", "pick_up", "placement")
ROLE_TO_ID = {name: index for index, name in enumerate(TEACHER_ROLES)}

# The teacher whose prior conditions a decision. Alignment decisions stay on the
# move-to teacher: the tail is inside the move-to stage, the gripper is still
# empty and above the object, and routing them to the pickup teacher would
# condition a rotation on a prompt about grasping.
ROLE_OF_STAGE: dict[int, int] = {
    STAGE_MOVE_TO: ROLE_TO_ID["move_to"],
    STAGE_ALIGN: ROLE_TO_ID["move_to"],
    STAGE_PICK_UP: ROLE_TO_ID["pick_up"],
    STAGE_PLACEMENT: ROLE_TO_ID["placement"],
    STAGE_SETTLE: ROLE_TO_ID["placement"],
}

FAILURE_NAMES: tuple[str, ...] = (
    "none",
    "move_budget_exhausted",
    "align_budget_exhausted",
    "pickup_budget_exhausted",
    "placement_budget_exhausted",
    "premature_grasp_before_handoff",
    "carry_loss",
    "wrong_place_settled",
    "settle_lost",
    "simulator_divergence",
    "global_budget_exhausted",
)
FAILURE_TO_ID = {name: index for index, name in enumerate(FAILURE_NAMES)}

# How an executed action was produced. Recorded per action so a consumer can
# tell a teacher command from a controller one without re-deriving the stage.
SOURCE_TEACHER = 0
SOURCE_YAW_TAIL = 1
SOURCE_YAW_HOLD = 2
SOURCE_SETTLE_HOLD = 3
SOURCE_GRIPPER_HOLD = 4
# The alignment tail WITH the XY centring bridge active. Distinct from
# `yaw_tail` so a bank can report exactly how many of its actions came from a
# controller that translated the gripper, not merely rotated it -- that is a
# materially larger share of the demonstration and it must be countable.
SOURCE_ALIGN_BRIDGE = 5
SOURCE_NAMES: tuple[str, ...] = (
    "teacher",
    "yaw_tail",
    "yaw_hold",
    "settle_hold",
    "gripper_hold",
    "align_bridge",
)


# --------------------------------------------------------------------------
# Instruction text
# --------------------------------------------------------------------------


def teacher_instruction_text(role: str, label: str, destination: str) -> str:
    """The prompt each teacher was TRAINED with, spelled exactly as its own
    reset spells it.

    Not normalized, on purpose. A teacher conditioned on "put apple on the
    plate" is a different function of "put apple into plate", and swapping the
    wording at collection time would measure a prompt transfer nobody asked
    for. The student's wording is applied later, by relabelling.
    """

    if role == "move_to":
        return f"move to {label}"
    if role == "pick_up":
        return f"pick up {label}"
    if role != "placement":
        raise ValueError(f"Unknown teacher role {role!r}.")
    if destination == "bowl":
        return f"put {label} into bowl"
    if destination == "plate":
        return f"put {label} on the plate"
    raise ValueError(f"Unknown destination {destination!r}.")


def student_instruction_text(label: str, destination: str) -> str:
    """The single final prompt the student sees from the first action.

    ``put <object> into plate`` is the user's requested wording and is used
    consistently in the dataset refresh, the SFT and the evaluation. The plate
    teacher keeps its familiar ``on the plate`` during collection; if the two
    ever drift apart, the student is trained on one template and tested on
    another, which reads as a capability loss.
    """

    if destination not in {"plate", "bowl"}:
        raise ValueError(f"Unknown destination {destination!r}.")
    return f"put {label} into {destination}"


def destination_instruction_id(destination: str) -> int:
    if destination == "plate":
        return int(INSTRUCTION_TO_ID["put_into_plate"])
    if destination == "bowl":
        return int(INSTRUCTION_TO_ID["put_into_bowl"])
    raise ValueError(f"Unknown destination {destination!r}.")


def object_label(catalog: str) -> str:
    return OBJECT_VARIANTS[catalog].label


# --------------------------------------------------------------------------
# Yaw
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PickupYawCalibration:
    """A MEASURED fixed world yaw for pickup, with its provenance.

    ``target_yaw`` is not a constant of this file. It is produced by
    ``tools/audit/calibrate_cdpr_pickup_yaw.py`` from the loaded model's camera
    extrinsics and the actual endpoint poses of the reach teacher, and stored
    with the calibration pose it was solved at. A hard-coded 0 or pi would be a
    guess about a mount transform that the MJCF is available to answer.

    ``tolerance_rad`` is a tolerance because this is a dynamical simulator with
    a compliant wrist: the yaw joint is driven by a position actuator through a
    ball-jointed stabilizer, so "equal" is not a thing that happens. The
    acceptance is the tolerance held for ``consecutive_decisions`` decisions,
    which rejects a world that merely swings through the target.
    """

    target_yaw: float
    tolerance_rad: float = math.radians(5.0)
    consecutive_decisions: int = 2
    # The height the wrist must be at before it is allowed to rotate. A reach
    # that ended low would sweep the fingers through the object; the bridge up
    # to this height is itself commanded and recorded.
    safe_rotation_z: float = 0.26
    source: str = "unset"
    calibration_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_joint_limits: tuple[float, float] = (-math.pi, math.pi)

    def validate(self) -> None:
        low, high = self.yaw_joint_limits
        if not low <= self.target_yaw <= high:
            raise ValueError(
                f"Calibrated pickup yaw {self.target_yaw} is outside the yaw "
                f"joint's range {self.yaw_joint_limits}. The servo would ask "
                "for motion the joint cannot make."
            )
        if self.tolerance_rad <= 0.0:
            raise ValueError("The yaw tolerance must be positive.")
        if self.consecutive_decisions < 1:
            raise ValueError("At least one decision must hold the tolerance.")
        if self.source == "unset":
            raise ValueError(
                "This calibration has no provenance. Run "
                "tools/audit/calibrate_cdpr_pickup_yaw.py and pass its result; "
                "do not invent a numeric yaw."
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "target_yaw": float(self.target_yaw),
            "tolerance_rad": float(self.tolerance_rad),
            "consecutive_decisions": int(self.consecutive_decisions),
            "safe_rotation_z": float(self.safe_rotation_z),
            "source": str(self.source),
            "calibration_pose": [float(v) for v in self.calibration_pose],
            "yaw_joint_limits": [float(v) for v in self.yaw_joint_limits],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PickupYawCalibration":
        return cls(
            target_yaw=float(payload["target_yaw"]),
            tolerance_rad=float(payload.get("tolerance_rad", math.radians(5.0))),
            consecutive_decisions=int(payload.get("consecutive_decisions", 2)),
            safe_rotation_z=float(payload.get("safe_rotation_z", 0.26)),
            source=str(payload.get("source", "unset")),
            calibration_pose=tuple(
                float(v) for v in payload.get("calibration_pose", (0.0, 0.0, 0.0))
            ),
            yaw_joint_limits=tuple(
                float(v)
                for v in payload.get("yaw_joint_limits", (-math.pi, math.pi))
            ),
        )


def wrap_to_pi(torch: Any, angle: Any) -> Any:
    """Shortest signed representative of an angle, in (-pi, pi]."""

    return angle - 2.0 * math.pi * torch.round(angle / (2.0 * math.pi))


def reachable_yaw_error(
    torch: Any, current: Any, target: float, limits: tuple[float, float]
) -> Any:
    """Signed yaw error the joint can actually travel.

    The shortest-angle error is the right answer for a free revolute joint and
    the wrong one here: ``ee_yaw`` is limited to [-pi, pi], so a shortest path
    that crosses the boundary asks for motion into a hard stop and the servo
    would push against it for the whole tail. Where the short way is blocked,
    the long way around is taken instead; where BOTH ends are blocked the error
    is clamped to what the joint can reach, which is reported rather than
    silently satisfied.
    """

    low, high = (float(limits[0]), float(limits[1]))
    goal = torch.full_like(current, float(target)).clamp(low, high)
    short = wrap_to_pi(torch, goal - current)
    long_way = short - 2.0 * math.pi * torch.sign(short)
    short_ok = ((current + short) >= low) & ((current + short) <= high)
    error = torch.where(short_ok, short, long_way)
    # Whatever survives, keep the commanded destination inside the joint.
    destination = (current + error).clamp(low, high)
    return destination - current


class YawTailController:
    """A bounded yaw servo expressed in the ordinary five-dim action.

    Holds XY and the hand OPEN, and drives the wrist toward
    the calibrated yaw at whatever fraction of ``action_step_yaw`` the
    remaining error justifies. Every command it produces is executed by the
    plant and recorded. With a pickup height configured, it climbs before
    rotating and descends to the pickup pose once yaw is aligned.
    """

    def __init__(
        self,
        *,
        torch: Any,
        calibration: PickupYawCalibration,
        action_step_yaw: float,
        action_step_xyz: float,
        action_step_gripper: float = 0.05,
        pickup_height_above_grasp: float | None = None,
        pickup_height_tolerance: float = 0.003,
        xy_centring_deadband: float | None = None,
        xy_centring_abort: float | None = None,
        handoff_at_clearance: bool = False,
        yaw_servo_gain: float = 0.35,
        descent_gain: float = 1.0,
    ) -> None:
        self.torch = torch
        self.calibration = calibration
        self.action_step_yaw = float(action_step_yaw)
        self.action_step_xyz = float(action_step_xyz)
        self.action_step_gripper = float(action_step_gripper)
        self.pickup_height_above_grasp = pickup_height_above_grasp
        self.pickup_height_tolerance = float(pickup_height_tolerance)
        # Stop the tail at the rotation clearance and hand off from there,
        # instead of descending to the pickup teacher's trained height.
        #
        # Every abort lives in the descent, and the geometry there is genuinely
        # awkward: at grasp point + 0.01 m the finger tips are 0.029 m BELOW the
        # object's centre, straddling it, so neither rotating nor translating is
        # free. The pickup teacher's own curriculum trains it from within a
        # 0.20 m three-dimensional cap, so a centred handoff at the clearance
        # height is inside its distribution -- it simply has to descend itself,
        # which is the thing it was trained to do.
        #
        # An arm, not a replacement: it trades a handoff at the teacher's exact
        # trained pose for one it can reach, and which of those the teacher
        # prefers is a measurement.
        self.handoff_at_clearance = bool(handoff_at_clearance)
        # DAMPING on the yaw servo, and without it the tail never promotes.
        #
        # The command is recomputed every ACTION, four times per decision, and
        # the plant integrates it: `setpoint += a3 * action_step_yaw` with
        # `a3 = error / action_step_yaw` means the setpoint absorbs the FULL
        # measured error every action while a kp=30 actuator, through a damped
        # ball joint, on a cable-suspended platform, is still travelling toward
        # the previous one. That is textbook integrator windup and it rings.
        #
        # Measured: median yaw error 0.0824 rad against an 0.0873 rad
        # acceptance band -- 94% of tolerance, sitting exactly on the boundary
        # -- with 47% of tail steps outside it and 0 of 10 chains promoted,
        # while centring in the same runs held 0.0039 m.
        #
        # Large errors are untouched: at this gain an error of 1.73 rad still
        # saturates the command, so the initial rotation runs at full rate and
        # only the final approach is damped. The XY servo is deliberately NOT
        # damped -- it converges to 4 mm at unity gain, because the cable
        # platform tracks a translation far faster than the wrist tracks a
        # rotation.
        self.yaw_servo_gain = float(yaw_servo_gain)
        if not 0.0 < self.yaw_servo_gain <= 1.0:
            raise ValueError("The yaw servo gain must be in (0, 1].")
        # Scale only the recorded descent toward the pickup pose. At unity the
        # 60 mm clearance-to-grasp error saturates a 15 mm/action controller;
        # the cable platform descends before its lateral loop can settle.
        self.descent_gain = float(descent_gain)
        if not 0.0 < self.descent_gain <= 1.0:
            raise ValueError("The descent gain must be in (0, 1].")
        # None disables the XY centring bridge entirely, which is the default.
        self.xy_centring_deadband = xy_centring_deadband
        # HYSTERESIS. Entering the descent needs the tight deadband; staying in
        # it tolerates more drift. Without the split the descend gate's else
        # branch is a CLIMB, so a wrist that drifts a millimetre past the
        # deadband on the way down climbs all the way back to the rotation
        # clearance and starts again -- chatter that burns the tail budget
        # while every instantaneous reading looks correct.
        #
        # A real loss of centring pauses vertical motion while the already
        # active lateral servo moves toward the object's centre. Earlier code
        # climbed all the way back to clearance, producing hundreds of
        # climb/restart cycles. Moving toward the centre is away from the near
        # finger; pausing Z makes that correction safer than combining it with
        # either descent or a full retreat.
        self.xy_centring_abort = (
            xy_centring_abort
            if xy_centring_abort is not None
            else (
                None
                if xy_centring_deadband is None
                else 2.0 * float(xy_centring_deadband)
            )
        )
        if self.action_step_yaw <= 0.0 or self.action_step_xyz <= 0.0:
            raise ValueError("Action steps must be positive.")
        if self.action_step_gripper <= 0.0:
            raise ValueError("The gripper action step must be positive.")

    def open_command(self, gripper_opening: Any) -> Any:
        """Normalized gripper action that opens toward 1.0 and never closes.

        Clamped at zero from below on purpose: this is a HOLD, not control of
        the gripper. It can undo a closure the approach did not need and it can
        never squeeze.
        """

        deficit = (1.0 - gripper_opening).clamp_min(0.0)
        return (deficit / self.action_step_gripper).clamp(0.0, 1.0)

    def xy_command(self, *, ee_position: Any, target_xy: Any) -> Any:
        """[W, 2] translation that closes the lateral error, with a deadband.

        THE PRIVILEGED STEP, and it is worth naming as such. Every other
        channel this controller drives is a function of proprioception or of a
        world constant: the gripper opening it can read, the yaw target is
        calibrated once, the safe height is declared. This one reads the
        object's live XY, which is exactly the oracle vector the campaign
        removed from the policy's OBSERVATION for being deployment-invalid.

        The distinction that makes it admissible is that it is a CONTROLLER,
        not an input: its output is a recorded action the student has to learn
        to reproduce from pixels, and the student is never given the vector.
        The design contemplates precisely this -- "any lift/repositioning must
        also be an explicit recorded bridge, or the chain is rejected".

        What it costs is honest provenance: these become demonstrations of a
        policy plus a controller that also translates, and the actions it
        produces carry their own source code so the share is countable.
        """

        torch = self.torch
        error = target_xy - ee_position[:, :2]
        distance = torch.linalg.vector_norm(error, dim=-1, keepdim=True)
        # Inside the deadband the command is exactly zero, so a centred wrist
        # does not jitter against the controller's own quantisation.
        inside = distance <= float(self.xy_centring_deadband)
        command = (error / self.action_step_xyz).clamp(-1.0, 1.0)
        return torch.where(inside, torch.zeros_like(command), command)

    def centred(self, *, ee_position: Any, target_xy: Any) -> Any:
        torch = self.torch
        distance = torch.linalg.vector_norm(
            target_xy - ee_position[:, :2], dim=-1
        )
        return distance <= float(self.xy_centring_deadband)

    def yaw_command(self, current_yaw: Any) -> Any:
        """Normalized yaw action in [-1, 1] that closes the error."""

        torch = self.torch
        error = reachable_yaw_error(
            torch,
            current_yaw,
            self.calibration.target_yaw,
            self.calibration.yaw_joint_limits,
        )
        return (
            self.yaw_servo_gain * error / self.action_step_yaw
        ).clamp(-1.0, 1.0)

    def actions(
        self, *, ee_position: Any, ee_yaw: Any, gripper_opening: Any = None,
        grasp_point_z: Any = None, target_xy: Any = None,
    ) -> Any:
        """[W, 5] alignment command: bridge Z, open the hand, rotate, centre.

        The sequence is ordered and each step gates the next, because doing
        them together is how a finger gets swept through the object: climb to
        the rotation clearance, then rotate and (when the bridge is enabled)
        translate at that height where nothing can be struck, and only descend
        once the wrist is both aligned and centred.
        """

        torch = self.torch
        worlds = int(ee_yaw.shape[0])
        command = torch.zeros(
            (worlds, 5), dtype=torch.float32, device=ee_yaw.device
        )
        # The legacy yaw-only mode bridges upward. Three-stage collection also
        # supplies a pickup height, adding a recorded descent after alignment.
        rise = (
            float(self.calibration.safe_rotation_z) - ee_position[:, 2]
        ).clamp_min(0.0)
        command[:, 2] = (rise / self.action_step_xyz).clamp(0.0, 1.0)
        command[:, 3] = self.yaw_command(ee_yaw)
        if self.pickup_height_above_grasp is not None and not self.handoff_at_clearance:
            if grasp_point_z is None:
                raise ValueError("Pickup alignment requires the live grasp point height.")
            # Finish at the teacher's aligned training height, not at the
            # rotation clearance. All motion remains recorded plant actions.
            yaw_ready = self.aligned(ee_yaw)
            clearance = ee_position[:, 2] >= (
                float(self.calibration.safe_rotation_z) - self.pickup_height_tolerance
            )
            if target_xy is None or self.xy_centring_deadband is None:
                centred = torch.ones_like(yaw_ready)
            else:
                error = torch.linalg.vector_norm(
                    target_xy - ee_position[:, :2], dim=-1
                )
                # Below the clearance the wrist is already on its way down, so
                # judge it by the wider band.
                descending_now = ee_position[:, 2] < (
                    float(self.calibration.safe_rotation_z)
                    - self.pickup_height_tolerance
                )
                centred = torch.where(
                    descending_now,
                    error <= float(self.xy_centring_abort),
                    error <= float(self.xy_centring_deadband),
                )
            target_z = grasp_point_z + float(self.pickup_height_above_grasp)
            descend = (
                self.descent_gain
                * (target_z - ee_position[:, 2])
                / self.action_step_xyz
            ).clamp(-1.0, 1.0)
            # Descend only once the wrist is aligned AND over the object. A
            # descent from a lateral offset is what lands the fingers on the
            # object's shoulder and stops the grasp dead -- measured, 0 grasps
            # in 48 chains at a 0.0166-0.0186 m handoff offset.
            command[:, 2] = torch.where(
                yaw_ready & centred, descend, command[:, 2]
            )
            # Below clearance, yaw is already safe. If lateral compliance
            # exceeds the abort band, HOLD Z and re-centre rather than climbing
            # back to clearance. The previous climb/restart loop consumed most
            # of the 48-decision tail: 683 recorded aborts in the first
            # 512-chain bank, with only 20/79 reaches promoted. A zero Z action
            # is an explicit position hold under this controller.
            below_clearance = ~clearance
            recentering_pause = yaw_ready & ~centred & below_clearance
            command[:, 2] = torch.where(
                recentering_pause,
                torch.zeros_like(command[:, 2]),
                command[:, 2],
            )
            # Do not sweep the fingers through the object while climbing.
            # Once aligned, small yaw corrections may hold that angle on descent.
            command[:, 3] = torch.where(
                clearance | yaw_ready, command[:, 3], torch.zeros_like(ee_yaw)
            )
            if target_xy is not None and self.xy_centring_deadband is not None:
                # THE LATERAL SERVO RUNS WHENEVER THE YAW SERVO DOES, descent
                # included, and zeroing it on the way down was the bug.
                #
                # A zero XY action is not "hold position". Under the production
                # controller `proposed_target = ee_position + delta`, so a zero
                # delta makes the setpoint CHASE the measurement: drift is
                # accepted rather than corrected, and on a cable-suspended,
                # ball-jointed platform it ratchets. Measured over a
                # 48-decision tail: lateral error p90 0.0294 m, 128 descent
                # aborts across 8 worlds, 0 of 10 chains promoted.
                #
                # Correcting during the descent is also the SAFE direction. The
                # servo only ever moves toward the object's centre, which is
                # away from whichever finger is closest; at the <=9 mm error the
                # descent tolerates, the correction is under one action step.
                # Widening the abort band was treating the symptom.
                lateral = self.xy_command(
                    ee_position=ee_position, target_xy=target_xy
                )
                command[:, :2] = torch.where(
                    (clearance | yaw_ready)[:, None],
                    lateral,
                    torch.zeros_like(lateral),
                )
        if self.handoff_at_clearance and target_xy is not None and (
            self.xy_centring_deadband is not None
        ):
            # No descent phase, so the only unsafe moment is the initial climb.
            clearance = ee_position[:, 2] >= (
                float(self.calibration.safe_rotation_z)
                - self.pickup_height_tolerance
            )
            lateral = self.xy_command(
                ee_position=ee_position, target_xy=target_xy
            )
            command[:, :2] = torch.where(
                clearance[:, None], lateral, torch.zeros_like(lateral)
            )
            command[:, 3] = torch.where(
                clearance, command[:, 3], torch.zeros_like(ee_yaw)
            )
        if gripper_opening is not None:
            command[:, 4] = self.open_command(gripper_opening)
        return command

    def aligned(self, ee_yaw: Any) -> Any:
        torch = self.torch
        error = reachable_yaw_error(
            torch,
            ee_yaw,
            self.calibration.target_yaw,
            self.calibration.yaw_joint_limits,
        )
        return error.abs() <= float(self.calibration.tolerance_rad)

    def handoff_ready(self, *, ee_yaw: Any, ee_position: Any, grasp_point_z: Any) -> Any:
        """Is the wrist in the pose the pickup teacher should be handed?

        The height term is skipped in the clearance-handoff arm, and leaving it
        in was a bug that made that arm untestable: the arm exists precisely so
        the tail does NOT descend, so `ee_z - grasp_z` stays at the rotation
        clearance (0.0696 m measured) against a 0.003 m band around 0.010 m.
        The gate required the descent that the flag removes, so the streak
        could never start and every chain died at align_budget_exhausted --
        with the yaw error at exactly 0.0 and the centring at 5.6 mm.

        Two runs were spent on an arm that could not promote whatever the
        controller did.
        """

        ready = self.aligned(ee_yaw)
        if (
            self.pickup_height_above_grasp is not None
            and not self.handoff_at_clearance
        ):
            error = ee_position[:, 2] - grasp_point_z - float(self.pickup_height_above_grasp)
            ready = ready & (error.abs() <= self.pickup_height_tolerance)
        return ready


# --------------------------------------------------------------------------
# Stage machine
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageBudgets:
    """Collection CAPS, not required stage lengths.

    The design's starting budgets, restated with their arithmetic: at four
    executed actions per policy decision, 32 + 32 + 64 decisions is at most 512
    env steps. Every existing config's ``max_env_steps`` and curriculum horizon
    is smaller than that, which is exactly the counter conversion the design
    tells us to audit -- a chain silently truncated to the old 40-decision task
    would end mid-carry and be scored a failure of the policy.
    """

    move_decisions: int = 32
    pickup_decisions: int = 32
    placement_decisions: int = 64
    # The alignment tail's OWN cap, not a share of the move budget.
    #
    # This class's docstring used to claim the two shared one, and the code
    # never did: `stage_decisions` resets on every transition, so the tail
    # always got a fresh `move_decisions`. The claim and the code disagreeing
    # is how the global loop came to be shorter than the worst case a chain
    # could need -- 32 + 32 + 32 + 64 = 160 decisions against a 128-decision
    # loop -- and a chain that ran past it stopped with no failure code and no
    # acceptance. Measured: 6 of 64 worlds per move-to screen ended that way,
    # invisible in the failure histogram because nothing decided anything
    # about them.
    #
    # Fixed by counting the tail rather than by starving a late reach: a chain
    # that reaches at decision 29 and needs twelve decisions to rotate is a
    # perfectly good chain, and the observed tail length is ~21 decisions.
    # None means "the same cap as move_decisions".
    #
    # Declared AFTER placement_decisions on purpose: the positional form
    # StageBudgets(move, pickup, placement) is used across the tests and would
    # silently re-map if this field were inserted before them.
    align_decisions: int | None = None
    # 0 disables the stability check entirely, which is the default: turning it
    # on makes the bank's acceptance strictly stricter than the production
    # placement predicate, and the two must then be reported separately rather
    # than one silently rewriting the other's historical scores.
    settle_decisions: int = 0

    @property
    def align_budget(self) -> int:
        return int(
            self.move_decisions
            if self.align_decisions is None
            else self.align_decisions
        )

    @property
    def total_decisions(self) -> int:
        """The loop length, and it must COVER the sum of the stage caps.

        If it is shorter, a chain is cut off mid-stage with no failure code and
        no acceptance -- invisible in every census, because nothing decided
        anything about it.
        """

        return (
            int(self.move_decisions)
            + self.align_budget
            + int(self.pickup_decisions)
            + int(self.placement_decisions)
            + int(self.settle_decisions)
        )

    def validate(self) -> None:
        for name in ("move_decisions", "pickup_decisions", "placement_decisions"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least one decision.")
        if self.align_budget < 1:
            raise ValueError("align_decisions must be at least one decision.")
        if int(self.settle_decisions) < 0:
            raise ValueError("settle_decisions cannot be negative.")


@dataclass(frozen=True)
class PickupReadiness:
    """What "the reach is finished" means beyond the XY predicate.

    XY success alone does not establish a usable pickup pose, which is the
    §3 warning made operational: a reach that ends with the gripper closed, or
    already holding the object, or far above it, satisfies ``move_to`` and hands
    the pickup teacher a state it was never trained on.

    THE HEIGHT BAND IS RELATIVE TO THE GRASP POINT, and the first version of
    this was an absolute [0.20, 0.34] that rejected everything. The grasp point
    is ``object_z + pick_grasp_height_offset``, which for the four target
    catalogs is 0.185-0.192 m, and the pickup teacher's own aligned start -- the
    pose it was TRAINED to begin from -- is one centimetre above that, so
    0.195-0.202 m. An absolute 0.20 m floor therefore sits on top of the
    correct answer and is below it for three of the four objects.

    Worse, nothing pushes the policy up to compensate: under
    ``sparse_binary_reward`` the move-to reward's ``z_penalty_weight`` is zeroed
    and ``distance_include_z`` is off, so the reach has no Z term at all and
    settles wherever its warm start puts it -- plausibly against the 0.19 m
    reset floor. Measured on the first teacher screen, that combination scored
    0 of 64 chains for every candidate of every role.

    The band is deliberately PERMISSIVE. Its job is to exclude a reach that
    could not descend from here -- a hand that stalled near the ceiling, or one
    that has already closed -- not to second-guess the pickup teacher. Whether
    a pose is actually graspable is measured by the pickup stage's own yield,
    which is the number the screen exists to produce.
    """

    # Relative to the grasp point (object top plus the pad offset). Slightly
    # negative at the bottom because the pads may sit a hair below the nominal
    # point without anything being wrong.
    min_height_above_grasp: float = -0.005
    max_height_above_grasp: float = 0.12
    # Absolute rails from the controller workspace, not from a guess: 0.18 is
    # the configured controller floor and 0.40 is well above the reset band.
    # These catch a diverged pose, not a low reach.
    min_ee_z: float = 0.18
    max_ee_z: float = 0.40
    # The gripper must still be essentially open. The production release test
    # is max(0.55, fitted + 0.04); this is stricter because the hand should not
    # have moved at all yet.
    min_gripper_opening: float = 0.90
    # LATERAL tolerance, and it is the gate the third screen was missing.
    #
    # The production move_to success window is 0.02 m. The open gripper's
    # lateral slack is 0.0475 minus the object's hull radius: 0.0130 m for an
    # apple and 0.0185 m for the others. The window is WIDER than the grasp
    # tolerance, so a chain promoted at the edge of it hands the pickup teacher
    # a pose from which the fingers cannot bracket the object -- they descend
    # 3.9 cm, land on its shoulder, and stop.
    #
    # Measured 2026-09-10: handoff XY error 0.0166-0.0186 m, closest approach
    # during pickup stopping 0.053-0.059 m above the grasp point with the
    # object never moving (max lift 0.002 m) and not one grasp in 48 chains
    # across three screens.
    #
    # So readiness carries its own XY bound, derived per object from the
    # measured aperture rather than inherited from a reward window that was
    # never about grasping.
    grasp_xy_margin: float = 0.003
    # Whether the lateral bound also gates the REACH transition.
    #
    # It always gates the handoff (align -> pickup), which is where it matters.
    # Gating the reach as well is right only when nothing downstream can fix an
    # off-centre pose: without a bridge, a chain that reaches 17 mm off will
    # still be 17 mm off at the handoff, so promoting it just burns the tail
    # budget. With the XY centring bridge enabled the opposite holds -- the
    # tail exists to close exactly that error, and gating the reach on it means
    # the bridge never runs on the 90% of chains that need it.
    #
    # Set False by the recorder whenever the bridge is on. Measured on the
    # teacher screen: 57-60 of 64 chains die at move_budget_exhausted, and the
    # reach predicate itself fires on roughly a third of them.
    require_centred_at_reach: bool = True
    # No grasp: a reach that has already closed on the object has skipped the
    # stage this chain exists to record.
    forbid_grasp: bool = True


CARRY_ACCEPTANCE_VERSION = "opening_over_goal_v2"


def release_opening_over_goal(
    *,
    command: Any,
    opening: Any,
    previous_opening: Any,
    target_xy: Any,
    receptacle_xy: Any,
    radius: Any,
) -> Any:
    """Observed opening above the goal, before the release threshold.

    Works with NumPy arrays and torch tensors so recording acceptance and the
    live stage machine use the same test. Height/settling remain the native
    placement predicate's responsibility: release starts above the receptacle.

    Three conjuncts, and each excludes a different false positive. The COMMAND
    must ask for opening, so a hand prised apart by a collision is not a
    release. The opening must actually INCREASE, so a saturated command against
    a stuck finger is not one either. And the object must be over the goal, so
    a hand that opens mid-carry and drops the object is still a slip -- which
    is what stops this exemption from forgiving the failure it is next to.
    """

    delta = target_xy - receptacle_xy
    return ((command > 1e-6) & (opening > previous_opening + 1e-6)
            & (delta[..., 0] ** 2 + delta[..., 1] ** 2 <= radius ** 2))


def contact_ended_without_release(
    *,
    physical_grasp: Any,
    released: Any,
    release_in_progress: Any,
) -> Any:
    """The live slip test, in ONE place for both callers.

    ``physical_grasp`` is a contact test that already includes
    ``~release_open``, so it goes False the instant the pads unload -- which is
    what the first steps of an intentional release look like, several env steps
    before the opening crosses the threshold and ``released`` becomes true. A
    rule that read "not holding it any more" therefore fires on every correct
    placement, at the moment the policy starts letting go.

    Measured on the remote screen, scene_44833e357637587f: contact ends at env
    step 248 with the opening at 0.456 and the object 7 mm from the bowl
    centre; the opening crosses its 0.55 threshold at step 251. Three steps in
    which the object is being placed correctly and every naive test calls it a
    dropped carry.

    This is the campaign's §7.11 shape for the third time -- a terminal
    condition sharing a conjunct with success and firing because the conjunct
    is not satisfied YET. It was fixed in the production ``wrong_place_settled``
    predicate, then again in this module's stage machine, and the recorder's
    acceptance check and the student evaluation each still carried their own
    copy of the mistake. Hence one function.
    """

    return ~physical_grasp & ~released & ~release_in_progress


class StageMachine:
    """Per-world stage status for a batch of chains.

    Pure tensor state with no simulator dependency, so the transition rules can
    be tested on CPU without a GPU, a renderer or a checkpoint. Everything it
    reads is passed in.
    """

    def __init__(
        self,
        *,
        torch: Any,
        device: Any,
        worlds: int,
        budgets: StageBudgets,
        calibration: PickupYawCalibration,
        readiness: PickupReadiness | None = None,
    ) -> None:
        budgets.validate()
        calibration.validate()
        self.torch = torch
        self.device = device
        self.worlds = int(worlds)
        self.budgets = budgets
        self.calibration = calibration
        self.readiness = readiness or PickupReadiness()

        zeros_int = lambda: torch.zeros(  # noqa: E731 - four identical allocs
            (self.worlds,), dtype=torch.int64, device=device
        )
        zeros_bool = lambda: torch.zeros(  # noqa: E731
            (self.worlds,), dtype=torch.bool, device=device
        )
        self.stage = zeros_int()
        self.stage_decisions = zeros_int()
        self.decisions = zeros_int()
        self.failure = zeros_int()
        self.aligned_streak = zeros_int()
        self.active = torch.ones(
            (self.worlds,), dtype=torch.bool, device=device
        )
        self.reach_event = torch.full(
            (self.worlds,), -1, dtype=torch.int64, device=device
        )
        self.align_event = self.reach_event.clone()
        self.pickup_event = self.reach_event.clone()
        self.placement_event = self.reach_event.clone()
        self.ever_held = zeros_bool()
        self.handoff_lift = torch.zeros(
            (self.worlds,), dtype=torch.float32, device=device
        )
        self.pickup_regrasp = zeros_int()

    # -- queries --------------------------------------------------------

    def stage_role_ids(self) -> Any:
        """Teacher role per world, for inference batching."""

        torch = self.torch
        roles = torch.zeros(
            (self.worlds,), dtype=torch.int64, device=self.device
        )
        for stage, role in ROLE_OF_STAGE.items():
            roles = torch.where(
                self.stage == stage,
                torch.full_like(roles, int(role)),
                roles,
            )
        return roles

    def in_stage(self, stage: int) -> Any:
        return (self.stage == int(stage)) & self.active

    def budget_for_stage(self) -> Any:
        """Per-world decision cap of the stage that world is currently in.

        The alignment tail carries its OWN cap. It is a substage of move-to for
        labelling and for the stage-balanced sampler, but not for budgeting:
        `stage_decisions` resets at the transition, and pretending otherwise is
        what made the global loop shorter than the worst case.
        """

        torch = self.torch
        budget = torch.full(
            (self.worlds,),
            int(self.budgets.placement_decisions),
            dtype=torch.int64,
            device=self.device,
        )
        budget = torch.where(
            self.stage == STAGE_MOVE_TO,
            torch.full_like(budget, int(self.budgets.move_decisions)),
            budget,
        )
        budget = torch.where(
            self.stage == STAGE_ALIGN,
            torch.full_like(budget, self.budgets.align_budget),
            budget,
        )
        budget = torch.where(
            self.stage == STAGE_PICK_UP,
            torch.full_like(budget, int(self.budgets.pickup_decisions)),
            budget,
        )
        budget = torch.where(
            self.stage == STAGE_SETTLE,
            torch.full_like(budget, max(int(self.budgets.settle_decisions), 1)),
            budget,
        )
        return budget

    # -- transitions ----------------------------------------------------

    def advance(
        self,
        *,
        decision: int,
        reach_success: Any,
        pickup_success: Any,
        placement_success: Any,
        placement_geometry_ok: Any,
        wrong_place_settled: Any,
        physical_grasp: Any,
        released: Any,
        gripper_opening: Any,
        ee_position: Any,
        grasp_point_z: Any,
        target_xy_error: Any,
        max_grasp_xy_offset: Any,
        target_lift: Any,
        yaw_aligned: Any,
        diverged: Any,
        alignment_ready: Any = None,
        release_in_progress: Any = None,
    ) -> dict[str, Any]:
        """One decision-boundary update. Every argument is a [W] tensor.

        Read at the boundary rather than at the instant the event fires, which
        is the design's rule and not a convenience: a native stage success can
        happen inside the four executed actions of a chunk, and the remaining
        actions of that chunk keep running. Asking again at the boundary is what
        makes "and readiness STILL holds" a real condition instead of a
        snapshot of a state the world has already left.
        """

        torch = self.torch
        active = self.active.clone()
        self.decisions = torch.where(
            active, self.decisions + 1, self.decisions
        )
        self.stage_decisions = torch.where(
            active, self.stage_decisions + 1, self.stage_decisions
        )

        opening_ok = gripper_opening >= float(
            self.readiness.min_gripper_opening
        )
        above_grasp = ee_position[:, 2] - grasp_point_z
        height_ok = (
            (above_grasp >= float(self.readiness.min_height_above_grasp))
            & (above_grasp <= float(self.readiness.max_height_above_grasp))
            & (ee_position[:, 2] >= float(self.readiness.min_ee_z))
            & (ee_position[:, 2] <= float(self.readiness.max_ee_z))
        )
        # The lateral gate, per object. Without it the reach window (0.02 m)
        # promotes poses the open gripper cannot bracket.
        centred_ok = target_xy_error <= max_grasp_xy_offset
        pickup_ready = reach_success & opening_ok & height_ok
        if self.readiness.require_centred_at_reach:
            pickup_ready = pickup_ready & centred_ok
        if self.readiness.forbid_grasp:
            pickup_ready = pickup_ready & ~physical_grasp

        moved = self.in_stage(STAGE_MOVE_TO)
        aligning = self.in_stage(STAGE_ALIGN)
        picking = self.in_stage(STAGE_PICK_UP)
        placing = self.in_stage(STAGE_PLACEMENT)
        settling = self.in_stage(STAGE_SETTLE)

        # A grasp before the pickup stage is a chain that skipped a stage; it
        # is not a lucky shortcut and must not become a "move-to" demonstration
        # that ends holding something.
        premature = (moved | aligning) & physical_grasp

        # Contact can end before the opening crosses the release threshold.
        # Allow observed, commanded opening over the goal to finish. Offline
        # acceptance additionally requires an uninterrupted opening suffix to
        # threshold crossing; a later release cannot erase an earlier slip.
        opening_release = (torch.zeros_like(released) if release_in_progress is None
                           else release_in_progress)
        carry_loss = (
            placing
            & contact_ended_without_release(
                physical_grasp=physical_grasp,
                released=released,
                release_in_progress=opening_release,
            )
            & ~placement_success
            & ~wrong_place_settled
        )

        # Losing the grasp during PICKUP is not a failure and does not end the
        # chain: the teacher may close, slip and close again, and the design
        # rejects regrasp during transport only. It is counted so a bank can
        # report how many of its pickups were first attempts.
        self.pickup_regrasp = torch.where(
            picking & self.ever_held & ~physical_grasp & ~pickup_success,
            self.pickup_regrasp + 1,
            self.pickup_regrasp,
        )
        self.ever_held = self.ever_held | (picking & physical_grasp)

        promote_align = moved & pickup_ready & ~premature
        self.reach_event = torch.where(
            promote_align & (self.reach_event < 0),
            torch.full_like(self.reach_event, int(decision)),
            self.reach_event,
        )

        pose_ready = yaw_aligned if alignment_ready is None else yaw_aligned & alignment_ready
        self.aligned_streak = torch.where(
            aligning & pose_ready & opening_ok & height_ok & centred_ok & ~physical_grasp,
            self.aligned_streak + 1,
            torch.where(aligning, torch.zeros_like(self.aligned_streak), self.aligned_streak),
        )
        promote_pick = aligning & (
            self.aligned_streak >= int(self.calibration.consecutive_decisions)
        ) & ~premature
        self.align_event = torch.where(
            promote_pick & (self.align_event < 0),
            torch.full_like(self.align_event, int(decision)),
            self.align_event,
        )

        # Still holding at the boundary AND a real lift off the true desk
        # datum. `pickup_success` already carries the production 5 cm test, but
        # it is latched at the instant it fired; the handoff condition is that
        # the world is STILL in that state when the placement teacher takes
        # over.
        promote_place = (
            picking & pickup_success & physical_grasp & (target_lift >= 0.05 - 1e-6)
        )
        self.pickup_event = torch.where(
            promote_place & (self.pickup_event < 0),
            torch.full_like(self.pickup_event, int(decision)),
            self.pickup_event,
        )
        self.handoff_lift = torch.where(
            promote_place & (self.pickup_event == int(decision)),
            target_lift.to(dtype=self.handoff_lift.dtype),
            self.handoff_lift,
        )

        released_ok = placing & placement_success
        self.placement_event = torch.where(
            released_ok & (self.placement_event < 0),
            torch.full_like(self.placement_event, int(decision)),
            self.placement_event,
        )
        settle_window = int(self.budgets.settle_decisions)
        # With no settle window the placement predicate IS the verdict, which
        # keeps this bank's success identical to every historical score. With
        # one, the chain has to still be in the receptacle when the window ends.
        enter_settle = released_ok & (settle_window > 0)
        settle_done = settling & (
            self.stage_decisions >= settle_window
        ) & placement_geometry_ok
        settle_lost = settling & ~placement_geometry_ok
        finished = (released_ok & (settle_window <= 0)) | settle_done

        over_budget = active & (
            self.stage_decisions >= self.budget_for_stage()
        )

        def fail(mask: Any, reason: str) -> None:
            code = int(FAILURE_TO_ID[reason])
            self.failure = torch.where(
                mask & (self.failure == 0),
                torch.full_like(self.failure, code),
                self.failure,
            )

        fail(diverged & active, "simulator_divergence")
        fail(premature, "premature_grasp_before_handoff")
        fail(carry_loss, "carry_loss")
        fail(placing & wrong_place_settled & ~placement_success, "wrong_place_settled")
        fail(settle_lost, "settle_lost")
        fail(over_budget & moved & ~promote_align, "move_budget_exhausted")
        fail(over_budget & aligning & ~promote_pick, "align_budget_exhausted")
        fail(over_budget & picking & ~promote_place, "pickup_budget_exhausted")
        fail(over_budget & placing & ~released_ok, "placement_budget_exhausted")

        failed = active & (self.failure != 0)
        # Success wins over an exhausted budget on the same boundary: the
        # placement predicate fired, and a cap is a limit on how long a world
        # may keep trying, not a retroactive veto on a chain that finished.
        failed = failed & ~finished

        next_stage = self.stage.clone()
        next_stage = torch.where(
            promote_align & ~failed,
            torch.full_like(next_stage, STAGE_ALIGN),
            next_stage,
        )
        next_stage = torch.where(
            promote_pick & ~failed,
            torch.full_like(next_stage, STAGE_PICK_UP),
            next_stage,
        )
        next_stage = torch.where(
            promote_place & ~failed,
            torch.full_like(next_stage, STAGE_PLACEMENT),
            next_stage,
        )
        next_stage = torch.where(
            enter_settle & ~failed & ~finished,
            torch.full_like(next_stage, STAGE_SETTLE),
            next_stage,
        )
        next_stage = torch.where(
            finished, torch.full_like(next_stage, STAGE_COMPLETE), next_stage
        )
        next_stage = torch.where(
            failed, torch.full_like(next_stage, STAGE_FAILED), next_stage
        )

        changed = active & (next_stage != self.stage)
        self.stage_decisions = torch.where(
            changed, torch.zeros_like(self.stage_decisions), self.stage_decisions
        )
        self.aligned_streak = torch.where(
            changed, torch.zeros_like(self.aligned_streak), self.aligned_streak
        )
        self.stage = next_stage
        self.active = active & ~finished & ~failed

        return {
            "promote_align": promote_align & ~failed,
            "promote_pick": promote_pick & ~failed,
            "promote_place": promote_place & ~failed,
            "enter_settle": enter_settle & ~failed & ~finished,
            "finished": finished,
            "failed": failed,
            "pickup_ready": pickup_ready,
        }

    def summary(self) -> dict[str, Any]:
        """Host-side counts, including the conditional stage yields."""

        stage = self.stage.detach().cpu().numpy()
        failure = self.failure.detach().cpu().numpy()
        reached = int((self.reach_event.detach().cpu().numpy() >= 0).sum())
        aligned = int((self.align_event.detach().cpu().numpy() >= 0).sum())
        picked = int((self.pickup_event.detach().cpu().numpy() >= 0).sum())
        placed = int((self.placement_event.detach().cpu().numpy() >= 0).sum())
        return {
            "worlds": int(self.worlds),
            "reached": reached,
            "aligned": aligned,
            "picked_up": picked,
            "placed": placed,
            # Conditional yields, which is what a collection budget is planned
            # against: an unconditional placement rate hides whether the loss
            # is in the approach or in the release.
            "align_given_reach": _ratio(aligned, reached),
            "pickup_given_align": _ratio(picked, aligned),
            "placement_given_pickup": _ratio(placed, picked),
            "stage_counts": {
                STAGE_NAMES[index]: int((stage == index).sum())
                for index in range(len(STAGE_NAMES))
            },
            "held_at_some_point": int(
                self.ever_held.detach().cpu().numpy().sum()
            ),
            "pickup_regrasp_decisions": int(
                self.pickup_regrasp.detach().cpu().numpy().sum()
            ),
            # Only real reasons. Code 0 is "no failure recorded" and reporting
            # it in this table reads as a failure mode called "none".
            "failure_counts": {
                name: int((failure == index).sum())
                for index, name in enumerate(FAILURE_NAMES)
                if index > 0 and int((failure == index).sum()) > 0
            },
        }


def _first_step_value(record: Any, values: Any, stage: int) -> Any:
    """The per-world value at the FIRST step of a stage, or NaN.

    The handoff state, as opposed to the best the stage ever achieved. A stage
    that starts badly and improves and a stage that starts well and degrades
    have the same minimum and different diagnoses.
    """

    import numpy as np

    in_stage = (record.step_stage == stage) & record.active
    first = np.argmax(in_stage, axis=0)
    entered = in_stage.any(axis=0)
    picked = values[first, np.arange(values.shape[1])]
    return np.where(entered, picked, np.nan)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


# --------------------------------------------------------------------------
# Stage-local task states
# --------------------------------------------------------------------------


def clone_task_state(
    torch: Any, state: BatchedTaskState, *, instruction_id: int
) -> BatchedTaskState:
    """An INDEPENDENT observer for one stage's native predicate.

    Every mutable field is cloned. ``evaluate_active_sparse_tasks`` writes
    ``ever_grasped``, ``grasped``, ``step_count``, ``peak_lift`` and
    ``release_clearance`` in place, so an observer that shared any of them with
    another would consume its history -- and the one that matters most is the
    placement observer's ``ever_grasped``, which the ``container_ok`` test
    requires and which the pickup evaluator would otherwise be setting and
    clearing on its own schedule.
    """

    return BatchedTaskState(
        instruction_ids=torch.full_like(
            state.instruction_ids, int(instruction_id)
        ),
        target_slots=state.target_slots.clone(),
        reference_slots=state.reference_slots.clone(),
        second_reference_slots=state.second_reference_slots.clone(),
        initial_target_positions=state.initial_target_positions.clone(),
        ever_grasped=state.ever_grasped.clone(),
        grasped=state.grasped.clone(),
        step_count=torch.zeros_like(state.step_count),
        release_threshold=state.release_threshold.clone(),
        support_surface_z=state.support_surface_z.clone(),
        target_rest_height=(
            None
            if state.target_rest_height is None
            else state.target_rest_height.clone()
        ),
        peak_lift=(
            None if state.peak_lift is None else torch.zeros_like(state.peak_lift)
        ),
        release_clearance=(
            None
            if state.release_clearance is None
            else torch.full_like(state.release_clearance, float("nan"))
        ),
    )


# --------------------------------------------------------------------------
# Teachers
# --------------------------------------------------------------------------

# Saved-args fields every teacher must agree on, because they define the
# observation, action and timing contract the chain is recorded under. A
# mismatch here does not raise anywhere downstream: it produces a bank whose
# rows were generated under two different action scales, and every number
# computed from it is then an average of two experiments.
TEACHER_CONTRACT_FIELDS: tuple[str, ...] = (
    "state_dim",
    "action_dim",
    "chunk_size",
    "replan_every",
    "image_size",
    "include_wrist",
    "include_aux_camera",
    "mask_empty_aux_camera",
    "smolvla_action_normalization",
    "smolvla_model_image_size",
    "base_checkpoint",
    "action_step_xyz",
    "action_step_yaw",
    "action_step_gripper",
    "hold_steps",
    "residual_vision_features",
    "residual_vision_dim",
    "residual_relative_target",
    "residual_vision_pooling",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class TeacherEntry:
    role: str
    checkpoint: Path
    sha256: str
    args: Mapping[str, Any]
    policy_state: Mapping[str, Any]
    lora_state: Mapping[str, Any] | None
    residual_scale: float


class TeacherBank:
    """One frozen SmolVLA, three sets of adapter weights, swapped by role.

    Three full runtimes would be three copies of the VLM; the adapters are the
    only part that differs, so the bank holds one runtime and one residual
    module and swaps the weights. ``activate`` is a no-op when the requested
    role is already resident, so a decision in which every live world is in the
    same stage costs no swap at all, and the worst case is three per decision.

    Weights are never averaged and never partially loaded: the residual is
    loaded strictly, and the LoRA load is checked key-for-key against what the
    attached adapter exposes. A LoRA state dict that matches nothing loads
    silently under ``strict=False``, which is how a probe once measured the
    stock prior while believing it was measuring an adapted one.
    """

    def __init__(
        self,
        *,
        torch: Any,
        runtime: Any,
        trainer: Any,
        entries: Sequence[TeacherEntry],
    ) -> None:
        self.torch = torch
        self.runtime = runtime
        self.trainer = trainer
        self.entries = {entry.role: entry for entry in entries}
        missing = [role for role in TEACHER_ROLES if role not in self.entries]
        if missing:
            raise ValueError(f"The teacher manifest has no entry for {missing}.")
        self.active_role: str | None = None
        self._lora_keys = {
            name
            for name in runtime.policy.state_dict()
            if "lora_" in name
        }

    def activate(self, role: str) -> None:
        if role == self.active_role:
            return
        entry = self.entries[role]
        base = self.trainer._unwrap(self.trainer.actor)
        base.load_state_dict(
            {
                key: value.to(self.trainer.device)
                for key, value in entry.policy_state.items()
            },
            strict=True,
        )
        base.residual_scale = float(entry.residual_scale)
        base.eval()
        if entry.lora_state:
            unknown = sorted(set(entry.lora_state) - self._lora_keys)
            if unknown:
                raise RuntimeError(
                    f"Teacher {role!r} carries {len(unknown)} LoRA tensors the "
                    f"attached adapter does not expose, e.g. {unknown[:3]}. "
                    "strict=False would load nothing and the prior would "
                    "silently be the stock SmolVLA."
                )
            absent = sorted(self._lora_keys - set(entry.lora_state))
            if absent:
                raise RuntimeError(
                    f"Teacher {role!r} is missing {len(absent)} of the "
                    "attached adapter's LoRA tensors, e.g. "
                    f"{absent[:3]}. Its residual was trained against a "
                    "complete adapter and cannot be reconstructed."
                )
            self.runtime.policy.load_state_dict(entry.lora_state, strict=False)
        elif self._lora_keys:
            raise RuntimeError(
                f"Teacher {role!r} carries no LoRA weights but the runtime has "
                f"{len(self._lora_keys)} LoRA tensors attached. Loading it "
                "would run its residual against another teacher's adapted "
                "prior."
            )
        self.runtime.policy.eval()
        self.active_role = role

    def manifest(self) -> dict[str, Any]:
        return {
            role: {
                "checkpoint": str(entry.checkpoint),
                "sha256": entry.sha256,
                "residual_scale": float(entry.residual_scale),
                "base_checkpoint": str(entry.args.get("base_checkpoint")),
                "contract": {
                    field_name: _jsonable(entry.args.get(field_name))
                    for field_name in TEACHER_CONTRACT_FIELDS
                },
            }
            for role, entry in sorted(self.entries.items())
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def load_teacher_entries(
    torch: Any, roles: Mapping[str, Path]
) -> list[TeacherEntry]:
    """Read each role's checkpoint and refuse an incompatible set.

    The compatibility test is on the saved ARGS, not on tensor shapes. Shapes
    catch a different state_dim; they do not catch two teachers trained at
    different ``action_step_yaw``, which would make one chain's recorded yaw
    commands mean two different rotations.
    """

    entries: list[TeacherEntry] = []
    for role in TEACHER_ROLES:
        path = Path(roles[role]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Teacher checkpoint for role {role!r} not found: {path}"
            )
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - PyTorch before weights_only
            payload = torch.load(path, map_location="cpu")
        if "policy" not in payload or not isinstance(
            payload.get("args"), Mapping
        ):
            raise ValueError(
                f"{path} is not a GRPO policy checkpoint with saved args."
            )
        args = dict(payload["args"])
        entries.append(
            TeacherEntry(
                role=role,
                checkpoint=path,
                sha256=sha256_file(path),
                args=args,
                policy_state={
                    key: value.clone()
                    for key, value in payload["policy"].items()
                },
                lora_state=(
                    {
                        key: value.clone()
                        for key, value in dict(payload["vla_lora"]).items()
                    }
                    if payload.get("vla_lora")
                    else None
                ),
                residual_scale=float(
                    payload.get("residual_scale", args.get("residual_scale", 1.0))
                ),
            )
        )
    reference = entries[0]
    for entry in entries[1:]:
        differing = {
            name: (
                _jsonable(reference.args.get(name)),
                _jsonable(entry.args.get(name)),
            )
            for name in TEACHER_CONTRACT_FIELDS
            if _jsonable(reference.args.get(name))
            != _jsonable(entry.args.get(name))
        }
        if differing:
            raise ValueError(
                f"Teachers {reference.role!r} and {entry.role!r} disagree on "
                f"the observation/action contract: {differing}. A chain "
                "recorded across them would mix two experiments."
            )
    return entries


# --------------------------------------------------------------------------
# The continuous three-stage rollout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StagedRolloutConfig:
    """Everything the loop needs that is not a live object."""

    actions_per_decision: int
    state_dim: int
    chunk_size: int
    budgets: StageBudgets
    calibration: PickupYawCalibration
    readiness: PickupReadiness = field(default_factory=PickupReadiness)
    include_relative_target: bool = False
    vision_feature_dim: int = 0
    microbatch_size: int = 0
    action_step_xyz: float = 0.015
    action_step_yaw: float = 0.08
    action_step_gripper: float = 0.05
    # HOLD THE HAND OPEN THROUGH THE APPROACH, as a recorded override.
    #
    # Measured 2026-09-10 on the teacher screen: the reach predicate fired on
    # 29-35 of 64 worlds, and 100% of those steps arrived with the gripper
    # already closed -- 1075 of 1075 for the first candidate. Seven to sixteen
    # chains of 64 closed hard enough to grasp the object outright, which is a
    # skipped stage.
    #
    # This is not a broken teacher. Under sparse_binary_reward the move-to
    # reward is where(success, 1.0, 0.0) with no gripper term at all, so that
    # channel is completely unconstrained for move_to -- and this is a shared
    # four-instruction policy whose pick_up and put_into experience is all
    # about closing. Nothing ever asked it to keep the hand open, so it does
    # not.
    #
    # A closed hand cannot be handed to the pickup teacher: its aligned start
    # is an OPEN gripper bracketing the object, and it was never trained to
    # open first. So the approach gets the same treatment the yaw does -- a
    # bounded, recorded, single-channel hold, with the raw teacher command
    # stored beside the applied one. It can only open; it can never squeeze.
    gripper_hold_open_before_pickup: bool = True
    # THE XY CENTRING BRIDGE, off by default.
    #
    # It closes the lateral error between the reach endpoint and the object
    # while the wrist is at the rotation clearance, so the descent starts from
    # over the object rather than beside it. It is the difference between a
    # bank and no bank: 57-60 of 64 chains currently die at the reach because
    # they land 17 mm off against a 13-18 mm tolerance.
    #
    # It is OFF by default because it is the one part of the tail that reads
    # privileged geometry -- the object's live XY, the oracle vector this
    # campaign removed from the policy's observation. As a controller whose
    # output is a recorded action it is admissible and the design contemplates
    # it by name, but it hands the controller a materially larger share of the
    # demonstration than the yaw tail does, and that is a decision to take
    # deliberately rather than inherit from a default.
    align_xy_centring: bool = False
    # Inside this the bridge commands exactly zero. 5 mm sits well inside the
    # apple's 13 mm slack, so a centred wrist is centred by a real margin.
    align_xy_deadband: float = 0.005
    # Hysteresis: the drift tolerated once the descent has begun. 9 mm still
    # sits inside the apple's 13 mm lateral slack, so a descent continuing at
    # this error is still one that can bracket the object.
    align_xy_abort: float = 0.009
    # Hand off at the rotation clearance instead of descending to the pickup
    # teacher's trained height. Every descent abort lives in that descent, and
    # the teacher is trained to approach from within 0.20 m anyway.
    align_handoff_at_clearance: bool = False
    # Damping on the yaw servo; see YawTailController for the windup this
    # exists to stop. 1.0 reproduces the undamped behaviour.
    align_yaw_servo_gain: float = 0.35
    # Gain on the alignment bridge's vertical descent only. The first
    # pause-and-recentre screen spent 72-86% of tail steps off-centre because
    # the unity-gain descent outran the lateral stabilization.
    align_descent_gain: float = 1.0
    # WHICH PROMPT DRIVES THE PICKUP STAGE.
    #
    # "pick_up" is the design's default and the teacher's own template.
    # "destination" runs the pickup stage under the episode's FINAL put_into
    # prompt instead, and it exists because of a retained measurement: the same
    # adapter commands +0.40 mean a_z while holding the object under a
    # put_into prompt and +0.02 under a pick_up one. The lift is the pickup
    # stage's known bottleneck (grasps 28-41%, lifts 3-12% of those), so which
    # prompt is asked is a screening variable, not a formality.
    #
    # It changes provenance, not physics: the stage is still the pickup
    # teacher's checkpoint driving a continuous trajectory, and the recorded
    # teacher text says which prompt produced it.
    pickup_prompt: str = "pick_up"
    # The pad offset below the body ``ee_position`` tracks, measured from the
    # MJCF. The grasp point of a resting object is its centre plus this, and
    # the readiness band is expressed against that point rather than against an
    # absolute height -- see PickupReadiness.
    pick_grasp_height_offset: float = 0.0075
    # Match the open aligned pickup reset in MjWarp's curriculum. The tail
    # physically descends here after rotating; it never resets the hand pose.
    pickup_height_above_grasp: float = 0.01
    pickup_height_tolerance: float = 0.003
    # Hold the calibrated yaw through the pickup stage. On by default as the
    # design's "initial controlled variant": the pickup teacher was trained
    # under a uniformly sampled yaw and has no reason to preserve one, so
    # without this the wrist can rotate back out of the presentation the tail
    # just established. Both the raw teacher command and the applied one are
    # recorded, so the arm can be re-run with this off and compared.
    yaw_hold_during_pickup: bool = True
    # Released by default during placement: the carry and the release are the
    # teacher's, and pinning the wrist through them would be a controller
    # writing part of the demonstration nobody asked it to write.
    yaw_hold_during_placement: bool = False
    record_frames: bool = True

    def validate(self) -> None:
        self.budgets.validate()
        self.calibration.validate()
        if not math.isfinite(self.pickup_height_tolerance) or self.pickup_height_tolerance <= 0:
            raise ValueError("pickup_height_tolerance must be finite and positive.")
        if not (self.readiness.min_height_above_grasp <= self.pickup_height_above_grasp
                <= self.readiness.max_height_above_grasp):
            raise ValueError("Pickup alignment height must lie inside the readiness band.")
        if self.actions_per_decision < 1:
            raise ValueError("actions_per_decision must be positive.")
        if not math.isfinite(self.align_descent_gain) or not (
            0.0 < self.align_descent_gain <= 1.0
        ):
            raise ValueError("align_descent_gain must be in (0, 1].")
        if self.pickup_prompt not in {"pick_up", "destination"}:
            raise ValueError(
                f"Unknown pickup_prompt {self.pickup_prompt!r}; expected "
                "'pick_up' or 'destination'."
            )
        if self.actions_per_decision > self.chunk_size:
            raise ValueError(
                f"{self.actions_per_decision} executed actions per decision "
                f"against a chunk of {self.chunk_size}. The plant cannot "
                "execute actions the actor did not emit."
            )


class StagedRolloutBuffers:
    """Preallocated host arrays for one round.

    Preallocated rather than appended because the frame arrays dominate
    everything else and their size is knowable in advance: at 240x320x3 and two
    cameras, one decision of W worlds costs W * 460 800 bytes, so a 64-world
    round over the design's 128-decision budget is 3.8 GB of pictures. That
    number belongs in the plan, not in an OOM three hours into a collection.
    """

    def __init__(
        self,
        *,
        worlds: int,
        decisions: int,
        actions_per_decision: int,
        state_dim: int,
        chunk_size: int,
        object_slots: int,
        record_frames: bool,
    ) -> None:
        import numpy as np

        steps = int(decisions) * int(actions_per_decision)
        self.np = np
        self.worlds = int(worlds)
        self.decisions = int(decisions)
        self.actions_per_decision = int(actions_per_decision)
        self.steps = steps
        self.record_frames = bool(record_frames)

        f32 = np.float32
        self.actions = np.zeros((steps, worlds, 5), f32)
        self.teacher_actions = np.zeros((steps, worlds, 5), f32)
        self.action_source = np.zeros((steps, worlds), np.int8)
        self.active = np.zeros((steps, worlds), bool)
        self.step_stage = np.zeros((steps, worlds), np.int8)
        self.ee_xyz = np.zeros((steps, worlds, 3), f32)
        self.ee_quaternion = np.zeros((steps, worlds, 4), f32)
        self.ee_yaw = np.zeros((steps, worlds), f32)
        self.gripper_opening = np.zeros((steps, worlds), f32)
        self.object_xyz = np.zeros((steps, worlds, object_slots, 3), f32)
        self.object_quaternion = np.zeros((steps, worlds, object_slots, 4), f32)
        self.physical_grasp = np.zeros((steps, worlds), bool)
        self.released = np.zeros((steps, worlds), bool)
        self.reach_success = np.zeros((steps, worlds), bool)
        self.pickup_success = np.zeros((steps, worlds), bool)
        self.placement_success = np.zeros((steps, worlds), bool)
        self.wrong_place_settled = np.zeros((steps, worlds), bool)
        self.placement_geometry_ok = np.zeros((steps, worlds), bool)
        self.target_lift = np.zeros((steps, worlds), f32)

        self.states = np.zeros((decisions, worlds, state_dim), f32)
        self.priors = np.zeros((decisions, worlds, chunk_size, 5), f32)
        self.teacher_role = np.zeros((decisions, worlds), np.int8)
        self.decision_stage = np.zeros((decisions, worlds), np.int8)
        self.decision_active = np.zeros((decisions, worlds), bool)
        self.stage_local_index = np.zeros((decisions, worlds), np.int32)

        self.overview: Any = None
        self.wrist: Any = None
        self.terminal_overview: Any = None
        self.terminal_wrist: Any = None

    def ensure_frames(self, height: int, width: int) -> None:
        if not self.record_frames or self.overview is not None:
            return
        np = self.np
        shape = (self.decisions, self.worlds, int(height), int(width), 3)
        self.overview = np.zeros(shape, np.uint8)
        self.wrist = np.zeros(shape, np.uint8)
        self.terminal_overview = np.zeros(
            (self.worlds, int(height), int(width), 3), np.uint8
        )
        self.terminal_wrist = np.zeros_like(self.terminal_overview)


def _to_uint8_frames(torch: Any, camera: Any) -> Any:
    """[W, C, H, Wd] float [0,1] -> [W, H, Wd, C] uint8 on the host."""

    picked = camera.permute(0, 2, 3, 1).float() * 255.0
    return picked.round().clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()


def sample_staged_teacher_actions(*, torch: Any, bank: Any, cameras: Any,
                                 proprio: Any, roles: Any, active: Any,
                                 role_texts: Mapping[str, Any], config: Any,
                                 sampling_seed: int = 0):
    """Evaluate each prior AND residual while that role's weights are resident."""
    worlds = int(proprio.shape[0])
    prior = torch.zeros((worlds, int(config.chunk_size), 5), device=proprio.device)
    state = torch.zeros((worlds, int(config.state_dim)), device=proprio.device)
    actions = torch.zeros((worlds, int(config.actions_per_decision), 5), device=proprio.device)
    switches = 0
    for role_index, role in enumerate(TEACHER_ROLES):
        selected = torch.nonzero(active & (roles == role_index), as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        switches += int(bank.active_role != role)
        bank.activate(role)
        devices = [proprio.device.index] if proprio.is_cuda else []
        # A downstream candidate must not consume noise that changes the next
        # decision of an upstream teacher. Preserve the caller's RNG as well.
        with torch.random.fork_rng(devices=devices):
            seed = int(sampling_seed) + role_index * 100_003
            torch.random.default_generator.manual_seed(seed)
            if proprio.is_cuda:
                with torch.cuda.device(proprio.device):
                    torch.cuda.manual_seed(seed)
            chunk_prior, vision = _sample_role(
                bank.runtime, cameras=cameras, states=proprio,
                instructions=role_texts[role], indices=selected,
                vision_dim=int(config.vision_feature_dim), microbatch=int(config.microbatch_size),
            )
        selected_state = proprio.index_select(0, selected)
        if config.vision_feature_dim:
            if vision is None:
                raise ValueError(f"Teacher {role} did not return configured vision features")
            selected_state = torch.cat([selected_state, vision.to(selected_state.dtype)], dim=-1)
        # Delaying this until after the loop mixes all priors with the LAST
        # teacher's residual as soon as different worlds reach different stages.
        chunk = bank.trainer.deterministic_action_chunks_tensor(
            states=selected_state, priors=chunk_prior,
            action_count=int(config.actions_per_decision),
        )
        prior.index_copy_(0, selected, chunk_prior.to(prior.dtype))
        state.index_copy_(0, selected, selected_state.to(state.dtype))
        actions.index_copy_(0, selected, chunk.to(actions.dtype))
    return state, prior, actions, switches


def run_staged_chains(
    *,
    backend: Any,
    collector: Any,
    resetter: Any,
    bank: TeacherBank,
    scenes: Sequence[Any],
    config: StagedRolloutConfig,
    round_index: int = 0,
    episode_uids: Sequence[str] | None = None,
    rollout_index: Sequence[int] | None = None,
) -> "StagedRound":
    """Drive one batch of chains from empty gripper to release.

    ``collector`` is a live ``RankLocalMJWarpGRPOCollector``. It is used for two
    things and nothing else: its physical-grasp detector and its task
    thresholds. Both are production code with subtle state (a persistence
    counter, a relative-pose slip test, the per-instruction container radii),
    and a second implementation of either is precisely the duplication this
    campaign has paid for repeatedly -- so they are called, not copied.
    """

    import numpy as np
    import torch

    config.validate()
    worlds = len(scenes)
    per_decision = int(config.actions_per_decision)
    total_decisions = int(config.budgets.total_decisions)

    labels = [object_label(scene.target_catalog) for scene in scenes]
    destinations = [scene.destination for scene in scenes]
    student_texts = [
        student_instruction_text(label, destination)
        for label, destination in zip(labels, destinations)
    ]
    role_texts = {
        role: [
            teacher_instruction_text(role, label, destination)
            for label, destination in zip(labels, destinations)
        ]
        for role in TEACHER_ROLES
    }
    if config.pickup_prompt == "destination":
        # The pickup stage is driven by the final goal instead of "pick up X".
        # Recorded, so `pickup_teacher_text` in the bank says which prompt this
        # chain's grasp was actually produced under.
        role_texts["pick_up"] = list(student_texts)
    instruction_ids = [
        destination_instruction_id(destination) for destination in destinations
    ]

    backend.pop_nonfinite_world_events()
    reset = resetter.reset(
        scenes,
        instruction_ids=instruction_ids,
        instruction_texts=student_texts,
        horizons=[total_decisions] * worlds,
    )
    device = backend.device

    # Three independently allocated observers. The placement one IS
    # reset.task_state -- persistent from env step zero, carrying the grasp
    # history and the lift datum across both handoffs.
    move_state = clone_task_state(
        torch, reset.task_state, instruction_id=INSTRUCTION_TO_ID["move_to_object"]
    )
    pick_state = clone_task_state(
        torch, reset.task_state, instruction_id=INSTRUCTION_TO_ID["pick_up"]
    )
    place_state = reset.task_state

    machine = StageMachine(
        torch=torch,
        device=device,
        worlds=worlds,
        budgets=config.budgets,
        calibration=config.calibration,
        readiness=config.readiness,
    )
    servo = YawTailController(
        torch=torch,
        calibration=config.calibration,
        action_step_yaw=config.action_step_yaw,
        action_step_xyz=config.action_step_xyz,
        action_step_gripper=config.action_step_gripper,
        pickup_height_above_grasp=config.pickup_height_above_grasp,
        pickup_height_tolerance=config.pickup_height_tolerance,
        xy_centring_deadband=(
            float(config.align_xy_deadband)
            if config.align_xy_centring
            else None
        ),
        xy_centring_abort=(
            float(config.align_xy_abort) if config.align_xy_centring else None
        ),
        handoff_at_clearance=bool(config.align_handoff_at_clearance),
        yaw_servo_gain=float(config.align_yaw_servo_gain),
        descent_gain=float(config.align_descent_gain),
    )

    # Per-object lateral slack, constant for the round. Reported loudly when a
    # catalog cannot be grasped at all rather than silently yielding nothing.
    initial_low_dim = backend.low_dim_observations()
    reset_object_xyz = initial_low_dim.object_positions.cpu().numpy().copy()
    reset_ee_xyz = initial_low_dim.ee_position.cpu().numpy().copy()
    pickup_z = (initial_low_dim.object_positions[:, 0, 2]
                + config.pick_grasp_height_offset + config.pickup_height_above_grasp)
    if bool(((pickup_z < backend.config.workspace_z[0]) |
             (pickup_z > backend.config.workspace_z[1])).any().item()):
        raise ValueError("Pickup alignment height is outside the controller Z bounds.")
    if config.align_handoff_at_clearance:
        print(
            "[staged] alignment handoff: rotation clearance "
            f"z={config.calibration.safe_rotation_z:.3f} m, fixed yaw; "
            "pickup teacher owns descent",
            flush=True,
        )
    else:
        print(
            "[staged] alignment handoff: grasp point + "
            f"{config.pickup_height_above_grasp:.3f} m "
            f"(tolerance {config.pickup_height_tolerance:.3f} m), fixed yaw; "
            "recorded descent with vertical pause for XY recentering, "
            f"descent gain {config.align_descent_gain:.2f}",
            flush=True,
        )
    quaternions = initial_low_dim.object_quaternions[:, 0].cpu().numpy()
    slack = [projected_grasp_xy_offset(
        scene.target_catalog, quaternion, config.calibration.target_yaw,
        margin=config.readiness.grasp_xy_margin,
    ) for scene, quaternion in zip(scenes, quaternions)]
    blocked = [scene for scene, value in zip(scenes, slack) if value <= 0.0]
    infeasible = sorted({scene.target_catalog for scene in blocked})
    if infeasible:
        # Split the blame. A manifest filtered with SceneGeometryConfig's
        # clearance_pickup_yaw predicted feasibility at the COMMANDED
        # orientation; anything infeasible here that the prediction passed was
        # turned broadside by SETTLING, which is a physical fact the manifest
        # cannot see and a reason to keep this measurement even when the
        # filter is on.
        settled_only = sum(
            1
            for scene in blocked
            if projected_grasp_xy_offset(
                scene.target_catalog,
                scene_object_quaternion(scene.target.yaw),
                config.calibration.target_yaw,
                margin=config.readiness.grasp_xy_margin,
            ) > 0.0
        )
        print(
            "[staged] WARNING: "
            f"{len(blocked)}/{len(scenes)} scenes {infeasible} have no "
            "positive centering clearance at their settled object orientation "
            "and calibrated pickup yaw; of those, "
            f"{settled_only} were graspable at the yaw the manifest commanded "
            "and were turned broadside by settling. This is a per-presentation "
            "gate, not a claim that every yaw of the catalog is ungraspable. "
            "Generate the manifest with --yaw-calibration to reject the "
            "commanded-orientation cases before they reach the GPU.",
            flush=True,
        )
    max_xy_offset = torch.tensor(
        slack, dtype=torch.float32, device=backend.device
    )

    object_slots = int(backend.low_dim_observations().object_positions.shape[1])
    buffers = StagedRolloutBuffers(
        worlds=worlds,
        decisions=total_decisions,
        actions_per_decision=per_decision,
        state_dim=int(config.state_dim),
        chunk_size=int(config.chunk_size),
        object_slots=object_slots,
        record_frames=bool(config.record_frames),
    )

    thresholds = collector._task_thresholds()
    move_reward = collector.move_to_distance_reward
    catch_reward = collector.catch_release_dense_reward
    # Never let a stage observer's own timeout end a world. The chain's budget
    # is the StageMachine's, and a predicate that terminated on step_count
    # would end a placement at the pick_up horizon.
    never_timeout = 1_000_000_000

    proprio_dim = int(config.state_dim) - int(config.vision_feature_dim)
    if proprio_dim <= 0:
        raise ValueError(
            f"state_dim {config.state_dim} is not wider than the vision "
            f"feature {config.vision_feature_dim}."
        )

    role_switches = 0
    world_rows = torch.arange(worlds, dtype=torch.int64, device=device)
    release_xy_radius = torch.tensor(
        [scene.destination_success_radius for scene in scenes], device=device
    )
    release_latched = torch.zeros((worlds,), dtype=torch.bool, device=device)
    diverged_any = torch.zeros((worlds,), dtype=torch.bool, device=device)
    with torch.inference_mode():
        for decision in range(total_decisions):
            if not bool(machine.active.any().item()):
                break
            cameras = backend.render_policy_cameras()
            low_dim = backend.low_dim_observations()
            if config.record_frames:
                buffers.ensure_frames(
                    int(cameras.overview.shape[2]), int(cameras.overview.shape[3])
                )
                buffers.overview[decision] = _to_uint8_frames(
                    torch, cameras.overview
                )
                buffers.wrist[decision] = _to_uint8_frames(torch, cameras.wrist)

            proprio = _build_state(
                torch,
                low_dim=low_dim,
                task_state=place_state,
                state_dim=proprio_dim,
                include_relative_target=config.include_relative_target,
            )
            roles = machine.stage_role_ids()
            state_tensor, prior, teacher_chunk, switches = sample_staged_teacher_actions(
                torch=torch, bank=bank, cameras=cameras, proprio=proprio,
                roles=roles, active=machine.active, role_texts=role_texts, config=config,
                sampling_seed=int(round_index) * 10_000_019 + decision * 1_000_003,
            )
            role_switches += switches

            buffers.states[decision] = state_tensor.float().cpu().numpy()
            buffers.priors[decision] = prior.float().cpu().numpy()
            buffers.teacher_role[decision] = roles.to(torch.int8).cpu().numpy()
            buffers.decision_stage[decision] = (
                machine.stage.to(torch.int8).cpu().numpy()
            )
            buffers.decision_active[decision] = (
                machine.active.cpu().numpy()
            )
            buffers.stage_local_index[decision] = (
                machine.stage_decisions.to(torch.int32).cpu().numpy()
            )

            # Latched over the chunk, not read at its last step. The stage
            # machine reads at decision boundaries, and an opening ramp is not
            # obliged to be increasing on the exact step the boundary lands on
            # -- the controller saturates, a finger sticks for a step. Reading
            # the instant would end a correct placement for a one-step pause in
            # a release that is plainly under way. The recording's own
            # acceptance check is the strict one; this is the live guard.
            release_latched.zero_()
            for action_index in range(per_decision):
                step = decision * per_decision + action_index
                step_active = machine.active.clone()
                raw = teacher_chunk[:, action_index].clone()
                applied, source = _apply_overrides(
                    torch,
                    raw=raw,
                    stage=machine.stage,
                    low_dim=low_dim,
                    servo=servo,
                    hold_pickup=config.yaw_hold_during_pickup,
                    hold_placement=config.yaw_hold_during_placement,
                    hold_gripper_open=config.gripper_hold_open_before_pickup,
                    grasp_point_z=(low_dim.object_positions[world_rows, place_state.target_slots, 2]
                                   + config.pick_grasp_height_offset),
                    target_xy=(
                        low_dim.object_positions[
                            world_rows, place_state.target_slots, :2
                        ]
                        if config.align_xy_centring
                        else None
                    ),
                )

                buffers.teacher_actions[step] = raw.float().cpu().numpy()
                buffers.actions[step] = applied.float().cpu().numpy()
                buffers.action_source[step] = source.to(torch.int8).cpu().numpy()
                buffers.active[step] = step_active.cpu().numpy()
                buffers.step_stage[step] = machine.stage.to(torch.int8).cpu().numpy()

                previous_opening = low_dim.gripper_opening.clone()
                low_dim = backend.step(applied, step_active)
                release_in_progress = release_opening_over_goal(
                    command=applied[:, 4], opening=low_dim.gripper_opening,
                    previous_opening=previous_opening,
                    target_xy=low_dim.object_positions[world_rows, place_state.target_slots, :2],
                    receptacle_xy=low_dim.object_positions[world_rows, place_state.reference_slots, :2],
                    radius=release_xy_radius,
                )
                release_latched |= release_in_progress & step_active
                low_dim, caught, grasp_diagnostics = (
                    collector._update_physical_grasp(reset, low_dim, step_active)
                )
                bilateral = grasp_diagnostics["bilateral_contact"]

                move_result = _evaluate(
                    state=move_state,
                    low_dim=low_dim,
                    caught=caught,
                    active=step_active,
                    thresholds=thresholds,
                    move_reward=move_reward,
                    catch_reward=catch_reward,
                    bilateral=bilateral,
                    max_steps=never_timeout,
                )
                pick_result = _evaluate(
                    state=pick_state,
                    low_dim=low_dim,
                    caught=caught,
                    active=step_active,
                    thresholds=thresholds,
                    move_reward=move_reward,
                    catch_reward=catch_reward,
                    bilateral=bilateral,
                    max_steps=never_timeout,
                )
                place_result = _evaluate(
                    state=place_state,
                    low_dim=low_dim,
                    caught=caught,
                    active=step_active,
                    thresholds=thresholds,
                    move_reward=move_reward,
                    catch_reward=catch_reward,
                    bilateral=bilateral,
                    max_steps=never_timeout,
                )

                buffers.ee_xyz[step] = low_dim.ee_position.float().cpu().numpy()
                buffers.ee_quaternion[step] = (
                    low_dim.ee_quaternion.float().cpu().numpy()
                )
                buffers.ee_yaw[step] = low_dim.ee_yaw.float().cpu().numpy()
                buffers.gripper_opening[step] = (
                    low_dim.gripper_opening.float().cpu().numpy()
                )
                buffers.object_xyz[step] = (
                    low_dim.object_positions.float().cpu().numpy()
                )
                buffers.object_quaternion[step] = (
                    low_dim.object_quaternions.float().cpu().numpy()
                )
                buffers.physical_grasp[step] = caught.cpu().numpy()
                buffers.released[step] = (
                    place_result.diagnostics["released"].cpu().numpy()
                )
                buffers.reach_success[step] = move_result.success.cpu().numpy()
                buffers.pickup_success[step] = pick_result.success.cpu().numpy()
                buffers.placement_success[step] = (
                    place_result.success.cpu().numpy()
                )
                buffers.wrong_place_settled[step] = (
                    place_result.diagnostics["wrong_place_drop"].cpu().numpy()
                )
                geometry_ok = _placement_geometry_ok(
                    place_result, thresholds=thresholds, catch_reward=catch_reward
                )
                buffers.placement_geometry_ok[step] = geometry_ok.cpu().numpy()
                buffers.target_lift[step] = (
                    pick_result.diagnostics["target_lift"].float().cpu().numpy()
                )

            # The boundary read. Native successes latched anywhere inside the
            # chunk are OR-ed over its steps; the readiness conditions are read
            # from the state the world is in NOW, which is the point of asking
            # at a boundary rather than at the event.
            window = slice(
                decision * per_decision, (decision + 1) * per_decision
            )
            def latched(array: Any) -> Any:
                return torch.as_tensor(
                    array[window].any(axis=0), device=device
                )

            diverged_now, _ = _pop_divergence(backend, worlds, device)
            diverged_any |= diverged_now
            machine.advance(
                decision=decision,
                reach_success=latched(buffers.reach_success),
                pickup_success=latched(buffers.pickup_success),
                placement_success=latched(buffers.placement_success),
                # Boundary state, NOT latched: the settle check asks whether
                # the object is in the receptacle NOW. An OR over the chunk
                # would pass a world for having been there at some point.
                placement_geometry_ok=geometry_ok.clone(),
                wrong_place_settled=latched(buffers.wrong_place_settled),
                physical_grasp=caught.clone(),
                released=place_result.diagnostics["released"].clone(),
                release_in_progress=release_latched,
                gripper_opening=low_dim.gripper_opening.clone(),
                ee_position=low_dim.ee_position.clone(),
                target_lift=pick_result.diagnostics["target_lift"].clone(),
                grasp_point_z=(
                    low_dim.object_positions[
                        world_rows, place_state.target_slots, 2
                    ]
                    + float(config.pick_grasp_height_offset)
                ),
                target_xy_error=torch.linalg.vector_norm(
                    low_dim.object_positions[
                        world_rows, place_state.target_slots, :2
                    ]
                    - low_dim.ee_position[:, :2],
                    dim=-1,
                ),
                max_grasp_xy_offset=max_xy_offset,
                yaw_aligned=servo.aligned(low_dim.ee_yaw),
                alignment_ready=servo.handoff_ready(
                    ee_yaw=low_dim.ee_yaw, ee_position=low_dim.ee_position,
                    grasp_point_z=(low_dim.object_positions[world_rows, place_state.target_slots, 2]
                                   + config.pick_grasp_height_offset),
                ),
                diverged=diverged_now,
            )

        # The terminal POST-action observation. A dataset that stores only
        # pre-action frames cannot show what the last action did, and the
        # release is the last action of every successful chain.
        if config.record_frames:
            cameras = backend.render_policy_cameras()
            buffers.ensure_frames(
                int(cameras.overview.shape[2]), int(cameras.overview.shape[3])
            )
            buffers.terminal_overview[:] = _to_uint8_frames(
                torch, cameras.overview
            )
            buffers.terminal_wrist[:] = _to_uint8_frames(torch, cameras.wrist)

    _, trailing_divergence = _pop_divergence(backend, worlds, device)
    diverged_mask = diverged_any | trailing_divergence
    round_result = StagedRound.from_buffers(
        buffers=buffers,
        machine=machine,
        scenes=scenes,
        student_texts=student_texts,
        role_texts=role_texts,
        instruction_ids=instruction_ids,
        config=config,
        round_index=int(round_index),
        role_switches=int(role_switches),
        diverged_mask=diverged_mask.cpu().numpy(),
        teacher_manifest=bank.manifest(),
        episode_uids=(
            list(episode_uids)
            if episode_uids is not None
            else [f"round{round_index}/r{round_index}w{i}" for i in range(worlds)]
        ),
        rollout_index=(
            list(rollout_index)
            if rollout_index is not None
            else [0] * worlds
        ),
    )
    # from_buffers historically used object_xyz[0], which is POST action.
    # Keep true reset poses and the exact gate used by this round instead.
    round_result.reset_object_xyz = reset_object_xyz
    round_result.reset_ee_xyz = reset_ee_xyz
    settings = json.loads(round_result.config_json)
    settings["grasp_xy_slack_m"] = slack
    settings["reset_pose_timing"] = "pre_action"
    settings["controller_workspace_z_bounds"] = list(backend.config.workspace_z)
    round_result.config_json = json.dumps(settings, sort_keys=True)
    round_result.frames = buffers
    return round_result


def _build_state(
    torch: Any,
    *,
    low_dim: Any,
    task_state: BatchedTaskState,
    state_dim: int,
    include_relative_target: bool,
) -> Any:
    from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
        goal_slots_for_reset,
    )
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
        build_smolvla_state_tensor,
    )

    return build_smolvla_state_tensor(
        ee_position=low_dim.ee_position,
        ee_yaw=low_dim.ee_yaw,
        gripper_opening=low_dim.gripper_opening,
        object_positions=low_dim.object_positions,
        target_slots=task_state.target_slots,
        state_dim=int(state_dim),
        include_relative_target=bool(include_relative_target),
        goal_slots=goal_slots_for_reset(torch, _TaskStateView(task_state)),
    )


@dataclass(frozen=True)
class _TaskStateView:
    """The one attribute ``goal_slots_for_reset`` reads.

    It takes a ``BatchedReset`` and uses only ``reset.task_state``. Passing this
    view keeps ONE definition of "which slot does the reward shape toward"
    instead of restating the placement/target rule here -- a second copy of that
    rule read -0.40 once when it was written backwards.
    """

    task_state: Any


def _sample_role(
    runtime: Any,
    *,
    cameras: Any,
    states: Any,
    instructions: Sequence[str],
    indices: Any,
    vision_dim: int,
    microbatch: int,
) -> tuple[Any, Any | None]:
    """VLA forward for one role's live worlds only.

    Subset inference, not a masked full-batch one: three roles over one batch
    would otherwise cost three full VLA forwards per decision, and the worlds
    of the other two roles would be conditioned on the wrong prompt anyway.
    """

    selected = [int(value) for value in indices.tolist()]
    kwargs = dict(
        primary_images=cameras.overview[indices],
        wrist_images=cameras.wrist[indices],
        states=states[indices],
        instructions=tuple(instructions[i] for i in selected),
        microbatch_size=int(microbatch),
    )
    if int(vision_dim) > 0:
        prior, features = runtime.sample_cdpr_chunks_and_vision_from_tensors(
            **kwargs, vision_dim=int(vision_dim)
        )
        return prior, features
    return runtime.sample_cdpr_chunks_from_tensors(**kwargs), None


def _apply_overrides(
    torch: Any,
    *,
    raw: Any,
    stage: Any,
    low_dim: Any,
    servo: YawTailController,
    hold_pickup: bool,
    hold_placement: bool,
    hold_gripper_open: bool = True,
    grasp_point_z: Any = None,
    target_xy: Any = None,
) -> tuple[Any, Any]:
    """Substitute the recorded controllers where the stage calls for them.

    Applied in order of increasing authority: the single-channel holds relabel
    a teacher action, and the whole-command controllers (the alignment tail and
    the settle hold) replace it. Every substitution is recorded per action in
    ``action_source``, and the raw teacher command is stored beside the applied
    one, so no row of the bank can be mistaken for pure policy output.
    """

    applied = raw.clone()
    source = torch.zeros(
        (raw.shape[0],), dtype=torch.int64, device=raw.device
    )
    # Channel 4 only, during the approach: keep the hand open so there is a
    # hand to grasp with when the pickup teacher takes over.
    approaching = stage == STAGE_MOVE_TO
    if hold_gripper_open and bool(approaching.any().item()):
        open_command = servo.open_command(low_dim.gripper_opening)
        applied[:, 4] = torch.where(approaching, open_command, applied[:, 4])
        source = torch.where(
            approaching, torch.full_like(source, SOURCE_GRIPPER_HOLD), source
        )
    aligning = stage == STAGE_ALIGN
    if bool(aligning.any().item()):
        tail = servo.actions(
            ee_position=low_dim.ee_position,
            ee_yaw=low_dim.ee_yaw,
            grasp_point_z=grasp_point_z,
            gripper_opening=(
                low_dim.gripper_opening if hold_gripper_open else None
            ),
            target_xy=target_xy,
        )
        applied = torch.where(aligning[:, None], tail, applied)
        # A distinct source when the bridge is active: these actions include a
        # translation derived from the object's live position, which is a
        # larger claim on the demonstration than a rotation and has to be
        # countable in the bank rather than inferred from a config flag.
        tail_source = (
            SOURCE_ALIGN_BRIDGE if target_xy is not None else SOURCE_YAW_TAIL
        )
        source = torch.where(
            aligning, torch.full_like(source, tail_source), source
        )
    settling = stage == STAGE_SETTLE
    if bool(settling.any().item()):
        # A HOLD, spelled as the zero action. Under the production controller a
        # zero XYZ command sets the target to the measured pose, so the arm
        # stays where it is and the fingers stay where they are -- the object
        # is left alone while the physics decides whether it stays put. These
        # steps are recorded like any other and carry their own source code, so
        # nothing downstream mistakes them for teacher commands.
        applied = torch.where(
            settling[:, None], torch.zeros_like(applied), applied
        )
        source = torch.where(
            settling, torch.full_like(source, SOURCE_SETTLE_HOLD), source
        )
    hold = torch.zeros_like(aligning)
    if hold_pickup:
        hold = hold | (stage == STAGE_PICK_UP)
    if hold_placement:
        hold = hold | (stage == STAGE_PLACEMENT)
    if bool(hold.any().item()):
        # ONLY the yaw channel. The teacher keeps XYZ and the gripper, which is
        # what makes this a hold rather than a takeover -- and the raw command
        # is recorded beside it so the substitution is auditable row by row.
        yaw_command = servo.yaw_command(low_dim.ee_yaw)
        applied[:, 3] = torch.where(hold, yaw_command, applied[:, 3])
        source = torch.where(
            hold & (source == SOURCE_TEACHER),
            torch.full_like(source, SOURCE_YAW_HOLD),
            source,
        )
    return applied.clamp(-1.0, 1.0), source


def _evaluate(
    *,
    state: BatchedTaskState,
    low_dim: Any,
    caught: Any,
    active: Any,
    thresholds: Any,
    move_reward: Any,
    catch_reward: Any,
    bilateral: Any,
    max_steps: int,
) -> Any:
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
        evaluate_active_sparse_tasks,
    )

    return evaluate_active_sparse_tasks(
        state=state,
        ee_position=low_dim.ee_position,
        object_positions=low_dim.object_positions,
        gripper_opening=low_dim.gripper_opening,
        caught_target=caught,
        active_mask=active,
        max_steps=int(max_steps),
        thresholds=thresholds,
        move_to_distance_reward=move_reward,
        catch_release_dense_reward=catch_reward,
        bilateral_contact=bilateral,
    )


def _placement_geometry_ok(
    result: Any, *, thresholds: Any, catch_reward: Any
) -> Any:
    """Is the object where it was asked to be put, gripper aside?

    The GEOMETRIC half of ``container_ok``, rebuilt from the predicate's own
    published diagnostics rather than recomputed from poses, so the settle check
    and the success test cannot drift apart on a radius or a tolerance.
    """

    diagnostics = result.diagnostics
    xy_ok = (
        diagnostics["container_xy_error"] <= diagnostics["container_xy_radius"]
    )
    z_error = diagnostics["container_z_error"]
    z_ok = z_error <= float(thresholds.container_z)
    if catch_reward is not None:
        z_ok = z_ok & (z_error <= float(catch_reward.container_z_tolerance))
    return xy_ok & z_ok


def _pop_divergence(backend: Any, worlds: int, device: Any) -> tuple[Any, Any]:
    """Which worlds went non-finite since the last call.

    A diverged world is neither a success nor a usable demonstration, and the
    campaign has already had to discard whole 512-world rounds because the
    backend could only report a COUNT. Prefer the per-world report; fall back to
    marking the whole batch when only a count is available, which is
    conservative and never a silent all-clear.
    """

    import torch

    report = getattr(backend, "pop_nonfinite_world_report", None)
    if report is None:
        count = int(backend.pop_nonfinite_world_events())
        mask = torch.full(
            (worlds,), bool(count), dtype=torch.bool, device=device
        )
        return mask, mask
    count, raw = report()
    if raw is None:
        mask = torch.full(
            (worlds,), bool(int(count)), dtype=torch.bool, device=device
        )
        return mask, mask
    mask = torch.as_tensor(raw, dtype=torch.bool, device=device).reshape(-1)
    return mask, mask


# --------------------------------------------------------------------------
# The durable recording
# --------------------------------------------------------------------------


@dataclass
class StagedRound:
    """One round of chains, exactly as physics executed them.

    Nothing in here is derived from a model of what should have happened: every
    array is a tensor the plant or a production predicate produced, copied to
    the host. Acceptance is computed FROM these arrays by ``acceptance()`` and
    is not stored as a verdict handed down by the loop, so a stricter contract
    can be applied later to a bank already on disk.
    """

    # Per env step, [S, W, ...].
    actions: Any
    teacher_actions: Any
    action_source: Any
    active: Any
    step_stage: Any
    ee_xyz: Any
    ee_quaternion: Any
    ee_yaw: Any
    gripper_opening: Any
    object_xyz: Any
    object_quaternion: Any
    physical_grasp: Any
    released: Any
    reach_success: Any
    pickup_success: Any
    placement_success: Any
    wrong_place_settled: Any
    placement_geometry_ok: Any
    target_lift: Any

    # Per policy decision, [D, W, ...].
    states: Any
    priors: Any
    teacher_role: Any
    decision_stage: Any
    decision_active: Any
    stage_local_index: Any

    # Per world, [W].
    episode_uid: Any
    rollout_index: Any
    scene_uid: Any
    split: Any
    destination: Any
    target_catalog: Any
    instruction_id: Any
    instruction_text: Any
    move_teacher_text: Any
    pickup_teacher_text: Any
    placement_teacher_text: Any
    approach_xy_distance: Any
    transport_xy_distance: Any
    destination_success_radius: Any
    reset_object_xyz: Any
    reset_ee_xyz: Any
    reach_event: Any
    align_event: Any
    pickup_event: Any
    placement_event: Any
    final_stage: Any
    failure_code: Any
    handoff_lift: Any
    diverged_world_mask: Any

    # Round level.
    actions_per_decision: int
    round_index: int
    role_switches: int
    budgets_json: str
    calibration_json: str
    teacher_manifest_json: str
    config_json: str

    # The pictures, held only while the round is in hand. Deliberately NOT
    # serialized with the record and not part of any of the field tuples below:
    # frames are by far the largest thing in the process (460.8 kB per world per
    # decision across the two cameras), and a round that carried them could not
    # be kept in a list of rounds. The recorder writes the accepted worlds'
    # frames to their own shard and drops this.
    frames: Any = None

    _PER_STEP = (
        "actions", "teacher_actions", "action_source", "active", "step_stage",
        "ee_xyz", "ee_quaternion", "ee_yaw", "gripper_opening", "object_xyz",
        "object_quaternion", "physical_grasp", "released", "reach_success",
        "pickup_success", "placement_success", "wrong_place_settled",
        "placement_geometry_ok", "target_lift",
    )
    _PER_DECISION = (
        "states", "priors", "teacher_role", "decision_stage",
        "decision_active", "stage_local_index",
    )
    _PER_WORLD = (
        "episode_uid", "rollout_index",
        "scene_uid", "split", "destination", "target_catalog",
        "instruction_id", "instruction_text", "move_teacher_text",
        "pickup_teacher_text", "placement_teacher_text",
        "approach_xy_distance", "transport_xy_distance",
        "destination_success_radius", "reset_object_xyz", "reset_ee_xyz",
        "reach_event", "align_event", "pickup_event", "placement_event",
        "final_stage", "failure_code", "handoff_lift", "diverged_world_mask",
    )
    _SCALARS = (
        ("actions_per_decision", "int64"),
        ("round_index", "int64"),
        ("role_switches", "int64"),
    )
    _STRINGS = (
        "budgets_json", "calibration_json", "teacher_manifest_json",
        "config_json",
    )

    @property
    def worlds(self) -> int:
        return int(self.actions.shape[1])

    @property
    def decisions(self) -> int:
        return int(self.states.shape[0])

    @classmethod
    def from_buffers(
        cls,
        *,
        buffers: StagedRolloutBuffers,
        machine: StageMachine,
        scenes: Sequence[Any],
        student_texts: Sequence[str],
        role_texts: Mapping[str, Sequence[str]],
        instruction_ids: Sequence[int],
        config: StagedRolloutConfig,
        round_index: int,
        role_switches: int,
        diverged_mask: Any,
        teacher_manifest: Mapping[str, Any],
        episode_uids: Sequence[str],
        rollout_index: Sequence[int],
    ) -> "StagedRound":
        import json

        import numpy as np

        host = lambda tensor: tensor.detach().cpu().numpy()  # noqa: E731

        return cls(
            actions=buffers.actions,
            teacher_actions=buffers.teacher_actions,
            action_source=buffers.action_source,
            active=buffers.active,
            step_stage=buffers.step_stage,
            ee_xyz=buffers.ee_xyz,
            ee_quaternion=buffers.ee_quaternion,
            ee_yaw=buffers.ee_yaw,
            gripper_opening=buffers.gripper_opening,
            object_xyz=buffers.object_xyz,
            object_quaternion=buffers.object_quaternion,
            physical_grasp=buffers.physical_grasp,
            released=buffers.released,
            reach_success=buffers.reach_success,
            pickup_success=buffers.pickup_success,
            placement_success=buffers.placement_success,
            wrong_place_settled=buffers.wrong_place_settled,
            placement_geometry_ok=buffers.placement_geometry_ok,
            target_lift=buffers.target_lift,
            states=buffers.states,
            priors=buffers.priors,
            teacher_role=buffers.teacher_role,
            decision_stage=buffers.decision_stage,
            decision_active=buffers.decision_active,
            stage_local_index=buffers.stage_local_index,
            episode_uid=np.asarray(list(episode_uids), dtype="U128"),
            rollout_index=np.asarray(list(rollout_index), dtype=np.int64),
            scene_uid=np.asarray(
                [scene.scene_uid for scene in scenes], dtype="U64"
            ),
            split=np.asarray([scene.split for scene in scenes], dtype="U24"),
            destination=np.asarray(
                [scene.destination for scene in scenes], dtype="U8"
            ),
            target_catalog=np.asarray(
                [scene.target_catalog for scene in scenes], dtype="U32"
            ),
            instruction_id=np.asarray(instruction_ids, dtype=np.int64),
            instruction_text=np.asarray(list(student_texts), dtype="U256"),
            move_teacher_text=np.asarray(
                list(role_texts["move_to"]), dtype="U256"
            ),
            pickup_teacher_text=np.asarray(
                list(role_texts["pick_up"]), dtype="U256"
            ),
            placement_teacher_text=np.asarray(
                list(role_texts["placement"]), dtype="U256"
            ),
            approach_xy_distance=np.asarray(
                [scene.approach_xy_distance for scene in scenes],
                dtype=np.float32,
            ),
            transport_xy_distance=np.asarray(
                [scene.transport_xy_distance for scene in scenes],
                dtype=np.float32,
            ),
            destination_success_radius=np.asarray(
                [scene.destination_success_radius for scene in scenes],
                dtype=np.float32,
            ),
            # The realized pre-action poses, taken from step 0 of the recording
            # rather than from the manifest. They are what the plant settled
            # to, and a scene's nominal value is not.
            reset_object_xyz=buffers.object_xyz[0].copy(),
            reset_ee_xyz=buffers.ee_xyz[0].copy(),
            reach_event=host(machine.reach_event),
            align_event=host(machine.align_event),
            pickup_event=host(machine.pickup_event),
            placement_event=host(machine.placement_event),
            final_stage=host(machine.stage),
            failure_code=host(machine.failure),
            handoff_lift=host(machine.handoff_lift),
            diverged_world_mask=np.asarray(diverged_mask, dtype=bool),
            actions_per_decision=int(config.actions_per_decision),
            round_index=int(round_index),
            role_switches=int(role_switches),
            budgets_json=json.dumps(
                {
                    "move_decisions": int(config.budgets.move_decisions),
                    "align_decisions": int(config.budgets.align_budget),
                    "pickup_decisions": int(config.budgets.pickup_decisions),
                    "placement_decisions": int(
                        config.budgets.placement_decisions
                    ),
                    "settle_decisions": int(config.budgets.settle_decisions),
                },
                sort_keys=True,
            ),
            calibration_json=json.dumps(
                config.calibration.to_json(), sort_keys=True
            ),
            teacher_manifest_json=json.dumps(
                dict(teacher_manifest), sort_keys=True
            ),
            config_json=json.dumps(
                {
                    "actions_per_decision": int(config.actions_per_decision),
                    "teacher_sampling": "role_decision_seeded_v1",
                    "state_dim": int(config.state_dim),
                    "chunk_size": int(config.chunk_size),
                    "vision_feature_dim": int(config.vision_feature_dim),
                    "include_relative_target": bool(
                        config.include_relative_target
                    ),
                    "action_step_xyz": float(config.action_step_xyz),
                    "action_step_yaw": float(config.action_step_yaw),
                    "pickup_height_above_grasp": float(config.pickup_height_above_grasp),
                    "pickup_height_tolerance": float(config.pickup_height_tolerance),
                    "carry_acceptance": CARRY_ACCEPTANCE_VERSION,
                    "yaw_hold_during_pickup": bool(
                        config.yaw_hold_during_pickup
                    ),
                    "yaw_hold_during_placement": bool(
                        config.yaw_hold_during_placement
                    ),
                    "gripper_hold_open_before_pickup": bool(
                        config.gripper_hold_open_before_pickup
                    ),
                    "align_xy_centring": bool(config.align_xy_centring),
                    "align_xy_deadband": float(config.align_xy_deadband),
                    "align_xy_abort": float(config.align_xy_abort),
                    "align_handoff_at_clearance": bool(
                        config.align_handoff_at_clearance
                    ),
                    "align_yaw_servo_gain": float(config.align_yaw_servo_gain),
                    "align_descent_gain": float(config.align_descent_gain),
                    "require_centred_at_reach": bool(
                        config.readiness.require_centred_at_reach
                    ),
                    "pickup_prompt": str(config.pickup_prompt),
                    "action_step_gripper": float(config.action_step_gripper),
                    "pick_grasp_height_offset": float(
                        config.pick_grasp_height_offset
                    ),
                    "readiness": {
                        "require_centred_at_reach": bool(
                            config.readiness.require_centred_at_reach
                        ),
                        "grasp_xy_margin": float(
                            config.readiness.grasp_xy_margin
                        ),
                        "min_height_above_grasp": float(
                            config.readiness.min_height_above_grasp
                        ),
                        "max_height_above_grasp": float(
                            config.readiness.max_height_above_grasp
                        ),
                        "min_ee_z": float(config.readiness.min_ee_z),
                        "max_ee_z": float(config.readiness.max_ee_z),
                        "min_gripper_opening": float(
                            config.readiness.min_gripper_opening
                        ),
                        "forbid_grasp": bool(config.readiness.forbid_grasp),
                    },
                },
                sort_keys=True,
            ),
        )

    # -- acceptance -----------------------------------------------------

    def acceptance(
        self,
        *,
        min_approach_xy: float = 0.06,
        min_handoff_lift: float = 0.05,
    ) -> tuple[Any, Any, dict[str, int]]:
        """Full-chain data acceptance, computed from the recording.

        Deliberately NOT the same thing as the production placement verdict.
        The production predicate answers "did this policy put the object in the
        receptacle"; this answers "is this trajectory a demonstration of the
        complete task". A chain that succeeds under the first and fails here is
        reported under both numbers rather than having one rewrite the other.
        """

        import numpy as np

        worlds = self.worlds
        reasons = np.array(["accepted"] * worlds, dtype="U48")
        accepted = np.ones((worlds,), dtype=bool)

        def reject(mask: Any, reason: str) -> None:
            # FIRST reason wins. A chain that diverged and then failed three
            # more tests should be counted once, under the thing that actually
            # ended it, or the rejection census sums to more than the batch.
            fresh = np.asarray(mask, dtype=bool) & accepted
            reasons[fresh] = reason
            accepted[fresh] = False

        reject(self.final_stage != STAGE_COMPLETE, "chain_did_not_complete")
        reject(np.asarray(self.diverged_world_mask, dtype=bool), "diverged")

        # Began empty, open, and with the object on the desk. The scene
        # guarantees it; this checks the PLANT agreed, which is the difference
        # between a manifest and a measurement.
        reject(self.physical_grasp[0], "started_grasping")
        reject(self.gripper_opening[0] < 0.90, "started_closed")

        # Genuine approach and a genuine transport, re-derived from the
        # realized reset poses rather than from the manifest's stored values.
        target_xy = self.reset_object_xyz[:, 0, :2]
        receptacle_xy = self.reset_object_xyz[:, 1, :2]
        approach = np.linalg.norm(self.reset_ee_xyz[:, :2] - target_xy, axis=-1)
        transport = np.linalg.norm(target_xy - receptacle_xy, axis=-1)
        reject(approach < float(min_approach_xy) - 1e-6, "approach_too_short")
        reject(
            transport <= self.destination_success_radius,
            "started_inside_goal",
        )

        # Every stage event happened, in order.
        reject(self.reach_event < 0, "no_reach_event")
        reject(self.align_event < self.reach_event, "align_before_reach")
        reject(self.pickup_event < self.align_event, "pickup_before_align")
        reject(
            self.placement_event < self.pickup_event, "placement_before_pickup"
        )
        # The production 5 cm, measured from the true desk datum at the moment
        # the placement teacher took over -- not at the instant the pickup
        # predicate first fired, which the world may already have left.
        reject(
            self.handoff_lift < float(min_handoff_lift) - 1e-6,
            "no_lift_at_handoff",
        )

        # Continuous hold through transport, up to the intentional release.
        reject(~self.continuous_carry(), "carry_interrupted")

        # Finite everywhere it was live. Inactive steps are exempt: the plant
        # is not being driven there and the arrays are padding.
        finite = np.isfinite(self.actions).all(axis=-1) & np.isfinite(
            self.ee_xyz
        ).all(axis=-1)
        reject(~(finite | ~self.active).all(axis=0), "non_finite_state")

        counts: dict[str, int] = {}
        for reason in np.unique(reasons):
            counts[str(reason)] = int((reasons == reason).sum())
        return accepted, reasons, counts

    def carry_release_start_steps(self) -> Any:
        """First contact-loss step in a verified opening sequence, or threshold.

        Only the suffix immediately leading to the first release threshold is
        eligible. Every step must command and realize opening over the goal,
        with no regrasp. This does not forgive a slip followed by later opening.
        A missing release leaves the entire active carry window checked.
        """
        import numpy as np

        previous = np.concatenate([self.gripper_opening[:1], self.gripper_opening[:-1]])
        opening = release_opening_over_goal(
            command=self.actions[..., 4], opening=self.gripper_opening,
            previous_opening=previous, target_xy=self.object_xyz[:, :, 0, :2],
            receptacle_xy=self.object_xyz[:, :, 1, :2], radius=self.destination_success_radius,
        )
        stops = np.full(self.worlds, self.actions.shape[0], dtype=np.int64)
        for world in range(self.worlds):
            if self.pickup_event[world] < 0:
                continue
            start = (int(self.pickup_event[world]) + 1) * int(self.actions_per_decision)
            opened = np.flatnonzero(self.released[start:, world] & self.active[start:, world])
            if not opened.size:
                continue
            stop = start + int(opened[0])
            stops[world] = stop
            lost = np.flatnonzero(self.active[start:stop, world] & ~self.physical_grasp[start:stop, world])
            if not lost.size:
                continue
            first_loss = start + int(lost[0])
            suffix = slice(first_loss, stop + 1)
            if (self.active[suffix, world].all() and opening[suffix, world].all()
                    and not self.physical_grasp[suffix, world].any()):
                stops[world] = first_loss
        return stops

    def continuous_carry(self) -> Any:
        """Continuous hold until contact ends during the verified release."""
        import numpy as np

        result = np.zeros(self.worlds, dtype=bool)
        stops = self.carry_release_start_steps()
        for world in range(self.worlds):
            if self.pickup_event[world] < 0:
                continue
            start = (int(self.pickup_event[world]) + 1) * int(self.actions_per_decision)
            live = self.active[start:stops[world], world]
            result[world] = bool(self.physical_grasp[start:stops[world], world][live].all())
        return result

    # -- diagnostics ----------------------------------------------------

    def reach_diagnostics(self) -> dict[str, Any]:
        """Decompose a failed reach stage into the gate that actually failed.

        ``reached: 0`` is not a finding, it is a question. The reach event is a
        CONJUNCTION -- the production XY predicate, an open gripper, no grasp,
        and a height band -- and a zero says nothing about which conjunct was
        false. The first teacher screen scored 0 of 64 for every candidate of
        every role, which is not a plausible reading of two move-to teachers
        that score 81% and 63% on their own protocols, and the answer turned
        out to be a height band whose floor sat above three of the four
        objects' natural pickup-ready heights.

        So this reports each gate separately, and the distances underneath
        them, so the next zero can be read in one pass instead of one GPU run
        per hypothesis.
        """

        import json

        import numpy as np

        settings = json.loads(self.config_json)
        readiness = dict(settings.get("readiness") or {})
        offset = float(settings.get("pick_grasp_height_offset", 0.0075))
        min_open = float(readiness.get("min_gripper_opening", 0.90))
        min_above = float(readiness.get("min_height_above_grasp", -0.005))
        max_above = float(readiness.get("max_height_above_grasp", 0.12))
        min_z = float(readiness.get("min_ee_z", 0.18))
        max_z = float(readiness.get("max_ee_z", 0.40))

        # Scoped to the stages the reach gate actually runs in. `reach_success`
        # is the move_to predicate evaluated on EVERY step, and once a world is
        # holding the object two centimetres from where it started, the
        # predicate keeps firing -- so an unscoped gate table reports the
        # pickup stage's deliberate closure as "gripper_closed" and reads as a
        # reach failure. Measured: the pick_up phase's table showed
        # gripper_closed 0.38 and already_grasping 0.15 that way, both of them
        # correct behaviour of a later stage.
        approach = (self.step_stage == STAGE_MOVE_TO) | (
            self.step_stage == STAGE_ALIGN
        )
        live = self.active & approach
        target = self.object_xyz[:, :, 0, :]
        xy_distance = np.linalg.norm(
            target[..., :2] - self.ee_xyz[..., :2], axis=-1
        )
        above_grasp = self.ee_xyz[..., 2] - (target[..., 2] + offset)

        slack = np.array(
            [
                max_grasp_xy_offset(
                    str(name),
                    margin=float(readiness.get("grasp_xy_margin", 0.003)),
                )
                for name in self.target_catalog
            ],
            dtype=np.float32,
        )
        if "grasp_xy_slack_m" in settings:
            slack = np.asarray(settings["grasp_xy_slack_m"], dtype=np.float32)
        centred_ok = xy_distance <= slack[None, :]
        opening_ok = self.gripper_opening >= min_open
        height_ok = (
            (above_grasp >= min_above)
            & (above_grasp <= max_above)
            & (self.ee_xyz[..., 2] >= min_z)
            & (self.ee_xyz[..., 2] <= max_z)
        )
        grasp_ok = ~self.physical_grasp
        fired = self.reach_success & live
        # Honour the gate that ACTUALLY ran. With the centring bridge enabled
        # the lateral bound does not gate the reach -- the tail exists to close
        # that error -- so ANDing it here would report a promotion rate lower
        # than the one the machine applied, which is the diagnostic telling a
        # different story from the run.
        # Early staged recordings stored this flag at the config root while
        # the diagnostic looked only inside ``readiness`` and therefore
        # silently reported the default ``True``.  The state machine used the
        # real value, so this is a reporting compatibility fallback, not a
        # change to the gate that ran.
        require_centred = bool(
            readiness.get(
                "require_centred_at_reach",
                settings.get("require_centred_at_reach", True),
            )
        )
        ready = fired & opening_ok & height_ok & grasp_ok
        if require_centred:
            ready = ready & centred_ok

        def percentiles(values: Any, mask: Any) -> dict[str, float] | None:
            selected = values[mask]
            if selected.size == 0:
                return None
            return {
                "p10": round(float(np.percentile(selected, 10)), 4),
                "median": round(float(np.median(selected)), 4),
                "p90": round(float(np.percentile(selected, 90)), 4),
            }

        # Closest approach per world, over the steps it was actually stepped.
        masked = np.where(live, xy_distance, np.inf)
        closest = masked.min(axis=0)
        finite = np.isfinite(closest)

        report: dict[str, Any] = {
            "worlds": int(self.worlds),
            # The PREDICATE alone, with no readiness attached. If this is zero
            # the teacher never got within the window; if it is high and
            # `reach_event` is zero, the readiness gate is what rejected them.
            "predicate_fired_worlds": int(fired.any(axis=0).sum()),
            "ready_worlds": int(ready.any(axis=0).sum()),
            "reach_event_worlds": int((self.reach_event >= 0).sum()),
            "closest_xy_distance_m": percentiles(closest, finite),
            "final_ee_z_m": percentiles(
                self.ee_xyz[..., 2], live
            ),
            "height_above_grasp_m": percentiles(above_grasp, live),
            # The fixed-yaw bridge uses the projected, per-presentation
            # aperture stored in ``grasp_xy_slack_m``.  Reporting the old
            # rotation-invariant catalog scalar here made every potato look
            # impossible even when its sampled orientation had positive
            # clearance.  Keep the catalog grouping, but summarize the exact
            # slack values that gated these worlds.
            "grasp_xy_slack_m": {
                str(name): percentiles(
                    slack,
                    np.asarray(self.target_catalog) == name,
                )
                for name in np.unique(self.target_catalog)
            },
            "readiness_settings": {
                "require_centred_at_reach": require_centred,
                "min_gripper_opening": min_open,
                "min_height_above_grasp": min_above,
                "max_height_above_grasp": max_above,
                "min_ee_z": min_z,
                "max_ee_z": max_z,
            },
        }
        if bool(fired.any()):
            # Which conjunct rejected the steps where the predicate DID fire.
            # Shares, not counts: a world can fire on many steps.
            report["among_predicate_steps"] = {
                "steps": int(fired.sum()),
                "gripper_closed": round(
                    float((~opening_ok)[fired].mean()), 4
                ),
                "already_grasping": round(
                    float((~grasp_ok)[fired].mean()), 4
                ),
                "too_low_above_grasp": round(
                    float((above_grasp < min_above)[fired].mean()), 4
                ),
                "too_high_above_grasp": round(
                    float((above_grasp > max_above)[fired].mean()), 4
                ),
                "outside_absolute_rails": round(
                    float(
                        (
                            (self.ee_xyz[..., 2] < min_z)
                            | (self.ee_xyz[..., 2] > max_z)
                        )[fired].mean()
                    ),
                    4,
                ),
                # Laterally too far to bracket the object, even though the
                # production reach window (0.02 m) says the reach succeeded.
                "not_centred_for_grasp": round(
                    float((~centred_ok)[fired].mean()), 4
                ),
                "xy_error_m": percentiles(xy_distance, fired),
                "height_above_grasp_m": percentiles(above_grasp, fired),
                "ee_z_m": percentiles(self.ee_xyz[..., 2], fired),
            }
        return report

    def align_diagnostics(self) -> dict[str, Any]:
        """Why the alignment tail runs out of budget.

        The tail has four jobs -- climb, rotate, centre, descend -- and each
        one gates the next, so "align_budget_exhausted" is four different
        failures wearing one name. This splits them, and additionally counts
        DESCENT ABORTS in legacy recordings and RECENTRING PAUSES in the
        current controller. The old descend gate's else branch climbed back to
        rotation clearance; the current one holds Z while its lateral servo
        restores centring.
        """

        import json

        import numpy as np

        settings = json.loads(self.config_json)
        calibration = json.loads(self.calibration_json)
        offset = float(settings.get("pick_grasp_height_offset", 0.0075))
        deadband = float(settings.get("align_xy_deadband", 0.005))
        safe_z = float(calibration.get("safe_rotation_z", 0.26))
        tolerance = float(settings.get("pickup_height_tolerance", 0.003))
        target_yaw = float(calibration.get("target_yaw", 0.0))
        yaw_tolerance = float(calibration.get("tolerance_rad", 0.0873))

        live = self.active & (self.step_stage == STAGE_ALIGN)
        entered = live.any(axis=0)
        if not bool(entered.any()):
            return {"entered_align": 0}

        target = self.object_xyz[:, :, 0, :]
        grasp_z = target[..., 2] + offset
        above = self.ee_xyz[..., 2] - grasp_z
        lateral = np.linalg.norm(
            self.ee_xyz[..., :2] - target[..., :2], axis=-1
        )
        yaw_error = np.abs(
            np.angle(np.exp(1j * (self.ee_yaw - target_yaw)))
        )
        at_clearance = self.ee_xyz[..., 2] >= (safe_z - tolerance)
        yaw_ok = yaw_error <= yaw_tolerance
        centred = lateral <= deadband
        descending = yaw_ok & centred

        # A climb commanded after a descent, inside the tail: the abort.
        climbed_back = np.zeros_like(live)
        climbed_back[1:] = (
            live[1:]
            & (self.actions[1:, :, 2] > 0.5)
            & descending[:-1]
            & live[:-1]
        )
        recentering_pause = (
            live
            & ~at_clearance
            & yaw_ok
            & (lateral > float(settings.get("align_xy_abort", 0.009)))
            & (np.abs(self.actions[..., 2]) <= 0.05)
        )

        # The stage machine promotes only at decision boundaries and requires
        # the full conjunction to remain true for consecutive boundaries.
        # Action-step medians can make every individual gate look healthy even
        # when they never overlap, which is exactly what the gain-0.20 screen
        # exposed (one promotion from 35 alignment entries).
        per = max(int(self.actions_per_decision), 1)
        boundary = np.zeros_like(live)
        boundary[per - 1 :: per] = live[per - 1 :: per]
        readiness = dict(settings.get("readiness") or {})
        target_height = float(settings.get("pickup_height_above_grasp", 0.01))
        if bool(settings.get("align_handoff_at_clearance", False)):
            handoff_height_ok = np.ones_like(live)
        else:
            handoff_height_ok = np.abs(above - target_height) <= tolerance
        broad_height_ok = (
            (above >= float(readiness.get("min_height_above_grasp", -0.005)))
            & (above <= float(readiness.get("max_height_above_grasp", 0.12)))
            & (self.ee_xyz[..., 2] >= float(readiness.get("min_ee_z", 0.18)))
            & (self.ee_xyz[..., 2] <= float(readiness.get("max_ee_z", 0.40)))
        )
        slack = np.asarray(
            settings.get(
                "grasp_xy_slack_m",
                [
                    max_grasp_xy_offset(
                        str(name),
                        margin=float(readiness.get("grasp_xy_margin", 0.003)),
                    )
                    for name in self.target_catalog
                ],
            ),
            dtype=np.float32,
        )
        grasp_centred = lateral <= slack[None, :]
        opening_ok = self.gripper_opening >= float(
            readiness.get("min_gripper_opening", 0.90)
        )
        boundary_ready = (
            boundary
            & yaw_ok
            & handoff_height_ok
            & broad_height_ok
            & grasp_centred
            & opening_ok
            & ~self.physical_grasp
        )
        ready_by_decision = boundary_ready[per - 1 :: per]
        live_by_decision = boundary[per - 1 :: per]
        max_streak = np.zeros((self.worlds,), dtype=np.int64)
        running = np.zeros((self.worlds,), dtype=np.int64)
        for ready_row, live_row in zip(ready_by_decision, live_by_decision):
            running = np.where(live_row & ready_row, running + 1, 0)
            max_streak = np.maximum(max_streak, running)

        boundary_count = int(boundary.sum())

        def boundary_failure_share(ok: Any) -> float | None:
            if boundary_count == 0:
                return None
            return round(float((~np.asarray(ok, dtype=bool))[boundary].mean()), 4)

        def percentiles(values: Any) -> dict[str, float] | None:
            selected = values[live]
            if selected.size == 0:
                return None
            return {
                "p10": round(float(np.percentile(selected, 10)), 4),
                "median": round(float(np.median(selected)), 4),
                "p90": round(float(np.percentile(selected, 90)), 4),
            }

        return {
            "entered_align": int(entered.sum()),
            "promoted": int((np.asarray(self.align_event) >= 0).sum()),
            "decisions_in_align": int(
                live.sum() / max(int(self.actions_per_decision), 1)
            ),
            # Share of tail steps failing each of the four jobs in turn.
            "share_below_clearance": round(
                float((~at_clearance)[live].mean()), 4
            ),
            "share_yaw_unaligned": round(float((~yaw_ok)[live].mean()), 4),
            "share_off_centre": round(float((~centred)[live].mean()), 4),
            "share_descending": round(float(descending[live].mean()), 4),
            # Worlds that ever got both gates open at once, i.e. ever started
            # the descent at all. A tail that never reaches this is failing at
            # rotation or centring; one that reaches it and still exhausts its
            # budget is chattering.
            "ever_started_descent": int((descending & live).any(axis=0).sum()),
            "descent_aborts": int(climbed_back.sum()),
            "worlds_with_a_descent_abort": int(
                climbed_back.any(axis=0).sum()
            ),
            "descent_recentering_pauses": int(recentering_pause.sum()),
            "worlds_with_a_recentering_pause": int(
                recentering_pause.any(axis=0).sum()
            ),
            "boundary_gate_diagnostics": {
                "decision_boundaries": boundary_count,
                "required_consecutive": int(
                    calibration.get("consecutive_decisions", 2)
                ),
                "worlds_ever_all_ready": int(
                    boundary_ready.any(axis=0).sum()
                ),
                "max_ready_streak": {
                    "p10": round(float(np.percentile(max_streak[entered], 10)), 2),
                    "median": round(float(np.median(max_streak[entered])), 2),
                    "p90": round(float(np.percentile(max_streak[entered], 90)), 2),
                },
                "share_yaw_not_ready": boundary_failure_share(yaw_ok),
                "share_handoff_height_not_ready": boundary_failure_share(
                    handoff_height_ok
                ),
                "share_broad_height_not_ready": boundary_failure_share(
                    broad_height_ok
                ),
                "share_not_centred_for_grasp": boundary_failure_share(
                    grasp_centred
                ),
                "share_gripper_not_open": boundary_failure_share(opening_ok),
                "share_already_grasping": boundary_failure_share(
                    ~self.physical_grasp
                ),
            },
            "yaw_error_rad": percentiles(yaw_error),
            "xy_error_m": percentiles(lateral),
            "height_above_grasp_m": percentiles(above),
        }

    def pickup_diagnostics(self) -> dict[str, Any]:
        """Where the grasp stage loses its chains: descend, close, or lift.

        The campaign has already been wrong about this once at a whole-phase
        scale: composed ``put_into`` was assumed to fail at placement and
        measured to fail at the grasp. Within the grasp there is a second
        split, and it is the one the retained pick_up measurements point at --
        grasps land at 28-41% while lifts are 3-12% OF THOSE, with the post-
        grasp rise sitting at 7-19 mm against a 50 mm success height.

        So this reports the ladder and the two distances underneath it, rather
        than one conversion rate. ``max_lift_when_grasped`` is the number that
        discriminates "it never got hold of the object" from "it held it and
        did not raise it".
        """

        import json

        import numpy as np

        settings = json.loads(self.config_json)
        offset = float(settings.get("pick_grasp_height_offset", 0.0075))
        live = self.active & (self.step_stage == STAGE_PICK_UP)
        entered = np.asarray(self.align_event) >= 0

        target = self.object_xyz[:, :, 0, :]
        grasp_point = target.copy()
        grasp_point[..., 2] += offset
        above = self.ee_xyz[..., 2] - grasp_point[..., 2]
        distance = np.linalg.norm(self.ee_xyz - grasp_point, axis=-1)
        # The lateral component, reported rather than left to be recovered from
        # sqrt(3d^2 - height^2) by whoever reads the table.
        lateral = np.linalg.norm(
            self.ee_xyz[..., :2] - grasp_point[..., :2], axis=-1
        )

        def per_world(values: Any, reduce: Any) -> Any:
            masked = np.where(live, values, reduce.identity)
            return reduce.fn(masked, axis=0)

        class _Min:
            identity = np.inf
            fn = staticmethod(np.min)

        class _Max:
            identity = -np.inf
            fn = staticmethod(np.max)

        closest_above = per_world(above, _Min)
        closest_distance = per_world(distance, _Min)
        peak_lift = per_world(self.target_lift, _Max)
        grasped = (self.physical_grasp & live).any(axis=0)
        lifted = (self.pickup_success & live).any(axis=0)
        handed_off = np.asarray(self.pickup_event) >= 0
        # The handoff is the LAST alignment observation, before the first
        # pickup action. Reading the first pickup row instead confounds the
        # initial pose with the teacher's first movement.
        handoff_steps = np.clip(
            (np.asarray(self.align_event) + 1) * int(self.actions_per_decision) - 1,
            0, above.shape[0] - 1,
        )
        world_indices = np.arange(above.shape[1])

        def percentiles(values: Any, mask: Any) -> dict[str, float] | None:
            selected = values[mask & np.isfinite(values)]
            if selected.size == 0:
                return None
            return {
                "p10": round(float(np.percentile(selected, 10)), 4),
                "median": round(float(np.median(selected)), 4),
                "p90": round(float(np.percentile(selected, 90)), 4),
            }

        return {
            "entered_pickup": int(entered.sum()),
            "grasped": int((grasped & entered).sum()),
            "lifted": int((lifted & entered).sum()),
            "handed_off": int(handed_off.sum()),
            "grasp_given_entered": _ratio(
                int((grasped & entered).sum()), int(entered.sum())
            ),
            "lift_given_grasp": _ratio(
                int((lifted & grasped).sum()), int(grasped.sum())
            ),
            # How far down it got. The pickup teacher's own aligned start is
            # 0.01 m above the grasp point; anything much larger means the
            # descend never happened and the close is irrelevant.
            "closest_height_above_grasp_m": percentiles(closest_above, entered),
            "closest_3d_distance_to_grasp_m": percentiles(
                closest_distance, entered
            ),
            # THE number this stage turns on. The open gripper's lateral slack
            # is 0.0130 m for an apple and 0.0185 m for the others; an XY error
            # above that means the fingers land on the object instead of
            # bracketing it, and the descent stops.
            "closest_xy_distance_m": percentiles(
                per_world(lateral, _Min), entered
            ),
            "handoff_xy_error_m": percentiles(
                lateral[handoff_steps, world_indices], entered
            ),
            "handoff_height_above_grasp_m": percentiles(
                above[handoff_steps, world_indices], entered
            ),
            # THE discriminator. A grasp that never rises is a different
            # failure from a grasp that never happens, and they have opposite
            # fixes.
            "max_lift_when_grasped_m": percentiles(peak_lift, grasped & entered),
            "max_lift_all_entered_m": percentiles(peak_lift, entered),
            "regrasp_decisions": int(np.asarray(self.pickup_regrasp_total())),
            # The COMMAND, not the outcome. The retained measurement this
            # reproduces: the same adapter commands +0.40 mean a_z under a
            # put_into prompt and +0.02 under a pick_up one, which is a
            # statement about the prompt rather than about the policy's
            # ability to lift. If the lift is failing and this is near zero,
            # the pickup stage is being asked the wrong question.
            "mean_action_z_while_grasped": (
                round(
                    float(
                        self.actions[..., 2][
                            self.physical_grasp & live
                        ].mean()
                    ),
                    4,
                )
                if bool((self.physical_grasp & live).any())
                else None
            ),
            "mean_action_gripper_while_grasped": (
                round(
                    float(
                        self.actions[..., 4][
                            self.physical_grasp & live
                        ].mean()
                    ),
                    4,
                )
                if bool((self.physical_grasp & live).any())
                else None
            ),
        }

    def pickup_regrasp_total(self) -> int:
        """Decisions spent having lost a grasp inside the pickup stage."""

        import numpy as np

        live = self.active & (self.step_stage == STAGE_PICK_UP)
        held = self.physical_grasp & live
        lost = np.zeros_like(held)
        lost[1:] = held[:-1] & ~held[1:] & live[1:]
        return int(lost.sum())

    def placement_diagnostics(self) -> dict[str, Any]:
        """Carry, release, and where the object ended up."""

        import numpy as np

        live = self.active & (
            (self.step_stage == STAGE_PLACEMENT)
            | (self.step_stage == STAGE_SETTLE)
        )
        entered = np.asarray(self.pickup_event) >= 0
        target = self.object_xyz[:, :, 0, :2]
        receptacle = self.object_xyz[:, :, 1, :2]
        separation = np.linalg.norm(target - receptacle, axis=-1)
        closest = np.where(live, separation, np.inf).min(axis=0)

        released = (self.released & live).any(axis=0)
        placed = (self.placement_success & live).any(axis=0)
        wrong = (self.wrong_place_settled & live).any(axis=0)
        geometry = (self.placement_geometry_ok & live).any(axis=0)

        def percentiles(values: Any, mask: Any) -> dict[str, float] | None:
            selected = values[mask & np.isfinite(values)]
            if selected.size == 0:
                return None
            return {
                "p10": round(float(np.percentile(selected, 10)), 4),
                "median": round(float(np.median(selected)), 4),
                "p90": round(float(np.percentile(selected, 90)), 4),
            }

        return {
            "entered_placement": int(entered.sum()),
            "reached_goal_geometry": int((geometry & entered).sum()),
            "released": int((released & entered).sum()),
            "placed": int((placed & entered).sum()),
            "wrong_place_settled": int((wrong & entered).sum()),
            "release_given_geometry": _ratio(
                int((released & geometry).sum()), int(geometry.sum())
            ),
            # How close the CARRY got, independent of the release. A carry that
            # never reaches the receptacle and a release that never happens are
            # different problems.
            "closest_target_receptacle_xy_m": percentiles(closest, entered),
        }

    # -- reporting ------------------------------------------------------

    def summary(
        self,
        *,
        min_approach_xy: float = 0.06,
        min_handoff_lift: float = 0.05,
    ) -> dict[str, Any]:
        import numpy as np

        accepted, _, reasons = self.acceptance(
            min_approach_xy=min_approach_xy,
            min_handoff_lift=min_handoff_lift,
        )
        native = self.placement_success.any(axis=0)
        by_destination: dict[str, Any] = {}
        for name in np.unique(self.destination):
            mask = self.destination == name
            by_destination[str(name)] = {
                "chains": int(mask.sum()),
                "native_placement": int(native[mask].sum()),
                "accepted": int(accepted[mask].sum()),
            }
        by_object: dict[str, Any] = {}
        for name in np.unique(self.target_catalog):
            mask = self.target_catalog == name
            by_object[str(name)] = {
                "chains": int(mask.sum()),
                "accepted": int(accepted[mask].sum()),
            }
        completion = self.active.sum(axis=0)
        return {
            "round_index": int(self.round_index),
            "worlds": int(self.worlds),
            "role_switches": int(self.role_switches),
            "reached": int((self.reach_event >= 0).sum()),
            "aligned": int((self.align_event >= 0).sum()),
            "picked_up": int((self.pickup_event >= 0).sum()),
            "native_placement_success": int(native.sum()),
            "accepted_chains": int(accepted.sum()),
            "rejection_reasons": reasons,
            "failure_counts": {
                FAILURE_NAMES[index]: int((self.failure_code == index).sum())
                for index in range(1, len(FAILURE_NAMES))
                if int((self.failure_code == index).sum()) > 0
            },
            "by_destination": by_destination,
            "by_object": by_object,
            "accepted_action_steps_mean": (
                round(float(completion[accepted].mean()), 2)
                if bool(accepted.any())
                else None
            ),
            "diverged_worlds": int(
                np.asarray(self.diverged_world_mask, dtype=bool).sum()
            ),
            # Always present, not only on a failure: a screen that reports
            # "reached 41" without saying how close the other 23 got cannot be
            # used to decide whether to widen a gate or change a teacher.
            "reach_diagnostics": self.reach_diagnostics(),
            "align_diagnostics": self.align_diagnostics(),
            "pickup_diagnostics": self.pickup_diagnostics(),
            "placement_diagnostics": self.placement_diagnostics(),
        }

    # -- serialization --------------------------------------------------

    def to_npz(self, path: Path) -> None:
        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        for name in (*self._PER_STEP, *self._PER_DECISION, *self._PER_WORLD):
            payload[name] = getattr(self, name)
        for name, dtype in self._SCALARS:
            payload[name] = np.asarray(getattr(self, name), dtype=dtype)
        for name in self._STRINGS:
            payload[name] = np.asarray(getattr(self, name), dtype="U65535")
        np.savez_compressed(path, **payload)

    @classmethod
    def from_npz(cls, path: Path) -> "StagedRound":
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            fields = {key: data[key] for key in data.files}
        expected = (
            *cls._PER_STEP,
            *cls._PER_DECISION,
            *cls._PER_WORLD,
            *(name for name, _ in cls._SCALARS),
            *cls._STRINGS,
        )
        missing = [name for name in expected if name not in fields]
        if missing:
            raise ValueError(
                f"{path} is missing {missing}; it is not a staged round."
            )
        kwargs: dict[str, Any] = {
            name: fields[name]
            for name in (*cls._PER_STEP, *cls._PER_DECISION, *cls._PER_WORLD)
        }
        for name, _ in cls._SCALARS:
            kwargs[name] = int(fields[name])
        for name in cls._STRINGS:
            kwargs[name] = str(fields[name])
        return cls(**kwargs)


def write_staged_frames(
    path: Path,
    *,
    buffers: StagedRolloutBuffers,
    episode_uids: Sequence[str],
    world_index: Sequence[int],
    keep: Any,
) -> dict[str, Any]:
    """Write the pictures for the KEPT worlds, keyed by explicit episode id.

    Two joins are written on purpose. ``world_index`` reproduces the historical
    ``frames_<stem>.npz`` layout that ``sil_sft.resolve_frame_rows`` parses out
    of a uid, so existing tooling keeps working. ``episode_uid`` is the explicit
    key the new schema uses, so a row finds its picture by identity rather than
    by a string convention that has already silently matched zero rows once.
    """

    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if buffers.overview is None:
        raise ValueError(
            "This round recorded no frames. Every accepted episode needs "
            "pictures; a bank without them cannot be refreshed onto a new "
            "student and cannot train the vision path."
        )
    selected = np.flatnonzero(np.asarray(keep, dtype=bool))
    payload = {
        "overview": buffers.overview[:, selected],
        "wrist": buffers.wrist[:, selected],
        "terminal_overview": buffers.terminal_overview[selected],
        "terminal_wrist": buffers.terminal_wrist[selected],
        "world_index": np.asarray(
            [int(world_index[i]) for i in selected], dtype=np.int64
        ),
        "episode_uid": np.asarray(
            [str(episode_uids[i]) for i in selected], dtype="U128"
        ),
        "decisions": np.asarray(buffers.decisions, dtype=np.int64),
    }
    np.savez_compressed(path, **payload)
    return {
        "path": str(path),
        "worlds": int(selected.size),
        "decisions": int(buffers.decisions),
        "bytes": int(path.stat().st_size),
    }
