#!/usr/bin/env python3
"""Solve the fixed pickup yaw from the model's camera extrinsics, and say how
well it travels.

The user describes the successful pickup pose as "the wrist camera facing the
overview camera". That is a statement about two camera frames, and the model
knows both of them, so this is a solve and not a guess. A hard-coded ``0`` or
``pi`` would be a claim about a mount transform (``camera_body`` sits at
``(0, 0.05, 0.045)`` under a ball-jointed stabilizer, and the camera itself
carries ``euler="-15 0 0"``) that the MJCF is right here to answer.

What is being solved
--------------------

The wrist camera looks along its own -Z. Its horizontal bearing is the XY
projection of that axis. "Facing the overview camera" is that bearing pointing
at the overview camera's world position, from the wrist camera's own world
position -- which itself moves with the yaw, because the camera hangs off the
axis rather than sitting on it. So the equation is implicit in ``q`` and is
solved numerically over the joint's range.

What a FIXED yaw cannot be
--------------------------

Yaw alone cannot point the camera's full optical axis at an elevated point:
the camera is tilted 15 degrees down and that tilt is not a free variable. Only
the horizontal bearing is aligned, which is what "facing" means here.

And a single fixed angle is exact at ONE place. The overview camera sits 0.54 m
in front of the desk, so the bearing toward it swings as the gripper moves
around the workspace. This tool therefore reports not only the calibrated angle
but the residual that angle leaves at a grid of workspace positions -- the
number that says how much of a per-position facing rule is being given up. A
per-position mode would be a different rule and is deliberately not what the
user selected.

Usage (no GPU, no checkpoint -- this is kinematics)::

    MUJOCO_GL=disable python tools/audit/calibrate_cdpr_pickup_yaw.py \\
        --xml robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml \\
        --output runs/three_stage/yaw_calibration.json

Feed the written JSON to the recorder with ``--yaw-calibration``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# The horizontal component of the wrist camera's optical axis goes to zero if
# the camera is ever pointed straight down; below this the bearing is numerical
# noise and no yaw "faces" anything.
_MIN_HORIZONTAL_AXIS = 1.0e-3
# Likewise for the bearing TOWARD the overview camera: if the wrist camera were
# directly beneath it, every yaw would be equally right and equally wrong.
_MIN_HORIZONTAL_SEPARATION = 1.0e-3


def _wrap(angle: float) -> float:
    return float(angle - 2.0 * math.pi * round(angle / (2.0 * math.pi)))


class YawGeometry:
    """Camera poses as a function of the yaw joint, from the loaded model.

    Kinematics only: the EE free joint and the yaw hinge are written into
    ``qpos`` and ``mj_forward`` is called. No actuators, no cables, no settling
    -- the question is where the camera frames are for a given joint value, and
    that is a pure function of the model tree.
    """

    def __init__(self, xml: Path) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.xml = Path(xml)

        def named(objtype: Any, name: str) -> int:
            index = mujoco.mj_name2id(self.model, objtype, name)
            if index < 0:
                raise SystemExit(f"{xml} has no {name!r}.")
            return int(index)

        self.overview_camera = named(mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        self.wrist_camera = named(mujoco.mjtObj.mjOBJ_CAMERA, "ee_camera")
        self.ee_body = named(mujoco.mjtObj.mjOBJ_BODY, "ee_base")
        self.yaw_joint = named(mujoco.mjtObj.mjOBJ_JOINT, "ee_yaw")
        self.free_joint = named(mujoco.mjtObj.mjOBJ_JOINT, "ee_free")
        self.yaw_qadr = int(self.model.jnt_qposadr[self.yaw_joint])
        self.free_qadr = int(self.model.jnt_qposadr[self.free_joint])
        low, high = (float(v) for v in self.model.jnt_range[self.yaw_joint])
        self.yaw_limits = (min(low, high), max(low, high))

    def pose(self, position: Sequence[float], yaw: float) -> dict[str, Any]:
        mujoco = self.mujoco
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.free_qadr : self.free_qadr + 3] = np.asarray(
            position, dtype=float
        )
        self.data.qpos[self.yaw_qadr] = float(
            min(max(yaw, self.yaw_limits[0]), self.yaw_limits[1])
        )
        mujoco.mj_forward(self.model, self.data)
        wrist_position = np.array(self.data.cam_xpos[self.wrist_camera])
        wrist_matrix = np.array(self.data.cam_xmat[self.wrist_camera]).reshape(3, 3)
        overview_position = np.array(self.data.cam_xpos[self.overview_camera])
        # A MuJoCo camera looks down its own -Z.
        forward = -wrist_matrix[:, 2]
        return {
            "yaw": float(self.data.qpos[self.yaw_qadr]),
            "ee_position": np.array(self.data.xpos[self.ee_body]).tolist(),
            "wrist_camera_position": wrist_position.tolist(),
            "wrist_camera_forward": forward.tolist(),
            "overview_camera_position": overview_position.tolist(),
        }

    def bearing_error(self, position: Sequence[float], yaw: float) -> float:
        """Signed angle from where the wrist looks to where the overview is."""

        pose = self.pose(position, yaw)
        forward = np.asarray(pose["wrist_camera_forward"])
        if float(np.linalg.norm(forward[:2])) < _MIN_HORIZONTAL_AXIS:
            raise SystemExit(
                "The wrist camera's optical axis is vertical at this pose, so "
                "it has no horizontal bearing and no yaw can face anything. "
                "Check the camera's mounting rotation before calibrating."
            )
        separation = np.asarray(pose["overview_camera_position"]) - np.asarray(
            pose["wrist_camera_position"]
        )
        if float(np.linalg.norm(separation[:2])) < _MIN_HORIZONTAL_SEPARATION:
            raise SystemExit(
                "The wrist camera sits directly beneath the overview camera at "
                "this pose; the bearing toward it is degenerate. Choose a "
                "different calibration position."
            )
        looking = math.atan2(float(forward[1]), float(forward[0]))
        toward = math.atan2(float(separation[1]), float(separation[0]))
        return _wrap(toward - looking)

    def solve(
        self, position: Sequence[float], *, samples: int = 721, refine: int = 40
    ) -> tuple[float, float]:
        """The yaw whose bearing error is smallest, and that residual.

        A coarse sweep then a bisection on the sign change, rather than a
        closed form: the camera's offset from the yaw axis makes the equation
        implicit, and a sweep over a bounded joint is cheap and cannot land in
        the wrong basin.
        """

        low, high = self.yaw_limits
        grid = np.linspace(low, high, int(samples))
        errors = np.array(
            [self.bearing_error(position, float(value)) for value in grid]
        )
        best = int(np.argmin(np.abs(errors)))
        left = max(best - 1, 0)
        right = min(best + 1, len(grid) - 1)
        a, b = float(grid[left]), float(grid[right])
        fa, fb = float(errors[left]), float(errors[right])
        if fa * fb <= 0.0 and fa != fb:
            for _ in range(int(refine)):
                middle = 0.5 * (a + b)
                fm = self.bearing_error(position, middle)
                if fa * fm <= 0.0:
                    b, fb = middle, fm
                else:
                    a, fa = middle, fm
            answer = 0.5 * (a + b)
        else:
            # No sign change inside the bracket: the minimum is at a joint
            # stop, which is a real answer and is reported with its residual
            # rather than refined into a fiction.
            answer = float(grid[best])
        return float(answer), float(self.bearing_error(position, answer))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=ROOT / "robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-xy",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        help=(
            "Where the fixed angle is solved. The desk centre by default, "
            "which is the position that minimises the worst-case residual "
            "over a symmetric workspace."
        ),
    )
    parser.add_argument(
        "--calibration-z",
        type=float,
        default=0.26,
        help="The safe rotation height the alignment tail rotates at.",
    )
    parser.add_argument(
        "--tolerance-degrees",
        type=float,
        default=5.0,
        help=(
            "Acceptance band for the alignment tail. Held for "
            "--consecutive-decisions decisions, so it rejects a wrist that "
            "merely swings through the target."
        ),
    )
    parser.add_argument("--consecutive-decisions", type=int, default=2)
    parser.add_argument(
        "--workspace-half-extent",
        type=float,
        default=0.19,
        help="Half-width of the grid the fixed angle's residual is reported on.",
    )
    parser.add_argument("--workspace-samples", type=int, default=7)
    args = parser.parse_args(argv)

    geometry = YawGeometry(args.xml.expanduser().resolve())
    position = (
        float(args.calibration_xy[0]),
        float(args.calibration_xy[1]),
        float(args.calibration_z),
    )
    target_yaw, residual = geometry.solve(position)
    pose = geometry.pose(position, target_yaw)

    # How far the ONE angle is from facing, elsewhere. This is the cost of the
    # user's fixed-yaw choice, stated as a number instead of a caveat.
    grid = np.linspace(
        -float(args.workspace_half_extent),
        float(args.workspace_half_extent),
        int(args.workspace_samples),
    )
    residuals: list[dict[str, Any]] = []
    for x in grid:
        for y in grid:
            error = geometry.bearing_error(
                (float(x), float(y), float(args.calibration_z)), target_yaw
            )
            residuals.append(
                {"x": float(x), "y": float(y), "error_deg": math.degrees(error)}
            )
    magnitudes = np.array([abs(row["error_deg"]) for row in residuals])

    report = {
        "xml": str(args.xml),
        "mode": "fixed_world_yaw",
        "target_yaw": target_yaw,
        "target_yaw_degrees": math.degrees(target_yaw),
        "residual_at_calibration_deg": math.degrees(residual),
        "calibration_pose": list(position),
        "yaw_joint_limits": list(geometry.yaw_limits),
        "tolerance_rad": math.radians(float(args.tolerance_degrees)),
        "consecutive_decisions": int(args.consecutive_decisions),
        "safe_rotation_z": float(args.calibration_z),
        "source": (
            f"calibrate_cdpr_pickup_yaw.py fixed_world_yaw from "
            f"{Path(args.xml).name}"
        ),
        "camera_geometry": pose,
        "workspace_residual_deg": {
            "samples": int(magnitudes.size),
            "mean_abs": round(float(magnitudes.mean()), 3),
            "median_abs": round(float(np.median(magnitudes)), 3),
            "p90_abs": round(float(np.percentile(magnitudes, 90)), 3),
            "max_abs": round(float(magnitudes.max()), 3),
            "within_tolerance_fraction": round(
                float(
                    (magnitudes <= float(args.tolerance_degrees)).mean()
                ),
                4,
            ),
        },
        "notes": [
            "Fixed mode. The angle is exact at the calibration pose only; "
            "workspace_residual_deg is how far it is from facing elsewhere.",
            "Yaw aligns the horizontal bearing only. The wrist camera is "
            "tilted 15 degrees down by its mounting and yaw cannot change "
            "that, so this is not a full optical-axis alignment.",
            "This is kinematics from the MJCF. It does not say the reach "
            "teacher can be brought here, or how long the tail takes; the "
            "recorder measures that and reports align_given_reach.",
        ],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[yaw] target {target_yaw:+.6f} rad "
        f"({math.degrees(target_yaw):+.3f} deg), residual at the calibration "
        f"pose {math.degrees(residual):+.4f} deg",
        flush=True,
    )
    print(
        "[yaw] fixed-angle residual over the workspace: mean "
        f"{report['workspace_residual_deg']['mean_abs']:.2f} deg, p90 "
        f"{report['workspace_residual_deg']['p90_abs']:.2f}, max "
        f"{report['workspace_residual_deg']['max_abs']:.2f}; "
        f"{report['workspace_residual_deg']['within_tolerance_fraction']:.1%} "
        f"of the grid is inside the {args.tolerance_degrees:.1f} deg band",
        flush=True,
    )
    print(f"[yaw] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
