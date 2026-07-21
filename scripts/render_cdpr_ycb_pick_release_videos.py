#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import mujoco as mj
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_cdpr_ycb_caught_object_videos import (  # noqa: E402
    DEFAULT_CDPR_XML,
    DEFAULT_OBJECTS,
    DEFAULT_YCB_ROOT,
    HELD_OFFSET,
    PLATE_CENTER,
    _annotate,
    _body_geoms,
    _build_wrapper_xml,
    _finger_geometry,
    _force_gripper_opening,
    _geom_interval_along_axis,
    _import_headless_simulation,
    _measure_fit,
    _object_position,
    _set_body_pose,
    _set_ee_position,
    _write_video,
)


DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_ycb_pick_release_videos"
TABLE_TOP_Z = 0.0
PLATE_TOP_Z = float(PLATE_CENTER[2] + 0.006)
OBJECT_START_XY = np.array([-0.14, -0.03], dtype=float)


def _step_physics(sim, *, steps: int = 1) -> None:
    for _ in range(max(1, int(steps))):
        mj.mj_step(sim.model, sim.data)


def _measure_physical_fit(sim, object_body: str, *, clearance: float, compression: float) -> dict[str, object]:
    base = _measure_fit(sim, object_body, clearance=0.0)
    width = float(base["object_projected_width_m"])
    closed_gap = float(base["closed_inner_gap_m"])
    joint_span = max(float(base["joint_span_m"]), 1e-6)
    desired_gap = max(0.0, width + 2.0 * max(0.0, clearance) - 2.0 * max(0.0, compression))
    raw_opening = (desired_gap - closed_gap) / (2.0 * joint_span)
    opening = float(np.clip(raw_opening, 0.0, 1.0))

    _force_gripper_opening(sim, opening)
    held = _finger_geometry(sim)
    base.update(
        {
            "clearance_m": float(clearance),
            "grip_compression_m": float(compression),
            "desired_inner_gap_m": float(desired_gap),
            "held_inner_gap_m": float(held["inner_gap"]),
            "raw_gripper_opening_01": float(raw_opening),
            "held_gripper_opening_01": float(opening),
            "target_finger_joint_qpos_m": float(sim.gripper_joint_min + opening * joint_span),
            "fits_without_clipping": bool(0.0 <= raw_opening <= 1.0),
            "finger_axis": [float(x) for x in np.asarray(held["axis"], dtype=float)],
            "left_tip_center_m": [float(x) for x in np.asarray(held["left_tip_center"], dtype=float)],
            "right_tip_center_m": [float(x) for x in np.asarray(held["right_tip_center"], dtype=float)],
            "hold_center_m": [float(x) for x in np.asarray(held["center"], dtype=float)],
        }
    )
    return base


def _body_interval_along_axis(sim, body_name: str, axis: np.ndarray) -> tuple[float, float]:
    lo = float("inf")
    hi = float("-inf")
    for gid in _body_geoms(sim, body_name):
        geom_lo, geom_hi = _geom_interval_along_axis(sim, gid, axis)
        lo = min(lo, float(geom_lo))
        hi = max(hi, float(geom_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise RuntimeError(f"Could not measure body interval for {body_name!r}.")
    return lo, hi


def _place_body_bottom_on_z(sim, body_name: str, xy: np.ndarray, surface_z: float, *, clearance: float = 0.001) -> None:
    pos = _object_position(sim, body_name)
    lo_z, _ = _body_interval_along_axis(sim, body_name, np.array([0.0, 0.0, 1.0], dtype=float))
    bottom_offset = float(lo_z - pos[2])
    target = np.array([float(xy[0]), float(xy[1]), float(surface_z) + float(clearance) - bottom_offset], dtype=float)
    _set_body_pose(sim, body_name, target)


def _object_center_for_bottom_on_z(sim, body_name: str, surface_z: float, *, clearance: float = 0.001) -> np.ndarray:
    pos = _object_position(sim, body_name)
    lo_z, _ = _body_interval_along_axis(sim, body_name, np.array([0.0, 0.0, 1.0], dtype=float))
    bottom_offset = float(lo_z - pos[2])
    out = pos.copy()
    out[2] = float(surface_z) + float(clearance) - bottom_offset
    return out


def _hold_center_relative_to_ee(sim) -> np.ndarray:
    return np.asarray(_finger_geometry(sim)["center"], dtype=float).reshape(3) - np.asarray(
        sim.get_end_effector_position(),
        dtype=float,
    ).reshape(3)


def _ee_position_for_held_object(sim, object_position: np.ndarray) -> np.ndarray:
    hold_center_rel = _hold_center_relative_to_ee(sim)
    hold_center = np.asarray(object_position, dtype=float).reshape(3) - HELD_OFFSET
    return hold_center - hold_center_rel


def _held_object_position(sim) -> np.ndarray:
    return np.asarray(_finger_geometry(sim)["center"], dtype=float).reshape(3) + HELD_OFFSET


def _setup_camera(sim) -> None:
    sim.overview_cam.lookat[:] = np.array([0.02, 0.0, 0.22], dtype=float)
    sim.overview_cam.distance = 1.05
    sim.overview_cam.azimuth = 90
    sim.overview_cam.elevation = -24


def _capture(
    sim,
    frames: list[np.ndarray],
    *,
    object_name: str,
    object_body: str,
    label: str,
    fit: dict[str, object],
    attached: bool,
) -> None:
    obj = _object_position(sim, object_body)
    ee = np.asarray(sim.get_end_effector_position(), dtype=float)
    frame = sim.capture_frame(sim.overview_cam, "overview")
    frames.append(
        _annotate(
            frame,
            [
                f"{object_name}: pick / lift / release",
                label,
                f"width={float(fit['object_projected_width_m']):.3f}m gap={float(fit['held_inner_gap_m']):.3f}m open={sim.get_gripper_opening():.2f}",
                f"ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:.2f}) obj=({obj[0]:+.2f},{obj[1]:+.2f},{obj[2]:.2f})",
            ],
        )
    )


def _move_ee(
    sim,
    frames: list[np.ndarray],
    *,
    object_name: str,
    object_body: str,
    label: str,
    fit: dict[str, object],
    target: np.ndarray,
    steps: int,
    attached: bool,
    gripper_opening: float,
) -> None:
    start = np.asarray(sim.get_end_effector_position(), dtype=float)
    target = np.asarray(target, dtype=float).reshape(3)
    for idx in range(max(1, int(steps))):
        alpha = (idx + 1) / float(max(1, int(steps)))
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        _set_ee_position(sim, (1.0 - smooth) * start + smooth * target)
        _force_gripper_opening(sim, gripper_opening)
        _step_physics(sim, steps=2)
        _capture(
            sim,
            frames,
            object_name=object_name,
            object_body=object_body,
            label=label,
            fit=fit,
            attached=attached,
        )


def _render_pick_lift_move(
    *,
    wrapper_xml: Path,
    object_name: str,
    run_dir: Path,
    fps: float,
    clearance: float,
    compression: float,
    keep_frames: bool,
) -> dict[str, object]:
    HeadlessCDPRSimulation = _import_headless_simulation()
    sim = HeadlessCDPRSimulation(str(wrapper_xml), output_dir=str(run_dir), record_trajectory=False)
    frames: list[np.ndarray] = []
    try:
        sim.initialize()
        _setup_camera(sim)
        object_body = str(sim.get_object_body_name())
        _place_body_bottom_on_z(sim, object_body, OBJECT_START_XY, TABLE_TOP_Z)
        fit = _measure_physical_fit(sim, object_body, clearance=clearance, compression=compression)

        object_pos = _object_position(sim, object_body)
        grasp_ee = _ee_position_for_held_object(sim, object_pos)
        hover = grasp_ee + np.array([0.0, 0.0, 0.25], dtype=float)
        lift = grasp_ee + np.array([0.0, 0.0, 0.30], dtype=float)
        carry = lift + np.array([0.28, 0.12, 0.0], dtype=float)

        _force_gripper_opening(sim, 1.0)
        _set_ee_position(sim, hover)
        for _ in range(12):
            _step_physics(sim, steps=2)
            _capture(
                sim,
                frames,
                object_name=object_name,
                object_body=object_body,
                label="open gripper above object",
                fit=fit,
                attached=False,
            )

        _move_ee(
            sim,
            frames,
            object_name=object_name,
            object_body=object_body,
            label="descend to grasp",
            fit=fit,
            target=grasp_ee,
            steps=32,
            attached=False,
            gripper_opening=1.0,
        )

        for idx in range(18):
            alpha = (idx + 1) / 18.0
            opening = (1.0 - alpha) * 1.0 + alpha * float(fit["held_gripper_opening_01"])
            _force_gripper_opening(sim, opening)
            _step_physics(sim, steps=4)
            _capture(
                sim,
                frames,
                object_name=object_name,
                object_body=object_body,
                label="closing fingers around object",
                fit=fit,
                attached=False,
            )

        for _ in range(10):
            _force_gripper_opening(sim, float(fit["held_gripper_opening_01"]))
            _step_physics(sim, steps=4)
            _capture(
                sim,
                frames,
                object_name=object_name,
                object_body=object_body,
                label="object captured",
                fit=fit,
                attached=True,
            )

        _move_ee(
            sim,
            frames,
            object_name=object_name,
            object_body=object_body,
            label="lift object",
            fit=fit,
            target=lift,
            steps=36,
            attached=True,
            gripper_opening=float(fit["held_gripper_opening_01"]),
        )
        _move_ee(
            sim,
            frames,
            object_name=object_name,
            object_body=object_body,
            label="move while holding",
            fit=fit,
            target=carry,
            steps=44,
            attached=True,
            gripper_opening=float(fit["held_gripper_opening_01"]),
        )

        output = run_dir / f"pick_lift_move_{object_name}.mp4"
        info = _write_video(frames, output, fps=fps, keep_frames=keep_frames)
        return {
            "object": object_name,
            "object_body": object_body,
            "kind": "pick_lift_move",
            "wrapper_xml": wrapper_xml.as_posix(),
            "final_object_position_m": [float(x) for x in _object_position(sim, object_body)],
            **fit,
            **info,
        }
    finally:
        sim.cleanup()


def _render_release(
    *,
    wrapper_xml: Path,
    object_name: str,
    run_dir: Path,
    fps: float,
    clearance: float,
    compression: float,
    keep_frames: bool,
) -> dict[str, object]:
    HeadlessCDPRSimulation = _import_headless_simulation()
    sim = HeadlessCDPRSimulation(str(wrapper_xml), output_dir=str(run_dir), record_trajectory=False)
    frames: list[np.ndarray] = []
    try:
        sim.initialize()
        _setup_camera(sim)
        object_body = str(sim.get_object_body_name())
        _place_body_bottom_on_z(sim, object_body, OBJECT_START_XY, TABLE_TOP_Z)
        fit = _measure_physical_fit(sim, object_body, clearance=clearance, compression=compression)

        object_on_plate = _object_center_for_bottom_on_z(sim, object_body, PLATE_TOP_Z, clearance=0.002)
        object_on_plate[:2] = PLATE_CENTER[:2]
        release_ee = _ee_position_for_held_object(sim, object_on_plate)
        high_ee = release_ee + np.array([0.0, 0.0, 0.24], dtype=float)

        _force_gripper_opening(sim, float(fit["held_gripper_opening_01"]))
        _set_ee_position(sim, high_ee)
        _set_body_pose(sim, object_body, _held_object_position(sim))
        _step_physics(sim, steps=8)
        for _ in range(14):
            _step_physics(sim, steps=2)
            _capture(
                sim,
                frames,
                object_name=object_name,
                object_body=object_body,
                label="held above release target",
                fit=fit,
                attached=True,
            )

        _move_ee(
            sim,
            frames,
            object_name=object_name,
            object_body=object_body,
            label="lower to release height",
            fit=fit,
            target=release_ee,
            steps=34,
            attached=True,
            gripper_opening=float(fit["held_gripper_opening_01"]),
        )

        for idx in range(18):
            alpha = (idx + 1) / 18.0
            opening = (1.0 - alpha) * float(fit["held_gripper_opening_01"]) + alpha * 1.0
            _force_gripper_opening(sim, opening)
            _step_physics(sim, steps=3)
            _capture(
                sim,
                frames,
                object_name=object_name,
                object_body=object_body,
                label="opening fingers to release",
                fit=fit,
                attached=False,
            )

        retreat_ee = release_ee + np.array([0.0, 0.0, 0.18], dtype=float)
        for idx in range(34):
            alpha = (idx + 1) / 34.0
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            _set_ee_position(sim, (1.0 - smooth) * release_ee + smooth * retreat_ee)
            _force_gripper_opening(sim, 1.0)
            _step_physics(sim, steps=2)
            _capture(
                sim,
                frames,
                object_name=object_name,
                object_body=object_body,
                label="released object on plate",
                fit=fit,
                attached=False,
            )

        output = run_dir / f"release_{object_name}.mp4"
        info = _write_video(frames, output, fps=fps, keep_frames=keep_frames)
        return {
            "object": object_name,
            "object_body": object_body,
            "kind": "release",
            "wrapper_xml": wrapper_xml.as_posix(),
            "final_object_position_m": [float(x) for x in _object_position(sim, object_body)],
            **fit,
            **info,
        }
    finally:
        sim.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render YCB CDPR pick/lift/move and release videos.")
    parser.add_argument("--cdpr-xml", type=Path, default=DEFAULT_CDPR_XML)
    parser.add_argument("--ycb-root", type=Path, default=DEFAULT_YCB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--clearance", type=float, default=0.0)
    parser.add_argument("--grip-compression", type=float, default=0.001)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    videos: list[dict[str, object]] = []
    for object_name in args.objects:
        wrapper_xml, object_xml = _build_wrapper_xml(
            cdpr_xml=args.cdpr_xml,
            ycb_root=args.ycb_root,
            object_name=str(object_name),
            output_dir=run_dir,
        )
        for render_fn in (_render_pick_lift_move, _render_release):
            video = render_fn(
                wrapper_xml=wrapper_xml,
                object_name=str(object_name),
                run_dir=run_dir,
                fps=float(args.fps),
                clearance=float(args.clearance),
                compression=float(args.grip_compression),
                keep_frames=bool(args.keep_frames),
            )
            video["source_object_xml"] = object_xml.as_posix()
            videos.append(video)

    manifest = {
        "cdpr_xml": args.cdpr_xml.resolve().as_posix(),
        "ycb_root": args.ycb_root.resolve().as_posix(),
        "clearance_m": float(args.clearance),
        "grip_compression_m": float(args.grip_compression),
        "videos": videos,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
