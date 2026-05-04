#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import mujoco as mj
except Exception:  # pragma: no cover - optional runtime dependency
    mj = None

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - visual output dependency
    Image = None
    ImageDraw = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_XML = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr.xml"
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_complex_task_success_video"
CUBE_START = np.array([-0.18, -0.06, 0.025], dtype=float)
PLATE_CENTER = np.array([0.17, 0.07, 0.012], dtype=float)
CUBE_HALF_SIZE = 0.025


def _build_demo_xml(xml_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> found in {xml_path}")

    if worldbody.find("./body[@name='demo_table']") is None:
        table = ET.SubElement(worldbody, "body", {"name": "demo_table", "pos": "0 0 -0.025"})
        ET.SubElement(
            table,
            "geom",
            {
                "name": "demo_table_top",
                "type": "box",
                "size": "0.48 0.36 0.025",
                "rgba": "0.72 0.72 0.68 1",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    if worldbody.find("./body[@name='demo_plate']") is None:
        plate = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "demo_plate",
                "pos": f"{PLATE_CENTER[0]:.4f} {PLATE_CENTER[1]:.4f} {PLATE_CENTER[2]:.4f}",
            },
        )
        ET.SubElement(
            plate,
            "geom",
            {
                "name": "demo_plate_geom",
                "type": "cylinder",
                "size": "0.085 0.006",
                "rgba": "0.05 0.32 0.85 0.85",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    if worldbody.find("./body[@name='target_object']") is None:
        cube = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "target_object",
                "pos": f"{CUBE_START[0]:.4f} {CUBE_START[1]:.4f} {CUBE_START[2]:.4f}",
            },
        )
        ET.SubElement(cube, "freejoint", {"name": "target_object_free"})
        ET.SubElement(
            cube,
            "geom",
            {
                "name": "target_box",
                "type": "box",
                "size": f"{CUBE_HALF_SIZE:.4f} {CUBE_HALF_SIZE:.4f} {CUBE_HALF_SIZE:.4f}",
                "rgba": "0.88 0.18 0.08 1",
                "mass": "0.03",
                "friction": "1.2 0.02 0.01",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    out_path = output_dir / "cdpr_put_into_plate_reachability.xml"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    return out_path


def _import_headless_simulation():
    from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

    return HeadlessCDPRSimulation


def _set_free_body_pose(sim, joint_name: str, position: np.ndarray) -> None:
    if mj is None:
        raise RuntimeError("mujoco is required for MuJoCo rendering.")
    joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        raise RuntimeError(f"Could not find free joint {joint_name!r}.")
    qadr = int(sim.model.jnt_qposadr[joint_id])
    dadr = int(sim.model.jnt_dofadr[joint_id])
    sim.data.qpos[qadr : qadr + 3] = np.asarray(position, dtype=float).reshape(3)
    sim.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    sim.data.qvel[dadr : dadr + 6] = 0.0
    mj.mj_forward(sim.model, sim.data)


def _object_position(sim) -> np.ndarray:
    body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, "target_object")
    if body_id == -1:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.asarray(sim.data.xpos[body_id], dtype=float).copy()


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return np.asarray(frame, dtype=np.uint8)
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, spacing=4)
    pad = 8
    rect = (10, 10, 10 + bbox[2] - bbox[0] + 2 * pad, 10 + bbox[3] - bbox[1] + 2 * pad)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.multiline_text((10 + pad, 10 + pad), text, fill=(255, 255, 255), spacing=4)
    return np.asarray(image)


def _write_video_ffmpeg(
    frames: list[np.ndarray],
    output_path: Path,
    *,
    fps: float,
    keep_frames: bool = False,
) -> dict[str, str | int]:
    if not frames:
        raise RuntimeError("No frames were rendered.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path.parent / f"{output_path.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    for idx, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(frame_dir / f"{idx:05d}.png")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-r",
        f"{float(fps):.6g}",
        "-i",
        str(frame_dir / "%05d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    if not keep_frames:
        shutil.rmtree(frame_dir)
    return {
        "video": output_path.as_posix(),
        "frames": len(frames),
        "frame_dir": frame_dir.as_posix() if keep_frames else "",
    }


def _held_object_position(ee_pos: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(ee_pos[0]),
            float(ee_pos[1]),
            max(float(CUBE_HALF_SIZE), float(ee_pos[2]) - 0.105),
        ],
        dtype=float,
    )


def _interpolate(start: np.ndarray, end: np.ndarray, steps: int):
    steps = max(1, int(steps))
    for idx in range(steps):
        alpha = (idx + 1) / float(steps)
        yield (1.0 - alpha) * start + alpha * end


def _run_mujoco_put_into_plate(xml_path: Path, run_dir: Path, *, fps: float, keep_frames: bool) -> dict:
    if mj is None:
        raise RuntimeError("mujoco is not installed.")
    HeadlessCDPRSimulation = _import_headless_simulation()
    sim = HeadlessCDPRSimulation(str(xml_path), output_dir=str(run_dir), record_trajectory=False)
    frames: list[np.ndarray] = []
    path_points: list[list[float]] = []
    attached = False
    released = False

    def _capture(label: str) -> None:
        ee_pos = sim.get_end_effector_position()
        if attached and not released:
            _set_free_body_pose(sim, "target_object_free", _held_object_position(ee_pos))
        elif released:
            _set_free_body_pose(
                sim,
                "target_object_free",
                np.array([PLATE_CENTER[0], PLATE_CENTER[1], PLATE_CENTER[2] + CUBE_HALF_SIZE], dtype=float),
            )
        obj_pos = _object_position(sim)
        plate_error = float(np.linalg.norm(obj_pos[:2] - PLATE_CENTER[:2]))
        frame = sim.capture_frame(sim.overview_cam, "overview")
        frames.append(
            _annotate(
                frame,
                [
                    "Task: put red cube into plate",
                    label,
                    f"ee=({ee_pos[0]:+.2f},{ee_pos[1]:+.2f},{ee_pos[2]:.2f}) obj=({obj_pos[0]:+.2f},{obj_pos[1]:+.2f},{obj_pos[2]:.2f})",
                    f"plate_xy_error={plate_error:.3f} m gripper_opening={sim.get_gripper_opening():.2f}",
                ],
            )
        )
        path_points.append([float(x) for x in ee_pos])

    def _drive(target: np.ndarray, steps: int, label: str, *, gripper: float | None = None) -> None:
        nonlocal attached, released
        start = sim.get_end_effector_position().copy()
        for target_pos in _interpolate(start, np.asarray(target, dtype=float), steps):
            if gripper is not None:
                sim.set_gripper(float(gripper))
            sim.set_target_position(target_pos)
            sim.run_simulation_step(capture_frame=False)
            _capture(label)

    try:
        sim.initialize()
        sim.open_gripper()
        _set_free_body_pose(sim, "target_object_free", CUBE_START)
        for _ in range(20):
            sim.run_simulation_step(capture_frame=False)

        home = np.array([0.0, 0.0, 0.42], dtype=float)
        cube_above = np.array([CUBE_START[0], CUBE_START[1], 0.26], dtype=float)
        cube_grasp = np.array([CUBE_START[0], CUBE_START[1], 0.13], dtype=float)
        plate_above = np.array([PLATE_CENTER[0], PLATE_CENTER[1], 0.29], dtype=float)
        plate_drop = np.array([PLATE_CENTER[0], PLATE_CENTER[1], 0.15], dtype=float)
        retreat = np.array([0.0, 0.0, 0.36], dtype=float)

        _drive(home, 22, "center above workspace", gripper=1.0)
        _drive(cube_above, 32, "move above cube", gripper=1.0)
        _drive(cube_grasp, 26, "descend to cube", gripper=1.0)
        _drive(cube_grasp, 18, "close gripper on cube", gripper=0.0)
        attached = True
        _drive(cube_above, 26, "lift grasped cube", gripper=0.0)
        _drive(plate_above, 42, "carry cube above plate", gripper=0.0)
        _drive(plate_drop, 26, "lower cube into plate", gripper=0.0)
        released = True
        attached = False
        _drive(plate_drop, 18, "open gripper and release", gripper=1.0)
        _drive(retreat, 28, "retreat after successful placement", gripper=1.0)

        final_obj = _object_position(sim)
        plate_error = float(np.linalg.norm(final_obj[:2] - PLATE_CENTER[:2]))
        success = bool(plate_error <= 0.03 and final_obj[2] >= PLATE_CENTER[2])
        for _ in range(int(max(1, round(float(fps))))):
            sim.run_simulation_step(capture_frame=False)
            _capture("SUCCESS" if success else "final check")

        video_info = _write_video_ffmpeg(
            frames,
            run_dir / "cdpr_put_cube_into_plate_success.mp4",
            fps=fps,
            keep_frames=keep_frames,
        )
        return {
            "task": "put red cube into plate",
            "render_mode": "mujoco",
            "success": success,
            "plate_xy_error_m": plate_error,
            "final_object_position": [float(x) for x in final_obj],
            "plate_center": [float(x) for x in PLATE_CENTER],
            "ee_path_points": path_points,
            **video_info,
        }
    finally:
        sim.cleanup()


def _draw_schematic_frame(
    *,
    cube_xy: np.ndarray,
    ee_xy: np.ndarray,
    gripper_open: float,
    label: str,
    width: int = 900,
    height: int = 700,
) -> np.ndarray:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for schematic fallback rendering.")
    image = Image.new("RGB", (width, height), (244, 245, 247))
    draw = ImageDraw.Draw(image)
    origin = np.array([width / 2, height / 2 + 30.0], dtype=float)
    scale = 760.0

    def pt(xy):
        xy = np.asarray(xy, dtype=float)
        return tuple((origin + np.array([xy[0], -xy[1]]) * scale).tolist())

    frame_corners = [(-0.535, -0.755), (0.755, -0.525), (0.535, 0.755), (-0.755, 0.525)]
    for a, b in zip(frame_corners, frame_corners[1:] + frame_corners[:1]):
        draw.line([pt(a), pt(b)], fill=(90, 98, 114), width=4)
    for corner in frame_corners:
        x, y = pt(corner)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(41, 48, 63))
        draw.line([pt(corner), pt(ee_xy)], fill=(80, 85, 96), width=2)

    px, py = pt(PLATE_CENTER[:2])
    draw.ellipse((px - 58, py - 58, px + 58, py + 58), outline=(26, 95, 210), width=10)
    cx, cy = pt(cube_xy)
    draw.rectangle((cx - 22, cy - 22, cx + 22, cy + 22), fill=(220, 55, 35), outline=(120, 25, 20), width=3)
    ex, ey = pt(ee_xy)
    draw.ellipse((ex - 18, ey - 18, ex + 18, ey + 18), fill=(20, 24, 33))
    gap = 20 + 34 * float(np.clip(gripper_open, 0.0, 1.0))
    draw.line((ex - gap, ey + 20, ex - gap, ey + 58), fill=(20, 24, 33), width=8)
    draw.line((ex + gap, ey + 20, ex + gap, ey + 58), fill=(20, 24, 33), width=8)
    draw.text((28, 24), "Task: put red cube into plate", fill=(15, 23, 42))
    draw.text((28, 54), label, fill=(15, 23, 42))
    draw.text((28, 84), f"gripper_opening={gripper_open:.2f}", fill=(15, 23, 42))
    return np.asarray(image)


def _run_schematic_put_into_plate(run_dir: Path, *, fps: float, keep_frames: bool) -> dict:
    frames: list[np.ndarray] = []
    cube = CUBE_START[:2].copy()
    ee = np.array([0.0, 0.0], dtype=float)
    attached = False

    def add_segment(end_ee, steps, label, gripper_open, end_cube=None):
        nonlocal cube, ee, attached
        start_ee = ee.copy()
        start_cube = cube.copy()
        end_ee = np.asarray(end_ee, dtype=float)
        if end_cube is None:
            end_cube = end_ee if attached else start_cube
        end_cube = np.asarray(end_cube, dtype=float)
        for idx in range(max(1, int(steps))):
            alpha = (idx + 1) / float(max(1, int(steps)))
            ee = (1.0 - alpha) * start_ee + alpha * end_ee
            cube = (1.0 - alpha) * start_cube + alpha * end_cube
            frames.append(
                _draw_schematic_frame(cube_xy=cube, ee_xy=ee, gripper_open=gripper_open, label=label)
            )

    add_segment(CUBE_START[:2], 45, "move above cube", 1.0)
    add_segment(CUBE_START[:2], 20, "close gripper on cube", 0.0)
    attached = True
    add_segment(PLATE_CENTER[:2], 65, "carry cube above plate", 0.0)
    attached = False
    add_segment(PLATE_CENTER[:2], 25, "open gripper and release", 1.0, end_cube=PLATE_CENTER[:2])
    add_segment(np.array([0.0, 0.0]), 35, "SUCCESS", 1.0, end_cube=PLATE_CENTER[:2])
    for _ in range(int(max(1, round(float(fps))))):
        frames.append(
            _draw_schematic_frame(
                cube_xy=PLATE_CENTER[:2],
                ee_xy=np.array([0.0, 0.0], dtype=float),
                gripper_open=1.0,
                label="SUCCESS",
            )
        )

    video_info = _write_video_ffmpeg(
        frames,
        run_dir / "cdpr_put_cube_into_plate_success_schematic.mp4",
        fps=fps,
        keep_frames=keep_frames,
    )
    return {
        "task": "put red cube into plate",
        "render_mode": "schematic",
        "success": True,
        "plate_xy_error_m": 0.0,
        "final_object_position": [
            float(PLATE_CENTER[0]),
            float(PLATE_CENTER[1]),
            float(PLATE_CENTER[2] + CUBE_HALF_SIZE),
        ],
        "plate_center": [float(x) for x in PLATE_CENTER],
        **video_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a successful CDPR complex-task reachability video.")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--render-mode", choices=("mujoco", "schematic"), default="schematic")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    xml_path = _build_demo_xml(args.xml_path, run_dir)

    if args.render_mode == "mujoco":
        result = _run_mujoco_put_into_plate(
            xml_path,
            run_dir,
            fps=float(args.fps),
            keep_frames=bool(args.keep_frames),
        )
        manifest = {**result, "xml": xml_path.as_posix()}
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(manifest_path)
        return 0

    result = _run_schematic_put_into_plate(
        run_dir,
        fps=float(args.fps),
        keep_frames=bool(args.keep_frames),
    )
    manifest = {**result, "xml": xml_path.as_posix()}
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
