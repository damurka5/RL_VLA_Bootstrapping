#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

import mujoco as mj

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - runtime dependency
    imageio = None

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - visual annotation is optional
    Image = None
    ImageDraw = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_XML = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr.xml"
DEFAULT_OUTPUT = ROOT / "runs" / "gripper_position_demos"


def _demo_xml_with_object(xml_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> found in {xml_path}")
    if worldbody.find("./body[@name='target_object']") is None:
        body = ET.SubElement(worldbody, "body", {"name": "target_object", "pos": "0.20 0.00 0.025"})
        ET.SubElement(
            body,
            "geom",
            {
                "name": "target_box",
                "type": "box",
                "size": "0.035 0.035 0.025",
                "rgba": "0.8 0.15 0.1 1",
                "contype": "1",
                "conaffinity": "1",
            },
        )
    out_path = output_dir / "cdpr_gripper_demo.xml"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    return out_path


def _annotate(frame: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return frame
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    text = "\n".join(str(line) for line in lines)
    bbox = draw.multiline_textbbox((0, 0), text)
    pad = 8
    rect = (8, 8, 8 + (bbox[2] - bbox[0]) + 2 * pad, 8 + (bbox[3] - bbox[1]) + 2 * pad)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.multiline_text((8 + pad, 8 + pad), text, fill=(255, 255, 255))
    return np.asarray(image)


def _draw_schematic_frame(
    *,
    title: str,
    target: float,
    actual: float,
    command: float | None,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    if Image is None or ImageDraw is None:
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        return frame

    target = float(np.clip(target, 0.0, 1.0))
    actual = float(np.clip(actual, 0.0, 1.0))
    image = Image.new("RGB", (width, height), (242, 244, 247))
    draw = ImageDraw.Draw(image)

    palm_x0, palm_y0, palm_x1, palm_y1 = 250, 110, 390, 165
    draw.rounded_rectangle((palm_x0, palm_y0, palm_x1, palm_y1), radius=8, fill=(42, 47, 58))
    draw.rectangle((width // 2 - 4, palm_y1, width // 2 + 4, 330), fill=(95, 101, 113))

    base_gap = 55
    travel_px = 95
    half_gap = base_gap + actual * travel_px
    finger_w = 32
    finger_h = 190
    left_x = width // 2 - half_gap - finger_w
    right_x = width // 2 + half_gap
    finger_y = 170
    draw.rounded_rectangle((left_x, finger_y, left_x + finger_w, finger_y + finger_h), radius=9, fill=(31, 36, 46))
    draw.rounded_rectangle((right_x, finger_y, right_x + finger_w, finger_y + finger_h), radius=9, fill=(31, 36, 46))
    draw.rounded_rectangle((left_x - 4, finger_y + finger_h - 32, left_x + finger_w + 4, finger_y + finger_h), radius=10, fill=(75, 84, 99))
    draw.rounded_rectangle((right_x - 4, finger_y + finger_h - 32, right_x + finger_w + 4, finger_y + finger_h), radius=10, fill=(75, 84, 99))

    target_half_gap = base_gap + target * travel_px
    for x in (width // 2 - target_half_gap, width // 2 + target_half_gap):
        draw.line((x, finger_y - 10, x, finger_y + finger_h + 20), fill=(37, 99, 235), width=3)

    bar_x0, bar_y0, bar_x1, bar_y1 = 90, 410, 550, 430
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=8, fill=(216, 222, 232))
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x0 + int((bar_x1 - bar_x0) * actual), bar_y1), radius=8, fill=(22, 163, 74))
    target_x = bar_x0 + int((bar_x1 - bar_x0) * target)
    draw.line((target_x, bar_y0 - 8, target_x, bar_y1 + 8), fill=(37, 99, 235), width=4)

    lines = [
        title,
        f"target={target:.2f}   actual={actual:.2f}",
        "0.00 closed      0.50 half open      1.00 fully open",
    ]
    if command is not None:
        lines.insert(2, f"delta command={command:+.2f}")
    draw.multiline_text((28, 24), "\n".join(lines), fill=(15, 23, 42), spacing=6)
    draw.text((bar_x0, bar_y0 - 26), "actual opening", fill=(15, 23, 42))
    draw.text((target_x + 8, bar_y0 - 30), "target", fill=(37, 99, 235))
    return np.asarray(image)


def _simulate_schematic(
    *,
    xml_path: Path,
    output_path: Path,
    fps: float,
    schedule: list[tuple[float, int, str, float | None]],
) -> dict[str, str]:
    if imageio is None:
        raise RuntimeError("imageio is required to write MP4 videos.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    act_gripper = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
    jnt_finger = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "finger_l")
    if act_gripper == -1 or jnt_finger == -1:
        raise RuntimeError("Could not find act_gripper/finger_l in the MuJoCo model.")
    qadr = int(model.jnt_qposadr[jnt_finger])
    joint_lo, joint_hi = [float(v) for v in model.jnt_range[jnt_finger]]
    joint_span = max(joint_hi - joint_lo, 1e-9)

    frames: list[np.ndarray] = []
    for target, steps, label, command in schedule:
        data.ctrl[act_gripper] = float(np.clip(target, 0.0, 1.0))
        for _ in range(max(1, int(steps))):
            mj.mj_step(model, data)
            actual = float(np.clip((float(data.qpos[qadr]) - joint_lo) / joint_span, 0.0, 1.0))
            frames.append(
                _draw_schematic_frame(
                    title=label,
                    target=float(target),
                    actual=actual,
                    command=command,
                )
            )

    imageio.mimsave(output_path, frames, fps=float(fps), macro_block_size=1)
    return {"video": output_path.as_posix(), "frames": str(len(frames))}


def _mujoco_overview_camera() -> mj.MjvCamera:
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.0, 0.0, 0.10])
    cam.distance = 1.5
    cam.azimuth = 90
    cam.elevation = -30
    return cam


def _simulate_mujoco_rendered(
    *,
    xml_path: Path,
    output_dir: Path,
    fps: float,
    schedule: list[tuple[float, int, str, float | None]],
) -> dict[str, str]:
    if imageio is None:
        raise RuntimeError("imageio is required to write MP4 videos.")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    # The demo isolates gripper actuation; disabling gravity keeps the suspended CDPR
    # pose steady without invoking the higher-level cable controller.
    model.opt.gravity[:] = 0.0

    act_gripper = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
    jnt_finger = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "finger_l")
    ee_camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
    if act_gripper == -1 or jnt_finger == -1 or ee_camera_id == -1:
        raise RuntimeError("Could not find act_gripper/finger_l/ee_camera in the MuJoCo model.")

    qadr = int(model.jnt_qposadr[jnt_finger])
    joint_lo, joint_hi = [float(v) for v in model.jnt_range[jnt_finger]]
    joint_span = max(joint_hi - joint_lo, 1e-9)

    overview_cam = _mujoco_overview_camera()
    renderer = mj.Renderer(model, width=640, height=480)
    overview_frames: list[np.ndarray] = []
    ee_frames: list[np.ndarray] = []

    try:
        for target, steps, label, command in schedule:
            data.ctrl[act_gripper] = float(np.clip(target, 0.0, 1.0))
            for _ in range(max(1, int(steps))):
                mj.mj_step(model, data)
                actual = float(np.clip((float(data.qpos[qadr]) - joint_lo) / joint_span, 0.0, 1.0))
                lines = [
                    label,
                    f"target={float(target):.2f} actual={actual:.2f}",
                ]
                if command is not None:
                    lines.append(f"delta_cmd={float(command):+.2f}")

                renderer.update_scene(data, camera=overview_cam)
                overview_frames.append(_annotate(renderer.render(), lines))

                renderer.update_scene(data, camera="ee_camera")
                ee_frames.append(_annotate(renderer.render(), lines))
    finally:
        renderer.close()

    overview_path = output_dir / "overview_video.mp4"
    ee_path = output_dir / "ee_camera_video.mp4"
    imageio.mimsave(overview_path, overview_frames, fps=float(fps), macro_block_size=1)
    imageio.mimsave(ee_path, ee_frames, fps=float(fps), macro_block_size=1)
    return {
        "overview_video": overview_path.as_posix(),
        "ee_camera_video": ee_path.as_posix(),
        "frames": str(len(overview_frames)),
    }


def _run_schematic_demos(args: argparse.Namespace, run_dir: Path, xml_path: Path) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}

    absolute_targets = [0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0]
    outputs["absolute_positions"] = _simulate_schematic(
        xml_path=xml_path,
        output_path=run_dir / "absolute_positions" / "schematic_video.mp4",
        fps=float(args.fps),
        schedule=[
            (target, int(args.hold_frames), f"absolute target {target:.2f}", None)
            for target in absolute_targets
        ],
    )

    half_targets = [0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0]
    outputs["half_open_hold"] = _simulate_schematic(
        xml_path=xml_path,
        output_path=run_dir / "half_open_hold" / "schematic_video.mp4",
        fps=float(args.fps),
        schedule=[
            (target, int(args.hold_frames), f"half-open hold target {target:.2f}", None)
            for target in half_targets
        ],
    )

    target = 0.0
    schedule: list[tuple[float, int, str, float | None]] = []
    for command in [1.0] * 22 + [-1.0] * 22 + [0.5] * 8 + [-0.5] * 8:
        target = float(np.clip(target + command * float(args.action_step_gripper), 0.0, 1.0))
        schedule.append((target, int(args.delta_hold_frames), "delta gripper control", float(command)))
    outputs["delta_open_close"] = _simulate_schematic(
        xml_path=xml_path,
        output_path=run_dir / "delta_open_close" / "schematic_video.mp4",
        fps=float(args.fps),
        schedule=schedule,
    )

    return outputs


def _run_mujoco_demos(args: argparse.Namespace, run_dir: Path, xml_path: Path) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}

    absolute_targets = [0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0]
    outputs["absolute_positions"] = _simulate_mujoco_rendered(
        xml_path=xml_path,
        output_dir=run_dir / "absolute_positions",
        fps=float(args.fps),
        schedule=[
            (target, int(args.hold_frames), f"absolute target {target:.2f}", None)
            for target in absolute_targets
        ],
    )

    half_targets = [0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0]
    outputs["half_open_hold"] = _simulate_mujoco_rendered(
        xml_path=xml_path,
        output_dir=run_dir / "half_open_hold",
        fps=float(args.fps),
        schedule=[
            (target, int(args.hold_frames), f"half-open hold target {target:.2f}", None)
            for target in half_targets
        ],
    )

    target = 0.0
    schedule: list[tuple[float, int, str, float | None]] = []
    for command in [1.0] * 22 + [-1.0] * 22 + [0.5] * 8 + [-0.5] * 8:
        target = float(np.clip(target + command * float(args.action_step_gripper), 0.0, 1.0))
        schedule.append((target, int(args.delta_hold_frames), "delta gripper control", float(command)))
    outputs["delta_open_close"] = _simulate_mujoco_rendered(
        xml_path=xml_path,
        output_dir=run_dir / "delta_open_close",
        fps=float(args.fps),
        schedule=schedule,
    )

    return outputs


def _import_headless_simulation():
    from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

    return HeadlessCDPRSimulation


def _capture_step(sim: HeadlessCDPRSimulation, *, label: str, command: float | None = None) -> None:
    sim.run_simulation_step(capture_frame=True)
    target = sim.get_gripper_target()
    opening = sim.get_gripper_opening()
    lines = [
        label,
        f"target={target:.2f}  actual={opening:.2f}",
    ]
    if command is not None:
        lines.append(f"delta_cmd={command:+.2f}")
    if sim.overview_frames:
        sim.overview_frames[-1] = _annotate(sim.overview_frames[-1], lines)
    if sim.ee_camera_frames:
        sim.ee_camera_frames[-1] = _annotate(sim.ee_camera_frames[-1], lines)


def _hold_absolute(sim: HeadlessCDPRSimulation, *, target: float, label: str, steps: int) -> None:
    sim.set_gripper(target)
    for _ in range(max(1, int(steps))):
        _capture_step(sim, label=label)


def _apply_delta(
    sim: HeadlessCDPRSimulation,
    *,
    command: float,
    action_step_gripper: float,
    label: str,
    hold_steps: int,
) -> None:
    target = np.clip(sim.get_gripper_target() + float(command) * float(action_step_gripper), 0.0, 1.0)
    sim.set_gripper(float(target))
    for _ in range(max(1, int(hold_steps))):
        _capture_step(sim, label=label, command=command)


def _save_demo(sim: HeadlessCDPRSimulation, demo_dir: Path, *, fps: float, summary: dict) -> dict[str, str]:
    demo_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    overview_path = demo_dir / "overview_video.mp4"
    ee_path = demo_dir / "ee_camera_video.mp4"
    sim.save_video(sim.overview_frames, str(overview_path), fps=fps)
    sim.save_video(sim.ee_camera_frames, str(ee_path), fps=fps)
    outputs["overview_video"] = overview_path.as_posix()
    outputs["ee_camera_video"] = ee_path.as_posix()
    summary_path = demo_dir / "summary.json"
    summary_path.write_text(json.dumps({**summary, "outputs": outputs}, indent=2), encoding="utf-8")
    outputs["summary"] = summary_path.as_posix()
    return outputs


def _new_sim(xml_path: Path, output_dir: Path) -> HeadlessCDPRSimulation:
    HeadlessCDPRSimulation = _import_headless_simulation()
    sim = HeadlessCDPRSimulation(str(xml_path), output_dir=str(output_dir), record_trajectory=True)
    sim.initialize()
    sim.set_gripper(0.0)
    for _ in range(20):
        sim.run_simulation_step(capture_frame=False)
    sim.overview_frames.clear()
    sim.ee_camera_frames.clear()
    sim.frame_capture_timestamps.clear()
    sim.trajectory_data.clear()
    return sim


def main() -> int:
    parser = argparse.ArgumentParser(description="Render CDPR normalized gripper position-control demos.")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--hold-frames", type=int, default=16)
    parser.add_argument("--delta-hold-frames", type=int, default=4)
    parser.add_argument("--action-step-gripper", type=float, default=0.05)
    parser.add_argument("--render-mode", choices=("mujoco", "schematic"), default="mujoco")
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    xml_path = _demo_xml_with_object(args.xml_path, run_dir)
    if args.render_mode == "schematic":
        all_outputs = _run_schematic_demos(args, run_dir, xml_path)
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(all_outputs, indent=2), encoding="utf-8")
        print(manifest_path)
        return 0

    if args.render_mode == "mujoco":
        all_outputs = _run_mujoco_demos(args, run_dir, xml_path)
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(all_outputs, indent=2), encoding="utf-8")
        print(manifest_path)
        return 0

    all_outputs: dict[str, dict[str, str]] = {}

    demos = [
        ("absolute_positions", (0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0)),
        ("half_open_hold", (0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0)),
    ]
    for demo_name, targets in demos:
        sim = _new_sim(xml_path, run_dir / demo_name)
        try:
            for target in targets:
                _hold_absolute(
                    sim,
                    target=float(target),
                    label=f"{demo_name}: absolute target {float(target):.2f}",
                    steps=int(args.hold_frames),
                )
            all_outputs[demo_name] = _save_demo(
                sim,
                run_dir / demo_name,
                fps=float(args.fps),
                summary={"demo": demo_name, "targets": list(targets), "control": "absolute normalized 0..1"},
            )
        finally:
            sim.cleanup()

    delta_demo = "delta_open_close"
    sim = _new_sim(xml_path, run_dir / delta_demo)
    try:
        commands = [1.0] * 22 + [-1.0] * 22 + [0.5] * 8 + [-0.5] * 8
        for command in commands:
            _apply_delta(
                sim,
                command=float(command),
                action_step_gripper=float(args.action_step_gripper),
                label=f"{delta_demo}: delta control",
                hold_steps=int(args.delta_hold_frames),
            )
        all_outputs[delta_demo] = _save_demo(
            sim,
            run_dir / delta_demo,
            fps=float(args.fps),
            summary={
                "demo": delta_demo,
                "commands": commands,
                "action_step_gripper": float(args.action_step_gripper),
                "control": "normalized delta command integrated into 0..1 target",
            },
        )
    finally:
        sim.cleanup()

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(all_outputs, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
