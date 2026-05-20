#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CDPR_MUJOCO_DIR = ROOT / "robots" / "cdpr" / "cdpr_mujoco"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CDPR_MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(CDPR_MUJOCO_DIR))

from cdpr_scene_switcher import (  # noqa: E402
    CDPR_XML,
    build_wrapper_mjcf,
    find_object_xml,
    find_scene_xml,
    make_placed_object_xml,
    preprocess_cdpr_set_ee_start,
    preprocess_scene_with_zoffset,
)
from headless_cdpr_egl import HeadlessCDPRSimulation  # noqa: E402
from robots.cdpr.cdpr_dataset.synthetic_tasks import (  # noqa: E402
    DEFAULT_LIFT_Z,
    DEFAULT_SAFETY_Z,
    _goto_if_available,
    body_bottom_offset,
    clear_sim_recording_buffers,
    follow_segment_minjerk,
    object_centers,
    prepare_cdpr_workspace,
    resolve_body_name,
    script_pick_and_hover,
    script_push,
    script_put_into_bowl,
    settle,
)


DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_ycb_curriculum_attempts"
CURRICULUM_INSTRUCTIONS = (
    "move_to_object",
    "grab_object",
    "pick_up",
    "push_left",
    "push_right",
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_between_objects",
)
INSTRUCTION_TEXT = {
    "move_to_object": "move to apple",
    "grab_object": "grab apple",
    "pick_up": "pick up apple",
    "push_left": "push apple left",
    "push_right": "push apple right",
    "put_into_plate": "put apple on plate",
    "move_left_of_object": "move apple to the left of pear",
    "move_right_of_object": "move apple to the right of pear",
    "move_between_objects": "move apple between pear and banana",
}


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    xy: tuple[float, float]
    dynamic: bool = True
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass
class Bodies:
    target: str
    plate: str
    reference: str
    second_reference: str


def _parse_instruction_types(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip() or raw.strip().lower() == "all":
        return CURRICULUM_INSTRUCTIONS
    requested = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [item for item in requested if item not in CURRICULUM_INSTRUCTIONS]
    if unknown:
        raise ValueError(f"Unknown instruction types: {unknown}. Available: {CURRICULUM_INSTRUCTIONS}")
    return requested


def _body_pos(sim: HeadlessCDPRSimulation, body_name: str) -> np.ndarray:
    body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise RuntimeError(f"Could not find body {body_name!r}.")
    return np.asarray(sim.data.xpos[body_id], dtype=float).copy()


def _set_free_body_pose(sim: HeadlessCDPRSimulation, body_name: str, pos: np.ndarray) -> bool:
    body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        return False
    jnt_num = int(sim.model.body_jntnum[body_id])
    jnt_adr = int(sim.model.body_jntadr[body_id])
    for offset in range(jnt_num):
        joint_id = jnt_adr + offset
        if sim.model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(sim.model.jnt_qposadr[joint_id])
        dadr = int(sim.model.jnt_dofadr[joint_id])
        sim.data.qpos[qadr : qadr + 3] = np.asarray(pos, dtype=float).reshape(3)
        sim.data.qpos[qadr + 3 : qadr + 7] = sim.model.body_quat[body_id].copy()
        sim.data.qvel[dadr : dadr + 6] = 0.0
        return True
    return False


def _ground_body_at_xy(sim: HeadlessCDPRSimulation, body_name: str, xy: tuple[float, float], *, table_z: float) -> None:
    mj.mj_forward(sim.model, sim.data)
    bottom_offset = float(body_bottom_offset(sim, body_name))
    pos = np.array([float(xy[0]), float(xy[1]), float(table_z) + bottom_offset], dtype=float)
    if not _set_free_body_pose(sim, body_name, pos):
        body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            sim.model.body_pos[body_id] = pos
    mj.mj_forward(sim.model, sim.data)


def _build_wrapper(run_dir: Path, *, scene: str, scene_z: float, ee_start: tuple[float, float, float], table_z: float) -> Path:
    scene_xml = find_scene_xml(scene)
    scene_for_include = run_dir / f"{scene}_zshift.xml"
    cdpr_for_include = run_dir / "cdpr_ee_override.xml"
    preprocess_scene_with_zoffset(scene_xml, float(scene_z), scene_for_include)
    preprocess_cdpr_set_ee_start(Path(CDPR_XML), np.asarray(ee_start, dtype=float), cdpr_for_include)

    object_specs = (
        ObjectSpec("ycb_apple", (-0.16, -0.05), True),
        ObjectSpec("plate", (0.15, 0.07), True),
        ObjectSpec("ycb_pear", (-0.14, 0.06), True),
        ObjectSpec("ycb_banana", (0.16, 0.06), True),
    )
    placed_xmls: list[Path] = []
    for idx, spec in enumerate(object_specs):
        object_xml = find_object_xml(spec.name)
        placed_path = run_dir / f"placed_{idx}_{spec.name}.xml"
        make_placed_object_xml(
            object_xml,
            placed_path,
            prefix=f"p{idx}",
            pos=np.array([spec.xy[0], spec.xy[1], table_z], dtype=float),
            quat=np.asarray(spec.quat_xyzw, dtype=float),
            force_dynamic=bool(spec.dynamic),
            logical_name=spec.name,
        )
        placed_xmls.append(placed_path)

    wrapper_xml = run_dir / "cdpr_ycb_curriculum_scene.xml"
    build_wrapper_mjcf(scene_for_include, cdpr_for_include, placed_xmls, wrapper_xml)
    return wrapper_xml


def _initial_layout(instruction: str) -> dict[str, tuple[float, float]]:
    layout = {
        "target": (-0.16, -0.05),
        "plate": (0.15, 0.07),
        "reference": (-0.14, 0.06),
        "second_reference": (0.16, 0.06),
    }
    if instruction in {"move_left_of_object", "move_right_of_object"}:
        layout["target"] = (-0.02, -0.03)
        layout["reference"] = (0.06, -0.03)
        layout["second_reference"] = (0.16, 0.06)
    elif instruction == "move_between_objects":
        layout["target"] = (-0.16, -0.03)
        layout["reference"] = (-0.14, 0.06)
        layout["second_reference"] = (0.16, 0.06)
    return layout


def _reset_objects_for_instruction(
    sim: HeadlessCDPRSimulation,
    bodies: Bodies,
    instruction: str,
    *,
    table_z: float,
) -> dict[str, list[float]]:
    layout = _initial_layout(instruction)
    _ground_body_at_xy(sim, bodies.target, layout["target"], table_z=table_z)
    _ground_body_at_xy(sim, bodies.plate, layout["plate"], table_z=table_z)
    _ground_body_at_xy(sim, bodies.reference, layout["reference"], table_z=table_z)
    _ground_body_at_xy(sim, bodies.second_reference, layout["second_reference"], table_z=table_z)
    for _ in range(8):
        sim.run_simulation_step(capture_frame=False)
    mj.mj_forward(sim.model, sim.data)
    return {
        "target": [float(x) for x in _body_pos(sim, bodies.target)],
        "plate": [float(x) for x in _body_pos(sim, bodies.plate)],
        "reference": [float(x) for x in _body_pos(sim, bodies.reference)],
        "second_reference": [float(x) for x in _body_pos(sim, bodies.second_reference)],
    }


def _script_move_to_object(sim: HeadlessCDPRSimulation, object_body_name: str, *, tol: float = 0.015) -> None:
    if hasattr(sim, "open_gripper"):
        sim.open_gripper()
    center_xy, top_z, _ = object_centers(sim, object_body_name)
    current = sim.get_end_effector_position().copy()
    above = np.array([center_xy[0], center_xy[1], max(DEFAULT_SAFETY_Z, top_z + 0.12)], dtype=float)
    near = np.array([center_xy[0], center_xy[1], max(top_z + 0.08, 0.22)], dtype=float)
    follow_segment_minjerk(sim, current, above, 1.0)
    _goto_if_available(sim, above, tol=tol)
    follow_segment_minjerk(sim, above, near, 0.6)
    _goto_if_available(sim, near, tol=tol)
    settle(sim, 20)


def _script_grab_object(sim: HeadlessCDPRSimulation, object_body_name: str, *, tol: float = 0.015) -> None:
    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=10)
    if hasattr(sim, "open_gripper"):
        sim.open_gripper()
    center_xy, top_z, _ = object_centers(sim, object_body_name)
    current = sim.get_end_effector_position().copy()
    above = np.array([center_xy[0], center_xy[1], max(DEFAULT_SAFETY_Z, top_z + 0.12)], dtype=float)
    grasp = np.array([center_xy[0], center_xy[1], top_z + 0.002], dtype=float)
    follow_segment_minjerk(sim, current, above, 1.0)
    _goto_if_available(sim, above, tol=tol)
    follow_segment_minjerk(sim, above, grasp, 0.6)
    _goto_if_available(sim, grasp, tol=tol)
    if hasattr(sim, "close_gripper"):
        sim.close_gripper()
    settle(sim, 40)


def _move_held_target_to_xy(
    sim: HeadlessCDPRSimulation,
    object_body_name: str,
    goal_xy: np.ndarray,
    *,
    tol: float = 0.015,
) -> None:
    script_pick_and_hover(
        sim,
        object_body_name=object_body_name,
        tol=tol,
        safety_z=DEFAULT_SAFETY_Z,
        lift_z=max(DEFAULT_LIFT_Z, DEFAULT_SAFETY_Z),
    )
    _, top_z, _ = object_centers(sim, object_body_name)
    current = sim.get_end_effector_position().copy()
    above_goal = np.array([float(goal_xy[0]), float(goal_xy[1]), max(DEFAULT_SAFETY_Z, top_z + 0.16)], dtype=float)
    follow_segment_minjerk(sim, current, above_goal, 1.2)
    _goto_if_available(sim, above_goal, tol=tol)
    settle(sim, 30)


def _script_relative_instruction(sim: HeadlessCDPRSimulation, instruction: str, bodies: Bodies) -> None:
    if instruction in {"move_left_of_object", "move_right_of_object"}:
        ref = _body_pos(sim, bodies.reference)
        sign = -1.0 if instruction == "move_left_of_object" else 1.0
        goal_xy = ref[:2] + np.array([0.12 * sign, 0.0], dtype=float)
        _move_held_target_to_xy(sim, bodies.target, goal_xy)
        return
    if instruction == "move_between_objects":
        ref = _body_pos(sim, bodies.reference)
        second = _body_pos(sim, bodies.second_reference)
        goal_xy = 0.5 * (ref[:2] + second[:2])
        _move_held_target_to_xy(sim, bodies.target, goal_xy)
        return
    raise KeyError(instruction)


def _run_scripted_attempt(sim: HeadlessCDPRSimulation, instruction: str, bodies: Bodies) -> None:
    if instruction == "move_to_object":
        _script_move_to_object(sim, bodies.target)
    elif instruction == "grab_object":
        _script_grab_object(sim, bodies.target)
    elif instruction == "pick_up":
        script_pick_and_hover(sim, object_body_name=bodies.target, tol=0.015)
    elif instruction == "push_left":
        script_push(sim, object_body_name=bodies.target, direction="left", distance=0.12, tol=0.015)
    elif instruction == "push_right":
        script_push(sim, object_body_name=bodies.target, direction="right", distance=0.12, tol=0.015)
    elif instruction == "put_into_plate":
        script_put_into_bowl(sim, object_body_name=bodies.target, bowl_body_name=bodies.plate, tol=0.015)
    elif instruction in {"move_left_of_object", "move_right_of_object", "move_between_objects"}:
        _script_relative_instruction(sim, instruction, bodies)
    else:
        raise KeyError(instruction)


def _install_sampler(sim: HeadlessCDPRSimulation, bodies: Bodies) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    original_step = sim.run_simulation_step

    def sample() -> None:
        try:
            contact = bool(sim.has_finger_contact())
        except Exception:
            contact = False
        samples.append(
            {
                "time": float(sim.data.time),
                "ee_pos": [float(x) for x in sim.get_end_effector_position()],
                "target_pos": [float(x) for x in _body_pos(sim, bodies.target)],
                "plate_pos": [float(x) for x in _body_pos(sim, bodies.plate)],
                "reference_pos": [float(x) for x in _body_pos(sim, bodies.reference)],
                "second_reference_pos": [float(x) for x in _body_pos(sim, bodies.second_reference)],
                "gripper_opening": float(sim.get_gripper_opening()),
                "finger_contact": contact,
            }
        )

    def wrapped_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        sample()
        return result

    sim.run_simulation_step = wrapped_step  # type: ignore[method-assign]
    sample()
    return samples


def _evaluate_samples(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not samples:
        return {}
    initial = samples[0]
    initial_target = np.asarray(initial["target_pos"], dtype=float)
    grasp_seen = False
    out = {
        key: {"success": False, "best_metric": None}
        for key in CURRICULUM_INSTRUCTIONS
    }

    for sample in samples:
        ee = np.asarray(sample["ee_pos"], dtype=float)
        target = np.asarray(sample["target_pos"], dtype=float)
        plate = np.asarray(sample["plate_pos"], dtype=float)
        reference = np.asarray(sample["reference_pos"], dtype=float)
        second = np.asarray(sample["second_reference_pos"], dtype=float)
        gripper_open = float(sample["gripper_opening"])
        closed = bool(gripper_open <= 0.35)
        contact = bool(sample["finger_contact"])
        ee_target_xy = float(np.linalg.norm(ee[:2] - target[:2]))

        move_to = ee_target_xy <= 0.025
        if move_to and not out["move_to_object"]["success"]:
            out["move_to_object"] = {"success": True, "best_metric": f"ee_target_xy={ee_target_xy:.3f} m"}

        grab = bool(closed and (contact or ee_target_xy <= 0.045))
        grasp_seen = bool(grasp_seen or grab)
        if grab and not out["grab_object"]["success"]:
            out["grab_object"] = {
                "success": True,
                "best_metric": f"closed={closed} contact={contact} ee_target_xy={ee_target_xy:.3f} m",
            }

        lift = float(target[2] - initial_target[2])
        pick = bool(grasp_seen and lift >= 0.05)
        if pick and not out["pick_up"]["success"]:
            out["pick_up"] = {"success": True, "best_metric": f"lift={lift:.3f} m"}

        left_motion = float(-(target[0] - initial_target[0]))
        right_motion = float(target[0] - initial_target[0])
        if left_motion >= 0.08 and not out["push_left"]["success"]:
            out["push_left"] = {"success": True, "best_metric": f"signed_x_motion={left_motion:.3f} m"}
        if right_motion >= 0.08 and not out["push_right"]["success"]:
            out["push_right"] = {"success": True, "best_metric": f"signed_x_motion={right_motion:.3f} m"}

        plate_xy_error = float(np.linalg.norm(target[:2] - plate[:2]))
        plate_z_error = float(abs(target[2] - plate[2]))
        if plate_xy_error <= 0.08 and plate_z_error <= 0.10 and not out["put_into_plate"]["success"]:
            out["put_into_plate"] = {
                "success": True,
                "best_metric": f"plate_xy_error={plate_xy_error:.3f} m z_error={plate_z_error:.3f} m",
            }

        target_motion_xy = float(np.linalg.norm(target[:2] - initial_target[:2]))
        left_offset = float(-(target[0] - reference[0]))
        right_offset = float(target[0] - reference[0])
        y_error = float(abs(target[1] - reference[1]))
        if left_offset >= 0.08 and y_error <= 0.12 and target_motion_xy >= 0.02 and not out["move_left_of_object"]["success"]:
            out["move_left_of_object"] = {
                "success": True,
                "best_metric": f"offset={left_offset:.3f} m y_error={y_error:.3f} m motion={target_motion_xy:.3f} m",
            }
        if right_offset >= 0.08 and y_error <= 0.12 and target_motion_xy >= 0.02 and not out["move_right_of_object"]["success"]:
            out["move_right_of_object"] = {
                "success": True,
                "best_metric": f"offset={right_offset:.3f} m y_error={y_error:.3f} m motion={target_motion_xy:.3f} m",
            }

        midpoint = 0.5 * (reference[:2] + second[:2])
        segment = second[:2] - reference[:2]
        seg_len_sq = float(np.dot(segment, segment))
        projection = 0.5 if seg_len_sq <= 1e-9 else float(np.dot(target[:2] - reference[:2], segment) / seg_len_sq)
        midpoint_error = float(np.linalg.norm(target[:2] - midpoint))
        between = bool(midpoint_error <= 0.07 and 0.0 <= projection <= 1.0 and target_motion_xy >= 0.02)
        if between and not out["move_between_objects"]["success"]:
            out["move_between_objects"] = {
                "success": True,
                "best_metric": f"midpoint_error={midpoint_error:.3f} m projection={projection:.2f} motion={target_motion_xy:.3f} m",
            }

    return out


def _write_video_ffmpeg(frames: list[np.ndarray], output_path: Path, *, fps: float, keep_frames: bool) -> dict[str, Any]:
    if not frames:
        raise RuntimeError("No frames to encode.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path.parent / f"{output_path.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    try:
        for idx, frame in enumerate(frames):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(frame_dir / f"{idx:05d}.png")
        subprocess.run(
            [
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
            ],
            check=True,
        )
        return {"video_path": output_path.as_posix(), "frame_count": len(frames), "fps": float(fps)}
    finally:
        if not keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)


def _run_attempt(
    *,
    wrapper_xml: Path,
    instruction: str,
    attempt_dir: Path,
    table_z: float,
    fps: float,
    keep_frames: bool,
) -> dict[str, Any]:
    sim = HeadlessCDPRSimulation(str(wrapper_xml), output_dir=str(attempt_dir), record_trajectory=True)
    error: str | None = None
    samples: list[dict[str, Any]] = []
    try:
        sim.initialize()
        prepare_cdpr_workspace(sim, initial_hold_warm_steps=10, clear_recordings=True)
        bodies = Bodies(
            target=resolve_body_name(sim, "ycb_apple"),
            plate=resolve_body_name(sim, "plate"),
            reference=resolve_body_name(sim, "ycb_pear"),
            second_reference=resolve_body_name(sim, "ycb_banana"),
        )
        initial_positions = _reset_objects_for_instruction(sim, bodies, instruction, table_z=table_z)
        clear_sim_recording_buffers(sim)
        samples = _install_sampler(sim, bodies)
        try:
            _run_scripted_attempt(sim, instruction, bodies)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        for _ in range(30):
            sim.run_simulation_step(capture_frame=True)
        if not sim.overview_frames:
            frame = sim.capture_frame(sim.overview_cam, "overview")
            if frame is not None:
                sim.overview_frames.append(frame)
        video_info = _write_video_ffmpeg(
            [np.asarray(frame, dtype=np.uint8) for frame in sim.overview_frames],
            attempt_dir / f"{instruction}_attempt_overview.mp4",
            fps=fps,
            keep_frames=keep_frames,
        )
        completed = _evaluate_samples(samples)
        return {
            "attempted_instruction": instruction,
            "instruction_text": INSTRUCTION_TEXT[instruction],
            "attempt_success": bool(completed.get(instruction, {}).get("success", False)),
            "completed_instructions": completed,
            "initial_positions": initial_positions,
            "final_positions": {
                "target": [float(x) for x in _body_pos(sim, bodies.target)],
                "plate": [float(x) for x in _body_pos(sim, bodies.plate)],
                "reference": [float(x) for x in _body_pos(sim, bodies.reference)],
                "second_reference": [float(x) for x in _body_pos(sim, bodies.second_reference)],
                "ee": [float(x) for x in sim.get_end_effector_position()],
            },
            "sample_count": len(samples),
            "error": error,
            **video_info,
        }
    finally:
        try:
            sim.cleanup()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real YCB-object CDPR scripted attempts and score partial LC-HOL++ successes.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--instruction-types", default="all", help="Comma-separated list, or all.")
    parser.add_argument("--scene", default="desk")
    parser.add_argument("--scene-z", type=float, default=-0.85)
    parser.add_argument("--table-z", type=float, default=0.15)
    parser.add_argument("--ee-start", default="0.0,0.0,0.40")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    ee_start_arr = np.fromstring(str(args.ee_start), sep=",", dtype=float)
    if ee_start_arr.size != 3:
        raise ValueError("--ee-start must be 'x,y,z'")

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    wrapper_xml = _build_wrapper(
        run_dir,
        scene=str(args.scene),
        scene_z=float(args.scene_z),
        ee_start=tuple(float(x) for x in ee_start_arr),
        table_z=float(args.table_z),
    )

    attempts = []
    for instruction in _parse_instruction_types(args.instruction_types):
        attempt_dir = run_dir / "attempts" / instruction
        attempt_dir.mkdir(parents=True, exist_ok=True)
        summary = _run_attempt(
            wrapper_xml=wrapper_xml,
            instruction=instruction,
            attempt_dir=attempt_dir,
            table_z=float(args.table_z),
            fps=float(args.fps),
            keep_frames=bool(args.keep_frames),
        )
        attempts.append(summary)
        print(
            f"{instruction}: attempted_success={summary['attempt_success']} "
            f"video={summary['video_path']}"
        )

    manifest = {
        "run_dir": run_dir.as_posix(),
        "generated_at": datetime.now().isoformat(),
        "render_mode": "mujoco_real_ycb_overview",
        "uses_openvla_oft": False,
        "scene": str(args.scene),
        "wrapper_xml": wrapper_xml.as_posix(),
        "objects": ["ycb_apple", "plate", "ycb_pear", "ycb_banana"],
        "instructions": attempts,
        "python": sys.executable,
        "mujoco_version": mj.__version__,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
