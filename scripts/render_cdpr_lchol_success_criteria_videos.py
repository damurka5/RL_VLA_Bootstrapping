#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_lchol_success_criteria_videos"
SUPPORT_Z = 0.15
TARGET_SIZE = 0.026

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

CRITERIA = {
    "move_to_object": "EE XY distance to target <= 0.025 m",
    "grab_object": "gripper closed and target caught or EE XY distance <= 0.045 m",
    "pick_up": "target grasped and lifted >= 0.05 m",
    "push_left": "target x displacement left >= 0.08 m",
    "push_right": "target x displacement right >= 0.08 m",
    "put_into_plate": "target-plate XY error <= 0.08 m and Z error <= 0.10 m",
    "move_left_of_object": "target left of reference >= 0.08 m, Y error <= 0.12 m, target moved >= 0.02 m",
    "move_right_of_object": "target right of reference >= 0.08 m, Y error <= 0.12 m, target moved >= 0.02 m",
    "move_between_objects": "target within 0.07 m of midpoint, projection in segment, target moved >= 0.02 m",
}

TEXT = {
    "move_to_object": "move to apple",
    "grab_object": "grab apple",
    "pick_up": "pick up apple",
    "push_left": "push apple left",
    "push_right": "push apple right",
    "put_into_plate": "put apple into plate",
    "move_left_of_object": "move apple to the left of pear",
    "move_right_of_object": "move apple to the right of pear",
    "move_between_objects": "move apple between pear and mug",
}


@dataclass(frozen=True)
class State:
    ee_xy: np.ndarray
    ee_z: float
    target_xy: np.ndarray
    target_z: float
    gripper_open: float = 1.0
    grasped: bool = False
    reference_xy: np.ndarray | None = None
    second_reference_xy: np.ndarray | None = None
    plate_xy: np.ndarray | None = None
    label: str = ""


def _arr(x: float, y: float) -> np.ndarray:
    return np.array([float(x), float(y)], dtype=float)


def _lerp(a: np.ndarray | float, b: np.ndarray | float, alpha: float):
    return (1.0 - float(alpha)) * a + float(alpha) * b


def _world_to_px(xy: np.ndarray, *, width: int, height: int) -> tuple[float, float]:
    center = np.array([width * 0.52, height * 0.56], dtype=float)
    scale = 780.0
    p = center + np.array([float(xy[0]), -float(xy[1])], dtype=float) * scale
    return float(p[0]), float(p[1])


def _draw_text_panel(draw: ImageDraw.ImageDraw, lines: list[str]) -> None:
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, spacing=4)
    pad = 10
    rect = (18, 18, 18 + bbox[2] - bbox[0] + 2 * pad, 18 + bbox[3] - bbox[1] + 2 * pad)
    draw.rounded_rectangle(rect, radius=6, fill=(11, 18, 32))
    draw.multiline_text((18 + pad, 18 + pad), text, fill=(245, 248, 252), spacing=4)


def _draw_workspace(draw: ImageDraw.ImageDraw, *, width: int, height: int, ee_xy: np.ndarray) -> None:
    corners = [_arr(-0.38, -0.32), _arr(0.38, -0.32), _arr(0.38, 0.32), _arr(-0.38, 0.32)]
    pts = [_world_to_px(c, width=width, height=height) for c in corners]
    draw.polygon(pts, fill=(230, 232, 235), outline=(70, 80, 96))
    for value in np.linspace(-0.30, 0.30, 5):
        draw.line(
            [_world_to_px(_arr(value, -0.32), width=width, height=height), _world_to_px(_arr(value, 0.32), width=width, height=height)],
            fill=(204, 208, 214),
            width=1,
        )
        draw.line(
            [_world_to_px(_arr(-0.38, value), width=width, height=height), _world_to_px(_arr(0.38, value), width=width, height=height)],
            fill=(204, 208, 214),
            width=1,
        )
    ee_px = _world_to_px(ee_xy, width=width, height=height)
    for corner in pts:
        draw.line([corner, ee_px], fill=(98, 105, 118), width=2)
        x, y = corner
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(33, 40, 53))


def _draw_disc(draw: ImageDraw.ImageDraw, xy: np.ndarray, radius_px: float, *, fill, outline, width: int, height: int) -> None:
    x, y = _world_to_px(xy, width=width, height=height)
    draw.ellipse((x - radius_px, y - radius_px, x + radius_px, y + radius_px), fill=fill, outline=outline, width=3)


def _draw_target(draw: ImageDraw.ImageDraw, state: State, *, width: int, height: int) -> None:
    x, y = _world_to_px(state.target_xy, width=width, height=height)
    lift = max(0.0, float(state.target_z) - SUPPORT_Z)
    shadow = max(0.0, min(1.0, lift / 0.10))
    shadow_r = 19 + 10 * shadow
    draw.ellipse((x - shadow_r, y - shadow_r * 0.45, x + shadow_r, y + shadow_r * 0.45), fill=(184, 190, 199))
    half = 17
    draw.rounded_rectangle((x - half, y - half, x + half, y + half), radius=5, fill=(218, 56, 43), outline=(124, 24, 21), width=3)
    if lift > 0.01:
        draw.text((x + 22, y - 24), f"lift {lift:.2f} m", fill=(110, 24, 21))


def _draw_reference(draw: ImageDraw.ImageDraw, xy: np.ndarray, label: str, *, width: int, height: int, color) -> None:
    x, y = _world_to_px(xy, width=width, height=height)
    draw.rounded_rectangle((x - 18, y - 18, x + 18, y + 18), radius=7, fill=color, outline=(34, 55, 70), width=3)
    draw.text((x - 18, y + 24), label, fill=(21, 31, 46))


def _draw_gripper(draw: ImageDraw.ImageDraw, state: State, *, width: int, height: int) -> None:
    x, y = _world_to_px(state.ee_xy, width=width, height=height)
    draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=(20, 27, 38), outline=(255, 255, 255), width=2)
    gap = 12 + 24 * float(np.clip(state.gripper_open, 0.0, 1.0))
    draw.line((x - gap, y + 14, x - gap, y + 44), fill=(20, 27, 38), width=6)
    draw.line((x + gap, y + 14, x + gap, y + 44), fill=(20, 27, 38), width=6)
    draw.text((x + 18, y - 10), f"z {state.ee_z:.2f}", fill=(20, 27, 38))


def _draw_z_inset(draw: ImageDraw.ImageDraw, state: State, *, width: int, height: int) -> None:
    left = width - 150
    top = 90
    bottom = 320
    draw.rounded_rectangle((left - 18, top - 24, width - 28, bottom + 36), radius=6, fill=(247, 248, 250), outline=(174, 181, 190))
    draw.text((left - 4, top - 18), "Z", fill=(25, 35, 52))
    def z_to_y(z: float) -> float:
        return bottom - (float(z) - SUPPORT_Z) / 0.30 * (bottom - top)
    support_y = z_to_y(SUPPORT_Z)
    lift_y = z_to_y(SUPPORT_Z + 0.05)
    target_y = z_to_y(state.target_z)
    ee_y = z_to_y(state.ee_z)
    draw.line((left, support_y, width - 50, support_y), fill=(100, 106, 116), width=2)
    draw.text((left, support_y + 4), "table", fill=(70, 76, 86))
    draw.line((left, lift_y, width - 50, lift_y), fill=(31, 111, 235), width=2)
    draw.text((left, lift_y - 18), "+0.05", fill=(31, 111, 235))
    draw.rectangle((left + 14, target_y - 8, left + 42, target_y + 8), fill=(218, 56, 43))
    draw.ellipse((left + 62 - 7, ee_y - 7, left + 62 + 7, ee_y + 7), fill=(20, 27, 38))


def _render_frame(instruction: str, state: State, metric_line: str, success: bool) -> np.ndarray:
    width, height = 960, 720
    image = Image.new("RGB", (width, height), (242, 244, 247))
    draw = ImageDraw.Draw(image)
    _draw_workspace(draw, width=width, height=height, ee_xy=state.ee_xy)

    if state.plate_xy is not None:
        _draw_disc(draw, state.plate_xy, 47, fill=(210, 226, 255), outline=(31, 111, 235), width=width, height=height)
        px, py = _world_to_px(state.plate_xy, width=width, height=height)
        draw.text((px - 24, py + 54), "plate", fill=(21, 46, 90))
    if state.reference_xy is not None:
        _draw_reference(draw, state.reference_xy, "ref", width=width, height=height, color=(76, 175, 125))
    if state.second_reference_xy is not None:
        _draw_reference(draw, state.second_reference_xy, "ref2", width=width, height=height, color=(155, 112, 220))

    if instruction in {"move_to_object", "grab_object"}:
        radius_m = 0.025 if instruction == "move_to_object" else 0.045
        _draw_disc(draw, state.target_xy, radius_m * 780.0, fill=None, outline=(245, 166, 35), width=width, height=height)
    if instruction in {"move_left_of_object", "move_right_of_object"} and state.reference_xy is not None:
        sign = -1.0 if instruction == "move_left_of_object" else 1.0
        target_line = state.reference_xy + _arr(0.08 * sign, 0.0)
        x, _ = _world_to_px(target_line, width=width, height=height)
        draw.line((x, 240, x, 610), fill=(245, 166, 35), width=3)
    if instruction == "move_between_objects" and state.reference_xy is not None and state.second_reference_xy is not None:
        a = _world_to_px(state.reference_xy, width=width, height=height)
        b = _world_to_px(state.second_reference_xy, width=width, height=height)
        draw.line((a, b), fill=(245, 166, 35), width=4)
        midpoint = 0.5 * (state.reference_xy + state.second_reference_xy)
        _draw_disc(draw, midpoint, 0.07 * 780.0, fill=None, outline=(245, 166, 35), width=width, height=height)

    _draw_target(draw, state, width=width, height=height)
    _draw_gripper(draw, state, width=width, height=height)
    _draw_z_inset(draw, state, width=width, height=height)

    _draw_text_panel(
        draw,
        [
            f"Instruction: {TEXT[instruction]}",
            f"Phase: {state.label}",
            f"Criterion: {CRITERIA[instruction]}",
            metric_line,
            f"SUCCESS: {success}",
        ],
    )
    return np.asarray(image, dtype=np.uint8)


def _metric_line(instruction: str, state: State, initial_target_xy: np.ndarray, initial_target_z: float) -> tuple[str, bool]:
    if instruction == "move_to_object":
        d = float(np.linalg.norm(state.ee_xy - state.target_xy))
        return f"ee_target_xy={d:.3f} m <= 0.025", bool(d <= 0.025)
    if instruction == "grab_object":
        d = float(np.linalg.norm(state.ee_xy - state.target_xy))
        caught = bool(state.grasped and d <= 0.045)
        return f"closed={state.gripper_open <= 0.05} caught_target={caught} xy={d:.3f} m", caught
    if instruction == "pick_up":
        lift = float(state.target_z - SUPPORT_Z)
        success = bool(state.grasped and lift >= 0.05)
        return f"grasped={state.grasped} lift={lift:.3f} m >= 0.05", success
    if instruction in {"push_left", "push_right"}:
        sign = -1.0 if instruction == "push_left" else 1.0
        motion = float(sign * (state.target_xy[0] - initial_target_xy[0]))
        return f"signed_x_motion={motion:.3f} m >= 0.08", bool(motion >= 0.08)
    if instruction == "put_into_plate":
        assert state.plate_xy is not None
        xy_error = float(np.linalg.norm(state.target_xy - state.plate_xy))
        z_error = float(abs(state.target_z - (SUPPORT_Z + 0.012)))
        return f"plate_xy_error={xy_error:.3f} m, z_error={z_error:.3f} m", bool(xy_error <= 0.08 and z_error <= 0.10)
    if instruction in {"move_left_of_object", "move_right_of_object"}:
        assert state.reference_xy is not None
        sign = -1.0 if instruction == "move_left_of_object" else 1.0
        offset = float(sign * (state.target_xy[0] - state.reference_xy[0]))
        y_error = float(abs(state.target_xy[1] - state.reference_xy[1]))
        motion = float(np.linalg.norm(state.target_xy - initial_target_xy))
        success = bool(offset >= 0.08 and y_error <= 0.12 and motion >= 0.02)
        return f"offset={offset:.3f} m, y_error={y_error:.3f} m, motion={motion:.3f} m", success
    if instruction == "move_between_objects":
        assert state.reference_xy is not None and state.second_reference_xy is not None
        midpoint = 0.5 * (state.reference_xy + state.second_reference_xy)
        segment = state.second_reference_xy - state.reference_xy
        seg_len_sq = float(np.dot(segment, segment))
        projection = 0.5 if seg_len_sq <= 1e-9 else float(np.dot(state.target_xy - state.reference_xy, segment) / seg_len_sq)
        error = float(np.linalg.norm(state.target_xy - midpoint))
        motion = float(np.linalg.norm(state.target_xy - initial_target_xy))
        success = bool(error <= 0.07 and 0.0 <= projection <= 1.0 and motion >= 0.02)
        return f"midpoint_error={error:.3f} m, projection={projection:.2f}, motion={motion:.3f} m", success
    raise KeyError(instruction)


def _append_transition(
    frames: list[np.ndarray],
    instruction: str,
    state: State,
    initial_target_xy: np.ndarray,
    initial_target_z: float,
    *,
    target_state: State,
    steps: int,
    label: str,
) -> State:
    current = state
    for idx in range(max(1, int(steps))):
        alpha = (idx + 1) / float(max(1, int(steps)))
        current = State(
            ee_xy=_lerp(state.ee_xy, target_state.ee_xy, alpha),
            ee_z=float(_lerp(state.ee_z, target_state.ee_z, alpha)),
            target_xy=_lerp(state.target_xy, target_state.target_xy, alpha),
            target_z=float(_lerp(state.target_z, target_state.target_z, alpha)),
            gripper_open=float(_lerp(state.gripper_open, target_state.gripper_open, alpha)),
            grasped=bool(target_state.grasped if alpha >= 0.5 else state.grasped),
            reference_xy=state.reference_xy if state.reference_xy is not None else target_state.reference_xy,
            second_reference_xy=state.second_reference_xy if state.second_reference_xy is not None else target_state.second_reference_xy,
            plate_xy=state.plate_xy if state.plate_xy is not None else target_state.plate_xy,
            label=label,
        )
        metric, success = _metric_line(instruction, current, initial_target_xy, initial_target_z)
        frames.append(_render_frame(instruction, current, metric, success))
    return replace(current, label=label)


def _hold(
    frames: list[np.ndarray],
    instruction: str,
    state: State,
    initial_target_xy: np.ndarray,
    initial_target_z: float,
    *,
    steps: int,
    label: str,
) -> State:
    current = replace(state, label=label)
    metric, success = _metric_line(instruction, current, initial_target_xy, initial_target_z)
    for _ in range(max(1, int(steps))):
        frames.append(_render_frame(instruction, current, metric, success))
    return current


def _initial_state(
    *,
    target_xy: np.ndarray = _arr(-0.12, -0.04),
    reference_xy: np.ndarray | None = None,
    second_reference_xy: np.ndarray | None = None,
    plate_xy: np.ndarray | None = None,
) -> State:
    return State(
        ee_xy=_arr(0.0, 0.0),
        ee_z=0.34,
        target_xy=target_xy.copy(),
        target_z=SUPPORT_Z + TARGET_SIZE,
        gripper_open=1.0,
        grasped=False,
        reference_xy=None if reference_xy is None else reference_xy.copy(),
        second_reference_xy=None if second_reference_xy is None else second_reference_xy.copy(),
        plate_xy=None if plate_xy is None else plate_xy.copy(),
        label="start",
    )


def _build_instruction_frames(instruction: str, fps: float) -> tuple[list[np.ndarray], dict[str, object]]:
    frames: list[np.ndarray] = []
    if instruction in {"move_left_of_object", "move_right_of_object"}:
        state = _initial_state(target_xy=_arr(-0.02, -0.03), reference_xy=_arr(0.06, -0.03))
    elif instruction == "move_between_objects":
        state = _initial_state(target_xy=_arr(-0.16, -0.03), reference_xy=_arr(-0.14, 0.06), second_reference_xy=_arr(0.16, 0.06))
    elif instruction == "put_into_plate":
        state = _initial_state(target_xy=_arr(-0.16, -0.05), plate_xy=_arr(0.15, 0.07))
    else:
        state = _initial_state()

    initial_target_xy = state.target_xy.copy()
    initial_target_z = float(state.target_z)
    state = _hold(frames, instruction, state, initial_target_xy, initial_target_z, steps=12, label="initial state")

    def go(**kwargs) -> State:
        nonlocal state
        steps = int(kwargs.pop("_steps", 28))
        label = str(kwargs.pop("_label", "move"))
        target = replace(state, **kwargs)
        state = _append_transition(
            frames,
            instruction,
            state,
            initial_target_xy,
            initial_target_z,
            target_state=target,
            steps=steps,
            label=label,
        )
        return state

    if instruction == "move_to_object":
        go(ee_xy=state.target_xy.copy(), ee_z=0.23, _steps=48, _label="approach target object")
    elif instruction == "grab_object":
        go(ee_xy=state.target_xy.copy(), ee_z=0.22, _steps=36, _label="approach target")
        go(gripper_open=0.0, grasped=True, _steps=20, _label="close gripper on target")
    elif instruction == "pick_up":
        go(ee_xy=state.target_xy.copy(), ee_z=0.22, _steps=30, _label="approach target")
        go(gripper_open=0.0, grasped=True, _steps=18, _label="grasp target")
        go(ee_z=0.32, target_z=SUPPORT_Z + 0.08, _steps=38, _label="lift target")
    elif instruction in {"push_left", "push_right"}:
        sign = -1.0 if instruction == "push_left" else 1.0
        go(ee_xy=state.target_xy + _arr(-0.04 * sign, 0.0), ee_z=0.21, _steps=26, _label="move beside target")
        final_target = state.target_xy + _arr(0.10 * sign, 0.0)
        go(ee_xy=final_target + _arr(-0.03 * sign, 0.0), target_xy=final_target, _steps=48, _label="push target")
    elif instruction == "put_into_plate":
        assert state.plate_xy is not None
        go(ee_xy=state.target_xy.copy(), ee_z=0.22, _steps=28, _label="approach target")
        go(gripper_open=0.0, grasped=True, _steps=18, _label="grasp target")
        go(ee_z=0.30, target_z=SUPPORT_Z + 0.08, _steps=26, _label="lift target")
        go(ee_xy=state.plate_xy.copy(), target_xy=state.plate_xy.copy(), _steps=48, _label="carry to plate")
        go(ee_z=0.22, target_z=SUPPORT_Z + 0.012, _steps=24, _label="place into plate")
        go(gripper_open=1.0, grasped=False, _steps=18, _label="release")
    elif instruction in {"move_left_of_object", "move_right_of_object"}:
        assert state.reference_xy is not None
        sign = -1.0 if instruction == "move_left_of_object" else 1.0
        target_xy = state.reference_xy + _arr(0.12 * sign, 0.0)
        go(ee_xy=state.target_xy.copy(), ee_z=0.22, _steps=24, _label="approach target")
        go(gripper_open=0.0, grasped=True, _steps=16, _label="grasp target")
        go(ee_xy=target_xy.copy(), target_xy=target_xy.copy(), _steps=48, _label="move relative to reference")
    elif instruction == "move_between_objects":
        assert state.reference_xy is not None and state.second_reference_xy is not None
        target_xy = 0.5 * (state.reference_xy + state.second_reference_xy)
        go(ee_xy=state.target_xy.copy(), ee_z=0.22, _steps=24, _label="approach target")
        go(gripper_open=0.0, grasped=True, _steps=16, _label="grasp target")
        go(ee_xy=target_xy.copy(), target_xy=target_xy.copy(), _steps=54, _label="move between references")
    else:
        raise KeyError(instruction)

    state = _hold(frames, instruction, state, initial_target_xy, initial_target_z, steps=int(max(1, fps)), label="final success check")
    metric, success = _metric_line(instruction, state, initial_target_xy, initial_target_z)
    return frames, {
        "instruction_type": instruction,
        "instruction_text": TEXT[instruction],
        "criterion": CRITERIA[instruction],
        "metric": metric,
        "success": bool(success),
        "final_ee_xy": [float(x) for x in state.ee_xy],
        "final_ee_z": float(state.ee_z),
        "initial_target_xy": [float(x) for x in initial_target_xy],
        "final_target_xy": [float(x) for x in state.target_xy],
        "final_target_z": float(state.target_z),
        "reference_xy": None if state.reference_xy is None else [float(x) for x in state.reference_xy],
        "second_reference_xy": None if state.second_reference_xy is None else [float(x) for x in state.second_reference_xy],
        "plate_xy": None if state.plate_xy is None else [float(x) for x in state.plate_xy],
    }


def _write_video_ffmpeg(frames: list[np.ndarray], output_path: Path, *, fps: float, keep_frames: bool) -> dict[str, object]:
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


def _parse_instruction_types(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip() or raw.strip().lower() == "all":
        return CURRICULUM_INSTRUCTIONS
    requested = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [item for item in requested if item not in CURRICULUM_INSTRUCTIONS]
    if unknown:
        raise ValueError(f"Unknown instruction types: {unknown}. Available: {CURRICULUM_INSTRUCTIONS}")
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(description="Render local scripted success videos for LC-HOL++ CDPR instructions.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--instruction-types", default="all", help="Comma-separated list, or all.")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for instruction in _parse_instruction_types(args.instruction_types):
        frames, summary = _build_instruction_frames(instruction, fps=float(args.fps))
        video_info = _write_video_ffmpeg(
            frames,
            videos_dir / f"{instruction}_success_overview.mp4",
            fps=float(args.fps),
            keep_frames=bool(args.keep_frames),
        )
        summaries.append({**summary, **video_info})

    manifest = {
        "run_dir": run_dir.as_posix(),
        "generated_at": datetime.now().isoformat(),
        "render_mode": "scripted_overview_schematic",
        "uses_openvla_oft": False,
        "instructions": summaries,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
