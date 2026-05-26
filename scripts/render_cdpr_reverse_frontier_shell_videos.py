#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CDPR_ROOT = ROOT / "robots" / "cdpr"
if str(CDPR_ROOT) not in sys.path:
    sys.path.insert(0, str(CDPR_ROOT))

DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_reverse_frontier_shell_videos"
DEFAULT_OBJECTS = ("ycb_apple", "ycb_pear", "ycb_baseball")
DEFAULT_INSTRUCTION = "put_into_plate"


def _task_metadata(objects: list[str]) -> dict[str, Any]:
    object_pool = list(dict.fromkeys([*objects, "plate", "bowl"]))
    return {
        "instruction_sampling": "uniform_cycle",
        "reward_mode": "sparse_binary",
        "sparse_success_reward": 1.0,
        "sparse_failure_reward": 0.0,
        "target_object_pool": objects,
        "catchable_object_pool": objects,
        "grippable_object_pool": objects,
        "container_object_pool": ["plate", "bowl"],
        "required_scene_object_pool": ["plate"],
        "scene_object_pool": object_pool,
        "distractor_object_pool": object_pool,
        "min_scene_objects": 3,
        "max_scene_objects": 4,
        "scene_variant_count": 64,
        "object_spawn_x_bounds": [-0.30, 0.30],
        "object_spawn_y_bounds": [-0.30, 0.30],
        "object_spawn_center_xy": [0.0, 0.0],
        "object_spawn_center_exclusion_radius": 0.10,
        "object_spawn_min_gap": 0.04,
        "object_spawn_min_ee_dist": 0.10,
        "object_spawn_max_tries": 200,
        "grab_require_caught": True,
        "grab_xy_tolerance": 0.025,
        "grab_closed_opening_threshold": 0.35,
        "catch_distance_threshold": 0.055,
        "catch_score_threshold": 0.30,
        "push_success_displacement": 0.08,
        "put_container_xy_tolerance": 0.08,
        "put_container_z_tolerance": 0.10,
        "put_min_target_motion": 0.04,
        "put_require_target_grasp": False,
        "put_require_release": True,
        "put_release_opening_threshold": 0.55,
        "relation_left_right_offset": 0.08,
        "relation_front_behind_offset": 0.08,
        "relation_y_tolerance": 0.12,
        "relation_x_tolerance": 0.12,
        "relation_min_target_motion": 0.04,
        "relation_require_target_grasp": True,
        "between_xy_tolerance": 0.07,
        "caught_object_start_object_offset": [0.0, 0.0, 0.005],
        "caught_object_start_fit_gripper": True,
        "caught_object_start_pin_object": False,
        "caught_object_start_gripper_clearance": 0.0,
        "caught_object_start_grip_compression": 0.001,
        "caught_object_start_min_height_above_table": 0.08,
        "caught_object_start_release_opening_threshold": 0.010,
        "caught_object_start_release_opening_margin": 0.04,
        "move_to_object_validation_distance_threshold": 0.025,
        "reverse_curriculum": {
            "enabled": True,
            "promotion_success": 0.50,
            "demotion_success": 0.20,
            "validation_rollouts_per_shell": 50,
            "min_train_updates_before_validation": 5,
            "max_shell_jump": 1,
            "saturation_abort_threshold": 0.30,
            "sample_frontier_probability": 0.80,
            "sample_rehearsal_probability": 0.20,
        },
    }


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, spacing=4)
    pad = 8
    rect = (10, 10, 10 + bbox[2] - bbox[0] + 2 * pad, 10 + bbox[3] - bbox[1] + 2 * pad)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.multiline_text((10 + pad, 10 + pad), text, fill=(255, 255, 255), spacing=4)
    return np.asarray(image)


def _write_video(frames: list[np.ndarray], output_path: Path, *, fps: float, keep_frames: bool) -> None:
    if not frames:
        raise RuntimeError("No frames captured for video.")
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


def _capture(env, *, object_name: str, instruction_type: str, shell_id: int, label: str) -> np.ndarray:
    sim = env.sim
    frame = sim.capture_frame(sim.overview_cam, "overview")
    info = env._base_info()
    ee = np.asarray(env._get_ee_position(), dtype=float)
    target = np.asarray(info.get("target_object_position_actual", [np.nan, np.nan, np.nan]), dtype=float)
    ref = np.asarray(info.get("reference_object_position", [np.nan, np.nan, np.nan]), dtype=float)
    return _annotate(
        frame,
        [
            f"{instruction_type} shell {shell_id} | {object_name}",
            label,
            f"target=({target[0]:+.2f},{target[1]:+.2f},{target[2]:.2f}) ref=({ref[0]:+.2f},{ref[1]:+.2f},{ref[2]:.2f})",
            f"ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:.2f}) gripper={float(info.get('gripper_opening', 0.0)):.2f}",
        ],
    )


def _scripted_action(env, *, instruction_type: str, shell_id: int) -> np.ndarray:
    del shell_id
    if instruction_type == "put_into_plate":
        target = np.asarray(env._current_target_reference_position(), dtype=np.float32)
        ee = np.asarray(env._get_ee_position(), dtype=np.float32)
        delta = target - ee
        action = np.zeros((5,), dtype=np.float32)
        if float(np.linalg.norm(delta[:3])) > 1e-6:
            action[:3] = np.clip(delta[:3] / max(float(getattr(env, "action_step_xyz", 0.02)), 1e-6), -0.5, 0.5)
        action[4] = 1.0
        return action
    return np.array([0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)


def _render_shell(
    *,
    object_name: str,
    instruction_type: str,
    shell_id: int,
    run_dir: Path,
    fps: float,
    keep_frames: bool,
    seed: int,
) -> dict[str, Any]:
    from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import get_cdpr_reverse_shell_specs
    from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv

    specs = {spec.instruction_id: spec for spec in get_cdpr_reverse_shell_specs([instruction_type])}
    if instruction_type not in specs:
        raise ValueError(f"No reverse shell spec for instruction_type={instruction_type!r}.")
    shell_count = int(specs[instruction_type].shell_count)

    env = CDPRLanguageRLEnv(
        max_steps=48,
        max_objects=4,
        action_step_xyz=0.015,
        action_step_yaw=0.08,
        action_step_gripper=0.05,
        hold_steps=4,
        capture_frames=False,
        instruction_types=[instruction_type],
        allowed_objects=[object_name, "plate", "bowl", "ycb_pear", "ycb_apple", "ycb_baseball"],
        wrapper_cleanup=False,
        use_wrapper_cache=True,
        reuse_existing_wrapper_variants=True,
        seed=int(seed),
    )
    frames: list[np.ndarray] = []
    try:
        obs, info = env.reset(
            options={
                "required_objects": [object_name, "plate"],
                "target_object": object_name,
                "reference_object": "plate",
            },
            instruction=f"put {object_name.replace('ycb_', '').replace('_', ' ')} into plate",
            curriculum_mode="reverse_frontier",
            curriculum_shell=shell_id,
        )
        del obs
        for _ in range(8):
            frames.append(
                _capture(
                    env,
                    object_name=object_name,
                    instruction_type=instruction_type,
                    shell_id=shell_id,
                    label="reverse shell reset",
                )
            )
            if hasattr(env.sim, "run_simulation_step"):
                env.sim.run_simulation_step(capture_frame=False)

        action = _scripted_action(env, instruction_type=instruction_type, shell_id=shell_id)
        for step_idx in range(18):
            _obs, _reward, done, truncated, step_info = env.step(action)
            frames.append(
                _capture(
                    env,
                    object_name=object_name,
                    instruction_type=instruction_type,
                    shell_id=shell_id,
                    label=f"scripted sparse probe step {step_idx + 1}",
                )
            )
            if done or truncated:
                break

        object_dir = run_dir / object_name
        video_path = object_dir / f"{instruction_type}_shell_{shell_id:02d}.mp4"
        _write_video(frames, video_path, fps=fps, keep_frames=keep_frames)
        summary = {
            "object": object_name,
            "instruction_type": instruction_type,
            "shell_id": int(shell_id),
            "shell_count": shell_count,
            "video": str(video_path),
            "frames": len(frames),
            "reset_info": info,
            "final_info": step_info if "step_info" in locals() else info,
        }
        (video_path.with_suffix(".json")).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="*", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--instruction-type", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()

    run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    old_metadata = os.environ.get("RLVLA_TASK_METADATA_JSON")
    os.environ["RLVLA_TASK_METADATA_JSON"] = json.dumps(_task_metadata([str(obj) for obj in args.objects]), sort_keys=True)
    try:
        from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import get_cdpr_reverse_shell_specs

        specs = {spec.instruction_id: spec for spec in get_cdpr_reverse_shell_specs([args.instruction_type])}
        shell_count = int(specs[str(args.instruction_type)].shell_count)
        summaries: list[dict[str, Any]] = []
        for object_idx, object_name in enumerate(args.objects):
            for shell_id in range(shell_count):
                print(f"[render] object={object_name} instruction={args.instruction_type} shell={shell_id}/{shell_count - 1}")
                summaries.append(
                    _render_shell(
                        object_name=str(object_name),
                        instruction_type=str(args.instruction_type),
                        shell_id=shell_id,
                        run_dir=run_dir,
                        fps=float(args.fps),
                        keep_frames=bool(args.keep_frames),
                        seed=int(args.seed) + object_idx * 100 + shell_id,
                    )
                )
        manifest = {
            "run_dir": str(run_dir),
            "objects": [str(obj) for obj in args.objects],
            "instruction_type": str(args.instruction_type),
            "shell_count": shell_count,
            "videos": summaries,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        if old_metadata is None:
            os.environ.pop("RLVLA_TASK_METADATA_JSON", None)
        else:
            os.environ["RLVLA_TASK_METADATA_JSON"] = old_metadata


if __name__ == "__main__":
    main()
