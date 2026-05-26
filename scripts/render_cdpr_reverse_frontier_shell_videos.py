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
DEFAULT_OBJECTS = ("ycb_apple",)
DEFAULT_INSTRUCTION_TYPES = (
    "move_to_object",
    "grab_object",
    "pick_up",
    "put_into_plate",
    "push_left",
    "push_right",
    "move_left_of_object",
    "move_right_of_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
)
REFERENCE_OBJECT = "ycb_pear"
SECOND_REFERENCE_OBJECT = "ycb_baseball"
CONTAINER_OBJECT = "plate"


def _task_metadata(objects: list[str]) -> dict[str, Any]:
    object_pool = list(
        dict.fromkeys([*objects, REFERENCE_OBJECT, SECOND_REFERENCE_OBJECT, CONTAINER_OBJECT, "bowl"])
    )
    return {
        "instruction_sampling": "uniform_cycle",
        "reward_mode": "sparse_binary",
        "sparse_success_reward": 1.0,
        "sparse_failure_reward": 0.0,
        "target_object_pool": objects,
        "catchable_object_pool": objects,
        "grippable_object_pool": objects,
        "container_object_pool": [CONTAINER_OBJECT, "bowl"],
        "required_scene_object_pool": [CONTAINER_OBJECT],
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


def _object_label(object_name: str) -> str:
    return str(object_name).replace("ycb_", "").replace("_", " ")


def _instruction_text(instruction_type: str, object_name: str) -> str:
    obj = _object_label(object_name)
    ref = _object_label(REFERENCE_OBJECT)
    second = _object_label(SECOND_REFERENCE_OBJECT)
    if instruction_type == "move_to_object":
        return f"move to {obj}"
    if instruction_type == "grab_object":
        return f"grab {obj}"
    if instruction_type == "pick_up":
        return f"pick up {obj}"
    if instruction_type == "put_into_plate":
        return f"put {obj} into plate"
    if instruction_type == "push_left":
        return f"push {obj} left"
    if instruction_type == "push_right":
        return f"push {obj} right"
    if instruction_type == "move_left_of_object":
        return f"move {obj} left of {ref}"
    if instruction_type == "move_right_of_object":
        return f"move {obj} right of {ref}"
    if instruction_type == "put_in_front_of_object":
        return f"put {obj} in front of {ref}"
    if instruction_type == "put_behind_object":
        return f"put {obj} behind {ref}"
    if instruction_type == "move_between_objects":
        return f"move {obj} between {ref} and {second}"
    return instruction_type.replace("_", " ")


def _reset_options(*, instruction_type: str, object_name: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "instruction_type": str(instruction_type),
        "target_object": str(object_name),
        "required_objects": [str(object_name)],
    }
    if instruction_type == "put_into_plate":
        options["reference_object"] = CONTAINER_OBJECT
        options["required_objects"] = [str(object_name), CONTAINER_OBJECT]
    elif instruction_type in {
        "move_left_of_object",
        "move_right_of_object",
        "put_in_front_of_object",
        "put_behind_object",
    }:
        options["reference_object"] = REFERENCE_OBJECT
        options["required_objects"] = [str(object_name), REFERENCE_OBJECT]
    elif instruction_type == "move_between_objects":
        options["reference_object"] = REFERENCE_OBJECT
        options["second_reference_object"] = SECOND_REFERENCE_OBJECT
        options["required_objects"] = [str(object_name), REFERENCE_OBJECT, SECOND_REFERENCE_OBJECT]
    return options


def _render_shell(
    *,
    object_name: str,
    instruction_type: str,
    shell_id: int,
    run_dir: Path,
    fps: float,
    hold_seconds: float,
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
        allowed_objects=[object_name, CONTAINER_OBJECT, "bowl", REFERENCE_OBJECT, SECOND_REFERENCE_OBJECT],
        wrapper_cleanup=False,
        use_wrapper_cache=True,
        reuse_existing_wrapper_variants=True,
        seed=int(seed),
    )
    frames: list[np.ndarray] = []
    try:
        options = _reset_options(instruction_type=instruction_type, object_name=object_name)
        obs, info = env.reset(
            options={
                **options,
            },
            instruction=_instruction_text(instruction_type, object_name),
            curriculum_mode="reverse_frontier",
            curriculum_shell=shell_id,
        )
        del obs
        if hasattr(env.sim, "hold_current_pose"):
            env.sim.hold_current_pose(warm_steps=0)

        hold_frames = max(1, int(round(float(fps) * max(float(hold_seconds), 0.1))))
        for _ in range(hold_frames):
            frames.append(
                _capture(
                    env,
                    object_name=object_name,
                    instruction_type=instruction_type,
                    shell_id=shell_id,
                    label=f"held reverse shell reset ({hold_seconds:.1f}s)",
                )
            )

        object_dir = run_dir / object_name / instruction_type
        video_path = object_dir / f"{object_name}_{instruction_type}_shell_{shell_id:02d}.mp4"
        final_info = env._base_info()
        _write_video(frames, video_path, fps=fps, keep_frames=keep_frames)
        summary = {
            "object": object_name,
            "instruction_type": instruction_type,
            "shell_id": int(shell_id),
            "shell_count": shell_count,
            "video": str(video_path),
            "frames": len(frames),
            "reset_info": info,
            "final_info": final_info,
        }
        (video_path.with_suffix(".json")).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="*", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--instruction-type", default=None)
    parser.add_argument("--instruction-types", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--hold-seconds", type=float, default=2.5)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()

    if args.instruction_type:
        instruction_types = [str(args.instruction_type)]
    elif args.instruction_types:
        instruction_types = [str(item) for item in args.instruction_types]
    else:
        instruction_types = list(DEFAULT_INSTRUCTION_TYPES)

    run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    old_metadata = os.environ.get("RLVLA_TASK_METADATA_JSON")
    os.environ["RLVLA_TASK_METADATA_JSON"] = json.dumps(_task_metadata([str(obj) for obj in args.objects]), sort_keys=True)
    try:
        from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import get_cdpr_reverse_shell_specs

        specs = {spec.instruction_id: spec for spec in get_cdpr_reverse_shell_specs(instruction_types)}
        summaries: list[dict[str, Any]] = []
        for object_idx, object_name in enumerate(args.objects):
            for instruction_idx, instruction_type in enumerate(instruction_types):
                if instruction_type not in specs:
                    raise ValueError(f"No reverse shell spec for instruction_type={instruction_type!r}.")
                shell_count = int(specs[instruction_type].shell_count)
                for shell_id in range(shell_count):
                    print(
                        f"[render] object={object_name} instruction={instruction_type} "
                        f"shell={shell_id}/{shell_count - 1}"
                    )
                    summaries.append(
                        _render_shell(
                            object_name=str(object_name),
                            instruction_type=str(instruction_type),
                            shell_id=shell_id,
                            run_dir=run_dir,
                            fps=float(args.fps),
                            hold_seconds=float(args.hold_seconds),
                            keep_frames=bool(args.keep_frames),
                            seed=int(args.seed) + object_idx * 1000 + instruction_idx * 100 + shell_id,
                        )
                    )
        manifest = {
            "run_dir": str(run_dir),
            "objects": [str(obj) for obj in args.objects],
            "instruction_types": instruction_types,
            "hold_seconds": float(args.hold_seconds),
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
