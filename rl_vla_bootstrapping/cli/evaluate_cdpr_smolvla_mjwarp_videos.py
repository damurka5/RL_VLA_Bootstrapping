#!/usr/bin/env python3
"""Record reproducible SmolVLA GRPO rollouts on the production MJWarp backend.

This evaluator reuses the held-out validation backend used by
``smolvla_grpo_mjwarp_cdpr`` while deliberately replacing its near-target
Reverse Frontier EE reset with a random-workspace qualitative audit. It
restores the checkpoint's residual policy and uses the same fixed four-slot
RoboCasa scene, frozen SmolVLA runtime, compact state, action chunking,
controller, reward, and success predicate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import textwrap
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ACTION_NAMES = ("x", "y", "z", "yaw", "gripper")
DEFAULT_TOLERANCES_M = (0.02, 0.025, 0.05, 0.10, 0.15)
GROUP_SIZE = 8


def _scene_object_count_for_round(
    round_index: int, minimum: int, maximum: int
) -> int:
    """Cycle inclusively through the configured scene-object counts."""

    return int(minimum) + int(round_index) % (
        int(maximum) - int(minimum) + 1
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a SmolVLA GRPO checkpoint from random workspace starts "
            "on the exact MJWarp backend and record telemetry MP4s."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-target", type=int, default=3)
    parser.add_argument("--success-distance", type=float, default=0.02)
    parser.add_argument(
        "--tolerances",
        nargs="+",
        type=float,
        default=list(DEFAULT_TOLERANCES_M),
        help="XY distances used for the qualitative best-distance sweep.",
    )
    parser.add_argument("--validation-seed", type=int, default=1_000_000)
    parser.add_argument(
        "--min-scene-objects",
        type=int,
        default=1,
        help="Minimum active scene objects; counts are cycled across episodes.",
    )
    parser.add_argument(
        "--max-scene-objects",
        type=int,
        default=3,
        help="Maximum active scene objects; counts are cycled across episodes.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smolvla-microbatch-size", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--terminal-hold-seconds", type=float, default=1.0)
    parser.add_argument(
        "--min-policy-inferences",
        type=int,
        default=10,
        help="Minimum fresh SmolVLA plus GRPO policy calls per episode.",
    )
    parser.add_argument(
        "--min-video-seconds",
        type=float,
        default=5.0,
        help="Pad the final annotated frame until every MP4 reaches this duration.",
    )
    parser.add_argument(
        "--random-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replace the Reverse Frontier near-target EE reset with a seeded "
            "random workspace pose shared by the eight candidate worlds."
        ),
    )
    parser.add_argument(
        "--random-start-x-bounds",
        nargs=2,
        type=float,
        default=(-0.24, 0.24),
    )
    parser.add_argument(
        "--random-start-y-bounds",
        nargs=2,
        type=float,
        default=(-0.24, 0.24),
    )
    parser.add_argument(
        "--random-start-z-bounds",
        nargs=2,
        type=float,
        default=(0.32, 0.52),
    )
    parser.add_argument(
        "--random-start-min-xy-distance",
        type=float,
        default=0.12,
        help="Reject random EE starts closer than this to the target in XY.",
    )
    parser.add_argument(
        "--curriculum-shell",
        type=int,
        default=None,
        help=(
            "Optional checkpoint-shell provenance override. The qualitative "
            "audit still enforces its minimum policy-call horizon."
        ),
    )
    return parser


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch before weights_only.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "policy" not in payload:
        raise ValueError(
            f"{path} is not a GRPO policy checkpoint with a 'policy' state."
        )
    if not isinstance(payload.get("args"), Mapping):
        raise ValueError(
            f"{path} does not contain the saved training arguments required "
            "to reproduce its MJWarp runtime."
        )
    return payload


def _evaluation_training_args(
    payload: Mapping[str, Any],
    *,
    config_path: Path,
    xml_path: Path,
    device: str,
    render_width: int,
    render_height: int,
    allowed_objects: Sequence[str],
    microbatch_size: int,
) -> Namespace:
    """Copy checkpoint arguments while shrinking only the independent batch."""

    values = dict(payload["args"])
    values.update(
        {
            "config": str(config_path),
            "device": str(device),
            "distributed": False,
            "simulator_backend": "mjlab_mjwarp",
            "worlds_per_rank": GROUP_SIZE,
            "groups_per_rank": 1,
            "grpo_group_size": GROUP_SIZE,
            "mjwarp_xml_path": str(xml_path),
            "render_width": int(render_width),
            "render_height": int(render_height),
            "allowed_objects": list(allowed_objects),
            "instruction_types": ["move_to_object"],
            "smolvla_inference_microbatch_size": max(
                1, min(int(microbatch_size), GROUP_SIZE)
            ),
            "smolvla_compile_model": False,
            "resume_checkpoint": None,
        }
    )
    return Namespace(**values)


def _validate_simulator_compatibility(
    payload: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
) -> None:
    """Require checkpoint physics/render identity, allowing a smaller batch."""

    stored = dict(payload.get("simulator_metadata") or {})
    if not stored:
        raise ValueError(
            "The checkpoint has no simulator metadata, so exact MJWarp "
            "compatibility cannot be established."
        )
    keys = (
        "backend",
        "versions",
        "physics_substeps_per_action",
        "physics_dtype",
        "controller_implementation",
        "action_step_xyz",
        "action_step_yaw",
        "action_step_gripper",
        "lock_non_commanded_axes",
        "lock_non_commanded_axes_threshold",
        "xml_sha256",
        "object_assets_sha256",
        "object_geometry",
        "nconmax_per_world",
        "njmax_per_world",
        "nccdmax_per_world",
        "render_width",
        "render_height",
        "object_slots",
        "object_catalogs",
        "camera_order",
    )
    differences = {
        key: (stored.get(key), runtime_metadata.get(key))
        for key in keys
        if stored.get(key) != runtime_metadata.get(key)
    }
    if differences:
        details = ", ".join(
            f"{key}: checkpoint={old!r}, runtime={new!r}"
            for key, (old, new) in differences.items()
        )
        raise RuntimeError(
            "Checkpoint simulator assumptions are incompatible with the "
            f"video evaluator: {details}"
        )


def _sample_random_start_xy(
    *,
    target_xy: Sequence[float],
    x_bounds: Sequence[float],
    y_bounds: Sequence[float],
    min_distance: float,
    seed: int,
    attempts: int = 256,
) -> np.ndarray:
    """Choose a reproducible workspace start that is not already near target."""

    target = np.asarray(target_xy, dtype=np.float64).reshape(2)
    x_low, x_high = (float(x_bounds[0]), float(x_bounds[1]))
    y_low, y_high = (float(y_bounds[0]), float(y_bounds[1]))
    if x_low >= x_high or y_low >= y_high:
        raise ValueError("Random-start workspace bounds must be increasing.")
    required = max(0.0, float(min_distance))
    rng = np.random.default_rng(int(seed))
    candidates = np.column_stack(
        (
            rng.uniform(x_low, x_high, size=max(1, int(attempts))),
            rng.uniform(y_low, y_high, size=max(1, int(attempts))),
        )
    )
    distances = np.linalg.norm(candidates - target[None, :], axis=1)
    valid = np.flatnonzero(distances >= required)
    if valid.size:
        return candidates[int(valid[0])].astype(np.float32)
    farthest = int(np.argmax(distances))
    raise ValueError(
        "Could not sample an EE start at least "
        f"{required:.3f} m from target {target.tolist()} inside "
        f"x=[{x_low}, {x_high}], y=[{y_low}, {y_high}]. "
        f"Farthest sampled distance was {distances[farthest]:.3f} m."
    )


def _apply_random_ee_start(
    *,
    backend: Any,
    reset: Any,
    round_index: int,
    validation_seed: int,
    x_bounds: Sequence[float],
    y_bounds: Sequence[float],
    z_bounds: Sequence[float],
    min_xy_distance: float,
) -> dict[str, Any]:
    """Override only the EE pose; object scene, cameras, and task stay intact."""

    import torch

    low_dim = backend.low_dim_observations()
    target_slot = int(reset.task_state.target_slots[0].item())
    target = (
        low_dim.object_positions[0, target_slot].detach().cpu().numpy()
    )
    sample_seed = int(validation_seed) + int(round_index) * 100_003 + 70_001
    xy = _sample_random_start_xy(
        target_xy=target[:2],
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        min_distance=min_xy_distance,
        seed=sample_seed,
    )
    rng = np.random.default_rng(sample_seed + 1)
    z_low, z_high = (float(z_bounds[0]), float(z_bounds[1]))
    if z_low >= z_high:
        raise ValueError("Random-start Z bounds must be increasing.")
    z = float(rng.uniform(z_low, z_high))
    yaw = float(rng.uniform(-math.pi, math.pi))
    position = torch.tensor(
        (float(xy[0]), float(xy[1]), z),
        dtype=torch.float32,
        device=backend.device,
    )
    backend.set_end_effector_poses(
        position[None, :].repeat(GROUP_SIZE, 1),
        torch.full(
            (GROUP_SIZE,), yaw, dtype=torch.float32, device=backend.device
        ),
    )
    backend.set_gripper_openings(
        torch.ones(
            (GROUP_SIZE,), dtype=torch.float32, device=backend.device
        )
    )
    return {
        "seed": sample_seed,
        "position": position.detach().cpu().numpy(),
        "yaw": yaw,
        "target": target,
    }


def _camera_frame(value: Any, world_index: int = 0) -> np.ndarray:
    array = (
        value[int(world_index)]
        .permute(1, 2, 0)
        .detach()
        .to(dtype=value.dtype)
        .cpu()
        .numpy()
    )
    return np.clip(np.rint(array * 255.0), 0.0, 255.0).astype(np.uint8)


def _vector(values: Iterable[float], digits: int = 3) -> str:
    return " ".join(f"{float(value):+.{digits}f}" for value in values)


def _telemetry_lines(telemetry: Mapping[str, Any]) -> list[str]:
    action = np.asarray(
        telemetry.get("action", np.zeros(5)), dtype=np.float64
    ).reshape(5)
    applied = np.asarray(
        telemetry.get("applied", np.zeros(5)), dtype=np.float64
    ).reshape(5)
    ee = np.asarray(telemetry["ee"], dtype=np.float64).reshape(3)
    target = np.asarray(telemetry["target"], dtype=np.float64).reshape(3)
    delta = target - ee
    status = "PASS" if bool(telemetry.get("strict_success", False)) else "FAIL"
    return [
        (
            f"{telemetry['instruction']} | catalog={telemetry['catalog']} | "
            f"episode={int(telemetry['episode'])}"
        ),
        (
            f"reset={telemetry['initialization_mode']} shell="
            f"{int(telemetry['shell'])} training_horizon="
            f"{int(telemetry['training_horizon'])}"
        ),
        (
            f"policy_call={int(telemetry['policy_call'])}/"
            f"{int(telemetry['evaluation_horizon'])} "
            f"chunk_action={int(telemetry['chunk_action'])} "
            f"env_action={int(telemetry['action_step'])}"
        ),
        f"normalized action [{_vector(action, 2)}]",
        (
            f"executed delta xyz=[{_vector(applied[:3], 4)}] m "
            f"yaw={applied[3]:+.4f} rad grip={applied[4]:+.4f}"
        ),
        f"ee_xyz=[{_vector(ee)}] target_xyz=[{_vector(target)}]",
        (
            f"target-ee=[{_vector(delta)}] "
            f"xy_distance={float(telemetry['distance']):.4f} m"
        ),
        (
            f"start={float(telemetry['start_distance']):.4f} m "
            f"best={float(telemetry['best_distance']):.4f} m "
            f"strict<={float(telemetry['success_distance']):.4f} m: {status}"
        ),
        (
            f"dense_reward={float(telemetry['reward']):.6f} "
            f"best_reward={float(telemetry['best_reward']):.6f} "
            f"gripper={float(telemetry['gripper']):.3f}"
        ),
    ]


def _font(size: int) -> Any:
    for name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotated_frame(
    overview: np.ndarray,
    wrist: np.ndarray,
    telemetry: Mapping[str, Any],
) -> np.ndarray:
    combined = np.concatenate(
        (
            np.asarray(overview, dtype=np.uint8),
            np.asarray(wrist, dtype=np.uint8),
        ),
        axis=1,
    )
    camera = Image.fromarray(combined).resize(
        (combined.shape[1] * 2, combined.shape[0] * 2),
        Image.Resampling.BILINEAR,
    )
    panel_height = 260
    image = Image.new(
        "RGB", (camera.width, camera.height + panel_height), (13, 17, 24)
    )
    image.paste(camera, (0, 0))
    draw = ImageDraw.Draw(image)
    font = _font(16)
    max_chars = max(36, int((image.width - 24) / 9.5))
    lines: list[str] = []
    for line in _telemetry_lines(telemetry):
        lines.extend(
            textwrap.wrap(
                line,
                width=max_chars,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    for index, line in enumerate(lines):
        color = (
            (255, 210, 94)
            if "normalized action" in line
            else (117, 235, 163)
            if "PASS" in line
            else (238, 241, 245)
        )
        draw.text(
            (12, camera.height + 10 + index * 23),
            line,
            fill=color,
            font=font,
        )
    return np.asarray(image, dtype=np.uint8)


class _FFmpegWriter:
    def __init__(self, path: Path, *, fps: float, shape: Sequence[int]) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to encode telemetry MP4s.")
        height, width = int(shape[0]), int(shape[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{float(fps):.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable.")
        try:
            self.process.stdin.write(
                np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
            )
        except BrokenPipeError as exc:
            details = (
                self.process.stderr.read().decode("utf-8", errors="replace")
                if self.process.stderr is not None
                else ""
            )
            raise RuntimeError(f"ffmpeg failed while writing {self.path}: {details}") from exc

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        details = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        code = self.process.wait()
        if code:
            raise RuntimeError(
                f"ffmpeg exited with status {code} for {self.path}: {details}"
            )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tolerance_rows(
    episodes: Sequence[Mapping[str, Any]],
    tolerances: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets = sorted({str(row["target_catalog"]) for row in episodes})
    for tolerance in sorted({float(value) for value in tolerances}):
        for target in ("all", *targets):
            selected = [
                row
                for row in episodes
                if target == "all" or str(row["target_catalog"]) == target
            ]
            successes = sum(
                float(row["best_xy_distance_m"]) <= tolerance
                for row in selected
            )
            rows.append(
                {
                    "target_catalog": target,
                    "tolerance_m": tolerance,
                    "episodes": len(selected),
                    "successes": successes,
                    "success_rate": (
                        float(successes) / len(selected) if selected else math.nan
                    ),
                }
            )
    return rows


def _summary_markdown(
    *,
    checkpoint: Path,
    shell: int,
    success_distance: float,
    initialization_mode: str,
    min_policy_inferences: int,
    min_video_seconds: float,
    episodes: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
) -> str:
    total = len(episodes)
    strict = sum(bool(row["strict_success"]) for row in episodes)
    lines = [
        "# SmolVLA MJWarp qualitative move-to evaluation",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Backend: exact `mjlab_mjwarp` training backend",
        f"- EE initialization: `{initialization_mode}`",
        f"- Checkpoint move-to Reverse Frontier shell (provenance): `{shell}`",
        f"- Minimum policy inferences per episode: `{min_policy_inferences}`",
        f"- Minimum video duration: `{min_video_seconds:.1f} s`",
        f"- Strict XY success threshold: `{success_distance:.4f} m`",
        f"- Strict successes: `{strict}/{total}` ({strict / max(total, 1):.1%})",
        "",
        "Each MP4 shows the exact overview and wrist observations plus policy, "
        "controller, geometry, reward, and strict-predicate telemetry. The "
        "recorder continues after first success so later closed-loop behavior "
        "remains visible.",
        "",
        (
            "Random workspace initialization and the extended minimum horizon "
            "are a qualitative generalization audit. They are intentionally "
            "harder than the checkpoint's Reverse Frontier validation reset "
            "and must not be reported as the original validation metric."
            if initialization_mode == "random_workspace"
            else
            "This run retains the Reverse Frontier initialization but extends "
            "the rollout horizon for qualitative inspection."
        ),
        "",
        "## Per-object results",
        "",
        "| object | episodes | strict success | mean best XY | <=5 cm |",
        "|---|---:|---:|---:|---:|",
    ]
    targets = sorted({str(row["target_catalog"]) for row in episodes})
    for target in targets:
        selected = [
            row for row in episodes if str(row["target_catalog"]) == target
        ]
        target_strict = sum(bool(row["strict_success"]) for row in selected)
        mean_best = float(
            np.mean([float(row["best_xy_distance_m"]) for row in selected])
        )
        loose = sum(
            float(row["best_xy_distance_m"]) <= 0.05 for row in selected
        )
        lines.append(
            f"| {target} | {len(selected)} | "
            f"{target_strict / max(len(selected), 1):.1%} | "
            f"{mean_best:.4f} m | {loose / max(len(selected), 1):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Tolerance sweep",
            "",
            "| XY tolerance | success rate |",
            "|---:|---:|",
        ]
    )
    for row in threshold_rows:
        if row["target_catalog"] == "all":
            lines.append(
                f"| {float(row['tolerance_m']):.4f} m | "
                f"{float(row['success_rate']):.1%} |"
            )
    lines.extend(
        [
            "",
            "This is a qualitative checkpoint audit, not a replacement for a "
            "large held-out benchmark. Review whether failures move toward the "
            "named object and whether best distance clusters just outside the "
            "strict 2 cm window.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_episode(
    *,
    backend: Any,
    runtime: Any,
    trainer: Any,
    resetter: Any,
    reward_config: Any,
    round_index: int,
    episode_index: int,
    success_distance: float,
    validation_seed: int,
    random_start: bool,
    random_start_x_bounds: Sequence[float],
    random_start_y_bounds: Sequence[float],
    random_start_z_bounds: Sequence[float],
    random_start_min_xy_distance: float,
    min_policy_inferences: int,
    min_video_seconds: float,
    fps: float,
    terminal_hold_seconds: float,
    video_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
        BatchedTaskThresholds,
        build_smolvla_state_tensor,
        evaluate_active_sparse_tasks,
    )
    from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
        ACTIVE_CDPR_CATALOGS,
        OBJECT_VARIANTS,
    )

    reset = resetter.reset(update_index=0, round_index=int(round_index))
    target_catalog_id = int(reset.group_target_catalog_ids[0].item())
    target_catalog = ACTIVE_CDPR_CATALOGS[target_catalog_id]
    target_label = OBJECT_VARIANTS[target_catalog].label
    instruction = str(reset.instructions[0])
    shell = int(reset.group_shell_ids[0].item())
    training_horizon = int(reset.horizons[0].item())
    evaluation_horizon = max(
        training_horizon, int(min_policy_inferences)
    )
    target_slot = int(reset.task_state.target_slots[0].item())

    random_start_state: dict[str, Any] | None = None
    if random_start:
        random_start_state = _apply_random_ee_start(
            backend=backend,
            reset=reset,
            round_index=round_index,
            validation_seed=validation_seed,
            x_bounds=random_start_x_bounds,
            y_bounds=random_start_y_bounds,
            z_bounds=random_start_z_bounds,
            min_xy_distance=random_start_min_xy_distance,
        )
    initialization_mode = (
        "random_workspace" if random_start else "reverse_frontier"
    )
    active = torch.ones(
        (GROUP_SIZE,), dtype=torch.bool, device=backend.device
    )
    initial_low = backend.low_dim_observations()
    initial_target = initial_low.object_positions[0, target_slot]
    initial_ee_np = initial_low.ee_position[0].detach().cpu().numpy()
    initial_target_np = initial_target.detach().cpu().numpy()
    start_distance = float(
        torch.linalg.vector_norm(
            initial_target[:2] - initial_low.ee_position[0, :2]
        ).item()
    )
    best_distance = start_distance
    final_distance = start_distance
    best_reward = float("-inf")
    final_reward = 0.0
    strict_success = False
    first_success_action_step: int | None = None
    action_rows: list[dict[str, Any]] = []
    action_step = 0
    policy_inference_count = 0

    def _env_on(name: str) -> bool:
        return os.environ.get(name, "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    _zero_overview = _env_on("RLVLA_EVAL_ZERO_OVERVIEW")
    _zero_wrist = _env_on("RLVLA_EVAL_ZERO_WRIST")

    cameras = backend.render_policy_cameras()
    base_telemetry = {
        "instruction": instruction,
        "catalog": target_catalog,
        "episode": episode_index,
        "shell": shell,
        "initialization_mode": initialization_mode,
        "training_horizon": training_horizon,
        "evaluation_horizon": evaluation_horizon,
        "policy_call": 0,
        "chunk_action": -1,
        "action_step": 0,
        "action": np.zeros(5),
        "applied": np.zeros(5),
        "ee": initial_ee_np,
        "target": initial_target_np,
        "distance": start_distance,
        "start_distance": start_distance,
        "best_distance": best_distance,
        "success_distance": success_distance,
        "strict_success": False,
        "reward": 0.0,
        "best_reward": 0.0,
        "gripper": float(initial_low.gripper_opening[0].item()),
    }
    frame = _annotated_frame(
        _camera_frame(cameras.overview),
        _camera_frame(cameras.wrist),
        base_telemetry,
    )
    writer = _FFmpegWriter(video_path, fps=fps, shape=frame.shape)
    writer.write(frame)
    last_frame = frame
    frame_count = 1
    try:
        with torch.inference_mode():
            for decision in range(evaluation_horizon):
                if decision > 0:
                    cameras = backend.render_policy_cameras()
                low_dim = backend.low_dim_observations()
                state = build_smolvla_state_tensor(
                    ee_position=low_dim.ee_position,
                    ee_yaw=low_dim.ee_yaw,
                    gripper_opening=low_dim.gripper_opening,
                    object_positions=low_dim.object_positions,
                    target_slots=reset.task_state.target_slots,
                    state_dim=int(trainer.state_dim),
                )
                # Vision-reliance ablation: optionally blank a camera in the
                # policy's observation only (the recorded video keeps the real
                # frames). RLVLA_EVAL_ZERO_OVERVIEW / RLVLA_EVAL_ZERO_WRIST=1.
                policy_overview = (
                    torch.zeros_like(cameras.overview)
                    if _zero_overview
                    else cameras.overview
                )
                policy_wrist = (
                    torch.zeros_like(cameras.wrist)
                    if _zero_wrist
                    else cameras.wrist
                )
                prior = runtime.sample_cdpr_chunks_from_tensors(
                    primary_images=policy_overview,
                    wrist_images=policy_wrist,
                    states=state,
                    instructions=reset.instructions,
                    microbatch_size=int(
                        trainer.args.smolvla_inference_microbatch_size
                    ),
                )
                actions = trainer.deterministic_action_chunks_tensor(
                    states=state,
                    priors=prior,
                    action_count=int(trainer.args.replan_every),
                )
                policy_inference_count += 1
                for chunk_action in range(int(actions.shape[1])):
                    step_active = active
                    normalized = actions[:, chunk_action]
                    low_dim = backend.step(normalized, step_active)
                    result = evaluate_active_sparse_tasks(
                        state=reset.task_state,
                        ee_position=low_dim.ee_position,
                        object_positions=low_dim.object_positions,
                        gripper_opening=low_dim.gripper_opening,
                        caught_target=torch.zeros_like(active),
                        active_mask=step_active,
                        max_steps=10_000,
                        thresholds=BatchedTaskThresholds(
                            move_to_xy_low=float(reward_config.xy_window_low),
                            move_to_xy=float(success_distance),
                        ),
                        move_to_distance_reward=reward_config,
                    )
                    action_step += 1
                    final_distance = float(
                        result.diagnostics["ee_xy_distance"][0].item()
                    )
                    best_distance = min(best_distance, final_distance)
                    final_reward = float(result.rewards[0].item())
                    best_reward = max(best_reward, final_reward)
                    success_this_step = bool(result.success[0].item())
                    if success_this_step and first_success_action_step is None:
                        first_success_action_step = action_step
                    strict_success = bool(
                        strict_success or success_this_step
                    )
                    action_np = normalized[0].detach().cpu().numpy()
                    applied_np = action_np * np.asarray(
                        (
                            trainer.args.action_step_xyz,
                            trainer.args.action_step_xyz,
                            trainer.args.action_step_xyz,
                            trainer.args.action_step_yaw,
                            trainer.args.action_step_gripper,
                        ),
                        dtype=np.float32,
                    )
                    ee_np = low_dim.ee_position[0].detach().cpu().numpy()
                    target_np = (
                        low_dim.object_positions[0, target_slot]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    action_rows.append(
                        {
                            "episode": episode_index,
                            "target_catalog": target_catalog,
                            "target_label": target_label,
                            "instruction": instruction,
                            "shell": shell,
                            "initialization_mode": initialization_mode,
                            "training_horizon_policy_inferences": training_horizon,
                            "evaluation_horizon_policy_inferences": evaluation_horizon,
                            "policy_inference": policy_inference_count,
                            "chunk_action": chunk_action,
                            "action_step": action_step,
                            **{
                                f"action_{name}": float(action_np[index])
                                for index, name in enumerate(ACTION_NAMES)
                            },
                            **{
                                f"applied_{name}": float(applied_np[index])
                                for index, name in enumerate(ACTION_NAMES)
                            },
                            "ee_x": float(ee_np[0]),
                            "ee_y": float(ee_np[1]),
                            "ee_z": float(ee_np[2]),
                            "target_x": float(target_np[0]),
                            "target_y": float(target_np[1]),
                            "target_z": float(target_np[2]),
                            "xy_distance_m": final_distance,
                            "best_xy_distance_m": best_distance,
                            "dense_reward": final_reward,
                            "strict_success_this_step": success_this_step,
                            "strict_success": strict_success,
                        }
                    )
                    telemetry = {
                        **base_telemetry,
                        "policy_call": policy_inference_count,
                        "chunk_action": chunk_action,
                        "action_step": action_step,
                        "action": action_np,
                        "applied": applied_np,
                        "ee": ee_np,
                        "target": target_np,
                        "distance": final_distance,
                        "best_distance": best_distance,
                        "strict_success": strict_success,
                        "reward": final_reward,
                        "best_reward": best_reward,
                        "gripper": float(low_dim.gripper_opening[0].item()),
                    }
                    rendered = backend.render_policy_cameras()
                    last_frame = _annotated_frame(
                        _camera_frame(rendered.overview),
                        _camera_frame(rendered.wrist),
                        telemetry,
                    )
                    writer.write(last_frame)
                    frame_count += 1
                    # This qualitative audit deliberately continues after the
                    # first strict success so every clip contains the requested
                    # number of fresh closed-loop policy calls.
        terminal_hold_frames = max(
            0, int(round(fps * terminal_hold_seconds))
        )
        minimum_total_frames = max(
            1, int(math.ceil(fps * min_video_seconds))
        )
        hold_frames = max(
            terminal_hold_frames, minimum_total_frames - frame_count
        )
        for _ in range(hold_frames):
            writer.write(last_frame)
            frame_count += 1
    finally:
        writer.close()

    episode = {
        "episode": episode_index,
        "round_index": round_index,
        "target_catalog": target_catalog,
        "target_label": target_label,
        "instruction": instruction,
        "curriculum_shell": shell,
        "initialization_mode": initialization_mode,
        "random_start_seed": (
            None if random_start_state is None else random_start_state["seed"]
        ),
        "initial_ee_yaw": (
            None if random_start_state is None else random_start_state["yaw"]
        ),
        "training_horizon_policy_inferences": training_horizon,
        "evaluation_horizon_policy_inferences": evaluation_horizon,
        "executed_policy_inferences": policy_inference_count,
        "executed_environment_actions": action_step,
        "initial_ee_x": float(initial_ee_np[0]),
        "initial_ee_y": float(initial_ee_np[1]),
        "initial_ee_z": float(initial_ee_np[2]),
        "initial_target_x": float(initial_target_np[0]),
        "initial_target_y": float(initial_target_np[1]),
        "initial_target_z": float(initial_target_np[2]),
        "initial_xy_distance_m": start_distance,
        "final_xy_distance_m": final_distance,
        "best_xy_distance_m": best_distance,
        "strict_success_distance_m": success_distance,
        "strict_success": strict_success,
        "first_success_action_step": first_success_action_step,
        "final_dense_reward": final_reward,
        "best_dense_reward": (
            best_reward if math.isfinite(best_reward) else final_reward
        ),
        "video_frames": frame_count,
        "video_duration_seconds": frame_count / float(fps),
        "video": str(video_path),
    }
    return episode, action_rows


def _run_vision_sensitivity_probe(
    *,
    backend: Any,
    runtime: Any,
    trainer: Any,
    resetter: Any,
    num_rounds: int,
    min_scene_objects: int,
    max_scene_objects: int,
    run_dir: Path,
) -> int:
    """Counterfactual test of whether the policy actually uses the cameras.

    The proprioceptive state is the ONLY non-vision input and the target never
    enters it (pure [ee_xyz, ee_yaw, gripper, 0]), so the target can only reach
    the action through SmolVLA's image encoding -> prior. We hold the state
    fixed at a canonical pose and vary only the images across many scenes:

      * real vs zeroed images -> if the action barely changes, vision is unused
        (cos ~ 1, |dA|/|A| ~ 0).
      * pairwise across different-target scenes -> if the action is ~identical
        (cos ~ 1) despite the target moving, the output is target-invariant.
      * grounding: cos(first-step XY action, direction to the true target) -> is
        any vision influence in the RIGHT direction (~+1) or task-blind (~0).

    All measured for the composed policy action AND the frozen prior alone, so we
    can tell whether blindness is in SmolVLA or in the residual composition.
    """

    import torch
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
        build_smolvla_state_tensor,
    )

    def _cos(a: "torch.Tensor", b: "torch.Tensor") -> float:
        return float(
            torch.nn.functional.cosine_similarity(
                a.reshape(1, -1), b.reshape(1, -1), dim=-1
            ).item()
        )

    worlds = int(GROUP_SIZE)
    device = next(trainer.actor.parameters()).device
    # Canonical fixed proprioceptive state: workspace centre, mid height.
    ee_const = torch.tensor([0.0, 0.0, 0.40], device=device).expand(worlds, 3)
    yaw_const = torch.zeros(worlds, device=device)
    grip_const = torch.zeros(worlds, device=device)
    seed = 1234
    action_count = max(1, int(trainer.args.replan_every))
    mb = int(trainer.args.smolvla_inference_microbatch_size)

    a_real: list[Any] = []
    a_zero: list[Any] = []
    p_real: list[Any] = []
    p_zero: list[Any] = []
    tgt_xy: list[Any] = []

    def _prior_action(overview: Any, wrist: Any, state: Any):
        # Fix the flow-matching noise so differences are attributable to images.
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        prior = runtime.sample_cdpr_chunks_from_tensors(
            primary_images=overview,
            wrist_images=wrist,
            states=state,
            instructions=reset.instructions,
            microbatch_size=mb,
        )
        act = trainer.deterministic_action_chunks_tensor(
            states=state, priors=prior, action_count=action_count
        )
        return prior, act

    # Capture SmolVLA's connector output ([B, 16, 960] per camera) -- the actual
    # vision->LM feature the vision-aware residual would tap. If a linear probe
    # cannot recover the target position from this frozen feature, feeding it to
    # the residual cannot work either.
    connector = None
    try:
        connector = runtime.policy.model.vlm_with_expert.vlm.model.connector
    except AttributeError:
        print("[vision-probe] connector module not found; skipping feature probe")
    captured: list[Any] = []
    hook_handle = (
        connector.register_forward_hook(
            lambda _m, _i, out: captured.append(out.detach().float().cpu())
        )
        if connector is not None
        else None
    )
    vfeat_joint: list[Any] = []
    vfeat_overview: list[Any] = []

    with torch.inference_mode():
        for round_index in range(int(num_rounds)):
            count = _scene_object_count_for_round(
                round_index, int(min_scene_objects), int(max_scene_objects)
            )
            resetter.set_scene_object_range(count, count)
            reset = resetter.reset(update_index=0, round_index=int(round_index))
            cameras = backend.render_policy_cameras()
            low_dim = backend.low_dim_observations()
            state = build_smolvla_state_tensor(
                ee_position=ee_const,
                ee_yaw=yaw_const,
                gripper_opening=grip_const,
                object_positions=low_dim.object_positions,
                target_slots=reset.task_state.target_slots,
                state_dim=int(trainer.state_dim),
            )
            rows = torch.arange(worlds, device=device)
            target = low_dim.object_positions[rows, reset.task_state.target_slots]
            tgt_xy.append((target[:, :2] - ee_const[:, :2])[0].detach().cpu())

            captured.clear()
            prior_r, act_r = _prior_action(cameras.overview, cameras.wrist, state)
            if hook_handle is not None and captured:
                # captured: one [B, 16, 960] per camera (overview, wrist, aux).
                # The 16 tokens are a ~4x4 spatial grid, so WHERE the object is
                # lives in WHICH token carries it. Keep the overview tokens
                # un-pooled and flattened (spatial probe) AND a mean-pooled copy
                # (the pooling destroys position, so this is the pessimistic
                # baseline).
                cams = torch.stack([c[0] for c in captured], dim=0)
                vfeat_joint.append(cams.mean(dim=(0, 1)).numpy())
                vfeat_overview.append(captured[0][0].reshape(-1).numpy())
            prior_z, act_z = _prior_action(
                torch.zeros_like(cameras.overview),
                torch.zeros_like(cameras.wrist),
                state,
            )
            a_real.append(act_r[0, 0].detach().cpu())
            a_zero.append(act_z[0, 0].detach().cpu())
            p_real.append(prior_r[0, 0].detach().cpu())
            p_zero.append(prior_z[0, 0].detach().cpu())

    if hook_handle is not None:
        hook_handle.remove()

    def _ridge_heldout_r2(feats: list[Any]) -> "tuple[float, float, float]":
        """Held-out R^2 of a linear map feature -> target XY, with a shuffled
        control, sweeping the ridge alpha and reporting the setting where the
        shuffled control is best-calibrated to ~0 (so D>>N overfitting cannot
        masquerade as signal). Uses the dual (kernel) ridge form so the cost is
        O(N^3), not O(D^3) -- essential for the 15360-dim spatial feature.
        Returns (real_R2, shuffled_R2, alpha)."""

        if len(feats) < 16:
            return float("nan"), float("nan"), 0.0
        F = np.stack(feats).astype(np.float64)
        T = np.stack([t.numpy() for t in tgt_xy]).astype(np.float64)
        m = len(F)
        idx = np.random.RandomState(0).permutation(m)
        tr, te = idx[: m // 2], idx[m // 2 :]
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        Xtr = (F[tr] - mu) / sd
        Xte = (F[te] - mu) / sd
        # Gram matrices (+1 folds in an untuned bias). Dual ridge:
        # pred_te = K_te,tr (K_tr,tr + alpha I)^-1 Y_tr.
        K_tr = Xtr @ Xtr.T + 1.0
        K_te = Xte @ Xtr.T + 1.0
        eye = np.eye(K_tr.shape[0])
        T_shuf = T[np.random.RandomState(1).permutation(m)]

        def fit_r2(Tm: Any, alpha: float) -> float:
            dual = np.linalg.solve(K_tr + alpha * eye, Tm[tr])
            pred = K_te @ dual
            ss_res = ((Tm[te] - pred) ** 2).sum(0)
            ss_tot = ((Tm[te] - Tm[te].mean(0)) ** 2).sum(0) + 1e-9
            return float((1.0 - ss_res / ss_tot).mean())

        # Pick the LEAST regularization whose shuffled control is calibrated to
        # ~0 (|shuf| <= 0.05): that is the most sensitive setting to real signal.
        # Over-regularizing crushes real and shuffled alike, hiding signal.
        alphas = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
        fallback = None
        for alpha in alphas:
            shuf = fit_r2(T_shuf, alpha)
            if fallback is None or abs(shuf) < abs(fallback[1]):
                fallback = (fit_r2(T, alpha), shuf, alpha)
            if abs(shuf) <= 0.05:
                return fit_r2(T, alpha), shuf, alpha
        return fallback if fallback is not None else (float("nan"), float("nan"), 0.0)

    joint_r2, joint_r2_shuf, joint_alpha = (
        _ridge_heldout_r2(vfeat_joint) if vfeat_joint else (float("nan"), float("nan"), 0.0)
    )
    overview_r2, overview_r2_shuf, overview_alpha = (
        _ridge_heldout_r2(vfeat_overview) if vfeat_overview else (float("nan"), float("nan"), 0.0)
    )

    n = len(a_real)
    use_action = np.mean([_cos(a_real[i], a_zero[i]) for i in range(n)])
    use_prior = np.mean([_cos(p_real[i], p_zero[i]) for i in range(n)])
    delta_action = np.mean([
        float((a_real[i] - a_zero[i]).norm() / (a_real[i].norm() + 1e-6))
        for i in range(n)
    ])
    ground_action = np.mean([_cos(a_real[i][:2], tgt_xy[i]) for i in range(n)])
    ground_prior = np.mean([_cos(p_real[i][:2], tgt_xy[i]) for i in range(n)])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    blind_action = np.mean([_cos(a_real[i], a_real[j]) for i, j in pairs])
    blind_prior = np.mean([_cos(p_real[i], p_real[j]) for i, j in pairs])

    report = textwrap.dedent(
        f"""
        ===== Vision-sensitivity probe ({n} scenes, fixed canonical state) =====
        cos(action | real vs zeroed images)   = {use_action:+.3f}   (~1 => images ignored)
        |dAction|/|Action| (real vs zeroed)   = {delta_action:.3f}    (~0 => images ignored)
        cos(prior  | real vs zeroed images)   = {use_prior:+.3f}   (isolates frozen SmolVLA)

        mean pairwise cos(action) across scenes= {blind_action:+.3f}   (~1 => target-invariant/blind)
        mean pairwise cos(prior)  across scenes= {blind_prior:+.3f}

        grounding cos(action_xy, target_dir)  = {ground_action:+.3f}   (+1 correct, 0 blind, - anti)
        grounding cos(prior_xy,  target_dir)  = {ground_prior:+.3f}

        --- Can the FROZEN connector feature be grounded? (linear probe R^2) ---
        pooled feature -> target XY   R^2 = {joint_r2:+.3f}  (shuf {joint_r2_shuf:+.3f}, a={joint_alpha:g})
        overview SPATIAL tokens -> XY R^2 = {overview_r2:+.3f}  (shuf {overview_r2_shuf:+.3f}, a={overview_alpha:g})
        =========================================================================
        Read: if the top block is ~1 / ~0, SmolVLA is not using the cameras at all
        (grounding wall or an image-plumbing bug). If it clearly uses vision
        (cos < ~0.9, |dA| > ~0.1) but grounding stays ~0, vision is read but not
        localized. The SPATIAL-tokens R^2 is the linchpin for the vision-aware
        residual (the pooled row is a pessimistic baseline -- pooling erases the
        4x4 token grid where position lives). With the shuffled control
        calibrated to ~0 by the alpha sweep, a clearly positive spatial R^2 means
        the frozen feature DOES encode target position, so a trainable head
        reading the token grid can ground --> wire it into the residual. R^2 ~ 0
        (== shuffled) means the frozen feature does not localize the object, so no
        head reading it can help (the vision encoder itself would have to adapt).
        Run with many scenes (EPISODES_PER_TARGET high) so the split is stable.
        """
    ).strip()
    print(report, flush=True)
    out = run_dir / "vision_sensitivity_probe.txt"
    out.write_text(report + "\n")
    print(f"[vision-probe] wrote {out}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.episodes_per_target < 1:
        parser.error("--episodes-per-target must be positive.")
    if args.success_distance <= 0.0:
        parser.error("--success-distance must be positive.")
    if args.fps <= 0.0:
        parser.error("--fps must be positive.")
    if args.terminal_hold_seconds < 0.0:
        parser.error("--terminal-hold-seconds cannot be negative.")
    if args.min_policy_inferences < 10:
        parser.error("--min-policy-inferences must be at least 10.")
    if args.min_video_seconds < 5.0:
        parser.error("--min-video-seconds must be at least 5.")
    if not 1 <= args.min_scene_objects <= 4:
        parser.error("--min-scene-objects must be in [1, 4].")
    if not args.min_scene_objects <= args.max_scene_objects <= 4:
        parser.error(
            "--max-scene-objects must be between --min-scene-objects and 4."
        )
    if args.random_start_min_xy_distance < 0.0:
        parser.error("--random-start-min-xy-distance cannot be negative.")
    if not (
        float(args.random_start_x_bounds[0])
        < float(args.random_start_x_bounds[1])
    ):
        parser.error("--random-start-x-bounds must be increasing.")
    if not (
        float(args.random_start_y_bounds[0])
        < float(args.random_start_y_bounds[1])
    ):
        parser.error("--random-start-y-bounds must be increasing.")
    if not (
        float(args.random_start_z_bounds[0])
        < float(args.random_start_z_bounds[1])
    ):
        parser.error("--random-start-z-bounds must be increasing.")

    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint_path}")
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from rl_vla_bootstrapping.core.config import load_project_config
    from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
        BatchedReverseFrontierResetter,
        RankLocalCurriculum,
    )
    from rl_vla_bootstrapping.policy.rank_local_grpo import RankLocalGroupLayout
    from rl_vla_bootstrapping.policy.smolvla_cdpr import load_smolvla_runtime
    from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
        SmolVLAGRPOTrainer,
    )
    from rl_vla_bootstrapping.simulation.cdpr_backend import (
        CDPRBackendConfig,
        create_cdpr_backend,
    )
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
        BatchedMoveToDistanceReward,
        INSTRUCTION_TO_ID,
    )
    from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
        ACTIVE_CDPR_CATALOGS,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Exact MJWarp checkpoint evaluation requires a CUDA GPU in the "
            "pinned cdpr-mjlab environment."
        )
    device = torch.device(str(args.device))
    project = load_project_config(config_path)
    if str(project.simulator.backend) != "mjlab_mjwarp":
        raise ValueError(
            f"Expected an mjlab_mjwarp config, got {project.simulator.backend!r}."
        )
    xml_path = project.resolve_path(project.simulator.fixed_scene_xml)
    if xml_path is None:
        raise ValueError("The config does not define simulator.fixed_scene_xml.")
    allowed_objects = tuple(
        str(value) for value in (project.task.target_objects or ())
    )
    if not allowed_objects:
        raise ValueError("The config does not define task.target_objects.")
    unknown = sorted(set(allowed_objects).difference(ACTIVE_CDPR_CATALOGS))
    if unknown:
        raise ValueError(f"Unsupported MJWarp target catalogs: {unknown}")

    payload = _load_torch_checkpoint(checkpoint_path)
    training_args = _evaluation_training_args(
        payload,
        config_path=config_path,
        xml_path=xml_path,
        device=str(device),
        render_width=int(project.simulator.render_width),
        render_height=int(project.simulator.render_height),
        allowed_objects=allowed_objects,
        microbatch_size=int(args.smolvla_microbatch_size),
    )
    layout = RankLocalGroupLayout(
        worlds_per_rank=GROUP_SIZE,
        groups_per_rank=1,
        group_size=GROUP_SIZE,
    )
    layout.validate()
    backend_config = CDPRBackendConfig(
        backend="mjlab_mjwarp",
        worlds_per_rank=GROUP_SIZE,
        groups_per_rank=1,
        grpo_group_size=GROUP_SIZE,
        hold_steps=int(training_args.hold_steps),
        action_step_xyz=float(training_args.action_step_xyz),
        action_step_yaw=float(training_args.action_step_yaw),
        action_step_gripper=float(training_args.action_step_gripper),
        lock_non_commanded_axes=bool(training_args.lock_non_commanded_axes),
        lock_non_commanded_axes_threshold=float(
            training_args.lock_non_commanded_axes_threshold
        ),
        render_width=int(training_args.render_width),
        render_height=int(training_args.render_height),
        object_slots=int(training_args.object_slots),
        nconmax=int(training_args.mjwarp_nconmax),
        njmax=int(training_args.mjwarp_njmax),
        nccdmax=training_args.mjwarp_nccdmax,
        device=str(device),
        xml_path=xml_path,
    )

    print(
        "[smolvla-mjwarp-video] allocating 8 worlds / 1 complete "
            f"reproducibly seeded group on {device}",
        flush=True,
    )
    backend = create_cdpr_backend(backend_config)
    runtime = None
    try:
        _validate_simulator_compatibility(payload, backend.metadata())
        print(
            "[smolvla-mjwarp-video] loading frozen SmolVLA "
            f"{training_args.base_checkpoint}",
            flush=True,
        )
        runtime = load_smolvla_runtime(
            checkpoint=str(training_args.base_checkpoint),
            device=str(device),
            mixed_precision=str(training_args.mixed_precision),
            image_size=int(training_args.image_size),
            state_dim=int(training_args.state_dim),
            image_feature_keys=(
                None
                if training_args.image_feature_keys is None
                else tuple(training_args.image_feature_keys)
            ),
            include_wrist=bool(training_args.include_wrist),
            include_aux_camera=bool(training_args.include_aux_camera),
            mask_empty_aux_camera=bool(
                getattr(training_args, "mask_empty_aux_camera", False)
            ),
            chunk_size=int(training_args.chunk_size),
            action_dim=int(training_args.action_dim),
            action_indices=(
                None
                if training_args.smolvla_action_indices is None
                else tuple(
                    int(value)
                    for value in training_args.smolvla_action_indices
                )
            ),
            action_normalization=str(
                training_args.smolvla_action_normalization
            ),
            model_image_size=(
                None
                if int(training_args.smolvla_model_image_size) <= 0
                else int(training_args.smolvla_model_image_size)
            ),
            compile_model=False,
            compile_mode=str(training_args.smolvla_compile_mode),
        )
        trainer = SmolVLAGRPOTrainer(
            args=training_args,
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
            chunk_size=int(payload["chunk_size"]),
            run_dir=run_dir,
            device=device,
        )
        trainer._unwrap(trainer.actor).load_state_dict(payload["policy"])
        trainer.actor.eval()

        curriculum = RankLocalCurriculum(
            device=device,
            promotion_success=float(
                training_args.reverse_frontier_promotion_success
            ),
            demotion_success=float(
                training_args.reverse_frontier_demotion_success
            ),
            validation_rollouts_per_shell=int(
                training_args.reverse_frontier_validation_episodes
            ),
            min_updates=int(training_args.reverse_frontier_min_train_updates),
            saturation_abort_threshold=float(
                training_args.reverse_frontier_saturation_abort_threshold
            ),
        )
        extra_state = dict(payload.get("extra_state") or {})
        curriculum_state = extra_state.get("curriculum")
        if not isinstance(curriculum_state, Mapping):
            curriculum_state = extra_state.get("complex_runtime")
        if isinstance(curriculum_state, Mapping):
            curriculum.restore(curriculum_state)
        if args.curriculum_shell is not None:
            move_index = INSTRUCTION_TO_ID["move_to_object"]
            max_shell = int(curriculum._shell_max[move_index].item())
            if not 0 <= int(args.curriculum_shell) <= max_shell:
                raise ValueError(
                    f"--curriculum-shell must be in [0, {max_shell}] for "
                    "move_to_object."
                )
            curriculum.current_shell[move_index] = int(args.curriculum_shell)
        move_shell = int(
            curriculum.current_shell[
                INSTRUCTION_TO_ID["move_to_object"]
            ].item()
        )
        metadata = dict(project.task.metadata or {})
        reward_config = BatchedMoveToDistanceReward.from_metadata(metadata)
        resetter = BatchedReverseFrontierResetter(
            backend=backend,
            layout=layout,
            curriculum=curriculum,
            rank=0,
            base_seed=int(args.validation_seed),
            instruction_types=("move_to_object",),
            allowed_objects=allowed_objects,
            frontier_probability=1.0,
            rehearsal_probability=0.0,
            balanced_target_catalogs=True,
            task_metadata={
                "min_scene_objects": int(args.min_scene_objects),
                "max_scene_objects": int(args.max_scene_objects),
            },
        )

        videos_dir = run_dir / "videos"
        episode_rows: list[dict[str, Any]] = []
        action_rows: list[dict[str, Any]] = []
        counts: defaultdict[str, int] = defaultdict(int)
        total_requested = len(allowed_objects) * int(args.episodes_per_target)
        audit_initialization_mode = (
            "random_workspace"
            if bool(args.random_start)
            else "reverse_frontier"
        )
        print(
            "[smolvla-mjwarp-video] qualitative "
            f"{audit_initialization_mode} audit: "
            f"{total_requested} reset(s), checkpoint move-to shell="
            f"{move_shell}, min_policy_inferences="
            f"{args.min_policy_inferences}, min_video="
            f"{args.min_video_seconds:.1f}s, strict_xy<="
            f"{args.success_distance:.4f}m, scene_objects="
            f"{args.min_scene_objects}-{args.max_scene_objects}",
            flush=True,
        )
        # Match the training validator: reset sampling has its own generator,
        # while the frozen SmolVLA flow sampler consumes this CUDA RNG stream.
        torch.manual_seed(int(args.validation_seed))
        torch.cuda.manual_seed(int(args.validation_seed))
        if os.environ.get("RLVLA_EVAL_VISION_PROBE", "").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            return _run_vision_sensitivity_probe(
                backend=backend,
                runtime=runtime,
                trainer=trainer,
                resetter=resetter,
                num_rounds=total_requested,
                min_scene_objects=int(args.min_scene_objects),
                max_scene_objects=int(args.max_scene_objects),
                run_dir=run_dir,
            )
        for round_index in range(total_requested):
            scene_object_count = _scene_object_count_for_round(
                round_index,
                int(args.min_scene_objects),
                int(args.max_scene_objects),
            )
            resetter.set_scene_object_range(
                scene_object_count, scene_object_count
            )
            target_catalog = allowed_objects[
                round_index % len(allowed_objects)
            ]
            target_episode = counts[target_catalog] + 1
            video_path = (
                videos_dir
                / target_catalog
                / f"{target_catalog}_episode_{target_episode:02d}.mp4"
            )
            episode, trace = _run_episode(
                backend=backend,
                runtime=runtime,
                trainer=trainer,
                resetter=resetter,
                reward_config=reward_config,
                round_index=round_index,
                episode_index=round_index + 1,
                success_distance=float(args.success_distance),
                validation_seed=int(args.validation_seed),
                random_start=bool(args.random_start),
                random_start_x_bounds=tuple(args.random_start_x_bounds),
                random_start_y_bounds=tuple(args.random_start_y_bounds),
                random_start_z_bounds=tuple(args.random_start_z_bounds),
                random_start_min_xy_distance=float(
                    args.random_start_min_xy_distance
                ),
                min_policy_inferences=int(args.min_policy_inferences),
                min_video_seconds=float(args.min_video_seconds),
                fps=float(args.fps),
                terminal_hold_seconds=float(args.terminal_hold_seconds),
                video_path=video_path,
            )
            actual_target = str(episode["target_catalog"])
            counts[actual_target] += 1
            episode["scene_object_count"] = scene_object_count
            for action_row in trace:
                action_row["scene_object_count"] = scene_object_count
            episode["target_episode"] = counts[actual_target]
            episode["video"] = str(
                Path(episode["video"]).relative_to(run_dir)
            )
            episode_rows.append(episode)
            action_rows.extend(trace)
            print(
                f"[{round_index + 1:02d}/{total_requested:02d}] "
                f"{episode['instruction']}: success={episode['strict_success']} "
                f"scene_objects={scene_object_count} "
                f"start={episode['initial_xy_distance_m']:.4f}m "
                f"best={episode['best_xy_distance_m']:.4f}m "
                f"policy_calls={episode['executed_policy_inferences']} "
                f"actions={episode['executed_environment_actions']} "
                f"duration={episode['video_duration_seconds']:.1f}s "
                f"video={episode['video']}",
                flush=True,
            )

        tolerances = sorted(
            {
                float(args.success_distance),
                *(float(value) for value in args.tolerances),
            }
        )
        threshold_rows = _tolerance_rows(episode_rows, tolerances)
        _write_csv(run_dir / "episode_results.csv", episode_rows)
        _write_csv(run_dir / "action_telemetry.csv", action_rows)
        _write_csv(
            run_dir / "move_to_object_threshold_sweep.csv", threshold_rows
        )
        report = _summary_markdown(
            checkpoint=checkpoint_path,
            shell=move_shell,
            success_distance=float(args.success_distance),
            initialization_mode=audit_initialization_mode,
            min_policy_inferences=int(args.min_policy_inferences),
            min_video_seconds=float(args.min_video_seconds),
            episodes=episode_rows,
            threshold_rows=threshold_rows,
        )
        (run_dir / "validation_report.md").write_text(
            report, encoding="utf-8"
        )
        manifest = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_global_step": int(payload.get("global_step", 0)),
            "config": str(config_path),
            "backend": "mjlab_mjwarp",
            "exact_training_backend": True,
            "training_validation_distribution": False,
            "policy_mode": (
                "seeded_smolvla_flow_plus_deterministic_residual_mean"
            ),
            "initialization_mode": audit_initialization_mode,
            "random_start_x_bounds": list(args.random_start_x_bounds),
            "random_start_y_bounds": list(args.random_start_y_bounds),
            "random_start_z_bounds": list(args.random_start_z_bounds),
            "random_start_min_xy_distance_m": float(
                args.random_start_min_xy_distance
            ),
            "minimum_policy_inferences": int(args.min_policy_inferences),
            "minimum_video_seconds": float(args.min_video_seconds),
            "continue_after_strict_success": True,
            "frozen_smolvla_checkpoint": str(training_args.base_checkpoint),
            "restored_move_to_curriculum_shell": move_shell,
            "validation_seed": int(args.validation_seed),
            "success_distance_m": float(args.success_distance),
            "episodes_per_target": int(args.episodes_per_target),
            "target_objects": list(allowed_objects),
            "min_scene_objects": int(args.min_scene_objects),
            "max_scene_objects": int(args.max_scene_objects),
            "scene_object_count_sampling": "deterministic_cycle",
            "worlds": GROUP_SIZE,
            "groups": 1,
            "group_size": GROUP_SIZE,
            "recorded_world_per_group": 1,
            "render_resolution": [
                int(training_args.render_width),
                int(training_args.render_height),
            ],
            "video_fps": float(args.fps),
            "backend_metadata": backend.metadata(),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(
            f"[smolvla-mjwarp-video] complete: {run_dir}",
            flush=True,
        )
    finally:
        if backend is not None:
            backend.close()
        runtime = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
