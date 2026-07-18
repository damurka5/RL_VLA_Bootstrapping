#!/usr/bin/env python3
"""Render strict GPU-physics pick/lift/release diagnostics for RoboCasa assets.

Physics, collision detection, contact solving, contact forces, and grasp
evidence stay in MJWarp/CUDA. Camera tensors and compact scalar diagnostics are
copied to the host only to encode the requested validation video and reports.
There is intentionally no CPU simulator or CPU-contact fallback in this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.simulation.cdpr_backend import (
    CDPRBackendConfig,
    create_cdpr_backend,
)
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
    BOWL_CATALOG,
    CATALOG_TO_ID,
    GRASPABLE_CDPR_CATALOGS,
    OBJECT_VARIANTS,
    PLATE_CATALOG,
)


SUPPORT_SURFACE_Z = 0.15
GRASPABLE_CATALOGS = GRASPABLE_CDPR_CATALOGS
FITTED_GRIPPER = {
    catalog: OBJECT_VARIANTS[catalog].fitted_gripper_opening
    for catalog in GRASPABLE_CATALOGS
}
MIN_PAD_FORCE_N = 0.05
MAX_RELATIVE_SLIP_M = 0.008
MAX_RELATIVE_ORIENTATION_SLIP_RAD = 0.15
MIN_LIFT_M = 0.015
PERSISTENCE_STEPS = 2

# Each tuple is (phase name, environment actions, x, y, z, yaw, gripper).
ACTION_SCHEDULE: tuple[tuple[str, int, float, float, float, float, float], ...] = (
    ("settle_closed", 8, 0.0, 0.0, 0.0, 0.0, 0.0),
    ("establish_pad_force", 4, 0.0, 0.0, 0.0, 0.0, -0.10),
    ("lift", 16, 0.0, 0.0, 0.70, 0.0, 0.0),
    ("transport", 12, 0.55, 0.0, 0.0, 0.0, 0.0),
    ("hold", 6, 0.0, 0.0, 0.0, 0.0, 0.0),
    ("release", 10, 0.0, 0.0, 0.0, 0.0, 1.0),
    ("retreat", 8, -0.45, 0.0, 0.55, 0.0, 0.0),
)


def _uint8_camera_batch(tensor: Any) -> np.ndarray:
    """Debug/export transfer only; physics and evidence remain on CUDA."""

    return (
        tensor.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .detach()
        .cpu()
        .numpy()
    )


def _quaternion_multiply(left: Any, right: Any) -> Any:
    import torch

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _relative_quaternion(parent: Any, child: Any) -> Any:
    conjugate = parent.clone()
    conjugate[..., 1:] *= -1.0
    relative = _quaternion_multiply(conjugate, child)
    return relative / relative.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _diagnostic_tile(
    frame: np.ndarray,
    *,
    catalog: str,
    phase: str,
    bilateral: bool,
    left_force: float,
    right_force: float,
    slip: float,
    orientation_slip: float,
    lifted: bool,
    grasp: bool,
    released: bool,
) -> Image.Image:
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + 48), (12, 15, 20))
    image.paste(Image.fromarray(frame), (0, 48))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    status = "RELEASE" if released else "GRASP" if grasp else "CONTACT" if bilateral else "FREE"
    color = (
        (108, 235, 148)
        if grasp
        else (109, 202, 255)
        if released
        else (255, 196, 92)
        if bilateral
        else (205, 210, 220)
    )
    draw.text((5, 4), f"{catalog} | {phase} | {status}", fill=color, font=font)
    draw.text(
        (5, 20),
        f"L/R={left_force:.3f}/{right_force:.3f} N  "
        f"slip={slip * 1000.0:.2f} mm/{np.degrees(orientation_slip):.2f} deg",
        fill=(238, 238, 238),
        font=font,
    )
    draw.text(
        (5, 34),
        f"bilateral={int(bilateral)} lifted={int(lifted)}",
        fill=(180, 198, 220),
        font=font,
    )
    return image


def _grid_frame(
    frames: np.ndarray,
    *,
    catalogs: Sequence[str],
    phase: str,
    bilateral: Sequence[bool],
    left_force: Sequence[float],
    right_force: Sequence[float],
    slip: Sequence[float],
    orientation_slip: Sequence[float],
    lifted: Sequence[bool],
    grasp: Sequence[bool],
    released: Sequence[bool],
) -> np.ndarray:
    tiles = [
        _diagnostic_tile(
            frames[index],
            catalog=catalogs[index],
            phase=phase,
            bilateral=bilateral[index],
            left_force=left_force[index],
            right_force=right_force[index],
            slip=slip[index],
            orientation_slip=orientation_slip[index],
            lifted=lifted[index],
            grasp=grasp[index],
            released=released[index],
        )
        for index in range(len(catalogs))
    ]
    width, height = tiles[0].size
    grid = Image.new("RGB", (4 * width, 2 * height), (8, 10, 14))
    for index, tile in enumerate(tiles):
        grid.paste(tile, ((index % 4) * width, (index // 4) * height))
    return np.asarray(grid)


def _detail_frame(
    overview: np.ndarray,
    wrist: np.ndarray,
    *,
    catalog: str,
    phase: str,
    bilateral: bool,
    left_force: float,
    right_force: float,
    slip: float,
    orientation_slip: float,
    lifted: bool,
    grasp: bool,
    released: bool,
) -> np.ndarray:
    left = _diagnostic_tile(
        overview,
        catalog=catalog,
        phase=phase,
        bilateral=bilateral,
        left_force=left_force,
        right_force=right_force,
        slip=slip,
        orientation_slip=orientation_slip,
        lifted=lifted,
        grasp=grasp,
        released=released,
    )
    right = _diagnostic_tile(
        wrist,
        catalog=f"{catalog} wrist",
        phase=phase,
        bilateral=bilateral,
        left_force=left_force,
        right_force=right_force,
        slip=slip,
        orientation_slip=orientation_slip,
        lifted=lifted,
        grasp=grasp,
        released=released,
    )
    output = Image.new("RGB", (left.width + right.width, left.height), (8, 10, 14))
    output.paste(left, (0, 0))
    output.paste(right, (left.width, 0))
    return np.asarray(output)


def _write_video(frames: Sequence[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.stem}_", dir=path.parent) as tmp:
        frame_dir = Path(tmp)
        for index, frame in enumerate(frames):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                frame_dir / f"{index:05d}.png"
            )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                f"{float(fps):.8g}",
                "-i",
                str(frame_dir / "%05d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            check=True,
        )


def _write_trace(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/examples/"
            "cdpr_smolvla_complex_reverse_frontier_grpo_mjlab.yaml"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/cdpr_mjwarp_physical_grasp_videos"),
    )
    parser.add_argument(
        "--require-all-graspable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero unless all eight RoboCasa targets show strict grasp and release.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ok": False,
        "backend": "mjlab_mjwarp",
        "physics_device": str(args.device),
        "cpu_contact_fallback": False,
        "errors": [],
    }
    backend = None
    try:
        import torch

        project = load_project_config(args.config.resolve())
        xml_path = project.resolve_path(project.simulator.fixed_scene_xml)
        if xml_path is None:
            raise ValueError("simulator.fixed_scene_xml is required.")
        worlds = 8
        backend = create_cdpr_backend(
            CDPRBackendConfig(
                backend="mjlab_mjwarp",
                worlds_per_rank=worlds,
                groups_per_rank=1,
                grpo_group_size=8,
                hold_steps=6,
                render_width=project.simulator.render_width,
                render_height=project.simulator.render_height,
                object_slots=4,
                nconmax=project.simulator.nconmax,
                njmax=project.simulator.njmax,
                nccdmax=project.simulator.nccdmax,
                device=args.device,
                xml_path=xml_path,
            )
        )
        device = backend.device
        # The eight selected RoboCasa graspables exactly fill one rank-local
        # GRPO group.
        target_catalogs = tuple(GRASPABLE_CATALOGS)
        target_ids = torch.tensor(
            [CATALOG_TO_ID[name] for name in target_catalogs],
            dtype=torch.int64,
            device=device,
        )
        catalogs = torch.stack(
            (
                target_ids,
                torch.full_like(target_ids, CATALOG_TO_ID[PLATE_CATALOG]),
                torch.full_like(target_ids, CATALOG_TO_ID[BOWL_CATALOG]),
                torch.full_like(
                    target_ids, CATALOG_TO_ID[GRASPABLE_CATALOGS[1]]
                ),
            ),
            dim=1,
        )
        all_worlds = torch.arange(worlds, dtype=torch.int64, device=device)
        backend.reset_worlds(all_worlds)
        backend.set_object_catalogs(catalogs)

        xy = torch.tensor(
            (
                (-0.08, -0.05),
                (0.13, 0.09),
                (0.13, -0.12),
                (-0.14, 0.13),
            ),
            dtype=torch.float32,
            device=device,
        )
        positions = torch.zeros(
            (worlds, 4, 3), dtype=torch.float32, device=device
        )
        positions[:, :, :2] = xy[None, :, :]
        for slot in range(4):
            slot_names = (
                target_catalogs
                if slot == 0
                else (PLATE_CATALOG,) * worlds
                if slot == 1
                else (BOWL_CATALOG,) * worlds
                if slot == 2
                else (GRASPABLE_CATALOGS[1],) * worlds
            )
            positions[:, slot, 2] = torch.tensor(
                [
                    SUPPORT_SURFACE_Z
                    + float(OBJECT_VARIANTS[name].rest_height)
                    for name in slot_names
                ],
                dtype=torch.float32,
                device=device,
            )
        quaternions = torch.zeros(
            (worlds, 4, 4), dtype=torch.float32, device=device
        )
        quaternions[..., 0] = 1.0
        backend.set_free_body_poses(
            backend.object_body_ids, positions, quaternions
        )

        graspable = torch.tensor(
            [name in GRASPABLE_CATALOGS for name in target_catalogs],
            dtype=torch.bool,
            device=device,
        )
        ee_positions = positions[:, 0].clone()
        ee_positions[:, 2] += 0.08
        backend.set_end_effector_poses(
            ee_positions,
            torch.zeros((worlds,), dtype=torch.float32, device=device),
        )
        fitted = torch.tensor(
            [FITTED_GRIPPER.get(name, 1.0) for name in target_catalogs],
            dtype=torch.float32,
            device=device,
        )
        initial_opening = torch.where(
            graspable,
            (fitted - (0.001 / 0.03)).clamp(0.0, 1.0),
            torch.ones_like(fitted),
        )
        backend.set_gripper_openings(initial_opening)
        # Reassert only the reset pose. No object pose is written after stepping starts.
        backend.set_free_body_poses(
            backend.object_body_ids, positions, quaternions
        )
        backend.set_visual_variants(
            torch.arange(worlds, dtype=torch.int64, device=device) % 7,
            torch.tensor(
                [[0.76, 0.84, 0.92, 1.0]],
                dtype=torch.float32,
                device=device,
            ).expand(worlds, -1),
            torch.linspace(
                0.58, 0.94, worlds, dtype=torch.float32, device=device
            ),
        )

        target_slots = torch.zeros(
            (worlds,), dtype=torch.int64, device=device
        )
        active = torch.ones((worlds,), dtype=torch.bool, device=device)
        initial_low = backend.low_dim_observations()
        previous = (
            initial_low.object_positions[:, 0] - initial_low.ee_position
        )
        previous_quaternion = _relative_quaternion(
            initial_low.ee_quaternion,
            initial_low.object_quaternions[:, 0],
        )
        baseline_z = torch.tensor(
            [
                SUPPORT_SURFACE_Z
                + float(OBJECT_VARIANTS[name].rest_height)
                for name in target_catalogs
            ],
            dtype=torch.float32,
            device=device,
        )
        lift_threshold = baseline_z + MIN_LIFT_M
        contact_steps = torch.zeros(
            (worlds,), dtype=torch.int64, device=device
        )
        ever_grasp = torch.zeros(
            (worlds,), dtype=torch.bool, device=device
        )
        ever_release = torch.zeros_like(ever_grasp)
        ever_bilateral = torch.zeros_like(ever_grasp)
        max_left_force = torch.zeros(
            (worlds,), dtype=torch.float32, device=device
        )
        max_right_force = torch.zeros_like(max_left_force)
        max_lift = torch.zeros_like(max_left_force)
        grid_frames: list[np.ndarray] = []
        detail_frames: list[np.ndarray] = []
        trace: list[dict[str, Any]] = []
        global_step = 0

        for phase, phase_steps, x, y, z, yaw, gripper in ACTION_SCHEDULE:
            action = torch.tensor(
                [x, y, z, yaw, gripper],
                dtype=torch.float32,
                device=device,
            )[None, :].expand(worlds, -1).clone()
            action[~graspable] = 0.0
            for _ in range(phase_steps):
                global_step += 1
                low = backend.step(action, active)
                contacts = backend.finger_object_contact_metrics(target_slots)
                relative = (
                    low.object_positions[:, 0] - low.ee_position
                )
                slip = torch.linalg.vector_norm(
                    relative - previous, dim=-1
                )
                previous = relative
                relative_quaternion = _relative_quaternion(
                    low.ee_quaternion,
                    low.object_quaternions[:, 0],
                )
                quaternion_dot = (
                    relative_quaternion * previous_quaternion
                ).sum(dim=-1).abs().clamp(max=1.0)
                orientation_slip = 2.0 * torch.acos(quaternion_dot)
                previous_quaternion = relative_quaternion
                bilateral = contacts.bilateral_contact
                force_ok = (
                    contacts.left_normal_force >= MIN_PAD_FORCE_N
                ) & (
                    contacts.right_normal_force >= MIN_PAD_FORCE_N
                )
                stable = (
                    slip <= MAX_RELATIVE_SLIP_M
                ) & (
                    orientation_slip
                    <= MAX_RELATIVE_ORIENTATION_SLIP_RAD
                )
                lifted = low.object_positions[:, 0, 2] >= lift_threshold
                candidate = (
                    graspable & bilateral & force_ok & stable
                )
                contact_steps = torch.where(
                    candidate,
                    contact_steps + 1,
                    torch.zeros_like(contact_steps),
                )
                grasp = (
                    candidate
                    & (contact_steps >= PERSISTENCE_STEPS)
                    & lifted
                )
                ever_grasp |= grasp
                release_threshold = torch.where(
                    graspable,
                    (fitted + 0.04).clamp(max=1.0),
                    torch.ones_like(fitted),
                )
                released = (
                    ever_grasp
                    & (low.gripper_opening >= release_threshold)
                    & ~bilateral
                )
                ever_release |= released
                ever_bilateral |= bilateral
                max_left_force = torch.maximum(
                    max_left_force, contacts.left_normal_force
                )
                max_right_force = torch.maximum(
                    max_right_force, contacts.right_normal_force
                )
                lift = low.object_positions[:, 0, 2] - baseline_z
                max_lift = torch.maximum(max_lift, lift)

                compact = torch.stack(
                    (
                        bilateral.to(dtype=torch.float32),
                        contacts.left_normal_force,
                        contacts.right_normal_force,
                        slip,
                        orientation_slip,
                        lifted.to(dtype=torch.float32),
                        grasp.to(dtype=torch.float32),
                        released.to(dtype=torch.float32),
                        low.gripper_opening,
                        low.object_positions[:, 0, 2],
                    ),
                    dim=1,
                ).detach().cpu().numpy()
                cameras = backend.render_policy_cameras()
                overview = _uint8_camera_batch(cameras.overview)
                wrist = _uint8_camera_batch(cameras.wrist)
                boolean = compact[:, [0, 5, 6, 7]].astype(bool)
                grid_frames.append(
                    _grid_frame(
                        overview,
                        catalogs=target_catalogs,
                        phase=phase,
                        bilateral=boolean[:, 0].tolist(),
                        left_force=compact[:, 1].tolist(),
                        right_force=compact[:, 2].tolist(),
                        slip=compact[:, 3].tolist(),
                        orientation_slip=compact[:, 4].tolist(),
                        lifted=boolean[:, 1].tolist(),
                        grasp=boolean[:, 2].tolist(),
                        released=boolean[:, 3].tolist(),
                    )
                )
                detail_frames.append(
                    _detail_frame(
                        overview[0],
                        wrist[0],
                        catalog=target_catalogs[0],
                        phase=phase,
                        bilateral=bool(boolean[0, 0]),
                        left_force=float(compact[0, 1]),
                        right_force=float(compact[0, 2]),
                        slip=float(compact[0, 3]),
                        orientation_slip=float(compact[0, 4]),
                        lifted=bool(boolean[0, 1]),
                        grasp=bool(boolean[0, 2]),
                        released=bool(boolean[0, 3]),
                    )
                )
                for world, catalog in enumerate(target_catalogs):
                    trace.append(
                        {
                            "step": global_step,
                            "phase": phase,
                            "world": world,
                            "catalog": catalog,
                            "bilateral_contact": int(boolean[world, 0]),
                            "left_pad_force_n": float(compact[world, 1]),
                            "right_pad_force_n": float(compact[world, 2]),
                            "relative_position_slip_m": float(compact[world, 3]),
                            "relative_orientation_slip_rad": float(
                                compact[world, 4]
                            ),
                            "lifted": int(boolean[world, 1]),
                            "physical_grasp": int(boolean[world, 2]),
                            "released": int(boolean[world, 3]),
                            "gripper_opening": float(compact[world, 8]),
                            "object_z": float(compact[world, 9]),
                        }
                    )

        grid_path = output_dir / "real_objects_physical_grasp_grid.mp4"
        detail_path = output_dir / "apple_overview_wrist_physical_grasp.mp4"
        trace_path = output_dir / "physical_grasp_trace.csv"
        _write_video(grid_frames, grid_path, args.fps)
        _write_video(detail_frames, detail_path, args.fps)
        _write_trace(trace_path, trace)

        summary_rows: list[dict[str, Any]] = []
        for world, catalog in enumerate(target_catalogs):
            summary_rows.append(
                {
                    "world": world,
                    "catalog": catalog,
                    "graspable": bool(catalog in GRASPABLE_CATALOGS),
                    "ever_bilateral_contact": bool(
                        ever_bilateral[world].item()
                    ),
                    "max_left_pad_force_n": float(
                        max_left_force[world].item()
                    ),
                    "max_right_pad_force_n": float(
                        max_right_force[world].item()
                    ),
                    "max_physical_lift_m": float(max_lift[world].item()),
                    "persistent_force_stable_lift": bool(
                        ever_grasp[world].item()
                    ),
                    "physical_release": bool(ever_release[world].item()),
                }
            )
        graspable_rows = [
            row for row in summary_rows if row["graspable"]
        ]
        strict_pass = all(
            row["ever_bilateral_contact"]
            and row["max_left_pad_force_n"] >= MIN_PAD_FORCE_N
            and row["max_right_pad_force_n"] >= MIN_PAD_FORCE_N
            and row["persistent_force_stable_lift"]
            and row["physical_release"]
            for row in graspable_rows
        )
        report.update(
            {
                "strict_all_graspable_pass": strict_pass,
                "worlds": summary_rows,
                "videos": {
                    "all_real_objects_grid": str(grid_path),
                    "apple_overview_wrist": str(detail_path),
                },
                "trace": str(trace_path),
                "simulator_metadata": backend.metadata(),
                "capacity": backend.capacity_status(),
            }
        )
        report["ok"] = bool(
            not report["capacity"]["contact_overflow"]
            and not report["capacity"]["constraint_overflow"]
            and (strict_pass or not args.require_all_graspable)
        )
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    finally:
        if backend is not None:
            backend.close()

    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"video_manifest={manifest}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
