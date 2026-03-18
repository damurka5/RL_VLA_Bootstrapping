from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rl_vla_bootstrapping.cli.run_cdpr_policy import (
    _control_spec_from_config,
    _default_instruction,
    _default_objects,
    _motion_diagnostics_for_log,
)
from rl_vla_bootstrapping.core.commands import ensure_directory
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.embodiments.mujoco import MujocoEmbodiment
from rl_vla_bootstrapping.simulation.scene_builder import build_scene_xml
from robots.cdpr.cdpr_dataset.synthetic_tasks import prepare_cdpr_workspace
from robots.cdpr.cdpr_mujoco.policy_control import (
    apply_normalized_cdpr_action,
    policy_action_frequency_hz,
    policy_action_period_s,
)


@dataclass(frozen=True)
class DiagnosticDemo:
    name: str
    kind: str
    description: str
    chunk: np.ndarray


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose CDPR controller response using scripted 8-action chunks that mirror OpenVLA/OFT rollout scaling."
    )
    parser.add_argument("--config", required=True, help="Path to bootstrap YAML/JSON/TOML config.")
    parser.add_argument("--scene", default=None, help="Scene name override. Defaults to config preview scene.")
    parser.add_argument("--target-object", default=None, help="Primary target object.")
    parser.add_argument(
        "--distractor",
        action="append",
        default=[],
        help="Additional distractor object. Repeat to add more than one.",
    )
    parser.add_argument("--instruction", default=None, help="Optional instruction label saved with the diagnostics.")
    parser.add_argument("--run-dir", default=None, help="Optional output directory.")
    parser.add_argument("--run-name", default="cdpr_policy_diagnostics", help="Artifact name prefix.")
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=None,
        help="Number of actions per demo. Defaults to the config action codec chunk size.",
    )
    parser.add_argument("--hold-steps", type=int, default=None, help="Override extra simulator substeps per policy action.")
    parser.add_argument("--action-step-xyz", type=float, default=None, help="Optional override for XYZ step size in meters.")
    parser.add_argument("--action-step-yaw", type=float, default=None, help="Optional override for yaw step size in radians.")
    parser.add_argument(
        "--axis-magnitude",
        type=float,
        default=1.0,
        help="Normalized magnitude used for axis demos before denormalization. Clipped to [-1, 1].",
    )
    parser.add_argument(
        "--include-negative-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also generate mirrored negative-axis demos.",
    )
    parser.add_argument(
        "--random-demos",
        type=int,
        default=1,
        help="Number of random 8-action chunks to generate after the axis demos.",
    )
    parser.add_argument(
        "--random-magnitude",
        type=float,
        default=1.0,
        help="Absolute normalized bound for random actions before denormalization.",
    )
    parser.add_argument(
        "--randomize-yaw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include random yaw actions in the random demos.",
    )
    parser.add_argument(
        "--randomize-gripper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include random gripper commands in the random demos.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random diagnostic chunks.")
    parser.add_argument(
        "--capture-all-substeps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record every MuJoCo substep instead of only the last frame of each policy action.",
    )
    parser.add_argument("--warm-steps", type=int, default=10, help="Warmup steps for hold_current_pose before each demo.")
    parser.add_argument("--log-every", type=int, default=1, help="Logging cadence in policy-action steps.")
    return parser


def _build_axis_demos(chunk_length: int, magnitude: float, *, include_negative: bool) -> list[DiagnosticDemo]:
    clipped = float(np.clip(magnitude, -1.0, 1.0))
    signs = (1.0, -1.0) if include_negative else (1.0,)
    axis_specs = (("x", 0), ("y", 1), ("z", 2))

    demos: list[DiagnosticDemo] = []
    for axis_name, axis_idx in axis_specs:
        for sign in signs:
            value = float(sign * clipped)
            chunk = np.zeros((int(chunk_length), 5), dtype=np.float32)
            chunk[:, axis_idx] = value
            direction = "pos" if value >= 0.0 else "neg"
            demos.append(
                DiagnosticDemo(
                    name=f"axis_{axis_name}_{direction}",
                    kind="axis",
                    description=f"{chunk_length} repeated normalized actions with {axis_name}={value:+.3f}",
                    chunk=chunk,
                )
            )
    return demos


def _build_random_demos(
    chunk_length: int,
    *,
    demo_count: int,
    magnitude: float,
    seed: int,
    randomize_yaw: bool,
    randomize_gripper: bool,
) -> list[DiagnosticDemo]:
    rng = np.random.default_rng(int(seed))
    clipped = float(abs(np.clip(magnitude, -1.0, 1.0)))
    demos: list[DiagnosticDemo] = []

    for idx in range(max(0, int(demo_count))):
        chunk = np.zeros((int(chunk_length), 5), dtype=np.float32)
        chunk[:, :3] = rng.uniform(-clipped, clipped, size=(int(chunk_length), 3)).astype(np.float32)
        if randomize_yaw:
            chunk[:, 3] = rng.uniform(-clipped, clipped, size=(int(chunk_length),)).astype(np.float32)
        if randomize_gripper:
            chunk[:, 4] = rng.uniform(-clipped, clipped, size=(int(chunk_length),)).astype(np.float32)
        demos.append(
            DiagnosticDemo(
                name=f"random_{idx:02d}",
                kind="random",
                description=(
                    f"{chunk_length} random normalized actions in [-{clipped:.3f}, +{clipped:.3f}] "
                    f"for xyz{', yaw' if randomize_yaw else ''}{', gripper' if randomize_gripper else ''}"
                ),
                chunk=chunk,
            )
        )
    return demos


def _mm(vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec, dtype=np.float32) * np.float32(1000.0)


def _format_triplet(vec: np.ndarray) -> str:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)[:3]
    return f"({arr[0]:+.4f}, {arr[1]:+.4f}, {arr[2]:+.4f})"


def _format_triplet_mm(vec: np.ndarray) -> str:
    arr = _mm(vec).reshape(-1)[:3]
    return f"({arr[0]:+.1f}, {arr[1]:+.1f}, {arr[2]:+.1f}) mm"


def _collect_demo_record(
    *,
    step: int,
    action: np.ndarray,
    ee_before: np.ndarray,
    yaw_before: float | None,
    gripper_before: float | None,
    result: dict[str, Any],
    motion_diag: dict[str, np.ndarray | float],
    frames_per_action: int,
) -> dict[str, Any]:
    ee_after = np.asarray(result.get("ee_position", ee_before), dtype=np.float32).reshape(-1)[:3]
    gain = float(motion_diag["realized_vs_command_gain"])
    cosine = float(motion_diag["realized_vs_command_cosine"])
    record = {
        "step": int(step),
        "normalized_action": np.asarray(np.clip(action, -1.0, 1.0), dtype=np.float32).reshape(5).tolist(),
        "ee_before": np.asarray(ee_before, dtype=np.float32).reshape(-1)[:3].tolist(),
        "ee_after": ee_after.tolist(),
        "yaw_before": None if yaw_before is None else float(yaw_before),
        "yaw_after": None if "ee_yaw" not in result else float(result["ee_yaw"]),
        "target_yaw": None if "target_yaw" not in result else float(result["target_yaw"]),
        "gripper_before": None if gripper_before is None else float(gripper_before),
        "gripper_after": None if "gripper_opening" not in result else float(result["gripper_opening"]),
        "gripper_command": None if "gripper_command" not in result else float(result["gripper_command"]),
        "target_xyz": np.asarray(motion_diag["target_xyz"], dtype=np.float32).reshape(-1)[:3].tolist(),
        "commanded_xyz_delta_raw": np.asarray(motion_diag["commanded_xyz_delta_raw"], dtype=np.float32).reshape(-1)[:3].tolist(),
        "commanded_xyz_delta_effective": np.asarray(motion_diag["commanded_xyz_delta_effective"], dtype=np.float32).reshape(-1)[:3].tolist(),
        "realized_xyz_delta": np.asarray(motion_diag["realized_xyz_delta"], dtype=np.float32).reshape(-1)[:3].tolist(),
        "realized_vs_command_gain": None if not np.isfinite(gain) else gain,
        "realized_vs_command_cosine": None if not np.isfinite(cosine) else cosine,
        "sim_steps": int(result.get("sim_steps", frames_per_action)),
        "captured_frames": int(frames_per_action),
    }
    return record


def _summarize_demo(demo: DiagnosticDemo, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "demo_name": demo.name,
            "kind": demo.kind,
            "description": demo.description,
            "policy_steps": 0,
        }

    initial_ee = np.asarray(records[0]["ee_before"], dtype=np.float32)
    final_ee = np.asarray(records[-1]["ee_after"], dtype=np.float32)
    cmd_raw = np.asarray([row["commanded_xyz_delta_raw"] for row in records], dtype=np.float32)
    cmd_eff = np.asarray([row["commanded_xyz_delta_effective"] for row in records], dtype=np.float32)
    realized = np.asarray([row["realized_xyz_delta"] for row in records], dtype=np.float32)
    gains = np.asarray(
        [np.nan if row["realized_vs_command_gain"] is None else row["realized_vs_command_gain"] for row in records],
        dtype=np.float32,
    )
    cosines = np.asarray(
        [np.nan if row["realized_vs_command_cosine"] is None else row["realized_vs_command_cosine"] for row in records],
        dtype=np.float32,
    )

    finite_gains = gains[np.isfinite(gains)]
    finite_cosines = cosines[np.isfinite(cosines)]
    realized_total = final_ee - initial_ee

    return {
        "demo_name": demo.name,
        "kind": demo.kind,
        "description": demo.description,
        "policy_steps": int(len(records)),
        "initial_ee_xyz": initial_ee.tolist(),
        "final_ee_xyz": final_ee.tolist(),
        "commanded_total_xyz_raw": np.sum(cmd_raw, axis=0, dtype=np.float32).tolist(),
        "commanded_total_xyz_effective": np.sum(cmd_eff, axis=0, dtype=np.float32).tolist(),
        "realized_total_xyz": realized_total.tolist(),
        "realized_step_sum_xyz": np.sum(realized, axis=0, dtype=np.float32).tolist(),
        "commanded_total_distance_mm": float(np.linalg.norm(_mm(np.sum(cmd_eff, axis=0, dtype=np.float32)))),
        "realized_total_distance_mm": float(np.linalg.norm(_mm(realized_total))),
        "mean_step_command_distance_mm": float(np.mean(np.linalg.norm(_mm(cmd_eff), axis=1))),
        "mean_step_realized_distance_mm": float(np.mean(np.linalg.norm(_mm(realized), axis=1))),
        "mean_gain": None if finite_gains.size == 0 else float(np.mean(finite_gains)),
        "min_gain": None if finite_gains.size == 0 else float(np.min(finite_gains)),
        "max_gain": None if finite_gains.size == 0 else float(np.max(finite_gains)),
        "mean_cosine": None if finite_cosines.size == 0 else float(np.mean(finite_cosines)),
        "max_abs_normalized_action": float(np.max(np.abs(demo.chunk))),
    }


def _save_raw_trajectory_npz(sim: Any, output_path: Path) -> None:
    trajectory_data = getattr(sim, "trajectory_data", None)
    if not isinstance(trajectory_data, list) or not trajectory_data:
        return

    payload: dict[str, np.ndarray] = {}
    keys = ("timestamp", "ee_position", "target_position", "slider_positions", "cable_lengths", "control_signals")
    for key in keys:
        values = [step[key] for step in trajectory_data if key in step]
        if len(values) != len(trajectory_data):
            continue
        payload[key] = np.asarray(values)

    if payload:
        np.savez(output_path, **payload)


def _save_demo_artifacts(
    *,
    sim: Any,
    demo_dir: Path,
    demo: DiagnosticDemo,
    demo_summary: dict[str, Any],
    step_records: list[dict[str, Any]],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    video_fps = float(sim._estimate_video_fps()) if hasattr(sim, "_estimate_video_fps") else 20.0

    if getattr(sim, "overview_frames", None):
        overview_path = demo_dir / "overview_video.mp4"
        sim.save_video(sim.overview_frames, str(overview_path), fps=video_fps)
        outputs["overview_video"] = overview_path.as_posix()

    if getattr(sim, "ee_camera_frames", None):
        ee_path = demo_dir / "ee_camera_video.mp4"
        sim.save_video(sim.ee_camera_frames, str(ee_path), fps=video_fps)
        outputs["ee_camera_video"] = ee_path.as_posix()

    chunk_npy = demo_dir / "normalized_action_chunk.npy"
    np.save(chunk_npy, demo.chunk)
    outputs["normalized_action_chunk_npy"] = chunk_npy.as_posix()

    chunk_json = demo_dir / "normalized_action_chunk.json"
    chunk_json.write_text(json.dumps(demo.chunk.tolist(), indent=2), encoding="utf-8")
    outputs["normalized_action_chunk_json"] = chunk_json.as_posix()

    records_path = demo_dir / "motion_diagnostics.json"
    records_path.write_text(
        json.dumps(
            {
                "demo": {
                    "name": demo.name,
                    "kind": demo.kind,
                    "description": demo.description,
                },
                "summary": demo_summary,
                "step_records": step_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs["motion_diagnostics"] = records_path.as_posix()

    raw_npz_path = demo_dir / "trajectory_raw.npz"
    _save_raw_trajectory_npz(sim, raw_npz_path)
    if raw_npz_path.exists():
        outputs["trajectory_raw"] = raw_npz_path.as_posix()

    if hasattr(sim, "save_summary"):
        sim.save_summary(str(demo_dir), demo.name)
        outputs["summary"] = (demo_dir / "summary.txt").as_posix()

    return outputs


def _run_demo(
    *,
    embodiment: MujocoEmbodiment,
    xml_path: Path,
    demo_dir: Path,
    demo: DiagnosticDemo,
    instruction: str,
    control_spec: Any,
    capture_all_substeps: bool,
    warm_steps: int,
    log_every: int,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    sim = embodiment.instantiate_controller(xml_path=xml_path, run_dir=demo_dir)
    records: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}

    try:
        sim.initialize()
        workspace_safety = prepare_cdpr_workspace(
            sim,
            initial_hold_warm_steps=int(warm_steps),
            clear_recordings=True,
        )
        sim_dt = float(getattr(getattr(sim, "controller", None), "dt", 1.0 / 60.0))
        setattr(sim, "language_instruction", instruction)

        print(
            f"[demo:{demo.name}] {demo.description} "
            f"(ee_min_z={workspace_safety['ee_min_z']:.4f}, ee_spawn_z={workspace_safety['ee_spawn_z']:.4f}, "
            f"lifted={workspace_safety['lifted_to_spawn_height']})"
        )
        for step_idx, action in enumerate(demo.chunk):
            ee_before = np.asarray(sim.get_end_effector_position(), dtype=np.float32).reshape(-1)[:3]
            yaw_before = float(sim.get_yaw()) if hasattr(sim, "get_yaw") else None
            grip_before = float(sim.get_gripper_opening()) if hasattr(sim, "get_gripper_opening") else None

            result = apply_normalized_cdpr_action(
                sim,
                action,
                control_spec,
                ee_min_z=float(workspace_safety["ee_min_z"]),
                capture_last_frame=not capture_all_substeps,
                capture_all_steps=capture_all_substeps,
            )
            motion_diag = _motion_diagnostics_for_log(
                action=action,
                control_spec=control_spec,
                ee_before=ee_before,
                result=result,
            )
            record = _collect_demo_record(
                step=step_idx,
                action=action,
                ee_before=ee_before,
                yaw_before=yaw_before,
                gripper_before=grip_before,
                result=result,
                motion_diag=motion_diag,
                frames_per_action=(
                    control_spec.sim_steps_per_policy_action if capture_all_substeps else 1
                ),
            )
            records.append(record)

            if step_idx % max(1, int(log_every)) == 0:
                cmd_eff = np.asarray(record["commanded_xyz_delta_effective"], dtype=np.float32)
                realized = np.asarray(record["realized_xyz_delta"], dtype=np.float32)
                ee_after = np.asarray(record["ee_after"], dtype=np.float32)
                yaw_after = record["yaw_after"]
                grip_after = record["gripper_after"]
                print(
                    f"  step={step_idx:02d} action={record['normalized_action']} "
                    f"ee={_format_triplet(ee_after)} "
                    f"yaw={0.0 if yaw_after is None else float(yaw_after):.4f} "
                    f"grip={0.0 if grip_after is None else float(grip_after):.4f}"
                )
                print(
                    "    motion "
                    f"cmd_eff={_format_triplet(cmd_eff)} {_format_triplet_mm(cmd_eff)} "
                    f"real={_format_triplet(realized)} {_format_triplet_mm(realized)} "
                    f"gain={float('nan') if record['realized_vs_command_gain'] is None else float(record['realized_vs_command_gain']):.4f} "
                    f"cosine={float('nan') if record['realized_vs_command_cosine'] is None else float(record['realized_vs_command_cosine']):.4f}"
                )

        summary = _summarize_demo(demo, records)
        outputs = _save_demo_artifacts(
            sim=sim,
            demo_dir=demo_dir,
            demo=demo,
            demo_summary=summary,
            step_records=records,
        )

        cmd_total = np.asarray(summary["commanded_total_xyz_effective"], dtype=np.float32)
        realized_total = np.asarray(summary["realized_total_xyz"], dtype=np.float32)
        print(
            f"[demo:{demo.name}] total cmd={_format_triplet(cmd_total)} {_format_triplet_mm(cmd_total)} "
            f"realized={_format_triplet(realized_total)} {_format_triplet_mm(realized_total)}"
        )
        return summary, outputs, sim_dt
    finally:
        if hasattr(sim, "cleanup"):
            sim.cleanup()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_project_config(args.config)
    run_dir = (
        ensure_directory(Path(args.run_dir).expanduser().resolve())
        if args.run_dir
        else ensure_directory(
            (config.resolve_path(config.project.output_root) or Path("runs"))
            / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )

    object_names = _default_objects(config, args.target_object, args.distractor)
    if not object_names:
        raise RuntimeError("No objects selected for the diagnostic scene.")
    scene_name = args.scene or config.simulation.preview_scene or "desk"
    instruction = args.instruction or _default_instruction(args.target_object, object_names)
    chunk_length = int(args.chunk_length or config.policy.action_codec.chunk_size)
    if chunk_length <= 0:
        raise ValueError(f"Chunk length must be positive, got {chunk_length}.")

    demos = _build_axis_demos(
        chunk_length,
        float(args.axis_magnitude),
        include_negative=bool(args.include_negative_axis),
    )
    demos.extend(
        _build_random_demos(
            chunk_length,
            demo_count=int(args.random_demos),
            magnitude=float(args.random_magnitude),
            seed=int(args.seed),
            randomize_yaw=bool(args.randomize_yaw),
            randomize_gripper=bool(args.randomize_gripper),
        )
    )
    if not demos:
        raise RuntimeError("No diagnostic demos were generated.")

    xml_path = build_scene_xml(
        config,
        output_dir=run_dir / "scene",
        scene_name=scene_name,
        object_names=object_names,
    )
    embodiment = MujocoEmbodiment(config=config, spec=config.embodiment)
    control_spec = _control_spec_from_config(config, args.hold_steps)
    if args.action_step_xyz is not None:
        control_spec = replace(control_spec, action_step_xyz=float(args.action_step_xyz))
    if args.action_step_yaw is not None:
        control_spec = replace(control_spec, action_step_yaw=float(args.action_step_yaw))

    print(f"Run directory: {run_dir}")
    print(f"Scene: {scene_name}")
    print(f"Objects: {object_names}")
    print(f"Instruction label: {instruction}")
    print(f"Chunk length: {chunk_length}")
    print(f"Capture mode: {'all_substeps' if args.capture_all_substeps else 'last_frame_only'}")

    demo_root = ensure_directory(run_dir / "demos")
    manifest_demos: list[dict[str, Any]] = []
    sim_dt: float | None = None

    for demo_idx, demo in enumerate(demos):
        demo_dir = ensure_directory(demo_root / f"{demo_idx:02d}_{demo.name}")
        summary, outputs, demo_sim_dt = _run_demo(
            embodiment=embodiment,
            xml_path=xml_path,
            demo_dir=demo_dir,
            demo=demo,
            instruction=instruction,
            control_spec=control_spec,
            capture_all_substeps=bool(args.capture_all_substeps),
            warm_steps=int(args.warm_steps),
            log_every=int(args.log_every),
        )
        if sim_dt is None:
            sim_dt = float(demo_sim_dt)
        manifest_demos.append(
            {
                "name": demo.name,
                "kind": demo.kind,
                "description": demo.description,
                "directory": demo_dir.as_posix(),
                "artifacts": outputs,
                "summary": summary,
            }
        )

    sim_dt = float(sim_dt or 1.0 / 60.0)
    action_period = policy_action_period_s(sim_dt, control_spec.hold_steps)
    action_hz = policy_action_frequency_hz(sim_dt, control_spec.hold_steps)
    print(
        "Control contract: "
        f"dt={sim_dt:.6f}s, hold_steps={control_spec.hold_steps}, "
        f"sim_steps_per_action={control_spec.sim_steps_per_policy_action}, "
        f"policy_period={action_period:.6f}s (~{action_hz:.2f} Hz), "
        f"action_step_xyz={control_spec.action_step_xyz}, action_step_yaw={control_spec.action_step_yaw}"
    )

    manifest = {
        "run_dir": run_dir.as_posix(),
        "scene_name": scene_name,
        "scene_xml": xml_path.as_posix(),
        "object_names": object_names,
        "instruction": instruction,
        "run_name": args.run_name,
        "chunk_length": chunk_length,
        "capture_mode": "all_substeps" if args.capture_all_substeps else "last_frame_only",
        "seed": int(args.seed),
        "control_spec": asdict(control_spec),
        "sim_dt": sim_dt,
        "policy_period_s": action_period,
        "policy_frequency_hz": action_hz,
        "demos": manifest_demos,
    }
    manifest_path = run_dir / "diagnostic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest saved: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
