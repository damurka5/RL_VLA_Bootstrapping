#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation
from robots.cdpr.cdpr_mujoco.policy_control import CDPRPolicyControlSpec, apply_normalized_cdpr_action


DEFAULT_XML = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr.xml"
DEFAULT_OUTPUT = ROOT / "runs" / "direct_actuator_verification"


def _verification_xml(xml_path: Path, run_dir: Path) -> Path:
    tree = ET.parse(xml_path)
    worldbody = tree.getroot().find("worldbody")
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
    output = run_dir / "cdpr_direct_actuator_verification.xml"
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output


def _run_idle(sim: HeadlessCDPRSimulation, steps: int, *, capture: bool = False) -> None:
    for _ in range(max(0, int(steps))):
        sim.run_simulation_step(capture_frame=capture)


def _apply_repeated(
    sim: HeadlessCDPRSimulation,
    spec: CDPRPolicyControlSpec,
    action: tuple[float, float, float, float, float],
    count: int,
    *,
    capture: bool,
    capture_every_actions: int,
) -> None:
    count = max(1, int(count))
    capture_every_actions = max(1, int(capture_every_actions))
    for action_index in range(count):
        apply_normalized_cdpr_action(
            sim,
            np.asarray(action, dtype=np.float32),
            spec,
            ee_min_z=0.20,
            capture_last_frame=bool(
                capture and (action_index % capture_every_actions == 0 or action_index == count - 1)
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that normalized policy actions can open/close and yaw the CDPR gripper."
    )
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-capture-every-actions", type=int, default=5)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    verification_xml = _verification_xml(args.xml_path.resolve(), run_dir)
    sim = HeadlessCDPRSimulation(
        str(verification_xml),
        output_dir=str(run_dir),
        record_trajectory=True,
        render_enabled=bool(args.record_video),
    )
    spec = CDPRPolicyControlSpec(
        xyz_limits=((-0.28, 0.28), (-0.28, 0.28), (0.20, 0.80)),
        action_step_xyz=0.03,
        action_step_yaw=0.08,
        action_step_gripper=0.05,
        hold_steps=2,
    )

    try:
        sim.initialize()
        sim.hold_current_pose(warm_steps=20)

        sim.set_gripper(0.0)
        sim.set_yaw(0.0)
        _run_idle(sim, 80)
        initial = {
            "ee_position": np.asarray(sim.get_end_effector_position(), dtype=float).tolist(),
            "gripper_opening": float(sim.get_gripper_opening()),
            "yaw": float(sim.get_yaw()),
        }

        _apply_repeated(
            sim,
            spec,
            (0.0, 0.0, 0.0, 0.0, 1.0),
            20,
            capture=args.record_video,
            capture_every_actions=args.video_capture_every_actions,
        )
        _run_idle(sim, 40)
        open_value = float(sim.get_gripper_opening())

        _apply_repeated(
            sim,
            spec,
            (0.0, 0.0, 0.0, 0.0, -1.0),
            20,
            capture=args.record_video,
            capture_every_actions=args.video_capture_every_actions,
        )
        _run_idle(sim, 40)
        closed_value = float(sim.get_gripper_opening())

        sim.set_yaw(0.0)
        _run_idle(sim, 80)
        clockwise_start = float(sim.get_yaw())
        _apply_repeated(
            sim,
            spec,
            (0.0, 0.0, 0.0, -1.0, 0.0),
            10,
            capture=args.record_video,
            capture_every_actions=args.video_capture_every_actions,
        )
        _run_idle(sim, 40)
        clockwise_delta = float(sim.get_yaw() - clockwise_start)

        sim.set_yaw(0.0)
        _run_idle(sim, 80)
        counterclockwise_start = float(sim.get_yaw())
        _apply_repeated(
            sim,
            spec,
            (0.0, 0.0, 0.0, 1.0, 0.0),
            10,
            capture=args.record_video,
            capture_every_actions=args.video_capture_every_actions,
        )
        _run_idle(sim, 40)
        counterclockwise_delta = float(sim.get_yaw() - counterclockwise_start)

        finite_state = bool(
            all(
                np.all(np.isfinite(np.asarray(getattr(sim.data, name), dtype=np.float64)))
                for name in ("qpos", "qvel", "qacc", "ctrl")
            )
        )
        checks = {
            "open_gripper": bool(open_value >= 0.80),
            "close_gripper": bool(closed_value <= 0.20),
            "rotate_gripper_clockwise": bool(clockwise_delta <= -0.50),
            "rotate_gripper_counterclockwise": bool(counterclockwise_delta >= 0.50),
            "finite_simulation_state": finite_state,
        }
        result = {
            "xml_path": str(verification_xml),
            "initial": initial,
            "measurements": {
                "open_gripper_opening": open_value,
                "close_gripper_opening": closed_value,
                "clockwise_yaw_delta_rad": clockwise_delta,
                "counterclockwise_yaw_delta_rad": counterclockwise_delta,
            },
            "thresholds": {
                "open_gripper_min": 0.80,
                "close_gripper_max": 0.20,
                "yaw_rotation_min_rad": 0.50,
            },
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
        if args.record_video:
            video_path = run_dir / "direct_actuator_verification.mp4"
            sim.save_video(sim.overview_frames, str(video_path), fps=float(args.fps))
            result["video_path"] = str(video_path)

        report_path = run_dir / "direct_actuator_verification.json"
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(report_path)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    finally:
        sim.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
