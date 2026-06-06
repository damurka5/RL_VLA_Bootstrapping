#!/usr/bin/env python3
"""Deterministic MuJoCo contact-stability tests for the stable CDPR object pack."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tools" / "audit" / "out"
TMP_DIR = OUT_DIR / "contact_tmp"
CDPR_XML = REPO_ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr.xml"
STABLE_DIR = REPO_ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "stable_objects"
TABLE_Z = 0.015
OBJECT_HALF_HEIGHT = {
    "stable_block": 0.035,
    "ycb_wood_block": 0.035,
    "stable_can": 0.045,
    "stable_sphere": 0.035,
    "ycb_baseball": 0.036,
    "ycb_apple": 0.038,
    "ycb_pear": 0.055,
    "ycb_peach": 0.034,
    "plate": 0.012,
    "bowl": 0.052,
    "ycb_b_cups": 0.050,
    "mug": 0.052,
}


def _write_scene(object_name: str, cdpr_xml: Path) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    obj_xml = STABLE_DIR / f"{object_name}.xml"
    if not obj_xml.exists():
        raise FileNotFoundError(obj_xml)
    scene = TMP_DIR / f"{object_name}_contact_scene.xml"
    scene.write_text(
        "\n".join(
            [
                '<mujoco model="cdpr_contact_audit">',
                '  <compiler autolimits="true"/>',
                '  <option timestep="0.002" solver="Newton" iterations="50"/>',
                f'  <include file="{cdpr_xml.as_posix()}"/>',
                "  <worldbody>",
                '    <body name="audit_table" pos="0 0 0">',
                '      <geom name="audit_table_top" type="box" size="0.45 0.35 0.015" friction="1.0 0.01 0.001" rgba="0.70 0.70 0.66 1"/>',
                "    </body>",
                "  </worldbody>",
                f'  <include file="{obj_xml.as_posix()}"/>',
                "</mujoco>",
            ]
        )
    )
    return scene


def _compile(object_name: str, cdpr_xml: Path) -> tuple[mj.MjModel, mj.MjData, int, int]:
    model = mj.MjModel.from_xml_path(str(_write_scene(object_name, cdpr_xml)))
    data = mj.MjData(model)
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, object_name)
    if body_id == -1:
        raise RuntimeError(f"Body {object_name!r} missing from compiled test scene.")
    joint_id = -1
    for offset in range(int(model.body_jntnum[body_id])):
        jid = int(model.body_jntadr[body_id]) + offset
        if int(model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_FREE):
            joint_id = jid
            break
    if joint_id == -1:
        raise RuntimeError(f"Body {object_name!r} has no freejoint.")
    return model, data, body_id, joint_id


def _set_free_pose(model: mj.MjModel, data: mj.MjData, joint_id: int, pos: np.ndarray) -> None:
    qadr = int(model.jnt_qposadr[joint_id])
    dofadr = int(model.jnt_dofadr[joint_id])
    data.qpos[qadr : qadr + 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[dofadr : dofadr + 6] = 0.0
    mj.mj_forward(model, data)


def _object_velocity(model: mj.MjModel, data: mj.MjData, joint_id: int) -> tuple[float, float]:
    dofadr = int(model.jnt_dofadr[joint_id])
    qvel = np.asarray(data.qvel[dofadr : dofadr + 6], dtype=np.float64)
    linear = float(np.linalg.norm(qvel[:3]))
    angular = float(np.linalg.norm(qvel[3:]))
    return linear, angular


def _max_contact_force(model: mj.MjModel, data: mj.MjData, body_id: int) -> float:
    max_force = 0.0
    for idx in range(int(data.ncon)):
        contact = data.contact[idx]
        b1 = int(model.geom_bodyid[int(contact.geom1)])
        b2 = int(model.geom_bodyid[int(contact.geom2)])
        if int(body_id) not in {b1, b2}:
            continue
        force = np.zeros(6, dtype=np.float64)
        try:
            mj.mj_contactForce(model, data, idx, force)
        except Exception:
            pass
        max_force = max(max_force, abs(float(force[0])))
    return float(max_force)


def _finger_contact_count(model: mj.MjModel, data: mj.MjData, body_id: int) -> int:
    finger_bodies = {
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "finger_left_car"),
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "finger_right_car"),
    }
    count = 0
    for idx in range(int(data.ncon)):
        contact = data.contact[idx]
        bodies = {
            int(model.geom_bodyid[int(contact.geom1)]),
            int(model.geom_bodyid[int(contact.geom2)]),
        }
        if int(body_id) in bodies and bodies.intersection(finger_bodies):
            count += 1
    return count


def _step_collect(
    model: mj.MjModel,
    data: mj.MjData,
    body_id: int,
    joint_id: int,
    steps: int,
    *,
    ignore_initial_steps: int = 0,
) -> dict[str, float]:
    max_linear = 0.0
    max_angular = 0.0
    max_force = 0.0
    min_z = float("inf")
    for step_idx in range(int(steps)):
        mj.mj_step(model, data)
        if step_idx < int(ignore_initial_steps):
            continue
        lin, ang = _object_velocity(model, data, joint_id)
        max_linear = max(max_linear, lin)
        max_angular = max(max_angular, ang)
        max_force = max(max_force, _max_contact_force(model, data, body_id))
        min_z = min(min_z, float(data.xpos[body_id, 2]))
    return {
        "max_linear_velocity": float(max_linear),
        "max_angular_velocity": float(max_angular),
        "max_contact_force": float(max_force),
        "min_body_z": float(min_z),
    }


def _verdict(metrics: dict[str, float], *, require_contact: bool = False, contact_count: int = 0) -> tuple[str, str]:
    if not np.isfinite(list(metrics.values())).all():
        return "fail", "non-finite state; inspect mass/inertia/contact parameters"
    if metrics["min_body_z"] < -0.08:
        return "fail", "object tunneled below support; increase margin/solver iterations or inspect collision proxy"
    if metrics["max_linear_velocity"] > 8.0 or metrics["max_angular_velocity"] > 80.0:
        return "fail", "object exploded; reduce timestep or soften contact after verifying inertia"
    if metrics["max_contact_force"] > 800.0:
        return "warn", "large contact force spike; check solref/solimp and gripper/object proxy scale"
    if require_contact and contact_count <= 0:
        return "fail", "no gripper/object contact; check object placement and finger pad reach"
    return "pass", ""


def _run_test(object_name: str, test_name: str, steps: int, cdpr_xml: Path) -> dict[str, Any]:
    model, data, body_id, joint_id = _compile(object_name, cdpr_xml)
    half_h = float(OBJECT_HALF_HEIGHT.get(object_name, 0.04))
    start = time.perf_counter()
    contact_count = 0

    if test_name == "drop":
        _set_free_pose(model, data, joint_id, np.array([0.25, 0.0, 0.35]))
        metrics = _step_collect(
            model,
            data,
            body_id,
            joint_id,
            steps,
            ignore_initial_steps=max(1, int(steps * 0.65)),
        )
    elif test_name == "rest_on_table":
        _set_free_pose(model, data, joint_id, np.array([0.25, 0.0, TABLE_Z + half_h + 0.002]))
        metrics = _step_collect(
            model,
            data,
            body_id,
            joint_id,
            steps,
            ignore_initial_steps=max(1, int(steps * 0.50)),
        )
    elif test_name == "push":
        _set_free_pose(model, data, joint_id, np.array([0.25, 0.0, TABLE_Z + half_h + 0.002]))
        for _ in range(80):
            mj.mj_step(model, data)
        data.xfrc_applied[body_id, 0] = 1.2
        metrics = _step_collect(model, data, body_id, joint_id, max(1, steps // 3))
        data.xfrc_applied[body_id, :] = 0.0
        settle = _step_collect(
            model,
            data,
            body_id,
            joint_id,
            max(1, steps // 3),
            ignore_initial_steps=max(1, steps // 6),
        )
        for key, value in settle.items():
            metrics[key] = max(float(metrics[key]), float(value))
    elif test_name in {"gripper_squeeze", "lift"}:
        mj.mj_forward(model, data)
        ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "ee_base")
        ee_pos = np.asarray(data.xpos[ee_id], dtype=np.float64).copy()
        _set_free_pose(model, data, joint_id, np.array([ee_pos[0], ee_pos[1], ee_pos[2] - 0.008]))
        act_gripper = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
        if act_gripper != -1:
            data.ctrl[act_gripper] = 1.0
            for _ in range(50):
                mj.mj_step(model, data)
            for value in np.linspace(1.0, 0.0, num=80):
                data.ctrl[act_gripper] = float(value)
                mj.mj_step(model, data)
        metrics = _step_collect(
            model,
            data,
            body_id,
            joint_id,
            max(1, steps // 2),
            ignore_initial_steps=max(1, steps // 8),
        )
        contact_count = _finger_contact_count(model, data, body_id)
        if test_name == "lift":
            ee_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "ee_free")
            if ee_joint != -1:
                qadr = int(model.jnt_qposadr[ee_joint])
                for _ in range(max(1, steps // 3)):
                    data.qpos[qadr + 2] += 0.0008
                    mj.mj_step(model, data)
            lift_metrics = _step_collect(model, data, body_id, joint_id, max(1, steps // 4))
            for key, value in lift_metrics.items():
                metrics[key] = max(float(metrics[key]), float(value))
    else:
        raise ValueError(test_name)

    status, recommendation = _verdict(
        metrics,
        require_contact=test_name in {"gripper_squeeze", "lift"},
        contact_count=contact_count,
    )
    return {
        "object": object_name,
        "test": test_name,
        "pass_fail": status,
        "max_linear_velocity_after_settling": metrics["max_linear_velocity"],
        "max_angular_velocity_after_settling": metrics["max_angular_velocity"],
        "max_normal_contact_force": metrics["max_contact_force"],
        "min_body_z": metrics["min_body_z"],
        "finger_contact_count": contact_count,
        "recommended_fix": recommendation,
        "duration_s": float(time.perf_counter() - start),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objects",
        nargs="*",
        default=["stable_block", "stable_can", "stable_sphere", "ycb_apple", "plate", "bowl", "ycb_b_cups"],
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=["drop", "rest_on_table", "push", "gripper_squeeze", "lift"],
        choices=["drop", "rest_on_table", "push", "gripper_squeeze", "lift"],
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--cdpr-xml", type=Path, default=CDPR_XML)
    parser.add_argument("--output-prefix", default="mujoco_contact_stability")
    args = parser.parse_args()
    cdpr_xml = Path(args.cdpr_xml).expanduser().resolve()

    rows: list[dict[str, Any]] = []
    for object_name in args.objects:
        for test_name in args.tests:
            try:
                rows.append(_run_test(str(object_name), test_name, int(args.steps), cdpr_xml))
            except Exception as exc:
                rows.append(
                    {
                        "object": str(object_name),
                        "test": test_name,
                        "pass_fail": "error",
                        "max_linear_velocity_after_settling": "",
                        "max_angular_velocity_after_settling": "",
                        "max_normal_contact_force": "",
                        "min_body_z": "",
                        "finger_contact_count": "",
                        "recommended_fix": f"{type(exc).__name__}: {exc}",
                        "duration_s": 0.0,
                    }
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = str(args.output_prefix).strip() or "mujoco_contact_stability"
    csv_path = OUT_DIR / f"{output_prefix}.csv"
    fields = [
        "object",
        "test",
        "pass_fail",
        "max_linear_velocity_after_settling",
        "max_angular_velocity_after_settling",
        "max_normal_contact_force",
        "min_body_z",
        "finger_contact_count",
        "recommended_fix",
        "duration_s",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    summary = {
        "csv": str(csv_path),
        "cdpr_xml": str(cdpr_xml),
        "objects": args.objects,
        "tests": args.tests,
        "rows": len(rows),
        "pass": sum(1 for row in rows if row.get("pass_fail") == "pass"),
        "warn": sum(1 for row in rows if row.get("pass_fail") == "warn"),
        "fail": sum(1 for row in rows if row.get("pass_fail") == "fail"),
        "error": sum(1 for row in rows if row.get("pass_fail") == "error"),
    }
    (OUT_DIR / f"{output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
