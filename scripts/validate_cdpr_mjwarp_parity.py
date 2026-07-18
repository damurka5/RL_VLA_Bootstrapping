#!/usr/bin/env python3
"""Quantify CPU MuJoCo versus MJWarp from one identical reset/action fixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    BatchedReverseFrontierResetter,
    RankLocalCurriculum,
)
from rl_vla_bootstrapping.policy.rank_local_grpo import RankLocalGroupLayout
from rl_vla_bootstrapping.simulation.cdpr_backend import (
    CDPRBackendConfig,
    create_cdpr_backend,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    ACTIVE_INSTRUCTION_TYPES,
    evaluate_active_sparse_tasks,
)
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
    PRIMITIVE_NAMES,
    build_variant_arrays,
)


def _name_id(mujoco: Any, model: Any, objtype: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, objtype, name))
    if value < 0:
        raise RuntimeError(f"Missing MJCF element {name!r}.")
    return value


def _apply_cpu_catalog(sim: Any, catalog_ids: np.ndarray) -> None:
    import mujoco

    variants = build_variant_arrays(np.asarray(catalog_ids).reshape(1, 4))
    for slot in range(4):
        for primitive_index, primitive in enumerate(PRIMITIVE_NAMES):
            geom = _name_id(
                mujoco,
                sim.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"mjwarp_slot_{slot}_{primitive}",
            )
            sim.model.geom_size[geom] = variants["geom_size"][
                0, slot, primitive_index
            ]
            sim.model.geom_pos[geom] = variants["geom_pos"][
                0, slot, primitive_index
            ]
            sim.model.geom_quat[geom] = variants["geom_quat"][
                0, slot, primitive_index
            ]
            sim.model.geom_rgba[geom] = variants["geom_rgba"][
                0, slot, primitive_index
            ]
        body = _name_id(
            mujoco,
            sim.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"mjwarp_object_slot_{slot}",
        )
        sim.model.body_mass[body] = variants["body_mass"][0, slot]
        sim.model.body_inertia[body] = variants["body_inertia"][0, slot]
    mujoco.mj_setConst(sim.model, sim.data)
    mujoco.mj_forward(sim.model, sim.data)


def _cpu_finger_contact(sim: Any, slot: int = 0) -> bool:
    import mujoco

    object_geoms = {
        _name_id(
            mujoco,
            sim.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"mjwarp_slot_{slot}_{primitive}",
        )
        for primitive in PRIMITIVE_NAMES
    }
    finger_geoms = {
        _name_id(mujoco, sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "finger_l_link",
            "finger_r_link",
            "finger_l_tip",
            "finger_r_tip",
            "left_finger_pad",
            "right_finger_pad",
        )
    }
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & object_geoms and pair & finger_geoms:
            return True
    return False


def _gpu_finger_contact(backend: Any) -> bool:
    import torch

    result = torch.zeros(
        (backend.worlds_per_rank,),
        dtype=torch.bool,
        device=backend.device,
    )
    for target in backend.slot_geom_ids_host[0]:
        for name in (
            "finger_l_link",
            "finger_r_link",
            "finger_l_tip",
            "finger_r_tip",
            "left_finger_pad",
            "right_finger_pad",
        ):
            finger = _name_id(
                backend.mujoco,
                backend.host_model,
                backend.mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            result.logical_or_(backend.contact_mask(int(target), finger))
    return bool(result[0].item())


def _error(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, float]:
    delta = np.asarray(cpu, dtype=np.float64) - np.asarray(gpu, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _cpu_sparse_success(
    *,
    instruction_type: str,
    ee_position: np.ndarray,
    object_positions: np.ndarray,
    target_slot: int,
    reference_slot: int,
    second_reference_slot: int,
    initial_target_position: np.ndarray,
    ever_grasped: bool,
    caught_target: bool,
    gripper_opening: float,
    support_surface_z: float,
) -> bool:
    from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
        InstructionSpec,
        compute_instruction_validation_success,
        init_reward_state,
    )

    positions = {
        "target": object_positions[target_slot],
        "reference": object_positions[max(0, reference_slot)],
        "second": object_positions[max(0, second_reference_slot)],
    }

    class FixtureEnv:
        def _get_body_position(self, name: str) -> np.ndarray:
            return np.asarray(positions[name], dtype=np.float32).copy()

    metadata = {
        "reward_mode": "sparse_binary",
        "move_to_object_validation_distance_threshold": 0.02,
        "move_to_object_require_z_window": False,
        "push_position_only_reward": True,
        "push_success_displacement": 0.08,
        "push_enforce_orthogonal_tolerance": True,
        "push_orthogonal_tolerance": 0.02,
        "push_enforce_max_overshoot": True,
        "push_max_overshoot": 0.025,
        "push_require_object_on_support": True,
        "push_support_min_clearance": 0.005,
        "push_support_vertical_tolerance": 0.04,
        "put_container_xy_tolerance": 0.03,
        "put_container_z_tolerance": 0.12,
        "put_require_release": True,
        "put_require_target_grasp_history": True,
        "put_min_target_motion": 0.04,
        "relation_left_right_offset": 0.10,
        "move_relation_success_zone_size": 0.03,
        "move_relation_min_target_motion": 0.04,
        "move_relation_require_target_grasp_history": True,
        "move_relation_require_release": True,
        "relation_min_target_motion": 0.04,
        "relation_require_target_grasp": False,
        "relation_require_target_grasp_history": True,
        "relation_require_release": True,
        "between_xy_tolerance": 0.03,
    }
    reward_state = init_reward_state(ee_position, initial_target_position)
    reward_state.ever_grasped = bool(ever_grasped)
    reward_state.grasped = bool(caught_target)
    spec = InstructionSpec(
        instruction_type=instruction_type,
        text=instruction_type,
        target_object="target",
        direction=np.zeros((3,), dtype=np.float32),
        target_displacement=0.0,
        lift_target=0.0,
        reference_object="reference",
        second_reference_object="second",
    )
    success, _ = compute_instruction_validation_success(
        spec=spec,
        ee_pos=np.asarray(ee_position, dtype=np.float32),
        reward_state=reward_state,
        task_metadata=metadata,
        obj_pos=positions["target"],
        goal_pos=positions["target"],
        env=FixtureEnv(),
        target_body_name="target",
        reference_body_name="reference",
        second_reference_body_name="second",
        gripper_opening=float(gripper_opening),
        support_surface_z=float(support_surface_z),
        caught_object_is_target=bool(caught_target),
    )
    return bool(success)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    metrics = report.get("discrepancies") or {}
    lines = [
        "# CDPR CPU MuJoCo / MJWarp parity",
        "",
        f"Status: **{'pass' if report.get('ok') else 'fail'}**",
        "",
        "| quantity | RMSE | max absolute |",
        "|---|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {values['rmse']:.6g} | {values['max_abs']:.6g} |"
        )
    lines.extend(
        [
            "",
            "MuJoCo Warp uses float32 parallel kernels, so bit identity is not "
            "expected. Camera shape, dtype, normalization, ordering, RGB channel "
            "order, and orientation are acceptance checks; pixel differences "
            "remain expected because the renderers differ.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--ee-rmse-limit", type=float, default=0.05)
    parser.add_argument("--object-rmse-limit", type=float, default=0.08)
    parser.add_argument("--tendon-rmse-limit", type=float, default=0.05)
    parser.add_argument("--gripper-rmse-limit", type=float, default=0.20)
    parser.add_argument("--camera-mae-limit", type=float, default=0.15)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/cdpr_mjwarp_parity")
    )
    parser.add_argument(
        "--render-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if int(args.steps) < 1:
        parser.error("--steps must be positive")
    args.xml = args.xml.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RLVLA_CDPR_OFFSCREEN_WIDTH", "320")
    os.environ.setdefault("RLVLA_CDPR_OFFSCREEN_HEIGHT", "240")
    os.environ.setdefault("RLVLA_CDPR_OFFSCREEN_SAMPLES", "1")
    os.environ.setdefault("RLVLA_CDPR_QUIET", "1")

    report: dict[str, Any] = {
        "ok": False,
        "xml": str(args.xml),
        "steps": int(args.steps),
        "errors": [],
    }
    backend = None
    sim = None
    try:
        import torch
        import mujoco
        from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import (
            HeadlessCDPRSimulation,
        )

        layout = RankLocalGroupLayout(
            worlds_per_rank=8, groups_per_rank=1, group_size=8
        )
        backend = create_cdpr_backend(
            CDPRBackendConfig(
                backend="mjlab_mjwarp",
                worlds_per_rank=8,
                groups_per_rank=1,
                grpo_group_size=8,
                hold_steps=6,
                device=args.device,
                xml_path=args.xml,
            )
        )
        resetter = BatchedReverseFrontierResetter(
            backend=backend,
            layout=layout,
            curriculum=RankLocalCurriculum(device=backend.device),
            rank=0,
            base_seed=args.seed,
        )
        resetter.reset(update_index=0, round_index=0)
        initial = backend.export_worlds([0])[0]

        sim = HeadlessCDPRSimulation(
            str(args.xml),
            output_dir=str(output),
            record_trajectory=False,
            use_model_cache=False,
            timestep=1.0 / 60.0,
            render_enabled=bool(args.render_cameras),
        )
        sim.initialize()
        _apply_cpu_catalog(sim, initial["catalog_ids"])
        sim.data.qpos[:] = initial["qpos"]
        sim.data.qvel[:] = initial["qvel"]
        mujoco.mj_forward(sim.model, sim.data)
        sim._sync_controller_geometry_from_state()
        sim._configure_controller_tendon_model(
            dlength_dq=np.asarray(backend._calibration["dlength_dq"])
        )
        sim.target_pos = np.asarray(initial["ee_position"], dtype=np.float64)
        yaw_target = float(sim.get_yaw())
        gripper_target = float(sim.get_gripper_opening())

        generator = np.random.default_rng(args.seed + 31)
        actions = generator.uniform(-0.75, 0.75, size=(args.steps, 5)).astype(
            np.float32
        )
        cpu_ee, gpu_ee = [], []
        cpu_object, gpu_object = [], []
        cpu_tendon, gpu_tendon = [], []
        cpu_gripper, gpu_gripper = [], []
        contact_agreement = []
        active = torch.zeros((8,), dtype=torch.bool, device=backend.device)
        active[0] = True
        action_batch = torch.zeros(
            (8, 5), dtype=torch.float32, device=backend.device
        )

        for action in actions:
            current = sim.get_end_effector_position()
            sim.target_pos = np.clip(
                current + action[:3] * 0.015,
                np.array([-0.28, -0.28, 0.20]),
                np.array([0.28, 0.28, 1.20]),
            )
            yaw_target = float(np.clip(yaw_target + action[3] * 0.08, -np.pi, np.pi))
            gripper_target = float(
                np.clip(gripper_target + action[4] * 0.05, 0.0, 1.0)
            )
            sim.set_yaw(yaw_target)
            sim.set_gripper(gripper_target)
            for _ in range(7):
                sim.run_simulation_step(capture_frame=False)

            action_batch.zero_()
            action_batch[0] = torch.as_tensor(action, device=backend.device)
            low = backend.step(action_batch, active)
            cpu_ee.append(sim.get_end_effector_position())
            gpu_ee.append(low.ee_position[0].detach().cpu().numpy())
            cpu_object.append(
                sim.data.xpos[
                    _name_id(
                        mujoco,
                        sim.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        "mjwarp_object_slot_0",
                    )
                ].copy()
            )
            gpu_object.append(
                low.object_positions[0, 0].detach().cpu().numpy()
            )
            cpu_tendon.append(sim.get_cable_lengths())
            gpu_tendon.append(low.tendon_lengths[0].detach().cpu().numpy())
            cpu_gripper.append(sim.get_gripper_opening())
            gpu_gripper.append(float(low.gripper_opening[0].item()))
            contact_agreement.append(
                _cpu_finger_contact(sim) == _gpu_finger_contact(backend)
            )

        discrepancies = {
            "end_effector_position_m": _error(
                np.asarray(cpu_ee), np.asarray(gpu_ee)
            ),
            "object_position_m": _error(
                np.asarray(cpu_object), np.asarray(gpu_object)
            ),
            "tendon_length_m": _error(
                np.asarray(cpu_tendon), np.asarray(gpu_tendon)
            ),
            "gripper_opening_normalized": _error(
                np.asarray(cpu_gripper), np.asarray(gpu_gripper)
            ),
        }
        report["discrepancies"] = discrepancies
        report["contact_boolean_agreement_rate"] = float(np.mean(contact_agreement))
        sparse_active = torch.zeros(
            (8,), dtype=torch.bool, device=backend.device
        )
        sparse_active[0] = True
        caught_target = backend.pinned_object_mask().clone()
        sparse_result = evaluate_active_sparse_tasks(
            state=reset.task_state,
            ee_position=low.ee_position,
            object_positions=low.object_positions,
            gripper_opening=low.gripper_opening,
            caught_target=caught_target,
            active_mask=sparse_active,
            max_steps=10_000,
        )
        task_id = int(reset.task_state.instruction_ids[0].item())
        cpu_sparse_success = _cpu_sparse_success(
            instruction_type=ACTIVE_INSTRUCTION_TYPES[task_id],
            ee_position=low.ee_position[0].detach().cpu().numpy(),
            object_positions=low.object_positions[0].detach().cpu().numpy(),
            target_slot=int(reset.task_state.target_slots[0].item()),
            reference_slot=int(
                reset.task_state.reference_slots[0].item()
            ),
            second_reference_slot=int(
                reset.task_state.second_reference_slots[0].item()
            ),
            initial_target_position=reset.task_state.initial_target_positions[
                0
            ]
            .detach()
            .cpu()
            .numpy(),
            ever_grasped=bool(
                reset.task_state.ever_grasped[0].item()
            ),
            caught_target=bool(caught_target[0].item()),
            gripper_opening=float(low.gripper_opening[0].item()),
            support_surface_z=float(
                reset.task_state.support_surface_z[0].item()
            ),
        )
        gpu_sparse_success = bool(sparse_result.success[0].item())
        report["sparse_success_output"] = {
            "instruction_type": ACTIVE_INSTRUCTION_TYPES[task_id],
            "cpu": cpu_sparse_success,
            "mjwarp": gpu_sparse_success,
            "mjwarp_reward": float(sparse_result.rewards[0].item()),
            "agreement": cpu_sparse_success == gpu_sparse_success,
        }
        report["thresholds"] = {
            "end_effector_position_m_rmse": args.ee_rmse_limit,
            "object_position_m_rmse": args.object_rmse_limit,
            "tendon_length_m_rmse": args.tendon_rmse_limit,
            "gripper_opening_normalized_rmse": args.gripper_rmse_limit,
        }
        checks = {
            "end_effector": discrepancies["end_effector_position_m"]["rmse"]
            <= args.ee_rmse_limit,
            "object": discrepancies["object_position_m"]["rmse"]
            <= args.object_rmse_limit,
            "tendon": discrepancies["tendon_length_m"]["rmse"]
            <= args.tendon_rmse_limit,
            "gripper": discrepancies["gripper_opening_normalized"]["rmse"]
            <= args.gripper_rmse_limit,
            "contacts": report["contact_boolean_agreement_rate"] >= 0.80,
            "sparse_success": report["sparse_success_output"]["agreement"],
        }

        if args.render_cameras:
            cpu_overview = sim.capture_frame(sim.overview_cam, "overview")
            cpu_wrist = sim.capture_frame(sim.ee_cam, "ee_camera")
            cameras = backend.render_policy_cameras()
            gpu_overview = (
                cameras.overview[0].permute(1, 2, 0).detach().cpu().numpy()
            )
            gpu_wrist = cameras.wrist[0].permute(1, 2, 0).detach().cpu().numpy()
            camera = {
                "ordering": ["overview", "ee_camera", "ee_camera"],
                "cpu_dtype": str(cpu_overview.dtype),
                "gpu_dtype": str(cameras.overview.dtype),
                "cpu_overview_shape": list(cpu_overview.shape),
                "cpu_wrist_shape": list(cpu_wrist.shape),
                "gpu_overview_shape": list(gpu_overview.shape),
                "gpu_wrist_shape": list(gpu_wrist.shape),
                "overview_rgb_mae": float(
                    np.mean(np.abs(cpu_overview.astype(np.float32) / 255.0 - gpu_overview))
                ),
                "wrist_rgb_mae": float(
                    np.mean(np.abs(cpu_wrist.astype(np.float32) / 255.0 - gpu_wrist))
                ),
                "third_slot_exact_wrist_duplicate": (
                    cameras.aux.data_ptr() == cameras.wrist.data_ptr()
                ),
            }
            report["camera_contract"] = camera
            checks["camera_shapes"] = (
                cpu_overview.shape == gpu_overview.shape == (240, 320, 3)
                and cpu_wrist.shape == gpu_wrist.shape == (240, 320, 3)
            )
            checks["camera_third_slot"] = camera[
                "third_slot_exact_wrist_duplicate"
            ]
            checks["camera_dtype_and_normalization"] = (
                cpu_overview.dtype == np.uint8
                and cpu_wrist.dtype == np.uint8
                and str(cameras.overview.dtype) == "torch.float32"
                and float(gpu_overview.min()) >= 0.0
                and float(gpu_overview.max()) <= 1.0
                and float(gpu_wrist.min()) >= 0.0
                and float(gpu_wrist.max()) <= 1.0
            )
            checks["camera_orientation_and_rgb_channels"] = (
                camera["overview_rgb_mae"] <= float(args.camera_mae_limit)
                and camera["wrist_rgb_mae"] <= float(args.camera_mae_limit)
            )
        report["checks"] = checks
        report["ok"] = all(checks.values())
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    finally:
        if sim is not None:
            sim.cleanup()
        if backend is not None:
            backend.close()

    json_path = output / "parity.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    markdown_path = output / "parity.md"
    _write_markdown(report, markdown_path)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"parity_artifact={json_path}")
    print(f"parity_report={markdown_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
