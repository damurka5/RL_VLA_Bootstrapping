#!/usr/bin/env python3
"""Executable MJWarp backend smoke: reset, controller, contacts, and cameras."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    BatchedReverseFrontierResetter,
    RankLocalCurriculum,
)
from rl_vla_bootstrapping.policy.rank_local_grpo import RankLocalGroupLayout
from rl_vla_bootstrapping.simulation.cdpr_backend import (
    CDPRBackendConfig,
    create_cdpr_backend,
)
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import CATALOG_TO_ID
from scripts.render_cdpr_mjlab_camera_videos import (
    SCENARIOS,
    _phase_complete,
    _policy_action,
    _scenario_metrics,
    _yaw_quaternions,
)


def _finite(torch: Any, *values: Any) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def _scenario_state(backend: Any) -> dict[str, Any]:
    low = backend.low_dim_observations()
    object_positions = low.object_positions[0].detach().cpu().numpy().copy()
    return {
        "ee_position": low.ee_position[0].detach().cpu().numpy().copy(),
        "ee_yaw": float(low.ee_yaw[0].item()),
        "gripper_opening": float(low.gripper_opening[0].item()),
        "gripper_target": float(backend._controller_gripper[0].item()),
        "target_position": (
            low.target_position[0].detach().cpu().numpy().copy()
        ),
        "tendon_lengths": (
            low.tendon_lengths[0].detach().cpu().numpy().copy()
        ),
        "object_position": object_positions[0],
        "reference_position": object_positions[1],
        "object_positions": object_positions,
        "pinned": False,
    }


def _run_training_scenario(
    backend: Any,
    scenario_name: str,
) -> dict[str, Any]:
    torch = backend.torch
    scenario = SCENARIOS[scenario_name]
    worlds = backend.worlds_per_rank

    def repeat(value: Any, *, dtype: Any) -> Any:
        tensor = torch.as_tensor(value, dtype=dtype, device=backend.device)
        return tensor.unsqueeze(0).repeat(worlds, *([1] * tensor.ndim))

    all_worlds = torch.arange(
        worlds, dtype=torch.int64, device=backend.device
    )
    backend.reset_worlds(all_worlds)
    backend.set_object_catalogs(
        repeat(
            [CATALOG_TO_ID[name] for name in scenario.catalogs],
            dtype=torch.int64,
        )
    )
    positions = repeat(scenario.object_positions, dtype=torch.float32)
    quaternions = repeat(
        _yaw_quaternions(scenario.object_yaws), dtype=torch.float32
    )
    backend.set_free_body_poses(
        backend.object_body_ids, positions, quaternions
    )
    backend.set_end_effector_poses(
        repeat(scenario.ee_start, dtype=torch.float32),
        torch.full(
            (worlds,),
            float(scenario.ee_yaw),
            dtype=torch.float32,
            device=backend.device,
        ),
    )
    backend.set_gripper_openings(
        torch.full(
            (worlds,),
            float(scenario.gripper_opening),
            dtype=torch.float32,
            device=backend.device,
        )
    )
    # The second write guarantees the caught object remains at the calibrated
    # pad center after controller-preload forward passes.
    backend.set_free_body_poses(
        backend.object_body_ids, positions, quaternions
    )
    backend.set_visual_variants(
        torch.full(
            (worlds,),
            int(scenario.texture_variant),
            dtype=torch.int64,
            device=backend.device,
        ),
        repeat(scenario.background_rgba, dtype=torch.float32),
        torch.full(
            (worlds,),
            float(scenario.gripper_shade),
            dtype=torch.float32,
            device=backend.device,
        ),
    )

    state = _scenario_state(backend)
    initial_grasp_offset = (
        np.asarray(state["object_position"], dtype=np.float64)
        - np.asarray(state["ee_position"], dtype=np.float64)
    )
    phase_steps: dict[str, int] = {}
    phase_completed: dict[str, bool] = {}
    held_openings: list[float] = []
    held_slips: list[float] = []
    active = torch.ones(
        (worlds,), dtype=torch.bool, device=backend.device
    )
    for phase in scenario.phases:
        completed = False
        for used in range(1, int(phase.max_steps) + 1):
            action = _policy_action(state, phase)
            backend.step(repeat(action, dtype=torch.float32), active)
            state = _scenario_state(backend)
            if phase.target_gripper < 0.55:
                held_openings.append(float(state["gripper_opening"]))
                held_slips.append(
                    float(
                        np.linalg.norm(
                            np.asarray(
                                state["object_position"], dtype=np.float64
                            )
                            - np.asarray(
                                state["ee_position"], dtype=np.float64
                            )
                            - initial_grasp_offset
                        )
                    )
                )
            if _phase_complete(state, phase, used):
                completed = True
                break
        phase_steps[phase.name] = used
        phase_completed[phase.name] = completed

    metrics = _scenario_metrics(scenario, state)
    stable_openings = np.asarray(held_openings[5:], dtype=np.float64)
    max_gripper_step = (
        float(np.max(np.abs(np.diff(stable_openings))))
        if stable_openings.size >= 2
        else 0.0
    )
    gripper_steady_max_error = (
        float(
            np.max(
                np.abs(
                    stable_openings - float(scenario.gripper_opening)
                )
            )
        )
        if stable_openings.size
        else 0.0
    )
    before_hold = np.asarray(state["ee_position"], dtype=np.float64)
    zero_actions = torch.zeros(
        (worlds, 5), dtype=torch.float32, device=backend.device
    )
    for _ in range(12):
        backend.step(zero_actions, active)
    hold_state = _scenario_state(backend)
    hold_drift = float(
        np.linalg.norm(
            np.asarray(hold_state["ee_position"], dtype=np.float64)
            - before_hold
        )
    )
    return {
        "phase_steps": phase_steps,
        "phase_completed": phase_completed,
        "placement": metrics,
        "held_object_max_slip": max(held_slips, default=0.0),
        "gripper_steady_max_error": gripper_steady_max_error,
        "gripper_max_step_after_settle": max_gripper_step,
        "controller_hold_drift": hold_drift,
    }


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
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/mjlab_backend_smoke.json")
    )
    args = parser.parse_args()
    report: dict[str, Any] = {"ok": False, "checks": {}, "errors": []}
    backend = None
    try:
        import torch

        if args.worlds < 8 or args.worlds % 8:
            raise ValueError("--worlds must be a positive multiple of eight.")
        project = load_project_config(args.config.resolve())
        xml_path = project.resolve_path(project.simulator.fixed_scene_xml)
        if xml_path is None:
            raise ValueError("simulator.fixed_scene_xml is required.")
        layout = RankLocalGroupLayout(
            worlds_per_rank=args.worlds,
            groups_per_rank=args.worlds // 8,
            group_size=8,
        )
        backend = create_cdpr_backend(
            CDPRBackendConfig(
                backend="mjlab_mjwarp",
                worlds_per_rank=args.worlds,
                groups_per_rank=args.worlds // 8,
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
        curriculum = RankLocalCurriculum(device=backend.device)
        resetter = BatchedReverseFrontierResetter(
            backend=backend,
            layout=layout,
            curriculum=curriculum,
            rank=0,
            base_seed=args.seed,
        )
        reset = resetter.reset(update_index=0, round_index=0)
        low_before = backend.low_dim_observations()
        ee_position_before = low_before.ee_position.clone()
        tendon_lengths_before = low_before.tendon_lengths.clone()
        report["checks"]["reset_finite"] = _finite(
            torch,
            low_before.ee_position,
            low_before.object_positions,
            low_before.tendon_lengths,
            low_before.gripper_opening,
        )
        tendon_upper = torch.as_tensor(
            backend.host_model.tendon_range[list(backend.tendon_ids), 1],
            dtype=torch.float32,
            device=backend.device,
        )
        preload_error = (
            low_before.tendon_lengths - tendon_upper[None, :]
        ).abs()
        report["preload_tendon_max_error"] = float(
            preload_error.max().item()
        )
        report["checks"]["calibrated_tendon_preload_active"] = (
            report["preload_tendon_max_error"] <= 5.0e-4
        )
        qpos_before = backend.export_worlds([0, 1])
        report["checks"]["group_state_broadcast_qpos"] = bool(
            (qpos_before[0]["qpos"] == qpos_before[1]["qpos"]).all()
        )
        report["capacity_after_reset"] = backend.capacity_status()
        reset_contact_count = int(backend._nacon.reshape(-1)[0].item())
        if reset_contact_count:
            reset_contact_distances = backend._contact_dist[
                :reset_contact_count
            ]
            report["reset_contact_min_distance"] = float(
                reset_contact_distances.min().item()
            )
        else:
            report["reset_contact_min_distance"] = None
        report["checks"]["no_deep_reset_contacts"] = bool(
            report["reset_contact_min_distance"] is None
            or (
                torch.isfinite(reset_contact_distances).all()
                and report["reset_contact_min_distance"] >= -0.05
            )
        )

        generator = torch.Generator(device=backend.device)
        generator.manual_seed(args.seed + 17)
        rollout_step_diagnostics = []
        for step in range(max(1, int(args.steps))):
            actions = torch.rand(
                (args.worlds, 5),
                generator=generator,
                dtype=torch.float32,
                device=backend.device,
            ) * 2.0 - 1.0
            # Exercise masked completed candidates without changing tensor shape.
            active = torch.ones(
                (args.worlds,), dtype=torch.bool, device=backend.device
            )
            active[(step + 1) % args.worlds] = False
            step_observation = backend.step(actions, active)
            step_capacity = backend.capacity_status()
            rollout_step_diagnostics.append(
                {
                    "step": step,
                    "finite": _finite(
                        torch,
                        step_observation.ee_position,
                        step_observation.object_positions,
                        step_observation.tendon_lengths,
                        step_observation.gripper_opening,
                    ),
                    "contacts": step_capacity["contacts"],
                    "max_constraints_per_world": step_capacity[
                        "max_constraints_per_world"
                    ],
                    "contact_overflow": step_capacity["contact_overflow"],
                    "constraint_overflow": step_capacity[
                        "constraint_overflow"
                    ],
                }
            )
        report["rollout_step_diagnostics"] = rollout_step_diagnostics
        low_after = backend.low_dim_observations()
        report["capacity_after_rollout"] = backend.capacity_status()
        report["checks"]["seven_substep_controller_finite"] = _finite(
            torch,
            low_after.ee_position,
            low_after.object_positions,
            low_after.tendon_lengths,
            low_after.gripper_opening,
        )
        report["checks"]["four_spatial_tendon_tensor"] = tuple(
            low_after.tendon_lengths.shape
        ) == (args.worlds, 4)
        report["checks"]["spatial_tendons_evolve_under_control"] = bool(
            (low_after.tendon_lengths - tendon_lengths_before)
            .abs()
            .max()
            > 1.0e-7
        )
        report["checks"]["controller_moved_active_worlds"] = bool(
            torch.linalg.vector_norm(
                low_after.ee_position - ee_position_before, dim=-1
            ).max()
            > 1.0e-5
        )
        right_finger_id = int(
            backend.mujoco.mj_name2id(
                backend.host_model,
                backend.mujoco.mjtObj.mjOBJ_JOINT,
                "finger_r",
            )
        )
        if right_finger_id < 0:
            raise RuntimeError("The required finger_r equality joint is missing.")
        right_finger_qadr = int(
            backend.host_model.jnt_qposadr[right_finger_id]
        )
        equality_error = (
            backend._qpos[:, backend.finger_qadr]
            - backend._qpos[:, right_finger_qadr]
        ).abs()
        report["finger_equality_max_error"] = float(equality_error.max().item())
        report["checks"]["finger_equality_active"] = bool(
            backend._eq_active is not None
            and tuple(backend._eq_active.shape)
            == (args.worlds, int(backend.host_model.neq))
            and bool(backend._eq_active[:, 0].all())
            and report["finger_equality_max_error"] <= 5.0e-3
        )
        sensor_data = getattr(backend, "_sensordata", None)
        report["checks"]["camera_frame_sensors_finite"] = bool(
            sensor_data is not None
            and tuple(sensor_data.shape)
            == (args.worlds, int(backend.host_model.nsensordata))
            and _finite(torch, sensor_data)
        )

        # The resetter places all object collision variants on the support.
        # Querying the first stable primitive collider against the desk exercises
        # the global contact arrays and world-id scatter path after stepping.
        desk = torch.full(
            (args.worlds,),
            int(backend.desk_geom_id),
            dtype=torch.int64,
            device=backend.device,
        )
        target_geom = torch.full(
            (args.worlds,),
            int(backend.slot_geom_ids_host[0][0]),
            dtype=torch.int64,
            device=backend.device,
        )
        contact = backend.contact_mask(target_geom, desk)
        report["contact_world_count"] = int(contact.sum().item())
        report["total_active_contacts"] = int(backend._nacon[0].item())
        report["checks"]["contact_step_generated_contacts"] = (
            report["total_active_contacts"] > 0
        )
        report["checks"]["contact_query_shape"] = tuple(contact.shape) == (
            args.worlds,
        )
        finger_contact = backend.finger_object_contact_metrics(
            reset.task_state.target_slots
        )
        report["checks"]["finger_contact_query_shape"] = all(
            tuple(value.shape) == (args.worlds,)
            for value in (
                finger_contact.left_contact,
                finger_contact.right_contact,
                finger_contact.left_normal_force,
                finger_contact.right_normal_force,
            )
        )
        report["checks"]["gpu_contact_forces_finite"] = _finite(
            torch,
            finger_contact.left_normal_force,
            finger_contact.right_normal_force,
        )
        report["gpu_contact_evidence"] = {
            "left_contact_worlds": int(
                finger_contact.left_contact.sum().item()
            ),
            "right_contact_worlds": int(
                finger_contact.right_contact.sum().item()
            ),
            "bilateral_contact_worlds": int(
                finger_contact.bilateral_contact.sum().item()
            ),
            "left_normal_force_max_n": float(
                finger_contact.left_normal_force.max().item()
            ),
            "right_normal_force_max_n": float(
                finger_contact.right_normal_force.max().item()
            ),
            "device": str(finger_contact.left_normal_force.device),
        }
        report["checks"]["objects_are_unpinned_free_bodies"] = (
            "_pinned_mask" not in vars(backend)
            and not hasattr(backend, "configure_pinned_objects")
        )

        cameras = backend.render_policy_cameras()
        expected = (
            args.worlds,
            3,
            project.simulator.render_height,
            project.simulator.render_width,
        )
        report["checks"]["camera_shapes"] = (
            tuple(cameras.overview.shape) == expected
            and tuple(cameras.wrist.shape) == expected
        )
        report["checks"]["camera_gpu_float_rgb"] = (
            cameras.overview.device == backend.device
            and cameras.wrist.device == backend.device
            and cameras.overview.dtype == torch.float32
            and cameras.wrist.dtype == torch.float32
            and float(cameras.overview.min().item()) >= 0.0
            and float(cameras.overview.max().item()) <= 1.0
            and float(cameras.wrist.min().item()) >= 0.0
            and float(cameras.wrist.max().item()) <= 1.0
        )
        report["checks"]["third_slot_exact_wrist_duplicate"] = (
            cameras.aux.data_ptr() == cameras.wrist.data_ptr()
        )
        report["checks"]["physical_cameras_are_distinct"] = bool(
            (cameras.overview - cameras.wrist).abs().mean() > 1.0e-4
        )
        report["camera"] = {
            "overview_shape": list(cameras.overview.shape),
            "wrist_shape": list(cameras.wrist.shape),
            "overview_mean": float(cameras.overview.mean().item()),
            "wrist_mean": float(cameras.wrist.mean().item()),
            "overview_center_rgb": cameras.overview[
                0,
                :,
                project.simulator.render_height // 2,
                project.simulator.render_width // 2,
            ]
            .detach()
            .cpu()
            .tolist(),
            "wrist_center_rgb": cameras.wrist[
                0,
                :,
                project.simulator.render_height // 2,
                project.simulator.render_width // 2,
            ]
            .detach()
            .cpu()
            .tolist(),
        }

        training_scenarios = {
            name: _run_training_scenario(backend, name)
            for name in (
                "training_put_into_bowl",
                "training_put_on_plate",
            )
        }
        report["training_scenarios"] = training_scenarios
        report["checks"]["controller_reaches_xyz_targets"] = all(
            all(result["phase_completed"].values())
            for result in training_scenarios.values()
        )
        report["checks"]["controller_holds_xyz_target"] = all(
            float(result["controller_hold_drift"]) <= 0.02
            for result in training_scenarios.values()
        )
        report["checks"]["gripper_holds_command_without_jitter"] = all(
            float(result["gripper_steady_max_error"]) <= 0.05
            and float(result["gripper_max_step_after_settle"]) <= 0.02
            for result in training_scenarios.values()
        )
        report["checks"]["caught_object_remains_in_gripper"] = all(
            float(result["held_object_max_slip"]) <= 0.03
            for result in training_scenarios.values()
        )
        report["checks"]["training_put_into_bowl_succeeds"] = bool(
            training_scenarios["training_put_into_bowl"]["placement"][
                "success"
            ]
        )
        report["checks"]["training_put_on_plate_succeeds"] = bool(
            training_scenarios["training_put_on_plate"]["placement"][
                "success"
            ]
        )

        backend.reset_worlds(
            torch.tensor([0, args.worlds - 1], device=backend.device)
        )
        partial = backend.low_dim_observations()
        report["checks"]["partial_reset_finite"] = _finite(
            torch, partial.ee_position, partial.tendon_lengths
        )
        report["checks"]["task_batch_complete_groups"] = tuple(
            reset.group_ids.shape
        ) == (args.worlds,)
        report["capacity"] = backend.capacity_status()
        report["simulator_metadata"] = backend.metadata()
        report["checks"]["robocasa_mesh_assets_active"] = (
            report["simulator_metadata"].get("object_geometry")
            == "robocasa_visual_plus_cdpr_native_primitives_v1"
            and len(
                report["simulator_metadata"].get(
                    "object_assets_sha256", ""
                )
            )
            == 64
        )
        report["checks"]["contact_pipeline_is_gpu_resident"] = (
            finger_contact.left_normal_force.is_cuda
            and finger_contact.right_normal_force.is_cuda
        )
        report["checks"]["no_contact_capacity_overflow"] = (
            not report["capacity"]["contact_overflow"]
        )
        report["checks"]["no_constraint_capacity_overflow"] = (
            not report["capacity"]["constraint_overflow"]
        )
        report["metadata"] = backend.metadata()
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    finally:
        if backend is not None:
            backend.close()

    failed = [name for name, value in report["checks"].items() if not bool(value)]
    report["failed_checks"] = failed
    report["ok"] = not report["errors"] and not failed
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"smoke_report={output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
