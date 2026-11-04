#!/usr/bin/env python3
"""Check scaled WidowX objects with CPU MuJoCo and optional real RGB renders.

This is a geometry/physics diagnostic, not a trained-policy evaluation or a
replacement for the CUDA preflight. It compiles mixed object/receptacle scenes
so MuJoCo rebuilds the collision BVHs for the actual sizes. Every graspable is
dropped centrally into each receptacle at every requested lattice cell, then
tested with a feedback grasp/lift/arc-carry/release script. The script's
failures are reported separately from the simple containment checks.

    python tools/audit/audit_widowx200_object_scale.py --output outputs/wx_scale
    python tools/audit/audit_widowx200_object_scale.py --output outputs/wx_scale --render

The output records the source hashes, mesh dimensions, first-contact grip
checks, final body motion, object/receptacle contact and support, and whether
the script actually retained the object until its deliberate release. Images
show the final physical state, with visual geoms only, from the policy overview
and an additional inspection camera. No training files are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("RLVLA_EMBODIMENT", "widowx200")

# Initialize NumPy before Torch (imported by the shared pipeline helpers).
import numpy as np  # noqa: E402
import mujoco as mj  # noqa: E402

from rl_vla_bootstrapping.simulation import cdpr_object_catalog as catalog  # noqa: E402
from robots.widowx200.widowx200_mujoco import workspace as workspace  # noqa: E402
from robots.widowx200.widowx200_mujoco.controller import WidowX200TaskSpaceController  # noqa: E402
from tools.audit.measure_widowx200_fitted_openings import measure  # noqa: E402
from tools.audit.widowx200_scene_coverage import control_spec, eval_config, pipeline_view  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/examples/widowx200_smolvla_pipeline.yaml"
SCENE = ROOT / "robots/widowx200/widowx200_mujoco/wx200_mjwarp_scene.xml"


def mixed_model(names: list[str]) -> Any:
    """Compile correct per-slot colliders, masses, inertias and collision BVHs."""
    spec = mj.MjSpec.from_file(str(SCENE))
    catalog._scale_catalog_meshes(spec)
    geoms = {geom.name: geom for geom in spec.geoms}
    for slot in range(4):
        variant = catalog.OBJECT_VARIANTS[names[slot] if slot < len(names) else names[0]]
        visual = geoms[catalog.slot_geom_name(slot, "visual")]
        visual.meshname = variant.visual_mesh_name
        visual.material = variant.material_name
        primitives = {primitive.primitive: primitive for primitive in variant.primitives}
        for name in catalog.COLLISION_GEOM_SLOT_NAMES:
            geom = geoms[catalog.slot_geom_name(slot, name)]
            primitive = primitives.get(name)
            geom.size = primitive.size if primitive else (1.0e-4,) * 3
            geom.pos = primitive.pos if primitive else (0.0, 0.0, 10.0)
            geom.quat = primitive.quat if primitive else (1.0, 0.0, 0.0, 0.0)
        body = catalog._spec_body(spec, f"mjwarp_object_slot_{slot}")
        body.mass, body.inertia = variant.mass, variant.inertia
        body.explicitinertial = True
    model = spec.compile()
    workspace.mount_widowx200(model, mujoco=mj)
    model.opt.timestep = 0.002
    return model


def body_id(model: Any, slot: int) -> int:
    return int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"mjwarp_object_slot_{slot}"))


def mesh_bounds(model: Any, data: Any, slot: int) -> tuple[Any, Any]:
    geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, catalog.slot_geom_name(slot, "visual"))
    mesh = int(model.geom_dataid[geom])
    start = int(model.mesh_vertadr[mesh])
    vertices = model.mesh_vert[start : start + int(model.mesh_vertnum[mesh])]
    world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    return world.min(axis=0), world.max(axis=0)


def prepare(model: Any, names: list[str], positions: list[Any], spec: Any) -> tuple[Any, Any]:
    data = mj.MjData(model)
    for slot in range(4):
        body = body_id(model, slot)
        address = int(model.jnt_qposadr[model.body_jntadr[body]])
        name = names[slot] if slot < len(names) else names[0]
        xy = positions[slot] if slot < len(positions) else (2.0 + slot, 2.0)
        data.qpos[address : address + 3] = [
            *xy, workspace.DESK_SURFACE_Z + catalog.OBJECT_VARIANTS[name].rest_height,
        ]
        data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
    controller = WidowX200TaskSpaceController(model, data, spec=spec, mujoco=mj)
    controller.reset_to_pose([*positions[0], 0.25], gripper=1.0)
    for _ in range(250):
        controller.run_simulation_step()
    return data, controller


def move(controller: Any, goal: Any, *, steps: int = 80) -> None:
    for _ in range(steps):
        error = np.asarray(goal) - controller.get_end_effector_position()
        controller.apply_normalized_action(np.r_[np.clip(error / 0.025, -1, 1), 0.0, 0.0])
        if np.linalg.norm(error) < 0.002:
            break


def close_gripper(controller: Any, opening: float) -> None:
    for _ in range(24):
        action = float(np.clip((opening - controller.get_gripper_target())
                               / controller.spec.action_step_gripper, -1.0, 1.0))
        controller.apply_normalized_action([0, 0, 0, 0, action])


def stationary_hold(name: str, spec: Any, cell: Any, tiles: list[Any] | None = None) -> dict[str, Any]:
    """A successful short lift must also survive five seconds without motion."""
    model = mixed_model([name])
    data, controller = prepare(model, [name], [cell], spec)
    body = body_id(model, 0)
    start = data.xpos[body].copy()
    move(controller, start)
    close_gripper(controller, max(0.0, catalog.OBJECT_VARIANTS[name].fitted_gripper_opening
                                 - catalog.CAUGHT_START_GRIP_SQUEEZE))
    move(controller, [*cell, 0.25])
    if tiles is not None:
        capture(model, data, f"Immediately after lift: {name}; wrist | inspection", tiles,
                focus_slot=0, policy_camera="ee_camera")
    trace = []
    for step in range(101):
        if step % 20 == 0:
            trace.append({
                "seconds_after_lift": round(step * model.opt.timestep * spec.sim_steps_per_policy_action, 2),
                "object_lift_m": round(float(data.xpos[body, 2] - start[2]), 5),
                "object_to_ee_m": round(float(np.linalg.norm(data.xpos[body] - controller.get_end_effector_position())), 5),
            })
        if step < 100:
            controller.apply_normalized_action([0, 0, 0, 0, 0])
    return {"catalog": name, "trace": trace,
            "held_after_5s": bool(trace[-1]["object_lift_m"] >= 0.04 and trace[-1]["object_to_ee_m"] < 0.03)}


def outcome(model: Any, data: Any, container: str, initial_reference: Any) -> dict[str, Any]:
    target, reference = body_id(model, 0), body_id(model, 1)
    rotation = data.xmat[reference].reshape(3, 3)
    offset = rotation.T @ (data.xpos[target] - data.xpos[reference])
    target_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == target}
    reference_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == reference}
    contact = False
    for index in range(data.ncon):
        c = data.contact[index]
        a, b = int(c.geom1), int(c.geom2)
        if c.dist <= 0.002 and (
            (a in target_geoms and b in reference_geoms)
            or (b in target_geoms and a in reference_geoms)
        ):
            contact = True
    dof = int(model.jnt_dofadr[model.body_jntadr[target]])
    speed = float(np.linalg.norm(data.qvel[dof : dof + 3]))
    # A conservative whole-mesh XY bound in the receptacle's local frame.
    # This is intentionally stronger than the training centre-distance test.
    lo, hi = mesh_bounds(model, data, 0)
    # Containers remain nearly upright in these tests. Project the world AABB
    # corners; rotation can only make this conservative rather than optimistic.
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    local = (corners - data.xpos[reference]) @ rotation
    if container == "robocasa_bowl":
        inner = (0.055 - 0.005) * catalog.WIDOWX200_OBJECT_SCALE[container]
        inside = bool(np.max(np.abs(local[:, :2])) <= inner)
    else:
        radius = catalog.OBJECT_VARIANTS[container].primitives[0].size[0]
        inside = bool(np.max(np.linalg.norm(local[:, :2], axis=1)) <= radius)
    return {
        "object_container_offset_m": np.round(offset, 5).tolist(),
        "xy_error_m": round(float(np.linalg.norm(offset[:2])), 5),
        "linear_speed_m_s": round(speed, 6),
        "container_shift_m": round(float(np.linalg.norm(data.xpos[reference] - initial_reference)), 5),
        "has_container_contact": contact,
        "whole_visual_xy_inside_conservative": inside,
        "settled_on_container": bool(contact and speed < 0.01 and offset[2] > 0.0),
    }


def settle(model: Any, data: Any, controller: Any, container: str, reference: Any,
           *, seconds: float = 2.5) -> dict[str, Any]:
    """Judge support over 0.5 s, rather than one chattering contact sample."""
    for _ in range(max(0, int(math.ceil((seconds - 0.5) / model.opt.timestep)))):
        controller.run_simulation_step()
    positions, contacts = [], []
    for _ in range(25):
        for _ in range(10):
            controller.run_simulation_step()
        sample = outcome(model, data, container, reference)
        positions.append(data.xpos[body_id(model, 0)].copy())
        contacts.append(sample["has_container_contact"])
    result = outcome(model, data, container, reference)
    excursion = float(np.max(np.linalg.norm(np.asarray(positions) - positions[0], axis=1)))
    contact_fraction = float(np.mean(contacts))
    result.update(
        support_contact_fraction=round(contact_fraction, 3),
        settling_window_excursion_m=round(excursion, 6),
        settled_on_container=bool(contact_fraction >= 0.5 and excursion < 0.002
                                  and result["object_container_offset_m"][2] > 0.0),
    )
    return result


def capture(model: Any, data: Any, label: str, tiles: list[Any], *,
            focus_slot: int = 1, policy_camera: str = "overview") -> None:
    from PIL import Image, ImageDraw

    option = mj.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 1, 0]
    option.sitegroup[:] = 0
    with mj.Renderer(model, height=300, width=400) as renderer:
        renderer.update_scene(data, camera=policy_camera, scene_option=option)
        overview = renderer.render().copy()
        camera = mj.MjvCamera()
        camera.lookat[:] = data.xpos[body_id(model, focus_slot)] + [0.0, 0.0, 0.01]
        camera.distance, camera.azimuth, camera.elevation = (0.18 if focus_slot == 0 else 0.24), 135, -40
        renderer.update_scene(data, camera=camera, scene_option=option)
        closeup = renderer.render().copy()
    tile = Image.fromarray(np.concatenate([overview, closeup], axis=1))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, tile.width, 22), fill="black")
    draw.text((5, 5), label, fill="white")
    tiles.append(tile)


def save_sheet(tiles: list[Any], path: Path) -> None:
    if not tiles:
        return
    from PIL import Image

    sheet = Image.new("RGB", (tiles[0].width * 2, tiles[0].height * math.ceil(len(tiles) / 2)))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * tile.width, (index // 2) * tile.height))
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, nargs="+", choices=range(6), default=list(range(6)))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--grip-mode", choices=("fitted", "closed"), default="fitted",
                        help="Use the calibrated caught-start squeeze, or command fully closed fingers.")
    parser.add_argument("--settle-seconds", type=float, default=2.5,
                        help="Wait this many simulated seconds before the final support verdict (minimum 0.5).")
    args = parser.parse_args()
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0.5:
        parser.error("--settle-seconds must be finite and at least 0.5")
    if catalog.ACTIVE_EMBODIMENT != "widowx200":
        parser.error("RLVLA_EMBODIMENT must be widowx200 before the catalog is imported")
    args.output.mkdir(parents=True, exist_ok=True)
    config = eval_config(pipeline_view(args.config), "put_into", True)
    metadata = config["task"]["metadata"]
    graspables = metadata["catchable_object_pool"]
    containers = metadata["container_object_pool"]
    spec = control_spec(config)
    cells = workspace.sector_lattice()
    report: dict[str, Any] = {
        "scope": "CPU MuJoCo deterministic geometry/physics diagnostic; no policy or CUDA evaluation",
        "mujoco_version": mj.__version__,
        "visual_assets_sha256": catalog.object_assets_sha256(SCENE),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.config.resolve(), Path(catalog.__file__), SCENE, Path(__file__),
                         SCENE.with_name("wx200.xml"), SCENE.with_name("wx200_scene.xml"),
                         SCENE.with_name("controller.py"), SCENE.with_name("workspace.py"),
                         ROOT / "robots/cdpr/cdpr_mujoco/cdpr_mjwarp_object_slots.xml")
        },
        "object_scales": catalog.WIDOWX200_OBJECT_SCALE,
        "cells": args.cells,
        "script_grip_mode": args.grip_mode,
        "settle_seconds": args.settle_seconds,
        "geometry": [], "grips": [], "stationary_holds": [], "centered_drops": [], "scripted_placements": [],
    }
    for name in metadata["scene_object_pool"]:
        model = mixed_model([name])
        data, _ = prepare(model, [name], [cells[3]], spec)
        lo, hi = mesh_bounds(model, data, 0)
        report["geometry"].append({
            "catalog": name,
            "visual_dimensions_mm": np.round((hi - lo) * 1000, 2).tolist(),
            "settled_visual_bottom_above_desk_mm": round(float((lo[2] - workspace.DESK_SURFACE_Z) * 1000), 2),
            "mass_g": round(catalog.OBJECT_VARIANTS[name].mass * 1000, 2),
        })
    grip_tiles: list[Any] = []
    for name in graspables:
        result = measure(name, cell=cells[0], squeezes=[catalog.CAUGHT_START_GRIP_SQUEEZE])
        result["configured_opening"] = catalog.OBJECT_VARIANTS[name].fitted_gripper_opening
        report["grips"].append(result)
        report["stationary_holds"].append(stationary_hold(name, spec, cells[3], grip_tiles if args.render else None))

    tiles: list[Any] = []
    for name in graspables:
        for container in containers:
            model = mixed_model([name, container])
            key = "put_bowl_release_height" if container == "robocasa_bowl" else "put_plate_release_height"
            for cell_index in args.cells:
                source, destination = cells[(cell_index + 3) % 6], cells[cell_index]
                data, controller = prepare(model, [name, container], [source, destination], spec)
                reference = data.xpos[body_id(model, 1)].copy()
                # Isolate physical receptacle fit from grasp/carry-controller errors.
                body = body_id(model, 0)
                address = int(model.jnt_qposadr[model.body_jntadr[body]])
                data.qpos[address : address + 3] = reference + [0.0, 0.0, float(metadata[key])]
                data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
                data.qvel[:] = 0.0
                mj.mj_forward(model, data)
                common = {"catalog": name, "container": container, "destination_cell": cell_index}
                report["centered_drops"].append({**common, **settle(model, data, controller, container, reference,
                                                                 seconds=args.settle_seconds)})
                if args.render and cell_index == args.cells[0]:
                    capture(model, data, f"Centered drop: {name} / {container}", tiles)

                data, controller = prepare(model, [name, container], [source, destination], spec)
                reference = data.xpos[body_id(model, 1)].copy()
                start = data.xpos[body].copy()
                move(controller, start)
                grip_target = (max(0.0, catalog.OBJECT_VARIANTS[name].fitted_gripper_opening
                                   - catalog.CAUGHT_START_GRIP_SQUEEZE)
                               if args.grip_mode == "fitted" else 0.0)
                close_gripper(controller, grip_target)
                move(controller, [*source, 0.25])
                lift = float(data.xpos[body, 2] - start[2])
                gap_max = float(np.linalg.norm(data.xpos[body] - controller.get_end_effector_position()))
                cx, cy, _ = workspace.DEFAULT_LAYOUT.base_position
                r0, r1 = math.hypot(source[0] - cx, source[1] - cy), math.hypot(destination[0] - cx, destination[1] - cy)
                a0, a1 = math.atan2(source[1] - cy, source[0] - cx), math.atan2(destination[1] - cy, destination[0] - cx)
                for fraction in np.linspace(0.0, 1.0, 32)[1:]:
                    radius, angle = r0 + fraction * (r1 - r0), a0 + fraction * (a1 - a0)
                    move(controller, [cx + radius * math.cos(angle), cy + radius * math.sin(angle), 0.25], steps=8)
                    gap_max = max(gap_max, float(np.linalg.norm(data.xpos[body] - controller.get_end_effector_position())))
                release = data.xpos[body_id(model, 1)].copy() + [0.0, 0.0, float(metadata[key])]
                move(controller, release)
                release_gap = float(np.linalg.norm(data.xpos[body] - controller.get_end_effector_position()))
                clearance = float(data.xpos[body, 2] - data.xpos[body_id(model, 1), 2])
                for _ in range(24):
                    controller.apply_normalized_action([0, 0, 0, 0, 1])
                move(controller, [*destination, 0.265])
                result = {
                    **common, "source_cell": (cell_index + 3) % 6,
                    "lift_m": round(lift, 5), "max_carry_gap_m": round(gap_max, 5),
                    "gap_before_open_m": round(release_gap, 5),
                    "clearance_before_open_m": round(clearance, 5),
                    "retained_until_open": bool(lift >= 0.04 and max(gap_max, release_gap) < 0.03),
                    **settle(model, data, controller, container, reference, seconds=args.settle_seconds),
                }
                report["scripted_placements"].append(result)
                print(f"[scale] {name} {container} cell {cell_index}: drop={report['centered_drops'][-1]['settled_on_container']} script={result['settled_on_container']} retained={result['retained_until_open']}", flush=True)
                if args.render and cell_index == args.cells[0]:
                    capture(model, data, f"Scripted placement: {name} / {container}; retained={result['retained_until_open']}", tiles)
    report["summary"] = {
        "grips_held": sum(all(h["held"] for h in r["holds_after_lift"].values()) for r in report["grips"]),
        "grips_total": len(report["grips"]),
        "stationary_holds_after_5s": sum(r["held_after_5s"] for r in report["stationary_holds"]),
        "centered_drops_settled": sum(r["settled_on_container"] for r in report["centered_drops"]),
        "centered_drops_total": len(report["centered_drops"]),
        "centered_drops_whole_visual_inside": sum(r["whole_visual_xy_inside_conservative"] for r in report["centered_drops"]),
        "scripted_settled": sum(r["settled_on_container"] for r in report["scripted_placements"]),
        "scripted_retained_until_open": sum(r["retained_until_open"] for r in report["scripted_placements"]),
        "scripted_total": len(report["scripted_placements"]),
    }
    report["fit_checks_passed"] = bool(
        report["summary"]["centered_drops_settled"] == len(report["centered_drops"])
        and report["summary"]["centered_drops_whole_visual_inside"] == len(report["centered_drops"])
        and report["summary"]["grips_held"] == len(graspables)
    )
    report["motion_checks_passed"] = bool(
        all(row["held_after_5s"] for row in report["stationary_holds"])
        and all(row["settled_on_container"] and row["retained_until_open"]
                for row in report["scripted_placements"])
    )
    report["passed"] = report["fit_checks_passed"] and report["motion_checks_passed"]
    save_sheet(tiles, args.output / "placements.png")
    save_sheet(grip_tiles, args.output / "grips.png")
    (args.output / "scale_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"[scale] {'PASSED' if report['passed'] else 'FAILED'}; {args.output / 'scale_audit.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
