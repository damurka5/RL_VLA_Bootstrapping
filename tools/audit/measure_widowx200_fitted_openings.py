#!/usr/bin/env python3
"""Measure the WidowX-200 catalog's first-contact openings and caught-start seat.

``ObjectVariant.fitted_gripper_opening`` is the normalized opening at which the
pads FIRST make bilateral contact with an object. The CDPR values were measured
by closing on a pad-centred object until the physics reported contact, after
mesh-bounds estimates overstated every one of them by 0.15-0.33; this repeats
that protocol on the WidowX gripper, on the compiled ``wx200_mjwarp_scene.xml``
with the rescaled catalog, rather than computing anything from widths.

For each graspable catalog it:

1. rests the object on the desk at a lattice cell and parks the gripper open
   with the pads centred on it;
2. closes in small steps until both pads touch the object (MJWarp's 2 mm
   contact tolerance), and records the measured opening there;
3. reseats the grip at ``fitted - squeeze`` for a few squeezes, lifts 5 cm
   through the normalized action interface, and reports whether the object
   came with it -- the check that the caught-object start actually holds.

CPU MuJoCo, no GPU. Prints one JSON report; ``--write`` patches the measured
openings into ``cdpr_object_catalog.WIDOWX200_FITTED_GRIPPER_OPENING``.

Usage::

    python tools/audit/measure_widowx200_fitted_openings.py
    python tools/audit/measure_widowx200_fitted_openings.py --write
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must precede the catalog import: the catalog sizes itself at import time.
os.environ["RLVLA_EMBODIMENT"] = "widowx200"

import numpy as np  # noqa: E402

from rl_vla_bootstrapping.simulation import cdpr_object_catalog as catalog  # noqa: E402
from robots.widowx200.widowx200_mujoco import workspace as W  # noqa: E402
from robots.widowx200.widowx200_mujoco.controller import (  # noqa: E402
    WidowX200ControlSpec,
    WidowX200MountPose,
    WidowX200TaskSpaceController,
)

SCENE = ROOT / "robots" / "widowx200" / "widowx200_mujoco" / "wx200_mjwarp_scene.xml"
CONTACT_TOLERANCE = 0.002
# Closing on the mug across its handle would measure the handle; the WidowX
# configs never ask for a mug grasp, but its number is recorded across the body.
GRASP_YAW = {"robocasa_mug": 0.5 * math.pi}


def _spec(layout: W.WidowX200SceneLayout) -> WidowX200ControlSpec:
    return WidowX200ControlSpec(
        mount=WidowX200MountPose(layout.base_position, layout.base_yaw),
        workspace_x=layout.workspace_x,
        workspace_y=layout.workspace_y,
        workspace_z=layout.workspace_z,
        min_reach_radius=layout.min_reach_radius,
    )


def _bilateral_contact(mujoco, model, data, pads, object_geoms) -> tuple[bool, bool]:
    left = right = False
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if float(contact.dist) > CONTACT_TOLERANCE:
            continue
        pair = {int(contact.geom1), int(contact.geom2)}
        if not pair & object_geoms:
            continue
        left |= pads[0] in pair
        right |= pads[1] in pair
    return left, right


def measure(name: str, *, cell: tuple[float, float], squeezes: list[float]) -> dict:
    import mujoco

    variant = catalog.OBJECT_VARIANTS[name]
    models = catalog.compile_catalog_variant_models(mujoco, SCENE)
    model = models[name]
    model.opt.timestep = 0.002
    layout = W.DEFAULT_LAYOUT
    W.mount_widowx200(model, layout, mujoco=mujoco)

    def geom(label: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, label))

    pads = (geom("left_finger_pad"), geom("right_finger_pad"))
    object_geoms = {
        geom(catalog.slot_geom_name(0, slot))
        for slot in catalog.COLLISION_GEOM_SLOT_NAMES
    }
    joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mjwarp_object_slot_0_free"))
    qadr = int(model.jnt_qposadr[joint])
    body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mjwarp_object_slot_0"))
    yaw = float(GRASP_YAW.get(name, 0.0))

    def fresh(opening: float):
        data = mujoco.MjData(model)
        rest = W.DESK_SURFACE_Z + float(variant.rest_height)
        data.qpos[qadr : qadr + 3] = (cell[0], cell[1], rest)
        data.qpos[qadr + 3 : qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        controller = WidowX200TaskSpaceController(model, data, spec=_spec(layout), mujoco=mujoco)
        controller.reset_to_pose([cell[0], cell[1], rest], yaw=yaw, gripper=opening)
        for _ in range(200):  # let the object settle between the open pads
            controller.run_simulation_step()
        return data, controller, rest

    data, controller, rest = fresh(1.0)
    fitted = float("nan")
    for _ in range(600):
        target = max(0.0, controller.get_gripper_target() - 0.002)
        controller.set_gripper(target)
        for _ in range(10):
            controller.run_simulation_step()
        left, right = _bilateral_contact(mujoco, model, data, pads, object_geoms)
        if left and right:
            fitted = controller.get_gripper_opening()
            break
        if target <= 0.0:
            break

    holds: dict[str, dict] = {}
    if math.isfinite(fitted):
        for squeeze in squeezes:
            data, controller, rest = fresh(max(0.0, fitted - squeeze))
            z0 = float(data.xpos[body][2])
            for _ in range(int(round(0.05 / 0.0104)) + 12):
                controller.apply_normalized_action([0.0, 0.0, 1.0, 0.0, 0.0])
            for _ in range(10):
                controller.apply_normalized_action([0.0, 0.0, 0.0, 0.0, 0.0])
            lift = float(data.xpos[body][2]) - z0
            ee_gap = float(np.linalg.norm(data.xpos[body] - controller.get_end_effector_position()))
            holds[f"{squeeze:.3f}"] = {
                "object_lift_m": round(lift, 4),
                "object_to_ee_m": round(ee_gap, 4),
                "held": bool(lift >= 0.03 and ee_gap <= 0.02),
            }
    return {
        "catalog": name,
        "scale": catalog.WIDOWX200_OBJECT_SCALE[name],
        "rest_height_m": round(float(variant.rest_height), 5),
        "grasp_yaw_rad": round(yaw, 4),
        # Radius (or half-extent) of the first collider, doubled: a sanity
        # read on the scaling, not an input to anything.
        "first_primitive_width_m": round(2.0 * float(variant.primitives[0].size[0]), 4),
        "fitted_gripper_opening": round(fitted, 3) if math.isfinite(fitted) else None,
        "holds_after_lift": holds,
    }


def _write(measured: dict[str, float]) -> None:
    path = ROOT / "rl_vla_bootstrapping" / "simulation" / "cdpr_object_catalog.py"
    text = path.read_text(encoding="utf-8")
    start = text.index("WIDOWX200_FITTED_GRIPPER_OPENING: dict[str, float] = {")
    end = text.index("}", start)
    block = text[start:end]
    for name, value in measured.items():
        block = re.sub(
            rf'"{name}": [0-9.]+,', f'"{name}": {value:.3f},', block
        )
    path.write_text(text[:start] + block + text[end:], encoding="utf-8")
    print(f"[measure] wrote {len(measured)} openings into {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--catalogs",
        nargs="+",
        default=[
            "robocasa_apple",
            "robocasa_bell_pepper",
            "robocasa_tomato",
            "robocasa_orange",
            "robocasa_potato",
            "robocasa_mug",
        ],
    )
    parser.add_argument("--squeezes", nargs="+", type=float, default=[0.03, 0.055, 0.08])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    cell = W.sector_lattice()[0]
    report = [measure(name, cell=cell, squeezes=args.squeezes) for name in args.catalogs]
    print(json.dumps({"cell": cell, "objects": report}, indent=2), flush=True)
    if args.write:
        _write(
            {
                item["catalog"]: float(item["fitted_gripper_opening"])
                for item in report
                if item["fitted_gripper_opening"] is not None
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
