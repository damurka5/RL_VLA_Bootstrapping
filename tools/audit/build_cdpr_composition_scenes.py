#!/usr/bin/env python3
"""Generate the full-task scene manifest, with disjoint splits, or audit one.

Scene generation is separated from collection on purpose. The collection run
takes hours on a GPU; the question "can these bounds even be met, and does the
resulting set cover every object and both receptacles" is arithmetic and should
be answered in a second on a laptop. It is also the artefact the splits are
keyed to, so it has to exist before any teacher is scored.

The geometry bounds come from the run config's task metadata where the config
owns them -- the production success radii above all, which a scene must never
loosen -- and from the flags otherwise.

Usage::

    python tools/audit/build_cdpr_composition_scenes.py \\
        --config configs/examples/cdpr_smolvla_three_stage_put_into.yaml \\
        --count 512 --seed 20260910 --output runs/three_stage/scenes.json

    python tools/audit/build_cdpr_composition_scenes.py \\
        --audit runs/three_stage/scenes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import (  # noqa: E402
    DEFAULT_SPLIT_WEIGHTS,
    SCENE_MANIFEST_VERSION,
    SPLIT_NAMES,
    SceneGeometryConfig,
    catalog_xy_radius,
    generate_scenes,
    manifest_payload,
    read_manifest,
    scene_counts,
    write_manifest,
)


def geometry_from_config(path: Path | None, overrides: Any) -> SceneGeometryConfig:
    """Success radii and the workspace come from the run config when given.

    Not from this file's defaults. A scene generated against a looser radius
    than the one training and evaluation use would produce demonstrations of an
    easier task while every number reported about them said otherwise.
    """

    metadata: dict[str, Any] = {}
    if path is not None:
        from rl_vla_bootstrapping.core.config import load_project_config

        project = load_project_config(path)
        metadata = dict(project.task.metadata or {})

    def number(key: str, default: float) -> float:
        try:
            return float(metadata.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def bounds(key: str, default: tuple[float, float]) -> tuple[float, float]:
        raw = metadata.get(key, default)
        try:
            low, high = (float(value) for value in raw)
        except (TypeError, ValueError):
            low, high = default
        return (min(low, high), max(low, high))

    return SceneGeometryConfig(
        target_catalogs=tuple(overrides.target_catalogs),
        destinations=tuple(overrides.destinations),
        destination_success_radius={
            "plate": number("put_plate_xy_tolerance", 0.091),
            "bowl": number("put_bowl_xy_tolerance", 0.057),
        },
        approach_xy_bounds=(
            float(overrides.approach_min),
            float(overrides.approach_max),
        ),
        ee_z_bounds=(float(overrides.ee_z_min), float(overrides.ee_z_max)),
        transport_margin=float(overrides.transport_margin),
        transport_xy_max=float(overrides.transport_max),
        min_object_gap=float(overrides.min_object_gap),
        workspace_x_bounds=bounds("ee_workspace_x_bounds", (-0.19, 0.19)),
        workspace_y_bounds=bounds("ee_workspace_y_bounds", (-0.19, 0.19)),
        support_surface_z=number("support_surface_z", 0.15),
        pick_grasp_height_offset=number("pick_grasp_height_offset", 0.0075),
        include_second_receptacle=bool(overrides.second_receptacle),
        distractor_catalogs=tuple(overrides.distractors),
    )


def audit(path: Path) -> int:
    scenes, payload = read_manifest(path)
    counts = scene_counts(scenes)
    print(f"[scenes] {path}", flush=True)
    print(f"[scenes] manifest sha256 {payload.get('manifest_sha256')}", flush=True)
    print(json.dumps(counts, indent=2, sort_keys=True), flush=True)

    overlap: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        overlap[split] = {
            scene.scene_uid for scene in scenes if scene.split == split
        }
    for first in SPLIT_NAMES:
        for second in SPLIT_NAMES:
            if first >= second:
                continue
            shared = overlap[first] & overlap[second]
            if shared:
                raise SystemExit(
                    f"Splits {first} and {second} share {len(shared)} scenes. "
                    "Teacher selection would then be scored on scenes the "
                    "final test also uses."
                )
    approach = [scene.approach_xy_distance for scene in scenes]
    transport = [scene.transport_xy_distance for scene in scenes]
    inside = [
        scene
        for scene in scenes
        if scene.transport_xy_distance <= scene.destination_success_radius
    ]
    print(
        f"[scenes] approach XY {min(approach):.4f}-{max(approach):.4f} m, "
        f"transport XY {min(transport):.4f}-{max(transport):.4f} m",
        flush=True,
    )
    # The phase-7 finding, checked rather than assumed: 92.6% of composed plate
    # episodes began INSIDE the success radius, so the metric moved with the
    # reset. Zero is the only acceptable answer here.
    print(
        f"[scenes] scenes starting inside the goal radius: {len(inside)} "
        "(must be zero)",
        flush=True,
    )
    if inside:
        raise SystemExit(
            f"{len(inside)} scenes begin with the object already inside its "
            "destination's success radius. Those are not put_into tasks."
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--target-catalogs",
        nargs="+",
        default=[
            "robocasa_apple",
            "robocasa_orange",
            "robocasa_potato",
            "robocasa_tomato",
        ],
    )
    parser.add_argument("--destinations", nargs="+", default=["plate", "bowl"])
    parser.add_argument("--approach-min", type=float, default=0.06)
    parser.add_argument("--approach-max", type=float, default=0.10)
    parser.add_argument("--ee-z-min", type=float, default=0.26)
    parser.add_argument("--ee-z-max", type=float, default=0.32)
    parser.add_argument("--transport-margin", type=float, default=0.04)
    parser.add_argument("--transport-max", type=float, default=0.18)
    parser.add_argument("--min-object-gap", type=float, default=0.015)
    parser.add_argument(
        "--second-receptacle",
        action="store_true",
        help=(
            "Put the OTHER receptacle in the scene as well. This is the "
            "ambiguity stratum and is a different task: the instruction has to "
            "pick between two candidates. Off for the first bank."
        ),
    )
    parser.add_argument("--distractors", nargs="*", default=[])
    parser.add_argument(
        "--split-weights",
        nargs=4,
        type=float,
        metavar=("COLLECTION", "TEACHER_SELECTION", "STUDENT_VALIDATION", "FINAL_TEST"),
        default=[
            DEFAULT_SPLIT_WEIGHTS["collection"],
            DEFAULT_SPLIT_WEIGHTS["teacher_selection"],
            DEFAULT_SPLIT_WEIGHTS["student_validation"],
            DEFAULT_SPLIT_WEIGHTS["final_test"],
        ],
    )
    args = parser.parse_args(argv)

    if args.audit is not None:
        return audit(args.audit.expanduser().resolve())
    if args.output is None:
        raise SystemExit("--output is required unless --audit is given.")

    config = geometry_from_config(
        args.config.expanduser().resolve() if args.config else None, args
    )
    weights = dict(zip(SPLIT_NAMES, [float(v) for v in args.split_weights]))
    scenes = generate_scenes(
        count=int(args.count),
        seed=int(args.seed),
        config=config,
        split_weights=weights,
    )
    payload = manifest_payload(
        scenes,
        config=config,
        seed=int(args.seed),
        split_weights=weights,
        split_salt=SCENE_MANIFEST_VERSION,
        notes=[
            "Full-task put_into scenes: empty open gripper away from the "
            "object, object on the desk outside its destination's success "
            "radius by a real margin.",
            "Splits are hash-derived from the scene identity, so a scene keeps "
            "its split when the manifest is regenerated at a different count.",
            "Success radii are the PRODUCTION ones and are not widened here.",
        ],
    )
    output = args.output.expanduser().resolve()
    write_manifest(output, payload)
    print(f"[scenes] wrote {output}", flush=True)
    print(
        "[scenes] hull radii: "
        + ", ".join(
            f"{name}={catalog_xy_radius(name):.4f}"
            for name in sorted(
                set(config.target_catalogs)
                | {"robocasa_plate", "robocasa_bowl"}
            )
        ),
        flush=True,
    )
    print(json.dumps(payload["counts"], indent=2, sort_keys=True), flush=True)
    return audit(output)


if __name__ == "__main__":
    raise SystemExit(main())
