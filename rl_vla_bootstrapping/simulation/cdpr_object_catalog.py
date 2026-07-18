from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PRIMITIVE_NAMES: tuple[str, ...] = (
    "sphere_0",
    "sphere_1",
    "sphere_2",
    "cylinder_0",
    "cylinder_1",
    "cylinder_2",
    "box_0",
    "box_1",
    "box_2",
    "box_3",
    "capsule_0",
)
VISUAL_MESH_SLOT = "visual"
COLLISION_GEOM_SLOT_NAMES: tuple[str, ...] = PRIMITIVE_NAMES
GEOM_SLOT_NAMES: tuple[str, ...] = (
    VISUAL_MESH_SLOT,
    *COLLISION_GEOM_SLOT_NAMES,
)

ACTIVE_CDPR_CATALOGS: tuple[str, ...] = (
    "robocasa_apple",
    "robocasa_banana",
    "robocasa_carrot",
    "robocasa_bell_pepper",
    "robocasa_tomato",
    "robocasa_orange",
    "robocasa_potato",
    "robocasa_mug",
    "robocasa_plate",
    "robocasa_bowl",
)
GRASPABLE_CDPR_CATALOGS: tuple[str, ...] = ACTIVE_CDPR_CATALOGS[:8]
PLATE_CATALOG = "robocasa_plate"
BOWL_CATALOG = "robocasa_bowl"
CATALOG_TO_ID = {name: index for index, name in enumerate(ACTIVE_CDPR_CATALOGS)}
ID_TO_CATALOG = {index: name for name, index in CATALOG_TO_ID.items()}
INACTIVE_CATALOG_ID = -1


@dataclass(frozen=True)
class PrimitiveSpec:
    primitive: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ObjectVariant:
    name: str
    label: str
    asset_directory: str
    asset_files: tuple[str, ...]
    mass: float
    inertia: tuple[float, float, float]
    rest_height: float
    fitted_gripper_opening: float
    primitives: tuple[PrimitiveSpec, ...]

    @property
    def visual_mesh_name(self) -> str:
        return f"{self.name}_visual_mesh"

    @property
    def material_name(self) -> str:
        return f"{self.name}_material"


# Visuals are a deliberately small RoboCasa Objaverse subset. The selected
# variants have one visual OBJ each, so per-world mesh switching remains
# compatible with MJWarp's fixed topology. Contact stays on native primitives:
# RoboCasa's large convex decompositions are intentionally not placed in the
# hot path.
OBJECT_VARIANTS: dict[str, ObjectVariant] = {
    "robocasa_apple": ObjectVariant(
        name="robocasa_apple",
        label="apple",
        asset_directory="objects/objaverse/apple/apple_10",
        asset_files=(
            "model.xml",
            "visual/model_normalized_0.obj",
            "visual/material.mtl",
            "visual/material_0.jpeg",
            "visual/image0.png",
        ),
        mass=0.12,
        inertia=(0.0000874, 0.0000900, 0.0000794),
        rest_height=0.03456,
        fitted_gripper_opening=0.785,
        primitives=(PrimitiveSpec("sphere_0", (0.0345, 0.0, 0.0)),),
    ),
    "robocasa_banana": ObjectVariant(
        name="robocasa_banana",
        label="banana",
        asset_directory="objects/objaverse/banana/banana_8",
        asset_files=(
            "model.xml",
            "visual/model_normalized_0.obj",
            "visual/material.mtl",
            "visual/material_0.jpeg",
            "visual/image0.png",
        ),
        mass=0.12,
        inertia=(0.000368, 0.000090, 0.000370),
        rest_height=0.03328,
        fitted_gripper_opening=0.368,
        primitives=(
            PrimitiveSpec(
                "box_0",
                (0.018, 0.045, 0.028),
                (0.006, -0.035, 0.004),
                (0.9959527, 0.0, 0.0, -0.0898785),
            ),
            PrimitiveSpec(
                "box_1",
                (0.018, 0.045, 0.028),
                (-0.006, 0.035, -0.004),
                (0.9959527, 0.0, 0.0, 0.0898785),
            ),
        ),
    ),
    "robocasa_carrot": ObjectVariant(
        name="robocasa_carrot",
        label="carrot",
        asset_directory="objects/objaverse/carrot/carrot_1",
        asset_files=(
            "model.xml",
            "visual/model_normalized_0.obj",
            "visual/material.mtl",
            "visual/material_0.png",
            "visual/image0.png",
        ),
        mass=0.10,
        inertia=(0.000297, 0.0000171, 0.000297),
        rest_height=0.01593,
        fitted_gripper_opening=0.20,
        primitives=(
            PrimitiveSpec(
                "capsule_0",
                (0.014, 0.065, 0.0),
                quat=(0.7071068, 0.7071068, 0.0, 0.0),
            ),
            PrimitiveSpec("sphere_0", (0.012, 0.0, 0.0), (0.0, 0.070, 0.0)),
        ),
    ),
    "robocasa_bell_pepper": ObjectVariant(
        name="robocasa_bell_pepper",
        label="bell pepper",
        asset_directory="objects/objaverse/bell_pepper/bell_pepper_0",
        asset_files=(
            "model.xml",
            "visual/model_normalized_0.obj",
            "visual/material.mtl",
            "visual/material_0.png",
            "visual/image0.png",
        ),
        mass=0.12,
        inertia=(0.000169, 0.000116, 0.000181),
        rest_height=0.03610,
        fitted_gripper_opening=0.868,
        primitives=(
            PrimitiveSpec("sphere_0", (0.037, 0.0, 0.0), (0.0, -0.012, 0.0)),
            PrimitiveSpec("sphere_1", (0.037, 0.0, 0.0), (0.0, 0.012, 0.0)),
        ),
    ),
    "robocasa_tomato": ObjectVariant(
        name="robocasa_tomato",
        label="tomato",
        asset_directory="objects/objaverse/tomato/tomato_1",
        asset_files=(
            "model.xml",
            "visual/model_normalized_0.obj",
            "visual/material.mtl",
            "visual/material_0.jpeg",
            "visual/image0.png",
        ),
        mass=0.10,
        inertia=(0.0000622, 0.0000622, 0.0000683),
        rest_height=0.02885,
        fitted_gripper_opening=0.685,
        primitives=(PrimitiveSpec("sphere_0", (0.0315, 0.0, 0.0)),),
    ),
    "robocasa_orange": ObjectVariant(
        name="robocasa_orange",
        label="orange",
        asset_directory="objects/objaverse/orange/orange_2",
        asset_files=(
            "model.xml",
            "visual/model_normalized_0.obj",
            "visual/material.mtl",
            "visual/material_0.png",
            "visual/image0.png",
        ),
        mass=0.13,
        inertia=(0.0000973, 0.0000973, 0.000100),
        rest_height=0.03285,
        fitted_gripper_opening=0.768,
        primitives=(PrimitiveSpec("sphere_0", (0.034, 0.0, 0.0)),),
    ),
    "robocasa_potato": ObjectVariant(
        name="robocasa_potato",
        label="potato",
        asset_directory="objects/objaverse/potato/potato_13",
        asset_files=(
            "model.xml",
            "visual/model_normalized_0.obj",
            "visual/material.mtl",
            "visual/material_0.png",
            "visual/image0.png",
        ),
        mass=0.15,
        inertia=(0.000197, 0.0000962, 0.000191),
        rest_height=0.03208,
        fitted_gripper_opening=0.602,
        primitives=(
            PrimitiveSpec(
                "capsule_0",
                (0.029, 0.025, 0.0),
                quat=(0.7071068, 0.7071068, 0.0, 0.0),
            ),
        ),
    ),
    "robocasa_mug": ObjectVariant(
        name="robocasa_mug",
        label="mug",
        asset_directory="objects/objaverse/mug/mug_10",
        asset_files=(
            "model.xml",
            "visual/model_normalized_0.obj",
            "visual/material.mtl",
            "visual/material_0.png",
            "visual/image0.png",
        ),
        mass=0.22,
        inertia=(0.000343, 0.000269, 0.000471),
        rest_height=0.03073,
        fitted_gripper_opening=0.768,
        primitives=(
            PrimitiveSpec("cylinder_0", (0.034, 0.031, 0.0)),
            PrimitiveSpec("capsule_0", (0.006, 0.018, 0.0), (0.045, 0.0, 0.0)),
        ),
    ),
    "robocasa_plate": ObjectVariant(
        name="robocasa_plate",
        label="plate",
        asset_directory="objects/objaverse/plate/plate_4",
        asset_files=(
            "model.xml",
            "visual/model_normalized_0.obj",
            "visual/material.mtl",
            "visual/material_0.jpeg",
            "visual/image0.png",
        ),
        mass=0.22,
        inertia=(0.000622, 0.000622, 0.001215),
        rest_height=0.01426,
        fitted_gripper_opening=0.0,
        primitives=(PrimitiveSpec("cylinder_0", (0.091, 0.0143, 0.0)),),
    ),
    "robocasa_bowl": ObjectVariant(
        name="robocasa_bowl",
        label="bowl",
        asset_directory="objects/objaverse/bowl/bowl_0",
        asset_files=(
            "model.xml",
            "visual/model_normalized_0.obj",
            "visual/material.mtl",
            "visual/material_0.jpeg",
            "visual/image0.png",
        ),
        mass=0.20,
        inertia=(0.000296, 0.000296, 0.000546),
        rest_height=0.01840,
        fitted_gripper_opening=0.0,
        primitives=(
            PrimitiveSpec("cylinder_0", (0.064, 0.009, 0.0), (0.0, 0.0, -0.009)),
            PrimitiveSpec("box_0", (0.005, 0.057, 0.015), (0.061, 0.0, 0.003)),
            PrimitiveSpec("box_1", (0.005, 0.057, 0.015), (-0.061, 0.0, 0.003)),
            PrimitiveSpec("box_2", (0.057, 0.005, 0.015), (0.0, 0.061, 0.003)),
            PrimitiveSpec("box_3", (0.057, 0.005, 0.015), (0.0, -0.061, 0.003)),
        ),
    ),
}


def slot_geom_name(slot: int, geom_slot: str) -> str:
    if slot < 0 or slot >= 4:
        raise IndexError(slot)
    if geom_slot not in GEOM_SLOT_NAMES:
        raise KeyError(geom_slot)
    return f"mjwarp_slot_{slot}_{geom_slot}"


def catalog_id(name: str | None) -> int:
    if name is None or not str(name).strip():
        return INACTIVE_CATALOG_ID
    normalized = str(name).strip().lower()
    aliases: dict[str, str] = {}
    for catalog, variant in OBJECT_VARIANTS.items():
        aliases[variant.label] = catalog
        aliases[variant.label.replace(" ", "_")] = catalog
    normalized = aliases.get(normalized, normalized)
    if normalized not in CATALOG_TO_ID:
        raise KeyError(
            f"Unsupported MJWarp object catalog {name!r}; supported values are "
            f"{', '.join(ACTIVE_CDPR_CATALOGS)}."
        )
    return CATALOG_TO_ID[normalized]


def catalog_ids(names: Iterable[str | None]) -> list[int]:
    return [catalog_id(name) for name in names]


def robocasa_asset_root(xml_path: Path) -> Path:
    """Return the repository-staged RoboCasa root for the fixed MJCF."""

    xml = Path(xml_path).expanduser().resolve()
    repo_root = xml.parents[3]
    return repo_root / "assets" / "externals" / "robocasa"


def required_asset_paths(xml_path: Path) -> tuple[Path, ...]:
    """Return only visual assets; collision topology is native MJCF geometry."""

    root = robocasa_asset_root(xml_path)
    paths: list[Path] = []
    for catalog in ACTIVE_CDPR_CATALOGS:
        variant = OBJECT_VARIANTS[catalog]
        directory = root / variant.asset_directory
        paths.extend(directory / relative for relative in variant.asset_files)
    return tuple(paths)


def validate_object_assets(xml_path: Path) -> tuple[Path, ...]:
    """Fail early when the curated RoboCasa visual subset is not staged."""

    paths = required_asset_paths(xml_path)
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:12])
        suffix = (
            f"\n  ... and {len(missing) - 12} more"
            if len(missing) > 12
            else ""
        )
        raise FileNotFoundError(
            "The real-object MJWarp scene requires the curated RoboCasa "
            "visual subset. Run `python scripts/stage_cdpr_robocasa_assets.py` "
            f"before starting MJ-Lab. Missing files:\n{preview}{suffix}"
        )
    return paths


def object_assets_sha256(xml_path: Path) -> str:
    """Content fingerprint for the active visual meshes and textures."""

    hasher = hashlib.sha256()
    root = robocasa_asset_root(xml_path)
    for path in validate_object_assets(xml_path):
        relative = path.relative_to(root).as_posix()
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _configure_variant_spec(spec: object, variant: ObjectVariant) -> None:
    primitive_by_name = {
        primitive.primitive: primitive for primitive in variant.primitives
    }
    for slot in range(4):
        visual = next(
            geom
            for geom in spec.geoms
            if geom.name == slot_geom_name(slot, VISUAL_MESH_SLOT)
        )
        visual.meshname = variant.visual_mesh_name
        visual.material = variant.material_name
        for primitive_name in COLLISION_GEOM_SLOT_NAMES:
            geom = next(
                item
                for item in spec.geoms
                if item.name == slot_geom_name(slot, primitive_name)
            )
            primitive = primitive_by_name.get(primitive_name)
            if primitive is None:
                geom.size = (1.0e-4, 1.0e-4, 1.0e-4)
                # A plane is a half-space: parking below it creates a deep
                # penetration and launches the body. Above the workspace is
                # contact-free while preserving fixed topology.
                geom.pos = (0.0, 0.0, 10.0)
                geom.quat = (1.0, 0.0, 0.0, 0.0)
            else:
                geom.size = primitive.size
                geom.pos = primitive.pos
                geom.quat = primitive.quat
        body = spec.find_body(f"mjwarp_object_slot_{slot}")
        body.mass = variant.mass
        body.inertia = variant.inertia
        body.explicitinertial = True


def compile_catalog_variant_models(mujoco: object, xml_path: Path) -> dict[str, object]:
    """Compile fixed-topology models for every visual/collider variant.

    Per-world switching copies one visual mesh reference plus primitive size
    and transform fields. No collision mesh is loaded or selected at runtime.
    """

    validate_object_assets(xml_path)
    resolved_xml = str(Path(xml_path).resolve())
    compiled: dict[str, object] = {}
    for catalog in ACTIVE_CDPR_CATALOGS:
        spec = mujoco.MjSpec.from_file(resolved_xml)
        _configure_variant_spec(spec, OBJECT_VARIANTS[catalog])
        compiled[catalog] = spec.compile()
    return compiled
