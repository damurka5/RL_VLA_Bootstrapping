from __future__ import annotations

from dataclasses import dataclass
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

ACTIVE_CDPR_CATALOGS: tuple[str, ...] = (
    "ycb_apple",
    "ycb_pear",
    "ycb_peach",
    "ycb_b_cups",
    "ycb_baseball",
    "plate",
    "bowl",
)
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
    mass: float
    inertia: tuple[float, float, float]
    rest_height: float
    rgba: tuple[float, float, float, float]
    primitives: tuple[PrimitiveSpec, ...]


# Collision dimensions, masses, and inertias are copied from the checked-in
# stable object packs.  The YCB meshes are intentionally represented by their
# existing collision primitives so every world has identical topology.
OBJECT_VARIANTS: dict[str, ObjectVariant] = {
    "ycb_apple": ObjectVariant(
        "ycb_apple",
        0.11,
        (0.000064, 0.000064, 0.000064),
        0.038,
        (0.76, 0.08, 0.07, 1.0),
        (PrimitiveSpec("sphere_0", (0.038, 0.0, 0.0)),),
    ),
    "ycb_pear": ObjectVariant(
        "ycb_pear",
        0.10,
        (0.000065, 0.000065, 0.000051),
        0.040,
        (0.63, 0.72, 0.10, 1.0),
        (
            PrimitiveSpec("sphere_0", (0.036, 0.0, 0.0), (0.0, 0.0, -0.004)),
            PrimitiveSpec("sphere_1", (0.030, 0.0, 0.0), (0.0, 0.0, 0.019)),
            PrimitiveSpec("sphere_2", (0.018, 0.0, 0.0), (0.0, 0.0, 0.044)),
        ),
    ),
    "ycb_peach": ObjectVariant(
        "ycb_peach",
        0.09,
        (0.000042, 0.000042, 0.000042),
        0.034,
        (0.95, 0.47, 0.28, 1.0),
        (PrimitiveSpec("sphere_0", (0.034, 0.0, 0.0)),),
    ),
    "ycb_b_cups": ObjectVariant(
        "ycb_b_cups",
        0.12,
        (0.00012, 0.00012, 0.000074),
        0.050,
        (0.05, 0.55, 0.70, 1.0),
        (
            PrimitiveSpec("cylinder_0", (0.038, 0.050, 0.0)),
            PrimitiveSpec(
                "capsule_0",
                (0.006, 0.020, 0.0),
                (0.050, 0.0, 0.004),
                (0.9238795, 0.0, 0.3826834, 0.0),
            ),
        ),
    ),
    "ycb_baseball": ObjectVariant(
        "ycb_baseball",
        0.09,
        (0.000047, 0.000047, 0.000047),
        0.036,
        (0.95, 0.95, 0.90, 1.0),
        (PrimitiveSpec("sphere_0", (0.036, 0.0, 0.0)),),
    ),
    "plate": ObjectVariant(
        "plate",
        0.16,
        (0.00038, 0.00038, 0.00072),
        0.012,
        (0.12, 0.35, 0.85, 1.0),
        (PrimitiveSpec("cylinder_0", (0.085, 0.012, 0.0)),),
    ),
    "bowl": ObjectVariant(
        "bowl",
        0.18,
        (0.00045, 0.00045, 0.00082),
        0.024,
        (0.88, 0.88, 0.82, 1.0),
        (
            PrimitiveSpec("cylinder_0", (0.070, 0.014, 0.0)),
            PrimitiveSpec("box_0", (0.007, 0.062, 0.024), (0.069, 0.0, 0.014)),
            PrimitiveSpec("box_1", (0.007, 0.062, 0.024), (-0.069, 0.0, 0.014)),
            PrimitiveSpec("box_2", (0.062, 0.007, 0.024), (0.0, 0.069, 0.014)),
            PrimitiveSpec("box_3", (0.062, 0.007, 0.024), (0.0, -0.069, 0.014)),
        ),
    ),
}


def catalog_id(name: str | None) -> int:
    if name is None or not str(name).strip():
        return INACTIVE_CATALOG_ID
    normalized = str(name).strip().lower()
    aliases = {
        "apple": "ycb_apple",
        "pear": "ycb_pear",
        "peach": "ycb_peach",
        "cups": "ycb_b_cups",
        "baseball": "ycb_baseball",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in CATALOG_TO_ID:
        raise KeyError(
            f"Unsupported MJWarp object catalog {name!r}; supported values are "
            f"{', '.join(ACTIVE_CDPR_CATALOGS)}."
        )
    return CATALOG_TO_ID[normalized]


def catalog_ids(names: Iterable[str | None]) -> list[int]:
    return [catalog_id(name) for name in names]


def build_variant_arrays(ids: object) -> dict[str, object]:
    """Build host arrays for per-world fixed-slot MJWarp model fields.

    `ids` has shape ``[world, 4]``.  The primitive topology is fixed and the
    arrays only change sizes, local transforms, colors, mass, and inertia.
    Inactive primitives are tiny and placed above their owning body.  Moving
    them below the workspace is unsafe because MuJoCo planes are infinite:
    a geom below the floor is deeply penetrating it, not collision-disabled.
    """

    import numpy as np

    catalog_array = np.asarray(ids, dtype=np.int32)
    if catalog_array.ndim != 2 or catalog_array.shape[1] != 4:
        raise ValueError(
            f"catalog ids must have shape [world, 4], got {catalog_array.shape}."
        )
    nworld, nslot = catalog_array.shape
    nprimitive = len(PRIMITIVE_NAMES)
    sizes = np.full((nworld, nslot, nprimitive, 3), 1.0e-4, dtype=np.float32)
    positions = np.zeros_like(sizes)
    positions[..., 2] = 10.0
    quaternions = np.zeros((nworld, nslot, nprimitive, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    rgba = np.zeros((nworld, nslot, nprimitive, 4), dtype=np.float32)
    mass = np.full((nworld, nslot), 1.0e-4, dtype=np.float32)
    inertia = np.full((nworld, nslot, 3), 1.0e-8, dtype=np.float32)
    rest_height = np.zeros((nworld, nslot), dtype=np.float32)

    primitive_index = {name: index for index, name in enumerate(PRIMITIVE_NAMES)}
    for world_index in range(nworld):
        for slot_index in range(nslot):
            object_id = int(catalog_array[world_index, slot_index])
            if object_id == INACTIVE_CATALOG_ID:
                continue
            if object_id not in ID_TO_CATALOG:
                raise KeyError(f"Unknown catalog id {object_id}.")
            variant = OBJECT_VARIANTS[ID_TO_CATALOG[object_id]]
            mass[world_index, slot_index] = variant.mass
            inertia[world_index, slot_index] = variant.inertia
            rest_height[world_index, slot_index] = variant.rest_height
            for primitive in variant.primitives:
                index = primitive_index[primitive.primitive]
                sizes[world_index, slot_index, index] = primitive.size
                positions[world_index, slot_index, index] = primitive.pos
                quaternions[world_index, slot_index, index] = primitive.quat
                rgba[world_index, slot_index, index] = variant.rgba

    return {
        "catalog_ids": catalog_array,
        "geom_size": sizes,
        "geom_pos": positions,
        "geom_quat": quaternions,
        "geom_rgba": rgba,
        "body_mass": mass,
        "body_inertia": inertia,
        "rest_height": rest_height,
    }
