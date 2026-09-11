"""Explicit full-task ``put_into`` scenes, generated and validated up front.

Why this is a manifest and not another resetter branch
------------------------------------------------------

Every start distribution in this campaign so far has been a function of a
checkpoint's stored curriculum: ``BatchedReverseFrontierResetter`` reads the
approach cap, the caught fraction and the shell out of the restored training
state, and the scene follows from those. That is right for GRPO and wrong for a
demonstration bank, for three reasons the campaign has already paid for.

A checkpoint's curriculum owns the geometry. Three teachers with three
curricula would produce three different scenes for what is supposed to be one
episode, and the pickup teacher's own reset would silently redefine the scene
the move-to teacher just finished approaching.

The container reset overwrites the gripper XY. ``uncaught_container`` places
the end-effector one centimetre above the object -- there is no approach at
all. 92.6% of composed plate episodes measured under it begin with the object
already inside the success radius, so the metric moved with the reset rather
than with the policy. A full-task demonstration has to begin away from the
object, and that is a property of the SCENE, not of a curriculum rung.

Scenes must be splittable and reproducible. Teacher selection, student
validation and the final test have to run on disjoint scene sets, and a split
is only meaningful if a scene has a stable identity. Here that identity is a
hash of the realized geometry, so a scene means the same thing in the
collector, in the dataset builder and in the evaluation.

What a scene fixes, and what it deliberately does not
-----------------------------------------------------

Fixed: which catalogs occupy which slots, where they rest, their yaws, where
the end-effector starts, its yaw and opening, and the visual variant. Realized
distances are stored alongside so nothing downstream has to recompute them
from a rule that might drift.

Not fixed: anything about the policy. A scene is a start, not a trajectory,
and the same scene run under two teachers is two rollouts of one start.

Geometric acceptance
--------------------

A scene is REJECTED and resampled rather than clamped. Clamping is how a
sampler quietly makes a task easier than its declared bounds: the phase-4
container reset clamps its object into the workspace box after sampling a
distance, so the realized distance is smaller than the sampled one and the
logged cap is not the cap that ran.

The tests applied here, all in metres:

* end-effector to target XY inside ``approach_xy_bounds``, and the EE height
  inside ``ee_z_bounds``;
* target to destination XY at least ``destination_radius + transport_margin``
  and at most ``transport_xy_max``, so the object starts OUTSIDE the success
  region by a real margin for both receptacles;
* no mesh overlap between any two objects, using rotation-invariant XY
  circumradii from the catalog primitives rather than a centre-distance rule --
  a centre-distance predicate does not stop an apple resting on a bowl rim;
* every object inside the workspace AND inside the overview camera's coverage
  half-extent, with no post-hoc clamping;
* the gripper starts open, empty, and clear of the target in 3-D.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
    ACTIVE_CDPR_CATALOGS,
    BOWL_CATALOG,
    CATALOG_TO_ID,
    GRASPABLE_CDPR_CATALOGS,
    INACTIVE_CATALOG_ID,
    OBJECT_VARIANTS,
    PLATE_CATALOG,
)


SCENE_MANIFEST_VERSION = "cdpr_three_stage_put_into/v1"

DESTINATION_CATALOGS: dict[str, str] = {
    "plate": PLATE_CATALOG,
    "bowl": BOWL_CATALOG,
}

# The production success radii, restated here ONLY as defaults for a caller
# that has no config in hand. Scene generation must not loosen success to
# improve yield, so a caller with a config passes its own values through
# SceneGeometryConfig and these are never consulted.
PRODUCTION_SUCCESS_RADIUS: dict[str, float] = {"plate": 0.091, "bowl": 0.057}

# Slot layout for a full-task scene. Manipulation tasks in the production
# collector assume slot 0 is the handled object and slot 1 the receptacle the
# reward and the predicate measure against; the ambiguity mode adds the OTHER
# receptacle at slot 2, which is exactly where the production reset puts it.
TARGET_SLOT = 0
DESTINATION_SLOT = 1
SECOND_RECEPTACLE_SLOT = 2
DISTRACTOR_SLOT = 3
OBJECT_SLOTS = 4

SPLIT_NAMES: tuple[str, ...] = (
    "collection",
    "teacher_selection",
    "student_validation",
    "final_test",
)


def _quaternion_matrix(quat: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """Row-major rotation matrix for a (w, x, y, z) quaternion."""

    w, x, y, z = (float(value) for value in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("A primitive carries a zero quaternion.")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _primitive_half_extents(primitive: str, size: Sequence[float]) -> tuple[float, float, float]:
    """Local axis-aligned half extents of one collision primitive.

    MuJoCo's conventions, not a guess: a sphere takes one radius, a cylinder
    takes (radius, half-length) about local Z, a capsule the same with the caps
    added to the length, and a box three half sizes.
    """

    sx, sy, sz = (float(value) for value in size)
    if primitive.startswith("sphere"):
        return (sx, sx, sx)
    if primitive.startswith("cylinder"):
        return (sx, sx, sy)
    if primitive.startswith("capsule"):
        return (sx, sx, sy + sx)
    if primitive.startswith("box"):
        return (sx, sy, sz)
    raise ValueError(f"Unknown collision primitive {primitive!r}.")


def catalog_xy_radius(catalog: str) -> float:
    """Rotation-invariant XY circumradius of a catalog's collision geometry.

    Rotation-invariant because scene objects are spawned at a sampled yaw: the
    quantity an overlap test needs is the largest horizontal distance from the
    body origin to any point of the collision hull, which does not depend on
    that yaw.

    Each primitive is bounded in its OWN natural form rather than by its
    axis-aligned box, and the difference is not cosmetic. The plate is a
    cylinder of radius 0.091 and half-height 0.01; boxing it gives
    hypot(0.091, 0.091) = 0.129, a 41% overstatement that makes a plate scene
    unplaceable inside the camera envelope. Spheres and round shafts are round.
    """

    variant = OBJECT_VARIANTS[catalog]
    radius = 0.0
    for primitive in variant.primitives:
        name = primitive.primitive
        size = primitive.size
        rotation = _quaternion_matrix(primitive.quat)
        centre = (float(primitive.pos[0]), float(primitive.pos[1]))
        if name.startswith("sphere"):
            radius = max(radius, math.hypot(*centre) + float(size[0]))
            continue
        if name.startswith(("cylinder", "capsule")):
            # A swept segment along the primitive's local Z, of radius size[0]
            # and half-length size[1]. Its XY reach is the farther endpoint
            # plus the radius; a capsule's caps have the same radius, so no
            # extra term is needed.
            shaft = float(size[0])
            half_length = float(size[1])
            axis_x = rotation[0][2] * half_length
            axis_y = rotation[1][2] * half_length
            reach = max(
                math.hypot(centre[0] + axis_x, centre[1] + axis_y),
                math.hypot(centre[0] - axis_x, centre[1] - axis_y),
            )
            radius = max(radius, reach + shaft)
            continue
        if not name.startswith("box"):
            raise ValueError(f"Unknown collision primitive {name!r}.")
        half = _primitive_half_extents(name, size)
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                for sign_z in (-1.0, 1.0):
                    local = (
                        sign_x * half[0],
                        sign_y * half[1],
                        sign_z * half[2],
                    )
                    x = sum(rotation[0][i] * local[i] for i in range(3))
                    y = sum(rotation[1][i] * local[i] for i in range(3))
                    radius = max(
                        radius, math.hypot(centre[0] + x, centre[1] + y)
                    )
    return radius


def catalog_top_height(catalog: str) -> float:
    """How far the catalog's collision hull reaches above the body origin."""

    variant = OBJECT_VARIANTS[catalog]
    top = 0.0
    for primitive in variant.primitives:
        half = _primitive_half_extents(primitive.primitive, primitive.size)
        rotation = _quaternion_matrix(primitive.quat)
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                for sign_z in (-1.0, 1.0):
                    local = (
                        sign_x * half[0],
                        sign_y * half[1],
                        sign_z * half[2],
                    )
                    z = sum(rotation[2][i] * local[i] for i in range(3))
                    top = max(top, float(primitive.pos[2]) + z)
    return top


@dataclass(frozen=True)
class SceneGeometryConfig:
    """Declared bounds for a full-task scene. Nothing here is clamped."""

    # Which catalogs may be the named target, and which receptacles are used.
    target_catalogs: tuple[str, ...] = (
        "robocasa_apple",
        "robocasa_orange",
        "robocasa_potato",
        "robocasa_tomato",
    )
    destinations: tuple[str, ...] = ("plate", "bowl")
    # Production success radii. Passed in from the run config; a scene must
    # never widen them to make demonstrations easier to obtain.
    destination_success_radius: Mapping[str, float] = field(
        default_factory=lambda: dict(PRODUCTION_SUCCESS_RADIUS)
    )

    # Approach: how far the empty gripper starts from the object, in XY.
    # 0.06-0.10 is deliberately OUTSIDE the native reaching success region
    # (move_to succeeds inside 0.02-0.08 depending on the window), so the
    # move-to teacher has a real approach to perform.
    approach_xy_bounds: tuple[float, float] = (0.06, 0.10)
    # A declared SAFE band: above every catalog's resting top (the mug reaches
    # 0.24 m with the desk at 0.15) and below the controller ceiling, so the
    # empty gripper starts clear of the scene and the yaw tail can rotate the
    # fingers without sweeping an object. The phase-7 reset band is
    # [0.19, 0.32]; this is its upper half.
    ee_z_bounds: tuple[float, float] = (0.26, 0.32)
    # Minimum 3-D clearance from the gripper's grasp point at reset, so no
    # scene begins with the pads already bracketing the object.
    min_ee_target_xyz_distance: float = 0.05

    # Transport: how far the object starts from its receptacle, in XY. The
    # MINIMUM is per-destination -- radius + margin -- so plate and bowl get
    # different centre distances but the same outside-radius margin.
    transport_margin: float = 0.04
    transport_xy_max: float = 0.18

    # Mesh separation on top of the circumradius sum. A pure circumradius test
    # allows two hulls to touch; this keeps a gap.
    min_object_gap: float = 0.015

    # Workspace and framing. The bounds are the END-EFFECTOR's, matching the
    # phase-7 reset box and comfortably inside the controller's own +-0.28
    # clamp -- a start outside the clamp would be silently moved, which is the
    # clamping this generator exists to avoid.
    #
    # Objects are bounded separately, by the camera: max |coord| 0.205 is the
    # envelope the dollied-in overview camera still covers at the near desk
    # edge, and the constraint is applied to the object's HULL, not its centre.
    # A plate centred at 0.19 has 0.091 m of itself outside the frame.
    workspace_x_bounds: tuple[float, float] = (-0.19, 0.19)
    workspace_y_bounds: tuple[float, float] = (-0.19, 0.19)
    camera_half_extent: float = 0.205
    support_surface_z: float = 0.15
    # The pad offset below the body ``ee_position`` tracks, measured from the
    # MJCF. The grasp point of an object at rest sits this far above it.
    pick_grasp_height_offset: float = 0.0075

    # Optional coverage strata.
    include_second_receptacle: bool = False
    distractor_catalogs: tuple[str, ...] = ()

    # Pickup-yaw feasibility, OFF by default.
    #
    # The gripper's yaw is pinned to one calibrated angle for pickup while the
    # object's yaw is sampled, so an elongated catalog presents a different
    # width to the closing axis in every scene. A potato drawn broadside has
    # NEGATIVE centring slack: the open fingers cannot bracket it at all, and
    # no reach, however accurate, can be followed by a grasp. Screens measured
    # this on roughly half of potato presentations.
    #
    # When this holds the calibrated pickup yaw, such a presentation is
    # rejected at generation time and the draw is retried. Because
    # ``generate_scenes`` balances by cycling over catalogs, the effect is that
    # potato scenes are RESAMPLED into graspable orientations, not dropped --
    # the census keeps its strata and stops containing impossible work.
    #
    # None by default because the pickup yaw is a calibration, not a property
    # of the scene: leaving it unset keeps every manifest written before this
    # filter existed validating unchanged, and makes the coupling explicit in
    # the manifest of any set that was filtered.
    #
    # The prediction is made at the COMMANDED orientation the full-task reset
    # writes. The collector measures the SETTLED quaternion and reports any
    # scene where the two disagree about feasibility.
    clearance_pickup_yaw: float | None = None
    # Matches StageReadiness.grasp_xy_margin. A scene accepted against a wider
    # margin than the collector enforces would hand back the same impossible
    # presentations through the other door.
    grasp_xy_margin: float = 0.003

    # Rejection budget. A generator that cannot meet its bounds must say so
    # rather than relax them.
    max_attempts_per_scene: int = 512

    def __post_init__(self) -> None:
        unknown = [
            name
            for name in self.target_catalogs
            if name not in GRASPABLE_CDPR_CATALOGS
        ]
        if unknown:
            raise ValueError(
                f"Target catalogs {unknown} are not graspable; a full-task "
                "scene must name an object the gripper can pick up."
            )
        bad = [name for name in self.destinations if name not in DESTINATION_CATALOGS]
        if bad:
            raise ValueError(f"Unknown destinations {bad}.")
        for name in self.destinations:
            if name not in self.destination_success_radius:
                raise ValueError(
                    f"No production success radius supplied for {name!r}."
                )
        low, high = self.approach_xy_bounds
        if not 0.0 < low <= high:
            raise ValueError("approach_xy_bounds must be a positive interval.")
        if self.ee_z_bounds[0] > self.ee_z_bounds[1]:
            raise ValueError("ee_z_bounds must be ordered.")
        for name in self.distractor_catalogs:
            if name not in ACTIVE_CDPR_CATALOGS:
                raise ValueError(f"Unknown distractor catalog {name!r}.")

    def transport_min(self, destination: str) -> float:
        return (
            float(self.destination_success_radius[destination])
            + float(self.transport_margin)
        )


@dataclass(frozen=True)
class SceneObject:
    slot: int
    catalog: str
    xy: tuple[float, float]
    z: float
    yaw: float

    @property
    def catalog_id(self) -> int:
        return int(CATALOG_TO_ID[self.catalog])


@dataclass(frozen=True)
class CompositionScene:
    """One complete, verified full-task start."""

    scene_uid: str
    split: str
    scene_index: int
    seed: int
    target_catalog: str
    destination: str
    objects: tuple[SceneObject, ...]
    ee_xyz: tuple[float, float, float]
    ee_yaw: float
    gripper_opening: float
    approach_xy_distance: float
    approach_xyz_distance: float
    transport_xy_distance: float
    destination_success_radius: float
    texture_id: int
    background_rgba: tuple[float, float, float, float]
    shade: float
    manifest_version: str = SCENE_MANIFEST_VERSION

    def object_at(self, slot: int) -> SceneObject | None:
        for entry in self.objects:
            if entry.slot == int(slot):
                return entry
        return None

    @property
    def target(self) -> SceneObject:
        entry = self.object_at(TARGET_SLOT)
        if entry is None:  # pragma: no cover - constructed invariant
            raise ValueError(f"{self.scene_uid} has no target object.")
        return entry

    @property
    def receptacle(self) -> SceneObject:
        entry = self.object_at(DESTINATION_SLOT)
        if entry is None:  # pragma: no cover - constructed invariant
            raise ValueError(f"{self.scene_uid} has no receptacle.")
        return entry

    def catalog_ids(self) -> list[int]:
        """Per-slot catalog ids, with unused slots explicitly disabled."""

        ids = [INACTIVE_CATALOG_ID] * OBJECT_SLOTS
        for entry in self.objects:
            ids[entry.slot] = entry.catalog_id
        return ids

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objects"] = [asdict(entry) for entry in self.objects]
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CompositionScene":
        values = dict(payload)
        values["objects"] = tuple(
            SceneObject(
                slot=int(entry["slot"]),
                catalog=str(entry["catalog"]),
                xy=(float(entry["xy"][0]), float(entry["xy"][1])),
                z=float(entry["z"]),
                yaw=float(entry["yaw"]),
            )
            for entry in payload["objects"]
        )
        values["ee_xyz"] = tuple(float(v) for v in payload["ee_xyz"])
        values["background_rgba"] = tuple(
            float(v) for v in payload["background_rgba"]
        )
        return cls(**values)


def _scene_uid(
    *,
    target: SceneObject,
    receptacle: SceneObject,
    others: Sequence[SceneObject],
    ee_xyz: Sequence[float],
    ee_yaw: float,
    destination: str,
) -> str:
    """Identity from the REALIZED geometry, not from a counter.

    A counter would let two runs with different bounds mint the same name for
    different scenes, and the split assignment is keyed off this string. Rounded
    to a tenth of a millimetre so a float replay that reproduces the geometry
    reproduces the identity.
    """

    def q(value: float) -> str:
        return f"{float(value):+.4f}"

    parts = [
        SCENE_MANIFEST_VERSION,
        destination,
        target.catalog,
        q(target.xy[0]),
        q(target.xy[1]),
        q(target.yaw),
        receptacle.catalog,
        q(receptacle.xy[0]),
        q(receptacle.xy[1]),
        q(receptacle.yaw),
    ]
    for entry in sorted(others, key=lambda item: item.slot):
        parts.extend(
            [str(entry.slot), entry.catalog, q(entry.xy[0]), q(entry.xy[1]), q(entry.yaw)]
        )
    parts.extend([q(ee_xyz[0]), q(ee_xyz[1]), q(ee_xyz[2]), q(ee_yaw)])
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"scene_{digest[:16]}"


class SceneRejection(ValueError):
    """A geometry proposal that failed a declared bound."""


# Half the gap between the finger pads with the gripper fully open, in metres,
# measured from the MJCF: the finger slide range is [0, 0.03] on a pad whose
# body sits at x = +-0.02, and the pad geom is 0.0025 half-thick, so the inner
# faces sit at +-(0.02 + 0.03 - 0.0025) = +-0.0475. Both fingers are driven --
# a weld equality couples finger_l and finger_r -- so the aperture is centred
# on ee_base and is 0.095 m wide, which is the "0.0969 m open gap" the object
# catalog's banana/mug note refers to.
#
# `tests/test_cdpr_staged_put_into.py` re-derives this from the model with
# MuJoCo and fails if it moves, because it is a geometry fact that decides
# which objects can be grasped at all and it must not drift into a config.
#
# It lives here, beside the object hulls it is compared against, so that scene
# generation can reject an ungraspable presentation without importing the
# policy stack -- the staged collector re-exports it for its own callers.
OPEN_GRIPPER_HALF_APERTURE_M = 0.0475


def max_grasp_xy_offset(catalog: str, *, margin: float = 0.003) -> float:
    """How far off-centre the gripper may be and still bracket this object.

    The half-aperture minus the object's rotation-invariant hull radius, minus
    a margin. Rotation-invariant because the object's yaw is sampled and the
    gripper's is pinned, so the object may present any of its widths to the
    closing axis; taking the widest is the only bound that holds for every
    draw. A catalog whose widest presentation exceeds the aperture returns a
    NEGATIVE slack and cannot be grasped at this yaw at all -- which is the
    same geometry fact that removed banana and mug from the target pool.
    """

    return (
        OPEN_GRIPPER_HALF_APERTURE_M
        - catalog_xy_radius(catalog)
        - float(margin)
    )


def projected_grasp_xy_offset(catalog: str, object_quaternion: Sequence[float],
                              gripper_yaw: float, *, margin: float = 0.003) -> float:
    """Conservative radial centering slack for this object's actual presentation.

    Project the primitive hull onto the fixed gripper's closing axis. A long
    potato can exceed the aperture along Y while still fitting between X pads;
    its XY circumradius is not its width along the closing axis.
    """

    world_axis = (math.cos(gripper_yaw), math.sin(gripper_yaw), 0.0)
    rotation = _quaternion_matrix(object_quaternion)
    axis = [sum(rotation[j][i] * world_axis[j] for j in range(3)) for i in range(3)]
    extent = 0.0
    for primitive in OBJECT_VARIANTS[catalog].primitives:
        rotation = _quaternion_matrix(primitive.quat)
        direction = [sum(rotation[j][i] * axis[j] for j in range(3)) for i in range(3)]
        centre = sum(axis[i] * primitive.pos[i] for i in range(3))
        name, size = primitive.primitive, primitive.size
        if name.startswith("sphere"):
            radius = float(size[0])
        elif name.startswith("capsule"):
            radius = float(size[0]) + abs(direction[2]) * float(size[1])
        elif name.startswith("cylinder"):
            radius = math.hypot(*direction[:2]) * float(size[0]) + abs(direction[2]) * float(size[1])
        elif name.startswith("box"):
            radius = sum(abs(direction[i]) * float(size[i]) for i in range(3))
        else:
            raise ValueError(f"Unsupported grasp primitive {name}")
        extent = max(extent, abs(centre) + radius)
    return OPEN_GRIPPER_HALF_APERTURE_M - extent - float(margin)


def scene_object_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """The (w, x, y, z) the full-task reset writes for a scene object's yaw.

    Kept beside the check that consumes it so the manifest-time feasibility
    prediction and ``FullTaskSceneReset`` cannot drift apart. This is the
    COMMANDED orientation; what the collector reads back is the settled one.
    """

    return (math.cos(0.5 * float(yaw)), 0.0, 0.0, math.sin(0.5 * float(yaw)))


def validate_scene(scene: CompositionScene, config: SceneGeometryConfig) -> None:
    """Re-derive every acceptance test from the stored geometry.

    Called by the generator on what it just built AND by every consumer on what
    it just loaded, so a manifest that was hand-edited, or written by an older
    generator, cannot enter a collection run unchecked.
    """

    target = scene.target
    receptacle = scene.receptacle
    if scene.destination not in config.destinations:
        raise SceneRejection(
            f"{scene.scene_uid}: destination {scene.destination!r} is not in "
            f"{list(config.destinations)}."
        )
    if receptacle.catalog != DESTINATION_CATALOGS[scene.destination]:
        raise SceneRejection(
            f"{scene.scene_uid}: destination {scene.destination!r} but slot "
            f"{DESTINATION_SLOT} holds {receptacle.catalog!r}."
        )
    if target.catalog not in config.target_catalogs:
        raise SceneRejection(
            f"{scene.scene_uid}: target {target.catalog!r} is not in the "
            "configured target catalogs."
        )
    if config.clearance_pickup_yaw is not None:
        slack = projected_grasp_xy_offset(
            target.catalog,
            scene_object_quaternion(target.yaw),
            float(config.clearance_pickup_yaw),
            margin=float(config.grasp_xy_margin),
        )
        if slack <= 0.0:
            raise SceneRejection(
                f"{scene.scene_uid}: {target.catalog} presented at yaw "
                f"{target.yaw:.4f} leaves {slack:+.4f} m of centring slack at "
                f"the calibrated pickup yaw {config.clearance_pickup_yaw:.4f}. "
                "The open fingers cannot bracket it in this orientation."
            )

    approach = math.dist(scene.ee_xyz[:2], target.xy)
    low, high = config.approach_xy_bounds
    if not (low - 1e-9) <= approach <= (high + 1e-9):
        raise SceneRejection(
            f"{scene.scene_uid}: approach XY {approach:.4f} outside "
            f"[{low}, {high}]."
        )
    if abs(approach - scene.approach_xy_distance) > 1e-4:
        raise SceneRejection(
            f"{scene.scene_uid}: stored approach {scene.approach_xy_distance} "
            f"disagrees with the geometry ({approach:.4f})."
        )
    if not (
        config.ee_z_bounds[0] - 1e-9
        <= scene.ee_xyz[2]
        <= config.ee_z_bounds[1] + 1e-9
    ):
        raise SceneRejection(
            f"{scene.scene_uid}: EE height {scene.ee_xyz[2]:.4f} outside "
            f"{config.ee_z_bounds}."
        )
    grasp_point = (
        target.xy[0],
        target.xy[1],
        target.z + float(config.pick_grasp_height_offset),
    )
    clearance = math.dist(scene.ee_xyz, grasp_point)
    if clearance < float(config.min_ee_target_xyz_distance) - 1e-9:
        raise SceneRejection(
            f"{scene.scene_uid}: gripper starts {clearance:.4f} m from the "
            "grasp point; the episode would begin already aligned."
        )

    transport = math.dist(target.xy, receptacle.xy)
    minimum = config.transport_min(scene.destination)
    if transport < minimum - 1e-9:
        raise SceneRejection(
            f"{scene.scene_uid}: object starts {transport:.4f} m from the "
            f"{scene.destination}, inside the required {minimum:.4f} m "
            "(success radius plus margin)."
        )
    if transport > float(config.transport_xy_max) + 1e-9:
        raise SceneRejection(
            f"{scene.scene_uid}: transport {transport:.4f} m exceeds "
            f"{config.transport_xy_max}."
        )
    if abs(transport - scene.transport_xy_distance) > 1e-4:
        raise SceneRejection(
            f"{scene.scene_uid}: stored transport distance disagrees with the "
            "geometry."
        )
    if abs(
        scene.destination_success_radius
        - float(config.destination_success_radius[scene.destination])
    ) > 1e-9:
        raise SceneRejection(
            f"{scene.scene_uid}: stored success radius "
            f"{scene.destination_success_radius} is not the configured "
            f"{config.destination_success_radius[scene.destination]}. A scene "
            "must not loosen success to improve demonstration yield."
        )

    if not (
        config.workspace_x_bounds[0] - 1e-9
        <= scene.ee_xyz[0]
        <= config.workspace_x_bounds[1] + 1e-9
    ) or not (
        config.workspace_y_bounds[0] - 1e-9
        <= scene.ee_xyz[1]
        <= config.workspace_y_bounds[1] + 1e-9
    ):
        raise SceneRejection(
            f"{scene.scene_uid}: the gripper starts at {scene.ee_xyz[:2]}, "
            "outside the declared end-effector workspace. The backend would "
            "clamp it and the realized approach distance would not be the "
            "sampled one."
        )

    slots = [entry.slot for entry in scene.objects]
    if len(set(slots)) != len(slots):
        raise SceneRejection(f"{scene.scene_uid}: two objects share a slot.")
    for entry in scene.objects:
        reach = math.hypot(*entry.xy) + catalog_xy_radius(entry.catalog)
        if reach > float(config.camera_half_extent) + 1e-9:
            raise SceneRejection(
                f"{scene.scene_uid}: {entry.catalog} reaches {reach:.4f} m "
                f"from the desk centre, past the camera coverage "
                f"{config.camera_half_extent}."
            )
        expected_z = float(config.support_surface_z) + OBJECT_VARIANTS[
            entry.catalog
        ].rest_height
        if abs(entry.z - expected_z) > 1e-6:
            raise SceneRejection(
                f"{scene.scene_uid}: {entry.catalog} rests at {entry.z:.5f}, "
                f"not on the support surface ({expected_z:.5f}). A full-task "
                "scene starts with everything on the desk."
            )

    for first in range(len(scene.objects)):
        for second in range(first + 1, len(scene.objects)):
            a = scene.objects[first]
            b = scene.objects[second]
            gap = math.dist(a.xy, b.xy) - (
                catalog_xy_radius(a.catalog) + catalog_xy_radius(b.catalog)
            )
            if gap < float(config.min_object_gap) - 1e-9:
                raise SceneRejection(
                    f"{scene.scene_uid}: {a.catalog} and {b.catalog} are "
                    f"{gap:.4f} m apart at the hulls, under the required "
                    f"{config.min_object_gap}. A centre-distance rule would "
                    "have accepted this."
                )

    if scene.gripper_opening < 0.999:
        raise SceneRejection(
            f"{scene.scene_uid}: gripper opening {scene.gripper_opening} is "
            "not fully open; the episode must start with an empty open hand."
        )


def _sample_scene(
    rng: Any,
    config: SceneGeometryConfig,
    *,
    destination: str,
    target_catalog: str,
    scene_index: int,
    seed: int,
) -> CompositionScene:
    """One accepted proposal, or SceneRejection after the attempt budget."""

    receptacle_catalog = DESTINATION_CATALOGS[destination]
    radius = float(config.destination_success_radius[destination])
    transport_low = config.transport_min(destination)
    transport_high = float(config.transport_xy_max)
    if transport_low > transport_high:
        raise SceneRejection(
            f"{destination}: the required minimum transport distance "
            f"{transport_low:.4f} exceeds transport_xy_max "
            f"{transport_high:.4f}. Widen the cap rather than shrinking the "
            "margin -- the margin is what keeps the object outside the goal."
        )

    def place(catalog: str) -> tuple[float, float]:
        """A uniform draw over the disc this catalog's HULL fits inside.

        Sampling a square and rejecting is the same distribution but a far
        worse acceptance rate: the plate's admissible disc has radius 0.114 m
        against the 0.19 m reset box, so a square draw would be refused
        two thirds of the time before any of the other tests even ran.
        """

        limit = max(
            float(config.camera_half_extent) - catalog_xy_radius(catalog), 0.0
        )
        angle = float(rng.uniform(-math.pi, math.pi))
        # sqrt for a uniform AREA density; a linear radius draw would crowd
        # every scene toward the desk centre.
        distance = limit * math.sqrt(float(rng.random()))
        return (distance * math.cos(angle), distance * math.sin(angle))

    last_error = "no attempt was made"
    for _ in range(int(config.max_attempts_per_scene)):
        receptacle_xy = place(receptacle_catalog)
        transport = float(rng.uniform(transport_low, transport_high))
        bearing = float(rng.uniform(-math.pi, math.pi))
        target_xy = (
            receptacle_xy[0] + transport * math.cos(bearing),
            receptacle_xy[1] + transport * math.sin(bearing),
        )
        approach = float(rng.uniform(*config.approach_xy_bounds))
        approach_bearing = float(rng.uniform(-math.pi, math.pi))
        ee_xy = (
            target_xy[0] + approach * math.cos(approach_bearing),
            target_xy[1] + approach * math.sin(approach_bearing),
        )
        ee_z = float(rng.uniform(*config.ee_z_bounds))

        objects = [
            SceneObject(
                slot=TARGET_SLOT,
                catalog=target_catalog,
                xy=target_xy,
                z=float(config.support_surface_z)
                + OBJECT_VARIANTS[target_catalog].rest_height,
                yaw=float(rng.uniform(-math.pi, math.pi)),
            ),
            SceneObject(
                slot=DESTINATION_SLOT,
                catalog=receptacle_catalog,
                xy=receptacle_xy,
                z=float(config.support_surface_z)
                + OBJECT_VARIANTS[receptacle_catalog].rest_height,
                yaw=float(rng.uniform(-math.pi, math.pi)),
            ),
        ]
        if config.include_second_receptacle:
            other = "bowl" if destination == "plate" else "plate"
            other_catalog = DESTINATION_CATALOGS[other]
            objects.append(
                SceneObject(
                    slot=SECOND_RECEPTACLE_SLOT,
                    catalog=other_catalog,
                    xy=place(other_catalog),
                    z=float(config.support_surface_z)
                    + OBJECT_VARIANTS[other_catalog].rest_height,
                    yaw=float(rng.uniform(-math.pi, math.pi)),
                )
            )
        if config.distractor_catalogs:
            choice = str(
                config.distractor_catalogs[
                    int(rng.integers(len(config.distractor_catalogs)))
                ]
            )
            objects.append(
                SceneObject(
                    slot=DISTRACTOR_SLOT,
                    catalog=choice,
                    xy=place(choice),
                    z=float(config.support_surface_z)
                    + OBJECT_VARIANTS[choice].rest_height,
                    yaw=float(rng.uniform(-math.pi, math.pi)),
                )
            )

        # The EE yaw is ORDINARY -- uniform over the joint's range -- because
        # the student is evaluated from an ordinary yaw and the alignment is
        # the thing the demonstration has to teach. Seeding the calibrated yaw
        # here would hand the student the skill and then measure it.
        ee_yaw = float(rng.uniform(-math.pi, math.pi))
        scene = CompositionScene(
            scene_uid=_scene_uid(
                target=objects[0],
                receptacle=objects[1],
                others=objects[2:],
                ee_xyz=(ee_xy[0], ee_xy[1], ee_z),
                ee_yaw=ee_yaw,
                destination=destination,
            ),
            split="unassigned",
            scene_index=int(scene_index),
            seed=int(seed),
            target_catalog=target_catalog,
            destination=destination,
            objects=tuple(objects),
            ee_xyz=(ee_xy[0], ee_xy[1], ee_z),
            ee_yaw=ee_yaw,
            gripper_opening=1.0,
            approach_xy_distance=math.dist(ee_xy, target_xy),
            approach_xyz_distance=math.dist(
                (ee_xy[0], ee_xy[1], ee_z),
                (
                    target_xy[0],
                    target_xy[1],
                    objects[0].z + float(config.pick_grasp_height_offset),
                ),
            ),
            transport_xy_distance=math.dist(target_xy, receptacle_xy),
            destination_success_radius=radius,
            texture_id=int(rng.integers(7)),
            background_rgba=(
                float(0.65 + rng.random() * 0.30),
                float(0.65 + rng.random() * 0.30),
                float(0.65 + rng.random() * 0.30),
                1.0,
            ),
            shade=float(0.55 + rng.random() * 0.45),
        )
        try:
            validate_scene(scene, config)
        except SceneRejection as error:
            last_error = str(error)
            continue
        return scene
    raise SceneRejection(
        f"Could not place a {destination} scene for {target_catalog} in "
        f"{config.max_attempts_per_scene} attempts. Last rejection: "
        f"{last_error}"
    )


def assign_split(scene_uid: str, weights: Mapping[str, float], salt: str) -> str:
    """Deterministic split from the scene identity alone.

    Hash-based rather than index-based so a scene keeps its split when the
    manifest is regenerated with a different count, and so a retry or a second
    rollout seed of the same scene cannot land on the other side of the line.
    """

    total = sum(float(value) for value in weights.values())
    if total <= 0.0:
        raise ValueError("Split weights must sum to a positive number.")
    digest = hashlib.sha256(f"{salt}|{scene_uid}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)
    cumulative = 0.0
    for name in SPLIT_NAMES:
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        cumulative += weight / total
        if position < cumulative:
            return name
    # Floating point can leave the last bucket a hair short.
    for name in reversed(SPLIT_NAMES):
        if float(weights.get(name, 0.0)) > 0.0:
            return name
    raise ValueError("No split carries positive weight.")


DEFAULT_SPLIT_WEIGHTS: dict[str, float] = {
    "collection": 0.60,
    "teacher_selection": 0.15,
    "student_validation": 0.15,
    "final_test": 0.10,
}


def generate_scenes(
    *,
    count: int,
    seed: int,
    config: SceneGeometryConfig | None = None,
    split_weights: Mapping[str, float] | None = None,
    split_salt: str = SCENE_MANIFEST_VERSION,
) -> list[CompositionScene]:
    """``count`` accepted scenes, balanced over destination and target catalog.

    Balanced by CYCLING rather than by sampling, so a small manifest still has
    every stratum: a 64-scene screening set drawn at random would leave some
    (destination, object) cell with two members and the comparison would be
    reported per-cell anyway.
    """

    import numpy as np

    settings = config or SceneGeometryConfig()
    weights = dict(split_weights or DEFAULT_SPLIT_WEIGHTS)
    rng = np.random.default_rng(int(seed))
    scenes: list[CompositionScene] = []
    seen: set[str] = set()
    destinations = list(settings.destinations)
    catalogs = list(settings.target_catalogs)
    index = 0
    while len(scenes) < int(count):
        destination = destinations[index % len(destinations)]
        catalog = catalogs[(index // len(destinations)) % len(catalogs)]
        scene = _sample_scene(
            rng,
            settings,
            destination=destination,
            target_catalog=catalog,
            scene_index=len(scenes),
            seed=int(seed),
        )
        index += 1
        if scene.scene_uid in seen:
            # Two identical geometries would share a uid and therefore a split
            # and a parent key; keep drawing rather than merging them.
            continue
        seen.add(scene.scene_uid)
        scenes.append(
            replace(
                scene,
                split=assign_split(scene.scene_uid, weights, split_salt),
            )
        )
    return scenes


def manifest_payload(
    scenes: Sequence[CompositionScene],
    *,
    config: SceneGeometryConfig,
    seed: int,
    split_weights: Mapping[str, float],
    split_salt: str,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """The manifest, plus the hash that identifies this scene set."""

    body = {
        "manifest_version": SCENE_MANIFEST_VERSION,
        "seed": int(seed),
        "split_salt": str(split_salt),
        "split_weights": {
            key: float(value) for key, value in dict(split_weights).items()
        },
        "geometry": _config_payload(config),
        "scenes": [scene.to_json() for scene in scenes],
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    body["manifest_sha256"] = digest
    body["notes"] = list(notes)
    body["counts"] = scene_counts(scenes)
    return body


def _config_payload(config: SceneGeometryConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["destination_success_radius"] = {
        key: float(value)
        for key, value in dict(config.destination_success_radius).items()
    }
    return payload


def scene_counts(scenes: Sequence[CompositionScene]) -> dict[str, Any]:
    """Per-split, per-destination and per-object census of a scene set."""

    counts: dict[str, Any] = {"total": len(scenes)}
    for key, extract in (
        ("by_split", lambda scene: scene.split),
        ("by_destination", lambda scene: scene.destination),
        ("by_target", lambda scene: scene.target_catalog),
        (
            "by_split_destination",
            lambda scene: f"{scene.split}/{scene.destination}",
        ),
    ):
        table: dict[str, int] = {}
        for scene in scenes:
            name = str(extract(scene))
            table[name] = table.get(name, 0) + 1
        counts[key] = dict(sorted(table.items()))
    return counts


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def read_manifest(
    path: Path, *, validate: bool = True
) -> tuple[list[CompositionScene], dict[str, Any]]:
    """Load a manifest and, by default, re-check every scene against its own
    stored geometry configuration.

    The re-check is on by default because a manifest travels between hosts and
    between tools, and the failure it guards against -- a hand-edited bound, a
    scene written by an older generator -- is silent everywhere else.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("manifest_version")) != SCENE_MANIFEST_VERSION:
        raise ValueError(
            f"{path} is manifest version {payload.get('manifest_version')!r}, "
            f"not {SCENE_MANIFEST_VERSION!r}."
        )
    geometry = dict(payload["geometry"])
    geometry["target_catalogs"] = tuple(geometry["target_catalogs"])
    geometry["destinations"] = tuple(geometry["destinations"])
    geometry["distractor_catalogs"] = tuple(
        geometry.get("distractor_catalogs", ())
    )
    geometry["approach_xy_bounds"] = tuple(geometry["approach_xy_bounds"])
    geometry["ee_z_bounds"] = tuple(geometry["ee_z_bounds"])
    geometry["workspace_x_bounds"] = tuple(geometry["workspace_x_bounds"])
    geometry["workspace_y_bounds"] = tuple(geometry["workspace_y_bounds"])
    config = SceneGeometryConfig(**geometry)
    scenes = [
        CompositionScene.from_json(entry) for entry in payload["scenes"]
    ]
    if validate:
        for scene in scenes:
            validate_scene(scene, config)
    return scenes, payload


def select_split(
    scenes: Iterable[CompositionScene], split: str
) -> list[CompositionScene]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; known: {list(SPLIT_NAMES)}")
    return [scene for scene in scenes if scene.split == split]
