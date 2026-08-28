"""Where the WidowX-200 may work, derived rather than assumed.

The CDPR hangs from four cables above the desk, so its reachable set is a box
and the scene put objects in a symmetric +/-0.205 m square around the world
origin. A serial arm's reachable set is an annular sector around its own base,
and the CDPR's square does not fit inside it -- the far corners are 0.49 m from
any sensible mount, against a top-down reach of about 0.35 m. Reusing the old
envelope would spawn targets the arm provably cannot touch and charge the
policy for failing to reach them.

Everything here is computed from `kinematics`, checked by
`tests/test_widowx200_workspace.py`, and consumed by the scene wrapper and the
configs. The datasheet's "550 mm reach" is not used anywhere: that is the
fully-extended figure with the wrist straight, and a top-down grasp folds the
wrist to vertical, which costs roughly 15 cm of it.

Layout choices that are judgement, not arithmetic
-------------------------------------------------

* **The arm mounts at the back of the desk, facing the overview camera.** The
  camera sits at y = -0.54 looking toward +y. With the arm in front of the
  objects it occludes most of them; behind them it is visible as context and
  the objects are not. This also matches how a WidowX is filmed in the
  BridgeData setups.

* **Objects live in an annular sector, not a box.** A box inscribed in the
  reachable annulus wastes about 40% of the usable area, and this arm has
  little to spare.

* **A 0.15 m inner keep-out.** Not a reach limit -- the arm can fold closer --
  but the waist is singular on its own axis: near it, a centimetre of commanded
  XY motion needs a large, fast waist swing. The policy commands XY deltas, so
  that region turns small actions into violent ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kinematics import WX200, WidowX200Geometry, TOP_DOWN_PITCH, reach_envelope

__all__ = [
    "DESK_SURFACE_Z",
    "WidowX200SceneLayout",
    "DEFAULT_LAYOUT",
    "usable_radius",
    "sector_lattice",
    "mount_widowx200",
]

# Unchanged from the CDPR scene, deliberately. Object rest heights, the support
# clearance in the push predicates, the container-z threshold, and every
# `*_grasp_height_offset` are written against this surface; moving it would
# invalidate all of them for no gain.
DESK_SURFACE_Z: float = 0.1502


def usable_radius(
    world_z: float,
    *,
    base_z: float = DESK_SURFACE_Z,
    geometry: WidowX200Geometry = WX200,
    margin: float = 0.02,
) -> float:
    """Largest horizontal radius the tool can hold at ``world_z``, top-down.

    ``margin`` is held back from the kinematic limit. At the limit the arm is
    radially singular: the position servo lags the target, and a reward written
    against the commanded point starts scoring a pose the arm never reached.
    """

    _, high = reach_envelope(world_z - base_z, geometry=geometry, gamma=TOP_DOWN_PITCH)
    return max(0.0, high - float(margin))


@dataclass(frozen=True)
class WidowX200SceneLayout:
    """Mount pose, object-spawn sector, and controller workspace, in world m."""

    base_position: tuple[float, float, float]
    base_yaw: float
    spawn_radius: tuple[float, float]
    spawn_half_angle: float
    workspace_x: tuple[float, float]
    workspace_y: tuple[float, float]
    workspace_z: tuple[float, float]
    min_reach_radius: float

    @property
    def spawn_area(self) -> float:
        lo, hi = self.spawn_radius
        return (hi * hi - lo * lo) * self.spawn_half_angle

    def sample_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Axis-aligned world box enclosing the spawn sector.

        Rejection sampling inside this box against ``contains`` is the intended
        use: the sector is not a product of intervals, so a box alone would put
        objects out of reach.
        """

        lo, hi = self.spawn_radius
        half = self.spawn_half_angle
        cx, cy = self.base_position[0], self.base_position[1]
        angles = [self.base_yaw - half, self.base_yaw + half]
        xs, ys = [], []
        for radius in (lo, hi):
            for angle in angles:
                xs.append(cx + radius * math.cos(angle))
                ys.append(cy + radius * math.sin(angle))
        # The sector's outer arc bulges past its corner chords along base_yaw.
        xs.append(cx + hi * math.cos(self.base_yaw))
        ys.append(cy + hi * math.sin(self.base_yaw))
        return ((min(xs), max(xs)), (min(ys), max(ys)))

    def contains(self, x: float, y: float) -> bool:
        dx = x - self.base_position[0]
        dy = y - self.base_position[1]
        radius = math.hypot(dx, dy)
        lo, hi = self.spawn_radius
        if not (lo <= radius <= hi):
            return False
        offset = math.atan2(dy, dx) - self.base_yaw
        offset = (offset + math.pi) % (2.0 * math.pi) - math.pi
        return abs(offset) <= self.spawn_half_angle


# Derived from `usable_radius`, checked in tests, and sized by a constraint
# that is easy to miss: the scene must hold FOUR objects at least 0.16 m apart
# (the CDPR collector's separation, set by the widest realistic pair). A sector
# only 0.14 m deep fits one row of them, so the outer radius is set from the
# reach at GRASP height (0.38 m at the desk) rather than at the ceiling, and the
# ceiling is lowered to 0.27 m instead -- at which height 0.350 m is still
# reachable, so every lattice cell stays reachable through a lift and a carry.
# `sector_lattice` below achieves 0.180 m of separation on this sector, above
# the 0.16 m the collector needs.
#
# The base sits 0.24 m behind the sector centroid, which keeps the objects
# straddling the world origin and the desk unchanged.
DEFAULT_LAYOUT = WidowX200SceneLayout(
    base_position=(0.0, 0.24, DESK_SURFACE_Z),
    base_yaw=-0.5 * math.pi,
    spawn_radius=(0.16, 0.34),
    spawn_half_angle=math.radians(50.0),
    workspace_x=(-0.30, 0.30),
    workspace_y=(-0.12, 0.22),
    # 0.168 is the floor at which the finger pads (17 mm half-length along the
    # approach axis, measured from the compiled model) just clear the desk at
    # 0.1502. Grasp phases run right at it; hover phases should pass something
    # higher, exactly as the CDPR configs do. The 0.30 ceiling is not a
    # kinematic limit but the height above which the outer sector stops being
    # reachable -- see above.
    workspace_z=(0.168, 0.27),
    min_reach_radius=0.15,
)


def sector_lattice(
    layout: "WidowX200SceneLayout" = DEFAULT_LAYOUT,
    *,
    inner_count: int = 2,
    outer_count: int = 4,
) -> tuple[tuple[float, float], ...]:
    """Candidate object cells in world XY, well separated and all reachable.

    The CDPR collector draws four objects from a 3x3 Cartesian grid at 0.18 m
    spacing (`mjwarp_rank_local_collector.py`, "3x3 candidate grid"). That grid
    does not fit here: its corners fall outside the arm's annulus, and a
    rectangular lattice inscribed in a sector wastes most of it.

    A two-ring polar lattice fits the shape instead. With the shipped layout it
    yields six cells with a minimum pairwise separation of 0.180 m -- above the
    0.16 m the collector needs (set by the widest realistic object pair) -- so
    a group can still take a random four-cell subset and never place two
    objects in contact.

    Returned in world coordinates, already rotated and translated onto the
    arm's mount, so a caller never has to know the mount pose.
    """

    inner, outer = layout.spawn_radius
    half = layout.spawn_half_angle
    cells: list[tuple[float, float]] = []
    for radius, count in ((inner, inner_count), (outer, outer_count)):
        if count <= 0:
            continue
        angles = (
            [0.0]
            if count == 1
            else [-half + 2.0 * half * index / (count - 1) for index in range(count)]
        )
        for angle in angles:
            bearing = layout.base_yaw + angle
            cells.append(
                (
                    layout.base_position[0] + radius * math.cos(bearing),
                    layout.base_position[1] + radius * math.sin(bearing),
                )
            )
    return tuple(cells)


def mount_widowx200(
    model: object,
    layout: "WidowX200SceneLayout" = DEFAULT_LAYOUT,
    *,
    mujoco: object = None,
) -> None:
    """Stand the arm at ``layout``'s mount pose on an already-compiled model.

    ``wx200_mount`` is a static body, so writing ``body_pos``/``body_quat`` and
    running ``mj_forward`` places it -- no XML edit, no recompile, and the same
    per-world field MJWarp would mutate if the mount is ever randomized.

    Doing it here rather than in the scene XML keeps ONE mount pose in the
    project. The controller needs the pose to convert world targets into the
    base frame; if the XML held a second copy they could drift apart, and the
    result would be an IK solution computed in the wrong frame, which produces
    no error and a policy that never reaches anything.
    """

    if mujoco is None:
        import mujoco as mujoco_module

        mujoco = mujoco_module
    import numpy as np

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wx200_mount")
    if body == -1:
        raise KeyError("Scene has no `wx200_mount` body to place the arm on.")
    model.body_pos[body] = np.asarray(layout.base_position, dtype=np.float64)
    half = 0.5 * float(layout.base_yaw)
    model.body_quat[body] = np.asarray(
        [math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64
    )
