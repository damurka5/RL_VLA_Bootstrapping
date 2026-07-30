"""Where the CDPR gripper actually is, measured from the MJCF.

Every grasp task needs one number: how far below the tracked end-effector body
the finger pads sit. Getting it wrong does not fail loudly -- it produces a
reward whose optimum is a pose in which the gripper cannot touch anything, and
the policy then solves that reward accurately while never grasping.

That is not hypothetical. `pick_up` ran 10M steps with
``pick_grasp_height_offset: 0.08`` against a real offset of 0.0075 m, so the
reward's grasp point sat 7.25 cm above the object. The measured result: the
end-effector converged to within 1.2-1.5 cm of the reward's target while the
grasp rate PEAKED at 0.068 near 4.5M steps and then decayed to 0.056, because
converging onto the target removed the erratic excursions that had been
producing accidental contacts. Terminal successes went from 8/1024 to 0/1024.

So the offset is derived from the model here rather than written down in configs,
and `assert_grasp_offset_matches_model` lets a config be checked against it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CDPRGripperGeometry",
    "load_cdpr_gripper_geometry",
    "assert_grasp_offset_matches_model",
]

# The body chain from the tracked end-effector body down to the fingers, and the
# geoms that define the pad and the lowest finger surface. Named explicitly so a
# model change that renames or reparents any of them raises instead of silently
# returning a wrong offset.
_TRACKED_EE_BODY = "ee_base"
_PAD_GEOM = "left_finger_pad"
_TIP_GEOM = "finger_l_tip"


def _vec3(raw: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    if not raw:
        return default
    parts = tuple(float(value) for value in raw.split())
    if len(parts) != 3:
        raise ValueError(f"Expected three numbers, got {raw!r}.")
    return parts


@dataclass(frozen=True)
class CDPRGripperGeometry:
    """Gripper offsets relative to the tracked end-effector body, in metres.

    All offsets are signed z displacements from ``ee_base``: negative means
    below it. ``ee_position`` in every observation and reward is that body's
    world position, so these are exactly the corrections a task needs.
    """

    pad_center_offset: float
    pad_half_height: float
    finger_tip_offset: float

    @property
    def grasp_height_offset(self) -> float:
        """How far above an object's centre ``ee_base`` must sit to grasp it.

        This is the value a task's ``*_grasp_height_offset`` should carry: at
        this height the finger pads straddle the object's centre.
        """

        return -self.pad_center_offset

    @property
    def pad_span(self) -> tuple[float, float]:
        """(low, high) z extent of the pad, relative to ``ee_base``."""

        return (
            self.pad_center_offset - self.pad_half_height,
            self.pad_center_offset + self.pad_half_height,
        )

    def finger_tip_height(self, ee_height: float) -> float:
        """World z of the lowest finger surface for a given ``ee_base`` height."""

        return float(ee_height) + self.finger_tip_offset

    def minimum_ee_height(self, object_center_z: float) -> float:
        """Lowest ``ee_base`` height that still grasps an object at this centre.

        The pads span a range, so the grasp tolerates the end-effector sitting
        anywhere that keeps the object centre inside that span. This returns the
        bottom of that window.
        """

        return float(object_center_z) - self.pad_span[1]

    def maximum_ee_height(self, object_center_z: float) -> float:
        """Highest ``ee_base`` height that still grasps an object at this centre."""

        return float(object_center_z) - self.pad_span[0]

    def can_reach(self, object_center_z: float, *, controller_floor: float) -> bool:
        """Whether a controller clamped at ``controller_floor`` can grasp.

        Tests the LOOSEST grasp height -- the top of the pad span -- so this is
        false only when no part of the pad can reach the object at all.
        """

        return float(controller_floor) <= self.maximum_ee_height(object_center_z)


def load_cdpr_gripper_geometry(xml_path: Path | str) -> CDPRGripperGeometry:
    """Measure the gripper offsets by walking the MJCF body chain."""

    path = Path(xml_path).expanduser()

    # The training model is a thin MJWarp wrapper that <include>s the robot, so
    # resolving includes is required -- measuring cdpr.xml directly instead would
    # silently stop tracking whatever the run actually loads.
    def worldbodies(file_path: Path, seen: set[Path]) -> list:
        resolved = file_path.resolve()
        if resolved in seen:
            return []
        seen.add(resolved)
        root = ET.parse(resolved).getroot()
        found = list(root.findall("worldbody"))
        for include in root.findall("include"):
            target = include.get("file")
            if target:
                found.extend(
                    worldbodies(resolved.parent / target, seen)
                )
        return found

    bodies = worldbodies(path, set())
    if not bodies:
        raise ValueError(f"{path} has no <worldbody>, directly or via <include>.")

    def find_tracked(node):
        for body in node.findall("body"):
            if body.get("name") == _TRACKED_EE_BODY:
                return body
            found = find_tracked(body)
            if found is not None:
                return found
        return None

    tracked = None
    for worldbody in bodies:
        tracked = find_tracked(worldbody)
        if tracked is not None:
            break
    if tracked is None:
        raise ValueError(
            f"{path} has no body named {_TRACKED_EE_BODY!r}; the gripper offset "
            "cannot be measured."
        )

    offsets: dict[str, float] = {}

    def walk(body, z_offset: float) -> None:
        # A rotated body in the chain would make a scalar z offset meaningless.
        for attr in ("euler", "quat", "axisangle", "xyaxes", "zaxis"):
            if body.get(attr) and body is not tracked:
                raise ValueError(
                    f"Body {body.get('name')!r} in {path} carries {attr!r}; the "
                    "gripper offset is only a scalar z displacement while the "
                    "chain is unrotated. Measure it explicitly instead."
                )
        for geom in body.findall("geom"):
            name = geom.get("name")
            if name in (_PAD_GEOM, _TIP_GEOM):
                geom_z = z_offset + _vec3(geom.get("pos"))[2]
                size = geom.get("size")
                half = 0.0
                if size:
                    parts = tuple(float(v) for v in size.split())
                    # Box half-extents are (x, y, z); the z half-extent is what
                    # sets how much of the object the pad can straddle.
                    half = parts[2] if len(parts) >= 3 else 0.0
                offsets[name] = geom_z
                offsets[f"{name}__half"] = half
        for child in body.findall("body"):
            walk(child, z_offset + _vec3(child.get("pos"))[2])

    walk(tracked, 0.0)

    missing = [
        name for name in (_PAD_GEOM, _TIP_GEOM) if name not in offsets
    ]
    if missing:
        raise ValueError(
            f"{path} is missing gripper geom(s) {missing} under "
            f"{_TRACKED_EE_BODY!r}."
        )

    return CDPRGripperGeometry(
        pad_center_offset=offsets[_PAD_GEOM],
        pad_half_height=offsets[f"{_PAD_GEOM}__half"],
        finger_tip_offset=(
            offsets[_TIP_GEOM] - offsets[f"{_TIP_GEOM}__half"]
        ),
    )


def assert_grasp_offset_matches_model(
    configured_offset: float,
    *,
    xml_path: Path | str,
    tolerance: float | None = None,
    label: str = "grasp_height_offset",
) -> CDPRGripperGeometry:
    """Raise unless ``configured_offset`` puts the pads on the object.

    ``tolerance`` defaults to the pad's half-height, because anywhere inside the
    pad span is a legitimate grasp height -- it is only outside that span that
    the reward starts optimizing toward a pose the gripper cannot grasp from.
    """

    geometry = load_cdpr_gripper_geometry(xml_path)
    limit = geometry.pad_half_height if tolerance is None else float(tolerance)
    ideal = geometry.grasp_height_offset
    if abs(float(configured_offset) - ideal) > limit:
        raise ValueError(
            f"{label}={float(configured_offset):.4f} m places the finger pads "
            f"{float(configured_offset) - ideal:+.4f} m from the object centre, "
            f"outside the pad's {limit:.4f} m half-height. The model puts the "
            f"pads {ideal:.4f} m below the tracked end-effector body, so the "
            f"reward optimum would be a pose that cannot touch the object."
        )
    return geometry
