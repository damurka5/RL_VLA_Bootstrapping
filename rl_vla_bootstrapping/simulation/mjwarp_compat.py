from __future__ import annotations

import importlib
import importlib.metadata
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PINNED_MJLAB_VERSION = "1.5.0"
PINNED_MUJOCO_VERSION = "3.10.0"
PINNED_MJWARP_VERSION = "3.10.0.1"
PINNED_WARP_VERSION = "1.14.0"
PINNED_TORCH_VERSION = "2.7.1"
PINNED_CUDA_RUNTIME = "12.8"


@dataclass
class MJCFCompatibilityReport:
    xml_path: str
    parse_ok: bool = False
    host_compile_ok: bool = False
    put_model_ok: bool = False
    world_allocation_ok: bool = False
    render_context_ok: bool = False
    required_features: dict[str, bool] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def compatible(self) -> bool:
        return bool(
            self.parse_ok
            and self.host_compile_ok
            and self.put_model_ok
            and self.world_allocation_ok
            and self.render_context_ok
            and not self.unsupported
            and all(self.required_features.values())
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compatible"] = self.compatible
        return payload


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for dist_name, key in (
        ("mjlab", "mjlab"),
        ("mujoco", "mujoco"),
        ("mujoco-warp", "mujoco_warp"),
        ("warp-lang", "warp"),
        ("torch", "torch"),
    ):
        try:
            out[key] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            out[key] = "unavailable"
    return out


def pinned_version_mismatches(
    versions: dict[str, str] | None = None,
) -> dict[str, tuple[str, str]]:
    actual = versions or package_versions()
    expected = {
        "mjlab": PINNED_MJLAB_VERSION,
        "mujoco": PINNED_MUJOCO_VERSION,
        "mujoco_warp": PINNED_MJWARP_VERSION,
        "warp": PINNED_WARP_VERSION,
        "torch": PINNED_TORCH_VERSION,
    }
    mismatches: dict[str, tuple[str, str]] = {}
    for name, wanted in expected.items():
        found = str(actual.get(name, "unavailable"))
        # PyTorch distribution versions include their CUDA local tag.
        match = found == wanted or (name == "torch" and found.startswith(wanted + "+"))
        if not match:
            mismatches[name] = (wanted, found)
    return mismatches


def require_pinned_versions() -> dict[str, str]:
    versions = package_versions()
    mismatches = pinned_version_mismatches(versions)
    if mismatches:
        details = ", ".join(
            f"{name}: expected {wanted}, found {found}"
            for name, (wanted, found) in mismatches.items()
        )
        raise RuntimeError(
            "MJLab/MJWarp dependency versions do not match the tested lock: "
            + details
        )
    return versions


def _parse_mjcf_include_tree(path: Path) -> list[ET.Element]:
    roots: list[ET.Element] = []
    visited: set[Path] = set()

    def visit(current: Path) -> None:
        resolved = current.expanduser().resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        root = ET.parse(resolved).getroot()
        roots.append(root)
        for include in root.findall(".//include"):
            filename = str(include.get("file") or "").strip()
            if filename:
                visit(resolved.parent / filename)

    visit(path)
    return roots


def _findall(roots: list[ET.Element], pattern: str) -> list[ET.Element]:
    return [item for root in roots for item in root.findall(pattern)]


def _local_name_counts(roots: list[ET.Element]) -> dict[str, int]:
    return {
        "spatial_tendons": len(_findall(roots, ".//tendon/spatial")),
        "tendon_wrap_geoms": len(_findall(roots, ".//tendon/spatial/geom")),
        "joint_equalities": len(_findall(roots, ".//equality/joint")),
        "cameras": len(_findall(roots, ".//camera")),
        "frame_sensors": len(_findall(roots, ".//sensor/framepos"))
        + len(_findall(roots, ".//sensor/framequat")),
        "free_joints": len(_findall(roots, ".//freejoint"))
        + len(_findall(roots, ".//joint[@type='free']")),
        "ball_joints": len(_findall(roots, ".//joint[@type='ball']")),
        "position_actuators": len(_findall(roots, ".//actuator/position")),
    }


def inspect_cdpr_mjcf(xml_path: str | Path) -> MJCFCompatibilityReport:
    """Static, dependency-free compatibility scan.

    This scan is intentionally not equivalent to remote compatibility.  A
    report is only compatible after `put_model` and world allocation execute.
    """

    path = Path(xml_path).expanduser().resolve()
    report = MJCFCompatibilityReport(xml_path=path.as_posix(), versions=package_versions())
    try:
        roots = _parse_mjcf_include_tree(path)
        report.parse_ok = True
    except Exception as exc:
        report.error = f"MJCF parse failed: {exc}"
        return report

    counts = _local_name_counts(roots)
    report.counts.update(counts)
    cameras = _findall(roots, ".//camera")
    tendon_wrap_geoms = _findall(roots, ".//tendon/spatial/geom")
    report.required_features = {
        "four_spatial_tendons": counts["spatial_tendons"] == 4,
        "pulley_routing_sidesites": counts["tendon_wrap_geoms"] >= 8
        and all(
            bool(str(item.get("sidesite") or "").strip())
            for item in tendon_wrap_geoms
        ),
        "finger_joint_equality": counts["joint_equalities"] >= 1,
        "free_end_effector": counts["free_joints"] >= 1,
        "ball_stabilizer": counts["ball_joints"] >= 1,
        "slider_yaw_gripper_actuators": counts["position_actuators"] >= 6,
        "overview_camera": any(
            item.get("name") == "overview" for item in cameras
        ),
        "wrist_camera": any(
            item.get("name") == "ee_camera" for item in cameras
        ),
        "camera_frame_sensors": counts["frame_sensors"] >= 4,
    }

    option_nodes = _findall(roots, ".//option")
    for option in option_nodes:
        integrator = str(option.get("integrator") or "").upper()
        solver = str(option.get("solver") or "").upper()
        if integrator == "IMPLICITFAST":
            report.warnings.append(
                "IMPLICITFAST is supported, but MuJoCo Warp documents differences "
                "for midpoint-feature and fluid-force paths; validate this model "
                "numerically before enabling those features."
            )
        if solver == "PGS":
            report.unsupported.append("PGS solver is unsupported by MuJoCo Warp.")
        try:
            noslip_iterations = int(option.get("noslip_iterations", "0"))
        except ValueError:
            noslip_iterations = 0
        if noslip_iterations > 0:
            report.unsupported.append(
                "noslip_iterations > 0 is unsupported by MuJoCo Warp; use Newton "
                "contacts without the noslip post-solver."
            )
    if _findall(roots, ".//extension/plugin") or _findall(
        roots, ".//actuator/plugin"
    ):
        report.unsupported.append("Plugin actuators/sensors are unsupported by MuJoCo Warp.")
    missing = [name for name, present in report.required_features.items() if not present]
    if missing:
        report.warnings.append("Missing required CDPR features: " + ", ".join(missing))
    return report


def execute_mjwarp_compatibility_spike(
    xml_path: str | Path,
    *,
    nworld: int,
    nconmax: int,
    njmax: int,
    nccdmax: int | None = None,
    device: str | None = None,
    create_renderer: bool = True,
    render_width: int = 320,
    render_height: int = 240,
) -> MJCFCompatibilityReport:
    """Compile, copy, allocate, and optionally create the batch renderer."""

    report = inspect_cdpr_mjcf(xml_path)
    if not report.parse_ok or report.unsupported:
        return report
    try:
        mujoco = importlib.import_module("mujoco")
        mjw = importlib.import_module("mujoco_warp")
        wp = importlib.import_module("warp")
    except Exception as exc:
        report.error = (
            "MJLab/MuJoCo Warp dependencies are unavailable. Install the pinned "
            f"environment first. Import error: {exc}"
        )
        return report

    try:
        if device:
            wp.set_device(str(device))
        mjm = mujoco.MjModel.from_xml_path(report.xml_path)
        report.host_compile_ok = True
        report.counts.update(
            {
                "nq": int(mjm.nq),
                "nv": int(mjm.nv),
                "nu": int(mjm.nu),
                "nbody": int(mjm.nbody),
                "ngeom": int(mjm.ngeom),
                "ntendon": int(mjm.ntendon),
                "neq": int(mjm.neq),
                "ncam": int(mjm.ncam),
            }
        )
        model = mjw.put_model(mjm)
        report.put_model_ok = True
        make_kwargs: dict[str, Any] = {
            "nworld": int(nworld),
            "nconmax": int(nconmax),
            "njmax": int(njmax),
        }
        if nccdmax is not None:
            make_kwargs["nccdmax"] = int(nccdmax)
        data = mjw.make_data(mjm, **make_kwargs)
        report.world_allocation_ok = bool(int(data.nworld) == int(nworld))
        mjw.forward(model, data)
        if create_renderer:
            if int(mjm.ncam) < 2:
                raise RuntimeError(
                    "Batch camera creation requires named overview and ee_camera cameras."
                )
            render_context = mjw.create_render_context(
                mjm,
                nworld=int(nworld),
                cam_res=(int(render_width), int(render_height)),
                render_rgb=True,
                render_depth=False,
                use_textures=True,
                use_shadows=False,
            )
            overview_id = int(
                mujoco.mj_name2id(
                    mjm, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
                )
            )
            wrist_id = int(
                mujoco.mj_name2id(
                    mjm, mujoco.mjtObj.mjOBJ_CAMERA, "ee_camera"
                )
            )
            if overview_id < 0 or wrist_id < 0:
                raise RuntimeError("Named policy cameras are missing after MJCF compile.")
            overview = wp.zeros(
                (int(nworld), int(render_height), int(render_width)),
                dtype=wp.vec3,
            )
            wrist = wp.zeros(
                (int(nworld), int(render_height), int(render_width)),
                dtype=wp.vec3,
            )
            mjw.refit_bvh(model, data, render_context)
            mjw.render(model, data, render_context)
            try:
                mjw.get_rgb(render_context, overview_id, overview)
                mjw.get_rgb(render_context, wrist_id, wrist)
            except TypeError:
                mjw.get_rgb(render_context, rgb_data=overview, cam_id=overview_id)
                mjw.get_rgb(render_context, rgb_data=wrist, cam_id=wrist_id)
            overview_np = overview.numpy()
            wrist_np = wrist.numpy()
            expected = (
                int(nworld),
                int(render_height),
                int(render_width),
                3,
            )
            if tuple(overview_np.shape) != expected or tuple(wrist_np.shape) != expected:
                raise RuntimeError(
                    "Batched camera shape mismatch: "
                    f"overview={overview_np.shape}, wrist={wrist_np.shape}, "
                    f"expected={expected}."
                )
            report.counts["rendered_camera_batches"] = 2
            report.render_context_ok = True
        return report
    except Exception as exc:
        report.error = f"MJWarp compatibility spike failed: {type(exc).__name__}: {exc}"
        return report


def write_compatibility_report(
    report: MJCFCompatibilityReport,
    path: str | Path,
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")
    return output
