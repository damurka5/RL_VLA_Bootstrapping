#!/usr/bin/env python3
"""Static audit helpers for the simulator migration decision report.

The script intentionally avoids running training or changing project code. It
collects file/asset evidence from MJCF wrappers, known configs, and existing
contact-test logs into small CSV/JSON artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tools" / "audit" / "out"

XML_SEARCH_ROOTS = [
    REPO_ROOT / "robots" / "cdpr" / "cdpr_dataset" / "wrappers",
    REPO_ROOT / "robots" / "cdpr" / "cdpr_mujoco",
]
CONFIG_SEARCH_ROOT = REPO_ROOT / "configs"
CONTACT_RUN_ROOT = (
    REPO_ROOT / "runs" / "ycb_gripper_physical_pick_release_videos"
)

FILE_ATTRS = {"file"}
PATH_RE = re.compile(
    r"(?P<path>(?:(?:\.\./)+|\.\/|/)?[\w./~+\- ]+\.(?:xml|obj|stl|dae|png|jpg|jpeg|bmp|mtl|urdf))"
)


def classify_asset(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp"}:
        return "texture_or_image"
    if suffix in {".obj", ".stl", ".dae", ".ply"}:
        return "mesh"
    if suffix in {".xml", ".mjcf"}:
        return "xml"
    if suffix == ".mtl":
        return "material"
    if suffix == ".urdf":
        return "urdf"
    return "other"


def resolve_path(raw: str, base: Path) -> Path:
    raw = os.path.expanduser(raw)
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def iter_xml_refs(xml_path: Path) -> Iterable[tuple[Path, str]]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return
    for elem in root.iter():
        for attr, raw in elem.attrib.items():
            if attr in FILE_ATTRS:
                yield resolve_path(raw, xml_path.parent), f"{xml_path}:{elem.tag}@{attr}"


def collect_config_refs() -> dict[Path, list[str]]:
    refs: dict[Path, list[str]] = defaultdict(list)
    if not CONFIG_SEARCH_ROOT.exists():
        return refs
    for config in CONFIG_SEARCH_ROOT.rglob("*.yaml"):
        text = config.read_text(errors="ignore")
        for match in PATH_RE.finditer(text):
            path = resolve_path(match.group("path").strip("'\""), config.parent)
            if not path.exists() and not Path(match.group("path")).is_absolute():
                repo_relative = (REPO_ROOT / match.group("path").strip("'\"")).resolve()
                if repo_relative.exists():
                    path = repo_relative
            refs[path].append(str(config))
    return refs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_dimensions(path: Path) -> str:
    if classify_asset(path) != "texture_or_image" or not path.exists():
        return ""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return f"{im.width}x{im.height}"
    except Exception:
        return ""


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def collect_asset_inventory() -> tuple[list[dict], dict[str, list[Path]]]:
    refs: dict[Path, list[str]] = defaultdict(list)
    for root in XML_SEARCH_ROOTS:
        if not root.exists():
            continue
        for xml_path in root.rglob("*.xml"):
            for path, source in iter_xml_refs(xml_path):
                refs[path].append(source)
    for path, sources in collect_config_refs().items():
        refs[path].extend(sources)

    rows: list[dict] = []
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(refs):
        exists = path.exists()
        size = path.stat().st_size if exists and path.is_file() else 0
        digest = sha256(path) if exists and path.is_file() else ""
        if digest:
            hash_to_paths[digest].append(path)
        rows.append(
            {
                "file_path": str(path),
                "asset_type": classify_asset(path),
                "exists": exists,
                "size_mb": f"{size / (1024 * 1024):.6f}" if size else "0.000000",
                "dimensions": image_dimensions(path),
                "sha256": digest,
                "referenced_by": "; ".join(sorted(set(refs[path]))[:40]),
                "reference_count": len(refs[path]),
            }
        )
    return rows, hash_to_paths


def collect_duplicate_textures(hash_to_paths: dict[str, list[Path]]) -> list[dict]:
    rows: list[dict] = []
    for digest, paths in sorted(hash_to_paths.items()):
        texture_paths = [p for p in paths if classify_asset(p) == "texture_or_image"]
        if len(texture_paths) < 2:
            continue
        size = texture_paths[0].stat().st_size if texture_paths[0].exists() else 0
        wasted = size * (len(texture_paths) - 1)
        rows.append(
            {
                "sha256": digest,
                "duplicate_count": len(texture_paths),
                "duplicate_file_paths": "; ".join(str(p) for p in texture_paths),
                "total_wasted_mb": f"{wasted / (1024 * 1024):.6f}",
            }
        )
    return rows


def wrapper_xmls() -> list[Path]:
    root = REPO_ROOT / "robots" / "cdpr" / "cdpr_dataset" / "wrappers"
    if not root.exists():
        return []
    return sorted(root.rglob("*_wrapper.xml"))


def include_graph_assets(xml_path: Path, seen: set[Path] | None = None) -> set[Path]:
    if seen is None:
        seen = set()
    if xml_path in seen or not xml_path.exists():
        return set()
    seen.add(xml_path)
    assets = {xml_path}
    for ref, _source in iter_xml_refs(xml_path):
        assets.add(ref)
        if ref.suffix.lower() == ".xml":
            assets.update(include_graph_assets(ref, seen))
    return assets


def collect_mujoco_model_cache_report() -> list[dict]:
    rows: list[dict] = []
    for xml_path in wrapper_xmls():
        assets = include_graph_assets(xml_path)
        existing_files = [p for p in assets if p.exists() and p.is_file()]
        lower_bound_bytes = sum(p.stat().st_size for p in existing_files)
        include_count = sum(1 for p in assets if p.suffix.lower() == ".xml") - 1
        texture_count = sum(1 for p in assets if classify_asset(p) == "texture_or_image")
        mesh_count = sum(1 for p in assets if classify_asset(p) == "mesh")
        rows.append(
            {
                "environment_scene_id": xml_path.parent.name,
                "xml_path": str(xml_path),
                "compiled_model_count": 1,
                "estimated_memory_mb": f"{lower_bound_bytes / (1024 * 1024):.3f}",
                "memory_estimate_method": "lower_bound_sum_of_existing_referenced_files_not_MjModel_heap",
                "renderer_count": 1,
                "camera_count": 2,
                "included_xml_count": max(include_count, 0),
                "referenced_mesh_count": mesh_count,
                "referenced_texture_count": texture_count,
                "missing_referenced_file_count": sum(1 for p in assets if not p.exists()),
            }
        )
    return rows


def object_name_from_path(path: Path) -> str:
    name = path.stem
    name = re.sub(r"^placed_\d+_", "", name)
    return name


def parse_object_xml(xml_path: Path) -> dict:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return {}
    geoms = list(root.iter("geom"))
    collision_geoms = [
        g
        for g in geoms
        if g.attrib.get("contype", "1") != "0" and g.attrib.get("conaffinity", "1") != "0"
    ]
    visual_geoms = [
        g
        for g in geoms
        if g.attrib.get("contype") == "0" or g.attrib.get("group") == "1"
    ]
    mass = ""
    inertia = ""
    for inertial in root.iter("inertial"):
        mass = inertial.attrib.get("mass", "")
        inertia = inertial.attrib.get("diaginertia", "")
        break
    frictions = sorted({g.attrib.get("friction", "") for g in collision_geoms if g.attrib.get("friction")})
    collision_types = sorted(
        {
            "mesh" if g.attrib.get("mesh") else g.attrib.get("type", "implicit_geom")
            for g in collision_geoms
        }
    )
    return {
        "object": object_name_from_path(xml_path),
        "xml_path": str(xml_path),
        "collision_type": "+".join(collision_types) if collision_types else "unknown",
        "collision_geom_count": len(collision_geoms),
        "visual_geom_count": len(visual_geoms),
        "mass": mass,
        "diaginertia": inertia,
        "friction": "|".join(frictions),
    }


def load_latest_contact_manifest() -> tuple[Path | None, dict]:
    if not CONTACT_RUN_ROOT.exists():
        return None, {}
    manifests = sorted(CONTACT_RUN_ROOT.rglob("manifest.json"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        return None, {}
    manifest = manifests[-1]
    try:
        return manifest, json.loads(manifest.read_text())
    except Exception:
        return manifest, {}


def collect_contact_stability_report() -> list[dict]:
    rows_by_object: dict[str, dict] = {}
    wrappers = REPO_ROOT / "robots" / "cdpr" / "cdpr_dataset" / "wrappers"
    if wrappers.exists():
        for xml_path in wrappers.rglob("placed_*.xml"):
            parsed = parse_object_xml(xml_path)
            if not parsed:
                continue
            obj = parsed["object"]
            rows_by_object.setdefault(
                obj,
                {
                    "object": obj,
                    "asset_source": "YCB/LIBERO wrapper XML inferred",
                    "collision_type": parsed["collision_type"],
                    "mass": parsed["mass"],
                    "friction": parsed["friction"],
                    "test_result": "static_xml_only",
                    "failure_mode": "",
                    "recommended_fix": "run drop/rest/push/grasp/lift contact tests before training",
                    "evidence": parsed["xml_path"],
                },
            )

    manifest_path, manifest = load_latest_contact_manifest()
    manifest_entries = manifest.get("results") or manifest.get("videos") or []
    for entry in manifest_entries:
        obj = entry.get("object", "")
        phase = entry.get("phase", "") or entry.get("kind", "")
        if not obj:
            continue
        forces = [
            float(entry.get("max_left_force", 0.0) or 0.0),
            float(entry.get("max_right_force", 0.0) or 0.0),
            float(entry.get("max_actuator_force", 0.0) or 0.0),
            float(entry.get("max_left_normal_force", 0.0) or 0.0),
            float(entry.get("max_right_normal_force", 0.0) or 0.0),
        ]
        max_force = max(forces)
        row = rows_by_object.setdefault(
            obj,
            {
                "object": obj,
                "asset_source": "YCB physical-gripper test",
                "collision_type": "",
                "mass": "",
                "friction": "",
                "test_result": "",
                "failure_mode": "",
                "recommended_fix": "",
                "evidence": "",
            },
        )
        previous_result = row.get("test_result", "")
        status = "force_spike" if max_force > 1000 else "no_large_force_spike_in_manifest"
        row["test_result"] = "; ".join(filter(None, [previous_result, f"{phase}:{status}"]))
        if max_force > 1000:
            row["failure_mode"] = "large_contact_force_spikes_in_existing_grasp_or_lift_test"
            row["recommended_fix"] = (
                "replace with primitive/convex collision proxy, verify scale/mass/inertia, "
                "then retune gripper friction/contact solver"
            )
        elif not row.get("recommended_fix"):
            row["recommended_fix"] = "still run full drop/rest/push/grasp/lift test battery"
        if manifest_path:
            row["evidence"] = "; ".join(filter(None, [row.get("evidence", ""), str(manifest_path)]))

    return sorted(rows_by_object.values(), key=lambda r: r["object"])


def write_rollout_profile_placeholder() -> list[dict]:
    rows = [
        {
            "metric": "env_step_time_s",
            "value": "",
            "status": "not_measured",
            "evidence": "No runtime profiler was executed by this static audit.",
        },
        {
            "metric": "render_time_s",
            "value": "",
            "status": "not_measured",
            "evidence": "HeadlessCDPRSimulation.capture_frame uses mjr_readPixels CPU readback.",
        },
        {
            "metric": "reward_time_s",
            "value": "",
            "status": "not_measured",
            "evidence": "Reward code is pure Python predicate logic in rl_instruction_tasks.py.",
        },
        {
            "metric": "image_preprocessing_time_s",
            "value": "",
            "status": "not_measured",
            "evidence": "OpenVLA wrapper batches PIL/processor inputs after local patching.",
        },
        {
            "metric": "policy_forward_time_s",
            "value": "",
            "status": "not_measured",
            "evidence": "OpenVLA forward pass is external to this repo under openvla-oft.",
        },
        {
            "metric": "optimizer_time_s",
            "value": "",
            "status": "not_measured",
            "evidence": "External GRPO/PPO trainer script is generated/patched, not stored here.",
        },
        {
            "metric": "cpu_ram_mb",
            "value": "",
            "status": "not_measured",
            "evidence": "Run-time RSS requires psutil/profiler around rollout workers.",
        },
        {
            "metric": "gpu_vram_mb",
            "value": "",
            "status": "not_measured",
            "evidence": "Requires nvidia-smi sampling during actual rollout/training.",
        },
        {
            "metric": "gpu_utilization_pct",
            "value": "",
            "status": "not_measured",
            "evidence": "Requires nvidia-smi dmon or pynvml during actual rollout/training.",
        },
    ]
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory, hash_to_paths = collect_asset_inventory()
    duplicate_textures = collect_duplicate_textures(hash_to_paths)
    cache_report = collect_mujoco_model_cache_report()
    contact_report = collect_contact_stability_report()
    rollout_profile = write_rollout_profile_placeholder()

    write_csv(
        OUT_DIR / "asset_inventory.csv",
        [
            "file_path",
            "asset_type",
            "exists",
            "size_mb",
            "dimensions",
            "sha256",
            "referenced_by",
            "reference_count",
        ],
        inventory,
    )
    write_csv(
        OUT_DIR / "duplicate_textures.csv",
        ["sha256", "duplicate_count", "duplicate_file_paths", "total_wasted_mb"],
        duplicate_textures,
    )
    write_csv(
        OUT_DIR / "mujoco_model_cache_report.csv",
        [
            "environment_scene_id",
            "xml_path",
            "compiled_model_count",
            "estimated_memory_mb",
            "memory_estimate_method",
            "renderer_count",
            "camera_count",
            "included_xml_count",
            "referenced_mesh_count",
            "referenced_texture_count",
            "missing_referenced_file_count",
        ],
        cache_report,
    )
    write_csv(
        OUT_DIR / "contact_stability_report.csv",
        [
            "object",
            "asset_source",
            "collision_type",
            "mass",
            "friction",
            "test_result",
            "failure_mode",
            "recommended_fix",
            "evidence",
        ],
        contact_report,
    )
    write_csv(
        OUT_DIR / "rollout_profile.csv",
        ["metric", "value", "status", "evidence"],
        rollout_profile,
    )

    summary = {
        "repo_root": str(REPO_ROOT),
        "asset_inventory_rows": len(inventory),
        "duplicate_texture_groups": len(duplicate_textures),
        "wrapper_main_xml_count": len(wrapper_xmls()),
        "mujoco_model_cache_rows": len(cache_report),
        "contact_report_rows": len(contact_report),
        "rollout_profile_status": "not_measured_static_placeholder",
        "outputs": {
            "asset_inventory": str(OUT_DIR / "asset_inventory.csv"),
            "duplicate_textures": str(OUT_DIR / "duplicate_textures.csv"),
            "mujoco_model_cache_report": str(OUT_DIR / "mujoco_model_cache_report.csv"),
            "contact_stability_report": str(OUT_DIR / "contact_stability_report.csv"),
            "rollout_profile": str(OUT_DIR / "rollout_profile.csv"),
        },
    }
    (OUT_DIR / "simulator_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
