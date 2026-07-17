#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CDPR + LIBERO scene/object adapter.
- Chooses a LIBERO scene (e.g., desk) and places LIBERO objects with given poses.
- Generates a temporary wrapper MJCF that includes: [scene] + [cdpr.xml] + [placed objects].
- Boots existing HeadlessCDPRSimulation on that wrapper.

Usage examples:
  python cdpr_scene_switcher.py --scene desk
  python cdpr_scene_switcher.py --scene desk \
      --object orange_juice:0.50,0.50,0.00:0,0,0,1 \
      --object orange_juice:0.70,0.35,0.00:0,0,0.382683,0.92388

Notes:
- Object pose format: name: x,y,z : qx,qy,qz,qw   (quat optional; defaults to 0,0,0,1)
- Scene names and object names correspond to directory names under LIBERO assets.
"""

from __future__ import annotations

import os, sys, argparse, tempfile, shutil, textwrap, time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from typing import Optional

# --- repo-local module path (needed only for optional smoke demo import) ---
sys.path.append(str(Path(__file__).parent))

HERE = Path(__file__).resolve().parent
ROBOT_ROOT = HERE.parent
REPO = HERE.parents[2]


def _resolve_libero_assets() -> Path:
    env_candidates = [
        os.environ.get("LIBERO_ASSETS"),
        os.environ.get("LIBERO_ASSETS_ROOT"),
    ]
    candidates: list[Path] = []
    for env_value in env_candidates:
        if env_value:
            candidates.append(Path(env_value).expanduser())

    candidates.extend(
        [
            REPO / "assets" / "externals" / "libero",
            REPO / "assets" / "externals" / "libero" / "libero" / "libero" / "assets",
            REPO / "LIBERO" / "libero" / "libero" / "assets",
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "scenes").is_dir():
            return resolved
    return candidates[0].resolve()


LIBERO_ASSETS = _resolve_libero_assets()
SCENES_DIR = LIBERO_ASSETS / "scenes"

# --- NEW: two object roots ---
OBJECTS_DIR_MAIN  = LIBERO_ASSETS / "stable_hope_objects"      # sauces, milk, etc.
OBJECTS_DIR_EXTRA = LIBERO_ASSETS / "stable_scanned_objects"   # bowls, plates, basket, ...
OBJECTS_DIRS = [OBJECTS_DIR_MAIN, OBJECTS_DIR_EXTRA]
STABLE_OBJECTS_DIR = REPO / "robots" / "cdpr" / "cdpr_mujoco" / "stable_objects"

CONTACT_PRESETS = ("legacy", "stable_contact")
STABLE_CONTACT_TIMESTEP = 0.002
STABLE_CONTACT_OPTION_ATTRS = {
    "timestep": f"{STABLE_CONTACT_TIMESTEP:.6g}",
    "solver": "Newton",
    "iterations": "100",
    "tolerance": "1e-10",
    "cone": "elliptic",
    "noslip_iterations": "5",
}
GRIPPER_OBJECT_PAIR_ATTRS = {
    "condim": "4",
    "friction": "3.0 0.08 0.003",
    "solref": "0.005 1",
    "solimp": "0.95 0.99 0.001",
    "margin": "0.001",
}
TABLE_OBJECT_PAIR_ATTRS = {
    "condim": "3",
    "friction": "0.9 0.01 0.001",
    "solref": "0.008 1",
    "solimp": "0.90 0.95 0.001",
    "margin": "0.001",
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _normalize_contact_preset(value: Optional[str] = None) -> str:
    raw = str(value or os.environ.get("RLVLA_CDPR_CONTACT_PRESET", "legacy")).strip() or "legacy"
    if raw not in CONTACT_PRESETS:
        raise ValueError(f"Unsupported contact_preset={raw!r}. Expected one of {CONTACT_PRESETS}.")
    return raw


def _find_stable_object_xml(object_name: str, normalized_name: str) -> Path | None:
    candidates = [str(object_name), str(normalized_name)]
    aliases = {
        "apple": "ycb_apple",
        "pear": "ycb_pear",
        "peach": "ycb_peach",
        "baseball": "ycb_baseball",
        "ball": "stable_sphere",
        "sphere": "stable_sphere",
        "cube": "stable_block",
        "block": "stable_block",
        "can": "stable_can",
        "cup": "ycb_b_cups",
    }
    for candidate in list(candidates):
        if candidate in aliases:
            candidates.append(aliases[candidate])
    for candidate in dict.fromkeys(candidates):
        path = STABLE_OBJECTS_DIR / f"{candidate}.xml"
        if path.exists():
            return path.resolve()
    return None

# --- EXTRA ASSETS (YCB) ---
# Prefer env var, otherwise probe common layouts used across local and remote repos.
def _resolve_ycb_root() -> Path:
    env_candidates = [
        os.environ.get("YCB_ASSETS"),
        os.environ.get("YCB_ASSETS_ROOT"),
    ]
    for env_root in env_candidates:
        if env_root:
            resolved = Path(env_root).expanduser().resolve()
            if resolved.name == "ycb":
                return resolved
            if (resolved / "ycb").is_dir():
                return (resolved / "ycb").resolve()
            return resolved

    candidates = [
        REPO / "assets" / "externals" / "ycb",
        REPO / "assets" / "externals" / "ycb_dataset" / "ycb",
        REPO / "robots" / "cdpr" / "assets" / "externals" / "ycb",
        REPO / "CDPR-Dataset" / "cdpr_dataset" / "external_assets" / "ycb_dataset" / "ycb",
        REPO / "CDPR-Dataset" / "external_assets" / "ycb_dataset" / "ycb",
        REPO / "external_assets" / "ycb_dataset" / "ycb",
    ]
    for cand in candidates:
        p = cand.resolve()
        if p.exists():
            return p
    return candidates[0].resolve()

YCB_ROOT = _resolve_ycb_root()


def _resolve_robotwin_roots() -> list[Path]:
    env_candidates = [
        os.environ.get("ROBOTWIN_ASSETS"),
        os.environ.get("ROBOTWIN_ASSETS_ROOT"),
    ]
    roots: list[Path] = []
    for env_value in env_candidates:
        if env_value:
            roots.append(Path(env_value).expanduser().resolve())

    roots.extend(
        [
            REPO / "assets" / "externals" / "robotwin2_assets",
            REPO / "benchmarks" / "externals" / "robotwin2" / "assets",
            REPO / "RoboTwin2.0" / "assets",
        ]
    )

    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            out.append(resolved)
    return out


ROBOTWIN_ROOTS = _resolve_robotwin_roots()

CDPR_XML = HERE / "cdpr.xml"


def _atomic_temp_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _atomic_write_xml_tree(tree: ET.ElementTree, path: Path) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _atomic_temp_path(path)
    try:
        tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _atomic_write_text(path: Path, content: str) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _atomic_temp_path(path)
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


# replace your existing preprocess_scene_with_zoffset(...) with this:
def preprocess_scene_with_zoffset(scene_xml: Path, z_offset: float, out_xml: Path):
    """
    Load the LIBERO scene MJCF and:
      1) rewrite all <asset> file paths (mesh/texture/skin) to ABSOLUTE paths
         based on the scene directory,
      2) add z_offset (meters) to the 'pos' z of every top-level body in <worldbody>.
    Write to out_xml.
    """
    src_dir = scene_xml.parent.resolve()
    tree = ET.parse(scene_xml)
    root = tree.getroot()

    # 1) rewrite asset file paths to absolute
    for a in root.findall("asset"):
        for sub in list(a):
            if sub.tag in ("mesh", "texture", "skin") and "file" in sub.attrib:
                rel = sub.get("file")
                abs_path = (src_dir / rel).resolve()
                sub.set("file", str(abs_path))

    # 2) shift top-level bodies' z
    for wb in root.findall("worldbody"):
        for body in list(wb.findall("body")):
            pos_str = body.get("pos", "0 0 0").split()
            if len(pos_str) == 3:
                x, y, z = map(float, pos_str)
                body.set("pos", f"{x} {y} {z + z_offset}")

    try:
        ET.indent(tree)
    except Exception:
        pass
    _atomic_write_xml_tree(tree, out_xml)

def preprocess_cdpr_set_ee_start(cdpr_xml: Path, ee_xyz: np.ndarray, out_xml: Path):
    """
    Overwrite 'pos' of body name='ee_base' in cdpr.xml.

    The override is written into a wrapper-cache directory, so relative mesh
    and texture paths must remain relative to the source robot XML rather than
    silently changing meaning beside the generated file.
    """
    cdpr_xml = Path(cdpr_xml).expanduser().resolve()
    out_xml = Path(out_xml).expanduser().resolve()
    tree = ET.parse(cdpr_xml)
    root = tree.getroot()
    found = False
    for wb in root.findall("worldbody"):
        # search recursively
        stack = list(wb.findall("body"))
        while stack:
            b = stack.pop()
            if b.get("name") == "ee_base":
                b.set("pos", f"{ee_xyz[0]} {ee_xyz[1]} {ee_xyz[2]}")
                found = True
                break
            stack.extend(list(b.findall("body")))
    if not found:
        raise ValueError("Could not find body name='ee_base' in cdpr.xml.")
    for element in root.iter():
        file_attr = str(element.get("file") or "").strip()
        if not file_attr:
            continue
        source_path = Path(file_attr).expanduser()
        if not source_path.is_absolute():
            source_path = (cdpr_xml.parent / source_path).resolve()
        element.set("file", os.path.relpath(source_path, start=out_xml.parent))
    try:
        ET.indent(tree)
    except Exception:
        pass
    _atomic_write_xml_tree(tree, out_xml)


def parse_object_arg(arg: str):
    """
    Parse: name: x,y,z : qx,qy,qz,qw
    Quat is optional; defaults to identity.
    """
    parts = [p.strip() for p in arg.split(":")]
    if len(parts) < 2:
        raise ValueError(f"Bad --object spec '{arg}'. Expected 'name:x,y,z[:qx,qy,qz,qw]'")
    name = parts[0]
    pos = np.fromstring(parts[1], sep=",", dtype=float)
    if pos.size != 3:
        raise ValueError(f"Bad position in '{arg}'. Need x,y,z")
    if len(parts) >= 3:
        quat = np.fromstring(parts[2], sep=",", dtype=float)
        if quat.size != 4:
            raise ValueError(f"Bad quaternion in '{arg}'. Need qx,qy,qz,qw")
    else:
        quat = np.array([0,0,0,1.0], dtype=float)
    return name, pos, quat

def find_scene_xml(scene_name: str) -> Path:
    cand = SCENES_DIR / scene_name / f"{scene_name}.xml"
    if not cand.exists():
        raise FileNotFoundError(f"Scene '{scene_name}' not found at {cand}")
    return cand

def find_object_xml(object_name: str) -> Path:
    """
    Search for object xml in:
      1) LIBERO stable_hope_objects + stable_scanned_objects
      2) YCB dataset (ycb/<name>.xml)
    Supports prefix:
      - "ycb_apple" -> "apple"
    """
    name = object_name
    if name.startswith("ycb_"):
        name = name[len("ycb_"):]

    stable_first = _env_flag("RLVLA_CDPR_USE_STABLE_OBJECTS", default=False) or str(object_name).startswith("stable_")
    if stable_first:
        stable = _find_stable_object_xml(object_name, name)
        if stable is not None:
            return stable

    # 1) LIBERO roots (your current behavior)
    for root in OBJECTS_DIRS:
        d = root / name
        cand = d / f"{name}.xml"
        if cand.exists():
            return cand
        if d.exists():
            xmls = list(d.glob("*.xml"))
            if xmls:
                return xmls[0]

    # 2) YCB root: ycb/<name>.xml (your repo layout matches this)
    cand = YCB_ROOT / f"{name}.xml"
    if cand.exists():
        return cand

    # fallback: try any xml matching name
    xmls = list(YCB_ROOT.glob(f"{name}*.xml"))
    if xmls:
        return xmls[0]

    # 3) RoboTwin-style staged assets: search recursively to tolerate unknown layout.
    for root in ROBOTWIN_ROOTS:
        direct = root / f"{name}.xml"
        if direct.exists():
            return direct
        nested_dir = root / name
        if nested_dir.is_dir():
            nested_xml = nested_dir / f"{name}.xml"
            if nested_xml.exists():
                return nested_xml
        matches = list(root.rglob(f"{name}.xml"))
        if matches:
            return matches[0]

    stable = _find_stable_object_xml(object_name, name)
    if stable is not None:
        return stable

    raise FileNotFoundError(
        f"Object '{object_name}' not found.\n"
        f"Checked LIBERO roots: {', '.join(str(r) for r in OBJECTS_DIRS)}\n"
        f"Checked YCB root: {YCB_ROOT}\n"
        f"Checked RoboTwin roots: {', '.join(str(r) for r in ROBOTWIN_ROOTS) if ROBOTWIN_ROOTS else '(none)'}\n"
        f"Checked stable object pack: {STABLE_OBJECTS_DIR}"
    )


def make_placed_object_xml(orig_object_xml: Path,
                           out_xml: Path,
                           prefix: str,
                           pos,
                           quat,
                           force_dynamic: bool = False,
                           logical_name: Optional[str] = None):
    import xml.etree.ElementTree as ET

    def clone(elem):
        return ET.fromstring(ET.tostring(elem))

    src_dir = orig_object_xml.parent.resolve()
    tree = ET.parse(orig_object_xml)
    root = tree.getroot()

    default_children = []
    for default_elem in root.findall("default"):
        for child in list(default_elem):
            default_children.append(clone(child))

    class_map: dict[str, str] = {}

    def prefix_default_classes(elem):
        if elem.tag == "default" and "class" in elem.attrib:
            old = elem.get("class")
            if old:
                new = f"{prefix}_{old}"
                class_map[old] = new
                elem.set("class", new)
        for child in list(elem):
            prefix_default_classes(child)

    for child in default_children:
        prefix_default_classes(child)

    # --- collect <asset> and absolutize files
    asset_elems = []
    for a in root.findall("asset"):
        a_copy = clone(a)
        for sub in a_copy:
            if sub.tag in ("mesh", "texture", "skin") and "file" in sub.attrib:
                rel = sub.get("file")
                p1 = (src_dir / rel).resolve()

                # If not found, try <xml_basename>/<rel>
                # Example: ycb/apple.xml + "textured.obj" -> ycb/apple/textured.obj
                if not p1.exists():
                    base = orig_object_xml.stem  # e.g. "apple"
                    p2 = (src_dir / base / rel).resolve()
                    if p2.exists():
                        p1 = p2

                sub.set("file", str(p1))

        asset_elems.append(a_copy)

    # take first body under worldbody
    worldbodies = root.findall("worldbody")
    if not worldbodies:
        raise ValueError(f"{orig_object_xml} has no <worldbody> section.")
    bodies = []
    for wb in worldbodies:
        bodies.extend(list(wb.findall("body")))
    if not bodies:
        raise ValueError(f"{orig_object_xml} has no <body> inside <worldbody>.")
    body_clone = clone(bodies[0])
    if logical_name is not None:
        body_clone.set("name", logical_name)

    # --- prefix names for body tree (to avoid geom/site/joint clashes)
    NAME_TAGS = {"body", "geom", "site", "joint", "camera", "light"}
    def prefix_names(elem):
        if elem.tag in NAME_TAGS and "name" in elem.attrib:
            elem.set("name", f"{prefix}_{elem.get('name')}")
        for child in list(elem):
            prefix_names(child)
    prefix_names(body_clone)

    # --- prefix asset names and build remap dicts
    mesh_map, tex_map, mat_map, skin_map, hf_map = {}, {}, {}, {}, {}

    def maybe_pref(attr_map, old):
        if old is None:
            return None
        new = f"{prefix}_{old}"
        attr_map[old] = new
        return new

    for a in asset_elems:
        for sub in list(a):
            if "name" in sub.attrib:
                old = sub.get("name")
                if sub.tag == "mesh":
                    sub.set("name", maybe_pref(mesh_map, old))
                elif sub.tag == "texture":
                    sub.set("name", maybe_pref(tex_map, old))
                elif sub.tag == "material":
                    sub.set("name", maybe_pref(mat_map, old))
                elif sub.tag == "skin":
                    sub.set("name", maybe_pref(skin_map, old))
                elif sub.tag == "hfield":
                    sub.set("name", maybe_pref(hf_map, old))

    # --- update references *inside assets* (material.texture, skin.texture)
    for a in asset_elems:
        for sub in list(a):
            if sub.tag == "material" and "texture" in sub.attrib:
                t = sub.get("texture")
                if t in tex_map:
                    sub.set("texture", tex_map[t])
            if sub.tag == "skin" and "texture" in sub.attrib:
                t = sub.get("texture")
                if t in tex_map:
                    sub.set("texture", tex_map[t])

    # --- update references inside the body tree (geom.mesh/material/hfield/skin)
    def rewrite_body_refs(elem):
        if "class" in elem.attrib:
            cls = elem.get("class")
            if cls in class_map:
                elem.set("class", class_map[cls])
        if elem.tag == "geom":
            if "mesh" in elem.attrib:
                m = elem.get("mesh")
                if m in mesh_map:
                    elem.set("mesh", mesh_map[m])
            if "material" in elem.attrib:
                m = elem.get("material")
                if m in mat_map:
                    elem.set("material", mat_map[m])
            if "hfield" in elem.attrib:
                h = elem.get("hfield")
                if h in hf_map:
                    elem.set("hfield", hf_map[h])
            if "skin" in elem.attrib:
                s = elem.get("skin")
                if s in skin_map:
                    elem.set("skin", skin_map[s])
        for ch in list(elem):
            rewrite_body_refs(ch)
    rewrite_body_refs(body_clone)

    # --- pose ---
    body_clone.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
    
    # MuJoCo allows only ONE orientation specifier on <body>
    for k in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
        if k in body_clone.attrib:
            del body_clone.attrib[k]

    # incoming is qx,qy,qz,qw  -> MuJoCo wants w x y z
    wxyz = (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
    body_clone.set("quat", f"{wxyz[0]} {wxyz[1]} {wxyz[2]} {wxyz[3]}")

    # ensure dynamic if requested (normalize to exactly one 6-DoF freejoint)
    if force_dynamic:
        # Remove ALL <joint> tags in the subtree (rigid-object normalization)
        def strip_joints(elem):
            for ch in list(elem):
                if ch.tag == "joint":
                    elem.remove(ch)
                else:
                    strip_joints(ch)
        strip_joints(body_clone)

        # Remove any existing <freejoint> tags too (avoid duplicates)
        def strip_freejoints(elem):
            for ch in list(elem):
                if ch.tag == "freejoint":
                    elem.remove(ch)
                else:
                    strip_freejoints(ch)
        strip_freejoints(body_clone)

        # Add exactly one freejoint on the root body
        ET.SubElement(body_clone, "freejoint")


    # --- build minimal MJCF with assets + body
    mj = ET.Element("mujoco")
    comp = ET.SubElement(mj, "compiler"); comp.set("autolimits", "true")

    default = ET.SubElement(mj, "default")
    geomdef = ET.SubElement(default, "geom")
    geomdef.set("density", "1200")
    for child in default_children:
        if child.tag == "geom" and "class" not in child.attrib:
            for key, value in child.attrib.items():
                geomdef.set(key, value)
        else:
            default.append(child)

    if asset_elems:
        new_asset = ET.SubElement(mj, "asset")
        for a in asset_elems:
            for sub in list(a):
                new_asset.append(sub)

    new_wb = ET.SubElement(mj, "worldbody")
    new_wb.append(body_clone)

    try:
        ET.indent(mj)
    except Exception:
        pass
    _atomic_write_xml_tree(ET.ElementTree(mj), out_xml)

def _xml_attrs(attrs: dict[str, str]) -> str:
    return " ".join(f'{key}="{value}"' for key, value in attrs.items())


def _stable_contact_option_block(contact_preset: str) -> str:
    if contact_preset != "stable_contact":
        return ""
    return f"    <option {_xml_attrs(STABLE_CONTACT_OPTION_ATTRS)}/>"


def _stable_contact_defaults_block(contact_preset: str) -> str:
    if contact_preset != "stable_contact":
        return ""
    return """    <default>
      <default class="table_collision">
        <geom contype="4" conaffinity="1" group="0" condim="3"
              friction="0.9 0.01 0.001" solref="0.008 1"
              solimp="0.90 0.95 0.001" margin="0.001"/>
      </default>
    </default>"""


def _geom_looks_like_support_table(geom: ET.Element) -> bool:
    name = str(geom.get("name") or "").lower()
    cls = str(geom.get("class") or "").lower()
    if any(token in name or token in cls for token in ("table", "desk", "workbench", "counter", "surface")):
        return True
    size = geom.get("size")
    if not size:
        return False
    try:
        vals = [float(x) for x in str(size).replace(",", " ").split()]
    except Exception:
        return False
    if len(vals) < 3:
        return False
    gtype = str(geom.get("type") or "box").lower()
    sx, sy, sz = vals[0], vals[1], vals[2]
    return gtype == "box" and sx >= 0.15 and sy >= 0.15 and sz <= 0.06


def _collect_geom_names(xml_path: Path, predicate) -> list[str]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []
    names: list[str] = []
    for geom in root.iter("geom"):
        name = str(geom.get("name") or "").strip()
        if name and predicate(name, geom):
            names.append(name)
    return names


def _collect_table_geom_names(scene_xml: Path, extra_names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    names = list(extra_names or [])
    names.extend(_collect_geom_names(scene_xml, lambda _name, geom: _geom_looks_like_support_table(geom)))
    return list(dict.fromkeys(name for name in names if str(name).strip()))


def _is_stable_contact_object_geom(name: str, geom: ET.Element) -> bool:
    lname = str(name).lower()
    if "collision" not in lname:
        return False
    if not any(token in lname for token in ("ycb_apple", "apple", "ycb_pear", "pear")):
        return False
    return str(geom.get("contype", "1")).strip() != "0"


def _collect_stable_contact_object_geoms(placed_object_xmls: list[Path]) -> list[str]:
    names: list[str] = []
    for placed_xml in placed_object_xmls:
        names.extend(_collect_geom_names(placed_xml, _is_stable_contact_object_geom))
    return list(dict.fromkeys(names))


def _geom_exists(xml_path: Path, geom_name: str) -> bool:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return False
    return any(geom.get("name") == geom_name for geom in root.iter("geom"))


def _stable_contact_pairs_block(
    *,
    contact_preset: str,
    cdpr_xml: Path,
    scene_xml: Path,
    placed_object_xmls: list[Path],
    table_geom_names: list[str] | tuple[str, ...] | None,
) -> str:
    if contact_preset != "stable_contact":
        return ""

    object_geoms = _collect_stable_contact_object_geoms(placed_object_xmls)
    if not object_geoms:
        return ""

    finger_pads = [
        name for name in ("left_finger_pad", "right_finger_pad")
        if _geom_exists(cdpr_xml, name)
    ]
    table_geoms = _collect_table_geom_names(scene_xml, table_geom_names)
    lines = ["    <contact>"]
    for pad in finger_pads:
        for object_geom in object_geoms:
            attrs = {"geom1": pad, "geom2": object_geom, **GRIPPER_OBJECT_PAIR_ATTRS}
            lines.append(f"      <pair {_xml_attrs(attrs)}/>")
    for table_geom in table_geoms:
        for object_geom in object_geoms:
            attrs = {"geom1": table_geom, "geom2": object_geom, **TABLE_OBJECT_PAIR_ATTRS}
            lines.append(f"      <pair {_xml_attrs(attrs)}/>")
    lines.append("    </contact>")
    return os.linesep.join(lines) if len(lines) > 2 else ""


def build_wrapper_mjcf(
    scene_xml: Path,
    cdpr_xml: Path,
    placed_object_xmls: list[Path],
    out_xml: Path,
    *,
    contact_preset: Optional[str] = None,
    table_geom_names: Optional[list[str] | tuple[str, ...]] = None,
):
    scene_xml = scene_xml.resolve()
    cdpr_xml  = cdpr_xml.resolve()
    scene_dir = scene_xml.parent.resolve()
    contact_preset = _normalize_contact_preset(contact_preset)
    # includes for placed objects use absolute asset paths, so no meshdir needed
    includes_objects = os.linesep.join([f'<include file="{str(p.resolve())}"/>' for p in placed_object_xmls])
    option_block = _stable_contact_option_block(contact_preset)
    defaults_block = _stable_contact_defaults_block(contact_preset)
    contact_block = _stable_contact_pairs_block(
        contact_preset=contact_preset,
        cdpr_xml=cdpr_xml,
        scene_xml=scene_xml,
        placed_object_xmls=placed_object_xmls,
        table_geom_names=table_geom_names,
    )

    content = f"""<mujoco>
    <compiler autolimits="true"/>
{option_block}
{defaults_block}

    <!-- Set mesh/texture dirs for the SCENE only -->
    <compiler meshdir="{str(scene_dir)}" texturedir="{str(scene_dir)}"/>
    <include file="{str(scene_xml)}"/>

    <!-- Reset to neutral (our CDPR uses primitives / absolute includes) -->
    <compiler meshdir="" texturedir=""/>

    <!-- CDPR rig -->
    <include file="{str(cdpr_xml)}"/>

    <!-- Placed LIBERO objects (assets already absolute) -->
    {includes_objects}
{contact_block}
    </mujoco>
    """
    out_xml.parent.mkdir(parents=True, exist_ok=True)  # ensure /.../wrappers/ exists
    _atomic_write_text(out_xml, content)


    

def main():
    ap = argparse.ArgumentParser(description="Build CDPR+LIBERO wrapper MJCF and optionally run a smoke demo.")
    ap.add_argument("--scene", required=True, help="Scene name under assets/scenes (e.g., 'desk').")
    ap.add_argument("--object", action="append", default=[],
                    help="Object placement: name:x,y,z[:qx,qy,qz,qw]. Repeat for multiple.")
    ap.add_argument(
        "--outdir",
        default=str(HERE / "trajectory_results"),
        help="Output directory for smoke demo videos/data (used only with --run_demo).",
    )
    ap.add_argument(
        "--run_demo",
        action="store_true",
        help="After building wrapper, run a short smoke demo and save trajectory results.",
    )
    ap.add_argument(
        "--demo_name",
        type=str,
        default="scene_switcher",
        help="Subdirectory name under --outdir for smoke demo outputs.",
    )
    ap.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Optional instruction string saved into summary/npz and instruction.txt for demo runs.",
    )
    ap.add_argument("--hover", default="0.5,0.5,0.35", help="Hover xyz over object for the demo (x,y,z).")
    ap.add_argument("--graspz", type=float, default=0.06, help="Grasp height z.")
    ap.add_argument("--liftz",  type=float, default=0.35, help="Lift height z.")
    ap.add_argument("--yaw",    type=float, default=0.0,  help="Yaw target (rad) for the demo.")
    ap.add_argument("--steps",  type=int,   default=120,  help="Goto() max steps for segments.")
    ap.add_argument("--scene_z", type=float, default=0.0,
                help="Additive z-offset (meters) applied to EVERY body in the LIBERO scene (e.g., -0.25 lowers the table).")
    ap.add_argument("--ee_start", default=None,
                    help="Override CDPR ee_base start pos as 'x,y,z'. If unset, use whatever is in cdpr.xml.")
    ap.add_argument("--object_on_table", action="store_true",
                    help="Force object z to table_z.")
    ap.add_argument("--table_z", type=float, default=0.0,
                    help="Tabletop height (meters) used when --object_on_table is set.")
    ap.add_argument("--object_dynamic", action="store_true",
                help="Force placed objects to be dynamic (inject <freejoint/> if missing).")
    ap.add_argument("--settle_time", type=float, default=1.0,
                help="Seconds to simulate before the demo to let objects fall/settle.")
    ap.add_argument("--wrapper_out", default=None,
                help="Write final wrapper MJCF to this path (kept). If unset, uses a temp dir.")
    ap.add_argument("--keep", action="store_true",
                help="Don’t delete the temp working directory (for debugging).")
    ap.add_argument("--build_only", action="store_true",
                help="Only build wrapper MJCF and exit (no demo, no videos).")
    ap.add_argument(
        "--contact_preset",
        choices=CONTACT_PRESETS,
        default=os.environ.get("RLVLA_CDPR_CONTACT_PRESET", "legacy"),
        help="Contact/solver preset for generated wrappers.",
    )
    ap.add_argument(
        "--debug_render_collision_geoms",
        action="store_true",
        default=_env_flag("RLVLA_CDPR_DEBUG_RENDER_COLLISION_GEOMS", default=False),
        help="Render group-3 collision proxy geoms in HeadlessCDPRSimulation demos.",
    )
    
    args = ap.parse_args()

    # `--build_only` remains for backward compatibility; default behavior is build-only unless --run_demo.
    run_demo = bool(args.run_demo and not args.build_only)
    if args.build_only and args.run_demo:
        print("ℹ️ Both --build_only and --run_demo were provided; skipping demo due to --build_only.")

    work_tmpdir = Path(tempfile.mkdtemp(prefix="cdpr_scene_", dir=str(HERE)))
    sim = None
    try:
        # Where to write generated wrapper and helper XML files.
        if args.wrapper_out:
            wrapper_xml = Path(args.wrapper_out).expanduser().resolve()
            wrapper_xml.parent.mkdir(parents=True, exist_ok=True)
            gen_base = wrapper_xml.parent
        else:
            wrapper_xml = work_tmpdir / "cdpr_scene_wrapper.xml"
            gen_base = work_tmpdir

        scene_xml = find_scene_xml(args.scene)
        if not CDPR_XML.exists():
            raise FileNotFoundError(f"CDPR XML not found at {CDPR_XML}")

        if abs(args.scene_z) > 1e-6:
            scene_for_include = gen_base / f"{args.scene}_zshift.xml"
            preprocess_scene_with_zoffset(scene_xml, args.scene_z, scene_for_include)
        else:
            scene_for_include = scene_xml

        if args.ee_start is not None:
            ee_xyz = np.fromstring(args.ee_start, sep=",", dtype=float)
            if ee_xyz.size != 3:
                raise ValueError("--ee_start must be 'x,y,z'")
            cdpr_for_include = gen_base / "cdpr_ee_override.xml"
            preprocess_cdpr_set_ee_start(CDPR_XML, ee_xyz, cdpr_for_include)
        else:
            cdpr_for_include = CDPR_XML

        placements = []
        for ob in args.object:
            name, pos, quat = parse_object_arg(ob)
            if args.object_on_table:
                pos[2] = float(args.table_z)
            obj_xml = find_object_xml(name)
            placements.append((name, obj_xml, pos, quat))

        placed_xmls = []
        for idx, (name, obj_xml, pos, quat) in enumerate(placements):
            placed_path = gen_base / f"placed_{idx}_{name}.xml"
            make_placed_object_xml(
                obj_xml,
                placed_path,
                prefix=f"p{idx}",
                pos=pos,
                quat=quat,
                force_dynamic=args.object_dynamic,
                logical_name=name,
            )
            placed_xmls.append(placed_path)

        build_wrapper_mjcf(
            scene_for_include,
            cdpr_for_include,
            placed_xmls,
            wrapper_xml,
            contact_preset=args.contact_preset,
        )
        print(f"✅ Built wrapper: {wrapper_xml}")
        print(f"   Includes {len(placed_xmls)} object(s).")

        if not run_demo:
            print("✅ Wrapper build completed (demo disabled).")
            return

        if args.debug_render_collision_geoms:
            os.environ["RLVLA_CDPR_DEBUG_RENDER_COLLISION_GEOMS"] = "1"
        from headless_cdpr_egl import HeadlessCDPRSimulation
        sim_timestep = STABLE_CONTACT_TIMESTEP if args.contact_preset == "stable_contact" else None
        sim = HeadlessCDPRSimulation(
            str(wrapper_xml),
            output_dir=args.outdir,
            timestep=sim_timestep,
            debug_render_collision_geoms=bool(args.debug_render_collision_geoms),
        )
        sim.initialize()

        if args.settle_time > 0:
            steps = max(1, int(round(args.settle_time / sim.controller.dt)))
            print(f"⏳ settling objects for {args.settle_time:.2f}s ({steps} steps)")
            for _ in range(steps):
                sim.run_simulation_step(capture_frame=False)

        if placements:
            ox, oy, _ = placements[0][2]
        else:
            ox, oy = 0.0, 0.0
        hover = np.fromstring(args.hover, sep=",", dtype=float)
        if hover.size != 3:
            hover = np.array([ox, oy, 0.35], dtype=float)

        try:
            sim.goto(hover, max_steps=args.steps)
            sim.set_yaw(args.yaw)
            sim.open_gripper()
            for _ in range(40):
                sim.run_simulation_step(capture_frame=True)
            sim.close_gripper()
            for _ in range(40):
                sim.run_simulation_step(capture_frame=True)
        except Exception as e:
            print("Warning during smoke motion:", e)

        instruction = str(args.instruction).strip()
        if instruction:
            setattr(sim, "language_instruction", instruction)

        demo_root = Path(args.outdir).expanduser().resolve()
        demo_dir = demo_root / args.demo_name
        demo_dir.mkdir(parents=True, exist_ok=True)
        sim.save_trajectory_results(str(demo_dir), args.demo_name)
        if instruction:
            (demo_dir / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")

        print(
            f"✅ Loaded scene '{args.scene}' with {len(placements)} object(s). "
            f"Wrapper at: {wrapper_xml}"
        )
    finally:
        if sim is not None:
            try:
                sim.cleanup()
            except Exception:
                pass

        if args.keep:
            print(f"🧪 Keeping temporary workspace: {work_tmpdir}")
        else:
            shutil.rmtree(work_tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
