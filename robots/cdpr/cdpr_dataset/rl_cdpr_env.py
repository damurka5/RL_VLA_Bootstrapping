from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml

try:
    import mujoco as mj
except Exception:  # pragma: no cover - optional runtime dependency
    mj = None

try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - optional runtime dependency
    gym = None
    spaces = None

from .rl_instruction_tasks import (
    INSTRUCTION_TYPES,
    compute_instruction_reward,
    init_reward_state,
    instruction_to_onehot,
    sample_instruction,
)
from .synthetic_tasks import (
    clamp_xyz,
    compute_cdpr_workspace_safety,
    clear_sim_recording_buffers,
    lift_cdpr_ee_to_spawn_height,
    place_objects_non_overlapping,
    resolve_body_name,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = HERE / "datasets" / "cdpr_scene_catalog.yaml"
DEFAULT_VIDEO_DIR = HERE / "datasets" / "cdpr_synth" / "videos"
DEFAULT_ALLOWED_OBJECTS: tuple[str, ...] = (
    "ycb_apple",
    "ycb_banana",
    "ycb_pear",
    "ycb_peach",
    "bowl",
    "plate",
    "ycb_baseball",
    "ycb_lemon",
    "ycb_fork",
    "ycb_hammer",
    "ycb_spoon",
)
DEFAULT_DESK_GEOM_REGEX = r"(table|desk|workbench|counter|surface)"
WRAP_DIR = HERE / "wrappers"
MIN_EE_START_Z = 0.40
TASK_REWARD_PREFIX = "RLVLA_TASK_REWARD"
TASK_SUCCESS_PREFIX = "RLVLA_TASK_SUCCESS"
CDPR_LOCK_NON_COMMANDED_AXES_ENV = "RLVLA_CDPR_LOCK_NON_COMMANDED_AXES"
CDPR_LOCK_NON_COMMANDED_AXES_THRESHOLD_ENV = "RLVLA_CDPR_LOCK_NON_COMMANDED_AXES_THRESHOLD"
CDPR_RANDOMIZE_EE_START_ENV = "RLVLA_CDPR_RANDOMIZE_EE_START"
CDPR_EE_START_X_BOUNDS_ENV = "RLVLA_CDPR_EE_START_X_BOUNDS"
CDPR_EE_START_Y_BOUNDS_ENV = "RLVLA_CDPR_EE_START_Y_BOUNDS"
CDPR_EE_START_Z_ENV = "RLVLA_CDPR_EE_START_Z"
DEFAULT_RANDOM_EE_START_X_BOUNDS = (-0.25, 0.25)
DEFAULT_RANDOM_EE_START_Y_BOUNDS = (-0.25, 0.25)
DEFAULT_GOAL_CENTER_XY = (0.0, 0.0)
DEFAULT_GOAL_HEIGHT_ABOVE_TABLE = 0.10

ROBOT_BODY_PREFIXES = (
    "world",
    "rotor_",
    "slider_",
    "ee_",
    "camera_",
    "yaw_frame",
    "ee_platform",
    "finger_",
)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    objects: tuple[str, ...]
    target_object: str | None = None

@dataclass
class DeskTexturePatchResult:
    wrapper_xml: Path
    generated_xmls: list[Path]
    generated_files: list[Path]
    chosen_texture: Path
    matched_geoms: int


@dataclass(frozen=True)
class WrapperBuilderHandle:
    build_wrapper_if_needed: Any
    list_wrapper_bundle_paths: Any

    def __call__(self, *args, **kwargs):
        return self.build_wrapper_if_needed(*args, **kwargs)

    def __iter__(self):
        yield self.build_wrapper_if_needed
        yield self.list_wrapper_bundle_paths


def _load_catalog(catalog_path: Path) -> tuple[dict[str, Any], list[SceneSpec]]:
    with catalog_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = dict(cfg.get("defaults", {}))
    scenes_raw = cfg.get("scenes", [])

    scenes: list[SceneSpec] = []
    for row in scenes_raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        objects = tuple(str(x) for x in (row.get("objects") or []))
        if name and objects:
            scenes.append(SceneSpec(name=name, objects=objects))

    if not scenes:
        raise ValueError(f"No scenes with objects found in catalog: {catalog_path}")

    return defaults, scenes


def _filter_scenes_to_allowed_objects(
    scenes: Sequence[SceneSpec], allowed_objects: Sequence[str]
) -> list[SceneSpec]:
    allowed_set = {str(x) for x in allowed_objects}
    if not allowed_set:
        return list(scenes)

    filtered: list[SceneSpec] = []
    for scene in scenes:
        objects = tuple(obj for obj in scene.objects if obj in allowed_set)
        if objects:
            filtered.append(SceneSpec(name=scene.name, objects=objects))

    if not filtered:
        raise ValueError(
            "No catalog scenes remain after allowed-object filtering. "
            f"Allowed objects: {sorted(allowed_set)}"
        )
    return filtered


def _dedupe_names(values: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def _metadata_name_list(task_metadata: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = task_metadata.get(key)
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence):
        raise ValueError(f"Task metadata `{key}` must be a list of object names.")
    return _dedupe_names([str(item) for item in raw])


def _unique_scene_names(scenes: Sequence[SceneSpec]) -> tuple[str, ...]:
    names = _dedupe_names([scene.name for scene in scenes if getattr(scene, "name", "")])
    if names:
        return names
    return ("desk",)


def _sample_scene_object_count(
    *,
    rng: np.random.Generator,
    min_objects: int,
    max_objects: int,
    total_available: int,
) -> int:
    upper = max(1, min(int(max_objects), int(total_available)))
    lower = max(1, min(int(min_objects), upper))
    return int(rng.integers(lower, upper + 1))


def _build_scene_object_variants(
    *,
    scene_names: Sequence[str],
    object_pool: Sequence[str],
    min_scene_objects: int,
    max_scene_objects: int,
    scene_variant_count: int,
    seed: int | None,
) -> list[SceneSpec]:
    object_pool = _dedupe_names(object_pool)
    if not object_pool:
        raise ValueError("Scene object pool cannot be empty when building scene variants.")

    scene_names = _unique_scene_names([SceneSpec(name=name, objects=()) for name in scene_names])
    rng = np.random.default_rng(0 if seed is None else int(seed))
    requested = max(int(scene_variant_count), len(scene_names))
    variants: list[SceneSpec] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def _make_variant(scene_name: str) -> SceneSpec:
        desired_count = _sample_scene_object_count(
            rng=rng,
            min_objects=min_scene_objects,
            max_objects=max_scene_objects,
            total_available=len(object_pool),
        )
        chosen = list(rng.choice(object_pool, size=desired_count, replace=False))
        chosen.sort()
        return SceneSpec(name=scene_name, objects=tuple(chosen))

    for scene_name in scene_names:
        variant = _make_variant(scene_name)
        key = (variant.name, tuple(sorted(variant.objects)))
        if key in seen:
            continue
        seen.add(key)
        variants.append(variant)

    max_attempts = max(requested * 8, 32)
    attempts = 0
    while len(variants) < requested and attempts < max_attempts:
        attempts += 1
        scene_name = str(scene_names[int(rng.integers(0, len(scene_names)))])
        variant = _make_variant(scene_name)
        key = (variant.name, tuple(sorted(variant.objects)))
        if key in seen:
            continue
        seen.add(key)
        variants.append(variant)

    return variants


def _build_scene_variants(
    *,
    scene_names: Sequence[str],
    target_object_pool: Sequence[str],
    distractor_object_pool: Sequence[str],
    min_scene_objects: int,
    max_scene_objects: int,
    scene_variant_count: int,
    seed: int | None,
) -> list[SceneSpec]:
    if not target_object_pool:
        raise ValueError("Target object pool cannot be empty when building scene variants.")

    scene_names = _unique_scene_names([SceneSpec(name=name, objects=()) for name in scene_names])
    targets = _dedupe_names(target_object_pool)
    distractors = _dedupe_names([name for name in distractor_object_pool if name not in targets])
    total_pool = _dedupe_names([*targets, *distractors])
    rng = np.random.default_rng(0 if seed is None else int(seed))

    requested = max(int(scene_variant_count), len(scene_names) * len(targets))
    variants: list[SceneSpec] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    def _make_variant(scene_name: str, target_name: str) -> SceneSpec:
        desired_count = _sample_scene_object_count(
            rng=rng,
            min_objects=min_scene_objects,
            max_objects=max_scene_objects,
            total_available=len(total_pool),
        )
        max_distractors = max(0, desired_count - 1)
        distractor_candidates = [name for name in distractors if name != target_name]
        chosen: list[str] = []
        if distractor_candidates and max_distractors > 0:
            sample_size = min(max_distractors, len(distractor_candidates))
            chosen = list(rng.choice(distractor_candidates, size=sample_size, replace=False))
            chosen.sort()
        return SceneSpec(
            name=scene_name,
            objects=tuple([target_name, *chosen]),
            target_object=target_name,
        )

    for scene_name in scene_names:
        for target_name in targets:
            variant = _make_variant(scene_name, target_name)
            key = (variant.name, variant.target_object or "", tuple(sorted(variant.objects)))
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant)

    max_attempts = max(requested * 8, 32)
    attempts = 0
    while len(variants) < requested and attempts < max_attempts:
        attempts += 1
        scene_name = str(scene_names[int(rng.integers(0, len(scene_names)))])
        target_name = str(targets[int(rng.integers(0, len(targets)))])
        variant = _make_variant(scene_name, target_name)
        key = (variant.name, variant.target_object or "", tuple(sorted(variant.objects)))
        if key in seen:
            continue
        seen.add(key)
        variants.append(variant)

    return variants


def _configure_scene_sampling(
    *,
    base_scenes: Sequence[SceneSpec],
    allowed_objects: Sequence[str],
    task_metadata: dict[str, Any],
    seed: int | None,
) -> tuple[list[SceneSpec], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    scene_object_pool = _metadata_name_list(task_metadata, "scene_object_pool")
    if scene_object_pool:
        allowed = _dedupe_names(scene_object_pool)
        min_scene_objects = int(task_metadata.get("min_scene_objects", 1))
        max_scene_objects = int(task_metadata.get("max_scene_objects", min(3, max(1, len(allowed)))))
        scene_variant_count = int(
            task_metadata.get(
                "scene_variant_count",
                max(len(base_scenes), len(_unique_scene_names(base_scenes)) * max(1, len(allowed))),
            )
        )
        scenes = _build_scene_object_variants(
            scene_names=_unique_scene_names(base_scenes),
            object_pool=allowed,
            min_scene_objects=min_scene_objects,
            max_scene_objects=max_scene_objects,
            scene_variant_count=scene_variant_count,
            seed=seed,
        )
        return scenes, allowed, (), ()

    target_pool = _metadata_name_list(task_metadata, "target_object_pool")
    distractor_pool = _metadata_name_list(task_metadata, "distractor_object_pool")

    if not target_pool and not distractor_pool:
        allowed = _dedupe_names(allowed_objects)
        return list(base_scenes), allowed, allowed, ()

    if not target_pool:
        target_pool = _dedupe_names(allowed_objects)
    if not target_pool:
        raise ValueError("Task metadata target_object_pool is empty and no allowed_objects were provided.")

    distractor_pool = _dedupe_names([name for name in distractor_pool if name not in target_pool])
    allowed = _dedupe_names([*target_pool, *distractor_pool])

    min_scene_objects = int(task_metadata.get("min_scene_objects", 1))
    max_scene_objects = int(task_metadata.get("max_scene_objects", max(min_scene_objects, len(allowed))))
    scene_variant_count = int(
        task_metadata.get(
            "scene_variant_count",
            max(len(base_scenes), len(_unique_scene_names(base_scenes)) * max(1, len(target_pool))),
        )
    )

    scenes = _build_scene_variants(
        scene_names=_unique_scene_names(base_scenes),
        target_object_pool=target_pool,
        distractor_object_pool=distractor_pool,
        min_scene_objects=min_scene_objects,
        max_scene_objects=max_scene_objects,
        scene_variant_count=scene_variant_count,
        seed=seed,
    )
    return scenes, allowed, target_pool, distractor_pool


def _iter_includes(tree_root: ET.Element):
    for inc in tree_root.iter("include"):
        file_attr = inc.get("file")
        if file_attr:
            yield inc, file_attr


def _resolve_include_path(current_xml: Path, file_attr: str) -> Path:
    p = Path(file_attr)
    if p.is_absolute():
        return p
    return (current_xml.parent / p).resolve()


def _relpath_or_abs(target: Path, base_dir: Path) -> str:
    try:
        return target.relative_to(base_dir).as_posix()
    except Exception:
        return target.as_posix()


def _ensure_asset_first(root: ET.Element) -> ET.Element:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
        return asset

    children = list(root)
    idx = children.index(asset)
    if idx != 0:
        root.remove(asset)
        root.insert(0, asset)
    return asset


def _geom_looks_like_table(geom: ET.Element) -> bool:
    size = geom.get("size")
    gtype = (geom.get("type") or "").lower()
    if not size:
        return False
    try:
        vals = [float(x) for x in size.replace(",", " ").split()]
        if len(vals) < 3:
            return False
        sx, sy, sz = vals[0], vals[1], vals[2]
    except Exception:
        return False

    return gtype in ("box", "") and sx > 0.15 and sy > 0.15 and sz < 0.06


def _patch_xml_tree_for_desk_material(
    source_xml: Path,
    variant_tag: str,
    desk_mat_name: str,
    table_regex: re.Pattern[str],
    mapping: dict[Path, Path],
    generated_xmls: list[Path],
) -> int:
    source_xml = source_xml.resolve()
    if source_xml in mapping:
        return 0

    path_hash = hashlib.sha1(source_xml.as_posix().encode("utf-8")).hexdigest()[:10]
    patched_xml = WRAP_DIR / f"{source_xml.stem}__{path_hash}__desktex_{variant_tag}{source_xml.suffix}"
    mapping[source_xml] = patched_xml

    tree = ET.parse(source_xml)
    root = tree.getroot()
    matched = 0

    for inc_elem, file_attr in list(_iter_includes(root)):
        include_src = _resolve_include_path(source_xml, file_attr)
        if not include_src.exists():
            continue
        matched += _patch_xml_tree_for_desk_material(
            include_src,
            variant_tag=variant_tag,
            desk_mat_name=desk_mat_name,
            table_regex=table_regex,
            mapping=mapping,
            generated_xmls=generated_xmls,
        )
        include_dst = mapping.get(include_src.resolve(), include_src)
        inc_elem.set("file", _relpath_or_abs(include_dst, patched_xml.parent))

    for geom in root.iter("geom"):
        name = (geom.get("name") or "")
        cls = (geom.get("class") or "")
        mat = (geom.get("material") or "")
        if table_regex.search(name) or table_regex.search(cls) or table_regex.search(mat) or _geom_looks_like_table(geom):
            geom.set("material", desk_mat_name)
            matched += 1

    tree.write(patched_xml, encoding="utf-8", xml_declaration=True)
    generated_xmls.append(patched_xml)
    return matched


def _build_textured_wrapper_variant(
    base_wrapper_xml: Path,
    chosen_texture: Path,
    variant_tag: str,
    desk_geom_regex: str,
    desk_texrepeat: tuple[int, int],
) -> DeskTexturePatchResult:
    base_wrapper_xml = base_wrapper_xml.resolve()
    chosen_texture = chosen_texture.resolve()

    tex_dir = WRAP_DIR / "_desk_textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    copied_texture = tex_dir / f"{variant_tag}__{chosen_texture.name}"
    shutil.copy2(chosen_texture, copied_texture)

    desk_tex_name = f"desktex_{variant_tag}"
    desk_mat_name = f"deskmat_{variant_tag}"
    table_regex = re.compile(desk_geom_regex, re.IGNORECASE)

    mapping: dict[Path, Path] = {}
    generated_xmls: list[Path] = []
    matched_geoms = _patch_xml_tree_for_desk_material(
        base_wrapper_xml,
        variant_tag=variant_tag,
        desk_mat_name=desk_mat_name,
        table_regex=table_regex,
        mapping=mapping,
        generated_xmls=generated_xmls,
    )

    wrapper_copy = mapping.get(base_wrapper_xml.resolve(), base_wrapper_xml.resolve())
    tree = ET.parse(wrapper_copy)
    root = tree.getroot()
    asset = _ensure_asset_first(root)

    tex_file_attr = _relpath_or_abs(copied_texture, wrapper_copy.parent)

    tex_el = None
    for el in asset.findall("texture"):
        if el.get("name") == desk_tex_name:
            tex_el = el
            break
    if tex_el is None:
        tex_el = ET.SubElement(asset, "texture", {"name": desk_tex_name, "type": "2d"})
    tex_el.set("file", tex_file_attr)

    mat_el = None
    for el in asset.findall("material"):
        if el.get("name") == desk_mat_name:
            mat_el = el
            break
    if mat_el is None:
        mat_el = ET.SubElement(asset, "material", {"name": desk_mat_name})
    mat_el.set("texture", desk_tex_name)
    mat_el.set("texrepeat", f"{int(desk_texrepeat[0])} {int(desk_texrepeat[1])}")
    mat_el.set("texuniform", "false")

    tree.write(wrapper_copy, encoding="utf-8", xml_declaration=True)

    return DeskTexturePatchResult(
        wrapper_xml=wrapper_copy,
        generated_xmls=generated_xmls,
        generated_files=[copied_texture],
        chosen_texture=chosen_texture,
        matched_geoms=matched_geoms,
    )


def _import_wrapper_builder():
    # Lazy import to avoid importing cdpr_mujoco at module import time.
    from .generate_cdpr_dataset import build_wrapper_if_needed, list_wrapper_bundle_paths

    return WrapperBuilderHandle(
        build_wrapper_if_needed=build_wrapper_if_needed,
        list_wrapper_bundle_paths=list_wrapper_bundle_paths,
    )


def _load_json_env(name: str) -> dict[str, Any]:
    raw = os.environ.get(name)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Environment variable {name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Environment variable {name} must contain a JSON object.")
    return dict(payload)


def _load_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} is not a valid boolean: {raw!r}")


def _load_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} is not a valid float: {raw!r}") from exc


def _normalize_float_pair(values: Sequence[float], *, name: str) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != 2:
        raise ValueError(f"{name} must contain exactly two floats, got {values!r}")
    low = float(arr[0])
    high = float(arr[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{name} must contain finite floats, got {values!r}")
    if low <= high:
        return low, high
    return high, low


def _load_float_pair_env(name: str, default: Sequence[float]) -> tuple[float, float]:
    raw = os.environ.get(name)
    if raw is None:
        return _normalize_float_pair(default, name=name)

    payload: Sequence[Any]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [chunk for chunk in re.split(r"[\s,]+", str(raw).strip()) if chunk]
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        raise ValueError(
            f"Environment variable {name} must contain two floats as JSON or comma-separated text, got {raw!r}"
        )
    payload = parsed
    return _normalize_float_pair((float(payload[0]), float(payload[1])), name=name)


def _coerce_ee_start(values: Sequence[float]) -> np.ndarray:
    ee_start = np.asarray(values, dtype=float).reshape(3)
    if not np.all(np.isfinite(ee_start)):
        raise ValueError(f"ee_start must contain finite floats, got {values!r}")
    ee_start[2] = max(float(ee_start[2]), MIN_EE_START_Z)
    return ee_start


def _prepend_python_paths(paths_raw: str | None) -> None:
    if not paths_raw:
        return
    for part in reversed([chunk for chunk in paths_raw.split(os.pathsep) if chunk]):
        if part not in sys.path:
            sys.path.insert(0, part)


def _load_callable_from_env(prefix: str):
    attribute = os.environ.get(f"{prefix}_ATTRIBUTE")
    if not attribute:
        return None

    _prepend_python_paths(os.environ.get(f"{prefix}_PYTHONPATHS"))

    module_name = os.environ.get(f"{prefix}_MODULE")
    file_path = os.environ.get(f"{prefix}_FILE")
    if module_name:
        module = importlib.import_module(module_name)
    elif file_path:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{prefix}_FILE does not exist: {path}")
        unique_name = f"_rlvla_hook_{path.stem}_{hashlib.sha1(path.as_posix().encode('utf-8')).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(unique_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for task hook: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
    else:
        raise ValueError(f"{prefix}_ATTRIBUTE requires either {prefix}_MODULE or {prefix}_FILE.")

    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(f"Task hook `{attribute}` not found for prefix {prefix}.") from exc


def _call_with_supported_kwargs(func, **kwargs):
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return func(**kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in params}
    return func(**filtered)


def _normalize_reward_result(result: Any) -> tuple[float, bool, dict[str, Any]]:
    if isinstance(result, dict):
        reward = float(result.get("reward", 0.0))
        success = bool(result.get("success", False))
        info = dict(result.get("info") or {})
        for key, value in result.items():
            if key not in {"reward", "success", "info"}:
                info[key] = value
        return reward, success, info

    if isinstance(result, (tuple, list)):
        if len(result) == 3:
            reward, success, info = result
            return float(reward), bool(success), dict(info or {})
        if len(result) == 2:
            reward, success = result
            return float(reward), bool(success), {}
        if len(result) == 1:
            return float(result[0]), False, {}
        raise ValueError("Reward hook must return reward, (reward, success), or (reward, success, info).")

    return float(result), False, {}


def _normalize_success_result(result: Any, current_success: bool) -> tuple[bool, dict[str, Any]]:
    if result is None:
        return bool(current_success), {}
    if isinstance(result, dict):
        success = bool(result.get("success", current_success))
        info = {key: value for key, value in result.items() if key != "success"}
        return success, info
    if isinstance(result, (tuple, list)):
        if len(result) == 2:
            success, info = result
            return bool(success), dict(info or {})
        if len(result) == 1:
            return bool(result[0]), {}
        raise ValueError("Success hook must return success or (success, info).")
    return bool(result), {}


class _EnvBase:
    pass


if gym is not None:
    _EnvBase = gym.Env


class CDPRLanguageRLEnv(_EnvBase):
    """
    Language-conditioned RL environment over the CDPR MuJoCo simulation.

    Action space:
      - Box(5): [dx, dy, dz, dyaw, gripper_cmd], each in [-1, 1]
      - dx/dy/dz are delta end-effector commands scaled by action_step_xyz.
      - dyaw is scaled by action_step_yaw.
      - gripper_cmd > +0.2 closes gripper; < -0.2 opens gripper.

    Observation space:
      - ee_position: (3,)
      - target_object_position: (3,) waypoint goal position
      - all_object_positions: (max_objects, 3)
      - object_position_mask: (max_objects,)
      - instruction_onehot: (7,)
      - goal_direction: (3,) motion direction for the current waypoint goal.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        catalog_path: Path | str | None = None,
        max_steps: int = 150,
        max_objects: int = 8,
        action_step_xyz: float = 0.02,
        action_step_yaw: float = 0.25,
        hold_steps: int = 0,
        lock_non_commanded_axes: bool | None = None,
        lock_non_commanded_axes_threshold: float | None = None,
        randomize_ee_start: bool | None = None,
        ee_start_x_bounds: Sequence[float] | None = None,
        ee_start_y_bounds: Sequence[float] | None = None,
        ee_start_z: float | None = None,
        move_distance: float = 0.40,
        lift_distance: float = 0.10,
        capture_frames: bool = False,
        instruction_types: Optional[Sequence[str]] = None,
        allowed_objects: Optional[Sequence[str]] = DEFAULT_ALLOWED_OBJECTS,
        desk_textures_dir: Path | str | None = None,
        desk_geom_regex: str = DEFAULT_DESK_GEOM_REGEX,
        desk_texrepeat: Sequence[int] = (20, 20),
        wrapper_cleanup: bool = True,
        use_wrapper_cache: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if gym is None or spaces is None:
            raise ImportError(
                "CDPRLanguageRLEnv requires gym (tested with gym==0.26.2). "
                "Install it before creating this env."
            )
        if mj is None:
            raise ImportError("CDPRLanguageRLEnv requires mujoco. Install it before creating this env.")

        try:
            from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation
        except Exception as exc:  # pragma: no cover - runtime dependency check
            raise ImportError(
                "CDPRLanguageRLEnv requires cdpr_mujoco. Install it before creating this env."
            ) from exc

        self._sim_cls = HeadlessCDPRSimulation
        self.sim = None

        self.catalog_path = Path(catalog_path) if catalog_path is not None else DEFAULT_CATALOG_PATH
        self.defaults, scenes = _load_catalog(self.catalog_path)
        self.allowed_objects = tuple(str(x) for x in (allowed_objects or ()))
        self.scenes = _filter_scenes_to_allowed_objects(scenes, self.allowed_objects)

        self.max_steps = int(max_steps)
        self.max_objects = int(max_objects)
        self.action_step_xyz = float(action_step_xyz)
        self.action_step_yaw = float(action_step_yaw)
        self.hold_steps = max(0, int(hold_steps))
        if lock_non_commanded_axes is None:
            lock_non_commanded_axes = _load_bool_env(CDPR_LOCK_NON_COMMANDED_AXES_ENV, default=False)
        if lock_non_commanded_axes_threshold is None:
            lock_non_commanded_axes_threshold = _load_float_env(
                CDPR_LOCK_NON_COMMANDED_AXES_THRESHOLD_ENV,
                default=0.05,
            )
        if randomize_ee_start is None:
            randomize_ee_start = _load_bool_env(CDPR_RANDOMIZE_EE_START_ENV, default=False)
        if ee_start_x_bounds is None:
            ee_start_x_bounds = _load_float_pair_env(
                CDPR_EE_START_X_BOUNDS_ENV,
                default=DEFAULT_RANDOM_EE_START_X_BOUNDS,
            )
        if ee_start_y_bounds is None:
            ee_start_y_bounds = _load_float_pair_env(
                CDPR_EE_START_Y_BOUNDS_ENV,
                default=DEFAULT_RANDOM_EE_START_Y_BOUNDS,
            )
        if ee_start_z is None:
            loaded_ee_start_z = _load_float_env(CDPR_EE_START_Z_ENV, default=float("nan"))
            ee_start_z = None if not np.isfinite(loaded_ee_start_z) else float(loaded_ee_start_z)
        self.lock_non_commanded_axes = bool(lock_non_commanded_axes)
        self.lock_non_commanded_axes_threshold = max(0.0, float(lock_non_commanded_axes_threshold))
        self.randomize_ee_start = bool(randomize_ee_start)
        self.ee_start_x_bounds = _normalize_float_pair(ee_start_x_bounds, name="ee_start_x_bounds")
        self.ee_start_y_bounds = _normalize_float_pair(ee_start_y_bounds, name="ee_start_y_bounds")
        self.ee_start_z = None if ee_start_z is None else max(float(ee_start_z), MIN_EE_START_Z)
        self.move_distance = float(move_distance)
        self.lift_distance = float(lift_distance)
        self.capture_frames = bool(capture_frames)
        self.instruction_types = tuple(instruction_types) if instruction_types else None
        self.wrapper_cleanup = bool(wrapper_cleanup)
        self.use_wrapper_cache = bool(use_wrapper_cache)
        self.desk_geom_regex = str(desk_geom_regex)
        texrepeat_vals = tuple(desk_texrepeat)
        if len(texrepeat_vals) != 2:
            raise ValueError("desk_texrepeat must contain exactly two integers: X Y.")
        self.desk_texrepeat = (int(texrepeat_vals[0]), int(texrepeat_vals[1]))

        self.np_random = np.random.default_rng(seed)
        WRAP_DIR.mkdir(parents=True, exist_ok=True)
        self.wrapper_dir = WRAP_DIR

        self.desk_texture_files: list[Path] = []
        if desk_textures_dir is not None:
            tex_dir = Path(desk_textures_dir).expanduser().resolve()
            if not tex_dir.exists():
                raise ValueError(f"desk_textures_dir not found: {tex_dir}")
            self.desk_texture_files = sorted(
                p for p in tex_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
            if not self.desk_texture_files:
                raise ValueError(f"No texture files found in: {tex_dir}")

        self._goal_region = _load_json_env("RLVLA_TASK_GOAL_REGION_JSON")
        self._dense_reward_terms = _load_json_env("RLVLA_TASK_DENSE_REWARD_TERMS_JSON")
        self._task_metadata = _load_json_env("RLVLA_TASK_METADATA_JSON")
        self._goal_relation = os.environ.get("RLVLA_TASK_GOAL_RELATION")
        self._reward_fn = _load_callable_from_env(TASK_REWARD_PREFIX) or compute_instruction_reward
        self._success_fn = _load_callable_from_env(TASK_SUCCESS_PREFIX)
        (
            self.scenes,
            self.allowed_objects,
            self.target_object_pool,
            self.distractor_object_pool,
        ) = _configure_scene_sampling(
            base_scenes=self.scenes,
            allowed_objects=self.allowed_objects,
            task_metadata=self._task_metadata,
            seed=seed,
        )
        self.scene_object_pool = _metadata_name_list(self._task_metadata, "scene_object_pool")
        if not self.scene_object_pool:
            self.scene_object_pool = tuple(self.allowed_objects)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "ee_position": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float32),
                "target_object_position": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float32),
                "all_object_positions": spaces.Box(
                    -2.0, 2.0, shape=(self.max_objects, 3), dtype=np.float32
                ),
                "object_position_mask": spaces.Box(
                    0.0, 1.0, shape=(self.max_objects,), dtype=np.float32
                ),
                "instruction_onehot": spaces.Box(
                    0.0, 1.0, shape=(len(INSTRUCTION_TYPES),), dtype=np.float32
                ),
                "goal_direction": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            }
        )

        self._step_count = 0
        self._yaw = 0.0
        self._last_gripper_cmd = 0.0
        self._instruction_spec = None
        self._reward_state = None
        self._scene_name = ""
        self._target_catalog_name = ""
        self._target_body_name = ""
        self._catalog_to_body: dict[str, str] = {}
        self._object_body_names: list[str] = []
        self._scene_catalog_objects: list[str] = []
        self._cleanup_paths: list[Path] = []
        self._cleanup_path_set: set[Path] = set()
        self._desk_texture_name = ""
        self._current_wrapper_xml: Path | None = None
        self._inverse_catalog_to_body: dict[str, str] = {}
        self._prev_object_positions: dict[str, np.ndarray] = {}
        self._prev_ee_for_catch = np.zeros((3,), dtype=np.float32)
        self._last_caught_body = ""
        self._last_caught_catalog = ""
        self._support_surface_z = 0.0
        self._ee_min_z = float("-inf")
        self._ee_spawn_z = float("-inf")
        self._locked_target_xyz = np.zeros((3,), dtype=np.float32)
        self._episode_ee_start = self._default_ee_start().astype(np.float32)
        self._goal_position = np.zeros((3,), dtype=np.float32)
        self._goal_motion_direction = np.zeros((3,), dtype=np.float32)
        self._episode_index = -1
        self._reset_counter = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        self._prepare_episode_rng(seed)

        self.close()
        scene = self._sample_scene(options=options)
        self._scene_name = scene.name
        self._scene_catalog_objects = list(scene.objects)

        episode_ee_start = self._sample_episode_ee_start(options=options)
        self._episode_ee_start = episode_ee_start.astype(np.float32)
        wrapper_xml = self._build_episode_wrapper(scene=scene, ee_start=episode_ee_start)
        self._current_wrapper_xml = wrapper_xml
        self.sim = self._sim_cls(xml_path=str(wrapper_xml), output_dir=str(DEFAULT_VIDEO_DIR))
        self.sim.initialize()
        if hasattr(self.sim, "hold_current_pose"):
            self.sim.hold_current_pose(warm_steps=10)
        self._refresh_workspace_safety()
        self._move_ee_to_episode_start()
        self._clear_sim_recording_buffers()

        self._catalog_to_body, self._object_body_names = self._resolve_objects(scene.objects)
        self._inverse_catalog_to_body = {v: k for k, v in self._catalog_to_body.items()}
        if self._object_body_names:
            try:
                place_objects_non_overlapping(
                    self.sim,
                    self._object_body_names,
                    xy_bounds=((-0.20, 0.20), (-0.20, 0.20), self._support_surface_z),
                    min_gap=0.02,
                    min_ee_dist=0.10,
                )
            except Exception:
                # Continue if placement fails; wrapper-provided placement is still valid.
                pass

        self._target_catalog_name = ""
        self._target_body_name = ""

        self._instruction_spec = sample_instruction(
            target_object=None,
            rng=self.np_random,
            allowed_instruction_types=self.instruction_types,
            move_distance=self.move_distance,
            lift_distance=self.lift_distance,
        )
        setattr(self.sim, "language_instruction", self._instruction_spec.text)

        ee0 = self._get_ee_position()
        self._goal_position = self._compute_instruction_goal(
            spec=self._instruction_spec,
            initial_ee_pos=ee0,
            options=options,
        )
        self._goal_motion_direction = self._compute_goal_motion_direction(
            initial_ee_pos=ee0,
            goal_pos=self._goal_position,
            instruction_direction=self._instruction_spec.direction,
        )
        self._reward_state = init_reward_state(ee0, self._goal_position)
        self._step_count = 0
        self._yaw = self._read_current_yaw()
        self._last_gripper_cmd = 0.0
        self._prev_ee_for_catch = ee0.copy()
        self._prev_object_positions = {}
        for body_name in self._object_body_names:
            try:
                self._prev_object_positions[body_name] = self._get_body_position(body_name)
            except Exception:
                continue
        self._last_caught_body = ""
        self._last_caught_catalog = ""
        self._locked_target_xyz = self._get_ee_position().astype(np.float32)

        obs = self._get_obs()
        info = self._base_info()
        info["success"] = False
        return obs, info

    def step(self, action: np.ndarray):
        if self.sim is None:
            raise RuntimeError("Environment was not reset before step().")
        if self._instruction_spec is None or self._reward_state is None:
            raise RuntimeError("Internal state is missing. Call reset().")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 5:
            raise ValueError(f"Expected action shape (5,), got {action.shape}")

        action = np.clip(action, -1.0, 1.0)
        self._apply_action(action)

        ee = self._get_ee_position()
        goal_pos = self._goal_position.astype(np.float32)
        camera_alignment = self._get_ee_camera_alignment(direction=self._goal_motion_direction)
        caught_body = ""
        caught_catalog = ""
        caught_score = 0.0
        caught_is_target = False
        reward_kwargs = {
            "spec": self._instruction_spec,
            "ee_pos": ee,
            "obj_pos": goal_pos,
            "goal_pos": goal_pos,
            "reward_state": self._reward_state,
            "action": action,
            "camera_alignment": camera_alignment,
            "goal_direction": self._goal_motion_direction,
            "goal_region": self._goal_region,
            "goal_relation": self._goal_relation,
            "dense_reward_terms": self._dense_reward_terms,
            "task_metadata": self._task_metadata,
            "env": self,
            "sim": self.sim,
            "scene_name": self._scene_name,
            "target_catalog_name": self._target_catalog_name,
            "target_body_name": self._target_body_name,
            "caught_object_body": caught_body,
            "caught_object_catalog": caught_catalog,
            "caught_object_score": float(caught_score),
            "caught_object_is_target": bool(caught_is_target),
        }
        reward, success, reward_info = _normalize_reward_result(
            _call_with_supported_kwargs(self._reward_fn, **reward_kwargs)
        )
        success_kwargs = {
            **reward_kwargs,
            "reward": float(reward),
            "reward_info": reward_info,
            "current_success": bool(success),
        }
        if self._success_fn is not None:
            success, success_info = _normalize_success_result(
                _call_with_supported_kwargs(self._success_fn, **success_kwargs),
                bool(success),
            )
            reward_info.update(success_info)
        if caught_body:
            self._last_caught_body = caught_body
            self._last_caught_catalog = caught_catalog

        self._step_count += 1
        terminated = bool(success)
        truncated = bool(self._step_count >= self.max_steps and not terminated)

        obs = self._get_obs()
        info = self._base_info()
        info.update(reward_info)
        info["success"] = bool(success)
        info["reward"] = float(reward)
        info["step"] = int(self._step_count)
        info["target_grasped"] = bool(float(reward_info.get("grasped", 0.0)) >= 0.5)
        info["caught_object_body"] = caught_body
        info["caught_object_catalog"] = caught_catalog
        info["caught_object_score"] = float(caught_score)
        info["caught_object_is_target"] = bool(caught_is_target)
        info["last_caught_object_body"] = self._last_caught_body
        info["last_caught_object_catalog"] = self._last_caught_catalog
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self.sim is not None:
            try:
                self.sim.cleanup()
            except Exception:
                pass
        self.sim = None
        self._cleanup_generated_files()
        self._current_wrapper_xml = None
        self._desk_texture_name = ""
        self._prev_object_positions = {}
        self._inverse_catalog_to_body = {}
        self._goal_position = np.zeros((3,), dtype=np.float32)
        self._goal_motion_direction = np.zeros((3,), dtype=np.float32)

    def _sample_scene(self, options: Optional[dict[str, Any]]) -> SceneSpec:
        requested_scene = (options or {}).get("scene")
        if requested_scene is not None:
            requested_scene = str(requested_scene)
            for scene in self.scenes:
                if scene.name == requested_scene:
                    return scene
        idx = int(self.np_random.integers(0, len(self.scenes)))
        return self.scenes[idx]

    def _prepare_episode_rng(self, seed: Optional[int]) -> None:
        episode_index = int(self._reset_counter)
        self._reset_counter += 1
        self._episode_index = episode_index
        if seed is None:
            return
        seed_sequence = np.random.SeedSequence([int(seed), episode_index])
        self.np_random = np.random.default_rng(seed_sequence)

    def _default_ee_start(self) -> np.ndarray:
        ee_start = _coerce_ee_start(self.defaults.get("ee_start", (0.0, 0.0, MIN_EE_START_Z)))
        if self.ee_start_z is not None:
            ee_start[2] = max(float(self.ee_start_z), MIN_EE_START_Z)
        return ee_start

    def _sample_episode_ee_start(self, options: Optional[dict[str, Any]] = None) -> np.ndarray:
        requested = (options or {}).get("ee_start")
        if requested is not None:
            return _coerce_ee_start(requested)

        ee_start = self._default_ee_start()
        if not self.randomize_ee_start:
            return ee_start

        ee_start[0] = float(self.np_random.uniform(*self.ee_start_x_bounds))
        ee_start[1] = float(self.np_random.uniform(*self.ee_start_y_bounds))
        return ee_start

    def _goal_center(self) -> np.ndarray:
        raw_xy = self._task_metadata.get("goal_center_xy", self.defaults.get("goal_center_xy", DEFAULT_GOAL_CENTER_XY))
        xy = np.asarray(raw_xy, dtype=np.float32).reshape(-1)
        if xy.size < 2:
            raise ValueError(f"goal_center_xy must contain at least two values, got {raw_xy!r}")
        height_above_table = float(
            self._task_metadata.get(
                "goal_height_above_table",
                self.defaults.get("goal_height_above_table", DEFAULT_GOAL_HEIGHT_ABOVE_TABLE),
            )
        )
        center = np.array([xy[0], xy[1], self._support_surface_z + height_above_table], dtype=np.float32)
        center = np.asarray(clamp_xyz(center), dtype=np.float32)
        if np.isfinite(self._ee_min_z):
            center[2] = max(float(center[2]), float(self._ee_min_z))
        return center

    def _compute_instruction_goal(
        self,
        *,
        spec,
        initial_ee_pos: np.ndarray,
        options: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        requested_goal = (options or {}).get("goal_position")
        if requested_goal is None:
            requested_goal = (options or {}).get("target_position")
        if requested_goal is not None:
            goal = np.asarray(clamp_xyz(requested_goal), dtype=np.float32)
        else:
            center = self._goal_center()
            lateral_offset = float(self._task_metadata.get("lateral_goal_offset", spec.target_displacement))
            vertical_offset = float(self._task_metadata.get("vertical_goal_offset", spec.lift_target))
            goal = center.copy()
            if spec.instruction_type == "move_left":
                goal[0] -= lateral_offset
            elif spec.instruction_type == "move_right":
                goal[0] += lateral_offset
            elif spec.instruction_type == "move_top":
                goal[1] += lateral_offset
            elif spec.instruction_type == "move_bottom":
                goal[1] -= lateral_offset
            elif spec.instruction_type == "move_up":
                goal[0] = float(initial_ee_pos[0])
                goal[1] = float(initial_ee_pos[1])
                goal[2] = float(initial_ee_pos[2] + vertical_offset)
            elif spec.instruction_type == "move_down":
                goal[0] = float(initial_ee_pos[0])
                goal[1] = float(initial_ee_pos[1])
                goal[2] = float(initial_ee_pos[2] - vertical_offset)
            elif spec.instruction_type != "move_center":
                raise RuntimeError(f"Unsupported instruction type for goal generation: {spec.instruction_type}")
            goal = np.asarray(clamp_xyz(goal), dtype=np.float32)

        min_goal_height = float(self._task_metadata.get("min_goal_height_above_table", 0.02))
        goal[2] = max(float(goal[2]), float(self._support_surface_z + min_goal_height))
        if np.isfinite(self._ee_min_z):
            goal[2] = max(float(goal[2]), float(self._ee_min_z))
        return goal.astype(np.float32)

    def _compute_goal_motion_direction(
        self,
        *,
        initial_ee_pos: np.ndarray,
        goal_pos: np.ndarray,
        instruction_direction: np.ndarray,
    ) -> np.ndarray:
        goal_delta = np.asarray(goal_pos - initial_ee_pos, dtype=np.float32)
        goal_norm = float(np.linalg.norm(goal_delta))
        if goal_norm > 1e-8:
            return (goal_delta / goal_norm).astype(np.float32)

        fallback = np.asarray(instruction_direction, dtype=np.float32).reshape(-1)
        if fallback.size < 3:
            padded = np.zeros((3,), dtype=np.float32)
            padded[: fallback.size] = fallback
            fallback = padded
        else:
            fallback = fallback[:3]
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm > 1e-8:
            return (fallback / fallback_norm).astype(np.float32)
        return np.zeros((3,), dtype=np.float32)

    def _build_episode_wrapper(
        self,
        *,
        scene: SceneSpec,
        ee_start: Sequence[float] | np.ndarray,
    ) -> Path:
        build_wrapper = self._build_wrapper
        supports_ee_start = True
        try:
            signature = inspect.signature(build_wrapper)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            params = signature.parameters
            supports_ee_start = (
                "ee_start" in params
                or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
            )

        if supports_ee_start:
            try:
                return Path(build_wrapper(scene=scene, ee_start=ee_start)).resolve()
            except TypeError as exc:
                if "ee_start" not in str(exc):
                    raise

        return Path(build_wrapper(scene=scene)).resolve()

    def _build_wrapper(self, scene: SceneSpec, *, ee_start: Sequence[float] | np.ndarray | None = None) -> Path:
        build_wrapper_if_needed, list_wrapper_bundle_paths = _import_wrapper_builder()
        default_ee_start = self._default_ee_start()
        episode_ee_start = default_ee_start if ee_start is None else _coerce_ee_start(ee_start)
        unique_wrapper_bundle = bool(
            self.randomize_ee_start or not np.allclose(episode_ee_start, default_ee_start, atol=1e-9)
        )
        wrapper_out = None
        use_cache = bool(self.use_wrapper_cache and not unique_wrapper_bundle)
        if (not use_cache) or self.wrapper_cleanup:
            wrapper_out = self._temporary_wrapper_path(scene=scene)
            use_cache = False

        wrapper_xml = build_wrapper_if_needed(
            scene_name=scene.name,
            object_names=list(scene.objects),
            scene_z=self.defaults.get("scene_z", -0.85),
            ee_start=tuple(float(x) for x in episode_ee_start),
            table_z=self.defaults.get("table_z", 0.15),
            settle_time=self.defaults.get("settle_time", 0.0),
            wrapper_out=wrapper_out,
            use_cache=use_cache,
        )
        if self.wrapper_cleanup or unique_wrapper_bundle:
            for path in list_wrapper_bundle_paths(wrapper_xml):
                self._register_cleanup_path(path)

        self._desk_texture_name = ""
        if self.desk_texture_files:
            tex_idx = int(self.np_random.integers(0, len(self.desk_texture_files)))
            chosen_texture = self.desk_texture_files[tex_idx]
            variant_tag = f"rl_{int(time.time_ns())}"
            patched = _build_textured_wrapper_variant(
                base_wrapper_xml=wrapper_xml,
                chosen_texture=chosen_texture,
                variant_tag=variant_tag,
                desk_geom_regex=self.desk_geom_regex,
                desk_texrepeat=self.desk_texrepeat,
            )
            wrapper_xml = patched.wrapper_xml
            self._desk_texture_name = chosen_texture.name

            if self.wrapper_cleanup or unique_wrapper_bundle:
                for path in patched.generated_xmls:
                    self._register_cleanup_path(path)
                for path in patched.generated_files:
                    self._register_cleanup_path(path)

        return wrapper_xml

    def _refresh_workspace_safety(self):
        safety = compute_cdpr_workspace_safety(self.sim, fallback_z=0.0)
        self._support_surface_z = float(safety["support_surface_z"])
        self._ee_min_z = float(safety["ee_min_z"])
        self._ee_spawn_z = float(safety["ee_spawn_z"])

    def _move_ee_to_spawn_height(self):
        if self.sim is None:
            return
        lift_cdpr_ee_to_spawn_height(
            self.sim,
            ee_spawn_z=float(self._ee_spawn_z),
            max_steps=120,
            tol=0.01,
            warm_steps=6,
        )

    def _move_ee_to_episode_start(self):
        if self.sim is None:
            return

        target = np.asarray(self._episode_ee_start, dtype=np.float32).reshape(3).copy()
        if np.isfinite(self._ee_spawn_z):
            target[2] = max(float(target[2]), float(self._ee_spawn_z))
        if np.isfinite(self._ee_min_z):
            target[2] = max(float(target[2]), float(self._ee_min_z))
        target = np.asarray(clamp_xyz(target), dtype=np.float32)
        self._set_ee_target(target)

        moved_with_goto = False
        if hasattr(self.sim, "goto"):
            try:
                self.sim.goto(target, max_steps=120, tol=0.01)
                moved_with_goto = True
            except Exception:
                moved_with_goto = False

        if not moved_with_goto and hasattr(self.sim, "run_simulation_step"):
            for _ in range(8):
                self.sim.run_simulation_step(capture_frame=False)

        if hasattr(self.sim, "hold_current_pose"):
            try:
                self.sim.hold_current_pose(warm_steps=6)
            except Exception:
                pass
        self._locked_target_xyz = target.astype(np.float32)

    def _temporary_wrapper_path(self, scene: SceneSpec) -> Path:
        obj_part = "-".join(sorted(scene.objects))
        stamp = int(time.time_ns())
        return self.wrapper_dir / f"{scene.name}__{obj_part}__rltmp_{stamp}.xml"

    def _clear_sim_recording_buffers(self):
        if self.sim is None:
            return
        # Reset any simulator-side logs so reset-time warmup motion never appears in saved episodes.
        clear_sim_recording_buffers(self.sim)

    def _register_cleanup_path(self, path: Path):
        p = Path(path).resolve()
        if p in self._cleanup_path_set:
            return
        self._cleanup_path_set.add(p)
        self._cleanup_paths.append(p)

    def _cleanup_generated_files(self):
        if not self.wrapper_cleanup:
            self._cleanup_paths.clear()
            self._cleanup_path_set.clear()
            return
        for p in reversed(self._cleanup_paths):
            try:
                if p.exists() and p.is_file():
                    p.unlink()
            except Exception:
                pass
        self._cleanup_paths.clear()
        self._cleanup_path_set.clear()

    def _detect_caught_object(self, ee_pos: np.ndarray) -> tuple[str, str, float, bool]:
        if not self._object_body_names or self._reward_state is None:
            self._prev_ee_for_catch = np.asarray(ee_pos, dtype=np.float32).copy()
            return "", "", 0.0, False

        gripper_closed = bool(self._reward_state.gripper_closed)
        ee_now = np.asarray(ee_pos, dtype=np.float32)
        ee_step = ee_now - self._prev_ee_for_catch

        best_body = ""
        best_score = 0.0
        best_dist = 1e9
        for body_name in self._object_body_names:
            try:
                obj_now = self._get_body_position(body_name)
            except Exception:
                continue

            obj_prev = self._prev_object_positions.get(body_name, obj_now)
            obj_step = obj_now - obj_prev
            dist = float(np.linalg.norm(ee_now - obj_now))
            contact_score = float(np.exp(-28.0 * dist))
            follow_score = float(np.exp(-35.0 * np.linalg.norm(obj_step - ee_step)))
            score = contact_score * follow_score

            self._prev_object_positions[body_name] = obj_now

            if score > best_score:
                best_score = score
                best_body = body_name
                best_dist = dist

        self._prev_ee_for_catch = ee_now.copy()
        is_caught = bool(gripper_closed and best_score >= 0.30 and best_dist <= 0.09)
        if not is_caught:
            return "", "", float(best_score), False

        catalog_name = self._inverse_catalog_to_body.get(best_body, best_body)
        return (
            best_body,
            catalog_name,
            float(best_score),
            bool(catalog_name == self._target_catalog_name),
        )

    def _resolve_objects(self, catalog_objects: Sequence[str]) -> tuple[dict[str, str], list[str]]:
        mapping: dict[str, str] = {}
        for name in catalog_objects:
            try:
                mapping[name] = resolve_body_name(self.sim, name)
            except Exception:
                continue

        unique_bodies = []
        for body in mapping.values():
            if body not in unique_bodies:
                unique_bodies.append(body)

        if not unique_bodies:
            unique_bodies = self._discover_dynamic_bodies()
            if len(unique_bodies) == len(catalog_objects):
                for cat, body in zip(catalog_objects, unique_bodies):
                    mapping[cat] = body

        return mapping, unique_bodies

    def _discover_dynamic_bodies(self) -> list[str]:
        model = self.sim.model
        bodies: list[str] = []
        for bid in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
            if not name:
                continue
            if any(name.startswith(pfx) for pfx in ROBOT_BODY_PREFIXES):
                continue

            jnum = int(model.body_jntnum[bid])
            jadr = int(model.body_jntadr[bid])
            has_free = False
            for i in range(jnum):
                jid = jadr + i
                if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
                    has_free = True
                    break
            if has_free:
                bodies.append(name)
        return bodies

    def _read_current_yaw(self) -> float:
        if hasattr(self.sim, "get_yaw"):
            try:
                return float(self.sim.get_yaw())
            except Exception:
                return 0.0
        return 0.0

    def _set_ee_target(self, xyz: np.ndarray):
        if hasattr(self.sim, "set_end_effector_target"):
            self.sim.set_end_effector_target(xyz)
        elif hasattr(self.sim, "set_ee_target"):
            self.sim.set_ee_target(xyz)
        elif hasattr(self.sim, "set_target_position"):
            self.sim.set_target_position(xyz)
        else:  # pragma: no cover - depends on runtime simulator API
            raise RuntimeError("Simulator has no end-effector target setter.")

    def _get_ee_position(self) -> np.ndarray:
        if hasattr(self.sim, "get_end_effector_position"):
            return np.asarray(self.sim.get_end_effector_position(), dtype=np.float32)
        raise RuntimeError("Simulator has no get_end_effector_position method.")

    def _get_body_position(self, body_name: str) -> np.ndarray:
        bid = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_BODY, body_name)
        if bid == -1:
            raise RuntimeError(f"Body '{body_name}' not found in MuJoCo model.")
        return np.asarray(self.sim.data.body_xpos[bid], dtype=np.float32).copy()

    def _get_geom_position(self, geom_name: str) -> Optional[np.ndarray]:
        gid = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_GEOM, geom_name)
        if gid == -1:
            return None
        return np.asarray(self.sim.data.geom_xpos[gid], dtype=np.float32).copy()

    def _get_gripper_surface_alignment(self, obj_pos: np.ndarray) -> Optional[float]:
        """
        Alignment for the requirement:
        stick surface ⟂ line(to object)  => line aligns with stick-surface normal.
        We approximate the normal by the left-right finger separation axis.
        """
        left = self._get_geom_position("finger_l_link")
        right = self._get_geom_position("finger_r_link")
        if left is None or right is None:
            # Fallback to tips if link geoms are unavailable.
            left = self._get_geom_position("finger_l_tip")
            right = self._get_geom_position("finger_r_tip")
        if left is None or right is None:
            return None

        surface_normal_xy = right[:2] - left[:2]
        norm_surface = float(np.linalg.norm(surface_normal_xy))
        if norm_surface < 1e-8:
            return None
        surface_normal_xy /= norm_surface

        gripper_center_xy = 0.5 * (right[:2] + left[:2])
        to_obj_xy = np.asarray(obj_pos[:2] - gripper_center_xy, dtype=np.float32)
        norm_obj = float(np.linalg.norm(to_obj_xy))
        if norm_obj < 1e-8:
            return 1.0
        to_obj_xy /= norm_obj

        # Absolute because either finger can face the target.
        return float(np.clip(abs(np.dot(surface_normal_xy, to_obj_xy)), 0.0, 1.0))

    def _get_ee_camera_alignment(
        self,
        target_pos: Optional[np.ndarray] = None,
        *,
        direction: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        cam_id = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
        if cam_id == -1:
            return None

        cam_pos = np.asarray(self.sim.data.cam_xpos[cam_id], dtype=np.float32)
        cam_xmat = np.asarray(self.sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
        # MuJoCo fixed camera forward direction is local -Z in world frame.
        cam_forward = -cam_xmat[:, 2]
        norm_forward = float(np.linalg.norm(cam_forward))
        if norm_forward < 1e-8:
            return None
        cam_forward /= norm_forward

        if direction is not None:
            desired = np.asarray(direction, dtype=np.float32).reshape(-1)
            if desired.size < 3:
                padded = np.zeros((3,), dtype=np.float32)
                padded[: desired.size] = desired
                desired = padded
            else:
                desired = desired[:3]
        elif target_pos is not None:
            desired = np.asarray(target_pos - cam_pos, dtype=np.float32)
        else:
            return None

        norm_desired = float(np.linalg.norm(desired))
        if norm_desired < 1e-8:
            return 1.0
        desired /= norm_desired

        return float(np.clip(np.dot(cam_forward, desired), 0.0, 1.0))

    def _apply_action(self, action: np.ndarray):
        ee = self._get_ee_position()
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        dxyz = action_arr[:3] * self.action_step_xyz

        if self.lock_non_commanded_axes:
            reference = np.asarray(self._locked_target_xyz, dtype=np.float32).reshape(3).copy()
            if not np.all(np.isfinite(reference)):
                reference = ee.astype(np.float32)
            active_axes = np.abs(action_arr[:3]) > float(self.lock_non_commanded_axes_threshold)
            target = reference.copy()
            target[active_axes] = target[active_axes] + dxyz[active_axes]
        else:
            target = ee + dxyz

        target = clamp_xyz(target)
        if np.isfinite(self._ee_min_z):
            target[2] = max(float(target[2]), float(self._ee_min_z))
        target = target.astype(np.float32)
        self._set_ee_target(target)
        self._locked_target_xyz = target.copy()

        if hasattr(self.sim, "set_yaw"):
            self._yaw = self._yaw + float(action_arr[3]) * self.action_step_yaw
            try:
                self.sim.set_yaw(self._yaw)
            except Exception:
                pass

        self._last_gripper_cmd = float(action_arr[4])
        if self._last_gripper_cmd >= 0.2 and hasattr(self.sim, "close_gripper"):
            self.sim.close_gripper()
        elif self._last_gripper_cmd <= -0.2 and hasattr(self.sim, "open_gripper"):
            self.sim.open_gripper()

        total_sim_steps = 1 + int(self.hold_steps)
        for sub_idx in range(total_sim_steps):
            capture = bool(self.capture_frames and sub_idx == (total_sim_steps - 1))
            self.sim.run_simulation_step(capture_frame=capture)

    def _get_obs(self) -> dict[str, np.ndarray]:
        ee_pos = self._get_ee_position()
        target_pos = self._goal_position.astype(np.float32)

        obj_pos = np.zeros((self.max_objects, 3), dtype=np.float32)
        obj_mask = np.zeros((self.max_objects,), dtype=np.float32)
        for i, body_name in enumerate(self._object_body_names[: self.max_objects]):
            try:
                obj_pos[i] = self._get_body_position(body_name)
                obj_mask[i] = 1.0
            except Exception:
                continue

        onehot = instruction_to_onehot(self._instruction_spec)
        goal_direction = self._goal_motion_direction.astype(np.float32)

        obs = {
            "ee_position": ee_pos.astype(np.float32),
            "target_object_position": target_pos.astype(np.float32),
            "all_object_positions": obj_pos,
            "object_position_mask": obj_mask,
            "instruction_onehot": onehot,
            "goal_direction": goal_direction,
        }
        return obs

    def _base_info(self) -> dict[str, Any]:
        return {
            "scene": self._scene_name,
            "episode_index": int(self._episode_index),
            "scene_objects": list(self._scene_catalog_objects),
            "allowed_objects": list(self.allowed_objects),
            "scene_object_pool": list(self.scene_object_pool),
            "target_object_pool": list(self.target_object_pool),
            "distractor_object_pool": list(self.distractor_object_pool),
            "target_object_catalog": self._target_catalog_name,
            "target_object_body": self._target_body_name,
            "language_instruction": self._instruction_spec.text,
            "instruction_type": self._instruction_spec.instruction_type,
            "goal_position": [float(x) for x in self._goal_position.tolist()],
            "goal_motion_direction": [float(x) for x in self._goal_motion_direction.tolist()],
            "goal_region": dict(self._goal_region),
            "goal_relation": self._goal_relation or "",
            "dense_reward_terms": dict(self._dense_reward_terms),
            "gripper_command": float(self._last_gripper_cmd),
            "desk_texture": self._desk_texture_name,
            "wrapper_xml": str(self._current_wrapper_xml) if self._current_wrapper_xml else "",
            "ee_start": [float(x) for x in self._episode_ee_start.tolist()],
            "support_surface_z": float(self._support_surface_z),
            "ee_min_z": float(self._ee_min_z) if np.isfinite(self._ee_min_z) else float("nan"),
            "ee_spawn_z": float(self._ee_spawn_z) if np.isfinite(self._ee_spawn_z) else float("nan"),
            "lock_non_commanded_axes": bool(self.lock_non_commanded_axes),
            "lock_non_commanded_axes_threshold": float(self.lock_non_commanded_axes_threshold),
            "randomize_ee_start": bool(self.randomize_ee_start),
            "ee_start_x_bounds": [float(self.ee_start_x_bounds[0]), float(self.ee_start_x_bounds[1])],
            "ee_start_y_bounds": [float(self.ee_start_y_bounds[0]), float(self.ee_start_y_bounds[1])],
            "ee_start_z_override": (
                float(self.ee_start_z) if self.ee_start_z is not None else float("nan")
            ),
        }
