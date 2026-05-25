from __future__ import annotations

import copy
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

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None

from .rl_instruction_tasks import (
    CATCHABLE_TARGET_INSTRUCTION_TYPES,
    DEFAULT_CATCHABLE_OBJECTS,
    DEFAULT_CONTAINER_OBJECTS,
    INSTRUCTION_TYPES,
    canonical_object_name,
    compute_instruction_reward,
    init_reward_state,
    instruction_uses_target_object,
    instruction_to_onehot,
    sample_instruction,
)
from .synthetic_tasks import (
    aabb_of_body,
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
CDPR_RECORD_TRAJECTORY_ENV = "RLVLA_CDPR_RECORD_TRAJECTORY"
CDPR_ACTION_STEP_GRIPPER_ENV = "RLVLA_CDPR_ACTION_STEP_GRIPPER"
DEFAULT_RANDOM_EE_START_X_BOUNDS = (-0.25, 0.25)
DEFAULT_RANDOM_EE_START_Y_BOUNDS = (-0.25, 0.25)
DEFAULT_GOAL_CENTER_XY = (0.0, 0.0)
DEFAULT_GOAL_HEIGHT_ABOVE_TABLE = 0.10
DEFAULT_CAUGHT_OBJECT_START_INSTRUCTION_TYPES: tuple[str, ...] = (
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "put_in_front_of_object",
    "put_behind_object",
    "move_between_objects",
)
DEFAULT_CAUGHT_OBJECT_START_OFFSET = (0.0, 0.0, -0.035)
YCB_CAUGHT_OBJECT_MEASUREMENTS: dict[str, dict[str, float]] = {
    "ycb_apple": {"width_m": 0.0751, "opening": 0.8852, "finger_qpos_m": 0.0266},
    "apple": {"width_m": 0.0751, "opening": 0.8852, "finger_qpos_m": 0.0266},
    "ycb_pear": {"width_m": 0.0662, "opening": 0.7369, "finger_qpos_m": 0.0221},
    "pear": {"width_m": 0.0662, "opening": 0.7369, "finger_qpos_m": 0.0221},
    "ycb_peach": {"width_m": 0.0591, "opening": 0.6186, "finger_qpos_m": 0.0186},
    "peach": {"width_m": 0.0591, "opening": 0.6186, "finger_qpos_m": 0.0186},
    "ycb_baseball": {"width_m": 0.0720, "opening": 0.8337, "finger_qpos_m": 0.0250},
    "baseball": {"width_m": 0.0720, "opening": 0.8337, "finger_qpos_m": 0.0250},
}
_TEXTURE_VALIDATION_CACHE: dict[Path, tuple[tuple[tuple[str, int, int], ...], list[Path], list[Path]]] = {}
_TEXTURE_VALIDATION_WARNED: set[Path] = set()

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


def _infer_instruction_type_from_text(instruction: str) -> str | None:
    text = re.sub(r"\s+", " ", str(instruction).strip().lower())
    if not text:
        return None
    if text.startswith("move to ") or text.startswith("go to "):
        return "move_to_object"
    if text.startswith("grab "):
        return "grab_object"
    if text.startswith("pick up "):
        return "pick_up"
    if text.startswith("push ") and text.endswith(" left"):
        return "push_left"
    if text.startswith("push ") and text.endswith(" right"):
        return "push_right"
    if text.startswith("put ") and (" plate" in text or " bowl" in text or " into " in text or " on " in text):
        return "put_into_plate"
    if text.startswith("move ") and " to the left of " in text:
        return "move_left_of_object"
    if text.startswith("move ") and " to the right of " in text:
        return "move_right_of_object"
    if text.startswith("put ") and " in front of " in text:
        return "put_in_front_of_object"
    if text.startswith("put ") and " behind " in text:
        return "put_behind_object"
    if text.startswith("move ") and " between " in text and " and " in text:
        return "move_between_objects"
    return None


def _infer_instruction_object_options(
    instruction: str,
    *,
    candidate_catalogs: Sequence[str],
) -> dict[str, str]:
    text = re.sub(r"\s+", " ", str(instruction).strip().lower())
    if not text:
        return {}
    matches: list[str] = []
    for catalog in _dedupe_names(candidate_catalogs):
        name = canonical_object_name(str(catalog)).strip().lower()
        if name and re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text):
            matches.append(str(catalog))
    if not matches:
        return {}
    out = {"target_object": matches[0]}
    if len(matches) >= 2:
        out["reference_object"] = matches[1]
    if len(matches) >= 3:
        out["second_reference_object"] = matches[2]
    return out


def _metadata_name_list(task_metadata: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = task_metadata.get(key)
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence):
        raise ValueError(f"Task metadata `{key}` must be a list of object names.")
    return _dedupe_names([str(item) for item in raw])


def _metadata_float(task_metadata: dict[str, Any], key: str, default: float) -> float:
    raw = task_metadata.get(key, default)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Task metadata `{key}` must be numeric, got {raw!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Task metadata `{key}` must be finite, got {raw!r}")
    return value


def _metadata_float_pair(task_metadata: dict[str, Any], key: str, default: Sequence[float]) -> tuple[float, float]:
    raw = task_metadata.get(key, default)
    if raw is None:
        raw = default
    return _normalize_float_pair(raw, name=f"task metadata `{key}`")


def _metadata_xy_pair(task_metadata: dict[str, Any], key: str, default: Sequence[float]) -> tuple[float, float]:
    raw = task_metadata.get(key, default)
    if raw is None:
        raw = default
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Task metadata `{key}` must provide at least two floats, got {raw!r}")
    x = float(arr[0])
    y = float(arr[1])
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError(f"Task metadata `{key}` must be finite, got {raw!r}")
    return x, y


def _resolve_object_spawn_config(
    task_metadata: dict[str, Any],
    *,
    support_surface_z: float,
) -> dict[str, Any]:
    center_default = task_metadata.get("goal_center_xy", DEFAULT_GOAL_CENTER_XY)
    x_bounds = _metadata_float_pair(task_metadata, "object_spawn_x_bounds", (-0.20, 0.20))
    y_bounds = _metadata_float_pair(task_metadata, "object_spawn_y_bounds", (-0.20, 0.20))
    center_xy = _metadata_xy_pair(task_metadata, "object_spawn_center_xy", center_default)
    return {
        "xy_bounds": (x_bounds, y_bounds, float(support_surface_z)),
        "min_gap": max(0.0, _metadata_float(task_metadata, "object_spawn_min_gap", 0.02)),
        "max_tries": max(1, int(round(_metadata_float(task_metadata, "object_spawn_max_tries", 200.0)))),
        "min_ee_dist": max(0.0, _metadata_float(task_metadata, "object_spawn_min_ee_dist", 0.10)),
        "support_clearance": max(0.0, _metadata_float(task_metadata, "object_spawn_support_clearance", 0.002)),
        "avoid_xy_center": center_xy,
        "avoid_xy_radius": max(0.0, _metadata_float(task_metadata, "object_spawn_center_exclusion_radius", 0.0)),
    }


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
    required_object_pool: Sequence[str] = (),
    min_scene_objects: int,
    max_scene_objects: int,
    scene_variant_count: int,
    seed: int | None,
) -> list[SceneSpec]:
    if not target_object_pool:
        raise ValueError("Target object pool cannot be empty when building scene variants.")

    scene_names = _unique_scene_names([SceneSpec(name=name, objects=()) for name in scene_names])
    targets = _dedupe_names(target_object_pool)
    distractors = _dedupe_names(distractor_object_pool) if distractor_object_pool else targets
    required_objects = _dedupe_names(required_object_pool)
    total_pool = _dedupe_names([*targets, *distractors, *required_objects])
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
        chosen: list[str] = []

        required_candidates = [name for name in required_objects if name != target_name]
        if desired_count > 1 and required_candidates:
            chosen.append(str(required_candidates[int(rng.integers(0, len(required_candidates)))]))

        max_distractors = max(0, desired_count - 1 - len(chosen))
        distractor_candidates = [
            name for name in _dedupe_names([*distractors, *required_objects])
            if name != target_name and name not in chosen
        ]
        if distractor_candidates and max_distractors > 0:
            sample_size = min(max_distractors, len(distractor_candidates))
            chosen.extend(str(name) for name in rng.choice(distractor_candidates, size=sample_size, replace=False))
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
    required_scene_pool = _metadata_name_list(task_metadata, "required_scene_object_pool")
    container_pool = _metadata_name_list(task_metadata, "container_object_pool")
    if container_pool:
        required_scene_pool = _dedupe_names([*required_scene_pool, *container_pool])

    if not target_pool and not distractor_pool:
        allowed = _dedupe_names(allowed_objects)
        return list(base_scenes), allowed, allowed, ()

    if not target_pool:
        target_pool = _dedupe_names(allowed_objects)
    if not target_pool:
        raise ValueError("Task metadata target_object_pool is empty and no allowed_objects were provided.")

    distractor_pool = _dedupe_names(distractor_pool) if distractor_pool else tuple(target_pool)
    allowed = _dedupe_names([*target_pool, *distractor_pool, *required_scene_pool])

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
        required_object_pool=required_scene_pool,
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


def _candidate_texture_files(tex_dir: Path) -> list[Path]:
    return sorted(p for p in tex_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def _texture_dir_signature(paths: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    out: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
        out.append((path.name, int(stat.st_size), mtime_ns))
    return tuple(out)


def _is_texture_file_valid(path: Path) -> bool:
    if Image is None:
        return path.is_file()
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def _validated_texture_files(tex_dir: Path) -> list[Path]:
    tex_dir = tex_dir.expanduser().resolve()
    candidates = _candidate_texture_files(tex_dir)
    signature = _texture_dir_signature(candidates)

    cached = _TEXTURE_VALIDATION_CACHE.get(tex_dir)
    if cached is not None and cached[0] == signature:
        valid_paths, invalid_paths = cached[1], cached[2]
    else:
        valid_paths = []
        invalid_paths = []
        for path in candidates:
            if _is_texture_file_valid(path):
                valid_paths.append(path)
            else:
                invalid_paths.append(path)
        _TEXTURE_VALIDATION_CACHE[tex_dir] = (signature, valid_paths, invalid_paths)

    if invalid_paths and tex_dir not in _TEXTURE_VALIDATION_WARNED:
        sample = ", ".join(path.name for path in invalid_paths[:3])
        suffix = "" if len(invalid_paths) <= 3 else ", ..."
        print(
            f"[env] Skipping {len(invalid_paths)} invalid desk texture file(s) from {tex_dir}: {sample}{suffix}",
            flush=True,
        )
        _TEXTURE_VALIDATION_WARNED.add(tex_dir)

    return list(valid_paths)


def _wrapper_bundle_exists(wrapper_xml: Path) -> bool:
    wrapper_xml = Path(wrapper_xml).expanduser().resolve()
    if not wrapper_xml.exists():
        return False

    wrap_root = WRAP_DIR.resolve()
    queue = [wrapper_xml]
    seen: set[Path] = set()

    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        if not current.exists():
            return False

        try:
            tree = ET.parse(current)
        except Exception:
            return False
        root = tree.getroot()

        for _, file_attr in _iter_includes(root):
            include_path = _resolve_include_path(current, file_attr)
            include_is_local = current.parent in include_path.parents or wrap_root in include_path.parents
            if include_is_local:
                if not include_path.exists():
                    return False
                queue.append(include_path)

        for tag_name in ("texture", "mesh", "hfield"):
            for elem in root.iter(tag_name):
                file_attr = elem.get("file")
                if not file_attr:
                    continue
                asset_path = _resolve_include_path(current, file_attr)
                asset_is_local = current.parent in asset_path.parents or wrap_root in asset_path.parents
                if asset_is_local and not asset_path.exists():
                    return False

    return True


def _wrapper_cache_prefix(scene_name: str, object_names: Sequence[str]) -> str:
    obj_part = "-".join(sorted(str(name) for name in object_names))
    return f"{scene_name}__{obj_part}"


def _wrapper_bundle_dir(
    wrapper_dir: Path | str,
    *,
    scene_name: str,
    object_names: Sequence[str],
) -> Path:
    wrapper_root = Path(wrapper_dir).expanduser().resolve()
    return wrapper_root / _wrapper_cache_prefix(scene_name, object_names)


def _candidate_existing_wrapper_paths(
    wrapper_dir: Path,
    *,
    scene_name: str,
    object_names: Sequence[str],
) -> list[Path]:
    wrapper_root = Path(wrapper_dir).expanduser().resolve()
    if not wrapper_root.exists():
        return []

    prefix = _wrapper_cache_prefix(scene_name, object_names)
    patterns = (
        f"{prefix}_wrapper.xml",
        f"{prefix}_wrapper__*__desktex_*.xml",
        f"{prefix}__rltmp_*.xml",
        f"{prefix}__rltmp_*__*__desktex_*.xml",
    )

    candidates: list[Path] = []
    seen: set[Path] = set()
    search_roots = [wrapper_root, _wrapper_bundle_dir(wrapper_root, scene_name=scene_name, object_names=object_names)]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for pattern in patterns:
            for path in sorted(search_root.glob(pattern)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                if not _wrapper_bundle_exists(resolved):
                    continue
                seen.add(resolved)
                candidates.append(resolved)
    return candidates


def _stable_file_signature(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    payload = f"{resolved.as_posix()}::{int(stat.st_size)}::{mtime_ns}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _shared_desk_texture_cache_path(texture_path: Path) -> Path:
    resolved = Path(texture_path).expanduser().resolve()
    tex_dir = WRAP_DIR / "_desk_textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    return tex_dir / f"{_stable_file_signature(resolved)}__{resolved.name}"


def _desk_texture_variant_tag(base_wrapper_xml: Path, chosen_texture: Path) -> str:
    payload = (
        f"{Path(base_wrapper_xml).expanduser().resolve().as_posix()}::"
        f"{_stable_file_signature(chosen_texture)}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


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
    output_dir: Path,
    mapping: dict[Path, Path],
    generated_xmls: list[Path],
) -> int:
    source_xml = source_xml.resolve()
    if source_xml in mapping:
        return 0

    path_hash = hashlib.sha1(source_xml.as_posix().encode("utf-8")).hexdigest()[:10]
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patched_xml = output_dir / f"{source_xml.stem}__{path_hash}__desktex_{variant_tag}{source_xml.suffix}"
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
            output_dir=output_dir,
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
    output_dir = base_wrapper_xml.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _wrapper_bundle_exists(base_wrapper_xml):
        raise FileNotFoundError(
            "Base wrapper bundle is missing or incomplete before desk-texture patching. "
            f"wrapper={base_wrapper_xml}"
        )

    copied_texture = _shared_desk_texture_cache_path(chosen_texture)
    if not copied_texture.exists():
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
        output_dir=output_dir,
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
        generated_files=[],
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
      - gripper_cmd is a delta applied to normalized gripper target 0..1.
        Positive values open, negative values close, scaled by action_step_gripper.

    Observation space:
      - ee_position: (3,)
      - target_object_position: (3,) waypoint goal position
      - all_object_positions: (max_objects, 3)
      - object_position_mask: (max_objects,)
      - instruction_onehot: (len(INSTRUCTION_TYPES),)
      - goal_direction: (3,) motion direction for the current waypoint goal.
    """

    metadata = {"render.modes": []}
    _next_env_instance_id = 0

    def __init__(
        self,
        catalog_path: Path | str | None = None,
        max_steps: int = 150,
        max_objects: int = 8,
        action_step_xyz: float = 0.02,
        action_step_yaw: float = 0.25,
        action_step_gripper: float | None = None,
        hold_steps: int = 0,
        lock_non_commanded_axes: bool | None = None,
        lock_non_commanded_axes_threshold: float | None = None,
        randomize_ee_start: bool | None = None,
        ee_start_x_bounds: Sequence[float] | None = None,
        ee_start_y_bounds: Sequence[float] | None = None,
        ee_start_z: float | None = None,
        record_trajectory: bool | None = None,
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
        reuse_existing_wrapper_variants: bool = False,
        wrapper_dir: Path | str | None = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._env_instance_id = int(type(self)._next_env_instance_id)
        type(self)._next_env_instance_id += 1

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
        if action_step_gripper is None:
            action_step_gripper = _load_float_env(CDPR_ACTION_STEP_GRIPPER_ENV, default=0.05)
        self.action_step_gripper = max(0.0, float(action_step_gripper))
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
        if record_trajectory is None:
            record_trajectory = _load_bool_env(CDPR_RECORD_TRAJECTORY_ENV, default=False)
        self.lock_non_commanded_axes = bool(lock_non_commanded_axes)
        self.lock_non_commanded_axes_threshold = max(0.0, float(lock_non_commanded_axes_threshold))
        self.randomize_ee_start = bool(randomize_ee_start)
        self.ee_start_x_bounds = _normalize_float_pair(ee_start_x_bounds, name="ee_start_x_bounds")
        self.ee_start_y_bounds = _normalize_float_pair(ee_start_y_bounds, name="ee_start_y_bounds")
        self.ee_start_z = None if ee_start_z is None else max(float(ee_start_z), MIN_EE_START_Z)
        self.record_trajectory = bool(record_trajectory)
        self.move_distance = float(move_distance)
        self.lift_distance = float(lift_distance)
        self.capture_frames = bool(capture_frames)
        self.instruction_types = tuple(instruction_types) if instruction_types else None
        self.wrapper_cleanup = bool(wrapper_cleanup)
        self.use_wrapper_cache = bool(use_wrapper_cache)
        self.reuse_existing_wrapper_variants = bool(reuse_existing_wrapper_variants)
        self.desk_geom_regex = str(desk_geom_regex)
        texrepeat_vals = tuple(desk_texrepeat)
        if len(texrepeat_vals) != 2:
            raise ValueError("desk_texrepeat must contain exactly two integers: X Y.")
        self.desk_texrepeat = (int(texrepeat_vals[0]), int(texrepeat_vals[1]))

        self.np_random = np.random.default_rng(seed)
        wrapper_root = WRAP_DIR if wrapper_dir is None else Path(wrapper_dir).expanduser().resolve()
        wrapper_root.mkdir(parents=True, exist_ok=True)
        self.wrapper_dir = wrapper_root

        self.desk_texture_files: list[Path] = []
        if desk_textures_dir is not None:
            tex_dir = Path(desk_textures_dir).expanduser().resolve()
            if not tex_dir.exists():
                raise ValueError(f"desk_textures_dir not found: {tex_dir}")
            self.desk_texture_files = _validated_texture_files(tex_dir)
            if not self.desk_texture_files:
                raise ValueError(f"No texture files found in: {tex_dir}")

        self._goal_region = _load_json_env("RLVLA_TASK_GOAL_REGION_JSON")
        self._dense_reward_terms = _load_json_env("RLVLA_TASK_DENSE_REWARD_TERMS_JSON")
        self._task_metadata = _load_json_env("RLVLA_TASK_METADATA_JSON")
        self._goal_relation = os.environ.get("RLVLA_TASK_GOAL_RELATION")
        instruction_sampling_mode = str(self._task_metadata.get("instruction_sampling", "uniform_cycle")).strip().lower()
        if instruction_sampling_mode not in {"uniform_cycle", "random"}:
            raise ValueError(
                "Task metadata `instruction_sampling` must be either `uniform_cycle` or `random`, "
                f"got {instruction_sampling_mode!r}."
            )
        self.instruction_sampling = instruction_sampling_mode
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
        self._reference_catalog_name = ""
        self._reference_body_name = ""
        self._second_reference_catalog_name = ""
        self._second_reference_body_name = ""
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
        self._caught_object_start_active = False
        self._caught_object_start_body = ""
        self._caught_object_start_catalog = ""
        self._caught_object_start_position = np.zeros((3,), dtype=np.float32)
        self._caught_object_start_ee_offset = np.zeros((3,), dtype=np.float32)
        self._caught_object_start_hold_offset = np.zeros((3,), dtype=np.float32)
        self._caught_object_start_gripper_opening = 0.0
        self._curriculum_mode = ""
        self._curriculum_shell: int | None = None
        self._curriculum_reset_info: dict[str, Any] = {}
        self._support_surface_z = 0.0
        self._ee_min_z = float("-inf")
        self._ee_spawn_z = float("-inf")
        self._locked_target_xyz = np.zeros((3,), dtype=np.float32)
        self._episode_ee_start = self._default_ee_start().astype(np.float32)
        self._goal_position = np.zeros((3,), dtype=np.float32)
        self._goal_motion_direction = np.zeros((3,), dtype=np.float32)
        self._episode_index = -1
        self._reset_counter = 0
        self._invalid_wrapper_paths: set[Path] = set()
        self._last_wrapper_reused_from_cache = False
        self._instruction_cycle: list[str] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
        instruction: str | None = None,
        curriculum_shell: int | None = None,
        curriculum_mode: str | None = None,
    ):
        options = dict(options or {})
        if instruction is not None:
            options["instruction"] = str(instruction)
        if curriculum_shell is not None:
            options["curriculum_shell"] = int(curriculum_shell)
        if curriculum_mode is not None:
            options["curriculum_mode"] = str(curriculum_mode)
        if "instruction_type" not in options and options.get("instruction") is not None:
            inferred = _infer_instruction_type_from_text(str(options.get("instruction", "")))
            if inferred:
                options["instruction_type"] = inferred
            binding_options = _infer_instruction_object_options(
                str(options.get("instruction", "")),
                candidate_catalogs=(
                    list(getattr(self, "target_object_pool", ()))
                    + list(getattr(self, "allowed_objects", ()))
                    + list(getattr(self, "scene_object_pool", ()))
                ),
            )
            for key, value in binding_options.items():
                options.setdefault(key, value)

        self._prepare_episode_rng(seed)

        self.close()
        scene = self._sample_scene(options=options)
        self._scene_name = scene.name
        self._scene_catalog_objects = list(scene.objects)

        episode_ee_start = self._sample_episode_ee_start(options=options)
        self._episode_ee_start = episode_ee_start.astype(np.float32)
        wrapper_xml: Path | None = None
        original_reuse_existing = bool(getattr(self, "reuse_existing_wrapper_variants", False))
        for attempt in range(2):
            if attempt > 0:
                self.reuse_existing_wrapper_variants = False

            try:
                wrapper_xml = self._build_episode_wrapper(scene=scene, ee_start=episode_ee_start)
                self._current_wrapper_xml = wrapper_xml
                self.sim = self._sim_cls(
                    xml_path=str(wrapper_xml),
                    output_dir=str(DEFAULT_VIDEO_DIR),
                    record_trajectory=self.record_trajectory,
                )
                self.sim.initialize()
                break
            except Exception:
                if bool(getattr(self, "_last_wrapper_reused_from_cache", False)) and wrapper_xml is not None:
                    self._invalid_wrapper_paths.add(Path(wrapper_xml).resolve())
                if self.sim is not None:
                    try:
                        self.sim.cleanup()
                    except Exception:
                        pass
                self.sim = None
                self._current_wrapper_xml = None
                if attempt == 0 and bool(getattr(self, "_last_wrapper_reused_from_cache", False)):
                    continue
                raise
            finally:
                self.reuse_existing_wrapper_variants = original_reuse_existing
        if hasattr(self.sim, "hold_current_pose"):
            self.sim.hold_current_pose(warm_steps=10)
        self._refresh_workspace_safety()
        self._move_ee_to_episode_start()
        self._clear_sim_recording_buffers()

        self._catalog_to_body, self._object_body_names = self._resolve_objects(scene.objects)
        self._inverse_catalog_to_body = {v: k for k, v in self._catalog_to_body.items()}
        if self._object_body_names:
            try:
                object_spawn_config = _resolve_object_spawn_config(
                    self._task_metadata,
                    support_surface_z=self._support_surface_z,
                )
                place_objects_non_overlapping(
                    self.sim,
                    self._object_body_names,
                    xy_bounds=object_spawn_config["xy_bounds"],
                    min_gap=object_spawn_config["min_gap"],
                    max_tries=object_spawn_config["max_tries"],
                    min_ee_dist=object_spawn_config["min_ee_dist"],
                    support_clearance=object_spawn_config["support_clearance"],
                    avoid_xy_center=object_spawn_config["avoid_xy_center"],
                    avoid_xy_radius=object_spawn_config["avoid_xy_radius"],
                )
            except Exception:
                # Continue if placement fails; wrapper-provided placement is still valid.
                pass

        instruction_type = self._sample_instruction_type(options=options)
        (
            self._target_catalog_name,
            self._target_body_name,
            self._reference_catalog_name,
            self._reference_body_name,
            self._second_reference_catalog_name,
            self._second_reference_body_name,
        ) = self._select_instruction_objects(scene, instruction_type=instruction_type, options=options)

        self._instruction_spec = sample_instruction(
            target_object=self._target_catalog_name or None,
            rng=self.np_random,
            allowed_instruction_types=self.instruction_types,
            move_distance=self.move_distance,
            lift_distance=self.lift_distance,
            instruction_type=instruction_type,
            reference_object=self._reference_catalog_name or None,
            second_reference_object=self._second_reference_catalog_name or None,
        )
        setattr(self.sim, "language_instruction", self._instruction_spec.text)

        self._reset_caught_object_start_state()
        self._maybe_spawn_target_caught_at_ee(
            instruction_type=self._instruction_spec.instruction_type,
            options=options,
        )
        self._curriculum_mode = str(options.get("curriculum_mode") or "")
        self._curriculum_shell = (
            None
            if options.get("curriculum_shell") is None
            else int(options.get("curriculum_shell"))
        )
        self._curriculum_reset_info = {}
        if self._curriculum_mode == "reverse_frontier" and self._curriculum_shell is not None:
            from .cdpr_reverse_shells import apply_cdpr_reverse_shell

            self._curriculum_reset_info = dict(
                apply_cdpr_reverse_shell(
                    self,
                    shell_id=int(self._curriculum_shell),
                    rng=self.np_random,
                )
            )

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
        reward_target_pos = self._current_manipulated_object_position(default=self._goal_position)
        self._reward_state = init_reward_state(ee0, reward_target_pos)
        reward_initial = self._curriculum_reset_info.get("curriculum_reward_initial_obj_pos")
        if reward_initial is not None:
            reward_initial_arr = np.asarray(reward_initial, dtype=np.float32).reshape(-1)
            if reward_initial_arr.size >= 3 and np.all(np.isfinite(reward_initial_arr[:3])):
                self._reward_state.initial_obj_pos = reward_initial_arr[:3].astype(np.float32).copy()
        if bool(self._curriculum_reset_info.get("curriculum_target_grasped", False)):
            self._reward_state.grasped = True
        self._reward_state.gripper_closed = self._is_gripper_closed(self._get_gripper_opening())
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
        self._last_caught_body = self._caught_object_start_body if self._caught_object_start_active else ""
        self._last_caught_catalog = self._caught_object_start_catalog if self._caught_object_start_active else ""
        self._locked_target_xyz = self._get_ee_position().astype(np.float32)

        obs = self._get_obs()
        info = self._base_info()
        info.update(self._curriculum_reset_info)
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
        goal_pos = self._current_target_reference_position()
        gripper_opening = self._get_gripper_opening()
        gripper_closed = self._is_gripper_closed(gripper_opening)
        self._reward_state.gripper_closed = gripper_closed
        if not gripper_closed:
            self._reward_state.grasped = False
        current_goal_direction = self._current_goal_motion_direction(ee_pos=ee, goal_pos=goal_pos)
        camera_alignment = self._get_ee_camera_alignment(target_pos=goal_pos, direction=current_goal_direction)
        caught_body, caught_catalog, caught_score, caught_is_target = self._detect_caught_object(ee)
        reward_kwargs = {
            "spec": self._instruction_spec,
            "ee_pos": ee,
            "obj_pos": goal_pos,
            "goal_pos": goal_pos,
            "reward_state": self._reward_state,
            "action": action,
            "camera_alignment": camera_alignment,
            "goal_direction": current_goal_direction,
            "goal_region": self._goal_region,
            "goal_relation": self._goal_relation,
            "dense_reward_terms": self._dense_reward_terms,
            "task_metadata": self._task_metadata,
            "env": self,
            "sim": self.sim,
            "scene_name": self._scene_name,
            "target_catalog_name": self._target_catalog_name,
            "target_body_name": self._target_body_name,
            "reference_catalog_name": self._reference_catalog_name,
            "reference_body_name": self._reference_body_name,
            "second_reference_catalog_name": self._second_reference_catalog_name,
            "second_reference_body_name": self._second_reference_body_name,
            "gripper_opening": gripper_opening,
            "support_surface_z": self._support_surface_z,
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
        self._goal_position = goal_pos.astype(np.float32)
        self._goal_motion_direction = self._current_goal_motion_direction(ee_pos=ee, goal_pos=goal_pos)

        self._step_count += 1
        terminated = bool(success)
        truncated = bool(self._step_count >= self.max_steps and not terminated)

        obs = self._get_obs()
        info = self._base_info()
        info.update(reward_info)
        info["success"] = bool(success)
        info["reward"] = float(reward)
        info["step"] = int(self._step_count)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info["env_done"] = bool(terminated or truncated)
        info["episode_timeout"] = bool(truncated and not terminated)
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
        self._reference_catalog_name = ""
        self._reference_body_name = ""
        self._second_reference_catalog_name = ""
        self._second_reference_body_name = ""
        self._reset_caught_object_start_state()
        self._curriculum_mode = ""
        self._curriculum_shell = None
        self._curriculum_reset_info = {}
        self._goal_position = np.zeros((3,), dtype=np.float32)
        self._goal_motion_direction = np.zeros((3,), dtype=np.float32)

    def capture_state(self) -> dict[str, Any]:
        if self.sim is None:
            raise RuntimeError("Environment was not reset before capture_state().")
        if not hasattr(self.sim, "capture_state"):
            raise RuntimeError("Underlying simulator does not support capture_state().")

        return {
            "sim_state": self.sim.capture_state(),
            "step_count": int(self._step_count),
            "yaw": float(self._yaw),
            "last_gripper_cmd": float(self._last_gripper_cmd),
            "instruction_spec": copy.deepcopy(self._instruction_spec),
            "reward_state": copy.deepcopy(self._reward_state),
            "scene_name": str(self._scene_name),
            "target_catalog_name": str(self._target_catalog_name),
            "target_body_name": str(self._target_body_name),
            "reference_catalog_name": str(getattr(self, "_reference_catalog_name", "")),
            "reference_body_name": str(getattr(self, "_reference_body_name", "")),
            "second_reference_catalog_name": str(getattr(self, "_second_reference_catalog_name", "")),
            "second_reference_body_name": str(getattr(self, "_second_reference_body_name", "")),
            "catalog_to_body": dict(self._catalog_to_body),
            "object_body_names": list(self._object_body_names),
            "scene_catalog_objects": list(self._scene_catalog_objects),
            "desk_texture_name": str(self._desk_texture_name),
            "current_wrapper_xml": (
                str(self._current_wrapper_xml)
                if self._current_wrapper_xml is not None
                else ""
            ),
            "inverse_catalog_to_body": dict(self._inverse_catalog_to_body),
            "prev_object_positions": {
                str(name): np.asarray(pos, dtype=np.float32).copy()
                for name, pos in self._prev_object_positions.items()
            },
            "prev_ee_for_catch": np.asarray(self._prev_ee_for_catch, dtype=np.float32).copy(),
            "last_caught_body": str(self._last_caught_body),
            "last_caught_catalog": str(self._last_caught_catalog),
            "caught_object_start_active": bool(getattr(self, "_caught_object_start_active", False)),
            "caught_object_start_body": str(getattr(self, "_caught_object_start_body", "")),
            "caught_object_start_catalog": str(getattr(self, "_caught_object_start_catalog", "")),
            "caught_object_start_position": np.asarray(
                getattr(self, "_caught_object_start_position", np.zeros((3,), dtype=np.float32)),
                dtype=np.float32,
            ).copy(),
            "caught_object_start_ee_offset": np.asarray(
                getattr(self, "_caught_object_start_ee_offset", np.zeros((3,), dtype=np.float32)),
                dtype=np.float32,
            ).copy(),
            "caught_object_start_hold_offset": np.asarray(
                getattr(self, "_caught_object_start_hold_offset", np.zeros((3,), dtype=np.float32)),
                dtype=np.float32,
            ).copy(),
            "caught_object_start_gripper_opening": float(
                getattr(self, "_caught_object_start_gripper_opening", 0.0)
            ),
            "curriculum_mode": str(getattr(self, "_curriculum_mode", "")),
            "curriculum_shell": copy.deepcopy(getattr(self, "_curriculum_shell", None)),
            "curriculum_reset_info": copy.deepcopy(getattr(self, "_curriculum_reset_info", {})),
            "support_surface_z": float(self._support_surface_z),
            "ee_min_z": float(self._ee_min_z),
            "ee_spawn_z": float(self._ee_spawn_z),
            "locked_target_xyz": np.asarray(self._locked_target_xyz, dtype=np.float32).copy(),
            "episode_ee_start": np.asarray(self._episode_ee_start, dtype=np.float32).copy(),
            "goal_position": np.asarray(self._goal_position, dtype=np.float32).copy(),
            "goal_motion_direction": np.asarray(self._goal_motion_direction, dtype=np.float32).copy(),
            "episode_index": int(self._episode_index),
            "reset_counter": int(self._reset_counter),
            "instruction_cycle": list(self._instruction_cycle),
            "rng_state": copy.deepcopy(self.np_random.bit_generator.state),
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        if self.sim is None:
            raise RuntimeError("Environment was not reset before restore_state().")
        if not hasattr(self.sim, "restore_state"):
            raise RuntimeError("Underlying simulator does not support restore_state().")

        self.sim.restore_state(snapshot["sim_state"])
        self._step_count = int(snapshot["step_count"])
        self._yaw = float(snapshot["yaw"])
        self._last_gripper_cmd = float(snapshot["last_gripper_cmd"])
        self._instruction_spec = copy.deepcopy(snapshot["instruction_spec"])
        self._reward_state = copy.deepcopy(snapshot["reward_state"])
        self._scene_name = str(snapshot["scene_name"])
        self._target_catalog_name = str(snapshot["target_catalog_name"])
        self._target_body_name = str(snapshot["target_body_name"])
        self._reference_catalog_name = str(snapshot.get("reference_catalog_name", ""))
        self._reference_body_name = str(snapshot.get("reference_body_name", ""))
        self._second_reference_catalog_name = str(snapshot.get("second_reference_catalog_name", ""))
        self._second_reference_body_name = str(snapshot.get("second_reference_body_name", ""))
        self._catalog_to_body = dict(snapshot["catalog_to_body"])
        self._object_body_names = [str(name) for name in snapshot["object_body_names"]]
        self._scene_catalog_objects = [str(name) for name in snapshot["scene_catalog_objects"]]
        self._desk_texture_name = str(snapshot["desk_texture_name"])
        wrapper_xml = str(snapshot.get("current_wrapper_xml", "") or "").strip()
        self._current_wrapper_xml = Path(wrapper_xml) if wrapper_xml else None
        self._inverse_catalog_to_body = dict(snapshot["inverse_catalog_to_body"])
        self._prev_object_positions = {
            str(name): np.asarray(pos, dtype=np.float32).copy()
            for name, pos in dict(snapshot["prev_object_positions"]).items()
        }
        self._prev_ee_for_catch = np.asarray(snapshot["prev_ee_for_catch"], dtype=np.float32).copy()
        self._last_caught_body = str(snapshot["last_caught_body"])
        self._last_caught_catalog = str(snapshot["last_caught_catalog"])
        self._caught_object_start_active = bool(snapshot.get("caught_object_start_active", False))
        self._caught_object_start_body = str(snapshot.get("caught_object_start_body", ""))
        self._caught_object_start_catalog = str(snapshot.get("caught_object_start_catalog", ""))
        self._caught_object_start_position = np.asarray(
            snapshot.get("caught_object_start_position", np.zeros((3,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3).copy()
        self._caught_object_start_ee_offset = np.asarray(
            snapshot.get("caught_object_start_ee_offset", np.zeros((3,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3).copy()
        self._caught_object_start_hold_offset = np.asarray(
            snapshot.get("caught_object_start_hold_offset", np.zeros((3,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3).copy()
        self._caught_object_start_gripper_opening = float(snapshot.get("caught_object_start_gripper_opening", 0.0))
        self._curriculum_mode = str(snapshot.get("curriculum_mode", ""))
        self._curriculum_shell = copy.deepcopy(snapshot.get("curriculum_shell", None))
        self._curriculum_reset_info = dict(snapshot.get("curriculum_reset_info", {}) or {})
        self._support_surface_z = float(snapshot["support_surface_z"])
        self._ee_min_z = float(snapshot["ee_min_z"])
        self._ee_spawn_z = float(snapshot["ee_spawn_z"])
        self._locked_target_xyz = np.asarray(snapshot["locked_target_xyz"], dtype=np.float32).copy()
        self._episode_ee_start = np.asarray(snapshot["episode_ee_start"], dtype=np.float32).copy()
        self._goal_position = np.asarray(snapshot["goal_position"], dtype=np.float32).copy()
        self._goal_motion_direction = np.asarray(snapshot["goal_motion_direction"], dtype=np.float32).copy()
        self._episode_index = int(snapshot["episode_index"])
        self._reset_counter = int(snapshot["reset_counter"])
        self._instruction_cycle = [str(item) for item in snapshot.get("instruction_cycle") or []]
        self.np_random.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        if self._instruction_spec is not None:
            setattr(self.sim, "language_instruction", self._instruction_spec.text)

    def _reset_caught_object_start_state(self) -> None:
        self._caught_object_start_active = False
        self._caught_object_start_body = ""
        self._caught_object_start_catalog = ""
        self._caught_object_start_position = np.zeros((3,), dtype=np.float32)
        self._caught_object_start_ee_offset = np.zeros((3,), dtype=np.float32)
        self._caught_object_start_hold_offset = np.zeros((3,), dtype=np.float32)
        self._caught_object_start_gripper_opening = 0.0

    def _caught_object_start_instruction_types(self) -> tuple[str, ...]:
        configured = _metadata_name_list(self._task_metadata, "caught_object_start_instruction_types")
        return configured or DEFAULT_CAUGHT_OBJECT_START_INSTRUCTION_TYPES

    def _should_spawn_target_caught_at_ee(
        self,
        *,
        instruction_type: str,
        options: Optional[dict[str, Any]] = None,
    ) -> bool:
        options = dict(options or {})
        instruction_type = str(instruction_type)
        if instruction_type not in set(self._caught_object_start_instruction_types()):
            return False
        if not self._target_body_name:
            return False

        forced = options.get("start_with_caught_object", options.get("caught_object_start"))
        if forced is not None:
            if isinstance(forced, str):
                return forced.strip().lower() in {"1", "true", "yes", "on"}
            return bool(forced)

        probability = _metadata_float(self._task_metadata, "caught_object_start_probability", 0.0)
        probability = float(np.clip(probability, 0.0, 1.0))
        if probability <= 0.0:
            return False
        return bool(float(self.np_random.random()) < probability)

    def _caught_object_start_offset(self) -> np.ndarray:
        raw = self._task_metadata.get(
            "caught_object_start_object_offset",
            DEFAULT_CAUGHT_OBJECT_START_OFFSET,
        )
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            raise ValueError(
                "Task metadata `caught_object_start_object_offset` must provide three floats: dx dy dz."
            )
        offset = arr[:3].astype(np.float32).copy()

        xy_jitter = max(0.0, _metadata_float(self._task_metadata, "caught_object_start_xy_jitter", 0.0))
        if xy_jitter > 0.0:
            offset[:2] += self.np_random.uniform(-xy_jitter, xy_jitter, size=(2,)).astype(np.float32)

        z_jitter = max(0.0, _metadata_float(self._task_metadata, "caught_object_start_z_jitter", 0.0))
        if z_jitter > 0.0:
            offset[2] += float(self.np_random.uniform(-z_jitter, z_jitter))
        return offset

    def _caught_object_start_fit_gripper_enabled(self) -> bool:
        raw = self._task_metadata.get("caught_object_start_fit_gripper", True)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() not in {"0", "false", "no", "off"}
        return bool(raw)

    def _caught_object_start_pin_object_enabled(self) -> bool:
        raw = self._task_metadata.get("caught_object_start_pin_object", False)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)

    def _geom_half_extent_along_axis(self, geom_id: int, axis: np.ndarray) -> float:
        model = self.sim.model
        data = self.sim.data
        gid = int(geom_id)
        axis_arr = np.asarray(axis, dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis_arr))
        if axis_norm < 1e-9:
            return 0.0
        axis_arr /= axis_norm

        try:
            g_box = int(mj.mjtGeom.mjGEOM_BOX)
            g_cylinder = int(mj.mjtGeom.mjGEOM_CYLINDER)
            g_capsule = int(mj.mjtGeom.mjGEOM_CAPSULE)
            g_sphere = int(mj.mjtGeom.mjGEOM_SPHERE)
        except Exception:
            g_box, g_cylinder, g_capsule, g_sphere = 6, 4, 3, 0

        gtype = int(model.geom_type[gid])
        size = np.asarray(model.geom_size[gid], dtype=np.float64)
        if gtype == g_box:
            half_local = np.array([size[0], size[1], size[2]], dtype=np.float64)
        elif gtype == g_cylinder:
            half_local = np.array([size[0], size[0], size[1]], dtype=np.float64)
        elif gtype == g_capsule:
            half_local = np.array([size[0], size[0], size[1] + size[0]], dtype=np.float64)
        elif gtype == g_sphere:
            half_local = np.array([size[0], size[0], size[0]], dtype=np.float64)
        else:
            radius = float(model.geom_rbound[gid]) if hasattr(model, "geom_rbound") else float(size[0])
            half_local = np.array([radius, radius, radius], dtype=np.float64)

        xmat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        return float(np.sum(np.abs(xmat.T @ axis_arr) * half_local))

    def _finger_pair_geometry(self) -> dict[str, np.ndarray | float] | None:
        sim = getattr(self, "sim", None)
        if sim is None or not hasattr(sim, "model") or not hasattr(sim, "data"):
            return None

        def _first_geom_id(names: Sequence[str]) -> int:
            for name in names:
                gid = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_GEOM, str(name))
                if gid != -1:
                    return int(gid)
            return -1

        left_gid = _first_geom_id(("finger_l_tip", "finger_l_link"))
        right_gid = _first_geom_id(("finger_r_tip", "finger_r_link"))
        if left_gid == -1 or right_gid == -1:
            return None

        left_pos = np.asarray(self.sim.data.geom_xpos[left_gid], dtype=np.float64).reshape(3)
        right_pos = np.asarray(self.sim.data.geom_xpos[right_gid], dtype=np.float64).reshape(3)
        separation = left_pos - right_pos
        distance = float(np.linalg.norm(separation))
        if distance < 1e-9:
            return None
        axis = separation / distance
        left_half = self._geom_half_extent_along_axis(left_gid, axis)
        right_half = self._geom_half_extent_along_axis(right_gid, axis)
        return {
            "center": (0.5 * (left_pos + right_pos)).astype(np.float32),
            "axis": axis.astype(np.float32),
            "inner_gap": float(max(0.0, distance - left_half - right_half)),
        }

    def _body_width_along_axis(self, body_name: str, axis: np.ndarray) -> float | None:
        model = self.sim.model
        data = self.sim.data
        axis_arr = np.asarray(axis, dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis_arr))
        if axis_norm < 1e-9:
            return None
        axis_arr /= axis_norm

        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, str(body_name))
        if body_id == -1:
            return None

        children = {idx: [] for idx in range(model.nbody)}
        for idx in range(1, model.nbody):
            parent = int(model.body_parentid[idx])
            if parent >= 0:
                children.setdefault(parent, []).append(idx)

        stack = [int(body_id)]
        body_ids: set[int] = set()
        while stack:
            bid = int(stack.pop())
            body_ids.add(bid)
            stack.extend(children.get(bid, ()))

        try:
            mesh_type = int(mj.mjtGeom.mjGEOM_MESH)
        except Exception:
            mesh_type = 7

        lo = float("inf")
        hi = float("-inf")
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) not in body_ids:
                continue

            center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3)
            gtype = int(model.geom_type[geom_id])
            mesh_id = int(model.geom_dataid[geom_id]) if hasattr(model, "geom_dataid") else -1
            if (
                gtype == mesh_type
                and mesh_id >= 0
                and hasattr(model, "mesh_vert")
                and hasattr(model, "mesh_vertadr")
                and hasattr(model, "mesh_vertnum")
            ):
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                if count > 0:
                    verts = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
                    xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
                    projected = (center + verts @ xmat.T) @ axis_arr
                    lo = min(lo, float(np.min(projected)))
                    hi = max(hi, float(np.max(projected)))
                    continue

            center_projection = float(np.dot(center, axis_arr))
            half_extent = self._geom_half_extent_along_axis(int(geom_id), axis_arr)
            lo = min(lo, center_projection - half_extent)
            hi = max(hi, center_projection + half_extent)

        width = float(hi - lo)
        if np.isfinite(width) and width > 0.0:
            return width

        try:
            mn, mx = aabb_of_body(self.sim, str(body_name), include_subtree=True)
        except Exception:
            return None
        extents = np.asarray(mx - mn, dtype=np.float64).reshape(3)
        abs_axis = np.abs(axis_arr)
        width = float(np.dot(abs_axis, np.maximum(extents, 0.0)))
        return width if np.isfinite(width) and width > 0.0 else None

    def _caught_object_start_measurement_for_body(self, body_name: str) -> dict[str, float] | None:
        candidates: list[str] = []
        body = str(body_name or "")
        if body:
            candidates.append(body)
        if body and body == str(getattr(self, "_target_body_name", "")):
            candidates.append(str(getattr(self, "_target_catalog_name", "")))
        inverse = getattr(self, "_inverse_catalog_to_body", {}) or {}
        if body in inverse:
            candidates.append(str(inverse[body]))
        if body:
            match = re.search(r"ycb_[A-Za-z0-9_]+", body)
            if match:
                candidates.append(match.group(0))

        for candidate in candidates:
            raw = str(candidate or "").strip()
            if not raw:
                continue
            keys = (raw, canonical_object_name(raw).replace(" ", "_"), canonical_object_name(raw))
            for key in keys:
                measurement = YCB_CAUGHT_OBJECT_MEASUREMENTS.get(str(key))
                if measurement is not None:
                    return measurement
        return None

    def _caught_object_start_gripper_opening_for_body(self, body_name: str) -> float:
        override = self._task_metadata.get("caught_object_start_gripper_opening")
        if override is not None:
            return float(np.clip(float(override), 0.0, 1.0))
        if not self._caught_object_start_fit_gripper_enabled():
            return 0.0

        measurement = self._caught_object_start_measurement_for_body(body_name)
        if measurement is not None:
            return float(np.clip(float(measurement["opening"]), 0.0, 1.0))

        geometry = self._finger_pair_geometry()
        if geometry is None:
            return 0.0

        width = self._body_width_along_axis(str(body_name), np.asarray(geometry["axis"], dtype=np.float32))
        if width is None:
            return 0.0

        clearance = max(0.0, _metadata_float(self._task_metadata, "caught_object_start_gripper_clearance", 0.0))
        compression = max(0.0, _metadata_float(self._task_metadata, "caught_object_start_grip_compression", 0.001))
        desired_gap = max(0.0, float(width + 2.0 * clearance - 2.0 * compression))
        closed_gap = float(geometry["inner_gap"])
        joint_span = float(
            max(
                float(getattr(self.sim, "gripper_joint_max", 0.03))
                - float(getattr(self.sim, "gripper_joint_min", 0.0)),
                1e-6,
            )
        )
        opening = (desired_gap - closed_gap) / (2.0 * joint_span)
        min_opening = _metadata_float(self._task_metadata, "caught_object_start_min_gripper_opening", 0.0)
        max_opening = _metadata_float(self._task_metadata, "caught_object_start_max_gripper_opening", 1.0)
        return float(
            np.clip(
                opening,
                min(float(min_opening), float(max_opening)),
                max(float(min_opening), float(max_opening)),
            )
        )

    def _caught_object_start_hold_center(self) -> np.ndarray | None:
        geometry = self._finger_pair_geometry()
        if geometry is None:
            return None
        center = np.asarray(geometry["center"], dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(center)):
            return None
        return center

    def _caught_object_start_target_position(self) -> np.ndarray:
        hold_center = self._caught_object_start_hold_center()
        if hold_center is not None:
            offset = np.asarray(
                getattr(self, "_caught_object_start_hold_offset", np.zeros((3,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(3)
            return np.asarray(clamp_xyz(hold_center + offset), dtype=np.float32)

        ee_pos = self._get_ee_position().astype(np.float32)
        offset = np.asarray(
            getattr(self, "_caught_object_start_ee_offset", np.zeros((3,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3)
        return np.asarray(clamp_xyz(ee_pos + offset), dtype=np.float32)

    def _maybe_spawn_target_caught_at_ee(
        self,
        *,
        instruction_type: str,
        options: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not self._should_spawn_target_caught_at_ee(instruction_type=instruction_type, options=options):
            return False

        if self.sim is not None:
            self._force_gripper_opening(0.0)

        ee_pos = self._get_ee_position().astype(np.float32)
        target_offset = self._caught_object_start_offset()
        min_height = _metadata_float(
            self._task_metadata,
            "caught_object_start_min_height_above_table",
            0.08,
        )

        gripper_opening = self._caught_object_start_gripper_opening_for_body(self._target_body_name)
        self._force_gripper_opening(gripper_opening)

        hold_center = self._caught_object_start_hold_center()
        if hold_center is not None:
            target_pos = hold_center + target_offset
        else:
            target_pos = ee_pos + target_offset
        target_pos[2] = max(float(target_pos[2]), float(self._support_surface_z + max(0.0, min_height)))
        target_pos = np.asarray(clamp_xyz(target_pos), dtype=np.float32)

        if not self._set_body_position(self._target_body_name, target_pos):
            return False

        self._caught_object_start_active = True
        self._caught_object_start_body = str(self._target_body_name)
        self._caught_object_start_catalog = str(self._target_catalog_name)
        self._caught_object_start_position = target_pos.astype(np.float32)
        self._caught_object_start_ee_offset = (target_pos - ee_pos).astype(np.float32)
        self._caught_object_start_gripper_opening = float(gripper_opening)
        hold_center = self._caught_object_start_hold_center()
        if hold_center is not None:
            self._caught_object_start_hold_offset = (target_pos - hold_center).astype(np.float32)
        else:
            self._caught_object_start_hold_offset = np.zeros((3,), dtype=np.float32)

        warm_steps = max(0, int(round(_metadata_float(self._task_metadata, "caught_object_start_warm_steps", 0.0))))
        if warm_steps > 0 and hasattr(self.sim, "run_simulation_step"):
            for _ in range(warm_steps):
                self._maintain_caught_object_start_pose()
                self.sim.run_simulation_step(capture_frame=False)
                self._maintain_caught_object_start_pose()
            self._force_gripper_opening(gripper_opening)
        return True

    def _caught_object_start_release_opening_threshold(self) -> float:
        threshold = _metadata_float(
            self._task_metadata,
            "caught_object_start_release_opening_threshold",
            _metadata_float(self._task_metadata, "pick_gripper_closed_opening_threshold", 0.010),
        )
        if bool(getattr(self, "_caught_object_start_active", False)):
            hold_opening = float(getattr(self, "_caught_object_start_gripper_opening", 0.0))
            margin = max(0.0, _metadata_float(self._task_metadata, "caught_object_start_release_opening_margin", 0.08))
            if np.isfinite(hold_opening):
                threshold = max(float(threshold), hold_opening + margin)
        return float(np.clip(threshold, 0.0, 1.0))

    def _caught_object_start_gripper_is_closed(self) -> bool:
        threshold = float(max(0.0, self._caught_object_start_release_opening_threshold()))
        try:
            target = float(self._get_gripper_target())
            if np.isfinite(target):
                return bool(target <= threshold)
        except Exception:
            pass
        opening = self._get_gripper_opening()
        if opening is not None and np.isfinite(opening):
            return bool(float(opening) <= threshold)
        return True

    def _maintain_caught_object_start_pose(self) -> bool:
        if not bool(getattr(self, "_caught_object_start_active", False)):
            return False
        if not str(getattr(self, "_caught_object_start_body", "")):
            return False
        if not self._caught_object_start_gripper_is_closed():
            self._caught_object_start_active = False
            return False

        if self._caught_object_start_pin_object_enabled():
            target_pos = self._caught_object_start_target_position()
            if not self._set_body_position(self._caught_object_start_body, target_pos):
                return False
            self._caught_object_start_position = target_pos.astype(np.float32)
            return True

        try:
            self._caught_object_start_position = self._get_body_position(self._caught_object_start_body).astype(np.float32)
        except Exception:
            pass
        return True

    def _sample_scene(self, options: Optional[dict[str, Any]]) -> SceneSpec:
        requested_scene = (options or {}).get("scene")
        if requested_scene is not None:
            requested_scene = str(requested_scene)
            for scene in self.scenes:
                if scene.name == requested_scene:
                    return scene
        required_raw = (options or {}).get("required_objects")
        if required_raw is not None:
            if isinstance(required_raw, str):
                required = {required_raw}
            else:
                required = {str(item) for item in required_raw}
            candidates = [scene for scene in self.scenes if required.issubset({str(obj) for obj in scene.objects})]
            if candidates:
                idx = int(self.np_random.integers(0, len(candidates)))
                return candidates[idx]
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

    def _allowed_instruction_candidates(self) -> tuple[str, ...]:
        if self.instruction_types:
            base_candidates = tuple(str(item) for item in self.instruction_types)
        else:
            base_candidates = tuple(INSTRUCTION_TYPES)

        curriculum_candidates = self._instruction_curriculum_candidates()
        if curriculum_candidates is None:
            return base_candidates

        curriculum_set = set(curriculum_candidates)
        allowed = tuple(item for item in base_candidates if item in curriculum_set)
        if not allowed:
            raise ValueError(
                "Active instruction curriculum stage has no overlap with the allowed instruction types. "
                f"Stage: {list(curriculum_candidates)}; allowed: {list(base_candidates)}"
            )
        return allowed

    def _instruction_curriculum_candidates(self) -> tuple[str, ...] | None:
        raw = getattr(self, "_task_metadata", {}).get("instruction_curriculum")
        if raw is None:
            return None
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError("Task metadata `instruction_curriculum` must be a list of stage mappings.")

        episode_index = int(getattr(self, "_episode_index", 0))
        cumulative_episodes = 0
        fallback: tuple[str, ...] | None = None
        for stage in raw:
            if not isinstance(stage, dict):
                raise ValueError("Each instruction curriculum stage must be a mapping.")

            stage_types_raw = stage.get("instruction_types")
            if stage_types_raw is None:
                raise ValueError("Each instruction curriculum stage must include `instruction_types`.")
            if isinstance(stage_types_raw, str):
                stage_types_raw = [stage_types_raw]
            if isinstance(stage_types_raw, (bytes, str)) or not isinstance(stage_types_raw, Sequence):
                raise ValueError("Instruction curriculum `instruction_types` must be a list of names.")
            stage_candidates = _dedupe_names([str(item) for item in stage_types_raw])
            if not stage_candidates:
                continue
            fallback = stage_candidates

            if "until_episode" in stage:
                until_episode = int(stage["until_episode"])
                if episode_index < until_episode:
                    return stage_candidates
                continue

            if "episodes" in stage:
                cumulative_episodes += max(0, int(stage["episodes"]))
                if episode_index < cumulative_episodes:
                    return stage_candidates
                continue

            return stage_candidates

        return fallback

    def _sample_instruction_type(self, options: Optional[dict[str, Any]] = None) -> str:
        candidates = self._allowed_instruction_candidates()
        requested = (options or {}).get("instruction_type")
        if requested is not None:
            requested_type = str(requested).strip()
            if requested_type not in candidates:
                raise ValueError(
                    f"Requested instruction_type {requested_type!r} is not in the allowed set {list(candidates)}."
                )
            return requested_type

        if len(candidates) <= 1 or self.instruction_sampling != "uniform_cycle":
            return str(candidates[int(self.np_random.integers(0, len(candidates)))])

        cycle = [item for item in self._instruction_cycle if item in candidates]
        if not cycle:
            order = np.asarray(candidates, dtype=object)
            perm = self.np_random.permutation(len(order))
            cycle = [str(order[idx]) for idx in perm.tolist()]
        selected = str(cycle.pop(0))
        self._instruction_cycle = cycle
        return selected

    def _select_target_object(self, scene: SceneSpec) -> tuple[str, str]:
        preferred_catalogs: list[str] = []
        if scene.target_object:
            preferred_catalogs.append(str(scene.target_object))
        preferred_catalogs.extend(str(name) for name in scene.objects)

        resolved_catalogs = [name for name in preferred_catalogs if name in self._catalog_to_body]
        if resolved_catalogs:
            chosen_catalog = (
                str(scene.target_object)
                if scene.target_object in self._catalog_to_body
                else str(resolved_catalogs[int(self.np_random.integers(0, len(resolved_catalogs)))])
            )
            return chosen_catalog, str(self._catalog_to_body[chosen_catalog])

        if self._object_body_names:
            chosen_body = str(self._object_body_names[int(self.np_random.integers(0, len(self._object_body_names)))])
            chosen_catalog = next(
                (catalog for catalog, body in self._catalog_to_body.items() if body == chosen_body),
                chosen_body,
            )
            return chosen_catalog, chosen_body

        if preferred_catalogs:
            chosen_catalog = str(preferred_catalogs[0])
            return chosen_catalog, str(self._catalog_to_body.get(chosen_catalog, ""))
        return "", ""

    def _resolve_requested_catalog(self, raw: Any) -> tuple[str, str]:
        if raw is None:
            return "", ""
        catalog = str(raw).strip()
        if not catalog:
            return "", ""
        body = str(self._catalog_to_body.get(catalog, ""))
        return catalog, body

    def _metadata_catalog_pool(self, *keys: str, default: Sequence[str] = ()) -> tuple[str, ...]:
        for key in keys:
            pool = _metadata_name_list(self._task_metadata, key)
            if pool:
                return pool
        return _dedupe_names(default)

    @staticmethod
    def _catalogs_in_pool(catalogs: Sequence[str], pool: Sequence[str]) -> list[str]:
        allowed = {str(name) for name in pool}
        if not allowed:
            return []
        return [str(name) for name in catalogs if str(name) in allowed]

    def _catchable_scene_catalogs(self, scene_catalogs: Sequence[str]) -> list[str]:
        pool = self._metadata_catalog_pool(
            "catchable_object_pool",
            "grippable_object_pool",
            default=DEFAULT_CATCHABLE_OBJECTS,
        )
        return self._catalogs_in_pool(scene_catalogs, pool)

    def _container_scene_catalogs(self, scene_catalogs: Sequence[str]) -> list[str]:
        pool = self._metadata_catalog_pool("container_object_pool", default=DEFAULT_CONTAINER_OBJECTS)
        candidates = self._catalogs_in_pool(scene_catalogs, pool)
        if candidates:
            return candidates
        return [name for name in scene_catalogs if "plate" in name.lower() or "bowl" in name.lower()]

    def _choose_catalog(self, candidates: Sequence[str], *, fallback: Sequence[str] = ()) -> tuple[str, str]:
        pool = [str(name) for name in candidates if str(name) in self._catalog_to_body]
        if not pool:
            pool = [str(name) for name in fallback if str(name) in self._catalog_to_body]
        if not pool:
            return "", ""
        chosen = str(pool[int(self.np_random.integers(0, len(pool)))])
        return chosen, str(self._catalog_to_body.get(chosen, ""))

    def _select_instruction_objects(
        self,
        scene: SceneSpec,
        *,
        instruction_type: str,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str, str, str, str, str]:
        options = dict(options or {})
        requested_target = self._resolve_requested_catalog(
            options.get("target_object", options.get("target_catalog_name"))
        )
        requested_reference = self._resolve_requested_catalog(
            options.get("reference_object", options.get("reference_catalog_name"))
        )
        requested_second_reference = self._resolve_requested_catalog(
            options.get("second_reference_object", options.get("second_reference_catalog_name"))
        )

        scene_catalogs = [str(name) for name in scene.objects if str(name) in self._catalog_to_body]
        if not scene_catalogs:
            target_catalog, target_body = self._select_target_object(scene)
            return target_catalog, target_body, "", "", "", ""

        def _not_selected(*selected: str) -> list[str]:
            blocked = {str(item) for item in selected if str(item)}
            return [name for name in scene_catalogs if name not in blocked]

        target_catalog, target_body = requested_target
        reference_catalog, reference_body = requested_reference
        second_reference_catalog, second_reference_body = requested_second_reference

        if not target_catalog:
            if str(instruction_type) in CATCHABLE_TARGET_INSTRUCTION_TYPES:
                catchable = self._catchable_scene_catalogs(scene_catalogs)
                target_catalog, target_body = self._choose_catalog(catchable)
                if not target_catalog:
                    raise ValueError(
                        f"Instruction {instruction_type!r} requires a catchable target object. "
                        f"Scene {scene.name!r} contains {scene_catalogs}; catchable pool is "
                        f"{list(self._metadata_catalog_pool('catchable_object_pool', 'grippable_object_pool', default=DEFAULT_CATCHABLE_OBJECTS))}."
                    )
            else:
                target_catalog, target_body = self._select_target_object(scene)
                if target_catalog not in self._catalog_to_body:
                    target_catalog, target_body = self._choose_catalog(scene_catalogs)

        if str(instruction_type) == "put_into_plate" and not reference_catalog:
            container_like = self._container_scene_catalogs(scene_catalogs)
            reference_catalog, reference_body = self._choose_catalog(
                [name for name in container_like if name != target_catalog],
                fallback=[name for name in container_like if name != target_catalog],
            )
            if not reference_catalog:
                raise ValueError(
                    f"Instruction {instruction_type!r} requires a bowl/plate reference object. "
                    f"Scene {scene.name!r} contains {scene_catalogs}; container pool is "
                    f"{list(self._metadata_catalog_pool('container_object_pool', default=DEFAULT_CONTAINER_OBJECTS))}."
                )

        elif str(instruction_type) in {
            "move_left_of_object",
            "move_right_of_object",
            "put_in_front_of_object",
            "put_behind_object",
        } and not reference_catalog:
            reference_catalog, reference_body = self._choose_catalog(_not_selected(target_catalog), fallback=scene_catalogs)

        elif str(instruction_type) == "move_between_objects":
            if not reference_catalog:
                reference_catalog, reference_body = self._choose_catalog(_not_selected(target_catalog), fallback=scene_catalogs)
            if not second_reference_catalog:
                second_reference_catalog, second_reference_body = self._choose_catalog(
                    _not_selected(target_catalog, reference_catalog),
                    fallback=_not_selected(target_catalog),
                )

        return (
            str(target_catalog),
            str(target_body),
            str(reference_catalog),
            str(reference_body),
            str(second_reference_catalog),
            str(second_reference_body),
        )

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

    def _body_position_or_none(self, body_name: str) -> np.ndarray | None:
        if not body_name:
            return None
        try:
            return self._get_task_body_position(str(body_name)).astype(np.float32)
        except Exception:
            return None

    def _current_manipulated_object_position(self, *, default: np.ndarray | None = None) -> np.ndarray:
        pos = self._body_position_or_none(self._target_body_name)
        if pos is not None:
            return pos
        if default is None:
            default = self._goal_position
        return np.asarray(default, dtype=np.float32).reshape(-1)[:3].astype(np.float32)

    def _reference_object_position(self, *, second: bool = False, default: np.ndarray | None = None) -> np.ndarray:
        body_name = self._second_reference_body_name if second else self._reference_body_name
        pos = self._body_position_or_none(body_name)
        if pos is not None:
            return pos
        if default is None:
            default = self._goal_position
        return np.asarray(default, dtype=np.float32).reshape(-1)[:3].astype(np.float32)

    def _compute_relation_goal_position(self, *, spec, target_pos: np.ndarray) -> np.ndarray:
        instruction_type = str(spec.instruction_type)
        if instruction_type == "put_into_plate":
            ref_pos = self._reference_object_position(default=target_pos)
            goal = ref_pos.copy()
            goal[2] = max(float(goal[2]), float(self._support_surface_z + 0.02))
            return goal.astype(np.float32)

        if instruction_type in {"move_left_of_object", "move_right_of_object"}:
            ref_pos = self._reference_object_position(default=target_pos)
            offset = float(self._task_metadata.get("relation_left_right_offset", 0.08))
            sign = -1.0 if instruction_type == "move_left_of_object" else 1.0
            goal = ref_pos.copy()
            goal[0] += sign * offset
            return np.asarray(clamp_xyz(goal), dtype=np.float32)

        if instruction_type in {"put_in_front_of_object", "put_behind_object"}:
            ref_pos = self._reference_object_position(default=target_pos)
            offset = float(
                self._task_metadata.get(
                    "relation_front_behind_offset",
                    self._task_metadata.get("relation_left_right_offset", 0.08),
                )
            )
            sign = 1.0 if instruction_type == "put_in_front_of_object" else -1.0
            goal = ref_pos.copy()
            goal[1] += sign * offset
            return np.asarray(clamp_xyz(goal), dtype=np.float32)

        if instruction_type == "move_between_objects":
            ref_a = self._reference_object_position(default=target_pos)
            ref_b = self._reference_object_position(second=True, default=target_pos)
            goal = 0.5 * (ref_a + ref_b)
            return np.asarray(clamp_xyz(goal), dtype=np.float32)

        if instruction_type in {"push_left", "push_right"}:
            distance = float(self._task_metadata.get("push_success_displacement", 0.08))
            sign = -1.0 if instruction_type == "push_left" else 1.0
            goal = np.asarray(target_pos, dtype=np.float32).copy()
            initial = (
                np.asarray(self._reward_state.initial_obj_pos, dtype=np.float32)
                if self._reward_state is not None
                else goal
            )
            goal = initial.copy()
            goal[0] += sign * distance
            return np.asarray(clamp_xyz(goal), dtype=np.float32)

        return target_pos.astype(np.float32)

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
        elif instruction_uses_target_object(spec.instruction_type) and self._target_body_name:
            target_pos = self._get_body_position(self._target_body_name).astype(np.float32)
            if spec.instruction_type in {
                "put_into_plate",
                "move_left_of_object",
                "move_right_of_object",
                "put_in_front_of_object",
                "put_behind_object",
                "move_between_objects",
                "push_left",
                "push_right",
            }:
                goal = self._compute_relation_goal_position(spec=spec, target_pos=target_pos)
            else:
                goal = target_pos
        else:
            center = self._goal_center()
            lateral_offset = float(self._task_metadata.get("lateral_goal_offset", spec.target_displacement))
            goal = center.copy()
            if spec.instruction_type == "move_left":
                goal[0] -= lateral_offset
            elif spec.instruction_type == "move_right":
                goal[0] += lateral_offset
            elif spec.instruction_type == "move_top":
                goal[1] += lateral_offset
            elif spec.instruction_type == "move_bottom":
                goal[1] -= lateral_offset
            elif spec.instruction_type in {"move_up", "move_down", "move_center"}:
                # Center-anchored instructions all share the workspace center target.
                pass
            elif spec.instruction_type != "move_center":
                raise RuntimeError(f"Unsupported instruction type for goal generation: {spec.instruction_type}")
            goal = np.asarray(clamp_xyz(goal), dtype=np.float32)

        if instruction_uses_target_object(spec.instruction_type) and requested_goal is None:
            return goal.astype(np.float32)
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
        self._last_wrapper_reused_from_cache = False
        if self.use_wrapper_cache and bool(getattr(self, "reuse_existing_wrapper_variants", False)):
            existing_candidates = _candidate_existing_wrapper_paths(
                self.wrapper_dir,
                scene_name=scene.name,
                object_names=scene.objects,
            )
            invalid_paths = set(getattr(self, "_invalid_wrapper_paths", set()))
            if invalid_paths:
                existing_candidates = [
                    path for path in existing_candidates
                    if path.resolve() not in invalid_paths
                ]
            if existing_candidates:
                chosen_idx = int(self.np_random.integers(0, len(existing_candidates)))
                chosen_wrapper = existing_candidates[chosen_idx].resolve()
                self._last_wrapper_reused_from_cache = True
                print(f"[env] Reusing cached wrapper variant: {chosen_wrapper}", flush=True)
                self._desk_texture_name = ""
                return chosen_wrapper

        build_wrapper_if_needed, list_wrapper_bundle_paths = _import_wrapper_builder()
        default_ee_start = self._default_ee_start()
        episode_ee_start = default_ee_start if ee_start is None else _coerce_ee_start(ee_start)
        reuse_cached_wrapper_across_ee_starts = bool(self.use_wrapper_cache and self.randomize_ee_start)
        wrapper_ee_start = default_ee_start if reuse_cached_wrapper_across_ee_starts else episode_ee_start
        unique_wrapper_bundle = bool(not np.allclose(wrapper_ee_start, default_ee_start, atol=1e-9))
        wrapper_out = None
        use_cache = bool(self.use_wrapper_cache and not unique_wrapper_bundle)
        if (not use_cache) or self.wrapper_cleanup:
            wrapper_out = self._temporary_wrapper_path(scene=scene)
            use_cache = False

        wrapper_xml = build_wrapper_if_needed(
            scene_name=scene.name,
            object_names=list(scene.objects),
            scene_z=self.defaults.get("scene_z", -0.85),
            ee_start=tuple(float(x) for x in wrapper_ee_start),
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
            variant_tag = _desk_texture_variant_tag(wrapper_xml, chosen_texture)
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
        stamp = int(time.time_ns())
        bundle_dir = _wrapper_bundle_dir(self.wrapper_dir, scene_name=scene.name, object_names=scene.objects)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        prefix = _wrapper_cache_prefix(scene.name, scene.objects)
        return bundle_dir / f"{prefix}__rltmp_{stamp}.xml"

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
        score_threshold = _metadata_float(self._task_metadata, "catch_score_threshold", 0.30)
        distance_threshold = _metadata_float(self._task_metadata, "catch_distance_threshold", 0.055)
        is_caught = bool(
            gripper_closed
            and best_score >= float(score_threshold)
            and best_dist <= float(distance_threshold)
        )
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

    def _read_named_body_position(self, body_name: str, body_id: int) -> np.ndarray:
        data = self.sim.data
        for attr in ("body_xpos", "xpos"):
            positions = getattr(data, attr, None)
            if positions is None:
                continue
            try:
                return np.asarray(positions[body_id], dtype=np.float32).copy()
            except Exception:
                continue

        body_accessor = getattr(data, "body", None)
        if callable(body_accessor):
            for key in (body_name, body_id):
                try:
                    return np.asarray(body_accessor(key).xpos, dtype=np.float32).copy()
                except Exception:
                    continue

        raise AttributeError(
            "MuJoCo data object does not expose a compatible body position accessor. "
            "Tried `body_xpos`, `xpos`, and `body(...).xpos`."
        )

    def _get_body_position(self, body_name: str) -> np.ndarray:
        bid = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_BODY, body_name)
        if bid == -1:
            raise RuntimeError(f"Body '{body_name}' not found in MuJoCo model.")
        return self._read_named_body_position(body_name, bid)

    def _get_task_body_position(self, body_name: str) -> np.ndarray:
        pos = self._get_body_position(body_name)
        catalog = str(getattr(self, "_inverse_catalog_to_body", {}).get(str(body_name), ""))
        is_container = bool("plate" in catalog.lower() or "bowl" in catalog.lower())
        if catalog:
            try:
                is_container = is_container or catalog in self._container_scene_catalogs([catalog])
            except Exception:
                pass
        if is_container:
            try:
                mn, mx = aabb_of_body(self.sim, str(body_name), include_subtree=True)
                center = 0.5 * (np.asarray(mn, dtype=np.float32) + np.asarray(mx, dtype=np.float32))
                if np.all(np.isfinite(center)):
                    return center.astype(np.float32)
            except Exception:
                pass
        return pos

    def _set_body_position(self, body_name: str, xyz: Sequence[float] | np.ndarray) -> bool:
        if not body_name or self.sim is None:
            return False
        bid = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_BODY, str(body_name))
        if bid == -1:
            return False

        target = np.asarray(xyz, dtype=float).reshape(-1)
        if target.size < 3 or not np.all(np.isfinite(target[:3])):
            return False
        target = target[:3].astype(float)

        model = self.sim.model
        data = self.sim.data
        jnum = int(model.body_jntnum[bid])
        jadr = int(model.body_jntadr[bid])
        for offset in range(jnum):
            jid = jadr + offset
            if model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE:
                continue
            qadr = int(model.jnt_qposadr[jid])
            data.qpos[qadr : qadr + 3] = target
            if hasattr(data, "qvel") and hasattr(model, "jnt_dofadr"):
                dofadr = int(model.jnt_dofadr[jid])
                if 0 <= dofadr < len(data.qvel):
                    data.qvel[dofadr : min(dofadr + 6, len(data.qvel))] = 0.0
            mj.mj_forward(model, data)
            return True

        positions = getattr(data, "xpos", None)
        if positions is not None:
            try:
                positions[bid] = target
                mj.mj_forward(model, data)
                return True
            except Exception:
                return False
        return False

    def _current_target_reference_position(self) -> np.ndarray:
        if (
            self._instruction_spec is not None
            and instruction_uses_target_object(self._instruction_spec.instruction_type)
            and self._target_body_name
        ):
            try:
                target_pos = self._get_body_position(self._target_body_name).astype(np.float32)
                if self._instruction_spec.instruction_type in {
                    "put_into_plate",
                    "move_left_of_object",
                    "move_right_of_object",
                    "put_in_front_of_object",
                    "put_behind_object",
                    "move_between_objects",
                    "push_left",
                    "push_right",
                }:
                    return self._compute_relation_goal_position(
                        spec=self._instruction_spec,
                        target_pos=target_pos,
                    )
                return target_pos
            except Exception:
                pass
        return self._goal_position.astype(np.float32)

    def _get_gripper_opening(self) -> float | None:
        if hasattr(self.sim, "get_gripper_opening"):
            try:
                opening = float(self.sim.get_gripper_opening())
            except Exception:
                return None
            return opening if np.isfinite(opening) else None
        return None

    def _get_gripper_target(self) -> float:
        if hasattr(self.sim, "get_gripper_target"):
            try:
                target = float(self.sim.get_gripper_target())
                if np.isfinite(target):
                    return float(np.clip(target, 0.0, 1.0))
            except Exception:
                pass
        opening = self._get_gripper_opening()
        if opening is not None and np.isfinite(opening):
            return float(np.clip(opening, 0.0, 1.0))
        return 1.0

    def _set_gripper_target(self, target_01: float) -> None:
        target = float(np.clip(target_01, 0.0, 1.0))
        if hasattr(self.sim, "set_gripper"):
            self.sim.set_gripper(target)
        elif target <= 0.0 and hasattr(self.sim, "close_gripper"):
            self.sim.close_gripper()
        elif target >= 1.0 and hasattr(self.sim, "open_gripper"):
            self.sim.open_gripper()

    def _force_gripper_opening(self, target_01: float) -> None:
        self._set_gripper_target(target_01)
        if self.sim is None or not hasattr(self.sim, "data") or not hasattr(self.sim, "model"):
            return
        target = float(np.clip(target_01, 0.0, 1.0))
        joint_min = float(getattr(self.sim, "gripper_joint_min", 0.0))
        joint_max = float(getattr(self.sim, "gripper_joint_max", 0.03))
        joint_pos = joint_min + target * max(joint_max - joint_min, 0.0)
        try:
            joint_ids: list[int] = []
            for joint_name in ("finger_l", "finger_r"):
                jid = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
                if jid != -1:
                    joint_ids.append(int(jid))
            fallback_joint_id = getattr(self.sim, "jnt_finger_l", None)
            if fallback_joint_id is not None and int(fallback_joint_id) not in joint_ids:
                joint_ids.append(int(fallback_joint_id))

            if not joint_ids:
                qadr = getattr(self.sim, "jnt_finger_l_qadr", None)
                if qadr is None:
                    return
                self.sim.data.qpos[int(qadr)] = float(joint_pos)

            for joint_id in joint_ids:
                joint_qadr = int(self.sim.model.jnt_qposadr[int(joint_id)])
                self.sim.data.qpos[joint_qadr] = float(joint_pos)
                if hasattr(self.sim.data, "qvel") and hasattr(self.sim.model, "jnt_dofadr"):
                    dofadr = int(self.sim.model.jnt_dofadr[int(joint_id)])
                    if 0 <= dofadr < len(self.sim.data.qvel):
                        self.sim.data.qvel[dofadr] = 0.0
            mj.mj_forward(self.sim.model, self.sim.data)
        except Exception:
            return

    def _is_gripper_closed(self, opening: float | None) -> bool:
        if opening is None or not np.isfinite(opening):
            return bool(self._reward_state.gripper_closed) if self._reward_state is not None else False
        grip_min = float(getattr(self.sim, "gripper_min", 0.0))
        grip_max = float(getattr(self.sim, "gripper_max", 1.0))
        threshold = grip_min + 0.35 * max(grip_max - grip_min, 1e-6)
        if bool(getattr(self, "_caught_object_start_active", False)):
            threshold = max(threshold, self._caught_object_start_release_opening_threshold())
        return bool(float(opening) <= threshold)

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

    def _current_goal_motion_direction(
        self,
        *,
        ee_pos: Optional[np.ndarray] = None,
        goal_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._instruction_spec is None:
            return self._goal_motion_direction.astype(np.float32)
        if not instruction_uses_target_object(self._instruction_spec.instruction_type):
            return self._goal_motion_direction.astype(np.float32)

        if self._reward_state is not None and self._reward_state.grasped:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        live_ee = self._get_ee_position() if ee_pos is None else np.asarray(ee_pos, dtype=np.float32)
        live_goal = self._current_target_reference_position() if goal_pos is None else np.asarray(goal_pos, dtype=np.float32)
        return self._compute_goal_motion_direction(
            initial_ee_pos=live_ee,
            goal_pos=live_goal,
            instruction_direction=self._instruction_spec.direction,
        )

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
        gripper_target = self._get_gripper_target() + self._last_gripper_cmd * float(self.action_step_gripper)
        if (
            bool(getattr(self, "_caught_object_start_active", False))
            and self._caught_object_start_fit_gripper_enabled()
        ):
            gripper_target = max(gripper_target, float(getattr(self, "_caught_object_start_gripper_opening", 0.0)))
        self._set_gripper_target(gripper_target)

        total_sim_steps = 1 + int(self.hold_steps)
        for sub_idx in range(total_sim_steps):
            capture = bool(self.capture_frames and sub_idx == (total_sim_steps - 1))
            self.sim.run_simulation_step(capture_frame=capture)
            self._maintain_caught_object_start_pose()

    def _get_obs(self) -> dict[str, np.ndarray]:
        ee_pos = self._get_ee_position()
        target_pos = self._current_target_reference_position()

        obj_pos = np.zeros((self.max_objects, 3), dtype=np.float32)
        obj_mask = np.zeros((self.max_objects,), dtype=np.float32)
        for i, body_name in enumerate(self._object_body_names[: self.max_objects]):
            try:
                obj_pos[i] = self._get_body_position(body_name)
                obj_mask[i] = 1.0
            except Exception:
                continue

        onehot = instruction_to_onehot(self._instruction_spec)
        goal_direction = self._current_goal_motion_direction(ee_pos=ee_pos, goal_pos=target_pos)

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
        live_goal_position = self._current_target_reference_position()
        live_goal_direction = self._current_goal_motion_direction(goal_pos=live_goal_position)
        target_object_position = self._current_manipulated_object_position(default=live_goal_position)
        reference_object_position = self._reference_object_position(default=live_goal_position)
        second_reference_object_position = self._reference_object_position(second=True, default=live_goal_position)
        return {
            "env_instance_id": int(self._env_instance_id),
            "scene": self._scene_name,
            "episode_index": int(self._episode_index),
            "scene_objects": list(self._scene_catalog_objects),
            "allowed_objects": list(self.allowed_objects),
            "scene_object_pool": list(self.scene_object_pool),
            "target_object_pool": list(self.target_object_pool),
            "distractor_object_pool": list(self.distractor_object_pool),
            "target_object_catalog": self._target_catalog_name,
            "target_object_body": self._target_body_name,
            "target_object_position_actual": [float(x) for x in target_object_position.tolist()],
            "caught_object_start": bool(getattr(self, "_caught_object_start_active", False)),
            "caught_object_start_body": str(getattr(self, "_caught_object_start_body", "")),
            "caught_object_start_catalog": str(getattr(self, "_caught_object_start_catalog", "")),
            "caught_object_start_position": [
                float(x)
                for x in np.asarray(
                    getattr(self, "_caught_object_start_position", np.zeros((3,), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(3).tolist()
            ],
            "caught_object_start_gripper_opening": float(
                getattr(self, "_caught_object_start_gripper_opening", 0.0)
            ),
            "caught_object_start_release_opening_threshold": float(
                self._caught_object_start_release_opening_threshold()
                if bool(getattr(self, "_caught_object_start_active", False))
                else _metadata_float(
                    self._task_metadata,
                    "caught_object_start_release_opening_threshold",
                    _metadata_float(self._task_metadata, "pick_gripper_closed_opening_threshold", 0.010),
                )
            ),
            "reference_object_catalog": self._reference_catalog_name,
            "reference_object_body": self._reference_body_name,
            "reference_object_position": [float(x) for x in reference_object_position.tolist()],
            "second_reference_object_catalog": self._second_reference_catalog_name,
            "second_reference_object_body": self._second_reference_body_name,
            "second_reference_object_position": [float(x) for x in second_reference_object_position.tolist()],
            "language_instruction": self._instruction_spec.text,
            "instruction_type": self._instruction_spec.instruction_type,
            "goal_position": [float(x) for x in live_goal_position.tolist()],
            "goal_motion_direction": [float(x) for x in live_goal_direction.tolist()],
            "goal_region": dict(self._goal_region),
            "goal_relation": self._goal_relation or "",
            "dense_reward_terms": dict(self._dense_reward_terms),
            "gripper_command": float(self._last_gripper_cmd),
            "gripper_opening": float(self._get_gripper_opening() or 0.0),
            "gripper_target": float(self._get_gripper_target()),
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
            "record_trajectory": bool(self.record_trajectory),
            "action_step_gripper": float(self.action_step_gripper),
            "curriculum_mode": str(getattr(self, "_curriculum_mode", "")),
            "curriculum_shell": (
                -1
                if getattr(self, "_curriculum_shell", None) is None
                else int(getattr(self, "_curriculum_shell"))
            ),
            "curriculum_instruction_id": str(
                getattr(self, "_curriculum_reset_info", {}).get("curriculum_instruction_id", "")
            ),
        }
