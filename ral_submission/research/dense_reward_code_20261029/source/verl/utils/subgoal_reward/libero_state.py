from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Any, Mapping

import numpy as np


@dataclass
class LiberoState:
    task_name: str | None = None
    instruction: str | None = None
    gripper_position: np.ndarray | None = None
    gripper_open: float | None = None
    object_position: np.ndarray | None = None
    target_position: np.ndarray | None = None
    object_in_gripper: bool | None = None
    success: bool = False
    raw_info: dict[str, Any] = field(default_factory=dict)


class LiberoStateExtractor:
    """Best-effort LIBERO / robosuite state extraction."""

    def extract(
        self,
        env: Any = None,
        obs: Mapping[str, Any] | None = None,
        info: Mapping[str, Any] | None = None,
        task_metadata: Mapping[str, Any] | None = None,
        done: bool = False,
    ) -> LiberoState:
        obs = obs or {}
        info = dict(info or {})
        task_metadata = task_metadata or {}

        gripper_position = self._vec3(obs.get("robot0_eef_pos"))
        gripper_open = self._gripper_open(obs)
        object_name, target_name = self._task_object_and_target(task_metadata)
        object_position = self._object_position(obs, gripper_position, task_metadata, object_name)
        target_position = self._target_position(obs, task_metadata, target_name, env)
        success = self._success(env, info, done)

        return LiberoState(
            task_name=self._str_from(task_metadata, "task_name", "task_suite_name"),
            instruction=self._str_from(task_metadata, "instruction", "task_description", "language"),
            gripper_position=gripper_position,
            gripper_open=gripper_open,
            object_position=object_position,
            target_position=target_position,
            object_in_gripper=self._object_in_gripper(obs, info, object_position, gripper_position, gripper_open),
            success=success,
            raw_info=info,
        )

    def _str_from(self, mapping: Mapping[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return str(value)
        return None

    def _vec3(self, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return None
        return arr[:3]

    def _gripper_open(self, obs: Mapping[str, Any]) -> float | None:
        qpos = obs.get("robot0_gripper_qpos")
        if qpos is None:
            qpos = obs.get("gripper_qpos")
        if qpos is None:
            return None
        try:
            value = float(np.abs(np.asarray(qpos, dtype=np.float32).reshape(-1)).mean())
        except Exception:
            return None
        return float(np.clip(value / 0.04, 0.0, 1.0))

    def _candidate_pos_keys(self, obs: Mapping[str, Any]) -> list[str]:
        blocked = ("robot", "eef", "gripper", "camera", "image", "quat")
        keys = []
        for key, value in obs.items():
            lower = key.lower()
            if not lower.endswith("_pos"):
                continue
            if any(token in lower for token in blocked):
                continue
            if self._vec3(value) is not None:
                keys.append(key)
        return keys

    def _object_position(
        self,
        obs: Mapping[str, Any],
        gripper_position: np.ndarray | None,
        task_metadata: Mapping[str, Any],
        object_name: str | None = None,
    ) -> np.ndarray | None:
        if object_name:
            for key in self._position_key_candidates(str(object_name)):
                value = self._vec3(obs.get(key))
                if value is not None:
                    return value

        candidates = [(key, self._vec3(obs[key])) for key in self._candidate_pos_keys(obs)]
        candidates = [(key, value) for key, value in candidates if value is not None]
        if not candidates:
            object_state = np.asarray(obs.get("object-state", []), dtype=np.float32).reshape(-1)
            if object_state.size >= 3:
                return object_state[:3]
            return None

        if gripper_position is None:
            return candidates[0][1]
        return min(candidates, key=lambda item: float(np.linalg.norm(item[1] - gripper_position)))[1]

    def _target_position(
        self,
        obs: Mapping[str, Any],
        task_metadata: Mapping[str, Any],
        target_name: str | None = None,
        env: Any = None,
    ) -> np.ndarray | None:
        if target_name:
            for key in self._position_key_candidates(str(target_name)):
                value = self._vec3(obs.get(key))
                if value is not None:
                    return value

        for key in ("target_position", "target_pos", "goal_position", "goal_pos"):
            value = self._vec3(task_metadata.get(key))
            if value is not None:
                return value
            value = self._vec3(obs.get(key))
            if value is not None:
                return value

        if env is not None:
            sim = getattr(env, "sim", None)
            data = getattr(sim, "data", None)
            get_site_xpos = getattr(data, "get_site_xpos", None)
            if callable(get_site_xpos):
                if target_name:
                    try:
                        value = self._vec3(get_site_xpos(str(target_name)))
                    except Exception:
                        value = None
                    if value is not None:
                        return value
                site_names = getattr(env, "object_sites_dict", {})
                for site_name in site_names:
                    if any(token in str(site_name).lower() for token in ("target", "goal")):
                        try:
                            value = self._vec3(get_site_xpos(str(site_name)))
                        except Exception:
                            continue
                        if value is not None:
                            return value

        target_tokens = ("target", "goal", "zone", "region", "site")
        for key, value in obs.items():
            lower = key.lower()
            if any(token in lower for token in target_tokens):
                vec = self._vec3(value)
                if vec is not None:
                    return vec
        return None

    def _task_object_and_target(self, task_metadata: Mapping[str, Any]) -> tuple[str | None, str | None]:
        object_name = self._str_from(task_metadata, "object_name", "obj_name")
        target_name = self._str_from(task_metadata, "target_name", "target_object_name", "goal_name")
        if object_name and target_name:
            return object_name, target_name

        bddl_path = self._str_from(task_metadata, "bddl_file_path", "bddl_file")
        if bddl_path:
            bddl_object, bddl_target = _parse_bddl_object_and_target(bddl_path)
            object_name = object_name or bddl_object
            target_name = target_name or bddl_target

        return object_name, target_name

    def _position_key_candidates(self, name: str) -> list[str]:
        name = name.strip()
        if not name:
            return []
        normalized = name.lower().replace(" ", "_").replace("-", "_")
        candidates = [f"{name}_pos", f"{normalized}_pos", name, normalized]
        deduped = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _object_in_gripper(
        self,
        obs: Mapping[str, Any],
        info: Mapping[str, Any],
        object_position: np.ndarray | None,
        gripper_position: np.ndarray | None,
        gripper_open: float | None,
    ) -> bool | None:
        for source in (info, obs):
            for key in ("object_in_gripper", "in_gripper", "grasped", "grasp_success", "contact"):
                if key in source:
                    return bool(source[key])
        if object_position is None or gripper_position is None or gripper_open is None:
            return None
        return bool(np.linalg.norm(object_position - gripper_position) <= 0.04 and gripper_open < 0.5)

    def _success(self, env: Any, info: Mapping[str, Any], done: bool) -> bool:
        for key in ("success", "is_success", "complete"):
            if key in info:
                return bool(info[key])
        if env is not None:
            for checker_name in ("check_success", "_check_success"):
                checker = getattr(env, checker_name, None)
                if callable(checker):
                    try:
                        return bool(checker())
                    except Exception:
                        pass
        return bool(done)


@lru_cache(maxsize=256)
def _parse_bddl_object_and_target(bddl_path: str) -> tuple[str | None, str | None]:
    try:
        with open(bddl_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None, None

    obj_interest = re.search(r"\(:obj_of_interest\s+([^)]+)\)", text, flags=re.IGNORECASE | re.DOTALL)
    if obj_interest:
        names = re.findall(r"[A-Za-z0-9_]+", obj_interest.group(1))
        if len(names) >= 2:
            return names[0], names[1]

    goal = re.search(r"\(:goal\s+\(And\s+\((?:On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", text, flags=re.IGNORECASE)
    if goal:
        return goal.group(1), goal.group(2)

    return None, None
