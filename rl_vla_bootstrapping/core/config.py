from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .specs import (
    AssetsSpec,
    EmbodimentSpec,
    EvaluationSpec,
    PolicySpec,
    ProjectConfig,
    ProjectSpec,
    ReposSpec,
    RemoteSpec,
    SceneBuilderSpec,
    TaskSpec,
    TrainingSpec,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. "
            "Install `PyYAML` or use JSON/TOML."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _load_toml(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as handle:
        data = tomllib.load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_project_config(config_path: str | Path) -> ProjectConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = _load_yaml(path)
    elif suffix == ".toml":
        raw = _load_toml(path)
    elif suffix == ".json":
        raw = _load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return ProjectConfig(
        config_path=path,
        project=ProjectSpec.from_mapping(raw.get("project")),
        repos=ReposSpec.from_mapping(raw.get("repos")),
        remote=RemoteSpec.from_mapping(raw.get("remote")),
        assets=AssetsSpec.from_mapping(raw.get("assets")),
        embodiment=EmbodimentSpec.from_mapping(raw.get("embodiment")),
        task=TaskSpec.from_mapping(raw.get("task")),
        simulation=SceneBuilderSpec.from_mapping(raw.get("simulation")),
        policy=PolicySpec.from_mapping(raw.get("policy")),
        training=TrainingSpec.from_mapping(raw.get("training")),
        evaluation=EvaluationSpec.from_mapping(raw.get("evaluation")),
    )
