from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from rl_vla_bootstrapping.core.imports import call_with_supported_kwargs, import_object
from rl_vla_bootstrapping.core.specs import ProjectConfig


def preview_selection(config: ProjectConfig) -> tuple[str | None, list[str]]:
    scene = config.simulation.preview_scene
    objects = list(config.simulation.preview_objects)
    if not objects and config.task.target_objects:
        objects = [config.task.target_objects[0]]
    return scene, objects


def build_scene_xml(
    config: ProjectConfig,
    output_dir: Path,
    *,
    scene_name: str | None = None,
    object_names: list[str] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.simulation.entrypoint is None:
        xml_path = config.resolve_path(config.embodiment.xml_path)
        if xml_path is None:
            raise ValueError("Embodiment xml_path is missing.")
        return xml_path

    entry = config.simulation.entrypoint
    builder = import_object(
        file_path=config.resolve_path(entry.file),
        module_name=entry.module,
        attribute=entry.attribute,
        python_paths=config.all_python_paths(),
    )

    wrapper_out = output_dir / "preview_scene.xml"
    kwargs: dict[str, Any] = dict(config.simulation.build_kwargs)
    kwargs.update(
        {
            "scene_name": scene_name or config.simulation.preview_scene,
            "object_names": object_names or list(config.simulation.preview_objects),
            "wrapper_out": wrapper_out,
            "use_cache": False,
        }
    )

    result = call_with_supported_kwargs(builder, **kwargs)
    if result is None:
        return wrapper_out
    if isinstance(result, Path):
        return result.resolve()
    return Path(str(result)).expanduser().resolve()


def resolve_method(instance: Any, preferred_name: str | None, fallback_name: str):
    if preferred_name and hasattr(instance, preferred_name):
        return getattr(instance, preferred_name)
    if hasattr(instance, fallback_name):
        return getattr(instance, fallback_name)
    raise AttributeError(f"Object `{type(instance).__name__}` has neither `{preferred_name}` nor `{fallback_name}`.")


def accepts_keyword(func: Any, keyword: str) -> bool:
    signature = inspect.signature(func)
    if keyword in signature.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
