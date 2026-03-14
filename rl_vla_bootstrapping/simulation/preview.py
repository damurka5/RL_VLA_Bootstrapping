from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rl_vla_bootstrapping.core.specs import ProjectConfig
from rl_vla_bootstrapping.embodiments.mujoco import MujocoEmbodiment
from rl_vla_bootstrapping.simulation.scene_builder import (
    accepts_keyword,
    build_scene_xml,
    preview_selection,
    resolve_method,
)


def _save_image(array: np.ndarray, output_path: Path) -> None:
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(array).save(output_path)
        return
    except Exception:
        try:
            import imageio.v2 as imageio

            imageio.imwrite(output_path, array)
            return
        except Exception as exc:
            raise RuntimeError(f"Could not save preview image to {output_path}") from exc


def _extract_latest_frame(controller: Any, buffer_attr: str) -> np.ndarray | None:
    frames = getattr(controller, buffer_attr, None)
    if isinstance(frames, list) and frames:
        return np.asarray(frames[-1])
    return None


def render_preview(
    config: ProjectConfig,
    output_dir: Path,
    *,
    scene_name: str | None = None,
    object_names: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embodiment = MujocoEmbodiment(config=config, spec=config.embodiment)
    default_scene, default_objects = preview_selection(config)
    scene_name = scene_name or default_scene
    object_names = object_names or default_objects

    xml_path = build_scene_xml(
        config,
        output_dir=output_dir,
        scene_name=scene_name,
        object_names=object_names,
    )
    controller = embodiment.instantiate_controller(xml_path=xml_path, run_dir=output_dir)
    controller_spec = config.embodiment.controller
    methods = controller_spec.method_map

    initialize = resolve_method(controller, methods.get("initialize"), "initialize")
    cleanup = resolve_method(controller, methods.get("cleanup"), "cleanup")
    step = resolve_method(controller, methods.get("step"), "run_simulation_step")

    initialize()
    for _ in range(max(1, int(controller_spec.preview_steps))):
        if accepts_keyword(step, "capture_frame"):
            step(capture_frame=True)
        else:
            step()

    outputs: dict[str, str] = {}
    try:
        for slot, buffer_attr in controller_spec.frame_buffers.items():
            image = _extract_latest_frame(controller, buffer_attr)
            if image is None:
                continue
            output_path = output_dir / f"{slot}.png"
            _save_image(image, output_path)
            outputs[slot] = output_path.as_posix()

        if not outputs and controller_spec.camera_handles:
            capture = resolve_method(controller, methods.get("capture_frame"), "capture_frame")
            for slot, handle_attr in controller_spec.camera_handles.items():
                if not hasattr(controller, handle_attr):
                    continue
                image = capture(getattr(controller, handle_attr), config.embodiment.cameras.get(slot, slot))
                output_path = output_dir / f"{slot}.png"
                _save_image(np.asarray(image), output_path)
                outputs[slot] = output_path.as_posix()
    finally:
        cleanup()

    metadata = {
        "xml_path": xml_path.as_posix(),
        "scene_name": scene_name,
        "object_names": object_names,
        "images": outputs,
    }
    (output_dir / "preview_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata
