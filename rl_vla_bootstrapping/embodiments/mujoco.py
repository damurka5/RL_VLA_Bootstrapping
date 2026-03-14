from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_vla_bootstrapping.core.imports import import_object
from rl_vla_bootstrapping.core.specs import EmbodimentSpec, ProjectConfig


@dataclass
class MujocoEmbodiment:
    config: ProjectConfig
    spec: EmbodimentSpec

    def resolved_robot_root(self) -> Path:
        path = self.config.resolve_path(self.spec.robot_root)
        if path is None:
            raise ValueError("Embodiment robot_root is missing.")
        return path

    def resolved_xml_path(self) -> Path:
        path = self.config.resolve_path(self.spec.xml_path)
        if path is None:
            raise ValueError("Embodiment xml_path is missing.")
        return path

    def resolved_python_paths(self) -> list[Path]:
        paths = [self.resolved_robot_root()]
        paths.extend(self.config.all_python_paths())
        return paths

    def load_controller_class(self):
        entry = self.spec.controller.entrypoint
        return import_object(
            file_path=self.config.resolve_path(entry.file),
            module_name=entry.module,
            attribute=entry.attribute,
            python_paths=self.resolved_python_paths(),
        )

    def build_controller_kwargs(self, xml_path: Path, run_dir: Path) -> dict[str, Any]:
        kwargs = dict(self.spec.controller.init_kwargs)
        for key, value in list(kwargs.items()):
            if isinstance(value, str):
                kwargs[key] = value.format(
                    run_dir=run_dir.as_posix(),
                    xml_path=xml_path.as_posix(),
                    robot_root=self.resolved_robot_root().as_posix(),
                )

        if "xml_path" not in kwargs:
            kwargs["xml_path"] = str(xml_path)
        return kwargs

    def instantiate_controller(self, xml_path: Path, run_dir: Path):
        controller_cls = self.load_controller_class()
        kwargs = self.build_controller_kwargs(xml_path=xml_path, run_dir=run_dir)
        try:
            return controller_cls(**kwargs)
        except TypeError:
            signature = inspect.signature(controller_cls)
            if "xml_path" in signature.parameters:
                return controller_cls(xml_path=str(xml_path))
            raise

    def validate(self) -> list[str]:
        errors: list[str] = []
        xml_path = self.resolved_xml_path()
        if not xml_path.exists():
            errors.append(f"Embodiment XML not found: {xml_path}")
        controller_file = self.config.resolve_path(self.spec.controller.entrypoint.file)
        if controller_file is not None and not controller_file.exists():
            errors.append(f"Controller file not found: {controller_file}")
        if self.spec.dof and self.spec.dof != len(self.spec.action_adapter.common_action_keys):
            errors.append(
                "Embodiment `dof` does not match action key count: "
                f"dof={self.spec.dof}, action_keys={len(self.spec.action_adapter.common_action_keys)}"
            )
        return errors
