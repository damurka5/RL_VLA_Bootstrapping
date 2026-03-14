from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rl_vla_bootstrapping.core.commands import StagePlan, append_cli_arg
from rl_vla_bootstrapping.core.specs import ProjectConfig


def _join_pythonpath(paths: list[Path]) -> str:
    parts = [str(path.resolve()) for path in paths if path]
    if not parts:
        return os.environ.get("PYTHONPATH", "")
    current = os.environ.get("PYTHONPATH", "")
    if current:
        parts.append(current)
    return os.pathsep.join(parts)


def _shared_env(config: ProjectConfig, extra_paths: list[Path] | None = None) -> dict[str, str]:
    python_paths = list(config.all_python_paths())
    for raw in (config.repos.dataset_repo, config.repos.embodiment_repo, config.repos.openvla_oft):
        path = config.resolve_path(raw)
        if path is not None:
            python_paths.append(path)
    if extra_paths:
        python_paths.extend(extra_paths)
    return {
        **config.project.env,
        **config.remote.env_vars,
        "PYTHONPATH": _join_pythonpath(python_paths),
    }


def _maybe_infer_xyz_step(config: ProjectConfig) -> float | None:
    scales = config.embodiment.action_adapter.controller_scales
    xyz = [scales.get(axis) for axis in ("x", "y", "z")]
    if any(value is None for value in xyz):
        return None
    base = float(xyz[0])
    if max(abs(float(value) - base) for value in xyz) > 1e-9:
        return None
    return base


def build_openvla_rl_plan(config: ProjectConfig, run_dir: Path) -> StagePlan:
    script_path = config.resolve_path(config.training.rl.script_path or config.policy.rl_script)
    if script_path is None:
        raise ValueError("RL stage needs `training.rl.script_path` or `policy.rl_script`.")

    stage_dir = run_dir / "rl"
    argv = [config.project.python_executable, str(script_path)]

    injected: dict[str, Any] = dict(config.training.rl.args)
    if config.policy.base_checkpoint and "vla_path" not in injected:
        injected["vla_path"] = config.policy.base_checkpoint

    dataset_root = config.resolve_path(config.repos.dataset_repo)
    if dataset_root is not None and "cdpr_dataset_root" not in injected:
        injected["cdpr_dataset_root"] = dataset_root.as_posix()

    embodiment_repo = config.resolve_path(config.repos.embodiment_repo)
    if embodiment_repo is None:
        robot_root = config.resolve_path(config.embodiment.robot_root)
        if robot_root is not None:
            embodiment_repo = robot_root.parent
    if embodiment_repo is not None and "cdpr_mujoco_root" not in injected:
        injected["cdpr_mujoco_root"] = embodiment_repo.as_posix()

    if config.simulation.catalog_path and "catalog_path" not in injected:
        injected["catalog_path"] = config.resolve_path(config.simulation.catalog_path).as_posix()
    if config.simulation.desk_textures_dir and "desk_textures_dir" not in injected:
        injected["desk_textures_dir"] = config.resolve_path(config.simulation.desk_textures_dir).as_posix()
    if config.task.target_objects and "allowed_objects" not in injected:
        injected["allowed_objects"] = list(config.task.target_objects)
    if config.task.instruction_types and "instruction_types" not in injected:
        injected["instruction_types"] = list(config.task.instruction_types)

    xyz_step = _maybe_infer_xyz_step(config)
    if xyz_step is not None and "action_step_xyz" not in injected:
        injected["action_step_xyz"] = xyz_step
    yaw_step = config.embodiment.action_adapter.controller_scales.get("yaw")
    if yaw_step is not None and "action_step_yaw" not in injected:
        injected["action_step_yaw"] = float(yaw_step)
    if "num_images_in_input" not in injected:
        injected["num_images_in_input"] = config.policy.num_images_in_input

    injected.setdefault("run_root_dir", run_dir.as_posix())
    injected.setdefault("run_id", "rl")

    for key, value in injected.items():
        append_cli_arg(argv, key, value)

    notes = [
        "Zero-demo RL stage using an external OpenVLA-OFT-compatible trainer.",
    ]
    return StagePlan(
        name="rl",
        kind="external_python",
        command=argv,
        cwd=str(config.resolve_path(config.policy.repo_path) or run_dir),
        env=_shared_env(config),
        notes=notes,
        artifact_paths=[stage_dir.as_posix()],
    )


def _discover_latest_rl_dir(run_dir: Path) -> Path | None:
    rl_dir = run_dir / "rl"
    if not rl_dir.exists():
        return None
    adapter_dirs = sorted(rl_dir.rglob("vla_cdpr_adapter"))
    if adapter_dirs:
        return adapter_dirs[-1].parent
    return rl_dir


def build_openvla_sft_plan(config: ProjectConfig, run_dir: Path) -> StagePlan:
    script_path = config.resolve_path(config.training.sft.script_path or config.policy.sft_script)
    if script_path is None:
        raise ValueError("SFT stage needs `training.sft.script_path` or `policy.sft_script`.")

    argv = [config.project.python_executable, str(script_path)]
    injected: dict[str, Any] = dict(config.training.sft.args)

    resume_from_rl = bool(injected.pop("resume_from_rl", True))
    if resume_from_rl and "vla_path" not in injected:
        latest_rl = _discover_latest_rl_dir(run_dir)
        if latest_rl is not None:
            injected["vla_path"] = latest_rl.as_posix()
    if config.policy.base_checkpoint and "vla_path" not in injected:
        injected["vla_path"] = config.policy.base_checkpoint

    injected.setdefault("run_root_dir", str(run_dir / "sft"))

    for key, value in injected.items():
        append_cli_arg(argv, key, value)

    notes = [
        "Optional SFT refinement stage. Uses RL artifacts if they are available in the current run.",
    ]
    return StagePlan(
        name="sft",
        kind="external_python",
        command=argv,
        cwd=str(config.resolve_path(config.policy.repo_path) or run_dir),
        env=_shared_env(config),
        notes=notes,
        artifact_paths=[str(run_dir / "sft")],
    )
