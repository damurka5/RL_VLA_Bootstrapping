from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rl_vla_bootstrapping.core.commands import StagePlan, append_cli_arg
from rl_vla_bootstrapping.core.specs import ProjectConfig
from rl_vla_bootstrapping.policy.octo_cdpr_adapter import (
    DEFAULT_OCTO_REPO_PATH,
    DEFAULT_OCTO_SMALL_CHECKPOINT,
)
from rl_vla_bootstrapping.policy.openvla_oft import (
    _allowed_objects_from_config,
    _extract_cdpr_env_overrides,
    _maybe_infer_xyz_step,
    _resolve_desk_textures_dir,
    _shared_env,
    _task_hook_env,
)


def _build_stage_prefix(
    *,
    python_executable: str,
    script_path: Path,
    launcher: str | None,
    launcher_args: dict[str, Any],
) -> list[str]:
    if launcher:
        argv = [launcher]
        for key, value in launcher_args.items():
            append_cli_arg(argv, key, value)
        argv.append(str(script_path))
        return argv
    return [python_executable, str(script_path)]


def _append_octo_script_arg(argv: list[str], key: str, value: Any) -> None:
    append_cli_arg(argv, key, value, preserve_underscores=False)


def _join_pythonpath(paths: list[Path]) -> str:
    parts = [str(path.resolve()) for path in paths if path]
    current = os.environ.get("PYTHONPATH", "")
    if current:
        parts.append(current)
    return os.pathsep.join(parts)


def _repo_root(config: ProjectConfig) -> Path:
    return config.resolve_path("../..") or config.config_path.resolve().parents[2]


def _octo_env(config: ProjectConfig, *, extra_paths: list[Path] | None = None) -> dict[str, str]:
    env = _shared_env(config, extra_paths=extra_paths)
    octo_repo = Path(os.environ.get("OCTO_REPO_PATH", DEFAULT_OCTO_REPO_PATH)).expanduser()
    paths = [_repo_root(config), octo_repo]
    if env.get("PYTHONPATH"):
        paths.extend(Path(part) for part in env["PYTHONPATH"].split(os.pathsep) if part)
    env["PYTHONPATH"] = _join_pythonpath(paths)
    env["OCTO_REPO_PATH"] = octo_repo.as_posix()
    return env


def build_octo_rl_plan(config: ProjectConfig, run_dir: Path) -> StagePlan:
    script_path = config.resolve_path(config.training.rl.script_path or config.policy.rl_script)
    if script_path is None:
        raise ValueError("Octo RL stage needs `training.rl.script_path` or `policy.rl_script`.")

    stage_dir = run_dir / "rl"
    argv = _build_stage_prefix(
        python_executable=config.project.python_executable,
        script_path=script_path,
        launcher=config.training.rl.launcher,
        launcher_args=config.training.rl.launcher_args,
    )

    injected: dict[str, Any] = dict(config.training.rl.args)
    injected.setdefault("config", config.config_path.as_posix())
    injected.setdefault("base_checkpoint", config.policy.base_checkpoint or DEFAULT_OCTO_SMALL_CHECKPOINT)
    injected.setdefault("run_root_dir", run_dir.as_posix())
    injected.setdefault("run_id", "rl")
    injected.setdefault("chunk_size", config.policy.action_codec.chunk_size)
    injected.setdefault("action_dim", len(config.embodiment.action_adapter.common_action_keys))
    injected.setdefault("image_size", 256)
    resume_checkpoint = os.environ.get("RLVLA_OCTO_RESUME_CHECKPOINT", "").strip()
    if resume_checkpoint:
        injected.setdefault("resume_checkpoint", resume_checkpoint)

    dataset_root = config.resolve_path(config.repos.dataset_repo)
    if dataset_root is not None:
        injected.setdefault("cdpr_dataset_root", dataset_root.as_posix())

    embodiment_repo = config.resolve_path(config.repos.embodiment_repo)
    if embodiment_repo is None:
        robot_root = config.resolve_path(config.embodiment.robot_root)
        if robot_root is not None:
            embodiment_repo = robot_root.parent
    if embodiment_repo is not None:
        injected.setdefault("cdpr_mujoco_root", embodiment_repo.as_posix())

    if config.simulation.catalog_path:
        catalog_path = config.resolve_path(config.simulation.catalog_path)
        if catalog_path is not None:
            injected.setdefault("catalog_path", catalog_path.as_posix())
    desk_textures_dir, desk_texture_note = _resolve_desk_textures_dir(config)
    if desk_textures_dir is not None:
        injected.setdefault("desk_textures_dir", desk_textures_dir.as_posix())
    allowed_objects = _allowed_objects_from_config(config)
    if allowed_objects:
        injected.setdefault("allowed_objects", allowed_objects)
    if config.task.instruction_types:
        injected.setdefault("instruction_types", list(config.task.instruction_types))

    xyz_step = _maybe_infer_xyz_step(config)
    if xyz_step is not None:
        injected.setdefault("action_step_xyz", xyz_step)
    yaw_step = config.embodiment.action_adapter.controller_scales.get("yaw")
    if yaw_step is not None:
        injected.setdefault("action_step_yaw", float(yaw_step))
    gripper_step = config.embodiment.action_adapter.controller_scales.get("gripper")
    if gripper_step is not None:
        injected.setdefault("action_step_gripper", float(gripper_step))

    stage_env = _octo_env(config)
    stage_env.update(_task_hook_env(config))
    stage_env.update(_extract_cdpr_env_overrides(dict(injected)))

    for key, value in injected.items():
        _append_octo_script_arg(argv, key, value)

    notes = [
        "Octo-Small CDPR RL stage: frozen pretrained Octo diffusion policy plus a small residual/readout head.",
        "Octo/JAX imports are runtime-only; local config parsing does not download weights.",
    ]
    if desk_texture_note:
        notes.append(desk_texture_note)
    if config.task.reward is not None:
        notes.append("Task reward hook exported through RLVLA_TASK_REWARD_* env vars.")
    if config.task.success_predicate is not None:
        notes.append("Task success hook exported through RLVLA_TASK_SUCCESS_* env vars.")

    return StagePlan(
        name="rl",
        kind="external_python",
        command=argv,
        cwd=str(_repo_root(config)),
        env=stage_env,
        notes=notes,
        artifact_paths=[stage_dir.as_posix()],
    )
