from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rl_vla_bootstrapping.core.commands import StagePlan, append_cli_arg
from rl_vla_bootstrapping.core.specs import ProjectConfig
from rl_vla_bootstrapping.policy.openvla_oft import (
    _allowed_objects_from_config,
    _extract_cdpr_env_overrides,
    _maybe_infer_xyz_step,
    _resolve_desk_textures_dir,
    _shared_env,
    _task_hook_env,
)
from rl_vla_bootstrapping.policy.smolvla_cdpr import DEFAULT_SMOLVLA_CHECKPOINT


def _build_stage_prefix(
    *,
    python_executable: str,
    script_path: Path,
    launcher: str | None,
    launcher_args: dict[str, Any],
    module_name: str | None = None,
) -> list[str]:
    if launcher:
        argv = [launcher]
        for key, value in launcher_args.items():
            append_cli_arg(argv, key, value)
        if module_name:
            argv.extend(["-m", module_name])
        else:
            argv.append(str(script_path))
        return argv
    if module_name:
        return [python_executable, "-m", module_name]
    return [python_executable, str(script_path)]


def _append_smolvla_script_arg(argv: list[str], key: str, value: Any) -> None:
    append_cli_arg(argv, key, value, preserve_underscores=False)


def _repo_root(config: ProjectConfig) -> Path:
    return config.resolve_path("../..") or config.config_path.resolve().parents[2]


def _module_name_for_script(script_path: Path) -> str | None:
    parts = script_path.parts[-3:]
    if parts == ("rl_vla_bootstrapping", "policy", "smolvla_finetune_cdpr.py"):
        return "rl_vla_bootstrapping.policy.smolvla_finetune_cdpr"
    if parts == ("rl_vla_bootstrapping", "policy", "smolvla_grpo_finetune_cdpr.py"):
        return "rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr"
    return None


def _smolvla_env(config: ProjectConfig) -> dict[str, str]:
    env = _shared_env(config, extra_paths=[_repo_root(config)])
    paths = [_repo_root(config)]
    if env.get("PYTHONPATH"):
        paths.extend(Path(part) for part in env["PYTHONPATH"].split(os.pathsep) if part)
    current = os.environ.get("PYTHONPATH", "")
    if current:
        paths.extend(Path(part) for part in current.split(os.pathsep) if part)
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve().as_posix()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    env["PYTHONPATH"] = os.pathsep.join(deduped)
    return env


def build_smolvla_rl_plan(config: ProjectConfig, run_dir: Path) -> StagePlan:
    script_path = config.resolve_path(config.training.rl.script_path or config.policy.rl_script)
    if script_path is None:
        raise ValueError("SmolVLA RL stage needs `training.rl.script_path` or `policy.rl_script`.")

    stage_dir = run_dir / "rl"
    launcher_args = dict(config.training.rl.launcher_args)
    nproc_override = os.environ.get("RLVLA_SMOLVLA_NPROC_PER_NODE", "").strip()
    if nproc_override:
        launcher_args["nproc_per_node"] = int(nproc_override)
    argv = _build_stage_prefix(
        python_executable=config.project.python_executable,
        script_path=script_path,
        launcher=config.training.rl.launcher,
        launcher_args=launcher_args,
        module_name=_module_name_for_script(script_path),
    )

    injected: dict[str, Any] = dict(config.training.rl.args)
    injected.setdefault("config", config.config_path.as_posix())
    injected.setdefault("base_checkpoint", config.policy.base_checkpoint or DEFAULT_SMOLVLA_CHECKPOINT)
    injected.setdefault("run_root_dir", run_dir.as_posix())
    injected.setdefault("run_id", "rl")
    injected.setdefault("chunk_size", config.policy.action_codec.chunk_size)
    injected.setdefault("action_dim", len(config.embodiment.action_adapter.common_action_keys))
    injected.setdefault("image_size", 256)

    resume_checkpoint = os.environ.get("RLVLA_SMOLVLA_RESUME_CHECKPOINT", "").strip()
    if resume_checkpoint:
        injected["resume_checkpoint"] = resume_checkpoint
    max_train_steps = os.environ.get("RLVLA_SMOLVLA_MAX_TRAIN_STEPS", "").strip()
    if max_train_steps:
        injected["max_train_steps"] = int(max_train_steps)
    noise_schedule_start_step = os.environ.get(
        "RLVLA_SMOLVLA_NOISE_SCHEDULE_START_STEP", ""
    ).strip()
    if noise_schedule_start_step:
        injected["noise_schedule_start_step"] = int(noise_schedule_start_step)

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

    stage_env = _smolvla_env(config)
    stage_env.update(_task_hook_env(config))
    stage_env.update(_extract_cdpr_env_overrides(injected))

    for key, value in injected.items():
        _append_smolvla_script_arg(argv, key, value)

    algorithm = str(config.training.rl.algorithm or "").lower()
    notes = [
        "SmolVLA CDPR RL stage: frozen LeRobot SmolVLA prior plus a small Torch residual/readout head.",
        "MuJoCo stepping remains CPU-side while model inference/training runs on GPU.",
        "LeRobot/SmolVLA imports are runtime-only; local config parsing does not download weights.",
    ]
    if "grpo" in algorithm:
        notes.append("GRPO mode trains the residual policy with grouped relative advantages and no TD3 critics.")
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
