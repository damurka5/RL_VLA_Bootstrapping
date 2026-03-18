from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rl_vla_bootstrapping.core.commands import StagePlan, append_cli_arg
from rl_vla_bootstrapping.core.specs import EntrypointRef, ProjectConfig


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


def _resolved_python_paths(config: ProjectConfig, raw_paths: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        path = config.resolve_path(raw)
        if path is None or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _metadata_name_list(metadata: dict[str, Any], key: str) -> list[str]:
    raw = metadata.get(key)
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _allowed_objects_from_config(config: ProjectConfig) -> list[str]:
    metadata = config.task.metadata if isinstance(config.task.metadata, dict) else {}
    scene_object_pool = _metadata_name_list(metadata, "scene_object_pool")
    if scene_object_pool:
        return scene_object_pool
    target_pool = _metadata_name_list(metadata, "target_object_pool")
    distractor_pool = _metadata_name_list(metadata, "distractor_object_pool")
    if target_pool or distractor_pool:
        merged: list[str] = []
        seen: set[str] = set()
        for name in [*target_pool, *distractor_pool]:
            if name in seen:
                continue
            seen.add(name)
            merged.append(name)
        return merged
    return list(config.task.target_objects)


def _encode_entrypoint_env(
    env: dict[str, str],
    *,
    prefix: str,
    entrypoint: EntrypointRef | None,
    config: ProjectConfig,
) -> None:
    if entrypoint is None:
        return
    env[f"{prefix}_ATTRIBUTE"] = entrypoint.attribute
    if entrypoint.file:
        resolved_file = config.resolve_path(entrypoint.file)
        if resolved_file is not None:
            env[f"{prefix}_FILE"] = resolved_file.as_posix()
    if entrypoint.module:
        env[f"{prefix}_MODULE"] = entrypoint.module
    python_paths = _resolved_python_paths(config, entrypoint.python_paths)
    if python_paths:
        env[f"{prefix}_PYTHONPATHS"] = os.pathsep.join(path.as_posix() for path in python_paths)


def _task_hook_env(config: ProjectConfig) -> dict[str, str]:
    env: dict[str, str] = {}
    _encode_entrypoint_env(env, prefix="RLVLA_TASK_REWARD", entrypoint=config.task.reward, config=config)
    _encode_entrypoint_env(
        env,
        prefix="RLVLA_TASK_SUCCESS",
        entrypoint=config.task.success_predicate,
        config=config,
    )
    if config.task.goal_region:
        env["RLVLA_TASK_GOAL_REGION_JSON"] = json.dumps(config.task.goal_region, sort_keys=True)
    if config.task.goal_relation:
        env["RLVLA_TASK_GOAL_RELATION"] = config.task.goal_relation
    if config.task.dense_reward_terms:
        env["RLVLA_TASK_DENSE_REWARD_TERMS_JSON"] = json.dumps(
            config.task.dense_reward_terms,
            sort_keys=True,
        )
    if config.task.metadata:
        env["RLVLA_TASK_METADATA_JSON"] = json.dumps(config.task.metadata, sort_keys=True)
    return env


def _resolve_desk_textures_dir(config: ProjectConfig) -> tuple[Path | None, str | None]:
    primary = config.resolve_path(config.simulation.desk_textures_dir)
    fallback = config.resolve_path(config.simulation.desk_textures_fallback_dir)
    if primary is not None and primary.exists():
        return primary, None
    if fallback is not None and fallback.exists():
        note = f"Desk textures fallback selected: {fallback}"
        return fallback, note
    return primary or fallback, None


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


def _append_openvla_script_arg(argv: list[str], key: str, value: Any) -> None:
    # OpenVLA/OFT training scripts expose literal underscore option names,
    # e.g. `--desk_textures_dir` and `--no-wrapper_cleanup`.
    append_cli_arg(argv, key, value, preserve_underscores=True)


def _extract_cdpr_env_overrides(injected: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    if "lock_non_commanded_axes" in injected:
        env["RLVLA_CDPR_LOCK_NON_COMMANDED_AXES"] = "1" if bool(injected.pop("lock_non_commanded_axes")) else "0"
    if "lock_non_commanded_axes_threshold" in injected:
        env["RLVLA_CDPR_LOCK_NON_COMMANDED_AXES_THRESHOLD"] = str(
            float(injected.pop("lock_non_commanded_axes_threshold"))
        )
    if "randomize_ee_start" in injected:
        env["RLVLA_CDPR_RANDOMIZE_EE_START"] = "1" if bool(injected.pop("randomize_ee_start")) else "0"
    if "ee_start_x_bounds" in injected:
        env["RLVLA_CDPR_EE_START_X_BOUNDS"] = json.dumps(
            [float(value) for value in injected.pop("ee_start_x_bounds")]
        )
    if "ee_start_y_bounds" in injected:
        env["RLVLA_CDPR_EE_START_Y_BOUNDS"] = json.dumps(
            [float(value) for value in injected.pop("ee_start_y_bounds")]
        )
    if "ee_start_z" in injected:
        env["RLVLA_CDPR_EE_START_Z"] = str(float(injected.pop("ee_start_z")))
    return env


def build_openvla_rl_plan(config: ProjectConfig, run_dir: Path) -> StagePlan:
    script_path = config.resolve_path(config.training.rl.script_path or config.policy.rl_script)
    if script_path is None:
        raise ValueError("RL stage needs `training.rl.script_path` or `policy.rl_script`.")

    stage_dir = run_dir / "rl"
    argv = _build_stage_prefix(
        python_executable=config.project.python_executable,
        script_path=script_path,
        launcher=config.training.rl.launcher,
        launcher_args=config.training.rl.launcher_args,
    )

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
    desk_textures_dir, desk_texture_note = _resolve_desk_textures_dir(config)
    if desk_textures_dir is not None and "desk_textures_dir" not in injected:
        injected["desk_textures_dir"] = desk_textures_dir.as_posix()
    allowed_objects = _allowed_objects_from_config(config)
    if allowed_objects and "allowed_objects" not in injected:
        injected["allowed_objects"] = allowed_objects
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

    stage_env = _shared_env(config)
    stage_env.update(_task_hook_env(config))
    stage_env.update(_extract_cdpr_env_overrides(injected))

    for key, value in injected.items():
        _append_openvla_script_arg(argv, key, value)

    notes = [
        "Zero-demo RL stage using an external OpenVLA-OFT-compatible trainer.",
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
        cwd=str(config.resolve_path(config.policy.repo_path) or run_dir),
        env=stage_env,
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

    argv = _build_stage_prefix(
        python_executable=config.project.python_executable,
        script_path=script_path,
        launcher=config.training.sft.launcher,
        launcher_args=config.training.sft.launcher_args,
    )
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
        _append_openvla_script_arg(argv, key, value)

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
