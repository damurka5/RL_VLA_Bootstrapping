from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional runtime dependency
    tqdm = None

from rl_vla_bootstrapping.cli.run_cdpr_policy import (
    _control_spec_from_config,
    _load_openvla_modules,
    _make_observation,
    _predict_normalized_action_chunk,
    _resolve_llm_dim,
    _set_num_images_in_input,
)
from rl_vla_bootstrapping.core.commands import ensure_directory
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.openvla_oft import (
    _allowed_objects_from_config,
    _extract_cdpr_env_overrides,
    _resolve_desk_textures_dir,
    _task_hook_env,
)
from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv
from robots.cdpr.cdpr_dataset.rl_instruction_tasks import INSTRUCTION_TEXT, INSTRUCTION_TYPES
from robots.cdpr.cdpr_dataset.synthetic_tasks import clear_sim_recording_buffers


_ACTION_HEAD_FILENAMES = (
    "action_head.pt",
    "action_head_cdpr.pt",
    "action_head_latest.pt",
)


@dataclass(frozen=True)
class ResolvedPolicyArtifacts:
    checkpoint_dir: Path | None
    adapter_path: Path
    action_head_path: Path


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    seed: int | None
    instruction_type: str
    instruction_text: str
    success: bool
    terminated: bool
    truncated: bool
    steps: int
    reward_total: float
    scene: str
    goal_position: list[float]
    ee_start: list[float]


@dataclass(frozen=True)
class InstructionSummary:
    instruction_type: str
    instruction_text: str
    successes: int
    episodes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    video_path: str | None


def _rl_args(config: Any) -> dict[str, Any]:
    training = getattr(config, "training", None)
    rl = getattr(training, "rl", None)
    return dict(getattr(rl, "args", {}) or {})


def _runtime_python_paths(config: Any) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    def _append(raw_path: str | None) -> None:
        path = config.resolve_path(raw_path)
        if path is None:
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        paths.append(resolved)

    for path in config.all_python_paths():
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(resolved)

    _append(config.repos.dataset_repo)
    _append(config.repos.embodiment_repo)
    _append(config.repos.openvla_oft)
    _append(config.policy.repo_path)
    return paths


def _prepend_runtime_python_paths(config: Any) -> None:
    for path in reversed(_runtime_python_paths(config)):
        path_str = path.as_posix()
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an OpenVLA/OFT CDPR checkpoint by running fixed-count episodes for each "
            "instruction type and reporting per-instruction success rates."
        )
    )
    parser.add_argument("--config", required=True, help="Path to bootstrap YAML/JSON/TOML config.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Checkpoint step directory. If provided, the validator will infer "
            "`vla_cdpr_adapter` and try to locate an action-head checkpoint inside it."
        ),
    )
    parser.add_argument("--adapter-path", default=None, help="Optional explicit adapter directory override.")
    parser.add_argument("--action-head-path", default=None, help="Optional explicit action-head checkpoint override.")
    parser.add_argument("--base-ckpt", default=None, help="Optional override for the base VLA checkpoint.")
    parser.add_argument("--scene", default=None, help="Optional fixed scene override for every episode.")
    parser.add_argument(
        "--wrapper-dir",
        default=None,
        help="Optional wrapper cache directory override. Defaults to an existing remote cache if found.",
    )
    parser.add_argument(
        "--episodes-per-instruction",
        type=int,
        default=100,
        help="How many episodes to run for each instruction type.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode horizon. Defaults to validation_max_steps, then max_env_steps, then 150.",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=None,
        help="Open-loop chunk length. Defaults to the config action codec chunk size.",
    )
    parser.add_argument("--hold-steps", type=int, default=None, help="Override extra simulator substeps per action.")
    parser.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass through OpenVLA center-crop behavior.",
    )
    parser.add_argument("--run-dir", default=None, help="Optional output directory.")
    parser.add_argument("--run-name", default="cdpr_policy_validation", help="Artifact name prefix.")
    parser.add_argument(
        "--action-guard",
        type=float,
        default=1.25,
        help="Warn when predicted action absolute values exceed this before clipping to [-1, 1].",
    )
    parser.add_argument(
        "--record-success-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one overview video for the first successful episode of each instruction type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used to derive deterministic episode seeds. Pass --seed=-1 for entropy.",
    )
    parser.add_argument(
        "--log-every-episode",
        type=int,
        default=10,
        help="Logging cadence within each instruction bucket.",
    )
    parser.add_argument(
        "--success-distance",
        type=float,
        default=0.05,
        help="Validation success tolerance in meters for reaching the target point.",
    )
    parser.add_argument(
        "--reuse-existing-wrapper-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer randomly sampling an already existing matching wrapper bundle before building a new one.",
    )
    parser.add_argument(
        "--progress-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show only a tqdm progress bar and suppress the validator's other console output.",
    )
    return parser


def _candidate_checkpoint_dirs(raw_path: str | Path) -> list[Path]:
    base = Path(raw_path).expanduser().resolve()
    if base.is_file():
        return [base.parent]

    candidates: list[Path] = []
    if base.name == "vla_cdpr_adapter":
        candidates.append(base.parent)
    candidates.append(base)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _resolve_adapter_path(raw_path: str | Path) -> Path:
    base = Path(raw_path).expanduser().resolve()
    if base.is_dir() and (base / "adapter_config.json").is_file():
        return base

    candidate_dirs = _candidate_checkpoint_dirs(base)
    for candidate in candidate_dirs:
        adapter_dir = candidate / "vla_cdpr_adapter"
        if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").is_file():
            return adapter_dir.resolve()

    if base.name == "vla_cdpr_adapter":
        return base
    return base


def _resolve_action_head_path(raw_path: str | Path) -> Path:
    base = Path(raw_path).expanduser().resolve()
    if base.is_file():
        return base

    candidate_dirs = _candidate_checkpoint_dirs(base)
    for candidate in candidate_dirs:
        for filename in _ACTION_HEAD_FILENAMES:
            action_head_path = candidate / filename
            if action_head_path.is_file():
                return action_head_path.resolve()
        matches = sorted(
            path
            for path in candidate.glob("*")
            if path.is_file() and "action_head" in path.name.lower() and path.suffix in {".pt", ".pth", ".bin"}
        )
        if len(matches) == 1:
            return matches[0].resolve()

    return base


def _infer_checkpoint_dir(*, checkpoint_dir: str | None, adapter_path: Path, action_head_path: Path) -> Path | None:
    if checkpoint_dir:
        candidates = _candidate_checkpoint_dirs(checkpoint_dir)
        return candidates[0] if candidates else Path(checkpoint_dir).expanduser().resolve()
    if adapter_path.name == "vla_cdpr_adapter":
        return adapter_path.parent.resolve()
    if action_head_path.is_file():
        return action_head_path.parent.resolve()
    if action_head_path.is_dir():
        return action_head_path.resolve()
    return None


def _resolve_policy_artifacts(args: argparse.Namespace, config: Any) -> ResolvedPolicyArtifacts:
    rl_args = _rl_args(config)

    raw_adapter = args.adapter_path or args.checkpoint_dir or rl_args.get("adapter_path")
    raw_action_head = args.action_head_path or args.checkpoint_dir or rl_args.get("action_head_path")
    if not raw_adapter:
        raise RuntimeError(
            "Could not resolve an adapter path. Pass --checkpoint-dir or --adapter-path, "
            "or populate training.rl.args.adapter_path in the config."
        )
    if not raw_action_head:
        raise RuntimeError(
            "Could not resolve an action-head path. Pass --checkpoint-dir or --action-head-path, "
            "or populate training.rl.args.action_head_path in the config."
        )

    adapter_path = _resolve_adapter_path(raw_adapter)
    action_head_path = _resolve_action_head_path(raw_action_head)
    checkpoint_path = _infer_checkpoint_dir(
        checkpoint_dir=args.checkpoint_dir,
        adapter_path=adapter_path,
        action_head_path=action_head_path,
    )
    return ResolvedPolicyArtifacts(
        checkpoint_dir=checkpoint_path,
        adapter_path=adapter_path,
        action_head_path=action_head_path,
    )


def _default_max_steps(config: Any, args: argparse.Namespace) -> int:
    if args.max_steps is not None:
        return int(args.max_steps)
    rl_args = _rl_args(config)
    for key in ("validation_max_steps", "max_env_steps", "max_steps"):
        raw = rl_args.get(key)
        if raw is not None:
            return int(raw)
    return 150


def _episode_seed(base_seed: int | None, instruction_index: int, episode_index: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + int(instruction_index) * 100_000 + int(episode_index)


def _validation_task_metadata(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    metadata = dict(getattr(config.task, "metadata", {}) or {})
    metadata["success_distance"] = float(args.success_distance)
    return metadata


def _validation_env_vars(config: Any, args: argparse.Namespace) -> dict[str, str]:
    rl_args = _rl_args(config)
    env = {str(k): str(v) for k, v in getattr(config.project, "env", {}).items()}
    env.update({str(k): str(v) for k, v in getattr(config.remote, "env_vars", {}).items()})
    env.update(_task_hook_env(config))
    env.update(_extract_cdpr_env_overrides(dict(rl_args)))
    env["RLVLA_TASK_METADATA_JSON"] = json.dumps(
        _validation_task_metadata(config, args),
        sort_keys=True,
    )
    return env


def _resolve_wrapper_dir(config: Any, args: argparse.Namespace) -> Path | None:
    if args.wrapper_dir:
        return Path(args.wrapper_dir).expanduser().resolve()

    remote_candidate = Path("/robot/cdpr/cdpr_dataset/wrappers")
    if remote_candidate.exists():
        return remote_candidate.resolve()

    dataset_repo = config.resolve_path(config.repos.dataset_repo)
    if dataset_repo is not None:
        candidate = dataset_repo / "cdpr_dataset" / "wrappers"
        if candidate.exists():
            return candidate.resolve()

    return None


@contextmanager
def _silence_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def _progress_bar(total: int):
    if tqdm is None:  # pragma: no cover - exercised only when tqdm is unavailable
        raise RuntimeError("`tqdm` is required for progress display. Install it in the remote environment.")
    return tqdm(total=total, dynamic_ncols=True, file=sys.__stderr__, leave=True)


@contextmanager
def _temporary_env_vars(overrides: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _build_validation_env(
    *,
    config: Any,
    instruction_type: str,
    capture_frames: bool,
    max_steps: int,
    hold_steps: int | None,
    seed: int | None,
    args: argparse.Namespace,
    wrapper_dir: Path | None,
) -> CDPRLanguageRLEnv:
    rl_args = _rl_args(config)
    control_spec = _control_spec_from_config(config, hold_steps)
    desk_textures_dir, _ = _resolve_desk_textures_dir(config)
    metadata = dict(getattr(config.task, "metadata", {}) or {})

    return CDPRLanguageRLEnv(
        catalog_path=config.resolve_path(config.simulation.catalog_path),
        max_steps=int(max_steps),
        action_step_xyz=float(control_spec.action_step_xyz),
        action_step_yaw=float(control_spec.action_step_yaw),
        hold_steps=int(control_spec.hold_steps),
        lock_non_commanded_axes=rl_args.get("lock_non_commanded_axes"),
        lock_non_commanded_axes_threshold=rl_args.get("lock_non_commanded_axes_threshold"),
        randomize_ee_start=rl_args.get("randomize_ee_start"),
        ee_start_x_bounds=rl_args.get("ee_start_x_bounds"),
        ee_start_y_bounds=rl_args.get("ee_start_y_bounds"),
        ee_start_z=rl_args.get("ee_start_z"),
        move_distance=float(metadata.get("lateral_goal_offset", 0.40)),
        lift_distance=float(metadata.get("vertical_goal_offset", 0.10)),
        capture_frames=bool(capture_frames),
        instruction_types=[instruction_type],
        allowed_objects=_allowed_objects_from_config(config),
        desk_textures_dir=desk_textures_dir,
        wrapper_cleanup=bool(rl_args.get("wrapper_cleanup", True)),
        use_wrapper_cache=bool(rl_args.get("use_wrapper_cache", False)),
        reuse_existing_wrapper_variants=bool(args.reuse_existing_wrapper_variants),
        wrapper_dir=wrapper_dir,
        seed=seed,
    )


def _gripper_range(sim: Any, config: Any) -> tuple[float, float]:
    limits = config.embodiment.action_adapter.controller_limits["gripper"]
    return (
        float(getattr(sim, "gripper_min", limits[0])),
        float(getattr(sim, "gripper_max", limits[1])),
    )


def _predict_policy_chunk(
    *,
    runtime: dict[str, Any],
    sim: Any,
    instruction: str,
    config: Any,
) -> np.ndarray:
    chunk = np.asarray(
        _predict_normalized_action_chunk(
            vla=runtime["vla"],
            processor=runtime["processor"],
            action_head=runtime["action_head"],
            obs=_make_observation(sim, instruction, _gripper_range(sim, config))[0],
            instruction=instruction,
            chunk_length=int(runtime["chunk_length"]),
            num_images_in_input=int(runtime["num_images_in_input"]),
            device=runtime["device"],
            pixel_dtype=runtime["pixel_dtype"],
        ),
        dtype=np.float32,
    )
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    if chunk.shape[1] < 5:
        raise RuntimeError(f"Expected at least 5 action dimensions, got chunk shape {chunk.shape}.")
    return chunk[:, :5]


def _load_policy_runtime(
    *,
    config: Any,
    artifacts: ResolvedPolicyArtifacts,
    args: argparse.Namespace,
    quiet: bool,
) -> dict[str, Any]:
    policy_repo = config.resolve_path(config.policy.repo_path)
    if policy_repo is None:
        raise RuntimeError("Config is missing `policy.repo_path`.")

    (
        GenerateConfig,
        get_action_head,
        get_processor,
        _get_proprio_projector,
        get_vla,
        PeftModel,
        generate_config_note,
    ) = _load_openvla_modules(policy_repo)
    if generate_config_note and not quiet:
        print(f"[info] {generate_config_note}")

    chunk_length = int(args.chunk_length or config.policy.action_codec.chunk_size)
    cfg = GenerateConfig(
        pretrained_checkpoint=args.base_ckpt or config.policy.base_checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=int(config.policy.num_images_in_input),
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=bool(args.center_crop),
        num_open_loop_steps=chunk_length,
        unnorm_key=None,
    )
    cfg.cdpr_action_head_path = str(artifacts.action_head_path)

    if not artifacts.adapter_path.is_dir():
        raise RuntimeError(f"Adapter path is not a directory: {artifacts.adapter_path}")
    if not (artifacts.adapter_path / "adapter_config.json").is_file():
        raise RuntimeError(
            "Adapter directory does not contain `adapter_config.json`: "
            f"{artifacts.adapter_path}. If you passed a step directory, it should contain "
            "`vla_cdpr_adapter/`."
        )
    if not artifacts.action_head_path.exists():
        raise RuntimeError(f"Action-head path does not exist: {artifacts.action_head_path}")

    vla_base = get_vla(cfg)
    vla_base.eval()
    vla = PeftModel.from_pretrained(vla_base, str(artifacts.adapter_path))
    vla.eval()

    cfg.num_images_in_input = _set_num_images_in_input(vla, int(cfg.num_images_in_input))
    llm_dim = _resolve_llm_dim(vla)
    if llm_dim is None:
        raise RuntimeError("Could not resolve llm_dim from the wrapped OpenVLA model.")

    processor = get_processor(cfg)
    param = next(vla.parameters())
    action_head = get_action_head(cfg, llm_dim=llm_dim).to(device=param.device, dtype=param.dtype)
    action_head.eval()

    return {
        "cfg": cfg,
        "vla": vla,
        "processor": processor,
        "action_head": action_head,
        "device": param.device,
        "pixel_dtype": param.dtype,
        "chunk_length": chunk_length,
        "num_images_in_input": int(cfg.num_images_in_input),
    }


def _save_success_video(
    *,
    sim: Any,
    output_dir: Path,
    instruction_type: str,
    episode_result: EpisodeResult,
) -> str | None:
    frames = list(getattr(sim, "overview_frames", []) or [])
    if not frames or not hasattr(sim, "save_video"):
        return None

    fps = float(sim._estimate_video_fps()) if hasattr(sim, "_estimate_video_fps") else 20.0
    output_path = output_dir / f"{instruction_type}_episode_{episode_result.episode_index:03d}_overview.mp4"
    sim.save_video(frames, str(output_path), fps=fps)

    summary_path = output_dir / f"{instruction_type}_episode_{episode_result.episode_index:03d}_summary.json"
    summary_path.write_text(json.dumps(asdict(episode_result), indent=2), encoding="utf-8")
    return output_path.as_posix()


def _summarize_instruction_results(
    *,
    instruction_type: str,
    episode_results: list[EpisodeResult],
    video_path: str | None,
) -> InstructionSummary:
    successes = sum(1 for item in episode_results if item.success)
    rewards = np.asarray([item.reward_total for item in episode_results], dtype=np.float32)
    steps = np.asarray([item.steps for item in episode_results], dtype=np.float32)
    total = len(episode_results)
    return InstructionSummary(
        instruction_type=instruction_type,
        instruction_text=INSTRUCTION_TEXT.get(instruction_type, instruction_type.replace("_", " ")),
        successes=int(successes),
        episodes=int(total),
        success_rate=float(successes / max(total, 1)),
        mean_reward=float(np.mean(rewards)) if rewards.size > 0 else 0.0,
        mean_steps=float(np.mean(steps)) if steps.size > 0 else 0.0,
        video_path=video_path,
    )


def _write_success_rate_csv(output_path: Path, summaries: list[InstructionSummary]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instruction_type",
                "instruction_text",
                "successes",
                "episodes",
                "success_rate",
                "mean_reward",
                "mean_steps",
                "video_path",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.instruction_type,
                    summary.instruction_text,
                    summary.successes,
                    summary.episodes,
                    f"{summary.success_rate:.6f}",
                    f"{summary.mean_reward:.6f}",
                    f"{summary.mean_steps:.6f}",
                    summary.video_path or "",
                ]
            )


def _run_instruction_validation(
    *,
    instruction_type: str,
    instruction_index: int,
    config: Any,
    runtime: dict[str, Any],
    args: argparse.Namespace,
    videos_dir: Path,
    max_steps: int,
    base_seed: int | None,
    progress,
    wrapper_dir: Path | None,
) -> tuple[InstructionSummary, list[EpisodeResult]]:
    should_capture = bool(args.record_success_videos)
    env = _build_validation_env(
        config=config,
        instruction_type=instruction_type,
        capture_frames=should_capture,
        max_steps=max_steps,
        hold_steps=args.hold_steps,
        seed=base_seed,
        args=args,
        wrapper_dir=wrapper_dir,
    )

    reset_options = {"scene": args.scene} if args.scene else None
    episode_results: list[EpisodeResult] = []
    video_path: str | None = None
    successes = 0

    try:
        for episode_index in range(int(args.episodes_per_instruction)):
            env.capture_frames = bool(args.record_success_videos and video_path is None)
            seed = _episode_seed(base_seed, instruction_index, episode_index)
            with _silence_output(bool(args.progress_only)):
                _obs, reset_info = env.reset(seed=seed, options=reset_options)
            instruction = str(reset_info.get("language_instruction", INSTRUCTION_TEXT[instruction_type]))

            current_chunk = np.zeros((0, 5), dtype=np.float32)
            chunk_index = 0
            reward_total = 0.0
            terminated = False
            truncated = False
            final_info = dict(reset_info)

            while not (terminated or truncated):
                if chunk_index >= len(current_chunk):
                    with _silence_output(bool(args.progress_only)):
                        current_chunk = _predict_policy_chunk(
                            runtime=runtime,
                            sim=env.sim,
                            instruction=instruction,
                            config=config,
                        )
                    chunk_index = 0

                action = np.asarray(current_chunk[chunk_index], dtype=np.float32).reshape(5)
                chunk_index += 1
                max_abs = float(np.max(np.abs(action)))
                if max_abs > float(args.action_guard) and not args.progress_only:
                    print(
                        f"[warn] [{instruction_type}] episode={episode_index:03d} "
                        f"action max abs {max_abs:.4f} > {args.action_guard}; clipping to [-1, 1]."
                    )

                with _silence_output(bool(args.progress_only)):
                    _obs, reward, terminated, truncated, final_info = env.step(action)
                reward_total += float(reward)

            episode_result = EpisodeResult(
                episode_index=int(episode_index),
                seed=seed,
                instruction_type=instruction_type,
                instruction_text=instruction,
                success=bool(final_info.get("success", False)),
                terminated=bool(terminated),
                truncated=bool(truncated),
                steps=int(final_info.get("step", max_steps)),
                reward_total=float(reward_total),
                scene=str(final_info.get("scene", "")),
                goal_position=[float(value) for value in final_info.get("goal_position", [])],
                ee_start=[float(value) for value in final_info.get("ee_start", [])],
            )
            episode_results.append(episode_result)
            successes += int(episode_result.success)

            if episode_result.success and video_path is None and bool(args.record_success_videos):
                try:
                    with _silence_output(bool(args.progress_only)):
                        video_path = _save_success_video(
                            sim=env.sim,
                            output_dir=videos_dir,
                            instruction_type=instruction_type,
                            episode_result=episode_result,
                        )
                except Exception as exc:
                    if not args.progress_only:
                        print(f"[warn] Failed to save success video for {instruction_type}: {exc}")
                finally:
                    clear_sim_recording_buffers(env.sim)
            elif getattr(env, "sim", None) is not None:
                clear_sim_recording_buffers(env.sim)

            if progress is not None:
                progress.set_description_str(instruction_type)
                progress.set_postfix_str(f"success={successes}/{episode_index + 1}")
                progress.update(1)
            elif (
                episode_result.success
                or episode_index == 0
                or (episode_index + 1) % max(1, int(args.log_every_episode)) == 0
                or episode_index == (int(args.episodes_per_instruction) - 1)
            ):
                print(
                    f"[{instruction_type}] episode={episode_index + 1:03d}/{int(args.episodes_per_instruction):03d} "
                    f"success={episode_result.success} steps={episode_result.steps} "
                    f"reward={episode_result.reward_total:.4f} scene={episode_result.scene}"
                )

        summary = _summarize_instruction_results(
            instruction_type=instruction_type,
            episode_results=episode_results,
            video_path=video_path,
        )
        return summary, episode_results
    finally:
        with _silence_output(bool(args.progress_only)):
            env.close()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if int(args.episodes_per_instruction) <= 0:
        raise ValueError("--episodes-per-instruction must be positive.")

    config = load_project_config(args.config)
    _prepend_runtime_python_paths(config)
    artifacts = _resolve_policy_artifacts(args, config)

    run_dir = (
        ensure_directory(Path(args.run_dir).expanduser().resolve())
        if args.run_dir
        else ensure_directory(
            (config.resolve_path(config.project.output_root) or Path("runs"))
            / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    videos_dir = ensure_directory(run_dir / "videos")
    max_steps = _default_max_steps(config, args)
    base_seed = None if args.seed is None or int(args.seed) < 0 else int(args.seed)
    instruction_types = tuple(config.task.instruction_types) or INSTRUCTION_TYPES
    wrapper_dir = _resolve_wrapper_dir(config, args)

    if not args.progress_only:
        print(f"Run directory: {run_dir}")
        print(f"Checkpoint dir: {artifacts.checkpoint_dir}")
        print(f"Adapter path: {artifacts.adapter_path}")
        print(f"Action-head path: {artifacts.action_head_path}")
        print(f"Wrapper dir: {wrapper_dir}")
        print(f"Instruction types: {list(instruction_types)}")
        print(f"Episodes per instruction: {int(args.episodes_per_instruction)}")
        print(f"Episode max steps: {max_steps}")
        print(f"Record success videos: {bool(args.record_success_videos)}")
        print(f"Validation success distance: {float(args.success_distance):.3f} m")
        print(f"Reuse existing wrapper variants: {bool(args.reuse_existing_wrapper_variants)}")
        print(f"Seed mode: {'entropy' if base_seed is None else base_seed}")

    with _temporary_env_vars(_validation_env_vars(config, args)):
        with _silence_output(bool(args.progress_only)):
            runtime = _load_policy_runtime(
                config=config,
                artifacts=artifacts,
                args=args,
                quiet=bool(args.progress_only),
            )

        instruction_summaries: list[InstructionSummary] = []
        instruction_episodes: dict[str, list[dict[str, Any]]] = {}
        progress = _progress_bar(total=len(instruction_types) * int(args.episodes_per_instruction))
        try:
            for instruction_index, instruction_type in enumerate(instruction_types):
                summary, episode_results = _run_instruction_validation(
                    instruction_type=instruction_type,
                    instruction_index=instruction_index,
                    config=config,
                    runtime=runtime,
                    args=args,
                    videos_dir=videos_dir,
                    max_steps=max_steps,
                    base_seed=base_seed,
                    progress=progress,
                    wrapper_dir=wrapper_dir,
                )
                instruction_summaries.append(summary)
                instruction_episodes[instruction_type] = [asdict(result) for result in episode_results]
        finally:
            progress.close()

    manifest = {
        "run_dir": run_dir.as_posix(),
        "generated_at": datetime.now().isoformat(),
        "config_path": Path(args.config).expanduser().resolve().as_posix(),
        "checkpoint_dir": None if artifacts.checkpoint_dir is None else artifacts.checkpoint_dir.as_posix(),
        "adapter_path": artifacts.adapter_path.as_posix(),
        "action_head_path": artifacts.action_head_path.as_posix(),
        "base_checkpoint": args.base_ckpt or config.policy.base_checkpoint,
        "scene": args.scene,
        "wrapper_dir": None if wrapper_dir is None else wrapper_dir.as_posix(),
        "episodes_per_instruction": int(args.episodes_per_instruction),
        "max_steps": int(max_steps),
        "chunk_length": int(runtime["chunk_length"]),
        "num_images_in_input": int(runtime["num_images_in_input"]),
        "center_crop": bool(args.center_crop),
        "hold_steps": int(_control_spec_from_config(config, args.hold_steps).hold_steps),
        "seed": base_seed,
        "record_success_videos": bool(args.record_success_videos),
        "success_distance": float(args.success_distance),
        "reuse_existing_wrapper_variants": bool(args.reuse_existing_wrapper_variants),
        "instruction_summaries": [asdict(summary) for summary in instruction_summaries],
        "episodes": instruction_episodes,
    }

    manifest_path = run_dir / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    csv_path = run_dir / "instruction_success_rates.csv"
    _write_success_rate_csv(csv_path, instruction_summaries)

    if not args.progress_only:
        print(f"Manifest saved: {manifest_path}")
        print(f"CSV saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
