from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
except Exception as exc:  # pragma: no cover - optional local dependency
    _TORCH_IMPORT_ERROR = exc
    torch = None

try:  # pragma: no cover - cosmetic optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional local dependency
    tqdm = None

from rl_vla_bootstrapping.cli.validate_cdpr_policy import (
    EpisodeResult,
    InstructionSummary,
    _aggregate_episode_results,
    _annotate_latest_validation_frame,
    _build_validation_env,
    _default_max_steps,
    _episode_seed,
    _instruction_validation_task_metadata,
    _is_normal_scene_canonical_episode,
    _markdown_table,
    _move_to_object_threshold_sweep,
    _parse_instruction_types,
    _prepend_runtime_python_paths,
    _render_policy_prompt,
    _reset_validation_env_with_retries,
    _resolve_wrapper_dir,
    _save_episode_video,
    _scaled_action_vector,
    _summarize_instruction_results,
    _summarize_instruction_text_results,
    _temporary_env_vars,
    _validation_buckets,
    _write_episode_results_csv,
    _write_grouped_success_rate_csv,
    _write_instruction_text_csv,
    _write_move_to_object_threshold_sweep,
    _write_success_rate_csv,
    _write_video_audit,
)
from rl_vla_bootstrapping.core.commands import ensure_directory
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.octo_cdpr_adapter import CDPRStateLayout
from rl_vla_bootstrapping.policy.octo_finetune_cdpr import ResidualChunkActor
from rl_vla_bootstrapping.policy.smolvla_cdpr import (
    DEFAULT_SMOLVLA_CHECKPOINT,
    cdpr_state_from_observation,
    load_smolvla_runtime,
)
from robots.cdpr.cdpr_dataset.rl_instruction_tasks import INSTRUCTION_TEXT, INSTRUCTION_TYPES
from robots.cdpr.cdpr_dataset.synthetic_tasks import clear_sim_recording_buffers


@dataclass(frozen=True)
class ResolvedSmolVLAArtifacts:
    checkpoint_dir: Path | None
    checkpoint_path: Path
    base_checkpoint: str


class SmolVLACDPREvalRuntime:
    def __init__(
        self,
        *,
        base_checkpoint: str,
        checkpoint_path: Path,
        seed: int,
        image_size: int,
        state_dim: int,
        chunk_size: int,
        action_dim: int,
        action_indices: tuple[int, ...] | None,
        action_normalization: str,
        include_wrist: bool,
        include_aux_camera: bool,
        mixed_precision: str,
        device: Any,
        mask_empty_aux_camera: bool = False,
    ) -> None:
        _require_torch()
        load_t0 = time.perf_counter()
        print(
            f"[smolvla-eval] Loading SmolVLA base checkpoint: {base_checkpoint} "
            f"(device={device}, mixed_precision={mixed_precision})",
            flush=True,
        )
        self.smolvla = load_smolvla_runtime(
            checkpoint=base_checkpoint,
            device=str(device),
            mixed_precision=str(mixed_precision),
            image_size=int(image_size),
            state_dim=int(state_dim),
            include_wrist=bool(include_wrist),
            include_aux_camera=bool(include_aux_camera),
            mask_empty_aux_camera=bool(mask_empty_aux_camera),
            chunk_size=int(chunk_size),
            action_dim=int(action_dim),
            action_indices=action_indices,
            action_normalization=str(action_normalization),
        )
        print(
            f"[smolvla-eval] Loaded SmolVLA base in {time.perf_counter() - load_t0:.1f}s; "
            f"{self.smolvla.device_summary()}",
            flush=True,
        )
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.actor: ResidualChunkActor | None = None
        ckpt_t0 = time.perf_counter()
        print(f"[smolvla-eval] Loading residual adapter checkpoint on CPU: {checkpoint_path}", flush=True)
        self.payload = torch.load(checkpoint_path, map_location="cpu")
        self.actor_state_dim = _checkpoint_state_dim(self.payload)
        self.actor_state = _checkpoint_actor_state(self.payload)
        self.checkpoint_format = "grpo_policy" if "policy" in self.payload else "residual_actor"
        print(
            f"[smolvla-eval] Loaded residual adapter in {time.perf_counter() - ckpt_t0:.1f}s; "
            f"format={self.checkpoint_format}; actor_state_dim={self.actor_state_dim}",
            flush=True,
        )

    def ensure_actor(self, state_dim: int) -> None:
        if self.actor is not None:
            return
        chunk_size = int(self.payload.get("chunk_size", self.smolvla.action_spec.chunk_size))
        action_dim = int(self.payload.get("action_dim", self.smolvla.action_spec.action_dim))
        hidden_dim = int(self.payload.get("hidden_dim", 512))
        residual_scale = float(self.payload.get("residual_scale", 0.35))
        actor_state_dim = int(self.actor_state_dim or state_dim)
        actor = ResidualChunkActor(
            state_dim=actor_state_dim,
            chunk_size=chunk_size,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            residual_scale=residual_scale,
        ).to(self.device)
        print(
            f"[smolvla-eval] Materializing residual actor on {self.device} "
            f"(state_dim={actor_state_dim}, chunk_size={chunk_size}, action_dim={action_dim})",
            flush=True,
        )
        actor.load_state_dict(self.actor_state)
        actor.eval()
        self.actor = actor
        self.actor_state_dim = actor_state_dim
        print("[smolvla-eval] Residual actor ready.", flush=True)

    def predict_chunk(
        self,
        *,
        env: Any,
        obs: dict[str, np.ndarray],
        info: dict[str, Any],
        layout: CDPRStateLayout,
        instruction: str,
    ) -> np.ndarray:
        priors = self.smolvla.sample_cdpr_chunks_from_envs(
            envs=[env],
            observations=[obs],
            infos=[info],
            instructions=[instruction],
        )
        prior = priors[0]
        actor_state_dim = int(self.actor_state_dim or layout.state_dim)
        if actor_state_dim == int(layout.state_dim):
            state = layout.flatten(obs)
        else:
            state = cdpr_state_from_observation(
                obs,
                info,
                state_dim=actor_state_dim,
            )
        self.ensure_actor(int(state.shape[0]))
        assert self.actor is not None
        if self.actor_state_dim is not None and int(state.shape[0]) != int(self.actor_state_dim):
            expected_max_objects = _max_objects_from_state_dim(int(self.actor_state_dim))
            hint = (
                f" Set --max-objects {expected_max_objects} or leave --max-objects unset."
                if expected_max_objects is not None
                else ""
            )
            raise RuntimeError(
                "SmolVLA checkpoint/env state dimension mismatch: "
                f"checkpoint expects state_dim={int(self.actor_state_dim)}, "
                f"but validation env produced state_dim={int(state.shape[0])}.{hint}"
            )
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        prior_t = torch.as_tensor(prior, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            chunk = self.actor(state_t, prior_t)[0].cpu().numpy()
        return np.clip(chunk, -1.0, 1.0).astype(np.float32, copy=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a SmolVLA CDPR adapter checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-name", default="cdpr_smolvla_policy_validation")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--wrapper-dir", default=None)
    parser.add_argument("--episodes-per-instruction", type=int, default=20)
    parser.add_argument("--move-to-object-episodes-per-target", type=int, default=20)
    parser.add_argument("--move-to-object-success-distance", type=float, default=0.025)
    parser.add_argument("--success-distance", type=float, default=0.05)
    parser.add_argument("--directional-displacement-threshold", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--hold-steps", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--mixed-precision", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--smolvla-action-indices", nargs=5, type=int, default=None)
    parser.add_argument("--smolvla-action-normalization", choices=("tanh", "clip", "none"), default="tanh")
    parser.add_argument("--instruction-types", nargs="+", default=None)
    parser.add_argument("--action-guard", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every-episode", type=int, default=10)
    parser.add_argument("--max-reset-attempts", type=int, default=10)
    parser.add_argument("--stratify-move-to-object-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-object-scenes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-scene-objects", type=int, default=3)
    parser.add_argument("--max-scene-objects", type=int, default=4)
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Maximum object slots in the CDPR observation. Defaults to the checkpoint-compatible value.",
    )
    parser.add_argument("--include-synonyms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--synonyms-per-instruction", type=int, default=2)
    parser.add_argument("--synonym-shells", choices=("normal", "all"), default="normal")
    parser.add_argument("--evaluate-reverse-shells", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--arbitrary-instructions-count", type=int, default=0)
    parser.add_argument("--reuse-existing-wrapper-variants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-success-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-failure-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-all-success-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-all-failure-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-coverage", choices=("instruction", "case"), default="instruction")
    parser.add_argument("--video-action-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-video-validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-complete-video-coverage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-wrist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-aux-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-empty-aux-camera", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _require_torch() -> None:
    if torch is None:
        detail = f" Original import error: {_TORCH_IMPORT_ERROR!r}" if _TORCH_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "SmolVLA CDPR validation requires PyTorch to load the residual adapter checkpoint. "
            f"Install it in the remote `smolvla` environment before running evaluation.{detail}"
        )


def _checkpoint_state_dim(payload: dict[str, Any]) -> int | None:
    raw_state_dim = payload.get("state_dim")
    if raw_state_dim is not None:
        try:
            return int(raw_state_dim)
        except (TypeError, ValueError):
            pass

    actor = payload.get("actor")
    first_weight_key = "net.net.0.weight"
    if not isinstance(actor, dict):
        actor = payload.get("policy")
        first_weight_key = "actor.net.net.0.weight"
    if not isinstance(actor, dict):
        return None
    first_weight = actor.get(first_weight_key)
    if first_weight is None or not hasattr(first_weight, "shape"):
        return None
    try:
        input_dim = int(first_weight.shape[1])
        chunk_size = int(payload.get("chunk_size", 8))
        action_dim = int(payload.get("action_dim", 5))
    except (IndexError, TypeError, ValueError):
        return None
    state_dim = input_dim - chunk_size * action_dim
    return state_dim if state_dim > 0 else None


def _checkpoint_actor_state(payload: dict[str, Any]) -> dict[str, Any]:
    actor = payload.get("actor")
    if isinstance(actor, dict):
        return dict(actor)

    policy = payload.get("policy")
    if isinstance(policy, dict):
        actor_state = {
            str(key)[len("actor.") :]: value
            for key, value in policy.items()
            if str(key).startswith("actor.")
        }
        if actor_state:
            return actor_state
    raise KeyError(
        "Unsupported SmolVLA adapter checkpoint: expected TD3 `actor` weights "
        "or GRPO `policy` weights containing the residual `actor.*` module."
    )


def _max_objects_from_state_dim(state_dim: int) -> int | None:
    # Older SmolVLA residual checkpoints were trained before five instruction
    # families were appended to the one-hot catalog. Preserve their 32-wide
    # state layout while also accepting the current catalog width.
    for instruction_width in (len(INSTRUCTION_TYPES), 32):
        fixed_dim = 3 + 3 + int(instruction_width) + 3
        variable_dim = int(state_dim) - fixed_dim
        if variable_dim >= 0 and variable_dim % 4 == 0:
            max_objects = variable_dim // 4
            if max_objects > 0:
                return max_objects
    return None


def _configure_checkpoint_compatible_object_slots(
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> None:
    if args.max_objects is not None:
        args.max_objects = max(1, int(args.max_objects))
        return
    state_dim = _checkpoint_state_dim(payload)
    if state_dim is None:
        return
    max_objects = _max_objects_from_state_dim(state_dim)
    if max_objects is not None:
        args.max_objects = max_objects


def _resolve_checkpoint(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
        return path
    for name in ("smolvla_grpo_adapter.pt", "smolvla_cdpr_adapter.pt", "latest.pt"):
        candidate = path / name
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(path.glob("step_*/smolvla_cdpr_adapter.pt"))
    if matches:
        return matches[-1].resolve()
    raise RuntimeError(f"Could not resolve SmolVLA checkpoint from {path}")


def _resolve_artifacts(args: argparse.Namespace, config: Any) -> ResolvedSmolVLAArtifacts:
    checkpoint_path = _resolve_checkpoint(args.checkpoint_dir)
    checkpoint_dir = checkpoint_path.parent if checkpoint_path.name != "latest.pt" else checkpoint_path.parent
    base_checkpoint = str(args.base_checkpoint or config.policy.base_checkpoint or DEFAULT_SMOLVLA_CHECKPOINT)
    return ResolvedSmolVLAArtifacts(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        base_checkpoint=base_checkpoint,
    )


def _resolve_instruction_types(config: Any, args: argparse.Namespace) -> tuple[str, ...]:
    raw_values = getattr(args, "instruction_types", None)
    if raw_values:
        return _parse_instruction_types(raw_values)
    configured = tuple(getattr(config.task, "instruction_types", ()) or ())
    if configured:
        return _parse_instruction_types(configured)
    return _parse_instruction_types(("all",))


def _episode_result_from_final(
    *,
    episode_index: int,
    seed: int | None,
    bucket: Any,
    instruction: str,
    canonical_instruction: str,
    reset_info: dict[str, Any],
    final_info: dict[str, Any],
    reward_total: float,
    terminated: bool,
    truncated: bool,
    policy_output_calls: int,
    action_steps: int,
    reset_attempts: int,
    initial_move_to_object_distance_xy: float,
    min_move_to_object_distance_xy: float,
) -> EpisodeResult:
    initial_distance = (
        None
        if not np.isfinite(initial_move_to_object_distance_xy)
        else float(initial_move_to_object_distance_xy)
    )
    final_distance = (
        None
        if final_info.get("move_to_object_validation_distance_xy") is None
        else float(final_info["move_to_object_validation_distance_xy"])
    )
    min_distance = (
        None
        if not np.isfinite(min_move_to_object_distance_xy)
        else float(min_move_to_object_distance_xy)
    )
    return EpisodeResult(
        episode_index=int(episode_index),
        seed=seed,
        instruction_type=str(bucket.instruction_type),
        instruction_text=str(instruction),
        success=bool(final_info.get("success", False)),
        terminated=bool(terminated),
        truncated=bool(truncated),
        steps=int(final_info.get("step", action_steps)),
        reward_total=float(reward_total),
        scene=str(final_info.get("scene", "")),
        goal_position=[float(value) for value in final_info.get("goal_position", [])],
        ee_start=[float(value) for value in final_info.get("ee_start", [])],
        target_object_catalog=str(final_info.get("target_object_catalog", reset_info.get("target_object_catalog", "")))
        or None,
        reference_object_catalog=str(
            final_info.get("reference_object_catalog", reset_info.get("reference_object_catalog", ""))
        )
        or None,
        second_reference_object_catalog=str(
            final_info.get("second_reference_object_catalog", reset_info.get("second_reference_object_catalog", ""))
        )
        or None,
        scene_objects=tuple(str(value) for value in final_info.get("scene_objects", reset_info.get("scene_objects", ()))),
        canonical_instruction_text=str(canonical_instruction),
        prompt_kind=str(bucket.prompt_kind),
        prompt_variant=str(bucket.prompt_variant),
        curriculum_shell=bucket.curriculum_shell,
        curriculum_shell_count=bucket.curriculum_shell_count,
        metric_episode=True,
        policy_output_calls=int(policy_output_calls),
        action_steps=int(action_steps),
        reset_attempts=int(reset_attempts),
        simulation_instability=bool(final_info.get("simulation_instability", False)),
        final_ee_position=tuple(float(value) for value in final_info.get("ee_position", ())),
        final_ee_yaw=float(final_info.get("ee_yaw", 0.0)),
        final_gripper_opening=float(final_info.get("gripper_opening", 0.0)),
        final_gripper_target=float(final_info.get("gripper_target", 0.0)),
        initial_move_to_object_distance_xy=initial_distance,
        final_move_to_object_distance_xy=final_distance,
        min_move_to_object_distance_xy=min_distance,
        final_move_to_object_distance_reduction_xy=(
            None
            if initial_distance is None or final_distance is None
            else float(initial_distance - final_distance)
        ),
        best_move_to_object_distance_reduction_xy=(
            None
            if initial_distance is None or min_distance is None
            else float(initial_distance - min_distance)
        ),
        move_to_object_distance_threshold=(
            None
            if final_info.get("move_to_object_validation_distance_threshold") is None
            else float(final_info["move_to_object_validation_distance_threshold"])
        ),
    )


def _video_coverage_key(args: argparse.Namespace, bucket: Any) -> str:
    if str(getattr(args, "video_coverage", "instruction")) == "instruction":
        return str(bucket.instruction_type)
    return str(bucket.case_id)


def _progress_message(progress: Any | None, message: str) -> None:
    if progress is not None and hasattr(progress, "write"):
        progress.write(message)
    else:
        print(message, flush=True)


def _move_to_object_distance_xy(info: dict[str, Any]) -> float:
    raw_distance = info.get("move_to_object_validation_distance_xy")
    if raw_distance is not None:
        try:
            return float(raw_distance)
        except (TypeError, ValueError):
            pass
    ee = np.asarray(info.get("ee_position", ()), dtype=np.float32).reshape(-1)
    target = np.asarray(
        info.get("goal_position", info.get("target_object_position_actual", ())),
        dtype=np.float32,
    ).reshape(-1)
    if ee.size >= 2 and target.size >= 2:
        return float(np.linalg.norm(target[:2] - ee[:2]))
    return float("inf")


def _run_bucket(
    *,
    config: Any,
    runtime: SmolVLACDPREvalRuntime,
    args: argparse.Namespace,
    bucket: Any,
    instruction_index: int,
    base_seed: int | None,
    max_steps: int,
    wrapper_dir: Path | None,
    videos_dir: Path,
    video_registry: dict[str, dict[str, str]],
    progress: Any | None = None,
    progress_state: dict[str, int] | None = None,
) -> list[EpisodeResult]:
    coverage_key = _video_coverage_key(args, bucket)
    coverage_entry = video_registry.setdefault(coverage_key, {})
    progress_state = progress_state if progress_state is not None else {"completed": 0, "successes": 0}
    should_capture = bool(
        args.record_success_videos
        or args.record_failure_videos
        or args.record_all_success_videos
        or args.record_all_failure_videos
        or bucket.force_video
    )
    with _temporary_env_vars(bucket.env_vars):
        env = _build_validation_env(
            config=config,
            instruction_type=bucket.instruction_type,
            capture_frames=should_capture,
            max_steps=max_steps,
            hold_steps=args.hold_steps,
            seed=base_seed,
            args=args,
            wrapper_dir=wrapper_dir,
        )
        reset_options: dict[str, Any] = {"instruction_type": bucket.instruction_type}
        if args.scene:
            reset_options["scene"] = args.scene
        if bucket.target_object:
            reset_options["target_object"] = str(bucket.target_object)
        if bucket.curriculum_shell is not None:
            reset_options["curriculum_mode"] = "reverse_frontier"
            reset_options["curriculum_shell"] = int(bucket.curriculum_shell)

        results: list[EpisodeResult] = []
        try:
            for episode_index in range(int(bucket.episodes)):
                needs_success_video = bool(
                    (args.record_success_videos or bucket.force_video)
                    and (args.record_all_success_videos or coverage_entry.get("success") is None)
                )
                needs_failure_video = bool(
                    (args.record_failure_videos or args.record_all_failure_videos or bucket.force_video)
                    and (
                        args.record_all_failure_videos
                        or coverage_entry.get("failure") is None
                    )
                )
                env.capture_frames = bool(bucket.force_video or needs_success_video or needs_failure_video)
                seed = _episode_seed(base_seed, instruction_index, episode_index)
                obs, reset_info, reset_attempts = _reset_validation_env_with_retries(
                    env=env,
                    seed=seed,
                    reset_options=reset_options,
                    max_attempts=int(args.max_reset_attempts),
                    quiet=bool(args.progress_only),
                )
                layout = CDPRStateLayout.from_observation(obs)
                canonical_instruction = str(
                    reset_info.get("language_instruction", INSTRUCTION_TEXT[bucket.instruction_type])
                )
                instruction = _render_policy_prompt(
                    prompt_template=bucket.prompt_template,
                    canonical_instruction=canonical_instruction,
                    reset_info=dict(reset_info),
                )
                setattr(env.sim, "language_instruction", instruction)

                current_chunk = np.zeros((0, 5), dtype=np.float32)
                chunk_index = 0
                reward_total = 0.0
                terminated = False
                truncated = False
                final_info = dict(reset_info)
                policy_output_calls = 0
                action_steps = 0
                action_trace: list[dict[str, Any]] = []
                initial_move_to_object_distance_xy = _move_to_object_distance_xy(reset_info)
                min_move_to_object_distance_xy = initial_move_to_object_distance_xy
                replan_every = max(1, min(int(args.replan_every), int(runtime.smolvla.action_spec.chunk_size)))

                while not (terminated or truncated):
                    if chunk_index >= len(current_chunk) or chunk_index >= replan_every:
                        current_chunk = runtime.predict_chunk(
                            env=env,
                            obs=obs,
                            info=final_info,
                            layout=layout,
                            instruction=instruction,
                        )
                        chunk_index = 0
                        policy_output_calls += 1
                    action = np.asarray(current_chunk[chunk_index], dtype=np.float32).reshape(5)
                    chunk_index += 1
                    if float(np.max(np.abs(action))) > float(args.action_guard) and not args.progress_only:
                        print(
                            f"[warn] [{bucket.log_label}] episode={episode_index:03d} "
                            "action max abs exceeded guard; clipping.",
                            flush=True,
                        )
                    action = np.clip(action, -1.0, 1.0)
                    obs, reward, terminated, truncated, final_info = env.step(action)
                    reward_total += float(reward)
                    action_steps += 1
                    distance_xy = _move_to_object_distance_xy(final_info)
                    min_move_to_object_distance_xy = min(
                        min_move_to_object_distance_xy,
                        distance_xy,
                    )
                    scaled_action = _scaled_action_vector(action, config, args.hold_steps)
                    ee_pos = final_info.get("ee_position", ())
                    ee_xyz = (
                        np.asarray(ee_pos, dtype=np.float32).reshape(-1)[:3]
                        if ee_pos is not None
                        else np.zeros((0,))
                    )
                    target_pos = np.asarray(
                        final_info.get(
                            "goal_position",
                            final_info.get("target_object_position_actual", ()),
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    distance_threshold = final_info.get(
                        "move_to_object_validation_distance_threshold",
                        "",
                    )
                    action_trace.append(
                        {
                            "step": int(action_steps),
                            "policy_call": int(policy_output_calls),
                            "new_policy_output": int(chunk_index == 1),
                            "chunk_action_index": int(chunk_index - 1),
                            "chunk_length": int(len(current_chunk)),
                            "action_x": float(action[0]),
                            "action_y": float(action[1]),
                            "action_z": float(action[2]),
                            "action_yaw": float(action[3]),
                            "action_gripper": float(action[4]),
                            "applied_dx": float(scaled_action[0]),
                            "applied_dy": float(scaled_action[1]),
                            "applied_dz": float(scaled_action[2]),
                            "applied_dyaw": float(scaled_action[3]),
                            "applied_dgripper": float(scaled_action[4]),
                            "ee_x": float(ee_xyz[0]) if ee_xyz.size >= 1 else "",
                            "ee_y": float(ee_xyz[1]) if ee_xyz.size >= 2 else "",
                            "ee_z": float(ee_xyz[2]) if ee_xyz.size >= 3 else "",
                            "ee_yaw": final_info.get("ee_yaw", ""),
                            "gripper_opening": final_info.get("gripper_opening", ""),
                            "gripper_target": final_info.get("gripper_target", ""),
                            "reward": float(reward),
                            "reward_total": float(reward_total),
                            "target_x": float(target_pos[0]) if target_pos.size >= 1 else "",
                            "target_y": float(target_pos[1]) if target_pos.size >= 2 else "",
                            "target_z": float(target_pos[2]) if target_pos.size >= 3 else "",
                            "move_to_object_distance_xy": (
                                float(distance_xy) if np.isfinite(distance_xy) else ""
                            ),
                            "move_to_object_min_distance_xy": (
                                float(min_move_to_object_distance_xy)
                                if np.isfinite(min_move_to_object_distance_xy)
                                else ""
                            ),
                            "move_to_object_initial_distance_xy": (
                                float(initial_move_to_object_distance_xy)
                                if np.isfinite(initial_move_to_object_distance_xy)
                                else ""
                            ),
                            "move_to_object_distance_reduction_xy": (
                                float(initial_move_to_object_distance_xy - distance_xy)
                                if np.isfinite(initial_move_to_object_distance_xy)
                                and np.isfinite(distance_xy)
                                else ""
                            ),
                            "move_to_object_distance_threshold": distance_threshold,
                            "success": int(bool(final_info.get("success", False))),
                            "simulation_state_valid": int(bool(final_info.get("simulation_state_valid", True))),
                        }
                    )
                    if env.capture_frames and bool(args.video_action_overlay):
                        _annotate_latest_validation_frame(
                            sim=env.sim,
                            instruction=instruction,
                            step=action_steps,
                            policy_call=policy_output_calls,
                            chunk_action_index=int(chunk_index - 1),
                            chunk_length=int(len(current_chunk)),
                            new_policy_output=bool(chunk_index == 1),
                            action=action,
                            scaled_action=scaled_action,
                            info=dict(final_info),
                            reward=float(reward),
                            reward_total=float(reward_total),
                            initial_move_to_object_distance_xy=(
                                None
                                if not np.isfinite(initial_move_to_object_distance_xy)
                                else float(initial_move_to_object_distance_xy)
                            ),
                            min_move_to_object_distance_xy=(
                                None
                                if not np.isfinite(min_move_to_object_distance_xy)
                                else float(min_move_to_object_distance_xy)
                            ),
                        )

                result = _episode_result_from_final(
                    episode_index=episode_index,
                    seed=seed,
                    bucket=bucket,
                    instruction=instruction,
                    canonical_instruction=canonical_instruction,
                    reset_info=reset_info,
                    final_info=final_info,
                    reward_total=reward_total,
                    terminated=terminated,
                    truncated=truncated,
                    policy_output_calls=policy_output_calls,
                    action_steps=action_steps,
                    reset_attempts=reset_attempts,
                    initial_move_to_object_distance_xy=initial_move_to_object_distance_xy,
                    min_move_to_object_distance_xy=min_move_to_object_distance_xy,
                )
                saved_video_path: str | None = None
                saved_video_kind: str | None = None
                try:
                    if result.success and needs_success_video:
                        saved_video_path = _save_episode_video(
                            sim=env.sim,
                            output_dir=videos_dir,
                            instruction_type=str(bucket.instruction_type),
                            episode_result=result,
                            outcome="success",
                            action_trace=action_trace,
                        )
                        if saved_video_path and coverage_entry.get("success") is None:
                            coverage_entry["success"] = saved_video_path
                        saved_video_kind = "success" if saved_video_path else None
                    elif (not result.success) and needs_failure_video:
                        saved_video_path = _save_episode_video(
                            sim=env.sim,
                            output_dir=videos_dir,
                            instruction_type=str(bucket.instruction_type),
                            episode_result=result,
                            outcome="failure",
                            action_trace=action_trace,
                        )
                        if saved_video_path and coverage_entry.get("failure") is None:
                            coverage_entry["failure"] = saved_video_path
                        saved_video_kind = "failure" if saved_video_path else None
                finally:
                    clear_sim_recording_buffers(env.sim)
                if saved_video_path:
                    result = replace(
                        result,
                        video_path=saved_video_path,
                        video_kind=saved_video_kind,
                        action_trace_path=saved_video_path.replace("_overview.mp4", "_actions.csv"),
                    )
                results.append(result)
                progress_state["completed"] = int(progress_state.get("completed", 0)) + 1
                progress_state["successes"] = int(progress_state.get("successes", 0)) + int(result.success)
                if progress is not None:
                    progress.set_postfix(
                        instruction=str(bucket.instruction_type),
                        success=f"{sum(item.success for item in results)}/{len(results)}",
                        overall=f"{progress_state['successes']}/{progress_state['completed']}",
                        refresh=False,
                    )
                    progress.update(1)
                if (episode_index + 1) % max(1, int(args.log_every_episode)) == 0:
                    successes = sum(item.success for item in results)
                    _progress_message(
                        progress,
                        f"[smolvla-eval] {bucket.log_label} {episode_index + 1}/{bucket.episodes} "
                        f"success={successes}/{len(results)}",
                    )
        finally:
            env.close()
    return results


def _write_smolvla_report(
    *,
    run_dir: Path,
    artifacts: ResolvedSmolVLAArtifacts,
    metric_results: list[EpisodeResult],
    instruction_rows: list[dict[str, Any]],
    normal_canonical_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    move_to_object_threshold_rows: list[dict[str, Any]],
) -> Path:
    successes = sum(item.success for item in metric_results)
    move_rows = [row for row in normal_canonical_rows if row.get("instruction_type") == "move_to_object"]
    move_rate = float(move_rows[0]["success_rate"]) if move_rows else 0.0
    overall_rate = float(successes / max(len(metric_results), 1))
    lines = [
        "# CDPR SmolVLA validation report",
        "",
        f"- Checkpoint: `{artifacts.checkpoint_path}`",
        f"- Base SmolVLA checkpoint: `{artifacts.base_checkpoint}`",
        f"- Metric episodes: `{len(metric_results)}`",
        f"- Overall successes: `{successes}`",
        f"- Overall success rate: `{overall_rate:.4f}`",
        f"- Move-to-object canonical normal success rate: `{move_rate:.4f}`",
        "- OpenVLA beat threshold: overall `>0.167`, move_to_object `>0.090`.",
        "",
        "## Instruction success rates",
        "",
        _markdown_table(instruction_rows, ("instruction_type", "successes", "episodes", "success_rate", "mean_steps")),
        "",
        "## Canonical normal-scene success rates",
        "",
        _markdown_table(normal_canonical_rows, ("instruction_type", "successes", "episodes", "success_rate", "mean_steps")),
        "",
        "## Target-object success rates",
        "",
        _markdown_table(target_rows, ("instruction_type", "target_object_catalog", "successes", "episodes", "success_rate")),
        "",
        "## Move-to-object tolerance sweep",
        "",
        _markdown_table(move_to_object_threshold_rows, ("distance_threshold_m", "successes", "episodes", "success_rate")),
        "",
        "## Artifacts",
        "",
        "- `validation_manifest.json`",
        "- `episode_results.csv`",
        "- `instruction_success_rates.csv`",
        "- `normal_scene_canonical_success_rates.csv`",
        "- `instruction_prompt_success_rates.csv`",
        "- `target_object_success_rates.csv`",
        "- `instruction_text_success_rates.csv`",
        "- `move_to_object_threshold_sweep.csv`",
        "- `video_coverage.csv`",
        "- `video_validation.csv` and `video_validation.json`",
        "- `videos/`",
        "",
    ]
    path = run_dir / "validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_empty_video_files(run_dir: Path) -> None:
    for filename in ("video_validation.csv", "video_coverage.csv"):
        with (run_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["note"])
            writer.writerow(["SmolVLA validator did not request video capture."])
    (run_dir / "video_validation.json").write_text("[]\n", encoding="utf-8")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _require_torch()
    config = load_project_config(args.config)
    _prepend_runtime_python_paths(config)
    artifacts = _resolve_artifacts(args, config)

    run_dir = (
        ensure_directory(Path(args.run_dir).expanduser().resolve())
        if args.run_dir
        else ensure_directory(
            (config.resolve_path(config.project.output_root) or Path("runs"))
            / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    max_steps = _default_max_steps(config, args)
    wrapper_dir = _resolve_wrapper_dir(config, args)
    videos_dir = ensure_directory(run_dir / "videos")
    chunk_size = int(args.chunk_size or config.policy.action_codec.chunk_size)
    runtime = SmolVLACDPREvalRuntime(
        base_checkpoint=artifacts.base_checkpoint,
        checkpoint_path=artifacts.checkpoint_path,
        seed=int(args.seed),
        image_size=int(args.image_size),
        state_dim=int(args.state_dim),
        chunk_size=chunk_size,
        action_dim=5,
        action_indices=None if args.smolvla_action_indices is None else tuple(args.smolvla_action_indices),
        action_normalization=str(args.smolvla_action_normalization),
        include_wrist=bool(args.include_wrist),
        include_aux_camera=bool(args.include_aux_camera),
        mask_empty_aux_camera=bool(args.mask_empty_aux_camera),
        mixed_precision=str(args.mixed_precision),
        device=torch.device(args.device),
    )
    _configure_checkpoint_compatible_object_slots(args, runtime.payload)

    instruction_types = _resolve_instruction_types(config, args)
    instruction_plan = [
        (instruction_index, instruction_type, _validation_buckets(config, args, instruction_type=instruction_type))
        for instruction_index, instruction_type in enumerate(instruction_types)
    ]
    planned_episodes = sum(int(bucket.episodes) for _idx, _instruction_type, buckets in instruction_plan for bucket in buckets)
    print(
        f"[smolvla-eval] Starting validation: {planned_episodes} episode(s), "
        f"{len(instruction_plan)} instruction type(s), progress={bool(args.progress)}, "
        f"progress_only={bool(args.progress_only)}, "
        f"log_every_episode={int(args.log_every_episode)}",
        flush=True,
    )
    progress = None
    if bool(args.progress):
        if tqdm is None:
            print("[smolvla-eval] tqdm is not installed; falling back to periodic text progress.", flush=True)
        else:
            progress = tqdm(
                total=int(planned_episodes),
                desc="SmolVLA eval",
                unit="ep",
                dynamic_ncols=True,
                leave=True,
            )
    progress_state = {"completed": 0, "successes": 0}
    all_results: list[EpisodeResult] = []
    video_registry: dict[str, dict[str, str]] = {}
    expected_video_keys: list[str] = []
    try:
        for instruction_index, instruction_type, buckets in instruction_plan:
            for bucket in buckets:
                expected_video_keys.append(_video_coverage_key(args, bucket))
                results = _run_bucket(
                    config=config,
                    runtime=runtime,
                    args=args,
                    bucket=bucket,
                    instruction_index=instruction_index,
                    base_seed=None if int(args.seed) < 0 else int(args.seed),
                    max_steps=max_steps,
                    wrapper_dir=wrapper_dir,
                    videos_dir=videos_dir,
                    video_registry=video_registry,
                    progress=progress,
                    progress_state=progress_state,
                )
                all_results.extend(results)
    finally:
        if progress is not None:
            progress.close()
    expected_video_keys = list(dict.fromkeys(expected_video_keys))

    metric_results = [item for item in all_results if item.metric_episode]
    instruction_summaries: list[InstructionSummary] = []
    for instruction_type in instruction_types:
        items = [item for item in metric_results if item.instruction_type == instruction_type]
        success_video_path = next((item.video_path for item in items if item.video_kind == "success"), None)
        failure_video_path = next((item.video_path for item in items if item.video_kind == "failure"), None)
        instruction_summaries.append(
            _summarize_instruction_results(
                instruction_type=instruction_type,
                episode_results=items,
                video_path=success_video_path or failure_video_path,
                success_video_path=success_video_path,
                failure_video_path=failure_video_path,
            )
        )
    instruction_rows = _aggregate_episode_results(metric_results, group_fields=("instruction_type",))
    normal_canonical_results = [item for item in metric_results if _is_normal_scene_canonical_episode(item)]
    normal_canonical_rows = _aggregate_episode_results(normal_canonical_results, group_fields=("instruction_type",))
    prompt_rows = _aggregate_episode_results(
        metric_results,
        group_fields=("instruction_type", "prompt_kind", "prompt_variant"),
    )
    target_rows = _aggregate_episode_results(
        metric_results,
        group_fields=("instruction_type", "target_object_catalog"),
    )
    text_summaries = _summarize_instruction_text_results(metric_results)
    move_to_object_threshold_rows = _move_to_object_threshold_sweep(
        metric_results,
        thresholds=tuple(
            sorted(
                {
                    float(args.move_to_object_success_distance),
                    0.025,
                    0.05,
                    0.10,
                    0.15,
                }
            )
        ),
    )

    _write_episode_results_csv(run_dir / "episode_results.csv", all_results)
    _write_success_rate_csv(run_dir / "instruction_success_rates.csv", instruction_summaries)
    _write_grouped_success_rate_csv(
        run_dir / "normal_scene_canonical_success_rates.csv",
        normal_canonical_rows,
        group_fields=("instruction_type",),
    )
    _write_grouped_success_rate_csv(
        run_dir / "instruction_prompt_success_rates.csv",
        prompt_rows,
        group_fields=("instruction_type", "prompt_kind", "prompt_variant"),
    )
    _write_grouped_success_rate_csv(
        run_dir / "target_object_success_rates.csv",
        target_rows,
        group_fields=("instruction_type", "target_object_catalog"),
    )
    _write_instruction_text_csv(run_dir / "instruction_text_success_rates.csv", text_summaries)
    _write_move_to_object_threshold_sweep(run_dir / "move_to_object_threshold_sweep.csv", move_to_object_threshold_rows)
    if bool(args.record_success_videos or args.record_failure_videos or args.record_all_success_videos):
        video_probes, video_coverage = _write_video_audit(
            run_dir=run_dir,
            videos_dir=videos_dir,
            expected_keys=expected_video_keys,
            video_registry=video_registry,
        )
    else:
        _write_empty_video_files(run_dir)
        video_probes, video_coverage = [], []
    report_path = _write_smolvla_report(
        run_dir=run_dir,
        artifacts=artifacts,
        metric_results=metric_results,
        instruction_rows=instruction_rows,
        normal_canonical_rows=normal_canonical_rows,
        target_rows=target_rows,
        move_to_object_threshold_rows=move_to_object_threshold_rows,
    )
    manifest = {
        "policy_type": "smolvla_cdpr",
        "base_checkpoint": artifacts.base_checkpoint,
        "checkpoint_path": artifacts.checkpoint_path.as_posix(),
        "checkpoint_dir": artifacts.checkpoint_dir.as_posix() if artifacts.checkpoint_dir else None,
        "run_dir": run_dir.as_posix(),
        "report_path": report_path.as_posix(),
        "instruction_types": list(instruction_types),
        "episodes": len(metric_results),
        "successes": sum(item.success for item in metric_results),
        "success_rate": sum(item.success for item in metric_results) / max(len(metric_results), 1),
        "task_metadata": _instruction_validation_task_metadata(config, args),
        "record_success_videos": bool(args.record_success_videos),
        "record_failure_videos": bool(args.record_failure_videos),
        "record_all_success_videos": bool(args.record_all_success_videos),
        "record_all_failure_videos": bool(args.record_all_failure_videos),
        "video_coverage_level": str(args.video_coverage),
        "recorded_videos": int(len(video_probes)),
        "video_registry": video_registry,
        "video_validation": video_probes,
        "video_coverage": video_coverage,
    }
    (run_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    invalid_videos = [probe for probe in video_probes if not bool(probe.get("valid"))]
    incomplete_coverage = [item for item in video_coverage if not bool(item.get("complete"))]
    if bool(args.strict_video_validation) and invalid_videos:
        raise RuntimeError(f"SmolVLA video validation failed for {len(invalid_videos)} MP4 file(s).")
    if bool(args.require_complete_video_coverage) and incomplete_coverage:
        raise RuntimeError(f"SmolVLA video coverage is incomplete for {len(incomplete_coverage)} key(s).")
    print(f"SmolVLA validation output: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
