from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageDraw = None
    ImageFont = None
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional runtime dependency
    tqdm = None

from rl_vla_bootstrapping.cli.run_cdpr_policy import (
    _control_spec_from_config,
    _load_openvla_modules,
    _make_observation,
    _normalize_policy_chunk,
    _predict_normalized_action_chunk,
    _resolve_llm_dim,
    _set_num_images_in_input,
)
from rl_vla_bootstrapping.core.commands import ensure_directory
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.openvla_actor_critic import configure_openvla_dimension_env_from_config
from rl_vla_bootstrapping.policy.openvla_oft import (
    _allowed_objects_from_config,
    _extract_cdpr_env_overrides,
    _resolve_desk_textures_dir,
    _task_hook_env,
)
from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv
from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
    INSTRUCTION_TEXT,
    INSTRUCTION_TYPES,
    canonical_object_name,
)
from robots.cdpr.cdpr_dataset.synthetic_tasks import clear_sim_recording_buffers


_ACTION_HEAD_FILENAMES = (
    "action_head.pt",
    "action_head_cdpr.pt",
    "action_head_latest.pt",
)
_VALIDATION_VIDEO_FPS = 10.0
_MAX_SYNTHETIC_VIDEO_FRAMES = 600


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
    target_object_catalog: str | None = None
    reference_object_catalog: str | None = None
    second_reference_object_catalog: str | None = None
    scene_objects: tuple[str, ...] = ()
    canonical_instruction_text: str | None = None
    prompt_kind: str = "canonical"
    prompt_variant: str = "canonical"
    curriculum_shell: int | None = None
    curriculum_shell_count: int | None = None
    metric_episode: bool = True
    policy_output_calls: int = 0
    action_steps: int = 0
    reset_attempts: int = 1
    simulation_instability: bool = False
    final_ee_position: tuple[float, ...] = ()
    final_ee_yaw: float | None = None
    final_gripper_opening: float | None = None
    final_gripper_target: float | None = None
    final_move_to_object_distance_xy: float | None = None
    min_move_to_object_distance_xy: float | None = None
    move_to_object_distance_threshold: float | None = None
    action_trace_path: str | None = None
    video_path: str | None = None
    video_kind: str | None = None


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
    success_video_path: str | None = None
    failure_video_path: str | None = None


@dataclass(frozen=True)
class InstructionTextSummary:
    instruction_text: str
    instruction_types: tuple[str, ...]
    target_object_catalogs: tuple[str, ...]
    successes: int
    episodes: int
    success_rate: float
    mean_reward: float
    mean_steps: float


@dataclass(frozen=True)
class ValidationBucket:
    instruction_type: str
    target_object: str | None
    episodes: int
    env_vars: dict[str, str]
    log_label: str
    prompt_kind: str = "canonical"
    prompt_variant: str = "canonical"
    prompt_template: str | None = None
    curriculum_shell: int | None = None
    curriculum_shell_count: int | None = None
    force_video: bool = False

    @property
    def case_id(self) -> str:
        parts = [
            self.instruction_type,
            f"shell_{self.curriculum_shell:02d}" if self.curriculum_shell is not None else "no_shell",
            self.prompt_kind,
            self.prompt_variant,
        ]
        if self.target_object:
            parts.append(self.target_object)
        return "__".join(_safe_filename_token(part) for part in parts if part)


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
        help=(
            "How many metric episodes to run per evaluation case. A case is an instruction, "
            "shell, prompt variant, and optional fixed target combination."
        ),
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
    parser.add_argument(
        "--replan-every",
        type=int,
        default=None,
        help="Consume only the first N actions from each predicted chunk before replanning. Defaults to full chunk length.",
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
        "--instruction-types",
        nargs="+",
        default=None,
        help=(
            "Optional instruction override. Accepts internal names such as `move_left`, "
            "`move_top`, `move_to_object`, or human-friendly aliases like `left`, `right`, "
            "`forward`, `backward`, `move forward`, `move backward`."
        ),
    )
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
        "--record-failure-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one overview video for the first failed episode of each instruction type.",
    )
    parser.add_argument(
        "--record-all-success-videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save an overview video for every successful episode instead of only the first success.",
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
        "--move-to-object-success-distance",
        type=float,
        default=0.10,
        help="Validation XY success tolerance in meters for `move to <object>`.",
    )
    parser.add_argument(
        "--directional-displacement-threshold",
        type=float,
        default=0.05,
        help=(
            "Validation threshold in meters for directional commands. "
            "Left/right/forward/backward use signed distance from the workspace center; "
            "up/down use signed displacement from the episode start."
        ),
    )
    parser.add_argument(
        "--move-to-object-episodes-per-target",
        type=int,
        default=50,
        help=(
            "Minimum validation episodes to run for each target object when validating "
            "`move to <object>`. The validator will increase the total episode budget "
            "for that instruction type as needed."
        ),
    )
    parser.add_argument(
        "--stratify-move-to-object-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a separate move-to-object case for every configured target object.",
    )
    parser.add_argument(
        "--multi-object-scenes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep randomized distractors in validation scenes, including move-to-object cases. "
            "Use this to test whether the policy distinguishes the named target."
        ),
    )
    parser.add_argument(
        "--min-scene-objects",
        type=int,
        default=3,
        help="Minimum scene object count when --multi-object-scenes is enabled.",
    )
    parser.add_argument(
        "--max-scene-objects",
        type=int,
        default=4,
        help="Maximum scene object count when --multi-object-scenes is enabled.",
    )
    parser.add_argument(
        "--evaluate-reverse-shells",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate every reverse-frontier curriculum shell for supported instruction types.",
    )
    parser.add_argument(
        "--include-synonyms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate natural-language synonym/paraphrase prompts in addition to canonical prompts.",
    )
    parser.add_argument(
        "--synonyms-per-instruction",
        type=int,
        default=2,
        help="Maximum synonym prompt variants per instruction type.",
    )
    parser.add_argument(
        "--synonym-shells",
        choices=("normal", "all"),
        default="normal",
        help="Evaluate synonyms only on the final normal-reset shell, or on every shell.",
    )
    parser.add_argument(
        "--arbitrary-instructions-count",
        type=int,
        default=0,
        help=(
            "Add this many deterministic free-form prompt checks on normal randomized scenes. "
            "Each arbitrary check runs one episode and is always recorded."
        ),
    )
    parser.add_argument(
        "--video-coverage",
        choices=("instruction", "case"),
        default="instruction",
        help="Require success/failure video examples per instruction type or per exact evaluation case.",
    )
    parser.add_argument(
        "--video-search-extra-episodes",
        type=int,
        default=0,
        help=(
            "After metric evaluation, run up to this many extra canonical normal-scene attempts "
            "per instruction to fill missing success/failure video coverage. These attempts do not "
            "change reported success rates."
        ),
    )
    parser.add_argument(
        "--strict-video-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero after saving the report if any recorded MP4 fails ffprobe validation.",
    )
    parser.add_argument(
        "--max-reset-attempts",
        type=int,
        default=10,
        help="Retry randomized episode reset this many times when MuJoCo produces an invalid state.",
    )
    parser.add_argument(
        "--video-action-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Burn OpenVLA call/chunk/action telemetry into every recorded overview frame.",
    )
    parser.add_argument(
        "--require-complete-video-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero after saving the report if any requested success/failure example is missing.",
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


_INSTRUCTION_TYPE_ALIASES: dict[str, str] = {
    "up": "move_up",
    "move_up": "move_up",
    "down": "move_down",
    "move_down": "move_down",
    "left": "move_left",
    "move_left": "move_left",
    "right": "move_right",
    "move_right": "move_right",
    "forward": "move_top",
    "move_forward": "move_top",
    "top": "move_top",
    "move_top": "move_top",
    "backward": "move_bottom",
    "move_backward": "move_bottom",
    "bottom": "move_bottom",
    "move_bottom": "move_bottom",
    "center": "move_center",
    "centre": "move_center",
    "move_center": "move_center",
    "move_centre": "move_center",
    "object": "move_to_object",
    "move_to": "move_to_object",
    "move_to_object": "move_to_object",
    "pick": "pick_up",
    "pickup": "pick_up",
    "pick_up": "pick_up",
    "grab": "grab_object",
    "grab_object": "grab_object",
    "catch": "catch_object",
    "catch_object": "catch_object",
    "grip": "grip_object",
    "grip_object": "grip_object",
    "release": "release_object",
    "release_object": "release_object",
    "free": "free_object",
    "free_object": "free_object",
    "open_gripper": "open_gripper",
    "close_gripper": "close_gripper",
    "rotate_gripper_clockwise": "rotate_gripper_clockwise",
    "rotate_gripper_counterclockwise": "rotate_gripper_counterclockwise",
    "rotate_clockwise": "rotate_clockwise",
    "clockwise": "rotate_clockwise",
    "rotate_counterclockwise": "rotate_counterclockwise",
    "counterclockwise": "rotate_counterclockwise",
    "left_of_object": "move_left_of_object",
    "move_left_of_object": "move_left_of_object",
    "right_of_object": "move_right_of_object",
    "move_right_of_object": "move_right_of_object",
    "move_front_of_object": "move_in_front_of_object",
    "move_in_front_of_object": "move_in_front_of_object",
    "move_behind_object": "move_behind_object",
    "front_of_object": "put_in_front_of_object",
    "in_front_of_object": "put_in_front_of_object",
    "put_in_front_of_object": "put_in_front_of_object",
    "behind_object": "put_behind_object",
    "put_behind_object": "put_behind_object",
    "push_left": "push_left",
    "push_right": "push_right",
    "push_forward": "push_forward",
    "push_backward": "push_backward",
    "between_objects": "move_between_objects",
    "move_between_objects": "move_between_objects",
}

_SYNONYM_PROMPT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "move_left": (
        "move the end effector left",
        "shift the gripper to the left",
    ),
    "move_right": (
        "move the end effector right",
        "shift the gripper to the right",
    ),
    "move_top": (
        "move the end effector forward",
        "shift the gripper away from the robot",
    ),
    "move_bottom": (
        "move the end effector backward",
        "shift the gripper toward the robot",
    ),
    "move_up": (
        "raise the end effector",
        "lift the gripper upward",
    ),
    "move_down": (
        "lower the end effector",
        "bring the gripper downward",
    ),
    "move_to_object": (
        "go to {target}",
        "move the gripper toward {target}",
        "approach {target}",
    ),
    "grab_object": (
        "grasp {target}",
        "take hold of {target}",
        "secure {target} with the gripper",
    ),
    "pick_up": (
        "lift {target}",
        "pick {target} up",
        "raise {target} off the table",
    ),
    "push_left": (
        "slide {target} to the left",
        "move {target} left by pushing it",
    ),
    "push_right": (
        "slide {target} to the right",
        "move {target} right by pushing it",
    ),
    "push_forward": (
        "slide {target} forward",
        "move {target} away from the robot by pushing it",
    ),
    "push_backward": (
        "slide {target} backward",
        "move {target} toward the robot by pushing it",
    ),
    "put_into_plate": (
        "place {target} inside {reference}",
        "drop {target} into {reference}",
        "put {target} in {reference}",
    ),
    "move_left_of_object": (
        "place {target} left of {reference}",
        "move {target} beside the left side of {reference}",
    ),
    "move_right_of_object": (
        "place {target} right of {reference}",
        "move {target} beside the right side of {reference}",
    ),
    "move_in_front_of_object": (
        "place {target} in front of {reference}",
        "move {target} to the front side of {reference}",
    ),
    "move_behind_object": (
        "place {target} behind {reference}",
        "move {target} to the back side of {reference}",
    ),
    "put_in_front_of_object": (
        "set {target} in front of {reference}",
        "position {target} on the front side of {reference}",
    ),
    "put_behind_object": (
        "set {target} behind {reference}",
        "position {target} on the back side of {reference}",
    ),
    "move_between_objects": (
        "place {target} between {reference} and {second_reference}",
        "position {target} in the middle of {reference} and {second_reference}",
    ),
    "catch_object": (
        "close the gripper around {target}",
        "capture {target} with the gripper",
    ),
    "grip_object": (
        "hold {target} firmly",
        "clamp the gripper onto {target}",
    ),
    "release_object": (
        "let go of {target}",
        "open the gripper and release {target}",
    ),
    "free_object": (
        "stop holding {target}",
        "open the gripper to free {target}",
    ),
    "open_gripper": (
        "spread the gripper fingers",
        "open the fingers fully",
    ),
    "close_gripper": (
        "bring the gripper fingers together",
        "close the fingers fully",
    ),
    "rotate_gripper_clockwise": (
        "turn the end effector clockwise",
        "yaw the gripper clockwise",
    ),
    "rotate_gripper_counterclockwise": (
        "turn the end effector counterclockwise",
        "yaw the gripper counterclockwise",
    ),
}

_ARBITRARY_PROMPT_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("move_left", "translate_x_negative", "translate the tool along the negative x direction"),
    ("move_right", "translate_x_positive", "translate the tool along the positive x direction"),
    ("move_top", "translate_y_positive", "translate the tool straight forward"),
    ("move_bottom", "translate_y_negative", "translate the tool straight backward"),
    ("move_up", "translate_z_positive", "raise the tool vertically without moving sideways"),
    ("move_down", "translate_z_negative", "lower the tool vertically without moving sideways"),
    ("open_gripper", "spread_fingers", "spread the two gripper fingers as far apart as possible"),
    ("close_gripper", "shut_fingers", "bring both gripper fingers fully together"),
    (
        "rotate_gripper_clockwise",
        "turn_tool_cw",
        "turn the tool clockwise without translating it",
    ),
    (
        "rotate_gripper_counterclockwise",
        "turn_tool_ccw",
        "turn the tool counterclockwise without translating it",
    ),
    ("move_to_object", "navigate_to_named_item", "navigate the end effector to the {target}"),
    ("grab_object", "secure_named_item", "carefully secure the {target} between the gripper fingers"),
    ("pick_up", "remove_from_surface", "remove the {target} from the table by lifting it"),
    ("put_into_plate", "deposit_in_container", "deposit the {target} inside the {reference}"),
    ("push_left", "nudge_west", "give the {target} a firm nudge toward the left side"),
    ("push_forward", "nudge_away", "nudge the {target} farther away from the robot"),
    (
        "move_right_of_object",
        "relative_right",
        "reposition the {target} so it ends up on the right-hand side of the {reference}",
    ),
    (
        "move_between_objects",
        "relative_middle",
        "leave the {target} midway between the {reference} and the {second_reference}",
    ),
    ("release_object", "unclamp_named_item", "unclamp the gripper so the {target} is no longer held"),
    ("grip_object", "pinch_named_item", "pinch the {target} securely with both fingers"),
)


def _reverse_shell_counts(instruction_types: tuple[str, ...]) -> dict[str, int]:
    from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import get_cdpr_reverse_shell_specs

    return {
        str(spec.instruction_id): int(spec.shell_count)
        for spec in get_cdpr_reverse_shell_specs(instruction_types)
    }


def _prompt_variant_specs(
    instruction_type: str,
    args: argparse.Namespace,
    *,
    curriculum_shell: int | None,
    curriculum_shell_count: int | None,
) -> list[tuple[str, str, str | None]]:
    variants: list[tuple[str, str, str | None]] = [("canonical", "canonical", None)]
    if not bool(getattr(args, "include_synonyms", False)):
        return variants

    is_normal_shell = (
        curriculum_shell is None
        or curriculum_shell_count is None
        or int(curriculum_shell) >= int(curriculum_shell_count) - 1
    )
    if str(getattr(args, "synonym_shells", "normal")) == "normal" and not is_normal_shell:
        return variants

    limit = max(0, int(getattr(args, "synonyms_per_instruction", 0)))
    for index, template in enumerate(_SYNONYM_PROMPT_TEMPLATES.get(instruction_type, ())[:limit], start=1):
        variants.append(("synonym", f"synonym_{index:02d}", str(template)))
    return variants


def _render_policy_prompt(
    *,
    prompt_template: str | None,
    canonical_instruction: str,
    reset_info: dict[str, Any],
) -> str:
    if not prompt_template:
        return str(canonical_instruction)

    target = canonical_object_name(str(reset_info.get("target_object_catalog", "")))
    reference = canonical_object_name(str(reset_info.get("reference_object_catalog", "")))
    second_reference = canonical_object_name(str(reset_info.get("second_reference_object_catalog", "")))
    return str(prompt_template).format(
        target=target,
        reference=reference,
        second_reference=second_reference,
    )


def _parse_instruction_types(raw_values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    resolved: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for token in str(raw_value).split(","):
            normalized = str(token).strip().lower().replace("-", "_").replace(" ", "_")
            if not normalized:
                continue
            candidates = INSTRUCTION_TYPES if normalized == "all" else (_INSTRUCTION_TYPE_ALIASES.get(normalized),)
            if normalized != "all" and candidates[0] is None:
                if normalized in INSTRUCTION_TYPES:
                    candidates = (normalized,)
                else:
                    supported = ", ".join(sorted(set(_INSTRUCTION_TYPE_ALIASES) | set(INSTRUCTION_TYPES) | {"all"}))
                    raise ValueError(
                        f"Unknown instruction type alias {raw_value!r}. Supported names: {supported}"
                    )
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(str(candidate))
                resolved.append(str(candidate))
    if not resolved:
        raise ValueError("Instruction selection removed every instruction type.")
    return tuple(resolved)


def _resolve_instruction_types(config: Any, args: argparse.Namespace) -> tuple[str, ...]:
    raw_values = getattr(args, "instruction_types", None)
    if raw_values:
        return _parse_instruction_types(raw_values)

    configured = tuple(getattr(config.task, "instruction_types", ()) or ())
    if configured:
        return _parse_instruction_types(configured)
    return tuple(INSTRUCTION_TYPES)


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
    metadata["directional_success_displacement_threshold"] = float(args.directional_displacement_threshold)
    metadata["directional_success_center_threshold"] = float(args.directional_displacement_threshold)
    return metadata


def _dedupe_object_names(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]

    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _instruction_validation_task_metadata(
    config: Any,
    args: argparse.Namespace,
    *,
    instruction_type: str | None = None,
    target_object: str | None = None,
) -> dict[str, Any]:
    metadata = _validation_task_metadata(config, args)
    dense_instruction_types = {
        str(item)
        for item in metadata.get("dense_stage_instruction_types", ())
    }
    dense_stage_metadata = metadata.get("dense_stage_metadata")
    if (
        instruction_type in dense_instruction_types
        and isinstance(dense_stage_metadata, dict)
    ):
        metadata.update(dense_stage_metadata)
    multi_object_scenes = bool(getattr(args, "multi_object_scenes", False))
    if multi_object_scenes:
        min_objects = max(2, int(getattr(args, "min_scene_objects", 3)))
        max_objects = max(min_objects, int(getattr(args, "max_scene_objects", 4)))
        metadata.pop("scene_object_pool", None)
        metadata["min_scene_objects"] = min_objects
        metadata["max_scene_objects"] = max_objects
        distractors = _dedupe_object_names(metadata.get("distractor_object_pool"))
        if not distractors:
            distractors = _dedupe_object_names(_allowed_objects_from_config(config))
        metadata["distractor_object_pool"] = distractors

    if instruction_type != "move_to_object":
        return metadata

    metadata.pop("scene_object_pool", None)
    target_pool = _dedupe_object_names(metadata.get("target_object_pool"))
    if not target_pool:
        target_pool = _dedupe_object_names(_allowed_objects_from_config(config))
    if target_pool:
        metadata["target_object_pool"] = target_pool
    if target_object:
        metadata["target_object_pool"] = [str(target_object)]
    metadata["move_to_object_validation_distance_threshold"] = float(
        getattr(args, "move_to_object_success_distance", 0.10)
    )
    if not multi_object_scenes:
        metadata["distractor_object_pool"] = []
        metadata["min_scene_objects"] = 1
        metadata["max_scene_objects"] = 1
    return metadata


def _validation_env_vars(
    config: Any,
    args: argparse.Namespace,
    *,
    instruction_type: str | None = None,
    task_metadata_override: dict[str, Any] | None = None,
) -> dict[str, str]:
    rl_args = _rl_args(config)
    env = {str(k): str(v) for k, v in getattr(config.project, "env", {}).items()}
    env.update({str(k): str(v) for k, v in getattr(config.remote, "env_vars", {}).items()})
    env.update(_task_hook_env(config))
    env.update(_extract_cdpr_env_overrides(dict(rl_args)))
    metadata = (
        dict(task_metadata_override)
        if task_metadata_override is not None
        else _instruction_validation_task_metadata(config, args, instruction_type=instruction_type)
    )
    env["RLVLA_TASK_METADATA_JSON"] = json.dumps(
        metadata,
        sort_keys=True,
    )
    env["RLVLA_TASK_SUCCESS_ATTRIBUTE"] = "compute_instruction_validation_success"
    env["RLVLA_TASK_SUCCESS_FILE"] = (
        Path(__file__).resolve().parents[2] / "robots" / "cdpr" / "cdpr_dataset" / "rl_instruction_tasks.py"
    ).as_posix()
    env.pop("RLVLA_TASK_SUCCESS_MODULE", None)
    return env


def _move_to_object_validation_targets(config: Any, args: argparse.Namespace) -> tuple[str, ...]:
    metadata = _instruction_validation_task_metadata(config, args, instruction_type="move_to_object")
    targets = _dedupe_object_names(metadata.get("target_object_pool"))
    if targets:
        return tuple(targets)
    return tuple(_dedupe_object_names(_allowed_objects_from_config(config)))


def _move_to_object_validation_episodes_per_target(
    config: Any,
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], int]:
    targets = _move_to_object_validation_targets(config, args)
    if not targets:
        return (), int(args.episodes_per_instruction)

    base_total = max(1, int(args.episodes_per_instruction))
    minimum_per_target = max(1, int(args.move_to_object_episodes_per_target))
    episodes_per_target = max(minimum_per_target, int(math.ceil(base_total / len(targets))))
    return targets, episodes_per_target


def _validation_buckets(
    config: Any,
    args: argparse.Namespace,
    *,
    instruction_type: str,
) -> list[ValidationBucket]:
    target_cases: list[tuple[str | None, int]] = [(None, max(1, int(args.episodes_per_instruction)))]
    if instruction_type == "move_to_object" and bool(
        getattr(args, "stratify_move_to_object_targets", True)
    ):
        targets, episodes_per_target = _move_to_object_validation_episodes_per_target(config, args)
        if targets:
            target_cases = [(target_object, episodes_per_target) for target_object in targets]

    shell_counts = (
        _reverse_shell_counts((instruction_type,))
        if bool(getattr(args, "evaluate_reverse_shells", False))
        else {}
    )
    shell_count = shell_counts.get(instruction_type)
    shell_ids: tuple[int | None, ...] = (
        tuple(range(int(shell_count))) if shell_count is not None else (None,)
    )

    buckets: list[ValidationBucket] = []
    for target_object, episodes in target_cases:
        metadata = _instruction_validation_task_metadata(
            config,
            args,
            instruction_type=instruction_type,
            target_object=target_object,
        )
        env_vars = _validation_env_vars(
            config,
            args,
            instruction_type=instruction_type,
            task_metadata_override=metadata,
        )
        for shell_id in shell_ids:
            for prompt_kind, prompt_variant, prompt_template in _prompt_variant_specs(
                instruction_type,
                args,
                curriculum_shell=shell_id,
                curriculum_shell_count=shell_count,
            ):
                label_parts = [instruction_type]
                if target_object:
                    label_parts.append(str(target_object))
                if shell_id is not None:
                    label_parts.append(f"shell={int(shell_id)}")
                if prompt_kind != "canonical":
                    label_parts.append(prompt_variant)
                buckets.append(
                    ValidationBucket(
                        instruction_type=instruction_type,
                        target_object=target_object,
                        episodes=int(episodes),
                        env_vars=env_vars,
                        log_label=":".join(label_parts),
                        prompt_kind=prompt_kind,
                        prompt_variant=prompt_variant,
                        prompt_template=prompt_template,
                        curriculum_shell=shell_id,
                        curriculum_shell_count=shell_count,
                    )
                )
    return buckets


def _arbitrary_validation_buckets(
    config: Any,
    args: argparse.Namespace,
    *,
    instruction_types: tuple[str, ...],
) -> list[ValidationBucket]:
    requested = max(0, min(int(getattr(args, "arbitrary_instructions_count", 0)), 10))
    if requested <= 0:
        return []

    allowed = set(instruction_types)
    shell_counts = _reverse_shell_counts(instruction_types)
    candidates = [item for item in _ARBITRARY_PROMPT_CHECKS if item[0] in allowed]
    selected = candidates[:requested]
    buckets: list[ValidationBucket] = []
    for instruction_type, prompt_variant, prompt_template in selected:
        shell_count = shell_counts.get(instruction_type)
        shell_id = None if shell_count is None else int(shell_count) - 1
        metadata = _instruction_validation_task_metadata(
            config,
            args,
            instruction_type=instruction_type,
        )
        buckets.append(
            ValidationBucket(
                instruction_type=instruction_type,
                target_object=None,
                episodes=1,
                env_vars=_validation_env_vars(
                    config,
                    args,
                    instruction_type=instruction_type,
                    task_metadata_override=metadata,
                ),
                log_label=f"arbitrary:{instruction_type}:{prompt_variant}",
                prompt_kind="arbitrary",
                prompt_variant=prompt_variant,
                prompt_template=prompt_template,
                curriculum_shell=shell_id,
                curriculum_shell_count=shell_count,
                force_video=True,
            )
        )
    return buckets


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
    max_objects = getattr(args, "max_objects", None)

    env_kwargs = dict(
        catalog_path=config.resolve_path(config.simulation.catalog_path),
        max_steps=int(max_steps),
        action_step_xyz=float(control_spec.action_step_xyz),
        action_step_yaw=float(control_spec.action_step_yaw),
        action_step_gripper=float(control_spec.action_step_gripper),
        hold_steps=int(control_spec.hold_steps),
        lock_non_commanded_axes=rl_args.get("lock_non_commanded_axes"),
        lock_non_commanded_axes_threshold=rl_args.get("lock_non_commanded_axes_threshold"),
        randomize_ee_start=rl_args.get("randomize_ee_start"),
        ee_start_x_bounds=rl_args.get("ee_start_x_bounds"),
        ee_start_y_bounds=rl_args.get("ee_start_y_bounds"),
        ee_start_z=rl_args.get("ee_start_z"),
        randomize_ee_yaw=rl_args.get("randomize_ee_yaw"),
        ee_yaw_bounds=rl_args.get("ee_yaw_bounds"),
        move_distance=float(metadata.get("lateral_goal_offset", 0.40)),
        lift_distance=float(metadata.get("vertical_goal_offset", 0.10)),
        capture_frames=bool(capture_frames),
        record_trajectory=bool(capture_frames),
        instruction_types=[instruction_type],
        allowed_objects=_allowed_objects_from_config(config),
        desk_textures_dir=desk_textures_dir,
        wrapper_cleanup=bool(rl_args.get("wrapper_cleanup", True)),
        use_wrapper_cache=bool(rl_args.get("use_wrapper_cache", False)),
        reuse_existing_wrapper_variants=bool(args.reuse_existing_wrapper_variants),
        wrapper_dir=wrapper_dir,
        seed=seed,
    )
    if max_objects is not None:
        env_kwargs["max_objects"] = max(1, int(max_objects))
    return CDPRLanguageRLEnv(**env_kwargs)


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
    return _normalize_policy_chunk(
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
        replan_every=runtime.get("replan_every"),
    )


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

    configure_openvla_dimension_env_from_config(
        config,
        chunk_length=int(args.chunk_length or config.policy.action_codec.chunk_size),
    )
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
    replan_every = None if args.replan_every is None else max(1, min(int(args.replan_every), chunk_length))
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
        "replan_every": replan_every,
        "num_images_in_input": int(cfg.num_images_in_input),
    }


def _safe_filename_token(value: str | None) -> str:
    token = str(value or "").strip().lower().replace(" ", "_")
    token = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in token)
    return token.strip("_")


def _format_action_vector(values: Any) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{float(value):+.3f}" for value in arr[:5]) + "]"


def _annotate_latest_validation_frame(
    *,
    sim: Any,
    instruction: str,
    step: int,
    policy_call: int,
    chunk_action_index: int,
    chunk_length: int,
    new_policy_output: bool,
    action: np.ndarray,
    scaled_action: np.ndarray,
    info: dict[str, Any],
) -> None:
    if Image is None or ImageDraw is None:
        return
    frames = getattr(sim, "overview_frames", None)
    if not isinstance(frames, list) or not frames:
        return
    frame = np.asarray(frames[-1])
    if frame.ndim != 3 or frame.shape[2] < 3:
        return

    image = Image.fromarray(frame[:, :, :3].astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default() if ImageFont is not None else None
    ee = info.get("ee_position", ())
    state_line = (
        f"EE={_format_action_vector(list(ee)[:3])} "
        f"yaw={float(info.get('ee_yaw', 0.0)):+.3f} "
        f"grip={float(info.get('gripper_opening', 0.0)):.3f}"
        f"->{float(info.get('gripper_target', 0.0)):.3f}"
    )
    source = "NEW VLA OUTPUT" if new_policy_output else "cached VLA chunk"
    lines = [
        f"{source} | call #{int(policy_call)} | action {int(chunk_action_index) + 1}/{int(chunk_length)}",
        f"step {int(step)} | {instruction}",
        f"normalized [x y z yaw grip] = {_format_action_vector(action)}",
        f"applied    [m m m rad open] = {_format_action_vector(scaled_action)}",
        state_line,
    ]
    line_height = 15
    box_height = line_height * len(lines) + 10
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0, 190))
    color = (100, 255, 120, 255) if new_policy_output else (255, 255, 255, 255)
    for index, line in enumerate(lines):
        draw.text((7, 5 + line_height * index), line, fill=color if index == 0 else (255, 255, 255, 255), font=font)
    frames[-1] = np.asarray(image)


def _scaled_action_vector(action: np.ndarray, config: Any, hold_steps: int | None) -> np.ndarray:
    control = _control_spec_from_config(config, hold_steps)
    arr = np.asarray(action, dtype=np.float32).reshape(5)
    return np.asarray(
        [
            arr[0] * float(control.action_step_xyz),
            arr[1] * float(control.action_step_xyz),
            arr[2] * float(control.action_step_xyz),
            arr[3] * float(control.action_step_yaw),
            arr[4] * float(control.action_step_gripper),
        ],
        dtype=np.float32,
    )


def _write_action_trace_csv(output_path: Path, action_trace: list[dict[str, Any]]) -> None:
    columns = [
        "step",
        "policy_call",
        "new_policy_output",
        "chunk_action_index",
        "chunk_length",
        "action_x",
        "action_y",
        "action_z",
        "action_yaw",
        "action_gripper",
        "applied_dx",
        "applied_dy",
        "applied_dz",
        "applied_dyaw",
        "applied_dgripper",
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_yaw",
        "gripper_opening",
        "gripper_target",
        "success",
        "simulation_state_valid",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in action_trace:
            writer.writerow({column: row.get(column, "") for column in columns})


def _validation_episode_video_frames(sim: Any, episode_result: EpisodeResult) -> list[Any]:
    frames = list(getattr(sim, "overview_frames", []) or [])
    if len(frames) != 1:
        return frames

    # Older/non-trajectory simulation objects keep only the latest overview frame.
    # Preserve a non-zero duration in that fallback case, while the normal
    # validation path records one overview frame per env step.
    target_frame_count = max(2, min(int(episode_result.steps), _MAX_SYNTHETIC_VIDEO_FRAMES))
    return frames * target_frame_count


def _probe_video_file(video_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "video_path": video_path.as_posix(),
        "exists": video_path.is_file(),
        "size_bytes": int(video_path.stat().st_size) if video_path.is_file() else 0,
        "probe_backend": "filesystem",
        "valid": False,
    }
    if not video_path.is_file() or int(result["size_bytes"]) <= 0:
        result["error"] = "Video file is missing or empty."
        return result

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        try:
            import imageio.v2 as imageio

            reader = imageio.get_reader(video_path.as_posix())
            first_frame = np.asarray(reader.get_data(0))
            metadata = dict(reader.get_meta_data() or {})
            try:
                frame_count = int(reader.count_frames())
            except Exception:
                frame_count = int(metadata.get("nframes") or 0)
            reader.close()
            fps = float(metadata.get("fps") or 0.0)
            duration = float(metadata.get("duration") or 0.0)
            if duration <= 0.0 and fps > 0.0 and frame_count > 0:
                duration = float(frame_count / fps)
            height = int(first_frame.shape[0]) if first_frame.ndim >= 2 else 0
            width = int(first_frame.shape[1]) if first_frame.ndim >= 2 else 0
            result.update(
                {
                    "probe_backend": "imageio",
                    "codec_name": metadata.get("codec"),
                    "width": width,
                    "height": height,
                    "avg_frame_rate": fps,
                    "nb_frames": frame_count,
                    "duration_sec": duration,
                    "valid": bool(width > 0 and height > 0 and frame_count > 0 and duration > 0.0),
                }
            )
            if not result["valid"]:
                result["error"] = "imageio could not decode a positive-duration video stream."
            return result
        except Exception as exc:
            result["error"] = f"Neither ffprobe nor imageio could validate the MP4: {exc}"
            return result

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        video_path.as_posix(),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        result["probe_backend"] = "ffprobe"
        result["error"] = "ffprobe timed out after 30 seconds."
        return result
    result["probe_backend"] = "ffprobe"
    result["returncode"] = int(completed.returncode)
    if completed.returncode != 0:
        result["error"] = completed.stderr.strip() or "ffprobe rejected the video."
        return result
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        result["error"] = f"Could not parse ffprobe output: {exc}"
        return result

    streams = list(payload.get("streams") or [])
    stream = dict(streams[0]) if streams else {}
    video_format = dict(payload.get("format") or {})
    try:
        width = int(stream.get("width") or 0)
    except (TypeError, ValueError):
        width = 0
    try:
        height = int(stream.get("height") or 0)
    except (TypeError, ValueError):
        height = 0
    try:
        duration = float(stream.get("duration") or video_format.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    result.update(
        {
            "codec_name": stream.get("codec_name"),
            "width": width,
            "height": height,
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "nb_frames": stream.get("nb_frames"),
            "duration_sec": duration,
            "valid": bool(width > 0 and height > 0 and duration > 0.0),
        }
    )
    if not result["valid"]:
        result["error"] = "Video has no decodable positive-size stream with positive duration."
    return result


def _save_episode_video(
    *,
    sim: Any,
    output_dir: Path,
    instruction_type: str,
    episode_result: EpisodeResult,
    outcome: str,
    action_trace: list[dict[str, Any]] | None = None,
) -> str | None:
    frames = _validation_episode_video_frames(sim, episode_result)
    if not frames or not hasattr(sim, "save_video"):
        return None

    fps = _VALIDATION_VIDEO_FPS
    target_token = _safe_filename_token(episode_result.target_object_catalog)
    target_part = f"_{target_token}" if target_token else ""
    shell_part = (
        f"_shell_{int(episode_result.curriculum_shell):02d}"
        if episode_result.curriculum_shell is not None
        else ""
    )
    prompt_part = ""
    if episode_result.prompt_kind != "canonical" or episode_result.prompt_variant != "canonical":
        prompt_part = (
            f"_{_safe_filename_token(episode_result.prompt_kind)}"
            f"_{_safe_filename_token(episode_result.prompt_variant)}"
        )
    output_path = (
        output_dir
        / (
            f"{instruction_type}{target_part}{shell_part}{prompt_part}_{outcome}"
            f"_episode_{episode_result.episode_index:03d}_overview.mp4"
        )
    )
    sim.save_video(frames, str(output_path), fps=fps)

    summary_path = output_path.with_name(output_path.name.replace("_overview.mp4", "_summary.json"))
    action_trace_path = output_path.with_name(output_path.name.replace("_overview.mp4", "_actions.csv"))
    if action_trace:
        _write_action_trace_csv(action_trace_path, action_trace)
    summary_data = asdict(episode_result)
    summary_data["video_kind"] = str(outcome)
    summary_data["video_path"] = output_path.as_posix()
    summary_data["video_frame_count"] = len(frames)
    summary_data["video_fps"] = fps
    summary_data["video_duration_sec"] = len(frames) / fps
    summary_data["policy_output_calls"] = int(episode_result.policy_output_calls)
    summary_data["action_trace_path"] = action_trace_path.as_posix() if action_trace else None
    summary_data["video_probe"] = _probe_video_file(output_path)
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    return output_path.as_posix()


def _summarize_instruction_results(
    *,
    instruction_type: str,
    episode_results: list[EpisodeResult],
    video_path: str | None,
    success_video_path: str | None = None,
    failure_video_path: str | None = None,
) -> InstructionSummary:
    successes = sum(1 for item in episode_results if item.success)
    rewards = np.asarray([item.reward_total for item in episode_results], dtype=np.float32)
    steps = np.asarray([item.steps for item in episode_results], dtype=np.float32)
    total = len(episode_results)
    resolved_success_video = success_video_path
    if resolved_success_video is None and failure_video_path is None:
        resolved_success_video = video_path
    resolved_failure_video = failure_video_path
    resolved_video = resolved_success_video or resolved_failure_video or video_path
    return InstructionSummary(
        instruction_type=instruction_type,
        instruction_text=INSTRUCTION_TEXT.get(instruction_type, instruction_type.replace("_", " ")),
        successes=int(successes),
        episodes=int(total),
        success_rate=float(successes / max(total, 1)),
        mean_reward=float(np.mean(rewards)) if rewards.size > 0 else 0.0,
        mean_steps=float(np.mean(steps)) if steps.size > 0 else 0.0,
        video_path=resolved_video,
        success_video_path=resolved_success_video,
        failure_video_path=resolved_failure_video,
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
                "success_video_path",
                "failure_video_path",
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
                    summary.success_video_path or "",
                    summary.failure_video_path or "",
                ]
            )


def _summarize_instruction_text_results(episode_results: list[EpisodeResult]) -> list[InstructionTextSummary]:
    grouped: dict[str, list[EpisodeResult]] = {}
    for episode_result in episode_results:
        grouped.setdefault(str(episode_result.instruction_text), []).append(episode_result)

    summaries: list[InstructionTextSummary] = []
    for instruction_text in sorted(grouped):
        items = grouped[instruction_text]
        rewards = np.asarray([item.reward_total for item in items], dtype=np.float32)
        steps = np.asarray([item.steps for item in items], dtype=np.float32)
        successes = sum(1 for item in items if item.success)
        instruction_types = tuple(sorted({str(item.instruction_type) for item in items}))
        target_object_catalogs = tuple(
            sorted({str(item.target_object_catalog) for item in items if str(item.target_object_catalog or "").strip()})
        )
        total = len(items)
        summaries.append(
            InstructionTextSummary(
                instruction_text=instruction_text,
                instruction_types=instruction_types,
                target_object_catalogs=target_object_catalogs,
                successes=int(successes),
                episodes=int(total),
                success_rate=float(successes / max(total, 1)),
                mean_reward=float(np.mean(rewards)) if rewards.size > 0 else 0.0,
                mean_steps=float(np.mean(steps)) if steps.size > 0 else 0.0,
            )
        )
    return summaries


def _write_instruction_text_csv(output_path: Path, summaries: list[InstructionTextSummary]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instruction_text",
                "instruction_types",
                "target_object_catalogs",
                "successes",
                "episodes",
                "success_rate",
                "mean_reward",
                "mean_steps",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.instruction_text,
                    "|".join(summary.instruction_types),
                    "|".join(summary.target_object_catalogs),
                    summary.successes,
                    summary.episodes,
                    f"{summary.success_rate:.6f}",
                    f"{summary.mean_reward:.6f}",
                    f"{summary.mean_steps:.6f}",
                ]
            )


def _aggregate_episode_results(
    episode_results: list[EpisodeResult],
    *,
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[EpisodeResult]] = {}
    for item in episode_results:
        if not item.metric_episode:
            continue
        key = tuple(getattr(item, field) for field in group_fields)
        grouped.setdefault(key, []).append(item)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        items = grouped[key]
        successes = sum(1 for item in items if item.success)
        rewards = np.asarray([item.reward_total for item in items], dtype=np.float32)
        steps = np.asarray([item.steps for item in items], dtype=np.float32)
        row = {field: value for field, value in zip(group_fields, key)}
        row.update(
            {
                "successes": int(successes),
                "episodes": int(len(items)),
                "success_rate": float(successes / max(len(items), 1)),
                "mean_reward": float(np.mean(rewards)) if rewards.size else 0.0,
                "mean_steps": float(np.mean(steps)) if steps.size else 0.0,
            }
        )
        rows.append(row)
    return rows


def _write_grouped_success_rate_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
) -> None:
    columns = [
        *group_fields,
        "successes",
        "episodes",
        "success_rate",
        "mean_reward",
        "mean_steps",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("success_rate", "mean_reward", "mean_steps"):
                formatted[key] = f"{float(row[key]):.6f}"
            writer.writerow(formatted)


def _write_episode_results_csv(output_path: Path, episode_results: list[EpisodeResult]) -> None:
    columns = [
        "episode_index",
        "seed",
        "instruction_type",
        "prompt_kind",
        "prompt_variant",
        "instruction_text",
        "canonical_instruction_text",
        "curriculum_shell",
        "curriculum_shell_count",
        "target_object_catalog",
        "reference_object_catalog",
        "second_reference_object_catalog",
        "scene",
        "scene_objects",
        "success",
        "terminated",
        "truncated",
        "steps",
        "reward_total",
        "metric_episode",
        "policy_output_calls",
        "action_steps",
        "reset_attempts",
        "simulation_instability",
        "final_ee_position",
        "final_ee_yaw",
        "final_gripper_opening",
        "final_gripper_target",
        "final_move_to_object_distance_xy",
        "min_move_to_object_distance_xy",
        "move_to_object_distance_threshold",
        "video_kind",
        "video_path",
        "action_trace_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in episode_results:
            row = asdict(item)
            row["scene_objects"] = "|".join(item.scene_objects)
            row["final_ee_position"] = "|".join(str(value) for value in item.final_ee_position)
            writer.writerow({column: row.get(column, "") for column in columns})


def _is_normal_scene_canonical_episode(item: EpisodeResult) -> bool:
    is_normal_shell = (
        item.curriculum_shell is None
        or item.curriculum_shell_count is None
        or int(item.curriculum_shell) >= int(item.curriculum_shell_count) - 1
    )
    return bool(
        item.metric_episode
        and item.prompt_kind == "canonical"
        and item.prompt_variant == "canonical"
        and is_normal_shell
    )


def _move_to_object_threshold_sweep(
    episode_results: list[EpisodeResult],
    thresholds: tuple[float, ...] = (0.025, 0.05, 0.10, 0.15),
) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in episode_results
        if item.instruction_type == "move_to_object"
        and _is_normal_scene_canonical_episode(item)
        and item.min_move_to_object_distance_xy is not None
    ]
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        successes = sum(
            float(item.min_move_to_object_distance_xy) <= float(threshold)
            for item in candidates
        )
        rows.append(
            {
                "distance_threshold_m": float(threshold),
                "successes": int(successes),
                "episodes": int(len(candidates)),
                "success_rate": float(successes / max(len(candidates), 1)),
            }
        )
    return rows


def _write_move_to_object_threshold_sweep(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("distance_threshold_m", "successes", "episodes", "success_rate"),
        )
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["distance_threshold_m"] = f"{float(row['distance_threshold_m']):.6f}"
            formatted["success_rate"] = f"{float(row['success_rate']):.6f}"
            writer.writerow(formatted)


def _write_video_audit(
    *,
    run_dir: Path,
    videos_dir: Path,
    expected_keys: list[str],
    video_registry: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probes = [_probe_video_file(path) for path in sorted(videos_dir.glob("*.mp4"))]
    (run_dir / "video_validation.json").write_text(
        json.dumps(probes, indent=2),
        encoding="utf-8",
    )
    with (run_dir / "video_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = [
            "video_path",
            "valid",
            "size_bytes",
            "probe_backend",
            "codec_name",
            "width",
            "height",
            "duration_sec",
            "error",
            "warning",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for probe in probes:
            writer.writerow({column: probe.get(column, "") for column in columns})

    coverage: list[dict[str, Any]] = []
    for key in expected_keys:
        entry = video_registry.get(key, {})
        coverage.append(
            {
                "coverage_key": key,
                "success_video_path": entry.get("success", ""),
                "failure_video_path": entry.get("failure", ""),
                "has_success_video": bool(entry.get("success")),
                "has_failure_video": bool(entry.get("failure")),
                "complete": bool(entry.get("success") and entry.get("failure")),
            }
        )
    with (run_dir / "video_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = [
            "coverage_key",
            "has_success_video",
            "has_failure_video",
            "complete",
            "success_video_path",
            "failure_video_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(coverage)
    return probes, coverage


def _markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...], *, limit: int | None = None) -> str:
    selected = rows if limit is None else rows[:limit]
    if not selected:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in selected:
        values: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _write_validation_report(
    *,
    run_dir: Path,
    artifacts: ResolvedPolicyArtifacts,
    metric_results: list[EpisodeResult],
    all_results: list[EpisodeResult],
    instruction_rows: list[dict[str, Any]],
    normal_canonical_rows: list[dict[str, Any]],
    shell_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    video_probes: list[dict[str, Any]],
    video_coverage: list[dict[str, Any]],
    move_to_object_threshold_rows: list[dict[str, Any]],
) -> Path:
    successes = sum(1 for item in metric_results if item.success)
    arbitrary = [item for item in all_results if item.prompt_kind == "arbitrary"]
    invalid_videos = [item for item in video_probes if not bool(item.get("valid"))]
    incomplete_coverage = [item for item in video_coverage if not bool(item.get("complete"))]
    total_policy_calls = sum(int(item.policy_output_calls) for item in all_results)
    total_action_steps = sum(int(item.action_steps) for item in all_results)
    reset_retries = sum(max(0, int(item.reset_attempts) - 1) for item in all_results)
    instability_episodes = sum(bool(item.simulation_instability) for item in all_results)
    lines = [
        "# CDPR checkpoint validation report",
        "",
        f"- Checkpoint: `{artifacts.checkpoint_dir}`",
        f"- Metric episodes: `{len(metric_results)}`",
        f"- Overall successes: `{successes}`",
        f"- Overall success rate: `{successes / max(len(metric_results), 1):.4f}`",
        f"- Exact OpenVLA output generations: `{total_policy_calls}`",
        f"- Applied policy actions: `{total_action_steps}`",
        f"- Reset retries after invalid simulation state: `{reset_retries}`",
        f"- Episodes truncated for simulation instability: `{instability_episodes}`",
        f"- Recorded videos: `{len(video_probes)}`",
        f"- Invalid videos: `{len(invalid_videos)}`",
        f"- Incomplete success/failure video coverage entries: `{len(incomplete_coverage)}`",
        "",
        "## Instruction success rates",
        "",
        _markdown_table(
            instruction_rows,
            ("instruction_type", "successes", "episodes", "success_rate", "mean_steps"),
        ),
        "",
        "## Canonical normal-scene success rates",
        "",
        "These are the deployment-like scores: canonical wording on the final normal reset, "
        "without easier reverse-curriculum shells.",
        "",
        _markdown_table(
            normal_canonical_rows,
            ("instruction_type", "successes", "episodes", "success_rate", "mean_steps"),
        ),
        "",
        "## Reverse-shell success rates",
        "",
        _markdown_table(
            shell_rows,
            (
                "instruction_type",
                "curriculum_shell",
                "successes",
                "episodes",
                "success_rate",
            ),
        ),
        "",
        "## Prompt-variant success rates",
        "",
        _markdown_table(
            prompt_rows,
            (
                "instruction_type",
                "prompt_kind",
                "prompt_variant",
                "successes",
                "episodes",
                "success_rate",
            ),
        ),
        "",
        "## Target-object success rates",
        "",
        _markdown_table(
            target_rows,
            (
                "instruction_type",
                "target_object_catalog",
                "successes",
                "episodes",
                "success_rate",
            ),
        ),
        "",
        "## Move-to-object tolerance sweep",
        "",
        "The same canonical normal-scene trajectories are rescored by their minimum XY distance.",
        "",
        _markdown_table(
            move_to_object_threshold_rows,
            ("distance_threshold_m", "successes", "episodes", "success_rate"),
        ),
        "",
        "## Arbitrary recorded prompt checks",
        "",
        _markdown_table(
            [
                {
                    "instruction_type": item.instruction_type,
                    "instruction_text": item.instruction_text,
                    "scene_objects": ", ".join(item.scene_objects),
                    "success": item.success,
                    "video_path": item.video_path or "",
                }
                for item in arbitrary
            ],
            ("instruction_type", "instruction_text", "scene_objects", "success", "video_path"),
        ),
        "",
        "## Video coverage",
        "",
        _markdown_table(
            video_coverage,
            (
                "coverage_key",
                "has_success_video",
                "has_failure_video",
                "complete",
            ),
        ),
        "",
        "## Artifacts",
        "",
        "- `validation_manifest.json`",
        "- `episode_results.csv`",
        "- `instruction_success_rates.csv`",
        "- `normal_scene_canonical_success_rates.csv`",
        "- `instruction_shell_success_rates.csv`",
        "- `instruction_prompt_success_rates.csv`",
        "- `evaluation_case_success_rates.csv`",
        "- `target_object_success_rates.csv`",
        "- `instruction_text_success_rates.csv`",
        "- `move_to_object_threshold_sweep.csv`",
        "- `video_coverage.csv`",
        "- `video_validation.csv` and `video_validation.json`",
        "- `videos/`",
        "",
    ]
    report_path = run_dir / "validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _reset_validation_env_with_retries(
    *,
    env: CDPRLanguageRLEnv,
    seed: int | None,
    reset_options: dict[str, Any],
    max_attempts: int,
    quiet: bool,
) -> tuple[Any, dict[str, Any], int]:
    last_error: Exception | None = None
    errors: list[str] = []
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        retry_seed = None if seed is None else int(seed) + (attempt - 1) * 1_000_003
        try:
            with _silence_output(quiet):
                obs, info = env.reset(seed=retry_seed, options=reset_options)
            if not bool(info.get("simulation_state_valid", True)):
                raise RuntimeError(
                    f"invalid reset state: {info.get('simulation_state_reason', 'unknown')}"
                )
            reset_info = dict(info)
            reset_info["reset_retry_errors"] = tuple(errors)
            return obs, reset_info, attempt
        except Exception as exc:
            last_error = exc
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            with _silence_output(quiet):
                env.close()
    error_summary = " | ".join(errors)
    raise RuntimeError(
        f"CDPR validation reset failed after {max(1, int(max_attempts))} attempts. "
        f"Failures: {error_summary or last_error}"
    ) from last_error


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
    episodes_to_run: int,
    episode_index_offset: int = 0,
    log_label: str | None = None,
    validation_bucket: ValidationBucket | None = None,
    video_registry: dict[str, dict[str, str]] | None = None,
    metric_episode: bool = True,
    stop_when_video_coverage_complete: bool = False,
) -> tuple[InstructionSummary, list[EpisodeResult]]:
    bucket = validation_bucket or ValidationBucket(
        instruction_type=instruction_type,
        target_object=None,
        episodes=int(episodes_to_run),
        env_vars={},
        log_label=str(log_label or instruction_type),
    )
    coverage_key = (
        instruction_type
        if str(getattr(args, "video_coverage", "instruction")) == "instruction"
        else bucket.case_id
    )
    registry = video_registry if video_registry is not None else {}
    coverage_entry = registry.setdefault(coverage_key, {})
    should_capture = bool(
        args.record_success_videos or args.record_failure_videos or bucket.force_video
    )
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

    reset_options: dict[str, Any] = {"instruction_type": instruction_type}
    if args.scene:
        reset_options["scene"] = args.scene
    if bucket.target_object:
        reset_options["target_object"] = str(bucket.target_object)
    if bucket.curriculum_shell is not None:
        reset_options["curriculum_mode"] = "reverse_frontier"
        reset_options["curriculum_shell"] = int(bucket.curriculum_shell)
    episode_results: list[EpisodeResult] = []
    success_video_path: str | None = coverage_entry.get("success")
    failure_video_path: str | None = coverage_entry.get("failure")
    successes = 0

    try:
        effective_log_label = str(log_label or instruction_type)
        for episode_offset in range(int(episodes_to_run)):
            episode_index = int(episode_index_offset) + int(episode_offset)
            needs_success_video = bool(
                (args.record_success_videos or bucket.force_video)
                and (args.record_all_success_videos or coverage_entry.get("success") is None)
            )
            needs_failure_video = bool(
                (args.record_failure_videos or bucket.force_video)
                and coverage_entry.get("failure") is None
            )
            env.capture_frames = bool(
                bucket.force_video or needs_success_video or needs_failure_video
            )
            seed = _episode_seed(base_seed, instruction_index, episode_index)
            _obs, reset_info, reset_attempts = _reset_validation_env_with_retries(
                env=env,
                seed=seed,
                reset_options=reset_options,
                max_attempts=int(args.max_reset_attempts),
                quiet=bool(args.progress_only),
            )
            canonical_instruction = str(
                reset_info.get("language_instruction", INSTRUCTION_TEXT[instruction_type])
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
            min_move_to_object_distance_xy = float("inf")

            while not (terminated or truncated):
                new_policy_output = False
                if chunk_index >= len(current_chunk):
                    with _silence_output(bool(args.progress_only)):
                        current_chunk = _predict_policy_chunk(
                            runtime=runtime,
                            sim=env.sim,
                            instruction=instruction,
                            config=config,
                        )
                    chunk_index = 0
                    policy_output_calls += 1
                    new_policy_output = True

                action_index = int(chunk_index)
                action = np.asarray(current_chunk[action_index], dtype=np.float32).reshape(5)
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
                action_steps += 1
                distance_xy_raw = final_info.get("move_to_object_validation_distance_xy")
                if distance_xy_raw is not None:
                    try:
                        min_move_to_object_distance_xy = min(
                            min_move_to_object_distance_xy,
                            float(distance_xy_raw),
                        )
                    except (TypeError, ValueError):
                        pass

                scaled_action = _scaled_action_vector(action, config, args.hold_steps)
                ee_position = list(final_info.get("ee_position", ()))
                trace_row = {
                    "step": int(action_steps),
                    "policy_call": int(policy_output_calls),
                    "new_policy_output": bool(new_policy_output),
                    "chunk_action_index": int(action_index),
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
                    "ee_x": float(ee_position[0]) if len(ee_position) >= 1 else "",
                    "ee_y": float(ee_position[1]) if len(ee_position) >= 2 else "",
                    "ee_z": float(ee_position[2]) if len(ee_position) >= 3 else "",
                    "ee_yaw": float(final_info.get("ee_yaw", 0.0)),
                    "gripper_opening": float(final_info.get("gripper_opening", 0.0)),
                    "gripper_target": float(final_info.get("gripper_target", 0.0)),
                    "success": bool(final_info.get("success", False)),
                    "simulation_state_valid": bool(
                        final_info.get("simulation_state_valid", True)
                    ),
                }
                action_trace.append(trace_row)
                if env.capture_frames and bool(args.video_action_overlay):
                    _annotate_latest_validation_frame(
                        sim=env.sim,
                        instruction=instruction,
                        step=action_steps,
                        policy_call=policy_output_calls,
                        chunk_action_index=action_index,
                        chunk_length=len(current_chunk),
                        new_policy_output=new_policy_output,
                        action=action,
                        scaled_action=scaled_action,
                        info=dict(final_info),
                    )

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
                target_object_catalog=str(
                    final_info.get("target_object_catalog", reset_info.get("target_object_catalog", ""))
                )
                or None,
                reference_object_catalog=str(
                    final_info.get(
                        "reference_object_catalog",
                        reset_info.get("reference_object_catalog", ""),
                    )
                )
                or None,
                second_reference_object_catalog=str(
                    final_info.get(
                        "second_reference_object_catalog",
                        reset_info.get("second_reference_object_catalog", ""),
                    )
                )
                or None,
                scene_objects=tuple(
                    str(value)
                    for value in final_info.get(
                        "scene_objects",
                        reset_info.get("scene_objects", ()),
                    )
                ),
                canonical_instruction_text=canonical_instruction,
                prompt_kind=str(bucket.prompt_kind),
                prompt_variant=str(bucket.prompt_variant),
                curriculum_shell=bucket.curriculum_shell,
                curriculum_shell_count=bucket.curriculum_shell_count,
                metric_episode=bool(metric_episode),
                policy_output_calls=int(policy_output_calls),
                action_steps=int(action_steps),
                reset_attempts=int(reset_attempts),
                simulation_instability=bool(final_info.get("simulation_instability", False)),
                final_ee_position=tuple(
                    float(value) for value in final_info.get("ee_position", ())
                ),
                final_ee_yaw=float(final_info.get("ee_yaw", 0.0)),
                final_gripper_opening=float(final_info.get("gripper_opening", 0.0)),
                final_gripper_target=float(final_info.get("gripper_target", 0.0)),
                final_move_to_object_distance_xy=(
                    None
                    if final_info.get("move_to_object_validation_distance_xy") is None
                    else float(final_info["move_to_object_validation_distance_xy"])
                ),
                min_move_to_object_distance_xy=(
                    None
                    if not np.isfinite(min_move_to_object_distance_xy)
                    else float(min_move_to_object_distance_xy)
                ),
                move_to_object_distance_threshold=(
                    None
                    if final_info.get("move_to_object_validation_distance_threshold") is None
                    else float(final_info["move_to_object_validation_distance_threshold"])
                ),
            )
            episode_results.append(episode_result)
            successes += int(episode_result.success)

            saved_video_path: str | None = None
            saved_video_kind: str | None = None
            if (
                episode_result.success
                and bool(args.record_success_videos or bucket.force_video)
                and (
                    bucket.force_video
                    or bool(args.record_all_success_videos)
                    or coverage_entry.get("success") is None
                )
            ):
                try:
                    with _silence_output(bool(args.progress_only)):
                        saved_video_path = _save_episode_video(
                            sim=env.sim,
                            output_dir=videos_dir,
                            instruction_type=instruction_type,
                            episode_result=episode_result,
                            outcome="success",
                            action_trace=action_trace,
                        )
                    if saved_video_path and coverage_entry.get("success") is None:
                        coverage_entry["success"] = saved_video_path
                    if success_video_path is None and saved_video_path:
                        success_video_path = saved_video_path
                    saved_video_kind = "success" if saved_video_path else None
                except Exception as exc:
                    if not args.progress_only:
                        print(f"[warn] Failed to save success video for {instruction_type}: {exc}")
                finally:
                    clear_sim_recording_buffers(env.sim)
            elif (
                (not episode_result.success)
                and bool(args.record_failure_videos or bucket.force_video)
                and (bucket.force_video or coverage_entry.get("failure") is None)
            ):
                try:
                    with _silence_output(bool(args.progress_only)):
                        saved_video_path = _save_episode_video(
                            sim=env.sim,
                            output_dir=videos_dir,
                            instruction_type=instruction_type,
                            episode_result=episode_result,
                            outcome="failure",
                            action_trace=action_trace,
                        )
                    if saved_video_path and coverage_entry.get("failure") is None:
                        coverage_entry["failure"] = saved_video_path
                    failure_video_path = saved_video_path or failure_video_path
                    saved_video_kind = "failure" if saved_video_path else None
                except Exception as exc:
                    if not args.progress_only:
                        print(f"[warn] Failed to save failure video for {instruction_type}: {exc}")
                finally:
                    clear_sim_recording_buffers(env.sim)
            elif getattr(env, "sim", None) is not None:
                clear_sim_recording_buffers(env.sim)

            if saved_video_path:
                trace_path = saved_video_path.replace("_overview.mp4", "_actions.csv")
                episode_result = replace(
                    episode_result,
                    video_path=saved_video_path,
                    video_kind=saved_video_kind,
                    action_trace_path=trace_path,
                )
                episode_results[-1] = episode_result

            if progress is not None:
                progress.set_description_str(effective_log_label)
                progress.set_postfix_str(f"success={successes}/{episode_offset + 1}")
                progress.update(1)
            elif not args.progress_only and (
                episode_result.success
                or episode_offset == 0
                or (episode_offset + 1) % max(1, int(args.log_every_episode)) == 0
                or episode_offset == (int(episodes_to_run) - 1)
            ):
                print(
                    f"[{effective_log_label}] episode={episode_offset + 1:03d}/{int(episodes_to_run):03d} "
                    f"success={episode_result.success} steps={episode_result.steps} "
                    f"reward={episode_result.reward_total:.4f} scene={episode_result.scene}"
                )

            if (
                stop_when_video_coverage_complete
                and coverage_entry.get("success")
                and coverage_entry.get("failure")
            ):
                break

        summary = _summarize_instruction_results(
            instruction_type=instruction_type,
            episode_results=episode_results,
            video_path=success_video_path or failure_video_path,
            success_video_path=success_video_path,
            failure_video_path=failure_video_path,
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
    instruction_types = _resolve_instruction_types(config, args)
    wrapper_dir = _resolve_wrapper_dir(config, args)
    instruction_buckets = {
        instruction_type: _validation_buckets(config, args, instruction_type=instruction_type)
        for instruction_type in instruction_types
    }
    metric_buckets = [
        bucket
        for instruction_type in instruction_types
        for bucket in instruction_buckets[instruction_type]
    ]
    arbitrary_buckets = _arbitrary_validation_buckets(
        config,
        args,
        instruction_types=instruction_types,
    )
    total_metric_episodes = sum(bucket.episodes for bucket in metric_buckets)
    total_validation_episodes = total_metric_episodes + sum(
        bucket.episodes for bucket in arbitrary_buckets
    )
    expected_video_keys = (
        list(instruction_types)
        if str(args.video_coverage) == "instruction"
        else [bucket.case_id for bucket in metric_buckets]
    )
    expected_video_keys = list(dict.fromkeys(expected_video_keys))

    if int(args.min_scene_objects) <= 0 or int(args.max_scene_objects) < int(args.min_scene_objects):
        raise ValueError("--min-scene-objects/--max-scene-objects define an invalid range.")
    if int(args.synonyms_per_instruction) < 0:
        raise ValueError("--synonyms-per-instruction cannot be negative.")
    if int(args.video_search_extra_episodes) < 0:
        raise ValueError("--video-search-extra-episodes cannot be negative.")
    if int(args.max_reset_attempts) <= 0:
        raise ValueError("--max-reset-attempts must be positive.")

    if not args.progress_only:
        print(f"Run directory: {run_dir}")
        print(f"Checkpoint dir: {artifacts.checkpoint_dir}")
        print(f"Adapter path: {artifacts.adapter_path}")
        print(f"Action-head path: {artifacts.action_head_path}")
        print(f"Wrapper dir: {wrapper_dir}")
        print(f"Instruction types: {list(instruction_types)}")
        print(f"Metric evaluation cases: {len(metric_buckets)}")
        print(f"Metric episodes: {total_metric_episodes}")
        print(f"Arbitrary recorded checks: {len(arbitrary_buckets)}")
        print(f"Episodes per case: {int(args.episodes_per_instruction)}")
        print(f"Episode max steps: {max_steps}")
        print(f"Evaluate reverse shells: {bool(args.evaluate_reverse_shells)}")
        print(f"Include synonyms: {bool(args.include_synonyms)}")
        print(f"Multi-object scenes: {bool(args.multi_object_scenes)}")
        if args.multi_object_scenes:
            print(f"Scene object count: {int(args.min_scene_objects)}..{int(args.max_scene_objects)}")
        print(f"Record success videos: {bool(args.record_success_videos)}")
        print(f"Record all success videos: {bool(args.record_all_success_videos)}")
        print(f"Record failure videos: {bool(args.record_failure_videos)}")
        print(f"Video coverage level: {args.video_coverage}")
        print(f"Video search extra episodes: {int(args.video_search_extra_episodes)}")
        print(f"Maximum reset attempts: {int(args.max_reset_attempts)}")
        print(f"Video action overlay: {bool(args.video_action_overlay)}")
        print(f"Validation success distance: {float(args.success_distance):.3f} m")
        print(
            "Move-to-object validation XY threshold: "
            f"{float(args.move_to_object_success_distance):.3f} m"
        )
        print(
            "Directional validation threshold: "
            f"{float(args.directional_displacement_threshold):.3f} m"
        )
        print(f"Move-to-object minimum episodes per target: {int(args.move_to_object_episodes_per_target)}")
        print(f"Reuse existing wrapper variants: {bool(args.reuse_existing_wrapper_variants)}")
        print(f"Seed mode: {'entropy' if base_seed is None else base_seed}")

    with _silence_output(bool(args.progress_only)):
        runtime = _load_policy_runtime(
            config=config,
            artifacts=artifacts,
            args=args,
            quiet=bool(args.progress_only),
        )

    video_registry: dict[str, dict[str, str]] = {}
    metric_episode_results: list[EpisodeResult] = []
    arbitrary_episode_results: list[EpisodeResult] = []
    video_search_episode_results: list[EpisodeResult] = []
    instruction_offsets = {instruction_type: 0 for instruction_type in instruction_types}
    instruction_indexes = {
        instruction_type: index for index, instruction_type in enumerate(instruction_types)
    }

    progress = _progress_bar(total=total_validation_episodes)
    try:
        for bucket in metric_buckets:
            instruction_type = bucket.instruction_type
            episode_index_offset = instruction_offsets[instruction_type]
            with _temporary_env_vars(bucket.env_vars):
                _bucket_summary, episode_results = _run_instruction_validation(
                    instruction_type=instruction_type,
                    instruction_index=instruction_indexes[instruction_type],
                    config=config,
                    runtime=runtime,
                    args=args,
                    videos_dir=videos_dir,
                    max_steps=max_steps,
                    base_seed=base_seed,
                    progress=progress,
                    wrapper_dir=wrapper_dir,
                    episodes_to_run=int(bucket.episodes),
                    episode_index_offset=int(episode_index_offset),
                    log_label=bucket.log_label,
                    validation_bucket=bucket,
                    video_registry=video_registry,
                    metric_episode=True,
                )
            metric_episode_results.extend(episode_results)
            instruction_offsets[instruction_type] += int(bucket.episodes)

        for bucket in arbitrary_buckets:
            instruction_type = bucket.instruction_type
            episode_index_offset = instruction_offsets[instruction_type]
            with _temporary_env_vars(bucket.env_vars):
                _bucket_summary, episode_results = _run_instruction_validation(
                    instruction_type=instruction_type,
                    instruction_index=instruction_indexes[instruction_type],
                    config=config,
                    runtime=runtime,
                    args=args,
                    videos_dir=videos_dir,
                    max_steps=max_steps,
                    base_seed=base_seed,
                    progress=progress,
                    wrapper_dir=wrapper_dir,
                    episodes_to_run=1,
                    episode_index_offset=int(episode_index_offset),
                    log_label=bucket.log_label,
                    validation_bucket=bucket,
                    video_registry=video_registry,
                    metric_episode=False,
                )
            arbitrary_episode_results.extend(episode_results)
            instruction_offsets[instruction_type] += 1
    finally:
        progress.close()

    if int(args.video_search_extra_episodes) > 0:
        for instruction_type in instruction_types:
            coverage_key = instruction_type
            if str(args.video_coverage) == "case":
                missing_case_buckets = [
                    bucket
                    for bucket in metric_buckets
                    if bucket.instruction_type == instruction_type
                    and not (
                        video_registry.get(bucket.case_id, {}).get("success")
                        and video_registry.get(bucket.case_id, {}).get("failure")
                    )
                ]
                search_buckets = missing_case_buckets
            else:
                if (
                    video_registry.get(coverage_key, {}).get("success")
                    and video_registry.get(coverage_key, {}).get("failure")
                ):
                    continue
                candidates = [
                    bucket
                    for bucket in instruction_buckets[instruction_type]
                    if bucket.prompt_kind == "canonical"
                    and (
                        bucket.curriculum_shell is None
                        or bucket.curriculum_shell_count is None
                        or bucket.curriculum_shell >= bucket.curriculum_shell_count - 1
                    )
                ]
                search_buckets = candidates[:1]

            for search_bucket in search_buckets:
                with _temporary_env_vars(search_bucket.env_vars):
                    _summary, episode_results = _run_instruction_validation(
                        instruction_type=instruction_type,
                        instruction_index=instruction_indexes[instruction_type],
                        config=config,
                        runtime=runtime,
                        args=args,
                        videos_dir=videos_dir,
                        max_steps=max_steps,
                        base_seed=base_seed,
                        progress=None,
                        wrapper_dir=wrapper_dir,
                        episodes_to_run=int(args.video_search_extra_episodes),
                        episode_index_offset=10_000_000 + int(instruction_offsets[instruction_type]),
                        log_label=f"video-search:{search_bucket.log_label}",
                        validation_bucket=search_bucket,
                        video_registry=video_registry,
                        metric_episode=False,
                        stop_when_video_coverage_complete=True,
                    )
                video_search_episode_results.extend(episode_results)
                instruction_offsets[instruction_type] += len(episode_results)

    all_episode_results = [
        *metric_episode_results,
        *arbitrary_episode_results,
        *video_search_episode_results,
    ]
    instruction_summaries: list[InstructionSummary] = []
    for instruction_type in instruction_types:
        items = [
            result
            for result in metric_episode_results
            if result.instruction_type == instruction_type
        ]
        registry_entry = video_registry.get(instruction_type, {})
        instruction_summaries.append(
            _summarize_instruction_results(
                instruction_type=instruction_type,
                episode_results=items,
                video_path=registry_entry.get("success") or registry_entry.get("failure"),
                success_video_path=registry_entry.get("success"),
                failure_video_path=registry_entry.get("failure"),
            )
        )

    instruction_text_summaries = _summarize_instruction_text_results(metric_episode_results)
    instruction_rows = _aggregate_episode_results(
        metric_episode_results,
        group_fields=("instruction_type",),
    )
    normal_canonical_results = [
        item for item in metric_episode_results if _is_normal_scene_canonical_episode(item)
    ]
    normal_canonical_rows = _aggregate_episode_results(
        normal_canonical_results,
        group_fields=("instruction_type",),
    )
    shell_rows = _aggregate_episode_results(
        metric_episode_results,
        group_fields=("instruction_type", "curriculum_shell"),
    )
    prompt_rows = _aggregate_episode_results(
        metric_episode_results,
        group_fields=("instruction_type", "prompt_kind", "prompt_variant"),
    )
    case_rows = _aggregate_episode_results(
        metric_episode_results,
        group_fields=(
            "instruction_type",
            "curriculum_shell",
            "prompt_kind",
            "prompt_variant",
            "target_object_catalog",
        ),
    )
    target_rows = _aggregate_episode_results(
        metric_episode_results,
        group_fields=("instruction_type", "target_object_catalog"),
    )
    move_to_object_threshold_rows = _move_to_object_threshold_sweep(metric_episode_results)

    csv_path = run_dir / "instruction_success_rates.csv"
    _write_success_rate_csv(csv_path, instruction_summaries)
    text_csv_path = run_dir / "instruction_text_success_rates.csv"
    _write_instruction_text_csv(text_csv_path, instruction_text_summaries)
    _write_grouped_success_rate_csv(
        run_dir / "normal_scene_canonical_success_rates.csv",
        normal_canonical_rows,
        group_fields=("instruction_type",),
    )
    _write_grouped_success_rate_csv(
        run_dir / "instruction_shell_success_rates.csv",
        shell_rows,
        group_fields=("instruction_type", "curriculum_shell"),
    )
    _write_grouped_success_rate_csv(
        run_dir / "instruction_prompt_success_rates.csv",
        prompt_rows,
        group_fields=("instruction_type", "prompt_kind", "prompt_variant"),
    )
    _write_grouped_success_rate_csv(
        run_dir / "evaluation_case_success_rates.csv",
        case_rows,
        group_fields=(
            "instruction_type",
            "curriculum_shell",
            "prompt_kind",
            "prompt_variant",
            "target_object_catalog",
        ),
    )
    _write_grouped_success_rate_csv(
        run_dir / "target_object_success_rates.csv",
        target_rows,
        group_fields=("instruction_type", "target_object_catalog"),
    )
    _write_episode_results_csv(run_dir / "episode_results.csv", all_episode_results)
    _write_move_to_object_threshold_sweep(
        run_dir / "move_to_object_threshold_sweep.csv",
        move_to_object_threshold_rows,
    )

    video_probes, video_coverage = _write_video_audit(
        run_dir=run_dir,
        videos_dir=videos_dir,
        expected_keys=expected_video_keys,
        video_registry=video_registry,
    )
    report_path = _write_validation_report(
        run_dir=run_dir,
        artifacts=artifacts,
        metric_results=metric_episode_results,
        all_results=all_episode_results,
        instruction_rows=instruction_rows,
        normal_canonical_rows=normal_canonical_rows,
        shell_rows=shell_rows,
        prompt_rows=prompt_rows,
        target_rows=target_rows,
        video_probes=video_probes,
        video_coverage=video_coverage,
        move_to_object_threshold_rows=move_to_object_threshold_rows,
    )

    instruction_episodes = {
        instruction_type: [
            asdict(result)
            for result in all_episode_results
            if result.instruction_type == instruction_type
        ]
        for instruction_type in instruction_types
    }
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
        "episodes_per_case": int(args.episodes_per_instruction),
        "total_metric_episodes": int(len(metric_episode_results)),
        "total_arbitrary_episodes": int(len(arbitrary_episode_results)),
        "total_video_search_episodes": int(len(video_search_episode_results)),
        "max_steps": int(max_steps),
        "chunk_length": int(runtime["chunk_length"]),
        "replan_every": int(runtime["replan_every"] or runtime["chunk_length"]),
        "num_images_in_input": int(runtime["num_images_in_input"]),
        "center_crop": bool(args.center_crop),
        "hold_steps": int(_control_spec_from_config(config, args.hold_steps).hold_steps),
        "seed": base_seed,
        "record_success_videos": bool(args.record_success_videos),
        "record_all_success_videos": bool(args.record_all_success_videos),
        "record_failure_videos": bool(args.record_failure_videos),
        "video_coverage_level": str(args.video_coverage),
        "video_search_extra_episodes": int(args.video_search_extra_episodes),
        "max_reset_attempts": int(args.max_reset_attempts),
        "video_action_overlay": bool(args.video_action_overlay),
        "success_distance": float(args.success_distance),
        "move_to_object_success_distance": float(args.move_to_object_success_distance),
        "directional_displacement_threshold": float(args.directional_displacement_threshold),
        "move_to_object_episodes_per_target": int(args.move_to_object_episodes_per_target),
        "stratify_move_to_object_targets": bool(args.stratify_move_to_object_targets),
        "multi_object_scenes": bool(args.multi_object_scenes),
        "min_scene_objects": int(args.min_scene_objects),
        "max_scene_objects": int(args.max_scene_objects),
        "evaluate_reverse_shells": bool(args.evaluate_reverse_shells),
        "include_synonyms": bool(args.include_synonyms),
        "synonyms_per_instruction": int(args.synonyms_per_instruction),
        "synonym_shells": str(args.synonym_shells),
        "arbitrary_instructions_count": int(args.arbitrary_instructions_count),
        "reuse_existing_wrapper_variants": bool(args.reuse_existing_wrapper_variants),
        "evaluation_cases": [
            {
                "case_id": bucket.case_id,
                "instruction_type": bucket.instruction_type,
                "target_object": bucket.target_object,
                "episodes": bucket.episodes,
                "prompt_kind": bucket.prompt_kind,
                "prompt_variant": bucket.prompt_variant,
                "prompt_template": bucket.prompt_template,
                "curriculum_shell": bucket.curriculum_shell,
                "curriculum_shell_count": bucket.curriculum_shell_count,
            }
            for bucket in [*metric_buckets, *arbitrary_buckets]
        ],
        "instruction_summaries": [asdict(summary) for summary in instruction_summaries],
        "normal_scene_canonical_summaries": normal_canonical_rows,
        "instruction_text_summaries": [asdict(summary) for summary in instruction_text_summaries],
        "move_to_object_threshold_sweep": move_to_object_threshold_rows,
        "total_policy_output_calls": int(
            sum(item.policy_output_calls for item in all_episode_results)
        ),
        "total_action_steps": int(sum(item.action_steps for item in all_episode_results)),
        "total_reset_retries": int(
            sum(max(0, item.reset_attempts - 1) for item in all_episode_results)
        ),
        "simulation_instability_episodes": int(
            sum(bool(item.simulation_instability) for item in all_episode_results)
        ),
        "video_registry": video_registry,
        "video_validation": video_probes,
        "video_coverage": video_coverage,
        "report_path": report_path.as_posix(),
        "episodes": instruction_episodes,
    }

    manifest_path = run_dir / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    invalid_videos = [probe for probe in video_probes if not bool(probe.get("valid"))]
    incomplete_coverage = [item for item in video_coverage if not bool(item.get("complete"))]
    if not args.progress_only:
        print(f"Manifest saved: {manifest_path}")
        print(f"Report saved: {report_path}")
        print(f"Instruction CSV saved: {csv_path}")
        print(f"Instruction text CSV saved: {text_csv_path}")
        print(f"Video validation failures: {len(invalid_videos)}")
        print(f"Incomplete video coverage entries: {len(incomplete_coverage)}")
    if bool(args.strict_video_validation) and invalid_videos:
        return 3
    if bool(args.require_complete_video_coverage) and incomplete_coverage:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
