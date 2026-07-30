#!/usr/bin/env python3
"""Record oracle reference episodes for the manipulation curriculum.

These videos answer one question: *what does a complete, correct episode of this
instruction look like, and what does the training reward say while it happens?*

Nothing here is a learned policy.  A scripted oracle drives the same five
normalized actions SmolVLA emits, through the same controller, so the recorded
motion is the behaviour the reward is asking for.

Everything that defines the task is imported from the training code rather than
re-implemented:

* the episode start comes from ``BatchedReverseFrontierResetter`` -- the exact
  reset the trainer runs, including the approach curriculum cap, the caught
  object start for placement, and the fitted gripper opening;
* the reward and success predicate come from ``evaluate_active_sparse_tasks``
  with ``BatchedCatchReleaseDenseReward.from_metadata(<config metadata>)``;
* the grasp evidence comes from the trainer's own
  ``RankLocalMJWarpGRPOCollector._update_physical_grasp`` (persistent bilateral
  pad contact, solved normal force, and relative-pose stability).

The only substitution is the physics engine.  MJ-Lab/MJWarp is CUDA-only, so on
a machine without an NVIDIA GPU the scene runs on MuJoCo's CPU pipeline using
the *same* fixed-topology MJWarp MJCF, catalog meshes, colliders and cameras.
Every video, CSV row and manifest entry records which backend produced it.
"""

from __future__ import annotations

import os as _os

# MuJoCo picks its GL backend at import time. On a headless box GLFW fails with
# "gladLoadGL error" / "DISPLAY environment variable is missing"; EGL is what the
# training backend already uses there. Set before any mujoco import, and only if
# the caller has not chosen a backend.
_os.environ.setdefault("MUJOCO_GL", "egl")
_os.environ.setdefault("PYOPENGL_PLATFORM", _os.environ["MUJOCO_GL"])

# Finger-pad centre below ee_base, measured from the MJCF. See
# rl_vla_bootstrapping/simulation/cdpr_gripper_geometry.py.
_MEASURED_PAD_OFFSET_M = 0.0075

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CDPR_ROOT = ROOT / "robots" / "cdpr"
if str(CDPR_ROOT) not in sys.path:
    sys.path.insert(0, str(CDPR_ROOT))

from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (  # noqa: E402
    BatchedReverseFrontierResetter,
    RankLocalCurriculum,
    RankLocalMJWarpGRPOCollector,
)
from rl_vla_bootstrapping.policy.rank_local_grpo import (  # noqa: E402
    RankLocalGroupLayout,
)
from rl_vla_bootstrapping.simulation.cdpr_backend import (  # noqa: E402
    CDPRBackendConfig,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (  # noqa: E402
    ACTIVE_INSTRUCTION_TYPES,
    INSTRUCTION_TO_ID,
    BatchedCatchReleaseDenseReward,
    BatchedTaskThresholds,
    _fine_distance_reward,
    evaluate_active_sparse_tasks,
    inverse_polynomial_distance_reward,
)
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (  # noqa: E402
    ACTIVE_CDPR_CATALOGS,
    OBJECT_VARIANTS,
)
from rl_vla_bootstrapping.simulation.mujoco_reference_batched_backend import (  # noqa: E402
    MujocoReferenceBatchedBackend,
)

DEFAULT_CONFIG = (
    ROOT / "configs" / "examples" / "cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml"
)
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_task_reference_episodes"
SUPPORTED_INSTRUCTIONS = ("pick_up", "put_into_plate", "put_into_bowl")
ACTION_NAMES = ("x", "y", "z", "yaw", "gripper")
CAMERA_LABELS = {"overview": "OVERVIEW", "ee_camera": "WRIST (ee_camera)"}
# The reset seats a caught object with the fingers this far inside the fitted
# opening; the oracle closes to the same depth so the pads load identically.
_GRASP_SQUEEZE = 0.001 / 0.03


# --------------------------------------------------------------------- oracle


@dataclass
class OraclePhase:
    name: str
    goal: Callable[["EpisodeContext", dict[str, Any]], np.ndarray]
    gripper: Callable[["EpisodeContext", dict[str, Any]], float]
    done: Callable[["EpisodeContext", dict[str, Any], int], bool]
    speed: float = 1.0
    # The platform is a cable robot: the controller target is re-issued relative
    # to the *measured* pose every step, so a pure proportional command chases
    # its own overshoot and limit-cycles at a few centimetres. Damping on the
    # measured per-step velocity is what makes the reference motion settle.
    kp: float = 0.55
    kd: float = 1.30


@dataclass
class EpisodeContext:
    instruction_type: str
    instruction_text: str
    target_slot: int
    reference_slot: int
    target_catalog: str
    fitted_opening: float
    release_opening: float
    grasp_height_offset: float
    release_height: float
    support_surface_z: float
    lift_success_height: float
    frozen: dict[str, Any] = field(default_factory=dict)

    @property
    def close_opening(self) -> float:
        return float(max(0.0, self.fitted_opening - _GRASP_SQUEEZE))


def _xy_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def _grasp_point(ctx: EpisodeContext, obs: dict[str, Any]) -> np.ndarray:
    target = np.asarray(obs["object_positions"][ctx.target_slot], dtype=np.float64)
    return target + np.array((0.0, 0.0, ctx.grasp_height_offset))


def _placement_hover(ctx: EpisodeContext, obs: dict[str, Any]) -> np.ndarray:
    reference = np.asarray(obs["object_positions"][ctx.reference_slot], dtype=np.float64)
    return np.array(
        (
            reference[0],
            reference[1],
            reference[2] + ctx.release_height + ctx.grasp_height_offset,
        )
    )


def _settled(obs: dict[str, Any], goal: np.ndarray, tolerance: float) -> bool:
    """Inside the tolerance *and* not still swinging through it."""

    return (
        float(np.linalg.norm(np.asarray(obs["ee"]) - np.asarray(goal))) <= tolerance
        and float(np.linalg.norm(obs["ee_velocity"])) <= 0.0020
    )


def _pick_up_phases() -> tuple[OraclePhase, ...]:
    def above(ctx: EpisodeContext, obs: dict[str, Any]) -> np.ndarray:
        grasp = _grasp_point(ctx, obs)
        # Frozen at phase entry: a hover height recomputed from the live ee_z
        # ratchets upward with every overshoot and the approach never converges.
        height = ctx.frozen.setdefault(
            "hover_z", max(float(obs["ee"][2]), float(grasp[2]) + 0.045)
        )
        return np.array((grasp[0], grasp[1], height))

    return (
        OraclePhase(
            "align_above_object",
            above,
            lambda ctx, obs: 1.0,
            lambda ctx, obs, steps: steps >= 3 and _settled(obs, above(ctx, obs), 0.010),
        ),
        OraclePhase(
            "descend_to_grasp_point",
            _grasp_point,
            lambda ctx, obs: 1.0,
            lambda ctx, obs, steps: steps >= 3
            and _settled(obs, _grasp_point(ctx, obs), 0.008),
            speed=0.45,
        ),
        OraclePhase(
            # Close until the physics says the pads have latched, rather than to
            # a tabulated opening. A tabulated target is only as good as the
            # calibration behind it; closing on contact is what a competent
            # policy does and it makes the reference episode independent of that
            # table.
            "close_fingers",
            lambda ctx, obs: ctx.frozen.setdefault("grasp_hold", obs["ee"].copy()),
            lambda ctx, obs: 0.0,
            lambda ctx, obs, steps: steps >= 2 and _latched(ctx, obs),
            speed=0.20,
        ),
        OraclePhase(
            "lift_object",
            lambda ctx, obs: ctx.frozen.setdefault(
                "lift_goal",
                np.asarray(obs["ee"], dtype=np.float64)
                + np.array((0.0, 0.0, ctx.lift_success_height + 0.035)),
            ),
            _hold_grip,
            lambda ctx, obs, steps: float(obs["target_lift"])
            >= ctx.lift_success_height + 0.006,
            speed=0.60,
        ),
        OraclePhase(
            "hold_lifted",
            lambda ctx, obs: ctx.frozen.setdefault("hold_pose", obs["ee"].copy()),
            _hold_grip,
            lambda ctx, obs, steps: False,
            speed=0.20,
        ),
    )


def _latched(ctx: EpisodeContext, obs: dict[str, Any]) -> bool:
    if not bool(obs["physical_grasp"]):
        return False
    # Squeeze a little past the latch point so the hold survives the lift.
    ctx.frozen.setdefault(
        "hold_grip", max(0.0, float(obs["commanded_gripper"]) - 0.06)
    )
    return True


def _hold_grip(ctx: EpisodeContext, obs: dict[str, Any]) -> float:
    return float(ctx.frozen.get("hold_grip", ctx.close_opening))


def _placement_phases() -> tuple[OraclePhase, ...]:
    def transit(ctx: EpisodeContext, obs: dict[str, Any]) -> np.ndarray:
        hover = _placement_hover(ctx, obs)
        height = ctx.frozen.setdefault(
            "transit_z", max(float(obs["ee"][2]), float(hover[2]) + 0.055)
        )
        return np.array((float(obs["ee"][0]), float(obs["ee"][1]), height))

    def traverse(ctx: EpisodeContext, obs: dict[str, Any]) -> np.ndarray:
        hover = _placement_hover(ctx, obs)
        return np.array((hover[0], hover[1], ctx.frozen["transit_z"]))

    def hold_closed(ctx: EpisodeContext, obs: dict[str, Any]) -> float:
        # Whatever opening the reset actually left the fingers at: the placement
        # episode is defined to START holding the object, so the oracle's job is
        # to keep that grip, not to re-establish it.
        return float(ctx.frozen.setdefault("hold_grip", float(obs["commanded_gripper"])))

    return (
        OraclePhase(
            "raise_to_transit_height",
            transit,
            hold_closed,
            lambda ctx, obs, steps: steps >= 2
            and float(obs["ee"][2]) >= ctx.frozen["transit_z"] - 0.008,
        ),
        OraclePhase(
            "traverse_to_receptacle",
            traverse,
            hold_closed,
            lambda ctx, obs, steps: steps >= 2
            and _xy_error(obs["ee"], _placement_hover(ctx, obs)) <= 0.010
            and float(np.linalg.norm(obs["ee_velocity"][:2])) <= 0.0020,
        ),
        OraclePhase(
            "descend_to_release_height",
            _placement_hover,
            hold_closed,
            lambda ctx, obs, steps: steps >= 3
            and _settled(obs, _placement_hover(ctx, obs), 0.010),
            speed=0.45,
        ),
        OraclePhase(
            "open_gripper_release",
            lambda ctx, obs: ctx.frozen.setdefault("release_pose", obs["ee"].copy()),
            lambda ctx, obs: ctx.release_opening,
            lambda ctx, obs, steps: steps >= 2
            and float(obs["measured_gripper"]) >= ctx.release_opening - 1.0e-3,
            speed=0.20,
        ),
        OraclePhase(
            "settle_in_receptacle",
            lambda ctx, obs: ctx.frozen["release_pose"],
            lambda ctx, obs: 1.0,
            lambda ctx, obs, steps: False,
            speed=0.20,
        ),
    )


def oracle_phases(instruction_type: str) -> tuple[OraclePhase, ...]:
    if instruction_type == "pick_up":
        return _pick_up_phases()
    if instruction_type in ("put_into_plate", "put_into_bowl"):
        return _placement_phases()
    raise ValueError(f"No oracle defined for instruction {instruction_type!r}.")


def oracle_action(
    ctx: EpisodeContext,
    obs: dict[str, Any],
    phase: OraclePhase,
    action_step_xyz: float,
    action_step_gripper: float,
) -> np.ndarray:
    goal = np.asarray(phase.goal(ctx, obs), dtype=np.float64)
    error = goal - np.asarray(obs["ee"], dtype=np.float64)
    velocity = np.asarray(obs["ee_velocity"], dtype=np.float64)
    command = phase.kp * error - phase.kd * velocity
    action = np.zeros(5, dtype=np.float64)
    action[:3] = np.clip(command / float(action_step_xyz), -phase.speed, phase.speed)
    action[3] = 0.0
    gripper_goal = float(phase.gripper(ctx, obs))
    action[4] = float(
        np.clip(
            (gripper_goal - float(obs["commanded_gripper"])) / float(action_step_gripper),
            -1.0,
            1.0,
        )
    )
    return np.clip(action, -1.0, 1.0)


# ------------------------------------------------------------- reward tracing


def _reward_breakdown(
    *,
    instruction_type: str,
    reward_config: BatchedCatchReleaseDenseReward,
    diagnostics: dict[str, Any],
    grasp_flags: dict[str, float],
    success: bool,
    wrong_place: bool,
    torch: Any,
) -> dict[str, float]:
    """Recompute the reward term by term with the production helpers.

    The caller asserts the terms sum to the reward the trainer's own function
    returned, so the on-screen breakdown can never drift from the real reward.
    """

    def scalar(name: str) -> Any:
        return diagnostics[name][:1]

    terms: dict[str, float] = {}
    if instruction_type == "pick_up":
        distance = scalar("pick_grasp_distance")
        coarse = inverse_polynomial_distance_reward(
            distance,
            window_high=float(reward_config.pick_distance_window),
            scale=reward_config.distance_reward_scale,
            weight=reward_config.distance_reward_weight,
            exponent=reward_config.distance_reward_exponent,
        )
        fine = _fine_distance_reward(distance, config=reward_config)
        lift = float(diagnostics["target_lift"][0])
        normalized_lift = min(
            max(lift / float(reward_config.pick_lift_reward_scale), 0.0), 1.0
        )
        grasped = float(grasp_flags["grasped"])
        terms["distance_coarse"] = float(coarse[0])
        terms["distance_fine"] = float(fine[0])
        terms["contact_bonus"] = float(
            grasp_flags["bilateral_contact"] * reward_config.pick_contact_bonus
        )
        terms["grasp_bonus"] = float(grasped * reward_config.pick_grasp_bonus)
        terms["lift"] = float(
            normalized_lift * grasped * reward_config.pick_lift_reward_weight
        )
        terms["success_bonus"] = float(
            (1.0 if success else 0.0) * reward_config.pick_success_bonus
        )
        return terms

    distance = scalar("placement_distance")
    window = (
        float(reward_config.placement_distance_window)
        if float(reward_config.placement_distance_window) > 0.0
        else float(diagnostics["container_xy_radius"][0])
    )
    coarse = inverse_polynomial_distance_reward(
        distance,
        window_high=window,
        scale=reward_config.distance_reward_scale,
        weight=reward_config.distance_reward_weight,
        exponent=reward_config.distance_reward_exponent,
    )
    fine = _fine_distance_reward(distance, config=reward_config)
    terms["distance_coarse"] = float(coarse[0])
    terms["distance_fine"] = float(fine[0])
    terms["success_bonus"] = float(
        (1.0 if success else 0.0) * reward_config.placement_success_bonus
    )
    terms["wrong_drop_penalty"] = float(
        -(1.0 if wrong_place else 0.0) * reward_config.placement_failure_penalty
    )
    del torch
    return terms


# ------------------------------------------------------------------- overlays


def _telemetry_lines(row: dict[str, Any]) -> list[tuple[str, str]]:
    """(kind, text) pairs; kind selects the colour."""

    def vec(values: Sequence[float], digits: int = 3) -> str:
        return " ".join(f"{float(v):+.{digits}f}" for v in values)

    lines: list[tuple[str, str]] = [
        ("head", f"{row['instruction_text']}   [{row['instruction_type']}]"),
        (
            "info",
            f"episode {row['episode']}  phase={row['phase']}  "
            f"decision {row['decision']}/{row['horizon']}  env_step {row['env_step']}",
        ),
        (
            "action",
            f"action [{vec(row['action'], 2)}]  ->  d_xyz [{vec(row['delta_xyz'], 4)}] m  "
            f"d_grip {row['delta_gripper']:+.3f}",
        ),
        (
            "info",
            f"ee [{vec(row['ee'])}]  yaw {row['ee_yaw']:+.3f}  "
            f"gripper {row['gripper_opening']:.3f} (cmd {row['commanded_gripper']:.3f})",
        ),
    ]
    if row["instruction_type"] == "pick_up":
        lines.append(
            (
                "info",
                f"object [{vec(row['target_position'])}]  "
                f"grasp_point [{vec(row['grasp_point'])}]  "
                f"dist {row['task_distance']:.4f} m",
            )
        )
        lines.append(
            (
                "info",
                f"lift {row['target_lift']*1000:6.1f} mm / "
                f"{row['lift_success_height']*1000:.0f} mm needed",
            )
        )
    else:
        lines.append(
            (
                "info",
                f"object [{vec(row['target_position'])}]  "
                f"receptacle [{vec(row['reference_position'])}]  "
                f"hover_dist {row['task_distance']:.4f} m",
            )
        )
        lines.append(
            (
                "info",
                f"obj->receptacle xy {row['container_xy_error']:.4f} / "
                f"{row['container_xy_radius']:.3f} m   dz {row['container_z_error']:.4f} / "
                f"{row['container_z_tolerance']:.3f} m   settled={int(row['target_has_settled'])}",
            )
        )
    lines.append(
        (
            "grasp",
            f"pads L/R contact {int(row['left_contact'])}/{int(row['right_contact'])}  "
            f"force {row['left_pad_force_n']:.3f}/{row['right_pad_force_n']:.3f} N  "
            f"slip {row['relative_position_slip_m']*1000:.2f} mm  "
            f"grasp={int(row['physical_grasp'])} ever={int(row['ever_grasped'])} "
            f"released={int(row['released'])}",
        )
    )
    terms = "  ".join(
        f"{name}={value:+.3f}" for name, value in row["reward_terms"].items()
    )
    lines.append(("reward", f"REWARD {row['reward']:+.4f}  =  {terms}"))
    lines.append(
        (
            "reward",
            f"success={int(row['success'])}  terminated={int(row['terminated'])}  "
            f"backend={row['backend']}",
        )
    )
    return lines


_COLORS = {
    "head": (255, 214, 122),
    "info": (232, 236, 242),
    "action": (150, 200, 255),
    "grasp": (168, 236, 180),
    "reward": (255, 176, 176),
}


def _annotate(frame: np.ndarray, row: dict[str, Any], title: str) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    font = ImageFont.load_default()
    max_chars = max(30, int((width - 14) // 6))
    wrapped: list[tuple[str, str]] = []
    for kind, line in [("head", title), *_telemetry_lines(row)]:
        pieces = (
            textwrap.wrap(line, width=max_chars, break_long_words=False, break_on_hyphens=False)
            or [""]
        )
        wrapped.extend((kind, piece) for piece in pieces)
    line_height = 12
    panel_height = 10 + line_height * len(wrapped)
    image = Image.new("RGB", (width, height + panel_height), (13, 16, 22))
    image.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(image)
    for index, (kind, line) in enumerate(wrapped):
        draw.text(
            (7, height + 5 + line_height * index),
            line,
            fill=_COLORS.get(kind, _COLORS["info"]),
            font=font,
        )
    return np.asarray(image)


def _write_video(frames: Sequence[np.ndarray], output: Path, *, fps: float) -> None:
    if not frames:
        raise RuntimeError(f"No frames captured for {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.stem}_frames"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for index, frame in enumerate(frames):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                staging / f"{index:05d}.png"
            )
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", f"{float(fps):.8g}",
                "-i", str(staging / "%05d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------- the rollout


def seat_held_object(
    *,
    backend: MujocoReferenceBatchedBackend,
    reset: Any,
    grasp_height_offset: float,
    torch: Any,
) -> dict[str, Any]:
    """Re-seat a "already holding" reset so the object is really between the pads.

    The production reset drops the carried object ``0.08 m`` below ``ee_base``
    and calls it grasped. The finger pads sit ``0.0075 m`` below ``ee_base``, so
    that pose leaves the object hanging in free space. This puts it at the pad
    centre and then closes the fingers until the physics reports bilateral pad
    contact, which is what "the gripper starts holding the object" has to mean.

    Reset-time only: it costs no episode steps, exactly like the trainer's own
    ``set_gripper_openings``.
    """

    worlds = backend.worlds_per_rank
    low_dim = backend.low_dim_observations()
    slots = reset.task_state.target_slots
    positions = low_dim.object_positions.clone()
    rows = torch.arange(worlds, dtype=torch.int64, device=backend.device)
    seated = low_dim.ee_position.clone()
    seated[:, 2] = low_dim.ee_position[:, 2] - float(grasp_height_offset)
    positions[rows, slots] = seated
    backend.set_free_body_poses(
        backend.object_body_ids, positions, low_dim.object_quaternions
    )
    opening = float(backend.controller_state()["gripper"][0])
    closed_steps = 0
    while opening > 0.0:
        opening = max(0.0, opening - 0.02)
        backend.set_gripper_openings(torch.full((worlds,), opening))
        backend.set_free_body_poses(
            backend.object_body_ids, positions, low_dim.object_quaternions
        )
        contacts = backend.finger_object_contact_metrics(slots)
        closed_steps += 1
        if bool(contacts.bilateral_contact[0].item()) and float(
            contacts.left_normal_force[0].item()
        ) >= 0.05 and float(contacts.right_normal_force[0].item()) >= 0.05:
            break
    # A little extra squeeze so the grip survives the first transport step.
    opening = max(0.0, opening - 0.04)
    backend.set_gripper_openings(torch.full((worlds,), opening))
    backend.set_free_body_poses(
        backend.object_body_ids, positions, low_dim.object_quaternions
    )
    contacts = backend.finger_object_contact_metrics(slots)
    return {
        "seated_object_z_below_ee_m": float(grasp_height_offset),
        "closing_increments": closed_steps,
        "seated_gripper_opening": opening,
        "seated_left_pad_force_n": float(contacts.left_normal_force[0].item()),
        "seated_right_pad_force_n": float(contacts.right_normal_force[0].item()),
        "seated_bilateral_contact": bool(contacts.bilateral_contact[0].item()),
    }


class _GraspShim:
    """Minimal host for the trainer's own ``_update_physical_grasp``.

    Calling the production method unbound guarantees the grasp evidence shown in
    these videos is byte-for-byte the rule the trainer applies -- no second
    implementation to drift.
    """

    def __init__(self, *, backend: Any, torch: Any, worlds: int) -> None:
        self.backend = backend
        self.torch = torch
        self._world_rows = torch.arange(worlds, dtype=torch.int64, device=backend.device)

    def update(self, reset: Any, low_dim: Any, active: Any) -> Any:
        return RankLocalMJWarpGRPOCollector._update_physical_grasp(
            self, reset, low_dim, active
        )


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_episode(
    *,
    backend: MujocoReferenceBatchedBackend,
    resetter: BatchedReverseFrontierResetter,
    grasp: _GraspShim,
    reward_config: BatchedCatchReleaseDenseReward,
    thresholds: BatchedTaskThresholds,
    instruction_type: str,
    episode: int,
    round_index: int,
    actions_per_decision: int,
    action_step_xyz: float,
    action_step_gripper: float,
    support_surface_z: float,
    continue_after_terminal: bool,
    grasp_height_offset: float,
    reseat_held_object: bool,
    release_opening_threshold: float | None,
    render: bool,
    torch: Any,
) -> dict[str, Any]:
    reset = resetter.reset(update_index=0, round_index=round_index)
    world = 0
    task_state = reset.task_state
    if release_opening_threshold is not None:
        task_state.release_threshold.fill_(float(release_opening_threshold))
    seating: dict[str, Any] | None = None
    if reseat_held_object and bool(task_state.grasped[world].item()):
        seating = seat_held_object(
            backend=backend,
            reset=reset,
            grasp_height_offset=grasp_height_offset,
            torch=torch,
        )
        # The carried object moved, so the relative-pose baseline the grasp
        # detector compares against must be re-taken or step 1 reads as slip.
        refreshed = backend.low_dim_observations()
        rows = torch.arange(
            backend.worlds_per_rank, dtype=torch.int64, device=backend.device
        )
        target_now = refreshed.object_positions[rows, task_state.target_slots]
        reset.previous_relative_position.copy_(target_now - refreshed.ee_position)
    target_slot = int(task_state.target_slots[world].item())
    reference_slot = int(task_state.reference_slots[world].item())
    catalog_id = int(reset.group_target_catalog_ids[0].item())
    target_catalog = ACTIVE_CDPR_CATALOGS[catalog_id]
    variant = OBJECT_VARIANTS[target_catalog]
    release_height = (
        float(reward_config.bowl_release_height)
        if instruction_type == "put_into_bowl"
        else float(reward_config.plate_release_height)
    )
    ctx = EpisodeContext(
        instruction_type=instruction_type,
        instruction_text=reset.instructions[world],
        target_slot=target_slot,
        reference_slot=max(reference_slot, 0),
        target_catalog=target_catalog,
        fitted_opening=float(variant.fitted_gripper_opening),
        release_opening=float(min(1.0, float(task_state.release_threshold[world].item()) + 0.05)),
        grasp_height_offset=float(grasp_height_offset),
        release_height=release_height,
        support_surface_z=float(support_surface_z),
        lift_success_height=float(reward_config.pick_lift_success_height),
    )
    phases = oracle_phases(instruction_type)
    phase_index = 0
    phase_steps = 0

    worlds = backend.worlds_per_rank
    horizon = int(reset.horizons[world].item())
    active = torch.ones((worlds,), dtype=torch.bool, device=backend.device)
    low_dim = backend.low_dim_observations()

    frames: dict[str, list[np.ndarray]] = {"overview": [], "ee_camera": [], "composite": []}
    rows: list[dict[str, Any]] = []
    env_step = 0
    terminated_at: int | None = None
    reached_success = False

    start_snapshot = {
        "ee": np.asarray(low_dim.ee_position[world].tolist(), dtype=float).tolist(),
        "gripper_opening": float(low_dim.gripper_opening[world].item()),
        "object_positions": np.asarray(
            low_dim.object_positions[world].tolist(), dtype=float
        ).tolist(),
        "target_slot": target_slot,
        "reference_slot": reference_slot,
        "target_catalog": target_catalog,
        "fitted_gripper_opening": ctx.fitted_opening,
        "release_threshold": float(task_state.release_threshold[world].item()),
        "starts_grasped": bool(task_state.grasped[world].item()),
        "horizon_decisions": horizon,
    }
    ee0 = np.asarray(low_dim.ee_position[world].tolist(), dtype=float)
    obj0 = np.asarray(low_dim.object_positions[world][target_slot].tolist(), dtype=float)
    ref0 = np.asarray(
        low_dim.object_positions[world][max(reference_slot, 0)].tolist(), dtype=float
    )
    start_snapshot["ee_to_object_m"] = float(np.linalg.norm(ee0 - obj0))
    start_snapshot["ee_to_grasp_point_m"] = float(
        np.linalg.norm(ee0 - (obj0 + np.array((0.0, 0.0, ctx.grasp_height_offset))))
    )
    start_snapshot["ee_to_object_xy_m"] = float(np.linalg.norm((ee0 - obj0)[:2]))
    if reference_slot >= 0:
        hover0 = np.array(
            (ref0[0], ref0[1], ref0[2] + release_height + ctx.grasp_height_offset)
        )
        start_snapshot["ee_to_placement_hover_m"] = float(np.linalg.norm(ee0 - hover0))

    previous_ee = np.asarray(low_dim.ee_position[world].tolist(), dtype=np.float64)
    last_physical_grasp = bool(reset.physical_grasp[world].item())
    for decision in range(horizon):
        for _ in range(actions_per_decision):
            controller = backend.controller_state()
            current_ee = np.asarray(low_dim.ee_position[world].tolist(), dtype=np.float64)
            obs = {
                "ee": current_ee,
                "ee_velocity": current_ee - previous_ee,
                "physical_grasp": last_physical_grasp,
                "ee_yaw": float(low_dim.ee_yaw[world].item()),
                "measured_gripper": float(low_dim.gripper_opening[world].item()),
                "commanded_gripper": float(controller["gripper"][world]),
                "object_positions": np.asarray(
                    low_dim.object_positions[world].tolist(), dtype=np.float64
                ),
                "target_lift": 0.0,
            }
            obs["target_lift"] = max(
                0.0,
                float(obs["object_positions"][target_slot][2])
                - float(task_state.initial_target_positions[world][2].item()),
            )
            while phase_index < len(phases) - 1 and phases[phase_index].done(
                ctx, obs, phase_steps
            ):
                phase_index += 1
                phase_steps = 0
            phase = phases[phase_index]
            action = oracle_action(
                ctx, obs, phase, action_step_xyz, action_step_gripper
            )
            phase_steps += 1
            previous_ee = current_ee

            batched_action = torch.zeros((worlds, 5), dtype=torch.float32, device=backend.device)
            for w in range(worlds):
                batched_action[w] = torch.as_tensor(action, dtype=torch.float32)
            step_active = active.clone()
            low_dim = backend.step(batched_action, step_active)
            low_dim, caught_target, grasp_diagnostics = grasp.update(
                reset, low_dim, step_active
            )
            last_physical_grasp = bool(grasp_diagnostics["physical_grasp"][world].item())
            result = evaluate_active_sparse_tasks(
                state=task_state,
                ee_position=low_dim.ee_position,
                object_positions=low_dim.object_positions,
                gripper_opening=low_dim.gripper_opening,
                caught_target=caught_target,
                active_mask=step_active,
                max_steps=10_000,
                thresholds=thresholds,
                move_to_distance_reward=None,
                catch_release_dense_reward=reward_config,
                bilateral_contact=grasp_diagnostics["bilateral_contact"],
            )
            env_step += 1

            diagnostics = dict(result.diagnostics)
            reference_now = low_dim.object_positions[:, max(reference_slot, 0)]
            hover_point = reference_now.clone()
            hover_point[:, 2] = (
                reference_now[:, 2] + release_height + ctx.grasp_height_offset
            )
            diagnostics["placement_distance"] = torch.linalg.vector_norm(
                low_dim.ee_position - hover_point, dim=-1
            )
            success = bool(result.success[world].item())
            reached_success = reached_success or success
            wrong_place = bool(diagnostics["wrong_place_drop"][world].item())
            grasp_flags = {
                "grasped": float(task_state.grasped[world].item()),
                "bilateral_contact": float(
                    grasp_diagnostics["bilateral_contact"][world].item()
                ),
            }
            terms = _reward_breakdown(
                instruction_type=instruction_type,
                reward_config=reward_config,
                diagnostics=diagnostics,
                grasp_flags=grasp_flags,
                success=success,
                wrong_place=wrong_place,
                torch=torch,
            )
            reward = float(result.rewards[world].item())
            drift = abs(sum(terms.values()) - reward)
            if drift > 2.0e-4:
                raise AssertionError(
                    f"Reward breakdown disagrees with the training reward by {drift:.6f} "
                    f"({instruction_type}, env step {env_step}); the overlay would be wrong."
                )

            controller = backend.controller_state()
            row = {
                "episode": episode,
                "instruction_type": instruction_type,
                "instruction_text": ctx.instruction_text,
                "target_catalog": target_catalog,
                "phase": phase.name,
                "decision": decision,
                "horizon": horizon,
                "env_step": env_step,
                "action": action.tolist(),
                "delta_xyz": (action[:3] * action_step_xyz).tolist(),
                "delta_gripper": float(action[4] * action_step_gripper),
                "ee": np.asarray(low_dim.ee_position[world].tolist(), dtype=float),
                "ee_yaw": float(low_dim.ee_yaw[world].item()),
                "gripper_opening": float(low_dim.gripper_opening[world].item()),
                "commanded_gripper": float(controller["gripper"][world]),
                "target_position": np.asarray(
                    low_dim.object_positions[world][target_slot].tolist(), dtype=float
                ),
                "reference_position": np.asarray(
                    low_dim.object_positions[world][max(reference_slot, 0)].tolist(),
                    dtype=float,
                ),
                "grasp_point": np.asarray(
                    low_dim.object_positions[world][target_slot].tolist(), dtype=float
                )
                + np.array((0.0, 0.0, ctx.grasp_height_offset)),
                "task_distance": float(
                    diagnostics["pick_grasp_distance"][world].item()
                    if instruction_type == "pick_up"
                    else diagnostics["placement_distance"][world].item()
                ),
                "target_lift": float(diagnostics["target_lift"][world].item()),
                "lift_success_height": float(reward_config.pick_lift_success_height),
                "container_xy_error": float(diagnostics["container_xy_error"][world].item()),
                "container_xy_radius": float(diagnostics["container_xy_radius"][world].item()),
                "container_z_error": float(diagnostics["container_z_error"][world].item()),
                "container_z_tolerance": float(reward_config.container_z_tolerance),
                "target_has_settled": bool(diagnostics["target_has_settled"][world].item()),
                "left_contact": bool(grasp_diagnostics["bilateral_contact"][world].item())
                or bool(grasp_diagnostics["left_pad_force_n"][world].item() > 0.0),
                "right_contact": bool(grasp_diagnostics["bilateral_contact"][world].item())
                or bool(grasp_diagnostics["right_pad_force_n"][world].item() > 0.0),
                "left_pad_force_n": float(grasp_diagnostics["left_pad_force_n"][world].item()),
                "right_pad_force_n": float(grasp_diagnostics["right_pad_force_n"][world].item()),
                "relative_position_slip_m": float(
                    grasp_diagnostics["relative_position_slip_m"][world].item()
                ),
                "physical_grasp": bool(grasp_diagnostics["physical_grasp"][world].item()),
                "ever_grasped": bool(task_state.ever_grasped[world].item()),
                "released": bool(diagnostics["released"][world].item()),
                "reward": reward,
                "reward_terms": terms,
                "success": success,
                "terminated": bool(result.terminated[world].item()),
                "backend": backend.metadata()["backend"],
            }
            rows.append(row)

            # Rendering only. Must NOT skip the termination handling below, or a
            # no-video run would never break on terminal and terminated_at would
            # stay unset.
            if render:
                rendered = backend.render_world(world)
                title_over = (
                    f"{CAMERA_LABELS['overview']}  |  oracle reference episode"
                )
                title_wrist = (
                    f"{CAMERA_LABELS['ee_camera']}  |  oracle reference episode"
                )
                frames["overview"].append(
                    _annotate(rendered["overview"], row, title_over)
                )
                frames["ee_camera"].append(
                    _annotate(rendered["ee_camera"], row, title_wrist)
                )
                frames["composite"].append(
                    _annotate(
                        np.concatenate(
                            (rendered["overview"], rendered["ee_camera"]), axis=1
                        ),
                        row,
                        "OVERVIEW  +  WRIST (ee_camera)  |  oracle reference episode",
                    )
                )

            if bool(result.terminated[world].item()):
                if terminated_at is None:
                    terminated_at = env_step
                if not continue_after_terminal:
                    # Stop immediately, mid-decision. Stepping on with the world
                    # masked out would append frames whose reward and grasp
                    # flags are the all-inactive degenerate values, not anything
                    # the trainer would ever score.
                    active = active & ~result.terminated
                    break
        if not bool(active[world].item()):
            break

    summary = {
        "episode": episode,
        "instruction_type": instruction_type,
        "instruction_text": ctx.instruction_text,
        "target_catalog": target_catalog,
        "env_steps": env_step,
        "horizon_decisions": horizon,
        "env_step_budget": horizon * actions_per_decision,
        "terminated_at_env_step": terminated_at,
        "success": reached_success,
        "final_reward": rows[-1]["reward"] if rows else float("nan"),
        "peak_reward": max((row["reward"] for row in rows), default=float("nan")),
        "start": start_snapshot,
        "held_object_reseated": seating,
        "phases_reached": sorted({row["phase"] for row in rows}),
        "peak_pad_force_n": max(
            (max(row["left_pad_force_n"], row["right_pad_force_n"]) for row in rows),
            default=0.0,
        ),
        "ever_physically_grasped": any(row["physical_grasp"] for row in rows),
    }
    return {"summary": summary, "rows": rows, "frames": frames}


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if key == "reward_terms":
            for name, term in value.items():
                flat[f"reward_term_{name}"] = float(term)
        elif isinstance(value, (list, tuple, np.ndarray)):
            values = np.asarray(value, dtype=float).reshape(-1)
            names = ACTION_NAMES if key in {"action"} else ("x", "y", "z")
            for index, item in enumerate(values):
                suffix = names[index] if index < len(names) else str(index)
                flat[f"{key}_{suffix}"] = float(item)
        elif isinstance(value, bool):
            flat[key] = int(value)
        else:
            flat[key] = value
    return flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--instructions",
        nargs="+",
        default=["pick_up"],
        choices=list(SUPPORTED_INSTRUCTIONS),
    )
    parser.add_argument("--episodes-per-instruction", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--terminal-hold-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--start-distance-cap",
        type=float,
        default=None,
        help=(
            "Approach-curriculum cap on the start distance. Defaults to the "
            "config's random_workspace_start_distance_initial, i.e. step 0 of "
            "the run you are about to launch. Pass 0 to disable the cap."
        ),
    )
    parser.add_argument(
        "--continue-after-terminal",
        action="store_true",
        help="Keep stepping (and recording) after the episode terminates.",
    )
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument(
        "--grasp-height-offset",
        type=float,
        default=None,
        help=(
            "Override pick_grasp_height_offset (metres below ee_base that the "
            "reward calls the grasp point). Defaults to the config value. The "
            "measured finger-pad centre is 0.0075 m below ee_base."
        ),
    )
    parser.add_argument(
        "--controller-z-floor",
        type=float,
        default=None,
        help=(
            "Override the controller workspace floor (CDPRBackendConfig."
            "workspace_z low). Defaults to the config's "
            "controller_workspace_z_bounds, i.e. what the run will use, and only "
            "then to the dataclass default."
        ),
    )
    parser.add_argument(
        "--target-catalogs",
        nargs="+",
        default=None,
        help="Restrict the target-object pool (e.g. robocasa_apple).",
    )
    parser.add_argument(
        "--release-opening-threshold",
        type=float,
        default=None,
        help=(
            "Override the per-world release threshold the reset derives as "
            "max(0.55, fitted_gripper_opening + 0.04). Use it to check whether a "
            "placement failure is a real behaviour failure or only that the "
            "threshold sits above the opening at which the object leaves the pads."
        ),
    )
    parser.add_argument(
        "--metadata-override",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override task.metadata entries (numeric or boolean), e.g. "
            "put_plate_release_height=0.10. Applied before the reward and the "
            "resetter are built, so both see the same value the trainer would."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "Skip all rendering and write only telemetry.csv and manifest.json. "
            "This check is about physics -- whether the pads reach the object and "
            "the reward ladder is climbed -- so it must not depend on a working "
            "GL context. Use it when rendering fails or is not needed."
        ),
    )
    parser.add_argument(
        "--reseat-held-object",
        action="store_true",
        help=(
            "For placement tasks, re-seat the carried object at the finger pads "
            "and close until bilateral contact before the episode starts."
        ),
    )
    args = parser.parse_args()

    import torch

    config = _load_config(args.config)
    task = config.get("task", {}) or {}
    metadata = dict(task.get("metadata", {}) or {})
    rl_args = dict(
        ((config.get("training", {}) or {}).get("rl", {}) or {}).get("args", {}) or {}
    )
    simulator = dict(config.get("simulator", {}) or {})

    missing = [name for name in args.instructions if name not in ACTIVE_INSTRUCTION_TYPES]
    if missing:
        raise SystemExit(f"Unknown instruction types: {missing}")
    unsupported = [
        name for name in args.instructions if name not in (task.get("instruction_types") or [])
    ]

    for override in args.metadata_override:
        key, _, raw = str(override).partition("=")
        if not key or not raw:
            raise SystemExit(f"--metadata-override expects KEY=VALUE, got {override!r}")
        lowered = raw.strip().lower()
        if lowered in {"true", "false"}:
            metadata[key] = lowered == "true"
        else:
            metadata[key] = float(raw)
    if args.grasp_height_offset is not None:
        metadata["pick_grasp_height_offset"] = float(args.grasp_height_offset)
    grasp_height_offset = float(
        metadata.get("pick_grasp_height_offset", _MEASURED_PAD_OFFSET_M)
    )
    # Take the floor from the config the run will actually use, falling back to
    # the dataclass default only when the config is silent. Defaulting straight
    # to the dataclass made this script "verify" a combination the trainer would
    # never run: with the config at 0.18 and this at 0.25 the oracle sat pinned
    # at z=0.248 commanding a full -0.45 descent, reported 0/2 and reward 0.87,
    # and none of that said anything about the configuration under test.
    configured_floor = rl_args.get("controller_workspace_z_bounds")
    if args.controller_z_floor is not None:
        z_floor = float(args.controller_z_floor)
    elif configured_floor:
        z_floor = float(min(float(v) for v in configured_floor))
    else:
        z_floor = float(min(CDPRBackendConfig.workspace_z))
    target_objects = list(args.target_catalogs or task.get("target_objects") or [])

    xml_path = Path(
        simulator.get("fixed_scene_xml")
        or "../../robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml"
    )
    if not xml_path.is_absolute():
        xml_path = (args.config.parent / xml_path).resolve()

    # Say up front whether this offset/floor pair can grasp at all. Without this
    # an unreachable configuration just reports "no success" with a plausible
    # partial reward, which reads as a behaviour problem and is not one.
    try:
        from rl_vla_bootstrapping.simulation.cdpr_gripper_geometry import (
            load_cdpr_gripper_geometry,
        )
        from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
            OBJECT_VARIANTS,
        )

        table_z = float(
            ((config.get("simulation", {}) or {}).get("build_kwargs", {}) or {}).get(
                "table_z", 0.15
            )
        )
        geometry = load_cdpr_gripper_geometry(xml_path)
        print(
            f"[geometry] measured pad offset {geometry.grasp_height_offset:.4f} m"
            f" | configured grasp offset {grasp_height_offset:.4f} m"
            f" | controller floor {z_floor:.4f} m"
        )
        unreachable = []
        for name in target_objects:
            variant = OBJECT_VARIANTS.get(name)
            if variant is None or variant.fitted_gripper_opening <= 0.0:
                continue
            centre = table_z + float(variant.rest_height)
            if not geometry.can_reach(centre, controller_floor=z_floor):
                unreachable.append(
                    f"{name} (centre {centre:.3f} m, needs the end-effector at or"
                    f" below {geometry.maximum_ee_height(centre):.3f} m)"
                )
        if unreachable:
            print(
                "[geometry] WARNING: the controller floor is above the grasp "
                "height for:\n  - " + "\n  - ".join(unreachable)
            )
            print(
                "[geometry] No policy can grasp these. Lower the floor via "
                "--controller-z-floor or controller_workspace_z_bounds in the "
                "config before reading anything into the episode results."
            )
    except (ImportError, OSError, ValueError) as exc:
        # A missing model or an unparseable chain is worth noting, not fatal.
        # Anything else (NameError, TypeError) is a bug in this pre-flight and
        # must surface rather than be reported as "skipped".
        print(f"[geometry] skipped reachability pre-flight: {exc}")

    group_size = 2  # the layout requires >= 2; both worlds are identical clones
    backend_config = CDPRBackendConfig(
        backend="mujoco_cpu",
        worlds_per_rank=group_size,
        groups_per_rank=1,
        grpo_group_size=group_size,
        hold_steps=int(rl_args.get("hold_steps", 6)),
        action_step_xyz=float(rl_args.get("action_step_xyz", 0.015)),
        action_step_yaw=float(rl_args.get("action_step_yaw", 0.08)),
        action_step_gripper=float(rl_args.get("action_step_gripper", 0.05)),
        lock_non_commanded_axes=bool(rl_args.get("lock_non_commanded_axes", False)),
        lock_non_commanded_axes_threshold=float(
            rl_args.get("lock_non_commanded_axes_threshold", 0.05)
        ),
        render_width=int(simulator.get("render_width", 320)),
        render_height=int(simulator.get("render_height", 240)),
        device="cpu",
        xml_path=xml_path,
        workspace_z=(z_floor, float(max(CDPRBackendConfig.workspace_z))),
    )
    backend = MujocoReferenceBatchedBackend(config=backend_config, xml_path=xml_path)

    reward_config = BatchedCatchReleaseDenseReward.from_metadata(metadata)
    thresholds = BatchedTaskThresholds(
        container_xy=max(float(reward_config.plate_radius), float(reward_config.bowl_radius)),
        container_z=float(reward_config.container_z_tolerance),
        minimum_target_motion=0.0,
    )
    support_surface_z = float(
        ((config.get("simulation", {}) or {}).get("build_kwargs", {}) or {}).get(
            "table_z", 0.15
        )
    )
    actions_per_decision = int(rl_args.get("replan_every", 4))
    cap = args.start_distance_cap
    if cap is None:
        cap = float(metadata.get("random_workspace_start_distance_initial", 0.03))

    output_root = args.output
    if output_root.exists() and not args.keep_existing:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    grasp = _GraspShim(backend=backend, torch=torch, worlds=group_size)
    manifest: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "xml_path": str(xml_path),
        "renderer_backend": backend.metadata()["backend"],
        "exact_production_backend": False,
        "production_backend": "mjlab_mjwarp",
        "note": (
            "Oracle (scripted) reference episodes. Reset, reward, success "
            "predicate and grasp detection are the production training code; "
            "physics is MuJoCo CPU because MJ-Lab/MJWarp requires CUDA."
        ),
        "start_distance_cap_m": float(cap),
        "pick_grasp_height_offset_m": grasp_height_offset,
        "controller_workspace_z_floor_m": z_floor,
        "measured_finger_pad_offset_below_ee_base_m": 0.0075,
        "held_object_reseated": bool(args.reseat_held_object),
        "target_objects": target_objects,
        "metadata_overrides": list(args.metadata_override),
        "release_opening_threshold_override": args.release_opening_threshold,
        "actions_per_policy_decision": actions_per_decision,
        "physics_substeps_per_env_step": backend_config.physics_substeps,
        "reward_config": {
            key: getattr(reward_config, key) for key in vars(reward_config)
        }
        if hasattr(reward_config, "__dict__")
        else {},
        "instructions_not_in_config": unsupported,
        "episodes": [],
    }

    all_rows: list[dict[str, Any]] = []
    try:
        for instruction_type in args.instructions:
            curriculum = RankLocalCurriculum(device=backend.device)
            layout = RankLocalGroupLayout(
                worlds_per_rank=group_size, groups_per_rank=1, group_size=group_size
            )
            resetter = BatchedReverseFrontierResetter(
                backend=backend,
                layout=layout,
                curriculum=curriculum,
                rank=0,
                base_seed=int(args.seed),
                instruction_types=[instruction_type],
                allowed_objects=target_objects,
                support_surface_z=support_surface_z,
                task_metadata=metadata,
            )
            resetter.set_scene_object_range(
                int(metadata.get("min_scene_objects", 1)),
                int(metadata.get("max_scene_objects", 2)),
            )
            resetter.set_random_start_max_goal_distance(float(cap))

            for episode in range(int(args.episodes_per_instruction)):
                result = run_episode(
                    backend=backend,
                    resetter=resetter,
                    grasp=grasp,
                    reward_config=reward_config,
                    thresholds=thresholds,
                    instruction_type=instruction_type,
                    episode=episode,
                    round_index=episode,
                    actions_per_decision=actions_per_decision,
                    action_step_xyz=backend_config.action_step_xyz,
                    action_step_gripper=backend_config.action_step_gripper,
                    support_surface_z=support_surface_z,
                    continue_after_terminal=bool(args.continue_after_terminal),
                    grasp_height_offset=grasp_height_offset,
                    reseat_held_object=bool(args.reseat_held_object),
                    release_opening_threshold=args.release_opening_threshold,
                    render=not bool(args.no_video),
                    torch=torch,
                )
                summary = result["summary"]
                episode_dir = output_root / instruction_type / f"episode_{episode:02d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                hold = max(0, int(round(float(args.terminal_hold_seconds) * float(args.fps))))
                videos = {}
                for name, frame_list in result["frames"].items():
                    if not frame_list:
                        continue
                    padded = list(frame_list) + [frame_list[-1]] * hold
                    path = episode_dir / f"{instruction_type}_ep{episode:02d}_{name}.mp4"
                    _write_video(padded, path, fps=float(args.fps))
                    videos[name] = str(path.relative_to(output_root))
                summary["videos"] = videos
                manifest["episodes"].append(summary)
                all_rows.extend(_csv_row(row) for row in result["rows"])
                status = "SUCCESS" if summary["success"] else "no success"
                print(
                    f"[{instruction_type}] episode {episode}: {status} in "
                    f"{summary['env_steps']} env steps "
                    f"(budget {summary['env_step_budget']}), "
                    f"final reward {summary['final_reward']:+.3f}, "
                    f"start {summary['start']['ee_to_grasp_point_m']:.3f} m from grasp point"
                )
    finally:
        backend.close()

    if all_rows:
        fieldnames: list[str] = []
        for row in all_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (output_root / "telemetry.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    print(f"\nvideos + telemetry: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
