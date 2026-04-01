#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - optional runtime dependency
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    F = None
    nn = None

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.openvla_actor_critic import (
    OpenVLAActorCriticStack,
    configure_openvla_dimension_env_from_config,
    load_generate_config,
    load_openvla_runtime,
    resolve_llm_dim,
    set_num_images_in_input,
)


@dataclass(frozen=True)
class ObservationLayout:
    state_dim: int
    max_objects: int
    ee_slice: slice
    target_slice: slice
    all_objects_slice: slice
    object_mask_slice: slice
    instruction_slice: slice
    goal_direction_slice: slice

    @classmethod
    def from_observation(cls, obs: dict[str, np.ndarray]) -> "ObservationLayout":
        ee = np.asarray(obs["ee_position"], dtype=np.float32).reshape(-1)
        target = np.asarray(obs["target_object_position"], dtype=np.float32).reshape(-1)
        all_objects = np.asarray(obs["all_object_positions"], dtype=np.float32).reshape(-1)
        object_mask = np.asarray(obs["object_position_mask"], dtype=np.float32).reshape(-1)
        instruction = np.asarray(obs["instruction_onehot"], dtype=np.float32).reshape(-1)
        goal_direction = np.asarray(obs["goal_direction"], dtype=np.float32).reshape(-1)

        offset = 0
        ee_slice = slice(offset, offset + ee.size)
        offset = ee_slice.stop
        target_slice = slice(offset, offset + target.size)
        offset = target_slice.stop
        all_objects_slice = slice(offset, offset + all_objects.size)
        offset = all_objects_slice.stop
        object_mask_slice = slice(offset, offset + object_mask.size)
        offset = object_mask_slice.stop
        instruction_slice = slice(offset, offset + instruction.size)
        offset = instruction_slice.stop
        goal_direction_slice = slice(offset, offset + goal_direction.size)
        offset = goal_direction_slice.stop

        return cls(
            state_dim=offset,
            max_objects=object_mask.size,
            ee_slice=ee_slice,
            target_slice=target_slice,
            all_objects_slice=all_objects_slice,
            object_mask_slice=object_mask_slice,
            instruction_slice=instruction_slice,
            goal_direction_slice=goal_direction_slice,
        )

    def flatten(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(obs["ee_position"], dtype=np.float32).reshape(-1),
                np.asarray(obs["target_object_position"], dtype=np.float32).reshape(-1),
                np.asarray(obs["all_object_positions"], dtype=np.float32).reshape(-1),
                np.asarray(obs["object_position_mask"], dtype=np.float32).reshape(-1),
                np.asarray(obs["instruction_onehot"], dtype=np.float32).reshape(-1),
                np.asarray(obs["goal_direction"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class BarrierConfig:
    x_low: float
    x_high: float
    y_low: float
    y_high: float
    z_low: float
    z_high: float
    eta: float
    beta: float
    object_clearance: float = 0.0


@dataclass
class Transition:
    observation: dict[str, np.ndarray]
    instruction: str
    vector_state: np.ndarray
    action_chunk: np.ndarray
    reward: float
    safety_cost: float
    stability_cost: float
    next_observation: dict[str, np.ndarray]
    next_instruction: str
    next_vector_state: np.ndarray
    done: bool


class TransitionReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.items: list[Transition] = []
        self.ptr = 0

    def add(self, transition: Transition) -> None:
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.capacity

    @property
    def size(self) -> int:
        return len(self.items)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.items, k=min(int(batch_size), len(self.items)))


if torch is not None:
    class DynamicsModel(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )

        def forward(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
            flat_action = action_chunk.reshape(action_chunk.shape[0], -1)
            return state + self.net(torch.cat([state, flat_action], dim=-1))


    class LyapunovNetwork(nn.Module):
        def __init__(self, state_dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return F.softplus(self.net(state))

else:
    class DynamicsModel:  # pragma: no cover - runtime dependency gate
        def __init__(self, *args, **kwargs):
            raise RuntimeError("DynamicsModel requires torch.")


    class LyapunovNetwork:  # pragma: no cover - runtime dependency gate
        def __init__(self, *args, **kwargs):
            raise RuntimeError("LyapunovNetwork requires torch.")


def _soft_update(source: Any, target: Any, tau: float) -> None:
    for src, dst in zip(source.parameters(), target.parameters()):
        dst.data.mul_(1.0 - tau).add_(tau * src.data)


def _goal_distance_sq_np(state: np.ndarray, layout: ObservationLayout) -> float:
    delta = np.asarray(state[layout.ee_slice] - state[layout.target_slice], dtype=np.float32)
    return float(np.dot(delta, delta))


def _workspace_barriers(state: torch.Tensor, layout: ObservationLayout, cfg: BarrierConfig) -> dict[str, torch.Tensor]:
    ee = state[..., layout.ee_slice]
    out = {
        "x_min": ee[..., 0:1] - cfg.x_low,
        "x_max": cfg.x_high - ee[..., 0:1],
        "y_min": ee[..., 1:2] - cfg.y_low,
        "y_max": cfg.y_high - ee[..., 1:2],
        "z_min": ee[..., 2:3] - cfg.z_low,
        "z_max": cfg.z_high - ee[..., 2:3],
    }
    if cfg.object_clearance > 0.0:
        flat = state[..., layout.all_objects_slice]
        objects = flat.view(*state.shape[:-1], layout.max_objects, 3)
        mask = state[..., layout.object_mask_slice].unsqueeze(-1)
        distances = torch.linalg.norm(objects - ee.unsqueeze(-2), dim=-1, keepdim=True)
        masked = torch.where(mask > 0.5, distances, torch.full_like(distances, float("inf")))
        out["object_clearance"] = masked.min(dim=-2).values - cfg.object_clearance
    return out


def _barrier_violations(
    state: torch.Tensor,
    predicted_next_state: torch.Tensor,
    layout: ObservationLayout,
    cfg: BarrierConfig,
) -> dict[str, torch.Tensor]:
    current = _workspace_barriers(state, layout, cfg)
    nxt = _workspace_barriers(predicted_next_state, layout, cfg)
    return {name: F.relu(current[name] - nxt[name] - cfg.eta * current[name]) for name in current}


def _safety_cost_np(state: np.ndarray, layout: ObservationLayout, cfg: BarrierConfig) -> float:
    ee = np.asarray(state[layout.ee_slice], dtype=np.float32)
    total = 0.0
    total += max(0.0, cfg.x_low - float(ee[0]))
    total += max(0.0, float(ee[0]) - cfg.x_high)
    total += max(0.0, cfg.y_low - float(ee[1]))
    total += max(0.0, float(ee[1]) - cfg.y_high)
    total += max(0.0, cfg.z_low - float(ee[2]))
    total += max(0.0, float(ee[2]) - cfg.z_high)
    if cfg.object_clearance > 0.0:
        objects = np.asarray(state[layout.all_objects_slice], dtype=np.float32).reshape(layout.max_objects, 3)
        mask = np.asarray(state[layout.object_mask_slice], dtype=np.float32)
        valid = objects[mask > 0.5]
        if valid.size > 0:
            distances = np.linalg.norm(valid - ee[None, :], axis=1)
            total += max(0.0, cfg.object_clearance - float(np.min(distances)))
    return float(total)


def _should_use_backup(state: np.ndarray, action: np.ndarray, layout: ObservationLayout, cfg: BarrierConfig, step_xyz: float) -> bool:
    ee = np.asarray(state[layout.ee_slice], dtype=np.float32)
    proposed = ee + np.asarray(action[:3], dtype=np.float32) * float(step_xyz)
    return bool(
        proposed[0] < cfg.x_low
        or proposed[0] > cfg.x_high
        or proposed[1] < cfg.y_low
        or proposed[1] > cfg.y_high
        or proposed[2] < cfg.z_low
        or proposed[2] > cfg.z_high
    )


def _backup_project_action(
    state: np.ndarray,
    action: np.ndarray,
    layout: ObservationLayout,
    cfg: BarrierConfig,
    step_xyz: float,
    goal_gain: float,
) -> np.ndarray:
    ee = np.asarray(state[layout.ee_slice], dtype=np.float32)
    goal = np.asarray(state[layout.target_slice], dtype=np.float32)
    proposed = ee + np.asarray(action[:3], dtype=np.float32) * float(step_xyz)
    clipped = proposed.copy()
    clipped[0] = float(np.clip(clipped[0], cfg.x_low, cfg.x_high))
    clipped[1] = float(np.clip(clipped[1], cfg.y_low, cfg.y_high))
    clipped[2] = float(np.clip(clipped[2], cfg.z_low, cfg.z_high))
    goal_dir = goal - ee
    norm = float(np.linalg.norm(goal_dir))
    if norm > 1e-6:
        goal_dir = goal_dir / norm
    mixed = clipped + float(goal_gain) * goal_dir * float(step_xyz)
    mixed[0] = float(np.clip(mixed[0], cfg.x_low, cfg.x_high))
    mixed[1] = float(np.clip(mixed[1], cfg.y_low, cfg.y_high))
    mixed[2] = float(np.clip(mixed[2], cfg.z_low, cfg.z_high))
    safe = np.asarray(action, dtype=np.float32).copy()
    safe[:3] = np.clip((mixed - ee) / max(float(step_xyz), 1e-6), -1.0, 1.0)
    return safe


def _bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help_text: str) -> None:
    parser.add_argument(f"--{name}", default=default, action=argparse.BooleanOptionalAction, help=help_text)


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="openvla_blac_finetune_cdpr",
        description="In-tree OpenVLA BLAC-style finetuning for the CDPR env using the local actor-critic stack.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--action-head-path", required=True)
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_root_dir", default="runs")
    parser.add_argument("--run_id", default="openvla_blac")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--save_every_episodes", type=int, default=5)
    parser.add_argument("--replay_size", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--update_after", type=int, default=100)
    parser.add_argument("--updates_per_episode", type=int, default=20)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gamma_c", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--dynamics_lr", type=float, default=3e-4)
    parser.add_argument("--lyapunov_lr", type=float, default=3e-4)
    parser.add_argument("--constraint_lr", type=float, default=1e-2)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--critic_hidden_dim", type=int, default=1024)
    parser.add_argument("--exploration_noise", type=float, default=0.10)
    parser.add_argument("--target_policy_noise", type=float, default=0.10)
    parser.add_argument("--target_noise_clip", type=float, default=0.25)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--rho_init", type=float, default=1.0)
    parser.add_argument("--rho_growth", type=float, default=1.01)
    parser.add_argument("--rho_max", type=float, default=100.0)
    parser.add_argument("--object_clearance", type=float, default=0.0)
    _bool_arg(parser, "enable_backup_controller", default=True, help_text="Project unsafe chunk steps back into workspace.")
    parser.add_argument("--backup_goal_gain", type=float, default=0.5)
    _bool_arg(parser, "freeze_vla_backbone", default=True, help_text="Freeze VLA/LoRA weights and train only the action head + critics.")
    parser.add_argument("--chunk_length", type=int, default=None)
    _bool_arg(parser, "center_crop", default=True, help_text="Forwarded to the OpenVLA processor config.")

    parser.add_argument("--catalog_path", default=None)
    parser.add_argument("--max_env_steps", type=int, default=120)
    parser.add_argument("--max_objects", type=int, default=8)
    parser.add_argument("--action_step_xyz", type=float, default=0.02)
    parser.add_argument("--action_step_yaw", type=float, default=0.25)
    parser.add_argument("--hold_steps", type=int, default=0)
    _bool_arg(parser, "lock_non_commanded_axes", default=False, help_text="Forwarded to the CDPR env.")
    parser.add_argument("--lock_non_commanded_axes_threshold", type=float, default=0.05)
    _bool_arg(parser, "randomize_ee_start", default=False, help_text="Forwarded to the CDPR env.")
    parser.add_argument("--ee_start_x_bounds", type=float, nargs=2, default=(-0.25, 0.25))
    parser.add_argument("--ee_start_y_bounds", type=float, nargs=2, default=(-0.25, 0.25))
    parser.add_argument("--ee_start_z", type=float, default=None)
    parser.add_argument("--move_distance", type=float, default=0.40)
    parser.add_argument("--lift_distance", type=float, default=0.10)
    _bool_arg(parser, "capture_frames", default=False, help_text="Forwarded to the CDPR env.")
    parser.add_argument("--instruction_types", nargs="*", default=None)
    parser.add_argument("--allowed_objects", nargs="*", default=None)
    parser.add_argument("--desk_textures_dir", default=None)
    parser.add_argument("--desk_geom_regex", default=r"(table|desk|workbench|counter|surface)")
    parser.add_argument("--desk_texrepeat", type=int, nargs=2, default=(20, 20))
    _bool_arg(parser, "wrapper_cleanup", default=False, help_text="Forwarded to the CDPR env.")
    _bool_arg(parser, "use_wrapper_cache", default=True, help_text="Forwarded to the CDPR env.")

    parser.add_argument("--workspace_x_bounds", type=float, nargs=2, default=(-0.90, 0.90))
    parser.add_argument("--workspace_y_bounds", type=float, nargs=2, default=(-0.90, 0.90))
    parser.add_argument("--workspace_z_bounds", type=float, nargs=2, default=(0.05, 0.60))
    parser.add_argument("--num_images_in_input", type=int, default=None)

    return parser.parse_known_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _make_run_dir(root: str | Path, run_id: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(root).expanduser().resolve() / f"{run_id}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_env(args: argparse.Namespace):
    from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRVisionLanguageEnv

    return CDPRVisionLanguageEnv(
        catalog_path=args.catalog_path,
        max_steps=args.max_env_steps,
        max_objects=args.max_objects,
        action_step_xyz=args.action_step_xyz,
        action_step_yaw=args.action_step_yaw,
        hold_steps=args.hold_steps,
        lock_non_commanded_axes=args.lock_non_commanded_axes,
        lock_non_commanded_axes_threshold=args.lock_non_commanded_axes_threshold,
        randomize_ee_start=args.randomize_ee_start,
        ee_start_x_bounds=args.ee_start_x_bounds,
        ee_start_y_bounds=args.ee_start_y_bounds,
        ee_start_z=args.ee_start_z,
        move_distance=args.move_distance,
        lift_distance=args.lift_distance,
        capture_frames=args.capture_frames,
        instruction_types=args.instruction_types,
        allowed_objects=args.allowed_objects,
        desk_textures_dir=args.desk_textures_dir,
        desk_geom_regex=args.desk_geom_regex,
        desk_texrepeat=args.desk_texrepeat,
        wrapper_cleanup=args.wrapper_cleanup,
        use_wrapper_cache=args.use_wrapper_cache,
        seed=args.seed,
    )


def _make_vla_observation(env: Any, instruction: str) -> dict[str, np.ndarray]:
    sim = env.sim
    full_rgb = sim.capture_frame(sim.overview_cam, "overview")
    wrist_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")
    return {
        "full_image": np.ascontiguousarray(full_rgb),
        "wrist_image": np.ascontiguousarray(wrist_rgb),
        "state": np.zeros((5,), dtype=np.float32),
        "task_description": str(instruction),
    }


def _barrier_config_from_args(args: argparse.Namespace) -> BarrierConfig:
    return BarrierConfig(
        x_low=float(args.workspace_x_bounds[0]),
        x_high=float(args.workspace_x_bounds[1]),
        y_low=float(args.workspace_y_bounds[0]),
        y_high=float(args.workspace_y_bounds[1]),
        z_low=float(args.workspace_z_bounds[0]),
        z_high=float(args.workspace_z_bounds[1]),
        eta=float(args.eta),
        beta=float(args.beta),
        object_clearance=max(0.0, float(args.object_clearance)),
    )


def _stack_pixel_context(stack: OpenVLAActorCriticStack) -> tuple[Any, Any]:
    param = next(stack.action_head.parameters())
    return param.device, param.dtype


def _load_stack(config: Any, args: argparse.Namespace) -> tuple[OpenVLAActorCriticStack, Any]:
    policy_repo = config.resolve_path(config.policy.repo_path)
    if policy_repo is None:
        raise RuntimeError("Config is missing `policy.repo_path`.")

    configure_openvla_dimension_env_from_config(
        config,
        chunk_length=int(args.chunk_length or config.policy.action_codec.chunk_size),
    )
    get_action_head, get_processor, _get_proprio_projector, get_vla, PeftModel, _repo = load_openvla_runtime(policy_repo)
    GenerateConfig, note = load_generate_config()
    if note:
        print(f"[info] {note}", flush=True)

    chunk_length = int(args.chunk_length or config.policy.action_codec.chunk_size)
    requested_num_images = int(args.num_images_in_input or config.policy.num_images_in_input)
    cfg = GenerateConfig(
        pretrained_checkpoint=args.base_ckpt or config.policy.base_checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=requested_num_images,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=bool(args.center_crop),
        num_open_loop_steps=chunk_length,
        unnorm_key=None,
    )
    cfg.cdpr_action_head_path = str(Path(args.action_head_path).expanduser().resolve())

    vla_base = get_vla(cfg)
    vla_base.eval()
    adapter_path = Path(args.adapter_path).expanduser().resolve()
    if not adapter_path.is_dir():
        raise RuntimeError(f"Adapter path is not a directory: {adapter_path}")
    vla = PeftModel.from_pretrained(vla_base, str(adapter_path))
    vla.eval()

    cfg.num_images_in_input = set_num_images_in_input(vla, int(cfg.num_images_in_input))
    llm_dim = resolve_llm_dim(vla)
    if llm_dim is None:
        raise RuntimeError("Could not resolve llm_dim from the wrapped OpenVLA model.")

    processor = get_processor(cfg)
    param = next(vla.parameters())
    action_head = get_action_head(cfg, llm_dim=llm_dim).to(device=param.device, dtype=param.dtype)
    action_head.train()

    stack = OpenVLAActorCriticStack(
        vla=vla,
        processor=processor,
        action_head=action_head,
        chunk_length=chunk_length,
        action_dim=int(getattr(action_head, "action_dim", 5)),
        num_images_in_input=int(cfg.num_images_in_input),
        llm_dim=int(llm_dim),
        critic_hidden_dim=int(args.critic_hidden_dim),
        freeze_vla_backbone=bool(args.freeze_vla_backbone),
    ).to(device=param.device, dtype=param.dtype)
    stack.train()
    return stack, cfg


def _copy_target_action_head(action_head: nn.Module) -> nn.Module:
    target = copy.deepcopy(action_head)
    target.eval()
    return target


def _chunk_action_dim(stack: OpenVLAActorCriticStack) -> int:
    return int(stack.chunk_length) * int(stack.action_dim)


def _rollout_chunk(
    *,
    env: Any,
    vector_obs: dict[str, np.ndarray],
    instruction: str,
    action_chunk: np.ndarray,
    layout: ObservationLayout,
    barrier_cfg: BarrierConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str, float, float, float, bool]:
    current_vec = {key: np.asarray(value, dtype=np.float32).copy() for key, value in vector_obs.items()}
    reward_total = 0.0
    safety_total = 0.0
    stability_total = 0.0
    done = False
    info: dict[str, Any] = {"language_instruction": instruction}

    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    chunk = chunk[:, :5]

    for step_action in chunk:
        action = np.asarray(step_action, dtype=np.float32).reshape(5)
        state_flat = layout.flatten(current_vec)
        if args.enable_backup_controller and _should_use_backup(
            state_flat,
            action,
            layout,
            barrier_cfg,
            float(args.action_step_xyz),
        ):
            action = _backup_project_action(
                state_flat,
                action,
                layout,
                barrier_cfg,
                float(args.action_step_xyz),
                float(args.backup_goal_gain),
            )

        next_vec, reward, terminated, truncated, info = env.step(action)
        current_vec = {key: np.asarray(value, dtype=np.float32).copy() for key, value in next_vec.items()}
        flat_next = layout.flatten(current_vec)
        reward_total += float(reward)
        safety_total += _safety_cost_np(flat_next, layout, barrier_cfg)
        stability_total += _goal_distance_sq_np(flat_next, layout)
        done = bool(terminated or truncated)
        if done:
            break

    next_instruction = str(info.get("language_instruction", instruction))
    next_vla_obs = _make_vla_observation(env, next_instruction)
    return current_vec, next_vla_obs, next_instruction, reward_total, safety_total, stability_total, done


def _sample_batch(batch: list[Transition], device: Any, dtype: Any) -> dict[str, Any]:
    return {
        "observations": [item.observation for item in batch],
        "instructions": [item.instruction for item in batch],
        "states": torch.as_tensor(np.stack([item.vector_state for item in batch], axis=0), device=device, dtype=torch.float32),
        "actions": torch.as_tensor(np.stack([item.action_chunk for item in batch], axis=0), device=device, dtype=dtype),
        "rewards": torch.as_tensor(np.asarray([item.reward for item in batch], dtype=np.float32).reshape(-1, 1), device=device),
        "safety_costs": torch.as_tensor(
            np.asarray([item.safety_cost for item in batch], dtype=np.float32).reshape(-1, 1),
            device=device,
        ),
        "stability_costs": torch.as_tensor(
            np.asarray([item.stability_cost for item in batch], dtype=np.float32).reshape(-1, 1),
            device=device,
        ),
        "next_observations": [item.next_observation for item in batch],
        "next_instructions": [item.next_instruction for item in batch],
        "next_states": torch.as_tensor(
            np.stack([item.next_vector_state for item in batch], axis=0),
            device=device,
            dtype=torch.float32,
        ),
        "dones": torch.as_tensor(np.asarray([item.done for item in batch], dtype=np.float32).reshape(-1, 1), device=device),
    }


def _save_checkpoint(
    *,
    run_dir: Path,
    episode: int,
    stack: OpenVLAActorCriticStack,
    target_action_head: nn.Module,
    dynamics: DynamicsModel,
    lyapunov: LyapunovNetwork,
    target_critic1: nn.Module,
    target_critic2: nn.Module,
    barrier_cfg: BarrierConfig,
) -> Path:
    path = run_dir / f"checkpoint_ep{episode:04d}.pt"
    payload = {
        "episode": int(episode),
        "action_head": stack.action_head.state_dict(),
        "critic1": stack.critic1.state_dict(),
        "critic2": stack.critic2.state_dict(),
        "target_action_head": target_action_head.state_dict(),
        "target_critic1": target_critic1.state_dict(),
        "target_critic2": target_critic2.state_dict(),
        "dynamics": dynamics.state_dict(),
        "lyapunov": lyapunov.state_dict(),
        "barrier_config": asdict(barrier_cfg),
    }
    torch.save(payload, path)
    torch.save(payload, run_dir / "latest.pt")
    torch.save(stack.action_head.state_dict(), run_dir / "action_head_latest.pt")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    if torch is None:
        raise RuntimeError("openvla_blac_finetune_cdpr requires torch and the OpenVLA runtime dependencies.")

    args, unknown = parse_args(argv)
    if unknown:
        print("[openvla-blac] Ignoring unrecognized stage arguments: " + " ".join(str(x) for x in unknown), flush=True)

    _set_seed(int(args.seed))
    config = load_project_config(args.config)
    run_dir = _make_run_dir(args.run_root_dir, args.run_id)
    _write_json(run_dir / "config.json", vars(args))

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard")) if SummaryWriter is not None else None
    barrier_cfg = _barrier_config_from_args(args)
    env = None
    try:
        env = _build_env(args)
        vector_obs, info = env.reset(seed=int(args.seed))
        layout = ObservationLayout.from_observation(vector_obs)
        instruction = str(info.get("language_instruction", "pick up object"))
        vla_obs = _make_vla_observation(env, instruction)

        stack, _cfg = _load_stack(config, args)
        device, pixel_dtype = _stack_pixel_context(stack)

        target_action_head = _copy_target_action_head(stack.action_head).to(device=device, dtype=pixel_dtype)
        target_critic1 = copy.deepcopy(stack.critic1).to(device=device, dtype=pixel_dtype)
        target_critic2 = copy.deepcopy(stack.critic2).to(device=device, dtype=pixel_dtype)
        dynamics = DynamicsModel(layout.state_dim, _chunk_action_dim(stack), int(args.hidden_dim)).to(device=device)
        lyapunov = LyapunovNetwork(layout.state_dim, int(args.hidden_dim)).to(device=device)
        lyapunov_target = copy.deepcopy(lyapunov).to(device=device)

        actor_optim = torch.optim.Adam(stack.action_head.parameters(), lr=float(args.actor_lr))
        critic_optim = torch.optim.Adam(
            list(stack.critic1.parameters()) + list(stack.critic2.parameters()),
            lr=float(args.critic_lr),
        )
        dynamics_optim = torch.optim.Adam(dynamics.parameters(), lr=float(args.dynamics_lr))
        lyapunov_optim = torch.optim.Adam(lyapunov.parameters(), lr=float(args.lyapunov_lr))

        lambda_values = {name: 0.0 for name in _workspace_barriers(torch.zeros((1, layout.state_dim), device=device), layout, barrier_cfg)}
        rho_values = {name: float(args.rho_init) for name in lambda_values}
        zeta = 0.0
        rho_zeta = float(args.rho_init)
        replay = TransitionReplayBuffer(int(args.replay_size))
        metrics_path = run_dir / "metrics.jsonl"
        gradient_step = 0

        for episode in range(1, int(args.max_episodes) + 1):
            if episode > 1:
                vector_obs, info = env.reset()
                instruction = str(info.get("language_instruction", instruction))
                vla_obs = _make_vla_observation(env, instruction)

            episode_reward = 0.0
            episode_safety = 0.0
            episode_steps = 0
            done = False

            while not done and episode_steps < int(args.max_env_steps):
                with torch.no_grad():
                    action_chunk, _, _ = stack.actor(
                        observations=[vla_obs],
                        instructions=[instruction],
                        device=device,
                        pixel_dtype=pixel_dtype,
                    )
                action_chunk_np = action_chunk[0].detach().to(dtype=torch.float32).cpu().numpy()
                action_chunk_np = np.asarray(action_chunk_np[:, :5], dtype=np.float32)
                if float(args.exploration_noise) > 0.0:
                    noise = np.random.normal(
                        0.0,
                        float(args.exploration_noise),
                        size=action_chunk_np.shape,
                    ).astype(np.float32)
                    action_chunk_np = np.clip(action_chunk_np + noise, -1.0, 1.0)

                vector_state = layout.flatten(vector_obs)
                next_vector_obs, next_vla_obs, next_instruction, reward, safety_cost, stability_cost, done = _rollout_chunk(
                    env=env,
                    vector_obs=vector_obs,
                    instruction=instruction,
                    action_chunk=action_chunk_np,
                    layout=layout,
                    barrier_cfg=barrier_cfg,
                    args=args,
                )
                next_state = layout.flatten(next_vector_obs)
                replay.add(
                    Transition(
                        observation=vla_obs,
                        instruction=instruction,
                        vector_state=vector_state,
                        action_chunk=action_chunk_np,
                        reward=float(reward),
                        safety_cost=float(safety_cost),
                        stability_cost=float(stability_cost),
                        next_observation=next_vla_obs,
                        next_instruction=next_instruction,
                        next_vector_state=next_state,
                        done=bool(done),
                    )
                )

                vector_obs = next_vector_obs
                vla_obs = next_vla_obs
                instruction = next_instruction
                episode_reward += float(reward)
                episode_safety += float(safety_cost)
                episode_steps += int(action_chunk_np.shape[0])

            episode_metrics: dict[str, float] = {}
            if replay.size >= int(args.update_after):
                for _ in range(int(args.updates_per_episode)):
                    batch_items = replay.sample(int(args.batch_size))
                    batch = _sample_batch(batch_items, device=device, dtype=pixel_dtype)

                    with torch.no_grad():
                        next_hidden, next_pooled = stack.encode(
                            observations=batch["next_observations"],
                            instructions=batch["next_instructions"],
                            device=device,
                            pixel_dtype=pixel_dtype,
                        )
                        next_actions = torch.tanh(target_action_head.predict_action(next_hidden))[:, :, :5]
                        if float(args.target_policy_noise) > 0.0:
                            noise = torch.randn_like(next_actions) * float(args.target_policy_noise)
                            noise = noise.clamp(-float(args.target_noise_clip), float(args.target_noise_clip))
                            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
                        target_q1 = target_critic1(next_pooled, next_actions)
                        target_q2 = target_critic2(next_pooled, next_actions)
                        target_q = torch.minimum(target_q1, target_q2)
                        q_target = batch["rewards"] + (1.0 - batch["dones"]) * float(args.gamma) * target_q

                    hidden, pooled = stack.encode(
                        observations=batch["observations"],
                        instructions=batch["instructions"],
                        device=device,
                        pixel_dtype=pixel_dtype,
                    )
                    q1 = stack.critic1(pooled, batch["actions"])
                    q2 = stack.critic2(pooled, batch["actions"])
                    critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
                    critic_optim.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    critic_optim.step()

                    pred_next_state = dynamics(batch["states"], batch["actions"])
                    dynamics_loss = F.mse_loss(pred_next_state, batch["next_states"])
                    dynamics_optim.zero_grad(set_to_none=True)
                    dynamics_loss.backward()
                    dynamics_optim.step()

                    with torch.no_grad():
                        lyapunov_target_value = batch["stability_costs"] + (1.0 - batch["dones"]) * float(args.gamma_c) * lyapunov_target(batch["next_states"])
                    lyapunov_pred = lyapunov(batch["states"])
                    lyapunov_loss = F.mse_loss(lyapunov_pred, lyapunov_target_value)
                    lyapunov_optim.zero_grad(set_to_none=True)
                    lyapunov_loss.backward()
                    lyapunov_optim.step()

                    if gradient_step % int(args.policy_delay) == 0:
                        hidden, pooled = stack.encode(
                            observations=batch["observations"],
                            instructions=batch["instructions"],
                            device=device,
                            pixel_dtype=pixel_dtype,
                        )
                        policy_actions = torch.tanh(stack.action_head.predict_action(hidden))[:, :, :5]
                        q_actor = torch.minimum(
                            stack.critic1(pooled, policy_actions),
                            stack.critic2(pooled, policy_actions),
                        )
                        predicted_policy_next = dynamics(batch["states"], policy_actions)
                        barrier_terms = _barrier_violations(batch["states"], predicted_policy_next, layout, barrier_cfg)
                        penalty = torch.zeros((1,), device=device, dtype=torch.float32)
                        for name, violation in barrier_terms.items():
                            mean_violation = violation.mean()
                            penalty = penalty + lambda_values[name] * mean_violation + 0.5 * rho_values[name] * mean_violation.pow(2)

                        current_l = lyapunov(batch["states"])
                        next_l = lyapunov(predicted_policy_next)
                        clf_violation = F.relu(next_l - current_l + float(args.beta) * current_l)
                        penalty = penalty + zeta * clf_violation.mean() + 0.5 * rho_zeta * clf_violation.mean().pow(2)

                        actor_loss = -q_actor.mean() + penalty
                        actor_optim.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        actor_optim.step()

                        with torch.no_grad():
                            for name, violation in barrier_terms.items():
                                mean_violation = float(violation.mean().item())
                                lambda_values[name] = max(0.0, lambda_values[name] + float(args.constraint_lr) * mean_violation)
                                rho_values[name] = min(float(args.rho_max), rho_values[name] * float(args.rho_growth))
                            clf_mean = float(clf_violation.mean().item())
                            zeta = max(0.0, zeta + float(args.constraint_lr) * clf_mean)
                            rho_zeta = min(float(args.rho_max), rho_zeta * float(args.rho_growth))

                        _soft_update(stack.action_head, target_action_head, float(args.tau))
                        _soft_update(stack.critic1, target_critic1, float(args.tau))
                        _soft_update(stack.critic2, target_critic2, float(args.tau))
                        _soft_update(lyapunov, lyapunov_target, float(args.tau))
                        episode_metrics["actor_loss"] = float(actor_loss.item())
                        episode_metrics["constraint_clf"] = float(clf_violation.mean().item())

                    gradient_step += 1
                    episode_metrics.update(
                        {
                            "critic_loss": float(critic_loss.item()),
                            "dynamics_loss": float(dynamics_loss.item()),
                            "lyapunov_loss": float(lyapunov_loss.item()),
                            "q_mean": float(torch.minimum(q1, q2).mean().item()),
                        }
                    )
                    for name, value in lambda_values.items():
                        episode_metrics[f"lambda/{name}"] = float(value)
                    episode_metrics["lambda/clf"] = float(zeta)

            summary = {
                "episode": int(episode),
                "reward": float(episode_reward),
                "safety_cost": float(episode_safety),
                "env_steps": int(episode_steps),
                **episode_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, sort_keys=True) + "\n")

            print(
                f"[openvla-blac] episode={episode:04d} env_steps={episode_steps:04d} "
                f"reward={episode_reward:8.3f} safety_cost={episode_safety:8.3f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("episode/reward", float(episode_reward), int(episode))
                writer.add_scalar("episode/safety_cost", float(episode_safety), int(episode))
                writer.add_scalar("episode/env_steps", float(episode_steps), int(episode))
                for key, value in episode_metrics.items():
                    writer.add_scalar(key, float(value), int(episode))
                writer.flush()

            if episode % int(args.save_every_episodes) == 0 or episode == int(args.max_episodes):
                path = _save_checkpoint(
                    run_dir=run_dir,
                    episode=episode,
                    stack=stack,
                    target_action_head=target_action_head,
                    dynamics=dynamics,
                    lyapunov=lyapunov,
                    target_critic1=target_critic1,
                    target_critic2=target_critic2,
                    barrier_cfg=barrier_cfg,
                )
                print(f"[openvla-blac] Saved checkpoint: {path}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
