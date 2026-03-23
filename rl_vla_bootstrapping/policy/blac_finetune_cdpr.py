#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


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
        parts = [
            np.asarray(obs["ee_position"], dtype=np.float32).reshape(-1),
            np.asarray(obs["target_object_position"], dtype=np.float32).reshape(-1),
            np.asarray(obs["all_object_positions"], dtype=np.float32).reshape(-1),
            np.asarray(obs["object_position_mask"], dtype=np.float32).reshape(-1),
            np.asarray(obs["instruction_onehot"], dtype=np.float32).reshape(-1),
            np.asarray(obs["goal_direction"], dtype=np.float32).reshape(-1),
        ]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def ee_position(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., self.ee_slice]

    def target_position(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., self.target_slice]

    def object_positions(self, state: torch.Tensor) -> torch.Tensor:
        flat = state[..., self.all_objects_slice]
        return flat.view(*state.shape[:-1], self.max_objects, 3)

    def object_mask(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., self.object_mask_slice]


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


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.cost = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        cost: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr, 0] = float(reward)
        self.cost[self.ptr, 0] = float(cost)
        self.next_state[self.ptr] = next_state
        self.done[self.ptr, 0] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "state": torch.as_tensor(self.state[idx], device=device),
            "action": torch.as_tensor(self.action[idx], device=device),
            "reward": torch.as_tensor(self.reward[idx], device=device),
            "cost": torch.as_tensor(self.cost[idx], device=device),
            "next_state": torch.as_tensor(self.next_state[idx], device=device),
            "done": torch.as_tensor(self.done[idx], device=device),
        }


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], *, activate_last: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last or activate_last:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = MLP((state_dim, hidden_dim, hidden_dim))
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

    def act(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        mean, _ = self(state)
        if deterministic:
            return torch.tanh(mean)
        action, _, _ = self.sample(state)
        return action


class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP((state_dim, hidden_dim, hidden_dim, action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state))

    def act(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self(state)


class QEnsemble(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_q: int):
        super().__init__()
        self.models = nn.ModuleList(
            MLP((state_dim + action_dim, hidden_dim, hidden_dim, 1)) for _ in range(int(num_q))
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> list[torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return [model(x) for model in self.models]


class LyapunovNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP((state_dim, hidden_dim, hidden_dim, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(state))


class DynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP((state_dim + action_dim, hidden_dim, hidden_dim, state_dim))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([state, action], dim=-1))
        return state + delta


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for src, dst in zip(source.parameters(), target.parameters()):
        dst.data.mul_(1.0 - tau).add_(tau * src.data)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _algorithm_choices() -> tuple[str, ...]:
    return ("blac", "bac", "sac", "td3", "redq")


def _bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    parser.add_argument(
        f"--{name}",
        default=default,
        action=argparse.BooleanOptionalAction,
        help=help_text,
    )


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="blac_finetune_cdpr",
        description=(
            "Train a vector-state CDPR policy with BLAC-style barrier/Lyapunov constraints. "
            "Also supports BAC, SAC, TD3, and REDQ baselines."
        ),
    )
    parser.add_argument("--algorithm", choices=_algorithm_choices(), default="blac")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_root_dir", default="runs")
    parser.add_argument("--run_id", default="rl_safe_ac")
    parser.add_argument("--save_every_episodes", type=int, default=10)
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--max_env_steps", type=int, default=120)
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--replay_size", type=int, default=200_000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gamma_c", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--lyapunov_lr", type=float, default=3e-4)
    parser.add_argument("--dynamics_lr", type=float, default=3e-4)
    parser.add_argument("--constraint_lr", type=float, default=1e-2)
    parser.add_argument("--target_entropy_scale", type=float, default=1.0)
    _bool_arg(parser, "autotune_alpha", default=True, help_text="Learn SAC/REDQ entropy temperature.")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--target_policy_noise", type=float, default=0.2)
    parser.add_argument("--target_noise_clip", type=float, default=0.5)
    parser.add_argument("--exploration_noise", type=float, default=0.1)
    parser.add_argument("--num_q", type=int, default=10, help="REDQ ensemble size; ignored for other algorithms.")
    parser.add_argument("--redq_target_subset", type=int, default=2)
    parser.add_argument("--eta", type=float, default=0.25, help="CBF contraction factor from the BLAC paper.")
    parser.add_argument("--beta", type=float, default=0.10, help="CLF decay factor from the BLAC paper.")
    parser.add_argument("--rho_init", type=float, default=1.0)
    parser.add_argument("--rho_growth", type=float, default=1.01)
    parser.add_argument("--rho_max", type=float, default=100.0)
    parser.add_argument("--object_clearance", type=float, default=0.0)
    _bool_arg(
        parser,
        "enable_backup_controller",
        default=True,
        help_text="Enable a lightweight action projection fallback inspired by BLAC Eq. (21).",
    )
    parser.add_argument("--backup_goal_gain", type=float, default=0.5)

    parser.add_argument("--catalog_path", default=None)
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

    parser.add_argument("--vla_path", default=None, help="Accepted for stage compatibility; unused by this trainer.")
    parser.add_argument(
        "--num_images_in_input",
        type=int,
        default=2,
        help="Accepted for stage compatibility; unused by this trainer.",
    )

    args, unknown = parser.parse_known_args(argv)
    return args, unknown


def _make_run_dir(root: str | Path, run_id: str, algorithm: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(root).expanduser().resolve() / f"{run_id}_{algorithm}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _goal_distance_sq(state: torch.Tensor, layout: ObservationLayout) -> torch.Tensor:
    delta = layout.ee_position(state) - layout.target_position(state)
    return delta.pow(2).sum(dim=-1, keepdim=True)


def _goal_distance_sq_np(state: np.ndarray, layout: ObservationLayout) -> float:
    delta = np.asarray(state[layout.ee_slice] - state[layout.target_slice], dtype=np.float32)
    return float(np.dot(delta, delta))


def _workspace_barriers(
    state: torch.Tensor,
    layout: ObservationLayout,
    config: BarrierConfig,
) -> dict[str, torch.Tensor]:
    ee = layout.ee_position(state)
    out = {
        "x_min": ee[..., 0:1] - config.x_low,
        "x_max": config.x_high - ee[..., 0:1],
        "y_min": ee[..., 1:2] - config.y_low,
        "y_max": config.y_high - ee[..., 1:2],
        "z_min": ee[..., 2:3] - config.z_low,
        "z_max": config.z_high - ee[..., 2:3],
    }
    if config.object_clearance > 0.0:
        ee_expanded = ee.unsqueeze(-2)
        objects = layout.object_positions(state)
        mask = layout.object_mask(state).unsqueeze(-1)
        distances = torch.linalg.norm(objects - ee_expanded, dim=-1, keepdim=True)
        masked = torch.where(mask > 0.5, distances, torch.full_like(distances, float("inf")))
        out["object_clearance"] = masked.min(dim=-2).values - config.object_clearance
    return out


def _barrier_violations(
    state: torch.Tensor,
    predicted_next_state: torch.Tensor,
    layout: ObservationLayout,
    config: BarrierConfig,
) -> dict[str, torch.Tensor]:
    current = _workspace_barriers(state, layout, config)
    nxt = _workspace_barriers(predicted_next_state, layout, config)
    return {
        name: F.relu(current[name] - nxt[name] - config.eta * current[name])
        for name in current
    }


def _safety_cost(state: np.ndarray, layout: ObservationLayout, config: BarrierConfig) -> float:
    state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    values = _workspace_barriers(state_t, layout, config)
    total = 0.0
    for value in values.values():
        total += float(F.relu(-value).sum().item())
    return total


def _backup_project_action(
    state: np.ndarray,
    action: np.ndarray,
    layout: ObservationLayout,
    config: BarrierConfig,
    *,
    action_step_xyz: float,
    goal_gain: float,
) -> np.ndarray:
    ee = state[layout.ee_slice]
    goal = state[layout.target_slice]
    proposed_target = ee + np.asarray(action[:3], dtype=np.float32) * float(action_step_xyz)
    clipped = proposed_target.copy()
    clipped[0] = float(np.clip(clipped[0], config.x_low, config.x_high))
    clipped[1] = float(np.clip(clipped[1], config.y_low, config.y_high))
    clipped[2] = float(np.clip(clipped[2], config.z_low, config.z_high))

    toward_goal = goal - ee
    goal_norm = float(np.linalg.norm(toward_goal))
    if goal_norm > 1e-6:
        toward_goal = toward_goal / goal_norm
    goal_delta = toward_goal * float(action_step_xyz)
    mixed_target = clipped + float(goal_gain) * goal_delta
    mixed_target[0] = float(np.clip(mixed_target[0], config.x_low, config.x_high))
    mixed_target[1] = float(np.clip(mixed_target[1], config.y_low, config.y_high))
    mixed_target[2] = float(np.clip(mixed_target[2], config.z_low, config.z_high))

    safe_action = np.asarray(action, dtype=np.float32).copy()
    safe_action[:3] = np.clip((mixed_target - ee) / max(float(action_step_xyz), 1e-6), -1.0, 1.0)
    return safe_action


def _should_use_backup(
    state: np.ndarray,
    action: np.ndarray,
    layout: ObservationLayout,
    config: BarrierConfig,
    *,
    action_step_xyz: float,
) -> bool:
    ee = np.asarray(state[layout.ee_slice], dtype=np.float32)
    proposed_target = ee + np.asarray(action[:3], dtype=np.float32) * float(action_step_xyz)
    if proposed_target[0] < config.x_low or proposed_target[0] > config.x_high:
        return True
    if proposed_target[1] < config.y_low or proposed_target[1] > config.y_high:
        return True
    if proposed_target[2] < config.z_low or proposed_target[2] > config.z_high:
        return True
    return False


def _critic_min(q_values: list[torch.Tensor]) -> torch.Tensor:
    return torch.min(torch.cat(q_values, dim=-1), dim=-1, keepdim=True).values


def _redq_target_min(q_values: list[torch.Tensor], subset_size: int) -> torch.Tensor:
    if subset_size >= len(q_values):
        return _critic_min(q_values)
    indices = random.sample(range(len(q_values)), k=int(subset_size))
    chosen = [q_values[idx] for idx in indices]
    return _critic_min(chosen)


class Trainer:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        layout: ObservationLayout,
        action_dim: int,
        run_dir: Path,
        device: torch.device,
        writer: Any,
    ) -> None:
        self.args = args
        self.layout = layout
        self.action_dim = int(action_dim)
        self.run_dir = run_dir
        self.device = device
        self.writer = writer
        self.barrier_config = _barrier_config_from_args(args)

        self.is_stochastic = args.algorithm in {"blac", "bac", "sac", "redq"}
        self.use_constraints = args.algorithm in {"blac", "bac"}
        self.use_lyapunov = args.algorithm == "blac"
        num_q = int(args.num_q) if args.algorithm == "redq" else 2

        if self.is_stochastic:
            self.actor: nn.Module = GaussianPolicy(layout.state_dim, action_dim, args.hidden_dim).to(device)
            self.actor_target = None
        else:
            self.actor = DeterministicPolicy(layout.state_dim, action_dim, args.hidden_dim).to(device)
            self.actor_target = DeterministicPolicy(layout.state_dim, action_dim, args.hidden_dim).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())

        self.critics = QEnsemble(layout.state_dim, action_dim, args.hidden_dim, num_q).to(device)
        self.critics_target = QEnsemble(layout.state_dim, action_dim, args.hidden_dim, num_q).to(device)
        self.critics_target.load_state_dict(self.critics.state_dict())

        self.lyapunov = LyapunovNetwork(layout.state_dim, args.hidden_dim).to(device)
        self.lyapunov_target = LyapunovNetwork(layout.state_dim, args.hidden_dim).to(device)
        self.lyapunov_target.load_state_dict(self.lyapunov.state_dict())

        self.dynamics = DynamicsModel(layout.state_dim, action_dim, args.hidden_dim).to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critics.parameters(), lr=args.critic_lr)
        self.lyapunov_optim = torch.optim.Adam(self.lyapunov.parameters(), lr=args.lyapunov_lr)
        self.dynamics_optim = torch.optim.Adam(self.dynamics.parameters(), lr=args.dynamics_lr)

        self.log_alpha: torch.Tensor | None = None
        self.alpha_optim: torch.optim.Optimizer | None = None
        if self.is_stochastic and args.autotune_alpha:
            self.target_entropy = -float(action_dim) * float(args.target_entropy_scale)
            self.log_alpha = torch.tensor(math.log(max(args.alpha, 1e-6)), device=device, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.actor_lr)
        else:
            self.target_entropy = -float(action_dim)

        self.constraint_names = tuple(_workspace_barriers(
            torch.zeros((1, layout.state_dim), device=device),
            layout,
            self.barrier_config,
        ).keys())
        self.lambda_values = {name: 0.0 for name in self.constraint_names}
        self.rho_values = {name: float(args.rho_init) for name in self.constraint_names}
        self.zeta = 0.0
        self.rho_zeta = float(args.rho_init)
        self.gradient_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        if self.log_alpha is None:
            return torch.tensor(float(self.args.alpha), device=self.device)
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, *, deterministic: bool, global_step: int) -> np.ndarray:
        if global_step < int(self.args.start_steps):
            action = np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)
        else:
            state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_t = self.actor.act(state_t, deterministic=deterministic)
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            if self.args.algorithm == "td3" and not deterministic:
                noise = np.random.normal(0.0, float(self.args.exploration_noise), size=action.shape).astype(np.float32)
                action = np.clip(action + noise, -1.0, 1.0)

        if self.args.enable_backup_controller and _should_use_backup(
            state,
            action,
            self.layout,
            self.barrier_config,
            action_step_xyz=float(self.args.action_step_xyz),
        ):
            action = _backup_project_action(
                state,
                action,
                self.layout,
                self.barrier_config,
                action_step_xyz=float(self.args.action_step_xyz),
                goal_gain=float(self.args.backup_goal_gain),
            )
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _update_critics(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            if self.is_stochastic:
                next_action, next_log_prob, _ = self.actor.sample(next_state)
                target_qs = self.critics_target(next_state, next_action)
                if self.args.algorithm == "redq":
                    target_q = _redq_target_min(target_qs, int(self.args.redq_target_subset))
                else:
                    target_q = _critic_min(target_qs)
                target_q = target_q - self.alpha.detach() * next_log_prob
            else:
                assert self.actor_target is not None
                next_action = self.actor_target(next_state)
                noise = torch.randn_like(next_action) * float(self.args.target_policy_noise)
                noise = noise.clamp(-float(self.args.target_noise_clip), float(self.args.target_noise_clip))
                next_action = (next_action + noise).clamp(-1.0, 1.0)
                target_q = _critic_min(self.critics_target(next_state, next_action))
            target = reward + (1.0 - done) * float(self.args.gamma) * target_q

        pred_qs = self.critics(state, action)
        critic_loss = sum(F.mse_loss(pred_q, target) for pred_q in pred_qs) / len(pred_qs)

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()
        return {"critic_loss": float(critic_loss.item()), "target_q": float(target.mean().item())}

    def _update_dynamics_and_lyapunov(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        state = batch["state"]
        action = batch["action"]
        next_state = batch["next_state"]
        cost = batch["cost"]
        done = batch["done"]

        pred_next_state = self.dynamics(state, action)
        dynamics_loss = F.mse_loss(pred_next_state, next_state)
        self.dynamics_optim.zero_grad(set_to_none=True)
        dynamics_loss.backward()
        self.dynamics_optim.step()

        with torch.no_grad():
            lyapunov_target = cost + (1.0 - done) * float(self.args.gamma_c) * self.lyapunov_target(next_state)
        lyapunov_pred = self.lyapunov(state)
        lyapunov_loss = F.mse_loss(lyapunov_pred, lyapunov_target)
        self.lyapunov_optim.zero_grad(set_to_none=True)
        lyapunov_loss.backward()
        self.lyapunov_optim.step()

        return {
            "dynamics_loss": float(dynamics_loss.item()),
            "lyapunov_loss": float(lyapunov_loss.item()),
            "lyapunov_value": float(lyapunov_pred.mean().item()),
        }

    def _constraint_penalty(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        predicted_next_state = self.dynamics(state, action)
        penalties = torch.zeros((1,), device=self.device, dtype=state.dtype)
        metrics: dict[str, float] = {}

        barrier_terms = _barrier_violations(state, predicted_next_state, self.layout, self.barrier_config)
        for name, violation in barrier_terms.items():
            mean_violation = violation.mean()
            metrics[f"constraint/{name}"] = float(mean_violation.item())
            penalties = penalties + self.lambda_values[name] * mean_violation
            penalties = penalties + 0.5 * self.rho_values[name] * mean_violation.pow(2)

        if self.use_lyapunov:
            current_l = self.lyapunov(state)
            next_l = self.lyapunov(predicted_next_state)
            clf_violation = F.relu(next_l - current_l + float(self.args.beta) * current_l)
            clf_mean = clf_violation.mean()
            metrics["constraint/clf"] = float(clf_mean.item())
            penalties = penalties + self.zeta * clf_mean
            penalties = penalties + 0.5 * self.rho_zeta * clf_mean.pow(2)

        return penalties, metrics

    def _update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        state = batch["state"]
        metrics: dict[str, float] = {}

        if self.is_stochastic:
            action, log_prob, _ = self.actor.sample(state)
            actor_loss = (self.alpha.detach() * log_prob - _critic_min(self.critics(state, action))).mean()
            metrics["log_prob"] = float(log_prob.mean().item())
        else:
            action = self.actor(state)
            actor_loss = -self.critics(state, action)[0].mean()

        if self.use_constraints:
            penalty, constraint_metrics = self._constraint_penalty(state, action)
            actor_loss = actor_loss + penalty
            metrics.update(constraint_metrics)

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()
        metrics["actor_loss"] = float(actor_loss.item())
        return metrics

    def _update_alpha(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        if not self.is_stochastic or self.log_alpha is None or self.alpha_optim is None:
            return {"alpha": float(self.alpha.item())}
        state = batch["state"]
        with torch.no_grad():
            _, log_prob, _ = self.actor.sample(state)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()
        return {
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
        }

    def _update_multipliers(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        if not self.use_constraints:
            return {}

        state = batch["state"]
        with torch.no_grad():
            if self.is_stochastic:
                action, _, _ = self.actor.sample(state)
            else:
                action = self.actor(state)
            predicted_next_state = self.dynamics(state, action)
            barrier_terms = _barrier_violations(state, predicted_next_state, self.layout, self.barrier_config)
            metrics: dict[str, float] = {}
            for name, violation in barrier_terms.items():
                mean_violation = float(violation.mean().item())
                self.lambda_values[name] = max(
                    0.0,
                    self.lambda_values[name] + float(self.args.constraint_lr) * mean_violation,
                )
                self.rho_values[name] = min(
                    float(self.args.rho_max),
                    self.rho_values[name] * float(self.args.rho_growth),
                )
                metrics[f"lambda/{name}"] = self.lambda_values[name]
                metrics[f"rho/{name}"] = self.rho_values[name]

            if self.use_lyapunov:
                current_l = self.lyapunov(state)
                next_l = self.lyapunov(predicted_next_state)
                clf_violation = F.relu(next_l - current_l + float(self.args.beta) * current_l)
                clf_mean = float(clf_violation.mean().item())
                self.zeta = max(0.0, self.zeta + float(self.args.constraint_lr) * clf_mean)
                self.rho_zeta = min(float(self.args.rho_max), self.rho_zeta * float(self.args.rho_growth))
                metrics["lambda/clf"] = self.zeta
                metrics["rho/clf"] = self.rho_zeta
        return metrics

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        metrics.update(self._update_critics(batch))
        metrics.update(self._update_dynamics_and_lyapunov(batch))

        if self.args.algorithm != "td3" or self.gradient_step % int(self.args.policy_delay) == 0:
            metrics.update(self._update_actor(batch))
            metrics.update(self._update_alpha(batch))
            metrics.update(self._update_multipliers(batch))
            if self.actor_target is not None:
                _soft_update(self.actor, self.actor_target, float(self.args.tau))

        _soft_update(self.critics, self.critics_target, float(self.args.tau))
        _soft_update(self.lyapunov, self.lyapunov_target, float(self.args.tau))
        self.gradient_step += 1
        return metrics

    def save(self, episode: int) -> Path:
        path = self.run_dir / f"checkpoint_ep{episode:04d}.pt"
        payload = {
            "algorithm": self.args.algorithm,
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "lyapunov": self.lyapunov.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "layout": asdict(self.layout),
            "barrier_config": asdict(self.barrier_config),
            "lambda_values": dict(self.lambda_values),
            "rho_values": dict(self.rho_values),
            "zeta": float(self.zeta),
            "rho_zeta": float(self.rho_zeta),
            "episode": int(episode),
        }
        torch.save(payload, path)
        latest = self.run_dir / "latest.pt"
        torch.save(payload, latest)
        return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _log_scalars(writer: Any, metrics: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar(key, float(value), int(step))
    writer.flush()


def _print_episode_summary(
    *,
    episode: int,
    total_steps: int,
    episode_reward: float,
    episode_cost: float,
    episode_length: int,
    algorithm: str,
) -> None:
    print(
        f"[{algorithm}] episode={episode:04d} "
        f"steps={total_steps:07d} "
        f"len={episode_length:03d} "
        f"reward={episode_reward:8.3f} "
        f"safety_cost={episode_cost:8.3f}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args, unknown = parse_args(argv)
    if unknown:
        print(
            "[blac] Ignoring unrecognized stage arguments for compatibility: "
            + " ".join(str(part) for part in unknown),
            flush=True,
        )

    _set_seed(int(args.seed))
    device = torch.device(args.device)
    run_dir = _make_run_dir(args.run_root_dir, args.run_id, args.algorithm)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard")) if SummaryWriter is not None else None

    _write_json(run_dir / "config.json", vars(args))

    env = None
    try:
        env = _build_env(args)
        obs, _ = env.reset(seed=int(args.seed))
        layout = ObservationLayout.from_observation(obs)
        state = layout.flatten(obs)
        trainer = Trainer(
            args=args,
            layout=layout,
            action_dim=int(env.action_space.shape[0]),
            run_dir=run_dir,
            device=device,
            writer=writer,
        )
        buffer = ReplayBuffer(int(args.replay_size), layout.state_dim, int(env.action_space.shape[0]))

        total_steps = 0
        metrics_path = run_dir / "metrics.jsonl"

        for episode in range(1, int(args.max_episodes) + 1):
            if episode > 1:
                obs, _ = env.reset()
                state = layout.flatten(obs)
            episode_reward = 0.0
            episode_cost = 0.0
            episode_metrics: dict[str, float] = {}

            for step_idx in range(int(args.max_env_steps)):
                deterministic = False
                action = trainer.select_action(state, deterministic=deterministic, global_step=total_steps)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = layout.flatten(next_obs)
                safety_cost = _safety_cost(next_state, layout, trainer.barrier_config)
                stability_cost = _goal_distance_sq_np(next_state, layout)
                done = bool(terminated or truncated)

                buffer.add(state, action, float(reward), float(stability_cost), next_state, done)

                state = next_state
                total_steps += 1
                episode_reward += float(reward)
                episode_cost += float(safety_cost)

                if buffer.size >= int(args.update_after):
                    for _ in range(int(args.updates_per_step)):
                        batch = buffer.sample(int(args.batch_size), device=device)
                        episode_metrics = trainer.train_step(batch)
                        _log_scalars(writer, episode_metrics, trainer.gradient_step)

                if done:
                    break

            summary = {
                "episode": int(episode),
                "total_steps": int(total_steps),
                "episode_length": int(step_idx + 1),
                "episode_reward": float(episode_reward),
                "episode_safety_cost": float(episode_cost),
                **{key: float(value) for key, value in episode_metrics.items()},
            }
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, sort_keys=True) + "\n")

            _print_episode_summary(
                episode=episode,
                total_steps=total_steps,
                episode_reward=episode_reward,
                episode_cost=episode_cost,
                episode_length=step_idx + 1,
                algorithm=args.algorithm,
            )

            if writer is not None:
                writer.add_scalar("episode/reward", float(episode_reward), int(episode))
                writer.add_scalar("episode/safety_cost", float(episode_cost), int(episode))
                writer.add_scalar("episode/length", float(step_idx + 1), int(episode))
                writer.flush()

            if episode % int(args.save_every_episodes) == 0 or episode == int(args.max_episodes):
                checkpoint = trainer.save(episode)
                print(f"[blac] Saved checkpoint: {checkpoint}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
