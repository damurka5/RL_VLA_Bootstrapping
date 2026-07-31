from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUPPORTED_SIMULATOR_BACKENDS = ("mujoco_cpu", "mjlab_mjwarp")


class SimulatorDependencyError(RuntimeError):
    """Raised when an explicitly selected simulator backend is unavailable."""


# Importable without CUDA; the backend itself additionally needs mjlab.
MJWARP_RUNTIME_PACKAGES = ("torch", "warp", "mujoco_warp")


def missing_mjwarp_dependencies() -> tuple[str, ...]:
    """Names of the MJWarp runtime packages that are not installed."""

    import importlib.util

    missing: list[str] = []
    for name in MJWARP_RUNTIME_PACKAGES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            # A namespace-package parent that is itself broken counts as
            # missing rather than crashing the caller's backend selection.
            found = False
        if not found:
            missing.append(name)
    return tuple(missing)


def resolve_simulator_backend(
    requested: str,
    *,
    cuda_available: bool,
    missing_dependencies: Sequence[str] = (),
) -> tuple[str, str]:
    """Pick a physics backend, and say why, for tools that can run on either.

    ``requested`` is one of ``SUPPORTED_SIMULATOR_BACKENDS`` or ``"auto"``.
    Returns ``(backend, reason)``; the reason is meant to be printed and stored,
    because a result from the CPU reference physics and one from the production
    MJWarp physics are not interchangeable and a run that does not say which it
    used cannot be compared to anything.

    Naming an unavailable backend explicitly is an error rather than a silent
    downgrade -- asking for MJWarp and quietly getting CPU MuJoCo is exactly the
    kind of substitution that makes a "verified" result mean nothing.
    """

    missing = tuple(str(name) for name in missing_dependencies)
    if requested not in SUPPORTED_SIMULATOR_BACKENDS + ("auto",):
        raise ValueError(
            f"Unsupported simulator backend {requested!r}; expected one of "
            f"{SUPPORTED_SIMULATOR_BACKENDS + ('auto',)}."
        )
    if requested == "mujoco_cpu":
        return "mujoco_cpu", "requested explicitly"
    if requested == "mjlab_mjwarp":
        if not cuda_available:
            raise SimulatorDependencyError(
                "mjlab_mjwarp was requested but no CUDA device is available."
            )
        if missing:
            raise SimulatorDependencyError(
                "mjlab_mjwarp was requested but these packages are missing: "
                + ", ".join(missing)
            )
        return "mjlab_mjwarp", "requested explicitly"
    if not cuda_available:
        return "mujoco_cpu", "auto: no CUDA device"
    if missing:
        return "mujoco_cpu", "auto: missing " + ", ".join(missing)
    return "mjlab_mjwarp", "auto: CUDA and the MJWarp runtime are available"


@dataclass(frozen=True)
class CDPRBackendConfig:
    backend: str = "mujoco_cpu"
    worlds_per_rank: int = 1
    groups_per_rank: int = 1
    grpo_group_size: int = 8
    hold_steps: int = 6
    action_step_xyz: float = 0.015
    action_step_yaw: float = 0.08
    action_step_gripper: float = 0.05
    lock_non_commanded_axes: bool = False
    lock_non_commanded_axes_threshold: float = 0.05
    render_width: int = 320
    render_height: int = 240
    object_slots: int = 4
    nconmax: int = 256
    njmax: int = 1024
    nccdmax: int | None = None
    device: str = "cuda:0"
    xml_path: Path | None = None
    workspace_x: tuple[float, float] = (-0.28, 0.28)
    workspace_y: tuple[float, float] = (-0.28, 0.28)
    # This clamps the CONTROLLER target, i.e. everywhere the policy may drive,
    # which is a different (and stricter) thing than the task's spawn bounds or
    # the reward's hover height. The ceiling is well under the 1.31 m rotors,
    # where the cable geometry approaches a singularity.
    #
    # The 0.25 floor is a MOVE-TO value and it makes grasping impossible: the
    # finger pads sit 0.0075 m below ee_base (measured, see
    # cdpr_gripper_geometry), so at the floor the pads bottom out at 0.2425 while
    # graspable object centres sit at 0.178-0.195. pick_up ran 10M steps unable
    # to reach any object. Grasp phases must pass a lower floor explicitly via
    # --controller-workspace-z-bounds; 0.25 stays the default because a hover
    # task wants the fingers well clear of the 0.15 m desk.
    workspace_z: tuple[float, float] = (0.25, 0.60)

    def validate(self) -> None:
        if self.backend not in SUPPORTED_SIMULATOR_BACKENDS:
            raise ValueError(
                f"Unsupported simulator backend {self.backend!r}; "
                f"expected one of {SUPPORTED_SIMULATOR_BACKENDS}."
            )
        if self.worlds_per_rank < 1:
            raise ValueError("worlds_per_rank must be positive.")
        if self.groups_per_rank < 1:
            raise ValueError("groups_per_rank must be positive.")
        expected = int(self.groups_per_rank) * int(self.grpo_group_size)
        if self.backend == "mjlab_mjwarp" and int(self.worlds_per_rank) != expected:
            raise ValueError(
                "MJWarp worlds must be exactly groups_per_rank * grpo_group_size: "
                f"{self.worlds_per_rank} != {self.groups_per_rank} * "
                f"{self.grpo_group_size}."
            )
        if self.hold_steps < 0:
            raise ValueError("hold_steps must be non-negative.")
        if self.lock_non_commanded_axes_threshold < 0.0:
            raise ValueError(
                "lock_non_commanded_axes_threshold must be non-negative."
            )
        if self.object_slots != 4:
            raise ValueError(
                "The active CDPR scene contract requires exactly four fixed object slots."
            )
        if self.backend == "mjlab_mjwarp":
            if self.xml_path is None:
                raise ValueError("mjlab_mjwarp requires a fixed-topology xml_path.")
            if not Path(self.xml_path).expanduser().exists():
                raise FileNotFoundError(f"MJWarp MJCF does not exist: {self.xml_path}")
        for name, bounds in (
            ("workspace_x", self.workspace_x),
            ("workspace_y", self.workspace_y),
            ("workspace_z", self.workspace_z),
        ):
            if len(bounds) != 2 or float(bounds[0]) >= float(bounds[1]):
                raise ValueError(f"{name} must be an increasing pair, got {bounds}.")

    @property
    def physics_substeps(self) -> int:
        """One commanded step plus `hold_steps`, matching the CPU backend."""
        return 1 + int(self.hold_steps)


@dataclass(frozen=True)
class CDPRRenderBatch:
    """GPU-resident camera images in normalized BCHW RGB format."""

    overview: Any
    wrist: Any

    @property
    def aux(self) -> Any:
        """The active SmolVLA contract duplicates wrist into camera slot three."""
        return self.wrist


@dataclass(frozen=True)
class CDPRLowDimBatch:
    ee_position: Any
    ee_quaternion: Any
    ee_yaw: Any
    gripper_opening: Any
    target_position: Any
    tendon_lengths: Any
    object_positions: Any
    object_quaternions: Any


@dataclass(frozen=True)
class CDPRFingerContactBatch:
    """Per-world physical contact evidence for a selected free-body object."""

    left_contact: Any
    right_contact: Any
    left_normal_force: Any
    right_normal_force: Any

    @property
    def bilateral_contact(self) -> Any:
        return self.left_contact & self.right_contact


class CDPRSimulatorBackend(abc.ABC):
    """Backend boundary used by rollout code.

    Implementations own simulator/controller state and expose fixed-shape
    tensors.  Validation export is intentionally separate from the training
    hot path so it may transfer selected worlds to the host.
    """

    config: CDPRBackendConfig

    @property
    def worlds_per_rank(self) -> int:
        return int(self.config.worlds_per_rank)

    @property
    @abc.abstractmethod
    def device(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def reset_worlds(
        self,
        world_indices: Any,
        *,
        qpos: Any | None = None,
        qvel: Any | None = None,
        controller_state: Mapping[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def broadcast_group_state(self, base_world_indices: Any) -> None:
        """Copy one base state to every candidate in each local group."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, actions: Any, active_mask: Any) -> CDPRLowDimBatch:
        """Advance every world with fixed-shape actions and completion masks."""
        raise NotImplementedError

    @abc.abstractmethod
    def low_dim_observations(self) -> CDPRLowDimBatch:
        raise NotImplementedError

    @abc.abstractmethod
    def render_policy_cameras(self) -> CDPRRenderBatch:
        raise NotImplementedError

    @abc.abstractmethod
    def body_pose(self, body_names: Sequence[str]) -> tuple[Any, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def body_velocity(self, body_names: Sequence[str]) -> tuple[Any, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def contact_mask(self, geom_a_ids: Any, geom_b_ids: Any) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def finger_object_contact_metrics(
        self, target_slots: Any
    ) -> CDPRFingerContactBatch:
        """Return bilateral pad contacts and solved normal forces per world."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_object_catalogs(self, catalog_ids: Any) -> None:
        """Apply per-world fixed-slot geometry/material variants."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_visual_variants(
        self,
        texture_variant_ids: Any,
        background_rgba: Any,
        gripper_shade: Any,
    ) -> None:
        """Apply per-world visual domain randomization without recompilation."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_end_effector_poses(
        self,
        positions: Any,
        yaws: Any,
        *,
        zero_velocity: bool = True,
    ) -> None:
        """Teleport batched reset poses and recompute CDPR tendon preload."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_gripper_openings(self, openings: Any) -> None:
        """Force normalized gripper reset state and matching actuator targets."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_free_body_poses(
        self,
        body_ids: Any,
        positions: Any,
        quaternions: Any | None = None,
        *,
        zero_velocity: bool = True,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def export_worlds(self, world_indices: Sequence[int]) -> list[dict[str, Any]]:
        """Host-side debug/validation export; never used in training rollout."""
        raise NotImplementedError

    @abc.abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None


def create_cdpr_backend(
    config: CDPRBackendConfig,
    **kwargs: Any,
) -> CDPRSimulatorBackend:
    config.validate()
    if config.backend == "mjlab_mjwarp":
        from .mjlab_mjwarp_backend import MJLabMJWarpCDPRBackend

        return MJLabMJWarpCDPRBackend(config=config, **kwargs)
    from .mujoco_cpu_backend import MujocoCPUReferenceBackend

    return MujocoCPUReferenceBackend(config=config, **kwargs)
