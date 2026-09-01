#!/usr/bin/env python3
"""Record the deterministic policy's own episodes, and replay them.

Phase 3 (self-imitation) needs a dataset of trajectories the campaign's own RL
checkpoints produced, and part 3 of it needs to re-simulate smoothed action
sequences and keep only what still succeeds. Both halves are the same machine:
run one validation round, capture exactly what physics executed, then run it
again feeding those actions back.

Why this is a sibling of ``xy_approach_probe`` rather than a sixth leg inside
it. The probe measures a policy; this records one. It imports the probe's
``_build_world`` so there is still exactly one place that reproduces the
training stack -- the campaign has been bitten twice by a second builder
drifting from the first -- but the recording concerns are its own.

Why the probe's existing ``policy_trace.npz`` is not enough. It stores
``commanded0``, action index 0 of each chunk, which was the right call for a
cosine metric and is a deliberate documented choice. But the env steps the
WHOLE chunk::

    for action_index in range(self.actions_per_policy_decision):
        low_dim = self.backend.step(actions[:, action_index], step_active)

so three of every four executed actions are absent from that trace, along with
the per-step active mask. An episode terminates on success
(``terminated = success | wrong_place_settled | timeout``) and the policy keeps
emitting actions into a frozen world afterwards, so without the mask every
successful episode's demonstration would be padded with actions that moved
nothing. Both are silent dataset poisoners, which is why recording happens at
the two points below and not at the chunk producer.


Interception points
-------------------

``backend.step(actions, active_mask)`` -- the executed command and the mask, as
physics received them. This is the only place that is ground truth about what
the plant was driven with; anything upstream is what the policy *intended*.

``evaluate_active_sparse_tasks(...)`` -- patched in the collector's module
namespace, called through, never reimplemented. Its keyword arguments carry the
post-grasp-update ``ee_position``, ``object_positions``, ``gripper_opening``
and ``caught_target`` that production actually scores, and its return carries
``success`` and ``terminated``. Relabeling and survival must be decided by this
function and nothing else: ``robots/cdpr/cdpr_dataset/cdpr_lchol_spec.py``
already contains a second, independent implementation of these predicates
(``_grab_predicate``, ``_pick_predicate``, ...) keyed off a dict the MJWarp
collector never builds, and this campaign has paid for that class of
duplication more than once.

The two paths report ``active`` independently, so they cross-check each other;
a disagreement is raised rather than averaged away.


Replay is a seeded re-run, not a state restore
----------------------------------------------

The reset is a pure function of its seed::

    base_seed + rank * 1_000_003 + update_index * 10_000_019 + round_index * 100_003

so replaying does not need MuJoCo state serialization: rebuild the same world
at the same ``--start-distance-cap`` and the same ``--round-index``, and the
starts are identical by construction. Actions are substituted at
``backend.step``, so physics sees the recorded bytes exactly.

Be honest about what that measures. Identical actions into an identical reset
should reproduce the trajectory bitwise, so ``--mode replay`` against an
unmodified recording is expected to survive at 100%. It is a PLUMBING test --
chunk ordering, world ordering, horizon, mask, npz round-trip -- and not
evidence that the recordings are valid training targets. It is worth running
because an off-by-one here would silently corrupt every downstream number, and
it is worthless as a claim about recording fidelity on its own. That is what
``--repeat`` is for: two identical recording runs measure the simulator's own
determinism, and any replay survival at or above that floor is indistinguishable
from noise.

The policy stays loaded in replay mode even though its output is discarded, so
that a record arm and a replay arm differ in exactly one respect -- the five
numbers handed to the plant -- and nothing else can drift between them.


Usage
-----

Record twice at the near rung, which is arms A and B together::

    RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run -n cdpr-mjlab \\
    python tools/audit/sil_record.py --mode record --repeat 2 \\
        --checkpoint runs/.../step_15000502/smolvla_grpo_adapter.pt \\
        --config configs/examples/cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml \\
        --start-distance-cap 0.01 --output tools/audit/out/sil_cap001

Replay the first recording, which is arm C::

    ... --mode replay --actions tools/audit/out/sil_cap001/record_00.npz \\
        --start-distance-cap 0.01 --output tools/audit/out/sil_cap001_replay

Diagnose two runs against each other. Pure numpy over the written npz files --
no checkpoint, no CUDA, runs anywhere the artefacts do::

    python tools/audit/sil_record.py --mode compare \\
        --actions tools/audit/out/sil_cap001/record_00.npz \\
        --against tools/audit/out/sil_cap001/record_01.npz \\
        --output tools/audit/out/sil_cap001_compare
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imported before anything here can reach huggingface_hub. The probe configures
# both offline switches at import time, and they are read into module constants
# on the hub's first import, so setting them afterwards is silently too late.
from tools.audit.xy_approach_probe import _build_world  # noqa: E402

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (  # noqa: E402
    ACTIVE_INSTRUCTION_TYPES,
)
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (  # noqa: E402
    ACTIVE_CDPR_CATALOGS,
)


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


@dataclass
class _Recording:
    """One validation round, as physics executed it.

    Per env step, shaped ``[S, W, ...]`` where ``S`` is
    ``max_decisions * actions_per_policy_decision`` and ``W`` is worlds. Per
    episode, shaped ``[W, ...]``. Nothing here is derived: every array is a
    tensor that production produced, copied to the host.
    """

    # Per env step.
    actions: np.ndarray  # [S, W, 5] what backend.step received
    active: np.ndarray  # [S, W] bool, the step_active mask
    success: np.ndarray  # [S, W] bool, already masked by active upstream
    terminated: np.ndarray  # [S, W] bool
    caught_target: np.ndarray  # [S, W] bool
    ee_xyz: np.ndarray  # [S, W, 3]
    gripper_opening: np.ndarray  # [S, W]
    object_xyz: np.ndarray  # [S, W, K, 3]

    # Per episode.
    instruction_ids: np.ndarray  # [W]
    target_slots: np.ndarray  # [W]
    reference_slots: np.ndarray  # [W]
    second_reference_slots: np.ndarray  # [W]
    horizons: np.ndarray  # [W] in decisions
    initial_target_xyz: np.ndarray  # [W, 3]
    support_surface_z: np.ndarray  # [W]
    release_threshold: np.ndarray  # [W]
    target_rest_height: np.ndarray  # [W]
    physical_grasp_at_reset: np.ndarray  # [W] bool
    instructions: np.ndarray  # [W] unicode

    # Round-level.
    actions_per_decision: int
    round_index: int
    diverged_worlds: int
    pick_lift_success_height: float

    # Per DECISION, not per env step: the policy is consulted once per chunk
    # and the plant is stepped ``actions_per_decision`` times from it. Optional
    # so the recordings written before observation capture existed stay
    # loadable -- they carry verdicts that are still valid, and re-harvesting
    # them would cost GPU time to reproduce data we already have.
    #
    # ``states`` is the residual's actual input, proprioception with the frozen
    # vision feature already concatenated. ``priors`` is load-bearing and easy
    # to overlook: the actor computes ``action = tanh(prior + residual)``, so
    # fitting the residual to a recorded action is impossible without the prior
    # that action was produced against.
    states: Any = None  # [D, W, state_dim]
    priors: Any = None  # [D, W, actions_per_decision, 5]

    # Which object each episode was asked to manipulate. Selecting on success
    # selects on object as well as on skill: robocasa_banana and robocasa_mug
    # are wider than the gripper's open gap in the seeded pose, so they cannot
    # be grasped and cannot appear among the successes -- and a pick_up set
    # harvested without recording this looks like "pick_up demonstrations"
    # while being demonstrations of four objects out of six.
    target_catalog_ids: Any = None  # [W]

    # The rung of the recording plan this round was collected at. NaN when the
    # recording predates the field, which is why every consumer must treat it
    # as unknown rather than as zero. Stored because a dataset pooled across
    # caps reports a source success rate that describes no actual collection
    # condition: placement runs ~0.93 at cap 0.01 and ~0.55 at 0.10, and one
    # averaged number hides which of the two a slice came from.
    start_distance_cap: float = float("nan")

    _OPTIONAL_ARRAYS = ("states", "priors", "target_catalog_ids")
    _REQUIRED_ARRAYS = (
        "actions", "active", "success", "terminated", "caught_target",
        "ee_xyz", "gripper_opening", "object_xyz",
        "instruction_ids", "target_slots", "reference_slots",
        "second_reference_slots", "horizons", "initial_target_xyz",
        "support_surface_z", "release_threshold", "target_rest_height",
        "physical_grasp_at_reset", "instructions",
    )
    _SCALARS = (
        ("actions_per_decision", np.int64),
        ("round_index", np.int64),
        ("diverged_worlds", np.int64),
        ("pick_lift_success_height", np.float64),
    )
    _OPTIONAL_SCALARS = (("start_distance_cap", np.float64, float("nan")),)

    @property
    def has_observations(self) -> bool:
        return self.states is not None and self.priors is not None

    @property
    def worlds(self) -> int:
        return int(self.actions.shape[1])

    @property
    def episode_success(self) -> np.ndarray:
        """Per-world verdict, latched exactly as ``validate_round`` latches it.

        ``candidate_success.logical_or_(result.success)`` over every env step,
        and ``result.success`` was already ``&= active`` inside the predicate.
        """

        return self.success.any(axis=0)

    @property
    def first_success_step(self) -> np.ndarray:
        """Env step of first success, or -1. The demonstration ends here.

        An episode terminates on success, so there is nothing after this step
        but a frozen world -- the actions past it must not enter the dataset.
        """

        any_success = self.success.any(axis=0)
        first = np.argmax(self.success, axis=0).astype(np.int64)
        return np.where(any_success, first, -1)

    @property
    def episode_length(self) -> np.ndarray:
        """Env steps this world was actually stepped for."""

        return self.active.sum(axis=0).astype(np.int64)

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: getattr(self, name) for name in self._REQUIRED_ARRAYS
        }
        for name in self._OPTIONAL_ARRAYS:
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        for name, caster in self._SCALARS:
            payload[name] = caster(getattr(self, name))
        for name, caster, _ in self._OPTIONAL_SCALARS:
            payload[name] = caster(getattr(self, name))
        np.savez_compressed(path, **payload)

    @classmethod
    def from_npz(cls, path: Path) -> "_Recording":
        with np.load(path, allow_pickle=False) as data:
            fields = {key: data[key] for key in data.files}
        missing = [
            name for name in cls._REQUIRED_ARRAYS if name not in fields
        ]
        if missing:
            raise ValueError(f"{path} is missing {missing}; not a recording.")
        kwargs: dict[str, Any] = {
            name: fields[name] for name in cls._REQUIRED_ARRAYS
        }
        # Absent rather than empty: a recording written before observation
        # capture existed is still a valid verdict record, and compare/replay
        # work on it unchanged. Only dataset extraction needs the observations.
        for name in cls._OPTIONAL_ARRAYS:
            kwargs[name] = fields.get(name)
        for name, caster in cls._SCALARS:
            kwargs[name] = caster(fields[name]).item()
        for name, caster, default in cls._OPTIONAL_SCALARS:
            kwargs[name] = (
                caster(fields[name]).item() if name in fields else default
            )
        return cls(**kwargs)


def _apply_determinism(
    *, torch_seed: int | None, deterministic_kernels: bool
) -> dict[str, Any]:
    """Pin the two things that can make an identical round come out different.

    Two candidate sources produce indistinguishable evidence at the magnitudes
    observed, and they have opposite consequences:

    A global RNG draw inside the forward. SmolVLA is a flow-matching policy and
    its sampler starts from noise; if that noise comes from the global torch
    generator, two rounds in one process get different draws and
    ``deterministic_action_chunks_tensor`` is deterministic only in its
    residual, not in the prior it adds to. That would make every
    "deterministic" validation number this campaign has quoted stochastic.

    Nondeterministic reduction order under bf16 autocast. The config sets
    ``mixed_precision: bf16``, whose relative epsilon is ~4e-3, so a different
    split-k or atomic ordering in one matmul lands in the same 1e-2 band a
    fresh noise draw would. That is a noise floor to be measured, not a bug to
    be fixed.

    Seeding separates them. If the null passes with ``--seed-torch`` and failed
    without it, the source is the RNG draw. If it still fails, the arithmetic
    itself is unordered, and ``--deterministic-kernels`` is the follow-up --
    with ``warn_only`` so an op lacking a deterministic implementation degrades
    the coverage rather than killing the run. Note that fully deterministic
    cuBLAS additionally needs ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the
    environment, which this cannot set from inside the process after torch has
    initialized.

    MEASURED, 2026-08-14, placement checkpoint step_15000502, cap 0.01, 512
    worlds. It is the first cause, and it is worse than a probe artefact:
    LeRobot's ``sample_actions`` runs
    ``if noise is None: noise = self.sample_noise(...)`` and ``sample_noise`` is
    a bare ``torch.normal`` with no ``generator=``. The wrapper never passes
    ``noise``, so the prior is a fresh global-RNG draw on every forward and
    every "deterministic" validation this campaign has quoted was sampling.

    With ``--seed-torch`` the step-0 action is bitwise identical across 512
    worlds and 5 dimensions -- max delta exactly 0.0 -- which also exonerates
    bf16: the forward is deterministic given identical inputs, so
    ``--deterministic-kernels`` was never needed.

    What survives seeding is chaos, not arithmetic. MuJoCo Warp under pinned
    actions diverges by a mean of 7.6e-06 m (max 7.7e-03 m) and flips **zero**
    verdicts on its own. Fed back through the policy, that micron-scale noise
    amplifies to 0.218 m of EE divergence and 34/512 = 6.6% flipped verdicts.
    So a seeded validation round is reproducible in its first decision and not
    in its outcome, and a single round at n=512 cannot resolve a slice whose
    success rate is near or below that band -- which is the whole pick_up
    column.

    The consequence that matters for this file: with actions pinned the verdict
    noise floor is zero (replay agreement 1.0, 0 flips), so a smoothing
    survival rate measured by replay is clean and any loss belongs to smoothing.
    """

    import torch

    applied: dict[str, Any] = {
        "torch_seed": None if torch_seed is None else int(torch_seed),
        "deterministic_kernels": bool(deterministic_kernels),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ""
        ),
    }
    if torch_seed is not None:
        torch.manual_seed(int(torch_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(torch_seed))
    if deterministic_kernels:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return applied


class _OracleActionSource:
    """Actions from the scripted oracle instead of from the policy.

    Rebuilt at every reset, because the phase chain a world runs depends on
    what its reset produced: an ungrasped container start gets the grasp
    prefix, a caught one does not, and an instruction with no oracle gets an
    empty chain and a zero action.

    Everything else in the recording path is untouched. The states and priors
    still come from `patched_action`, so a demonstration recorded this way
    carries the same inputs the residual would have seen; the reward, the grasp
    detector, the horizon and the success predicate are all the trainer's own.
    Only the five numbers handed to the plant come from somewhere else.
    """

    def __init__(self) -> None:
        self.oracle: Any = None
        self._previous_ee: np.ndarray | None = None
        self._target_slots: np.ndarray | None = None
        self._initial_target_z: np.ndarray | None = None

    def begin(self, reset: Any, world: Any) -> None:
        from tools.audit.sil_oracle import BatchedOracle
        from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
            OBJECT_VARIANTS,
        )

        task_state = reset.task_state
        worlds = int(world.layout.worlds_per_rank)
        metadata = dict(world.task_metadata or {})

        def number(key: str, default: float) -> float:
            try:
                return float(metadata.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        instruction_ids = _host_int(task_state.instruction_ids).reshape(-1)
        names = [_instruction_name(int(i)) for i in instruction_ids]
        target_slots = _host_int(task_state.target_slots).reshape(-1)
        reference_slots = _host_int(task_state.reference_slots).reshape(-1)
        # Per GROUP in the reset; the recording expands it the same way.
        catalogs = getattr(reset, "group_target_catalog_ids", None)
        if catalogs is None:
            catalog_ids = np.zeros(worlds, dtype=np.int64)
        else:
            catalog_ids = _host_int(catalogs).reshape(-1)
            if catalog_ids.shape[0] != worlds:
                catalog_ids = np.repeat(
                    catalog_ids, worlds // max(catalog_ids.shape[0], 1)
                )
        catalog_names = [
            ACTIVE_CDPR_CATALOGS[int(index)] for index in catalog_ids
        ]
        release_thresholds = _host_float(task_state.release_threshold).reshape(-1)
        plate_release = number("put_plate_release_height", 0.10)
        bowl_release = number("put_bowl_release_height", 0.10)
        self.oracle = BatchedOracle(
            instruction_types=names,
            # The reset decides this, not the config: the caught-object stage
            # is drawn per group, so two worlds of one round can need different
            # chains.
            starts_grasped=[
                bool(v) for v in _host_bool(task_state.grasped).reshape(-1)
            ],
            instruction_texts=[str(t) for t in reset.instructions],
            target_slots=target_slots,
            reference_slots=reference_slots,
            target_catalogs=catalog_names,
            fitted_openings=[
                float(OBJECT_VARIANTS[name].fitted_gripper_opening)
                for name in catalog_names
            ],
            # The harness opens a little past the threshold so the release is
            # unambiguous rather than exactly on the boundary.
            release_openings=[
                float(min(1.0, value + 0.05)) for value in release_thresholds
            ],
            grasp_height_offset=number("pick_grasp_height_offset", 0.0075),
            release_heights=[
                bowl_release if name == "put_into_bowl" else plate_release
                for name in names
            ],
            support_surface_z=_host_float(task_state.support_surface_z).reshape(-1),
            lift_success_height=number("pick_lift_success_height", 0.05),
            action_step_xyz=float(world.action_step_xyz),
            action_step_gripper=float(world.action_step_gripper),
        )
        # Cleared per reset: carrying the previous round's end pose into the
        # next round's first velocity would put one bogus damping term on step
        # 0 of every round after the first.
        self._previous_ee = None
        self._target_slots = target_slots
        self._initial_target_z = _host_float(
            task_state.initial_target_positions
        ).reshape(worlds, 3)[:, 2]

    def __call__(self, *, low_dim: Any, caught_target: Any, backend: Any) -> np.ndarray:
        assert self.oracle is not None, "begin() must run at reset"
        ee = _host_float(low_dim.ee_position)
        previous = getattr(self, "_previous_ee", None)
        if previous is None or previous.shape != ee.shape:
            previous = ee
        self._previous_ee = ee
        controller = backend.controller_state()
        commanded = np.asarray(controller["gripper"], dtype=np.float32).reshape(-1)
        # caught_target lags one step: it is captured at the predicate, which
        # runs after the step. It only ever latches on, and the phase that
        # reads it is waiting for exactly that transition.
        grasp = (
            np.zeros(ee.shape[0], dtype=bool)
            if caught_target is None
            else _host_bool(caught_target).reshape(-1)
        )
        return self.oracle.actions(
            ee=ee,
            ee_velocity=ee - previous,
            ee_yaw=_host_float(low_dim.ee_yaw).reshape(-1),
            measured_gripper=_host_float(low_dim.gripper_opening).reshape(-1),
            commanded_gripper=commanded,
            object_positions=_host_float(low_dim.object_positions),
            physical_grasp=grasp,
            initial_target_z=self._initial_target_z,
        )


class _RoundRecorder:
    """Runs one ``validate_round``, capturing what the plant executed.

    ``playback`` substitutes recorded actions at ``backend.step``. Everything
    else -- reset, reward, grasp detector, horizon, termination, predicate --
    stays the trainer's own code.
    """

    def __init__(
        self,
        world: Any,
        *,
        playback: np.ndarray | None = None,
        action_source: Any = None,
        horizon_override: int = 0,
        torch_seed: int | None = None,
        deterministic_kernels: bool = False,
    ) -> None:
        self.world = world
        self.playback = playback
        # Closed-loop substitution, where `playback` is open-loop: the oracle
        # needs the live observation to decide, so it is handed the low-dim
        # state the PREVIOUS step returned. One step of lag on a quantity the
        # controller is already integrating over four env steps per decision.
        self.action_source = action_source
        self._last_low_dim: Any = None
        self._last_caught: Any = None
        self.horizon_override = max(0, int(horizon_override))
        self.torch_seed = torch_seed
        self.deterministic_kernels = bool(deterministic_kernels)
        self.determinism: dict[str, Any] = {}
        self.reset: Any = None
        self.env_step = 0
        self._rows_step: list[dict[str, np.ndarray]] = []
        self._rows_eval: list[dict[str, np.ndarray]] = []
        # Saved originals plus whether the name was already an INSTANCE
        # attribute before the patch. Both patched names are ordinary methods,
        # so restoring by assignment would leave a bound method shadowing the
        # class for the rest of the process. The probe's _ArmRunner takes the
        # same care for the same reason; several arms run back to back over one
        # live trainer.
        self._rows_decision: list[dict[str, np.ndarray]] = []
        self._original_step: Any = None
        self._original_reset: Any = None
        self._original_predicate: Any = None
        self._original_action: Any = None
        self._owned_step = False
        self._owned_reset = False
        self._owned_action = False
        self._collector_module: Any = None

    # -- installation ---------------------------------------------------

    def __enter__(self) -> "_RoundRecorder":
        import rl_vla_bootstrapping.policy.mjwarp_rank_local_collector as module

        torch = self.world.torch
        backend = self.world.backend
        trainer = self.world.collector.trainer
        resetter = self.world.collector.resetter
        self._collector_module = module

        self._original_step = backend.step
        self._original_reset = resetter.reset
        self._original_predicate = module.evaluate_active_sparse_tasks
        self._original_action = trainer.deterministic_action_chunks_tensor
        self._owned_step = "step" in vars(backend)
        self._owned_reset = "reset" in vars(resetter)
        self._owned_action = (
            "deterministic_action_chunks_tensor" in vars(trainer)
        )

        def patched_reset(**kwargs: Any) -> Any:
            reset = self._original_reset(**kwargs)
            if self.horizon_override:
                reset.horizons.fill_(self.horizon_override)
            self.reset = reset
            self.env_step = 0
            # The first step has no previous one to read, so seed from the
            # backend directly.
            if self.action_source is not None:
                self._last_low_dim = backend.low_dim_observations()
                self._last_caught = None
                self.action_source.begin(reset, self.world)
            self._rows_step.clear()
            self._rows_eval.clear()
            self._rows_decision.clear()
            # Cable-singularity blowups are a real outcome here, not an
            # exception: a diverged world is neither a success nor a usable
            # demonstration, and its count belongs next to the survival rate.
            backend.pop_nonfinite_world_events()
            return reset

        def patched_step(actions: Any, active_mask: Any) -> Any:
            if self.playback is not None:
                if self.env_step >= self.playback.shape[0]:
                    raise RuntimeError(
                        f"Playback ran past the recording at env step "
                        f"{self.env_step} of {self.playback.shape[0]}. The "
                        "replay horizon does not match the recorded one; pass "
                        "the same --horizon-decisions and --start-distance-cap."
                    )
                actions = torch.as_tensor(
                    self.playback[self.env_step],
                    dtype=actions.dtype,
                    device=actions.device,
                )
            if self.action_source is not None:
                actions = torch.as_tensor(
                    self.action_source(
                        low_dim=self._last_low_dim,
                        caught_target=self._last_caught,
                        backend=backend,
                    ),
                    dtype=actions.dtype,
                    device=actions.device,
                )
            self._rows_step.append(
                {
                    "actions": _host_float(actions),
                    "active": _host_bool(active_mask),
                }
            )
            self.env_step += 1
            result = self._original_step(actions, active_mask)
            self._last_low_dim = result
            return result

        def patched_predicate(**kwargs: Any) -> Any:
            result = self._original_predicate(**kwargs)
            self._rows_eval.append(
                {
                    "success": _host_bool(result.success),
                    "terminated": _host_bool(result.terminated),
                    "active": _host_bool(kwargs["active_mask"]),
                    "caught_target": _host_bool(kwargs["caught_target"]),
                    "ee_xyz": _host_float(kwargs["ee_position"]),
                    "gripper_opening": _host_float(kwargs["gripper_opening"]),
                    "object_xyz": _host_float(kwargs["object_positions"]),
                }
            )
            self._last_caught = kwargs["caught_target"]
            return result

        def patched_action(
            *, states: Any, priors: Any, action_count: int
        ) -> Any:
            chunk = self._original_action(
                states=states, priors=priors, action_count=action_count
            )
            # The input side of the demonstration. `states` already carries the
            # frozen vision feature concatenated onto proprioception, so this is
            # literally what the residual saw, not a reconstruction of it. The
            # prior is captured with it because `action = tanh(prior +
            # residual)` -- a recorded action alone does not determine a
            # residual target.
            self._rows_decision.append(
                {
                    "states": _host_float(states),
                    "priors": _host_float(priors),
                }
            )
            return chunk

        backend.step = patched_step
        resetter.reset = patched_reset
        module.evaluate_active_sparse_tasks = patched_predicate
        trainer.deterministic_action_chunks_tensor = patched_action
        return self

    def __exit__(self, *exc: Any) -> None:
        backend = self.world.backend
        trainer = self.world.collector.trainer
        resetter = self.world.collector.resetter
        if self._owned_step:
            backend.step = self._original_step
        else:
            vars(backend).pop("step", None)
        if self._owned_reset:
            resetter.reset = self._original_reset
        else:
            vars(resetter).pop("reset", None)
        if self._owned_action:
            trainer.deterministic_action_chunks_tensor = self._original_action
        else:
            vars(trainer).pop("deterministic_action_chunks_tensor", None)
        self._collector_module.evaluate_active_sparse_tasks = (
            self._original_predicate
        )

    # -- running --------------------------------------------------------

    def run(self, *, round_index: int) -> _Recording:
        collector = self.world.collector
        # Before the first forward of the round, so a seeded run starts each
        # round from the same generator state rather than inheriting whatever
        # the previous round left behind.
        self.determinism = _apply_determinism(
            torch_seed=self.torch_seed,
            deterministic_kernels=self.deterministic_kernels,
        )
        collector.validate_round(round_index=round_index)
        diverged = int(self.world.backend.pop_nonfinite_world_events())

        if len(self._rows_step) != len(self._rows_eval):
            raise RuntimeError(
                f"Recorded {len(self._rows_step)} plant steps but "
                f"{len(self._rows_eval)} predicate evaluations. They are "
                "called once each per env step in validate_round, so a "
                "mismatch means the loop changed shape and the arrays below "
                "would be misaligned."
            )
        if not self._rows_step:
            raise RuntimeError("The round executed no env steps.")

        def stack(rows: Sequence[Mapping[str, np.ndarray]], key: str) -> np.ndarray:
            return np.stack([row[key] for row in rows], axis=0)

        active_from_step = stack(self._rows_step, "active")
        active_from_eval = stack(self._rows_eval, "active")
        # Independent readings of the same mask, from the two ends of one env
        # step. Equal by construction in production; unequal means one of the
        # two interception points is not where this file claims it is, which
        # would make every per-step array here misindexed.
        if not np.array_equal(active_from_step, active_from_eval):
            raise RuntimeError(
                "The active mask handed to the plant and the one handed to the "
                "success predicate disagree. The recording is misaligned."
            )

        task_state = self.reset.task_state
        catch_release = getattr(
            self.world.collector, "catch_release_dense_reward", None
        )
        per_decision = int(self.world.collector.actions_per_policy_decision)
        decisions = len(self._rows_decision)
        # The plant is stepped once per action in the chunk, so the two
        # counters are locked together. If they drift, the per-decision arrays
        # cannot be indexed from an env step and every demonstration built from
        # them would pair the wrong observation with the wrong action.
        if decisions and decisions * per_decision != len(self._rows_step):
            raise RuntimeError(
                f"Captured {decisions} decisions and {len(self._rows_step)} "
                f"env steps at {per_decision} actions per decision; expected "
                f"{decisions * per_decision}."
            )
        # Per group in the reset, one entry per world here. Guarded rather than
        # assumed: a shape that is already per-world must not be repeated again.
        catalogs = getattr(self.reset, "group_target_catalog_ids", None)
        worlds = int(self.world.layout.worlds_per_rank)
        if catalogs is None:
            catalog_ids = None
        else:
            catalog_ids = _host_int(catalogs).reshape(-1)
            if catalog_ids.shape[0] != worlds:
                catalog_ids = np.repeat(
                    catalog_ids, worlds // max(catalog_ids.shape[0], 1)
                )
        return _Recording(
            target_catalog_ids=catalog_ids,
            states=(
                stack(self._rows_decision, "states")
                if self._rows_decision
                else None
            ),
            priors=(
                stack(self._rows_decision, "priors")
                if self._rows_decision
                else None
            ),
            actions=stack(self._rows_step, "actions"),
            active=active_from_step,
            success=stack(self._rows_eval, "success"),
            terminated=stack(self._rows_eval, "terminated"),
            caught_target=stack(self._rows_eval, "caught_target"),
            ee_xyz=stack(self._rows_eval, "ee_xyz"),
            gripper_opening=stack(self._rows_eval, "gripper_opening"),
            object_xyz=stack(self._rows_eval, "object_xyz"),
            instruction_ids=_host_int(task_state.instruction_ids),
            target_slots=_host_int(task_state.target_slots),
            reference_slots=_host_int(task_state.reference_slots),
            second_reference_slots=_host_int(task_state.second_reference_slots),
            horizons=_host_int(self.reset.horizons),
            initial_target_xyz=_host_float(task_state.initial_target_positions),
            support_surface_z=_host_float(task_state.support_surface_z),
            release_threshold=_host_float(task_state.release_threshold),
            target_rest_height=(
                _host_float(task_state.target_rest_height)
                if task_state.target_rest_height is not None
                else np.zeros(
                    (int(task_state.instruction_ids.shape[0]),),
                    dtype=np.float32,
                )
            ),
            physical_grasp_at_reset=_host_bool(self.reset.physical_grasp),
            instructions=np.asarray(list(self.reset.instructions), dtype="U256"),
            actions_per_decision=int(
                self.world.collector.actions_per_policy_decision
            ),
            round_index=int(round_index),
            diverged_worlds=diverged,
            pick_lift_success_height=float(
                getattr(catch_release, "pick_lift_success_height", 0.05)
            ),
        )


class _DecisionFrameTap:
    """Tee the camera tensors the policy was handed, for a subset of worlds.

    Phase 4 trains LoRA, and LoRA needs a grad-through-VLA forward, which needs
    the frames -- the dataset's 512-wide vision feature is a fixed random
    projection taken under no_grad and cannot stand in for them.

    Three decisions here, each of which is the difference between a usable
    dataset and a large one.

    FRAMES ARE TAPPED ON THE REPLAY, NOT THE HARVEST. The demonstration is the
    SMOOTHED trajectory, so its observations are the smoothed rollout's, and
    those only exist during the replay. Tapping the harvest would pair the
    original run's pictures with the smoothed run's actions. The replay also
    already knows which worlds succeeded, so the buffer can be narrowed to
    them: a full round is 512 worlds x 2 cameras x 240 x 320 x 3 = 236 MB per
    DECISION, which is the difference between 1.9 GB and 0.4 GB on a horizon of
    eight.

    UINT8, SELECTED ON THE GPU BEFORE THE TRANSFER. The backend hands out
    float32 in [0, 1]; keeping that would quadruple the file and the copy for a
    precision no camera delivers. The 1/255 round trip is far below the
    micron-scale physics noise that already flips 6.6% of verdicts, and these
    frames are training inputs, never a replay source.

    ONE CALL PER DECISION, ASSERTED. ``validate_round`` renders once per policy
    decision, so the frame count and the decision count are locked together. If
    they ever drift, every frame would be paired with the wrong action, and the
    dataset would look entirely normal while doing it.
    """

    def __init__(self, backend: Any, worlds: Sequence[int], torch: Any) -> None:
        self.backend = backend
        self.torch = torch
        self.worlds = [int(w) for w in worlds]
        self.overview: list[np.ndarray] = []
        self.wrist: list[np.ndarray] = []
        self._original = backend.render_policy_cameras
        self._index: Any = None

    def __enter__(self) -> "_DecisionFrameTap":
        torch = self.torch

        def to_uint8(camera: Any) -> np.ndarray:
            if self._index is None:
                self._index = torch.as_tensor(
                    self.worlds, dtype=torch.int64, device=camera.device
                )
            picked = camera.index_select(0, self._index)
            picked = picked.permute(0, 2, 3, 1).float() * 255.0
            return (
                picked.round().clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
            )

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            cameras = self._original(*args, **kwargs)
            if self.worlds:
                self.overview.append(to_uint8(cameras.overview))
                self.wrist.append(to_uint8(cameras.wrist))
            return cameras

        self.backend.render_policy_cameras = wrapped
        return self

    def __exit__(self, *exc: Any) -> None:
        if "render_policy_cameras" in vars(self.backend):
            del self.backend.render_policy_cameras
        else:  # pragma: no cover - bound-method backends
            self.backend.render_policy_cameras = self._original

    def stack(self, *, decisions: int) -> dict[str, np.ndarray]:
        """[decisions, worlds, H, W, 3] per camera, aligned to the decisions."""

        if not self.worlds:
            # Nothing was asked for, so nothing was captured and the render
            # count says nothing about the decision count. Return empty rather
            # than accusing the caller of a misalignment it did not cause.
            empty = np.zeros((0, 0, 0, 0, 3), dtype=np.uint8)
            return {
                "overview": empty,
                "wrist": empty.copy(),
                "world_index": np.zeros((0,), dtype=np.int64),
            }
        if len(self.overview) != decisions:
            raise RuntimeError(
                f"Captured {len(self.overview)} camera renders against "
                f"{decisions} recorded decisions. validate_round renders once "
                "per decision, so a mismatch means every frame would be paired "
                "with the wrong action."
            )
        return {
            "overview": np.stack(self.overview, axis=0),
            "wrist": np.stack(self.wrist, axis=0),
            "world_index": np.asarray(self.worlds, dtype=np.int64),
        }


def _host_float(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy().copy()


def _host_int(value: Any) -> np.ndarray:
    return value.detach().to("cpu").numpy().astype(np.int64).copy()


def _host_bool(value: Any) -> np.ndarray:
    import torch

    return value.detach().to(dtype=torch.bool).cpu().numpy().copy()


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _instruction_name(instruction_id: int) -> str:
    if 0 <= int(instruction_id) < len(ACTIVE_INSTRUCTION_TYPES):
        return ACTIVE_INSTRUCTION_TYPES[int(instruction_id)]
    return f"id_{int(instruction_id)}"


def _catalog_name(catalog_id: int) -> str:
    if 0 <= int(catalog_id) < len(ACTIVE_CDPR_CATALOGS):
        return ACTIVE_CDPR_CATALOGS[int(catalog_id)]
    return f"catalog_{int(catalog_id)}"


def _object_mix(
    recording: _Recording, mask: np.ndarray | None = None
) -> dict[str, Any] | None:
    """Attempted versus kept, per object.

    The gap between the two columns is the object selection the success filter
    performs silently. An object that cannot be grasped at all contributes
    attempts and no successes, so a dataset drawn from this pool is narrower
    than the pool it was drawn from.
    """

    if recording.target_catalog_ids is None:
        return None
    kept = recording.episode_success
    if mask is not None:
        kept = kept & mask
    mix: dict[str, Any] = {}
    for catalog_id in sorted(set(recording.target_catalog_ids.tolist())):
        selected = recording.target_catalog_ids == catalog_id
        if mask is not None:
            selected = selected & mask
        attempted = int(selected.sum())
        if not attempted:
            continue
        succeeded = int((selected & kept).sum())
        mix[_catalog_name(catalog_id)] = {
            "attempted": attempted,
            "kept": succeeded,
            "rate": round(succeeded / attempted, 4),
        }
    return mix


def _episode_rows(recording: _Recording) -> list[dict[str, Any]]:
    """One row per world. The unit the dataset is selected on."""

    success = recording.episode_success
    first = recording.first_success_step
    length = recording.episode_length
    # Peak lift above the RESET height, while the object was held. The same two
    # quantities the pick_up predicate multiplies together, kept apart on
    # purpose: reported as measurements, not as a reimplemented predicate.
    grasped = recording.caught_target & (recording.gripper_opening <= 0.94)
    lift = (
        recording.object_xyz[
            :, np.arange(recording.worlds), recording.target_slots, 2
        ]
        - recording.initial_target_xyz[None, :, 2]
    )
    held_lift = np.where(grasped & recording.active, lift, -np.inf)
    peak_held_lift = held_lift.max(axis=0)
    peak_held_lift = np.where(np.isfinite(peak_held_lift), peak_held_lift, 0.0)

    rows: list[dict[str, Any]] = []
    for world in range(recording.worlds):
        rows.append(
            {
                "world": world,
                "instruction": _instruction_name(
                    recording.instruction_ids[world]
                ),
                "success": bool(success[world]),
                "first_success_env_step": int(first[world]),
                "env_steps_active": int(length[world]),
                "horizon_decisions": int(recording.horizons[world]),
                "grasped_at_reset": bool(
                    recording.physical_grasp_at_reset[world]
                ),
                "ever_grasped": bool(
                    (grasped[:, world] & recording.active[:, world]).any()
                ),
                "peak_held_lift_m": round(float(peak_held_lift[world]), 5),
                "instruction_text": str(recording.instructions[world]),
            }
        )
    return rows


def _slice_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-instruction success, with the denominator kept visible.

    Selecting on success selects on alignment for any policy, including a blind
    one, so a dataset slice is not interpretable without the rate it was drawn
    from. Every consumer of this file should carry the source rate forward.
    """

    summary: dict[str, Any] = {}
    for name in sorted({str(row["instruction"]) for row in rows}):
        subset = [row for row in rows if row["instruction"] == name]
        successes = [row for row in subset if row["success"]]
        summary[name] = {
            "episodes": len(subset),
            "successes": len(successes),
            "source_success_rate": (
                round(len(successes) / len(subset), 4) if subset else 0.0
            ),
            "mean_env_steps_to_success": (
                round(
                    float(
                        np.mean(
                            [row["first_success_env_step"] for row in successes]
                        )
                    ),
                    2,
                )
                if successes
                else None
            ),
        }
    return summary


def _pick_up_prefix_report(recording: _Recording) -> dict[str, Any]:
    """Can a placement episode be relabelled as ``pick_up``?

    The brief calls this the strongest relabel in the set. It may instead be a
    gate the reset already satisfies. ``pick_success`` is
    ``grasped & (target_lift >= pick_lift_success_height)`` with ``target_lift``
    measured from ``initial_target_positions`` captured AT RESET -- and
    placement episodes start already grasped. So either the carry lifts the
    object past the threshold, in which case the relabelled trajectories
    contain no grasp acquisition at all and would train pick_up on demos that
    skip the only hard part; or it does not, and the relabel yields nothing.

    Both readings matter and neither is what the brief assumes, so this reports
    the two components rather than a verdict. The relabel itself, if it
    survives this, must go through ``evaluate_active_sparse_tasks`` with
    rewritten instruction ids -- never through the arithmetic here.
    """

    placement = np.isin(
        recording.instruction_ids,
        [
            ACTIVE_INSTRUCTION_TYPES.index("put_into_plate"),
            ACTIVE_INSTRUCTION_TYPES.index("put_into_bowl"),
        ],
    )
    grasped = recording.caught_target & (recording.gripper_opening <= 0.94)
    lift = (
        recording.object_xyz[
            :, np.arange(recording.worlds), recording.target_slots, 2
        ]
        - recording.initial_target_xyz[None, :, 2]
    )
    held_lift = np.where(grasped & recording.active, lift, -np.inf)
    peak = held_lift.max(axis=0)
    peak = np.where(np.isfinite(peak), peak, 0.0)

    threshold = float(recording.pick_lift_success_height)
    subset = placement & recording.episode_success
    return {
        "pick_lift_success_height_m": threshold,
        "placement_episodes": int(placement.sum()),
        "successful_placement_episodes": int(subset.sum()),
        "grasped_at_reset_fraction": round(
            float(recording.physical_grasp_at_reset[placement].mean()), 4
        )
        if placement.any()
        else 0.0,
        "peak_held_lift_m_mean": round(float(peak[subset].mean()), 5)
        if subset.any()
        else None,
        "peak_held_lift_m_p90": round(float(np.percentile(peak[subset], 90)), 5)
        if subset.any()
        else None,
        "would_relabel_as_pick_up": int((peak[subset] >= threshold).sum()),
        "would_relabel_fraction": round(
            float((peak[subset] >= threshold).mean()), 4
        )
        if subset.any()
        else 0.0,
    }


def _divergence(
    first: _Recording, second: _Recording
) -> dict[str, Any]:
    """Trajectory divergence, restricted to steps both runs actually stepped.

    An unmasked max over ``[S, W, 3]`` is not a statement about the plant. A
    world that terminated at different steps in the two runs has a frozen tail
    whose difference is an artefact of the termination step, not of physics,
    and failed episodes are free to wander. Both inflate the number without
    saying anything. This masks to ``active & active`` and reports the
    successful episodes separately, since those are the ones the dataset keeps.
    """

    steps = min(first.actions.shape[0], second.actions.shape[0])
    both = first.active[:steps] & second.active[:steps]
    action_delta = np.abs(first.actions[:steps] - second.actions[:steps])
    ee_delta = np.linalg.norm(
        first.ee_xyz[:steps] - second.ee_xyz[:steps], axis=-1
    )
    kept = first.episode_success & second.episode_success
    both_kept = both & kept[None, :]

    def summarize(mask: np.ndarray, values: np.ndarray) -> Any:
        selected = values[mask]
        return round(float(selected.max()), 8) if selected.size else None

    return {
        "compared_env_steps": int(steps),
        "active_in_both_steps": int(both.sum()),
        "max_abs_action_delta_active": summarize(
            both[..., None].repeat(action_delta.shape[-1], axis=-1),
            action_delta,
        ),
        "max_ee_delta_m_active": summarize(both, ee_delta),
        "mean_ee_delta_m_active": (
            round(float(ee_delta[both].mean()), 8) if both.any() else None
        ),
        "max_ee_delta_m_active_and_kept": summarize(both_kept, ee_delta),
        # The unmasked figure the first version of this tool printed, kept so
        # the two can be compared rather than silently replaced.
        "max_ee_delta_m_unmasked": round(float(ee_delta.max()), 8),
    }


def _first_decision_report(
    first: _Recording, second: _Recording
) -> dict[str, Any]:
    """Does the policy emit the same first action from the same reset?

    This is the discriminator the aggregate deltas cannot give. Env step 0 is
    taken from a reset that is a pure function of its seed, so the two runs
    hand the policy an identical world. If the commanded action still differs
    the nondeterminism is in the policy pipeline -- nondeterministic reduction
    kernels in the SmolVLA forward, or the render feeding it -- and physics is
    downstream of it. If it matches, physics diverged first and the policy is
    only responding.

    The magnitude separates the two cases that matter: ~1e-7 is float noise
    chaotically amplified over a hundred steps, ~1e-2 is a different policy.
    """

    delta = np.abs(first.actions[0] - second.actions[0])
    per_world = delta.max(axis=-1)
    differing = per_world > 0.0
    return {
        "identical": bool(not differing.any()),
        "worlds_differing": int(differing.sum()),
        "worlds": int(per_world.shape[0]),
        "max_abs_delta": float(per_world.max()),
        "median_abs_delta_where_differing": (
            float(np.median(per_world[differing])) if differing.any() else 0.0
        ),
        "verdict": (
            "policy pipeline is deterministic; physics diverges first"
            if not differing.any()
            else (
                "policy differs at step 0 from an identical reset -- the "
                "nondeterminism is upstream of physics"
            )
        ),
    }


def _reset_identity_report(
    first: _Recording, second: _Recording
) -> dict[str, Any]:
    """Are the two runs even the same episodes?

    Everything downstream is void if not. The reset is seeded by
    ``base_seed + rank*1_000_003 + update_index*10_000_019 + round_index*100_003``
    and nothing else, so a mismatch here means the run was launched with a
    different cap, round index or config -- not that the simulator is noisy.
    """

    return {
        "same_instruction_ids": bool(
            np.array_equal(first.instruction_ids, second.instruction_ids)
        ),
        "same_target_slots": bool(
            np.array_equal(first.target_slots, second.target_slots)
        ),
        "same_horizons": bool(np.array_equal(first.horizons, second.horizons)),
        "same_grasp_at_reset": bool(
            np.array_equal(
                first.physical_grasp_at_reset, second.physical_grasp_at_reset
            )
        ),
        "max_initial_target_delta_m": float(
            np.abs(
                first.initial_target_xyz - second.initial_target_xyz
            ).max()
        ),
    }


def _flip_report(first: _Recording, second: _Recording) -> dict[str, Any]:
    """Which episodes changed verdict, and were they marginal?

    A flip concentrated at the end of the budget is an episode that only just
    made it, and a coin-flip verdict on those is a different fact from a policy
    that behaves differently. ``late_success_fraction`` is the success step as a
    fraction of the episode's active length; near 1.0 means it succeeded on
    almost its last available step.
    """

    a, b = first.episode_success, second.episode_success
    only_a = a & ~b
    only_b = ~a & b
    flipped = only_a | only_b

    steps = first.first_success_step.astype(np.float64)
    length = np.maximum(first.episode_length.astype(np.float64), 1.0)
    late = np.where(steps >= 0, steps / length, np.nan)

    by_instruction: dict[str, Any] = {}
    for instruction_id in sorted(set(first.instruction_ids.tolist())):
        mask = first.instruction_ids == instruction_id
        by_instruction[_instruction_name(instruction_id)] = {
            "episodes": int(mask.sum()),
            "success_a": int((a & mask).sum()),
            "success_b": int((b & mask).sum()),
            "flipped": int((flipped & mask).sum()),
            "agreement": round(float((a[mask] == b[mask]).mean()), 5),
        }

    marginal = late[a & flipped]
    robust = late[a & ~flipped]
    return {
        "flipped_total": int(flipped.sum()),
        "won_in_a_only": int(only_a.sum()),
        "won_in_b_only": int(only_b.sum()),
        "agreement": round(float((a == b).mean()), 5),
        "late_success_fraction_flipped": (
            round(float(np.nanmean(marginal)), 4) if marginal.size else None
        ),
        "late_success_fraction_stable": (
            round(float(np.nanmean(robust)), 4) if robust.size else None
        ),
        "by_instruction": by_instruction,
    }


def _compare_report(
    first: _Recording, second: _Recording
) -> dict[str, Any]:
    """The whole forensic, from two npz files. No GPU, no simulator."""

    return {
        "reset_identity": _reset_identity_report(first, second),
        "first_decision": _first_decision_report(first, second),
        "divergence": _divergence(first, second),
        "verdict_flips": _flip_report(first, second),
    }


def _determinism_report(
    first: _Recording, second: _Recording
) -> dict[str, Any]:
    """The null. Two identical runs; anything that differs is the simulator.

    Without this number a replay survival of, say, 0.97 is unreadable -- it
    could be a recording bug or it could be the floor. If this comes back
    anything other than exact agreement, the seeded-replay design does not hold
    and part 3 needs state serialization instead.
    """

    steps = min(first.actions.shape[0], second.actions.shape[0])
    return {
        "env_steps": [int(first.actions.shape[0]), int(second.actions.shape[0])],
        "same_step_count": bool(
            first.actions.shape[0] == second.actions.shape[0]
        ),
        "same_instruction_ids": bool(
            np.array_equal(first.instruction_ids, second.instruction_ids)
        ),
        "same_horizons": bool(np.array_equal(first.horizons, second.horizons)),
        "success_agreement": round(
            float(
                (first.episode_success == second.episode_success).mean()
            ),
            5,
        ),
        "success_rate": [
            round(float(first.episode_success.mean()), 5),
            round(float(second.episode_success.mean()), 5),
        ],
        "actions_bitwise_identical": bool(
            np.array_equal(
                first.actions[:steps], second.actions[:steps]
            )
        ),
        "max_abs_action_delta": float(
            np.abs(
                first.actions[:steps] - second.actions[:steps]
            ).max()
        ),
        # Unmasked, and therefore blunt: it includes failed episodes and the
        # frozen tails of worlds that terminated at different steps. Read
        # `divergence` and `first_decision` instead; this stays for continuity.
        "max_abs_ee_delta_m": float(
            np.abs(first.ee_xyz[:steps] - second.ee_xyz[:steps]).max()
        ),
        "first_decision": _first_decision_report(first, second),
        "divergence": _divergence(first, second),
        "verdict_flips": _flip_report(first, second),
    }


def _replay_report(
    source: _Recording, replay: _Recording
) -> dict[str, Any]:
    """Arm C. Survival is agreement with the recording that produced it."""

    kept = source.episode_success
    survived = kept & replay.episode_success
    steps = min(source.actions.shape[0], replay.actions.shape[0])
    report: dict[str, Any] = {
        "same_instruction_ids": bool(
            np.array_equal(source.instruction_ids, replay.instruction_ids)
        ),
        "same_horizons": bool(
            np.array_equal(source.horizons, replay.horizons)
        ),
        "actions_fed_bitwise_identical": bool(
            np.array_equal(source.actions[:steps], replay.actions[:steps])
        ),
        "recorded_successes": int(kept.sum()),
        "survived": int(survived.sum()),
        "survival_rate": (
            round(float(survived.sum() / kept.sum()), 5) if kept.any() else None
        ),
        "max_abs_ee_delta_m": float(
            np.abs(source.ee_xyz[:steps] - replay.ee_xyz[:steps]).max()
        ),
        "divergence": _divergence(source, replay),
        # A survivor that succeeds at a different step did not reproduce the
        # trajectory; it reached the same verdict by a different route. Under a
        # bitwise replay this must be zero.
        "successes_at_a_different_step": int(
            (
                (source.first_success_step != replay.first_success_step)
                & survived
            ).sum()
        ),
    }
    by_instruction: dict[str, Any] = {}
    for instruction_id in sorted(set(source.instruction_ids.tolist())):
        mask = source.instruction_ids == instruction_id
        kept_here = kept & mask
        if not kept_here.any():
            continue
        by_instruction[_instruction_name(instruction_id)] = {
            "recorded_successes": int(kept_here.sum()),
            "survived": int((survived & mask).sum()),
            "survival_rate": round(
                float((survived & mask).sum() / kept_here.sum()), 5
            ),
        }
    report["by_instruction"] = by_instruction
    return report


def _group_label(recording: _Recording, fallback: str) -> str:
    """Which rung of the recording plan a round came from.

    Prefers the cap stored in the recording. Recordings written before that
    field existed fall back to a caller-supplied label -- in practice the
    input file's parent directory -- which is honest about being provenance
    rather than a measured cap.
    """

    cap = float(recording.start_distance_cap)
    if np.isnan(cap):
        return fallback
    return "cap_inf" if np.isinf(cap) else f"cap_{cap:g}"


SMOOTH_CHANNELS: Mapping[str, tuple[int, ...]] = {
    "xyz": (0, 1, 2),
    "xyz_yaw": (0, 1, 2, 3),
    "all": (0, 1, 2, 3, 4),
}


def _instruction_windows(
    instruction_ids: np.ndarray, *, default: int, overrides: Mapping[str, int]
) -> np.ndarray:
    """Per-world filter width, from a per-instruction override table.

    One global width is the wrong shape for this problem. The success
    tolerances differ by task -- plate 0.091 m, bowl 0.057 m, pick_up roughly
    2 cm of grasp precision -- and the measured survival tracks them: at
    window 5 plate loses nothing at all while bowl is already down to 0.864.
    A width that bowl can survive leaves plate smoothed far less than it could
    be, and a width tuned for plate would gut bowl.

    This is matching filter strength to a known physical tolerance, not
    searching for whatever maximises the survival number.
    """

    widths = np.full(instruction_ids.shape, int(default), dtype=np.int64)
    for name, width in overrides.items():
        if name not in ACTIVE_INSTRUCTION_TYPES:
            raise ValueError(
                f"Unknown instruction {name!r} in the window table. Known: "
                f"{list(ACTIVE_INSTRUCTION_TYPES)}"
            )
        widths[instruction_ids == ACTIVE_INSTRUCTION_TYPES.index(name)] = int(
            width
        )
    return widths


def _smooth_actions(
    actions: np.ndarray,
    active: np.ndarray,
    *,
    method: str,
    window: int,
    alpha: float,
    channels: str,
    per_world_window: np.ndarray | None = None,
) -> np.ndarray:
    """Filter each episode's action sequence along the env-step axis.

    Three things this must not do, each of which would quietly corrupt the
    survival number it exists to produce.

    It must not smooth across episode boundaries. Every world is an independent
    trajectory, so filtering runs per world.

    It must not pull the tail of a live episode toward the dead actions after
    it. A world stops being stepped once it terminates and the policy keeps
    emitting into a frozen world, so only the active steps are filtered and the
    frozen remainder is left exactly as recorded -- it is fed back but moves
    nothing.

    It must not pull the ends toward zero. Edge padding, not zero padding: the
    first actions are the approach and the last are the release, and a filter
    that fades them out would be testing a different trajectory rather than a
    smoother one.

    The gripper channel is excluded by default. It is closer to a discrete
    open/close than to a continuous path, and averaging it delays the grasp and
    softens the release that ``container_ok`` requires -- which would show up as
    a survival loss attributable to the filter's reach rather than to smoothing.
    """

    if method not in {"none", "moving_average", "ema", "median"}:
        raise ValueError(f"Unknown smoothing method {method!r}.")
    if channels not in SMOOTH_CHANNELS:
        raise ValueError(f"Unknown smoothing channels {channels!r}.")
    smoothed = actions.copy()
    if method == "none":
        return smoothed

    columns = list(SMOOTH_CHANNELS[channels])
    if per_world_window is not None and len(per_world_window) != actions.shape[1]:
        raise ValueError("per_world_window must be one width per world.")

    for world in range(actions.shape[1]):
        live = active[:, world]
        count = int(live.sum())
        if count < 2:
            continue
        width = max(
            1,
            int(window if per_world_window is None else per_world_window[world]),
        )
        if width % 2 == 0:
            width += 1  # centred filters need an odd window
        pad = width // 2
        block = smoothed[live, world]
        segment = block[:, columns].astype(np.float64)

        if method == "ema":
            # Causal on purpose: a controller can only filter samples it has
            # already seen, so a centred filter would flatter the method by
            # using the future.
            filtered = np.empty_like(segment)
            accumulator = segment[0].copy()
            for step in range(segment.shape[0]):
                accumulator = alpha * segment[step] + (1.0 - alpha) * accumulator
                filtered[step] = accumulator
        else:
            padded = np.pad(segment, ((pad, pad), (0, 0)), mode="edge")
            if method == "moving_average":
                kernel = np.ones(width) / float(width)
                filtered = np.stack(
                    [
                        np.convolve(padded[:, index], kernel, mode="valid")
                        for index in range(segment.shape[1])
                    ],
                    axis=1,
                )
            else:
                windows = np.lib.stride_tricks.sliding_window_view(
                    padded, width, axis=0
                )
                filtered = np.median(windows, axis=-1)

        block[:, columns] = filtered.astype(actions.dtype)
        smoothed[live, world] = block
    return smoothed


def _smoothness(actions: np.ndarray, active: np.ndarray) -> dict[str, float]:
    """How jagged the commanded sequences are, over live steps only.

    Reported next to survival because survival on its own is gameable: the
    identity filter survives perfectly and smooths nothing. A method is only
    interesting where both columns move.
    """

    both = active[1:] & active[:-1]
    if not both.any():
        return {"mean_abs_step_delta": 0.0, "max_abs_step_delta": 0.0}
    delta = np.abs(actions[1:] - actions[:-1])
    selected = delta[both]
    return {
        "mean_abs_step_delta": round(float(selected.mean()), 6),
        "max_abs_step_delta": round(float(selected.max()), 6),
    }


def _row_quota_mask(
    instruction_ids: np.ndarray,
    episode_uids: np.ndarray,
    *,
    rows_per_instruction: int,
    seed: int,
) -> np.ndarray:
    """Row mask holding each instruction to roughly ``rows_per_instruction``.

    In ROWS, not episodes. A decision is what the loss sees, and episode
    lengths differ by a factor of four across these families -- measured on the
    first mixed build, 21.2 decisions per move_to episode against 5.2-5.4 for
    the placement pair. A quota of 300 episodes each therefore hands move_to
    two thirds of the gradient while every count in the report reads as
    balanced, which is the kind of skew that only shows up as a policy that
    over-restored one family at another's expense.

    Whole episodes, in random order, until the budget is met. Never a partial
    episode: ``_episode_split`` holds out whole episodes so that consecutive
    decisions sharing an observation history cannot straddle the train/val
    line, and a quota that cut mid-episode would reintroduce exactly that.

    The episode that crosses the budget is kept rather than dropped, so a slice
    overshoots by up to one episode. Stopping short instead would bias the bank
    toward short episodes, which for move_to means the starts nearest the goal.
    """

    keep = np.zeros(instruction_ids.shape, dtype=bool)
    budget = int(rows_per_instruction)
    if budget <= 0:
        return ~keep
    rng = np.random.default_rng(int(seed))
    for instruction_id in np.unique(instruction_ids):
        rows = np.flatnonzero(instruction_ids == instruction_id)
        uids = episode_uids[rows]
        taken = 0
        chosen: list[str] = []
        for uid in rng.permutation(np.unique(uids)):
            if taken >= budget:
                break
            chosen.append(uid)
            taken += int((uids == uid).sum())
        keep[rows[np.isin(uids, chosen)]] = True
    return keep


def _build_dataset(
    recordings: Sequence[_Recording],
    labels: Sequence[str] | None = None,
    keys: Sequence[str] | None = None,
    *,
    rows_per_instruction: int = 0,
    quota_seed: int = 0,
    relabel_rules: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Successful episodes only, truncated at the step that made them succeed.

    One row per policy DECISION, because that is the unit the policy emits: a
    state, the prior it was conditioned on, and the chunk of actions the plant
    then executed from it.

    Two truncations, both load-bearing:

    An episode terminates on success, so every decision after ``first_success``
    ran against a frozen world. Those are not demonstrations of anything and
    are dropped entirely.

    The decision that *contains* the success is kept, but the world usually
    terminated partway through its chunk -- so the actions after that point
    were also executed against a frozen world. They are kept in place and
    marked in ``action_mask`` rather than dropped, so the chunk keeps its shape
    and the loss can ignore the dead tail. Dropping them would silently shorten
    chunks and misalign the action head.
    """

    names = list(labels or [f"input_{i}" for i in range(len(recordings))])
    # The uid key must be unique per SOURCE FILE, which the rung label is not:
    # a whole harvest replays into one output directory, so every file there
    # shares a parent name, and two families harvested at the same cap share a
    # rung. A colliding uid merges distinct episodes, which puts parts of the
    # same trajectory on both sides of the episode split.
    unique = list(keys or [f"input_{i}" for i in range(len(recordings))])
    if len(names) != len(recordings) or len(unique) != len(recordings):
        raise ValueError("labels and keys must be one per recording.")
    if len(set(unique)) != len(unique):
        raise ValueError(
            f"Dataset input keys are not unique: {sorted(unique)}. Episode "
            "ids built from them would merge distinct episodes."
        )
    paired = [
        (rec, _group_label(rec, name), key)
        for rec, name, key in zip(recordings, names, unique)
        if rec.has_observations
    ]
    usable = [rec for rec, _, _ in paired]
    if not usable:
        raise ValueError(
            "No recording carries observations. Re-record with a build that "
            "captures `states` and `priors`; verdict-only recordings cannot "
            "train anything."
        )

    states: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    instruction_ids: list[int] = []
    instruction_texts: list[str] = []
    episode_uids: list[str] = []
    decision_indices: list[int] = []
    groups: list[str] = []
    episodes_kept = 0

    for rec, group, key in paired:
        per_decision = int(rec.actions_per_decision)
        success = rec.episode_success
        first = rec.first_success_step
        for world in range(rec.worlds):
            if not success[world]:
                continue
            episodes_kept += 1
            last_decision = int(first[world]) // per_decision
            # The label, not just the rung. Two families harvested at the same
            # cap produce the same group ("cap_0.03" for both sil_harvest_0.03
            # and sil_pickup_0.03), so a rung-based uid collides across them --
            # and a colliding uid silently merges two real episodes into one,
            # which puts half of each on both sides of the episode split.
            uid = f"{key}/r{int(rec.round_index)}w{world}"
            for decision in range(last_decision + 1):
                start = decision * per_decision
                stop = start + per_decision
                states.append(rec.states[decision, world])
                priors.append(rec.priors[decision, world])
                actions.append(rec.actions[start:stop, world])
                masks.append(rec.active[start:stop, world])
                instruction_ids.append(int(rec.instruction_ids[world]))
                instruction_texts.append(str(rec.instructions[world]))
                episode_uids.append(uid)
                decision_indices.append(decision)
                groups.append(group)

    if not states:
        raise ValueError("No successful episodes to build a dataset from.")

    dataset = {
        "state": np.stack(states).astype(np.float32),
        "prior": np.stack(priors).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "action_mask": np.stack(masks).astype(bool),
        "instruction_id": np.asarray(instruction_ids, dtype=np.int64),
        "instruction_text": np.asarray(instruction_texts, dtype="U256"),
        "episode_uid": np.asarray(episode_uids, dtype="U128"),
        "decision_index": np.asarray(decision_indices, dtype=np.int64),
        # Carried per row so a consumer can filter or reweight by rung without
        # re-reading the recordings. The near rungs are cheap and easy; the far
        # ones are scarce and biased, and pooling them silently is the whole
        # selection-bias trap.
        "source_group": np.asarray(groups, dtype="U32"),
    }

    # BEFORE the quota, so a relabelled episode counts against the budget of
    # the instruction it now carries rather than the one it was recorded as.
    # Applied after the arrays are assembled rather than per row, because the
    # split is per EPISODE and needs every row's uid in hand.
    relabel_counts: dict[str, int] = {}
    if relabel_rules:
        (
            dataset["instruction_id"],
            dataset["instruction_text"],
            relabel_counts,
        ) = apply_instruction_relabel(
            dataset["instruction_id"],
            dataset["instruction_text"],
            dataset["episode_uid"],
            relabel_rules,
        )

    quota: dict[str, Any] | None = None
    if int(rows_per_instruction) > 0:
        keep = _row_quota_mask(
            dataset["instruction_id"],
            dataset["episode_uid"],
            rows_per_instruction=int(rows_per_instruction),
            seed=int(quota_seed),
        )
        before = int(dataset["state"].shape[0])
        dataset = {key: value[keep] for key, value in dataset.items()}
        episodes_kept = int(len(np.unique(dataset["episode_uid"])))
        quota = {
            "rows_per_instruction": int(rows_per_instruction),
            "seed": int(quota_seed),
            "decisions_before": before,
            "decisions_after": int(dataset["state"].shape[0]),
        }

    # The source success rate travels with every slice. A dataset drawn from a
    # 0.06 source is mostly luck; one drawn from 0.93 is mostly skill, and the
    # two must never be pooled without saying so.
    def slice_stats(
        instruction_id: int, subset: Sequence[tuple[_Recording, str]]
    ) -> dict[str, Any]:
        source_episodes = 0
        source_successes = 0
        for rec, *_ in subset:
            mask = rec.instruction_ids == instruction_id
            source_episodes += int(mask.sum())
            source_successes += int((rec.episode_success & mask).sum())
        return {
            "episodes": source_successes,
            "source_episodes": source_episodes,
            "source_success_rate": (
                round(source_successes / source_episodes, 4)
                if source_episodes
                else 0.0
            ),
        }

    per_slice: dict[str, Any] = {}
    all_groups = sorted({group for _, group, _ in paired})
    for instruction_id in sorted(set(instruction_ids)):
        name = _instruction_name(instruction_id)
        entry = slice_stats(instruction_id, paired)
        rows_for_name = dataset["instruction_id"] == instruction_id
        entry["decisions"] = int(rows_for_name.sum())
        # Distinct from "episodes", which counts what the RECORDINGS held. A
        # quota makes the two diverge, and reporting only the source count
        # would describe a dataset that was never written.
        entry["episodes_kept"] = int(
            len(np.unique(dataset["episode_uid"][rows_for_name]))
        )
        entry["decisions_per_episode"] = (
            round(entry["decisions"] / entry["episodes_kept"], 2)
            if entry["episodes_kept"]
            else None
        )
        # Pooling across rungs reports a rate that describes no collection
        # condition that ever ran. Placement is ~0.93 at the near rung and
        # ~0.55 at the far one; the average of those is not a fact about
        # either.
        by_group: dict[str, Any] = {}
        for group in all_groups:
            subset = [pair for pair in paired if pair[1] == group]
            group_entry = slice_stats(instruction_id, subset)
            if not group_entry["source_episodes"]:
                continue
            rows = (dataset["instruction_id"] == instruction_id) & (
                dataset["source_group"] == group
            )
            group_entry["decisions"] = int(rows.sum())
            by_group[group] = group_entry
        entry["by_group"] = by_group
        mixes: dict[str, dict[str, int]] = {}
        for rec, *_ in paired:
            mix = _object_mix(rec, rec.instruction_ids == instruction_id)
            for catalog, counts in (mix or {}).items():
                totals = mixes.setdefault(
                    catalog, {"attempted": 0, "kept": 0}
                )
                totals["attempted"] += counts["attempted"]
                totals["kept"] += counts["kept"]
        entry["by_object"] = {
            catalog: {
                **counts,
                "rate": round(counts["kept"] / counts["attempted"], 4)
                if counts["attempted"]
                else 0.0,
            }
            for catalog, counts in sorted(mixes.items())
        } or None
        per_slice[name] = entry

    stats = {
        "relabelled": relabel_counts,
        "recordings": len(usable),
        "recordings_without_observations": len(recordings) - len(usable),
        "episodes_kept": episodes_kept,
        "quota": quota,
        "decisions": int(dataset["state"].shape[0]),
        "state_dim": int(dataset["state"].shape[-1]),
        "actions_per_decision": int(dataset["action"].shape[1]),
        "dead_action_fraction": round(
            float(1.0 - dataset["action_mask"].mean()), 5
        ),
        "by_instruction": per_slice,
    }
    return dataset, stats


def _run_sharded(
    *,
    raw_argv: Sequence[str],
    devices: Sequence[str],
    first_round: int,
    rounds: int,
    output: Path,
) -> int:
    """Record one contiguous slice of the round range per device, in parallel.

    Each shard is this same script re-invoked with a single --device, so the
    recording path is byte-for-byte the one a serial harvest takes and there is
    no second implementation to keep correct. What the parent adds is the
    split, the live prefixed log, and the pooled summary.

    Every shard writes into the SAME --output directory. That is safe because
    record files are named by round index while walking, and the shards own
    disjoint round ranges -- so the harvest looks exactly like a serial one and
    --mode dataset needs no special case to read it.
    """

    import subprocess
    import threading

    shards = plan_device_shards(first_round, rounds, devices)
    if not shards:
        print("[sil][shard] nothing to run", flush=True)
        return 0
    if len(shards) < len(devices):
        print(
            f"[sil][shard] {rounds} rounds over {len(devices)} devices: "
            f"using {len(shards)} of them, the rest would record nothing.",
            flush=True,
        )
    inherited = strip_argv_flags(
        raw_argv,
        ("--device", "--devices", "--round-index", "--rounds",
         "--summary-suffix"),
    )

    processes: list[tuple[str, str, subprocess.Popen[str]]] = []
    readers: list[threading.Thread] = []
    for device, start, count in shards:
        tag = device.replace(":", "").replace("/", "")
        suffix = f"_{tag}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *inherited,
            "--device", device,
            "--round-index", str(start),
            "--rounds", str(count),
            "--summary-suffix", suffix,
        ]
        print(
            f"[sil][shard] {device}: rounds {start}..{start + count - 1} "
            f"-> summary{suffix}.json",
            flush=True,
        )
        log_path = output / f"log{suffix}.txt"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def pump(
            stream: Any, label: str, path: Path
        ) -> None:
            # Prefixed so two shards interleaving on one terminal stay
            # readable, and teed to a file so a long harvest is still
            # inspectable after the scrollback is gone.
            with path.open("w", encoding="utf-8") as handle:
                for line in stream:
                    handle.write(line)
                    handle.flush()
                    print(f"[{label}] {line.rstrip()}", flush=True)

        reader = threading.Thread(
            target=pump,
            args=(process.stdout, device, log_path),
            daemon=True,
        )
        reader.start()
        readers.append(reader)
        processes.append((device, suffix, process))

    failures: list[str] = []
    for device, _suffix, process in processes:
        code = process.wait()
        if code != 0:
            failures.append(f"{device} (exit {code})")
    for reader in readers:
        reader.join(timeout=30.0)

    # Every round the shards owned must have left a file. This is cheap and it
    # is the guard that was missing when two children computed the same file
    # index: they wrote the same path concurrently, one round was lost and the
    # survivor failed its CRC on the next read, hours later at replay. Naming
    # is fixed (see record_file_index) but the check stays, because the failure
    # it catches is silent at the only moment it could still be cheap to fix.
    missing = [
        f"record_{index:02d}.npz"
        for _device, start, count in shards
        for index in range(start, start + count)
        if not (output / f"record_{index:02d}.npz").is_file()
    ]
    if missing and not failures:
        failures.append(
            f"{len(missing)} round(s) left no recording: {', '.join(missing)}"
        )

    summaries: list[Mapping[str, Any]] = []
    for _device, suffix, _process in processes:
        path = output / f"summary{suffix}.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    pooled = _merge_shard_summaries(summaries)
    merged = {
        "mode": "record",
        "sharded": True,
        "devices": [device for device, _s, _p in processes],
        "round_index": int(first_round),
        "rounds": int(rounds),
        "shards": [
            {"device": device, "first_round": start, "rounds": count}
            for device, start, count in shards
        ],
        "shard_summaries": [
            f"summary{suffix}.json" for _d, suffix, _p in processes
        ],
        "by_instruction": pooled,
        "failed_shards": failures,
    }
    (output / "summary.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8"
    )
    for name, stats in sorted(pooled.items()):
        print(
            f"[sil][shard] pooled {name}: {stats['successes']}/"
            f"{stats['episodes']} = {stats['source_success_rate']:.3f}",
            flush=True,
        )
    print(f"[sil] wrote {output / 'summary.json'}", flush=True)
    if failures:
        # Loud and non-zero: a half-finished harvest that reports success is
        # how a dataset silently becomes one shard.
        print(
            f"[sil][shard] FAILED: {', '.join(failures)}. The rounds those "
            "shards owned were not recorded.",
            flush=True,
        )
        return 1
    return 0


def record_file_index(*, run_index: int, round_index: int, repeat: int) -> int:
    """Which number a record file carries: the round, unless repeating.

    ``--repeat`` pins one round index and re-runs it, so only the run number
    separates those files. Every other invocation walks the round index, and
    then the ROUND has to name the file -- including a one-round invocation,
    which is what a device shard receives.

    That last clause is the whole reason this is a function. It used to read
    "round index if rounds > 1", which is true for a serial harvest and false
    for the shards it was written to support: the parent hands each child
    ``--rounds 1`` when there is one round per device, so both children
    computed run index 0, both wrote record_00.npz, and the two processes
    raced. The surviving file failed its CRC and the other round was simply
    gone. Sharding four rounds over two devices hid it, because each child then
    got --rounds 2 and took the walking branch.
    """

    if int(repeat) > 1:
        return int(run_index)
    return int(round_index)


def plan_device_shards(
    first_round: int,
    rounds: int,
    devices: Sequence[str],
) -> list[tuple[str, int, int]]:
    """Split a contiguous round range across devices, as (device, start, n).

    Rounds are independent by construction -- ``round_index`` is part of the
    reset seed and nothing carries between rounds -- so sharding them is exact
    rather than an approximation. The split is CONTIGUOUS, not round-robin, so
    every shard owns a run of consecutive indices and the record files it
    writes are named by an index no other shard can produce.

    Devices beyond the round count get no shard rather than an empty one: a
    child process that recorded nothing would still pay the full model load,
    build a world, and write a summary describing zero episodes.

    The remainder goes to the earliest devices, so with 5 rounds on 2 devices
    the split is 3/2 and never 2/2 with a round quietly dropped.
    """

    rounds = max(int(rounds), 0)
    names = [str(device).strip() for device in devices if str(device).strip()]
    if not names or rounds <= 0:
        return []
    usable = min(len(names), rounds)
    base, extra = divmod(rounds, usable)
    shards: list[tuple[str, int, int]] = []
    start = int(first_round)
    for position in range(usable):
        count = base + (1 if position < extra else 0)
        shards.append((names[position], start, count))
        start += count
    return shards


def strip_argv_flags(argv: Sequence[str], flags: Sequence[str]) -> list[str]:
    """Drop ``--flag value`` and ``--flag=value`` pairs from an argv list.

    Used to rebuild a child command line from the parent's own, so a shard
    inherits every argument that was actually passed -- including ones added to
    this tool after the sharding was written -- rather than a hand-maintained
    subset that silently drops the flag someone needs.

    Only long options are handled, which is all this tool defines.
    """

    wanted = {str(flag) for flag in flags}
    out: list[str] = []
    skip_value = False
    for token in argv:
        if skip_value:
            skip_value = False
            continue
        text = str(token)
        head = text.split("=", 1)[0]
        if head in wanted:
            # "--flag=value" carries its value; "--flag value" does not.
            skip_value = "=" not in text
            continue
        out.append(text)
    return out


def _merge_shard_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pool per-instruction counts across shard summaries.

    Rates are recomputed from summed counts rather than averaged: shards can
    hold different numbers of rounds when the split has a remainder, and the
    mean of two rates over unequal denominators is not the pooled rate.
    """

    totals: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        for key, entry in summary.items():
            if not str(key).startswith("run_") or not isinstance(entry, Mapping):
                continue
            for name, stats in (entry.get("by_instruction") or {}).items():
                bucket = totals.setdefault(
                    str(name), {"successes": 0, "episodes": 0}
                )
                bucket["successes"] += int(stats.get("successes", 0))
                bucket["episodes"] += int(stats.get("episodes", 0))
    for bucket in totals.values():
        episodes = int(bucket["episodes"])
        bucket["source_success_rate"] = (
            round(bucket["successes"] / episodes, 5) if episodes else 0.0
        )
    return totals


_RELABEL_RECEPTACLE = {
    "put_into_plate": "plate",
    "put_into_bowl": "bowl",
}


def parse_relabel_rules(specs: Sequence[str]) -> dict[str, list[str]]:
    """``pick_up=put_into_plate,put_into_bowl`` -> {src: [dst, ...]}."""

    rules: dict[str, list[str]] = {}
    for spec in specs or ():
        source, _, targets = str(spec).partition("=")
        source = source.strip()
        names = [name.strip() for name in targets.split(",") if name.strip()]
        if not source or not names:
            raise ValueError(
                f"--relabel-instruction expects SRC=DST[,DST2], got {spec!r}"
            )
        for name in (source, *names):
            if name not in ACTIVE_INSTRUCTION_TYPES:
                raise ValueError(f"Unknown instruction in --relabel-instruction: {name}")
        rules[source] = names
    return rules


def relabel_instruction_text(text: str, source: str, target: str) -> str:
    """Rewrite one episode's prompt for its new instruction.

    The prompt is the ONLY channel a relabel travels down. The residual's state
    is proprioception plus a vision feature and carries no instruction, and
    sil_refresh_priors recomputes the SmolVLA prior from
    ``dataset["instruction_text"]`` -- so the text here becomes the prior the
    SFT trains against, and a malformed one trains against a malformed prompt.

    Generated to match what sample_instruction would have written for the
    target, so relabelled rows are indistinguishable in the prompt
    distribution: "pick up apple" -> "put apple into plate".
    """

    raw = str(text).strip()
    prefix = {
        "pick_up": "pick up ",
        "move_to_object": "move to ",
        "grab_object": "grab ",
    }.get(source)
    obj = raw[len(prefix):].strip() if prefix and raw.startswith(prefix) else ""
    if not obj:
        # Never invent a name. An unparsed prompt keeps the generic wording the
        # task itself falls back to when the reference object is unnamed.
        obj = "object"
    receptacle = _RELABEL_RECEPTACLE.get(target)
    if receptacle is None:
        return raw
    return f"put {obj} into {receptacle}"


def apply_instruction_relabel(
    instruction_id: np.ndarray,
    instruction_text: np.ndarray,
    episode_uid: np.ndarray,
    rules: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Retarget whole episodes onto a different instruction.

    Why this exists: a `put_into` episode has always begun with the object
    already between the fingers, so nothing in the bank shows what to do when
    the instruction is `put_into` and the object is still on the desk. A grasp
    recorded in a scene that contains the receptacle IS that missing prefix --
    the actions are real and were executed under the right physics, and only
    the label is wrong.

    Split per EPISODE, not per row, and by a hash of the uid rather than a
    counter: every decision of one trajectory must carry the same instruction
    or the prompt changes mid-episode, and a counter would make the assignment
    depend on which files were globbed in which order.
    """

    names = list(ACTIVE_INSTRUCTION_TYPES)
    new_id = np.asarray(instruction_id).copy()
    new_text = np.asarray(instruction_text).copy()
    counts: dict[str, int] = {}
    if not rules:
        return new_id, new_text, counts
    by_episode: dict[str, str] = {}
    for row in range(new_id.shape[0]):
        source = names[int(new_id[row])]
        targets = rules.get(source)
        if not targets:
            continue
        uid = str(episode_uid[row])
        target = by_episode.get(uid)
        if target is None:
            # Stable across runs and across input ordering.
            digest = hashlib.sha1(uid.encode("utf-8")).digest()
            target = list(targets)[digest[0] % len(targets)]
            by_episode[uid] = target
        new_id[row] = int(names.index(target))
        new_text[row] = relabel_instruction_text(
            str(instruction_text[row]), source, target
        )
        counts[f"{source}->{target}"] = counts.get(f"{source}->{target}", 0) + 1
    return new_id, new_text, counts


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Required for record and replay; unused by compare.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Required for record and replay; unused by compare.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("record", "replay", "compare", "dataset", "oracle"),
        default="record",
        help=(
            "record runs the deterministic policy and writes the executed "
            "actions; replay feeds a recording's actions back through the same "
            "reset and rescores with the production predicate; compare is a "
            "pure-numpy forensic over two existing npz files -- no GPU, no "
            "simulator, no checkpoint -- answering whether the two runs are "
            "the same episodes, whether the policy emitted the same first "
            "action from the same reset, and which verdicts flipped."
        ),
    )
    parser.add_argument(
        "--actions",
        type=Path,
        default=None,
        help="Recording npz. Required for replay and compare.",
    )
    parser.add_argument(
        "--against",
        type=Path,
        default=None,
        help="Second npz to compare --actions against. Required for compare.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Recording runs. 2 gives the determinism null: two runs that "
            "differ in nothing, so any disagreement is the simulator's own "
            "and is the noise floor under every survival number downstream."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help=(
            "Harvest N rounds at consecutive --round-index values. Different "
            "round indices are different starts, which is where dataset "
            "diversity comes from. Distinct from --repeat, which re-runs ONE "
            "round index to measure noise; passing both above 1 is ambiguous "
            "and is rejected."
        ),
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[],
        help="Recording npz files to build a dataset from (--mode dataset).",
    )
    parser.add_argument(
        "--rows-per-instruction",
        type=int,
        default=0,
        help=(
            "--mode dataset: hold each instruction to roughly this many "
            "DECISIONS by dropping whole episodes at random. 0 keeps "
            "everything. The unit is decisions and not episodes because the "
            "SFT loss is per decision and episode lengths differ ~4x across "
            "families, so an episode quota that reads as balanced is not."
        ),
    )
    parser.add_argument(
        "--relabel-instruction",
        action="append",
        default=[],
        metavar="SRC=DST[,DST2]",
        help=(
            "--mode dataset: retarget whole episodes onto another instruction, "
            "e.g. pick_up=put_into_plate,put_into_bowl. A put_into episode has "
            "always started with the object already held, so nothing in the "
            "bank shows what to do when the instruction is put_into and the "
            "object is still on the desk; a grasp recorded in a scene that "
            "CONTAINS the receptacle is that missing prefix, with real actions "
            "and only the label wrong. Multiple targets split evenly across "
            "episodes by a hash of the episode uid, so the assignment is "
            "stable across runs and independent of input ordering. The prompt "
            "is rewritten to match what the target instruction would have "
            "generated, because the prompt is the only channel the label "
            "travels down: the residual's state carries no instruction and "
            "sil_refresh_priors recomputes the prior from instruction_text. "
            "Applied BEFORE the row quota."
        ),
    )
    parser.add_argument(
        "--quota-seed",
        type=int,
        default=0,
        help="Seed for the --rows-per-instruction subsample.",
    )
    parser.add_argument(
        "--round-index",
        type=int,
        default=None,
        help=(
            "Reset round index. Defaults to 0 when recording and to the "
            "recording's own index when replaying -- replaying against a "
            "different index feeds one episode set's actions into another's "
            "starts, which the reset-identity check then rejects."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--devices",
        default="",
        help=(
            "Comma-separated devices to shard --rounds across, e.g. "
            "'cuda:0,cuda:1'. Record mode only. The parent process builds no "
            "world of its own: it re-invokes this script once per device with "
            "a contiguous slice of the round range and waits, so each shard is "
            "the ordinary single-device path and the only new machinery is the "
            "split. Rounds are independent -- round_index is part of the reset "
            "seed and nothing carries between them -- so this is exact, not an "
            "approximation of a serial harvest. Overrides --device."
        ),
    )
    parser.add_argument(
        "--summary-suffix",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--smolvla-microbatch", type=int, default=256)
    parser.add_argument(
        "--start-distance-cap",
        type=float,
        default=None,
        help=(
            "Override the checkpoint's approach-curriculum cap (m). This is "
            "the recording plan's rung. Must match between a recording and "
            "its replay, or the resets differ and the comparison is void."
        ),
    )
    parser.add_argument(
        "--horizon-decisions",
        type=int,
        default=0,
        help=(
            "Override the rollout budget in decisions (0 keeps the coupled "
            "one). Smoothing tends to shrink command magnitude, so a smoothed "
            "episode can fail for want of budget rather than for want of "
            "skill; this separates the two."
        ),
    )
    parser.add_argument(
        "--metadata-override",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="Same contract as the probe's flag of the same name.",
    )
    parser.add_argument(
        "--video-worlds",
        type=int,
        default=0,
        help=(
            "Replay only: write an mp4 for the first N SURVIVING episodes. "
            "Frames are teed off the backend's own render call, so the video "
            "is what the policy was shown along the smoothed rollout rather "
            "than a re-render. Watching is the fastest way to tell a smoothed "
            "demonstration from a smoothed stumble."
        ),
    )
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument(
        "--record-frames",
        action="store_true",
        help=(
            "Replay only: write frames_<stem>.npz alongside the replay -- the "
            "uint8 camera tensors the policy was handed, for the source's "
            "successful worlds. This is what LoRA fine-tuning needs and the "
            "512-wide vision feature cannot supply: that feature is a fixed "
            "random projection taken under no_grad, so no gradient reaches the "
            "vision tower through it."
        ),
    )
    parser.add_argument(
        "--frame-worlds",
        type=int,
        default=0,
        help=(
            "Cap on how many successful worlds keep frames (0 = all of them). "
            "A round is ~236 MB per decision at 512 worlds and two cameras, so "
            "this is the knob that decides whether a harvest fits on disk."
        ),
    )
    parser.add_argument(
        "--smooth",
        choices=("none", "moving_average", "ema", "median"),
        default="none",
        help=(
            "Filter the recorded actions before replaying them. `none` is the "
            "control and must be run first: the controller re-anchors its "
            "target to the measured EE pose every step, so a replay is not "
            "guaranteed to reproduce its recording, and a survival rate for a "
            "real filter means nothing until the identity filter has been "
            "shown to survive at 1.0."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Window for moving_average and median. Forced odd.",
    )
    parser.add_argument(
        "--smooth-window-by-instruction",
        nargs="+",
        default=[],
        metavar="NAME=WIDTH",
        help=(
            "Per-instruction window overriding --smooth-window, e.g. "
            "put_into_plate=13 put_into_bowl=7. The success tolerances differ "
            "by task (plate 0.091 m, bowl 0.057 m, pick_up ~2 cm), and the "
            "measured survival tracks them, so one global width either "
            "under-smooths the forgiving task or breaks the tight one."
        ),
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="EMA weight on the current sample. 1.0 is a no-op.",
    )
    parser.add_argument(
        "--smooth-channels",
        choices=tuple(SMOOTH_CHANNELS),
        default="xyz",
        help=(
            "Which action channels to filter. The gripper is excluded by "
            "default: it is closer to a discrete open/close than a continuous "
            "path, and averaging it delays the grasp and softens the release "
            "the container predicate requires, which would read as a survival "
            "loss caused by smoothing rather than by the filter's reach."
        ),
    )
    parser.add_argument(
        "--seed-torch",
        type=int,
        default=None,
        help=(
            "Seed torch's global RNG before each round. This is the experiment "
            "that separates a stochastic prior from unordered bf16 arithmetic: "
            "if the null passes with this and failed without it, the forward "
            "was drawing noise from the global generator. Omit to reproduce "
            "the trainer's own behaviour, which does not seed here."
        ),
    )
    parser.add_argument(
        "--deterministic-kernels",
        action="store_true",
        help=(
            "Ask torch for deterministic algorithms (warn_only, so an "
            "uncovered op degrades coverage instead of killing the run). The "
            "follow-up when --seed-torch alone does not close the null. Fully "
            "deterministic cuBLAS also needs CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "set in the environment before launch."
        ),
    )
    args = parser.parse_args(argv)
    # The child command lines are rebuilt from this, so a shard inherits every
    # argument that was actually passed rather than a subset kept in sync by
    # hand. Captured before any validation mutates the parsed values.
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    if args.repeat < 1:
        parser.error("--repeat must be at least 1.")
    if args.rounds < 1:
        parser.error("--rounds must be at least 1.")
    if args.repeat > 1 and args.rounds > 1:
        parser.error(
            "--repeat re-runs one round index to measure noise; --rounds "
            "walks consecutive round indices to harvest. Combining them makes "
            "the output ambiguous. Pick one."
        )
    # Parsed and checked BEFORE the dataset and compare branches return, so
    # "--devices with --mode dataset" is refused rather than silently ignored
    # on the way past.
    devices = [
        name.strip() for name in str(args.devices).split(",") if name.strip()
    ]
    if devices:
        if args.mode not in ("record", "oracle"):
            parser.error(
                "--devices shards a round range across GPUs and only --mode "
                f"record and --mode oracle walk one; --mode {args.mode} does "
                "not. Replay or build a dataset with --device."
            )
        if int(args.repeat) > 1:
            parser.error(
                "--devices shards --rounds across devices; --repeat pins one "
                "round index to measure noise, so there is nothing to shard "
                "and the two runs of the null would land on different GPUs."
            )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.mode == "dataset":
        if not args.inputs:
            parser.error("--mode dataset needs --inputs.")
        try:
            relabel_rules = parse_relabel_rules(args.relabel_instruction)
        except ValueError as exc:
            parser.error(str(exc))
        resolved = [path.expanduser().resolve() for path in args.inputs]
        recordings = [_Recording.from_npz(path) for path in resolved]
        # Provenance for recordings written before start_distance_cap was
        # stored. The harvest writes one directory per rung, so the parent
        # name is the rung -- but it is a filesystem convention, not a
        # measurement, and _group_label prefers the stored cap when present.
        dataset, stats = _build_dataset(
            recordings,
            [path.parent.name for path in resolved],
            # Directory AND stem. Neither alone is unique across both layouts
            # this tool produces: a harvest writes record_00..NN into one
            # directory PER RUNG, so stems repeat across rungs; replays of a
            # whole harvest land in ONE directory, so parents repeat there.
            [f"{path.parent.name}/{path.stem}" for path in resolved],
            rows_per_instruction=int(args.rows_per_instruction),
            quota_seed=int(args.quota_seed),
            relabel_rules=relabel_rules,
        )
        np.savez_compressed(output / "demonstrations.npz", **dataset)
        stats["inputs"] = [str(path) for path in args.inputs]
        (output / "dataset.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[sil][dataset] {stats['episodes_kept']} episodes -> "
            f"{stats['decisions']} decisions, state_dim "
            f"{stats['state_dim']}, dead actions "
            f"{stats['dead_action_fraction']:.3f}",
            flush=True,
        )
        for name, entry in stats["by_instruction"].items():
            print(
                f"[sil][dataset] {name}: {entry['episodes_kept']} episodes "
                f"of {entry['episodes']} available "
                f"({entry['decisions']} decisions, "
                f"{entry['decisions_per_episode']} per episode) pooled source "
                f"rate {entry['source_success_rate']:.3f}",
                flush=True,
            )
            for group, group_entry in entry["by_group"].items():
                print(
                    f"[sil][dataset]     {group}: "
                    f"{group_entry['episodes']}/"
                    f"{group_entry['source_episodes']} = "
                    f"{group_entry['source_success_rate']:.3f}",
                    flush=True,
                )
        for pair, rows in sorted(stats.get("relabelled", {}).items()):
            print(f"[sil][dataset] relabelled {pair}: {rows} rows", flush=True)
        print(f"[sil] wrote {output / 'demonstrations.npz'}", flush=True)
        return 0

    # compare reads two npz files and builds nothing. Kept ahead of the world
    # build so the forensic can be re-run on a laptop against artefacts the GPU
    # box already wrote, without a checkpoint or a CUDA device in sight.
    if args.mode == "compare":
        if args.actions is None or args.against is None:
            parser.error("--mode compare needs --actions and --against.")
        first = _Recording.from_npz(args.actions.expanduser().resolve())
        second = _Recording.from_npz(args.against.expanduser().resolve())
        report = _compare_report(first, second)
        report["a"] = str(args.actions)
        report["b"] = str(args.against)
        (output / "compare.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        identity = report["reset_identity"]
        decision = report["first_decision"]
        print(
            "[sil][compare] same_episodes="
            f"{identity['same_instruction_ids'] and identity['same_horizons']}",
            flush=True,
        )
        print(
            f"[sil][compare] step-0 actions identical={decision['identical']} "
            f"differing_worlds={decision['worlds_differing']}/"
            f"{decision['worlds']} max_delta={decision['max_abs_delta']:.3e}",
            flush=True,
        )
        print(f"[sil][compare] {decision['verdict']}", flush=True)
        print(
            "[sil][compare] ee delta over active steps: "
            f"max={report['divergence']['max_ee_delta_m_active']} "
            f"mean={report['divergence']['mean_ee_delta_m_active']} m",
            flush=True,
        )
        print(
            f"[sil][compare] flipped {report['verdict_flips']['flipped_total']}"
            f" agreement={report['verdict_flips']['agreement']}",
            flush=True,
        )
        print(f"[sil] wrote {output / 'compare.json'}", flush=True)
        return 0

    if args.checkpoint is None or args.config is None:
        # oracle still needs both: the checkpoint supplies the states and
        # priors the demonstration is conditioned on, even though its actions
        # are discarded, and the config supplies the scene and the reward.
        parser.error(f"--mode {args.mode} needs --checkpoint and --config.")
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if not config_path.is_file():
        parser.error(f"Config does not exist: {config_path}")
    if args.mode == "replay" and args.actions is None:
        parser.error("--mode replay needs --actions.")

    if len(devices) > 1:
        return _run_sharded(
            raw_argv=raw_argv,
            devices=devices,
            first_round=(
                0 if args.round_index is None else int(args.round_index)
            ),
            rounds=int(args.rounds),
            output=output,
        )
    if len(devices) == 1:
        # One device named is just that device. Fall through to the ordinary
        # path rather than spawning a child to do what this process can do.
        args.device = devices[0]

    start_cap = args.start_distance_cap
    if start_cap is not None and (start_cap <= 0.0 or start_cap == float("inf")):
        start_cap = float("inf")

    world = _build_world(
        checkpoint=checkpoint,
        config_path=config_path,
        device_str=str(args.device),
        worlds=int(args.worlds),
        group_size=int(args.group_size),
        microbatch=int(args.smolvla_microbatch),
        # The policy stays loaded in replay too, even though its output is
        # discarded, so a record arm and a replay arm differ in exactly one
        # respect: the five numbers handed to the plant.
        load_policy=True,
        run_dir=output,
        start_distance_cap=start_cap,
        metadata_overrides=list(args.metadata_override or []),
    )

    # None means "unspecified", which record resolves to 0 and replay resolves
    # to the recording's own index.
    first_round = 0 if args.round_index is None else int(args.round_index)
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "mode": str(args.mode),
        "worlds": int(args.worlds),
        "round_index": first_round,
        "start_distance_cap": (
            None if start_cap is None else float(start_cap)
        ),
        "horizon_decisions_override": int(args.horizon_decisions),
    }

    summary_name = f"summary{args.summary_suffix}.json"
    if args.mode in ("record", "oracle"):
        recordings: list[_Recording] = []
        runs = max(int(args.repeat), int(args.rounds))
        # --repeat pins the round index (the null); --rounds walks it (harvest).
        # Exactly one of them is above 1, enforced above.
        walking = int(args.rounds) > 1
        for index in range(runs):
            round_index = first_round + (index if walking else 0)
            print(
                f"[sil] recording run {index} (round_index {round_index})",
                flush=True,
            )
            with _RoundRecorder(
                world,
                action_source=(
                    _OracleActionSource() if args.mode == "oracle" else None
                ),
                horizon_override=int(args.horizon_decisions),
                torch_seed=args.seed_torch,
                deterministic_kernels=bool(args.deterministic_kernels),
            ) as recorder:
                recording = recorder.run(round_index=round_index)
            # NaN when no override was passed: the round then ran at whatever
            # cap the checkpoint earned, which this tool does not know and must
            # not invent a number for.
            recording.start_distance_cap = (
                float("nan") if start_cap is None else float(start_cap)
            )
            summary["determinism"] = recorder.determinism
            # See record_file_index: the round names the file unless --repeat
            # pinned it. For the default harvest the two agree and the name is
            # unchanged from before sharding existed.
            file_index = record_file_index(
                run_index=index,
                round_index=round_index,
                repeat=int(args.repeat),
            )
            path = output / f"record_{file_index:02d}.npz"
            recording.to_npz(path)
            recordings.append(recording)
            rows = _episode_rows(recording)
            _write_csv(output / f"episodes_{file_index:02d}.csv", rows)
            slices = _slice_summary(rows)
            summary[f"run_{file_index:02d}"] = {
                "npz": str(path),
                "env_steps": int(recording.actions.shape[0]),
                "actions_per_decision": recording.actions_per_decision,
                "diverged_worlds": recording.diverged_worlds,
                "overall_success_rate": round(
                    float(recording.episode_success.mean()), 5
                ),
                "by_instruction": slices,
                "object_mix": _object_mix(recording),
            }
            for name, stats in slices.items():
                print(
                    f"[sil][run {file_index}] {name}: "
                    f"{stats['successes']}/{stats['episodes']} "
                    f"= {stats['source_success_rate']:.3f}",
                    flush=True,
                )

        summary["pick_up_prefix"] = _pick_up_prefix_report(recordings[0])
        # Only when the round index was held fixed. Two DIFFERENT round indices
        # are different episodes, and reporting their disagreement as a
        # determinism null would be a control that is not a null -- it would
        # read as simulator noise while measuring the reset distribution.
        if len(recordings) >= 2 and not walking:
            null = _determinism_report(recordings[0], recordings[1])
            summary["determinism_null"] = null
            print(
                f"[sil][null] success_agreement="
                f"{null['success_agreement']:.5f} "
                f"actions_identical={null['actions_bitwise_identical']} "
                f"max_ee_delta={null['max_abs_ee_delta_m']:.3e} m",
                flush=True,
            )
    else:
        source_path = args.actions.expanduser().resolve()
        source = _Recording.from_npz(source_path)
        # The reset is a pure function of its seed, and round_index is part of
        # that seed. Replaying a recording against a different round index
        # feeds one episode set's actions into another's starts -- the actions
        # are valid, the world is not, and the survival rate that comes out is
        # a number about nothing. It defaults to the recording's own index
        # rather than to zero, because a default of zero is silently correct
        # for record_00 and silently wrong for every other round.
        replay_round = (
            int(args.round_index)
            if args.round_index is not None
            else int(source.round_index)
        )
        if replay_round != int(source.round_index):
            print(
                f"[sil][replay] WARNING: replaying a recording made at "
                f"round_index {source.round_index} against round_index "
                f"{replay_round}. These are different episodes.",
                flush=True,
            )
        window_table: dict[str, int] = {}
        for override in args.smooth_window_by_instruction or ():
            name, _, raw = str(override).partition("=")
            if not name or not raw:
                parser.error(
                    f"--smooth-window-by-instruction expects NAME=WIDTH, "
                    f"got {override!r}"
                )
            window_table[name] = int(raw)
        try:
            per_world_window = (
                _instruction_windows(
                    source.instruction_ids,
                    default=int(args.smooth_window),
                    overrides=window_table,
                )
                if window_table
                else None
            )
        except ValueError as error:
            parser.error(str(error))
        played = _smooth_actions(
            source.actions,
            source.active,
            method=str(args.smooth),
            window=int(args.smooth_window),
            alpha=float(args.smooth_alpha),
            channels=str(args.smooth_channels),
            per_world_window=per_world_window,
        )
        before = _smoothness(source.actions, source.active)
        after = _smoothness(played, source.active)
        smoothing = {
            "method": str(args.smooth),
            "window": int(args.smooth_window),
            "window_by_instruction": dict(window_table),
            "alpha": float(args.smooth_alpha),
            "channels": str(args.smooth_channels),
            # Per instruction, so a report can say what each slice was filtered
            # with rather than quoting a global width that half the rows never
            # saw.
            "step_delta_by_instruction": {
                _instruction_name(instruction_id): {
                    "before": _smoothness(
                        source.actions[:, source.instruction_ids == instruction_id],
                        source.active[:, source.instruction_ids == instruction_id],
                    )["mean_abs_step_delta"],
                    "after": _smoothness(
                        played[:, source.instruction_ids == instruction_id],
                        source.active[:, source.instruction_ids == instruction_id],
                    )["mean_abs_step_delta"],
                }
                for instruction_id in sorted(
                    set(source.instruction_ids.tolist())
                )
            },
            "step_delta_before": before["mean_abs_step_delta"],
            "step_delta_after": after["mean_abs_step_delta"],
            # Survival on its own is gameable -- the identity filter survives
            # perfectly and smooths nothing -- so the reduction it bought is
            # published beside it. A method is only interesting where both move.
            "step_delta_reduction": (
                round(
                    1.0
                    - after["mean_abs_step_delta"]
                    / before["mean_abs_step_delta"],
                    5,
                )
                if before["mean_abs_step_delta"]
                else 0.0
            ),
            "actions_changed": bool(
                not np.array_equal(played, source.actions)
            ),
        }
        print(
            f"[sil] replaying {source.actions.shape[0]} env steps "
            f"from {args.actions} (smooth={args.smooth}, "
            f"step delta {before['mean_abs_step_delta']:.5f} -> "
            f"{after['mean_abs_step_delta']:.5f})",
            flush=True,
        )
        # Which worlds to film has to be decided before the rollout, and the
        # ones worth filming are the ones that survive -- which is not known
        # until afterwards. The source's successes are the candidate set: a
        # world that failed before smoothing cannot survive it, so filming
        # those is guaranteed waste, and the survivors are a subset of these.
        filmed: list[int] = []
        if int(args.video_worlds) > 0:
            filmed = [
                int(w)
                for w in np.flatnonzero(source.episode_success)[
                    : int(args.video_worlds)
                ]
            ]
        # The worlds whose FRAMES are kept for the dataset: the source's
        # successes, capped. A world that failed before smoothing cannot
        # survive it, so anything outside this set is guaranteed waste; the
        # survivors are a subset and are filtered again at dataset build.
        frame_worlds: list[int] = []
        if bool(args.record_frames):
            candidates = np.flatnonzero(source.episode_success)
            limit = int(args.frame_worlds)
            frame_worlds = [
                int(w) for w in (candidates[:limit] if limit > 0 else candidates)
            ]
            if not frame_worlds:
                raise SystemExit(
                    "--record-frames was asked for but the source recording "
                    "has no successful episodes to keep frames for."
                )
        with _RoundRecorder(
            world,
            playback=played,
            horizon_override=int(args.horizon_decisions),
            torch_seed=args.seed_torch,
            deterministic_kernels=bool(args.deterministic_kernels),
        ) as recorder:
            # Both taps wrap the same method, so they nest rather than compete:
            # the inner one installs over the outer and calls through it.
            frame_tap = (
                _DecisionFrameTap(world.backend, frame_worlds, world.torch)
                if frame_worlds
                else None
            )
            if filmed:
                from tools.audit.success_episode_videos import _FrameTap

                with contextlib.ExitStack() as stack:
                    tap = stack.enter_context(
                        _FrameTap(world.backend, filmed, True)
                    )
                    if frame_tap is not None:
                        stack.enter_context(frame_tap)
                    replay = recorder.run(round_index=replay_round)
                    captured = {w: list(f) for w, f in tap.frames.items()}
            else:
                captured = {}
                if frame_tap is not None:
                    with frame_tap:
                        replay = recorder.run(round_index=replay_round)
                else:
                    replay = recorder.run(round_index=replay_round)
        summary["determinism"] = recorder.determinism
        # Prefer the cap actually requested, so a replay of a recording written
        # before the field existed still carries its rung forward to the
        # dataset instead of falling back to a directory name.
        replay.start_distance_cap = (
            float(start_cap)
            if start_cap is not None
            else source.start_distance_cap
        )
        identity = _reset_identity_report(source, replay)
        if not (
            identity["same_instruction_ids"] and identity["same_horizons"]
        ):
            raise SystemExit(
                "The replay drew a different reset than the recording it "
                "replays: instruction ids or horizons differ. Its survival "
                "rate would describe no episode that exists. Check that "
                "--start-distance-cap and --round-index match the recording "
                f"(round_index {source.round_index}, cap "
                f"{source.start_distance_cap})."
            )
        # Named after the source's RUNG and stem, not the stem alone. Every
        # rung numbers its rounds record_00..NN, so a stem-only name collides
        # across rungs exactly as the fixed name collided across rounds: the
        # last rung overwrites the rest and the dataset silently becomes one
        # rung while presenting itself as the ladder.
        stem = f"{source_path.parent.name}_{source_path.stem}"
        replay_path = output / f"replay_{stem}.npz"
        replay.to_npz(replay_path)
        _write_csv(output / f"episodes_{stem}.csv", _episode_rows(replay))
        summary["replay_npz"] = str(replay_path)
        if frame_tap is not None:
            decisions = (
                0 if replay.states is None else int(replay.states.shape[0])
            )
            payload = frame_tap.stack(decisions=decisions)
            # Stored so the SFT side can index this file without touching a
            # single picture -- the arrays are gigabytes and all it needs is
            # how many decisions they hold.
            payload["decisions"] = np.asarray(int(decisions), dtype=np.int64)
            payload["round_index"] = np.asarray(
                int(replay.round_index), dtype=np.int64
            )
            payload["start_distance_cap"] = np.asarray(
                float(replay.start_distance_cap), dtype=np.float64
            )
            frames_path = output / f"frames_{stem}.npz"
            np.savez_compressed(frames_path, **payload)
            megabytes = frames_path.stat().st_size / (1024.0 * 1024.0)
            summary["frames_npz"] = str(frames_path)
            summary["frames"] = {
                "worlds": int(payload["world_index"].shape[0]),
                "decisions": decisions,
                "height": int(payload["overview"].shape[2]),
                "width": int(payload["overview"].shape[3]),
                "megabytes": round(megabytes, 1),
            }
            print(
                f"[sil][replay] frames {payload['overview'].shape[1]} worlds "
                f"x {decisions} decisions x 2 cameras -> {megabytes:.1f} MB "
                f"({frames_path.name})",
                flush=True,
            )
        report = _replay_report(source, replay)
        summary["source"] = str(args.actions)
        summary["smoothing"] = smoothing
        summary["replay"] = report
        summary["replay_diverged_worlds"] = replay.diverged_worlds
        summary_name = f"summary_{stem}.json"
        print(
            f"[sil][replay] survived {report['survived']}/"
            f"{report['recorded_successes']} "
            f"rate={report['survival_rate']} "
            f"max_ee_delta={report['max_abs_ee_delta_m']:.3e} m",
            flush=True,
        )
        for name, entry in report["by_instruction"].items():
            print(
                f"[sil][replay]     {name}: {entry['survived']}/"
                f"{entry['recorded_successes']} = {entry['survival_rate']}",
                flush=True,
            )
        if captured:
            from tools.audit.success_episode_videos import _Mp4

            videos = output / "videos"
            videos.mkdir(parents=True, exist_ok=True)
            survived_mask = source.episode_success & replay.episode_success
            written = 0
            for filmed_world, frames in sorted(captured.items()):
                if not frames:
                    continue
                # Name the file with the verdict. A dropped episode is the more
                # informative thing to watch -- it shows what the filter broke.
                verdict = (
                    "kept" if survived_mask[filmed_world] else "dropped"
                )
                instruction = _instruction_name(
                    replay.instruction_ids[filmed_world]
                )
                path = (
                    videos
                    / f"{stem}_w{filmed_world:03d}_{instruction}_{verdict}.mp4"
                )
                writer = _Mp4(
                    path,
                    fps=float(args.video_fps),
                    height=int(frames[0].shape[0]),
                    width=int(frames[0].shape[1]),
                )
                for frame in frames:
                    writer.write(frame)
                writer.close()
                written += 1
            print(
                f"[sil][replay] wrote {written} videos to {videos}",
                flush=True,
            )

    (output / summary_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[sil] wrote {output / summary_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
