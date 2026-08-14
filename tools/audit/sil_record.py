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
import csv  # noqa: E402
import json  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (  # noqa: E402
    ACTIVE_INSTRUCTION_TYPES,
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
            name: getattr(self, name)
            for name in (
                "actions", "active", "success", "terminated", "caught_target",
                "ee_xyz", "gripper_opening", "object_xyz",
                "instruction_ids", "target_slots", "reference_slots",
                "second_reference_slots", "horizons", "initial_target_xyz",
                "support_surface_z", "release_threshold", "target_rest_height",
                "physical_grasp_at_reset", "instructions",
            )
        }
        payload["actions_per_decision"] = np.int64(self.actions_per_decision)
        payload["round_index"] = np.int64(self.round_index)
        payload["diverged_worlds"] = np.int64(self.diverged_worlds)
        payload["pick_lift_success_height"] = np.float64(
            self.pick_lift_success_height
        )
        np.savez_compressed(path, **payload)

    @classmethod
    def from_npz(cls, path: Path) -> "_Recording":
        with np.load(path, allow_pickle=False) as data:
            fields = {key: data[key] for key in data.files}
        return cls(
            **{
                key: value
                for key, value in fields.items()
                if key
                not in {
                    "actions_per_decision", "round_index", "diverged_worlds",
                    "pick_lift_success_height",
                }
            },
            actions_per_decision=int(fields["actions_per_decision"]),
            round_index=int(fields["round_index"]),
            diverged_worlds=int(fields["diverged_worlds"]),
            pick_lift_success_height=float(fields["pick_lift_success_height"]),
        )


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
        horizon_override: int = 0,
        torch_seed: int | None = None,
        deterministic_kernels: bool = False,
    ) -> None:
        self.world = world
        self.playback = playback
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
        self._original_step: Any = None
        self._original_reset: Any = None
        self._original_predicate: Any = None
        self._owned_step = False
        self._owned_reset = False
        self._collector_module: Any = None

    # -- installation ---------------------------------------------------

    def __enter__(self) -> "_RoundRecorder":
        import rl_vla_bootstrapping.policy.mjwarp_rank_local_collector as module

        torch = self.world.torch
        backend = self.world.backend
        resetter = self.world.collector.resetter
        self._collector_module = module

        self._original_step = backend.step
        self._original_reset = resetter.reset
        self._original_predicate = module.evaluate_active_sparse_tasks
        self._owned_step = "step" in vars(backend)
        self._owned_reset = "reset" in vars(resetter)

        def patched_reset(**kwargs: Any) -> Any:
            reset = self._original_reset(**kwargs)
            if self.horizon_override:
                reset.horizons.fill_(self.horizon_override)
            self.reset = reset
            self.env_step = 0
            self._rows_step.clear()
            self._rows_eval.clear()
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
            self._rows_step.append(
                {
                    "actions": _host_float(actions),
                    "active": _host_bool(active_mask),
                }
            )
            self.env_step += 1
            return self._original_step(actions, active_mask)

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
            return result

        backend.step = patched_step
        resetter.reset = patched_reset
        module.evaluate_active_sparse_tasks = patched_predicate
        return self

    def __exit__(self, *exc: Any) -> None:
        backend = self.world.backend
        resetter = self.world.collector.resetter
        if self._owned_step:
            backend.step = self._original_step
        else:
            vars(backend).pop("step", None)
        if self._owned_reset:
            resetter.reset = self._original_reset
        else:
            vars(resetter).pop("reset", None)
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
        return _Recording(
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
        choices=("record", "replay", "compare"),
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
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
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

    if args.repeat < 1:
        parser.error("--repeat must be at least 1.")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

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
        parser.error(f"--mode {args.mode} needs --checkpoint and --config.")
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if not config_path.is_file():
        parser.error(f"Config does not exist: {config_path}")
    if args.mode == "replay" and args.actions is None:
        parser.error("--mode replay needs --actions.")

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

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "mode": str(args.mode),
        "worlds": int(args.worlds),
        "round_index": int(args.round_index),
        "start_distance_cap": (
            None if start_cap is None else float(start_cap)
        ),
        "horizon_decisions_override": int(args.horizon_decisions),
    }

    if args.mode == "record":
        recordings: list[_Recording] = []
        for index in range(int(args.repeat)):
            print(f"[sil] recording run {index}", flush=True)
            with _RoundRecorder(
                world,
                horizon_override=int(args.horizon_decisions),
                torch_seed=args.seed_torch,
                deterministic_kernels=bool(args.deterministic_kernels),
            ) as recorder:
                recording = recorder.run(round_index=int(args.round_index))
            summary["determinism"] = recorder.determinism
            path = output / f"record_{index:02d}.npz"
            recording.to_npz(path)
            recordings.append(recording)
            rows = _episode_rows(recording)
            _write_csv(output / f"episodes_{index:02d}.csv", rows)
            slices = _slice_summary(rows)
            summary[f"run_{index:02d}"] = {
                "npz": str(path),
                "env_steps": int(recording.actions.shape[0]),
                "actions_per_decision": recording.actions_per_decision,
                "diverged_worlds": recording.diverged_worlds,
                "overall_success_rate": round(
                    float(recording.episode_success.mean()), 5
                ),
                "by_instruction": slices,
            }
            for name, stats in slices.items():
                print(
                    f"[sil][run {index}] {name}: "
                    f"{stats['successes']}/{stats['episodes']} "
                    f"= {stats['source_success_rate']:.3f}",
                    flush=True,
                )

        summary["pick_up_prefix"] = _pick_up_prefix_report(recordings[0])
        if len(recordings) >= 2:
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
        source = _Recording.from_npz(args.actions.expanduser().resolve())
        print(
            f"[sil] replaying {source.actions.shape[0]} env steps "
            f"from {args.actions}",
            flush=True,
        )
        with _RoundRecorder(
            world,
            playback=source.actions,
            horizon_override=int(args.horizon_decisions),
            torch_seed=args.seed_torch,
            deterministic_kernels=bool(args.deterministic_kernels),
        ) as recorder:
            replay = recorder.run(round_index=int(args.round_index))
        summary["determinism"] = recorder.determinism
        replay.to_npz(output / "replay.npz")
        _write_csv(output / "episodes_replay.csv", _episode_rows(replay))
        report = _replay_report(source, replay)
        summary["source"] = str(args.actions)
        summary["replay"] = report
        summary["replay_diverged_worlds"] = replay.diverged_worlds
        print(
            f"[sil][replay] survived {report['survived']}/"
            f"{report['recorded_successes']} "
            f"rate={report['survival_rate']} "
            f"max_ee_delta={report['max_abs_ee_delta_m']:.3e} m",
            flush=True,
        )

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[sil] wrote {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
