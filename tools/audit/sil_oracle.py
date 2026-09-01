#!/usr/bin/env python3
"""Drive the scripted oracle over a whole batch of worlds.

The composed `put_into` -- object on the desk, grasp it, carry it, release it --
cannot be seeded from any policy we have. Measured: the best single policy
scores 0.011 (bowl) and 0.005 (plate) on it, and relabelled free-scene grasps
miss the join, with only ~20% ending inside the distance placement demos cover.
So the demonstrations have to come from a script.

That script already exists. `render_cdpr_task_reference_episodes.py` defines the
phase chain and, for an ungrasped placement start, returns pick_up's approach
and close followed by the placement carry -- eight phases, the whole task. What
it does not do is produce many episodes or write them in bank format: it drives
two identical clone worlds and renders videos for inspection.

This wraps that same oracle for a batch. Nothing about the control law is
reimplemented here -- `oracle_phases` and `oracle_action` are imported from the
script, so a fix to the reference motion reaches the demonstrations and the two
can never disagree. What this adds is per-world phase state and a loop.

Per-world Python, not vectorised, on purpose. The goal/gripper/done callables
are closures over scalars and would each need rewriting to vectorise, which is a
second implementation of exactly the thing that must not drift. At 2048 worlds
and 104 env steps the loop costs a few seconds against a rollout that costs
minutes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Any, Sequence  # noqa: E402

import numpy as np  # noqa: E402

_ORACLE_SCRIPT = _ROOT / "scripts" / "render_cdpr_task_reference_episodes.py"


def load_oracle_module() -> Any:
    """Import the reference harness as a module.

    By path, because it lives in scripts/ and is not a package. Registered in
    sys.modules before execution: it defines dataclasses, and dataclasses
    resolve their own module out of sys.modules while being built.
    """

    existing = sys.modules.get("_cdpr_reference_oracle")
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        "_cdpr_reference_oracle", _ORACLE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the oracle from {_ORACLE_SCRIPT}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cdpr_reference_oracle"] = module
    spec.loader.exec_module(module)
    return module


class BatchedOracle:
    """One scripted episode per world, advanced together.

    Every argument is per-world and plain numpy, so the phase machine can be
    exercised without a simulator -- which is the only part of this that can be
    got wrong quietly. A phase that never reports done leaves the episode
    stalled in it for the whole horizon and produces a demonstration of
    hovering; a phase that reports done immediately skips the motion it was
    supposed to contribute. Neither raises.
    """

    def __init__(
        self,
        *,
        instruction_types: Sequence[str],
        starts_grasped: Sequence[bool],
        instruction_texts: Sequence[str],
        target_slots: Sequence[int],
        reference_slots: Sequence[int],
        target_catalogs: Sequence[str],
        fitted_openings: Sequence[float],
        release_openings: Sequence[float],
        grasp_height_offset: float,
        release_heights: Sequence[float],
        support_surface_z: Sequence[float],
        lift_success_height: float,
        action_step_xyz: float,
        action_step_gripper: float,
    ) -> None:
        oracle = load_oracle_module()
        self._oracle = oracle
        self.action_step_xyz = float(action_step_xyz)
        self.action_step_gripper = float(action_step_gripper)
        self.worlds = len(instruction_types)
        self.contexts: list[Any] = []
        self.phases: list[tuple[Any, ...]] = []
        self.phase_index = np.zeros(self.worlds, dtype=np.int64)
        self.phase_steps = np.zeros(self.worlds, dtype=np.int64)
        # Which worlds the oracle has no phase chain for -- move_to, the
        # relation tasks. They are driven to a zero action rather than dropped,
        # so the batch shape never depends on the instruction mix.
        self.supported = np.zeros(self.worlds, dtype=bool)
        for world in range(self.worlds):
            name = str(instruction_types[world])
            context = oracle.EpisodeContext(
                instruction_type=name,
                instruction_text=str(instruction_texts[world]),
                target_slot=int(target_slots[world]),
                reference_slot=max(int(reference_slots[world]), 0),
                target_catalog=str(target_catalogs[world]),
                fitted_opening=float(fitted_openings[world]),
                release_opening=float(release_openings[world]),
                grasp_height_offset=float(grasp_height_offset),
                release_height=float(release_heights[world]),
                support_surface_z=float(support_surface_z[world]),
                lift_success_height=float(lift_success_height),
            )
            self.contexts.append(context)
            try:
                chain = oracle.oracle_phases(
                    name, starts_grasped=bool(starts_grasped[world])
                )
                self.supported[world] = True
            except ValueError:
                chain = ()
            self.phases.append(tuple(chain))

    def phase_names(self) -> list[str]:
        """The phase each world is currently executing, for diagnostics."""

        out: list[str] = []
        for world in range(self.worlds):
            chain = self.phases[world]
            if not chain:
                out.append("unsupported")
                continue
            out.append(chain[int(self.phase_index[world])].name)
        return out

    def actions(
        self,
        *,
        ee: np.ndarray,
        ee_velocity: np.ndarray,
        ee_yaw: np.ndarray,
        measured_gripper: np.ndarray,
        commanded_gripper: np.ndarray,
        object_positions: np.ndarray,
        physical_grasp: np.ndarray,
        initial_target_z: np.ndarray,
    ) -> np.ndarray:
        """One action per world, in the trainer's normalised [-1, 1] space."""

        out = np.zeros((self.worlds, 5), dtype=np.float32)
        for world in range(self.worlds):
            chain = self.phases[world]
            if not chain:
                continue
            context = self.contexts[world]
            target_slot = int(context.target_slot)
            positions = np.asarray(object_positions[world], dtype=np.float64)
            observation = {
                "ee": np.asarray(ee[world], dtype=np.float64),
                "ee_velocity": np.asarray(ee_velocity[world], dtype=np.float64),
                "physical_grasp": bool(physical_grasp[world]),
                "ee_yaw": float(ee_yaw[world]),
                "measured_gripper": float(measured_gripper[world]),
                "commanded_gripper": float(commanded_gripper[world]),
                "object_positions": positions,
                "target_lift": max(
                    0.0,
                    float(positions[target_slot][2])
                    - float(initial_target_z[world]),
                ),
            }
            index = int(self.phase_index[world])
            steps = int(self.phase_steps[world])
            # `while`, not `if`: a phase whose exit condition is already true at
            # entry must not cost a step, or a chain of eight phases spends
            # eight steps doing nothing. The reference harness advances the same
            # way and the two must stay identical.
            while index < len(chain) - 1 and chain[index].done(
                context, observation, steps
            ):
                index += 1
                steps = 0
            phase = chain[index]
            out[world] = self._oracle.oracle_action(
                context,
                observation,
                phase,
                self.action_step_xyz,
                self.action_step_gripper,
            ).astype(np.float32)
            self.phase_index[world] = index
            self.phase_steps[world] = steps + 1
        return out
