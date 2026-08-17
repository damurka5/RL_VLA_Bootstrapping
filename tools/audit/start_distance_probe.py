#!/usr/bin/env python3
"""P0/M1 -- does the approach cap move the start the policy actually gets?

The phase-4 loop is triggered by "the cap went up". That trigger is worth
exactly as much as the link between the number the trainer logs
(``curriculum/start_max_goal_distance_m/{name}``) and the distance the resetter
realizes. Phase 2 spent five million steps on a cap that climbed to its ceiling
while success stayed at 0, so the link is not assumed here -- it is measured,
from the reset itself, per instruction, per rung.

What makes this cheap: the start geometry is pure torch inside
``BatchedReverseFrontierResetter.reset``. It needs no physics, no renderer and
no policy. ``--backend fake`` therefore runs the real resetter against a stub
that records the poses it is handed, on CPU, in seconds -- which is the whole
measurement for "what did the resetter command". ``--backend mjwarp`` runs the
real backend and reads the pose back through ``low_dim_observations``, which
additionally catches anything the simulator does to the commanded start.

Read both. They answer different questions and only the pair rules out the
failure where the reset is correct and the sim moves it.

The goal is not reimplemented here. A placement episode is rewarded on
gripper->receptacle while it already holds its target, and pick_up on
gripper->object; getting that backwards reads gripper slop instead of approach
(it scored -0.40 once, see phase-3 report §9.7). The probe calls
``goal_slots_for_reset`` -- the same function the collector's ``_goal_slots``
delegates to -- so there is one definition, not two.

Pre-registered verdict, so the run cannot be read after the fact:

* PASS requires, for every instruction, (a) the rung medians strictly
  increasing, (b) ``over_cap_fraction <= 0.01`` on every rung, and (c) at least
  one rung whose realized median differs from its neighbour by more than 20% of
  the rung gap.
* FAIL on (a) or (c) means the cap is decorative and the loop's trigger
  measures nothing. Most likely cause: ``random_workspace_gripper_start`` is
  false, so the whole cap block (mjwarp_rank_local_collector.py:1579) never
  executes.
* FAIL on (b) means the cap is a soft suggestion, and the "не дальше
  последнего достигнутого cap" guarantee the mini-dataset rests on is not one.

``ee_source`` is reported alongside because it is the thing that explains a
failure. The capped ``ee_group`` is overwritten later for several start
populations -- caught placement, pre-grasped and aligned pick_up
(:1988/:2003/:2026) -- and only the caught-placement path feeds the capped
value back in (:1881-1888, the held object is moved TO the gripper). A rung
whose starts are mostly ``aligned`` is not measuring the approach curriculum at
all.

Usage::

    # geometry as the resetter commands it -- no GPU, runs anywhere
    python tools/audit/start_distance_probe.py \\
      --config configs/examples/cdpr_smolvla_phase4_placement_loop.yaml \\
      --output runs/p0_start_distance --strict

    # the start after the simulator has applied it
    RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab \\
      python tools/audit/start_distance_probe.py --backend mjwarp \\
      --config configs/examples/cdpr_smolvla_phase4_placement_loop.yaml \\
      --output runs/p0_start_distance_mjwarp --strict
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# The stub backend
# --------------------------------------------------------------------------


def build_recording_backend(torch: Any, *, object_slots: int = 4) -> Any:
    """A backend that only records the poses the resetter hands it.

    Everything the reset writes goes through ``set_free_body_poses`` /
    ``set_end_effector_poses``, so recording those two is a complete reading of
    the commanded start. No physics runs, which is the point: this isolates the
    resetter's arithmetic from anything the simulator might do to it afterwards.
    """

    class RecordingBackend:
        def __init__(self) -> None:
            self.torch = torch
            self.device = torch.device("cpu")
            self.object_body_ids = torch.arange(
                int(object_slots), dtype=torch.int64
            )
            self.object_positions: Any = None
            self.object_quaternions: Any = None
            self.ee_positions: Any = None
            self.ee_yaw: Any = None
            self.gripper_openings: Any = None

        def reset_worlds(self, _worlds: Any) -> None:
            return None

        def set_object_catalogs(self, _catalogs: Any) -> None:
            return None

        def set_free_body_poses(
            self, _body_ids: Any, positions: Any, quaternions: Any
        ) -> None:
            self.object_positions = positions.clone()
            self.object_quaternions = quaternions.clone()

        def set_end_effector_poses(self, positions: Any, yaw: Any) -> None:
            self.ee_positions = positions.clone()
            self.ee_yaw = yaw.clone()

        def set_gripper_openings(self, openings: Any) -> None:
            self.gripper_openings = openings.clone()

        def set_visual_variants(self, *_args: Any) -> None:
            return None

        def broadcast_group_state(self, _base_worlds: Any) -> None:
            return None

        def low_dim_observations(self) -> Any:
            from types import SimpleNamespace

            ee_quaternion = torch.zeros(
                (self.ee_positions.shape[0], 4), dtype=torch.float32
            )
            ee_quaternion[:, 0] = torch.cos(0.5 * self.ee_yaw)
            ee_quaternion[:, 3] = torch.sin(0.5 * self.ee_yaw)
            return SimpleNamespace(
                ee_position=self.ee_positions,
                ee_quaternion=ee_quaternion,
                object_positions=self.object_positions,
                object_quaternions=self.object_quaternions,
            )

    return RecordingBackend()


# --------------------------------------------------------------------------
# Config reading
# --------------------------------------------------------------------------


def _apply_overrides(
    metadata: dict[str, Any], overrides: Sequence[str]
) -> dict[str, Any]:
    """``KEY=VALUE`` onto the task metadata, before anything reads it.

    Same contract as xy_approach_probe's flag of the same name: an override
    that reached only one of two consumers would be a control that is not a
    null, and this campaign has already paid for one of those.
    """

    for override in overrides or ():
        key, _, raw = str(override).partition("=")
        if not key or not raw:
            raise SystemExit(
                f"--metadata-override expects KEY=VALUE, got {override!r}"
            )
        lowered = raw.strip().lower()
        if lowered in {"true", "false"}:
            metadata[key] = lowered == "true"
        else:
            try:
                metadata[key] = float(raw)
            except ValueError:
                metadata[key] = raw
        print(f"[p0] metadata override {key}={metadata[key]!r}", flush=True)
    return metadata


def _ladders(
    metadata: Mapping[str, Any], instructions: Sequence[str]
) -> dict[str, list[float]]:
    """Each instruction's configured rungs, mirroring the curriculum's rules.

    Built by instantiating ``PerInstructionApproachCurriculum`` rather than by
    re-reading the metadata keys, because the precedence between
    ``..._ladder``, ``..._ladder_by_instruction`` and
    ``..._initial_by_instruction`` is genuinely intricate (an explicit ladder
    outranks the initial, so an initial override alone is a silent no-op) and a
    second reading of it would be a second implementation.
    """

    from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
        PerInstructionApproachCurriculum,
    )

    curriculum = PerInstructionApproachCurriculum(
        metadata, instruction_types=tuple(instructions)
    )
    out: dict[str, list[float]] = {}
    for name in instructions:
        item = curriculum._by_name[name]
        if item.ladder:
            out[name] = [float(rung) for rung in item.ladder]
        else:
            rungs: list[float] = []
            value = float(item.initial)
            while value <= float(item.final) + 1.0e-9 and len(rungs) < 24:
                rungs.append(round(value, 6))
                value += float(item.increment)
            out[name] = rungs
    return out


# --------------------------------------------------------------------------
# One rung
# --------------------------------------------------------------------------


def _percentile(values: Any, fraction: float) -> float:
    import torch

    if int(values.numel()) == 0:
        return float("nan")
    return float(
        torch.quantile(values.to(dtype=torch.float32), float(fraction)).item()
    )


def measure_rung(
    *,
    torch: Any,
    resetter: Any,
    backend: Any,
    caps_by_id: Mapping[int, float],
    rounds: int,
    grasp_offset: float,
    read_back: bool,
) -> dict[str, Any]:
    """Reset ``rounds`` times at one cap table and pool the realized starts.

    ``read_back`` reads the end-effector through ``low_dim_observations`` (what
    the simulator holds) instead of the recorded command. Identical on the stub;
    on MJWarp the difference is exactly the failure mode the stub cannot see.
    """

    from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
        goal_slots_for_reset,
    )
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
        INSTRUCTION_TO_ID,
    )

    resetter.set_random_start_max_goal_distance(dict(caps_by_id))
    rows: list[dict[str, Any]] = []
    for round_index in range(int(rounds)):
        reset = resetter.reset(
            update_index=round_index, round_index=round_index
        )
        low_dim = backend.low_dim_observations()
        ee = (
            low_dim.ee_position
            if read_back
            else backend.ee_positions
        ).to(dtype=torch.float32)
        objects = low_dim.object_positions.to(dtype=torch.float32)
        worlds = int(ee.shape[0])
        index = torch.arange(worlds, dtype=torch.int64, device=ee.device)
        goal = objects[index, goal_slots_for_reset(torch, reset)]
        planar = torch.linalg.vector_norm(ee[:, :2] - goal[:, :2], dim=-1)
        # The 3-D distance is measured to the point the cap is a radius around,
        # which the reset publishes, NOT to the goal object's centre. For a
        # placement task those differ by the release height plus the gripper
        # hang -- more than every rung -- so measuring to the centre reported
        # over_cap_fraction_3d = 1.0000 on a reset that was behaving exactly as
        # designed. That reading was this probe's bug, and it is the reason the
        # reset now publishes the point instead of leaving it to be rebuilt.
        curriculum_goal = reset.curriculum_goal_xyz
        if curriculum_goal is None:
            raise RuntimeError(
                "The reset published no curriculum_goal_xyz, which means the "
                "random-workspace start path did not run and there is no "
                "approach cap to audit."
            )
        curriculum_goal = curriculum_goal.to(dtype=torch.float32)
        spatial = torch.linalg.vector_norm(ee - curriculum_goal, dim=-1)
        # Independent reading of the same quantity from the two ends: the XY of
        # the published goal must be the goal slot's XY, or one of the two is
        # not the point the curriculum used and every number below is about a
        # different reset than the one that ran.
        drift = float(
            torch.linalg.vector_norm(
                curriculum_goal[:, :2] - goal[:, :2], dim=-1
            ).max().item()
        )
        if drift > 1.0e-4:
            raise RuntimeError(
                "The published curriculum goal and the goal slot disagree in "
                f"XY by up to {drift:.6f} m."
            )

        # Which code path put the end-effector where it is. Inferred from public
        # masks and geometry rather than from resetter internals: a caught start
        # sits exactly one grasp offset above its object (:1988), which is a
        # 1e-6 test, and the two pick_up stages announce themselves on the reset.
        target = objects[index, reset.task_state.target_slots]
        # Compared component-wise rather than against a constructed offset
        # vector: a fresh torch.tensor lands on the CPU while ee/target are on
        # the rank's GPU, which is a device mismatch that the stub backend
        # cannot reproduce and only shows up on the real one.
        offset = ee - target
        caught = (
            torch.linalg.vector_norm(offset[:, :2], dim=-1) < 1.0e-5
        ) & ((offset[:, 2] - grasp_offset).abs() < 1.0e-5)
        prelifted = (
            reset.prelifted
            if reset.prelifted is not None
            else torch.zeros(worlds, dtype=torch.bool)
        )
        aligned = (
            reset.aligned
            if reset.aligned is not None
            else torch.zeros(worlds, dtype=torch.bool)
        )
        rows.append(
            {
                "instruction_ids": reset.task_state.instruction_ids.to("cpu"),
                "planar": planar.to("cpu"),
                "spatial": spatial.to("cpu"),
                "caught": caught.to("cpu"),
                "prelifted": prelifted.to("cpu"),
                "aligned": aligned.to("cpu"),
            }
        )

    def pooled(key: str) -> Any:
        return torch.cat([row[key] for row in rows], dim=0)

    instruction_ids = pooled("instruction_ids")
    planar = pooled("planar")
    spatial = pooled("spatial")
    caught = pooled("caught")
    prelifted = pooled("prelifted")
    aligned = pooled("aligned")

    per_instruction: dict[str, Any] = {}
    for name, task_id in INSTRUCTION_TO_ID.items():
        cap = float(caps_by_id.get(int(task_id), float("inf")))
        mask = instruction_ids == int(task_id)
        n = int(mask.sum().item())
        if n == 0:
            continue
        mine_planar = planar[mask]
        mine_spatial = spatial[mask]
        mine_caught = caught[mask]
        mine_prelifted = prelifted[mask]
        mine_aligned = aligned[mask]
        plain = ~(mine_caught | mine_prelifted | mine_aligned)
        # 1e-4 m of slack: the pull-in clamps the start back into the workspace
        # box after placing it on the annulus (:1721), so a start in a corner can
        # land a float epsilon outside its cap without the cap being wrong.
        over = float((mine_planar > cap + 1.0e-4).float().mean().item())
        over_spatial = float(
            (mine_spatial > cap + 1.0e-4).float().mean().item()
        )
        per_instruction[name] = {
            "n": n,
            "cap": cap,
            "planar_mean": round(float(mine_planar.mean().item()), 5),
            "planar_p05": round(_percentile(mine_planar, 0.05), 5),
            "planar_median": round(_percentile(mine_planar, 0.50), 5),
            "planar_p95": round(_percentile(mine_planar, 0.95), 5),
            "planar_max": round(float(mine_planar.max().item()), 5),
            "spatial_median": round(_percentile(mine_spatial, 0.50), 5),
            "spatial_max": round(float(mine_spatial.max().item()), 5),
            "over_cap_fraction": round(over, 5),
            "over_cap_fraction_3d": round(over_spatial, 5),
            "ee_source": {
                "caught": round(float(mine_caught.float().mean().item()), 4),
                "prelifted": round(
                    float(mine_prelifted.float().mean().item()), 4
                ),
                "aligned": round(float(mine_aligned.float().mean().item()), 4),
                "curriculum": round(float(plain.float().mean().item()), 4),
            },
        }
    return per_instruction


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


_CONTAINER_INSTRUCTIONS = ("put_into_plate", "put_into_bowl")


def verdict(
    rungs_by_instruction: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    includes_z: bool,
    far_rung_min_cap: float = float("inf"),
) -> dict[str, Any]:
    """Pre-registered PASS/FAIL. Stated before the run, applied after.

    The far rung is exempt from the 3-D bound ON PURPOSE, and the exemption is
    in the reset, not here: at ``placement_far_rung_min_cap`` and above, a
    placement start is drawn from ``placement_far_rung_z_bounds`` as a start
    DISTRIBUTION -- begin low, as though the object had just been lifted off the
    desk, and carry up to the receptacle -- so the z_allowance window that keeps
    the 3-D offset inside the cap is deliberately not applied
    (mjwarp_rank_local_collector.py:1788-1800). Failing those rungs would report
    a design decision as a defect. They are reported separately with their
    measured overshoot instead, because the overshoot is a real fact about what
    "не дальше последнего достигнутого cap" guarantees on the top rung.
    """

    out: dict[str, Any] = {}
    for name, rungs in rungs_by_instruction.items():
        ordered = sorted(rungs, key=lambda row: float(row["cap"]))
        medians = [float(row["planar_median"]) for row in ordered]
        caps = [float(row["cap"]) for row in ordered]
        over_key = (
            "over_cap_fraction_3d" if includes_z else "over_cap_fraction"
        )
        exempt = (
            name in _CONTAINER_INSTRUCTIONS
            and includes_z
            and math.isfinite(far_rung_min_cap)
        )
        bounded = [
            row
            for row in ordered
            if not (exempt and float(row["cap"]) >= far_rung_min_cap)
        ]
        far_rungs = [
            {
                "cap": float(row["cap"]),
                "over_cap_fraction_3d": float(row["over_cap_fraction_3d"]),
                "spatial_max": float(row["spatial_max"]),
            }
            for row in ordered
            if exempt and float(row["cap"]) >= far_rung_min_cap
        ]
        worst_over = (
            max(float(row[over_key]) for row in bounded) if bounded else 0.0
        )
        monotone = all(
            later > earlier + 1.0e-6
            for earlier, later in zip(medians, medians[1:])
        )
        # The rung has to MOVE the start, not merely order it. 20% of the cap
        # gap is the smallest step that could not be a sampling artifact at
        # these counts, and it is the check a decorative cap fails while a
        # monotone-by-luck one passes.
        responsive = any(
            (later - earlier) > 0.20 * (cap_late - cap_early)
            for earlier, later, cap_early, cap_late in zip(
                medians, medians[1:], caps, caps[1:]
            )
        ) if len(medians) > 1 else False
        failures = []
        if not monotone:
            failures.append("medians_not_increasing")
        if not responsive:
            failures.append("medians_do_not_track_cap")
        if worst_over > 0.01:
            failures.append(f"over_cap_fraction={worst_over:.4f}")
        out[name] = {
            "pass": not failures,
            "failures": failures,
            "caps": caps,
            "medians": medians,
            "worst_over_cap_fraction": round(worst_over, 5),
            "over_cap_metric": over_key,
            "far_rungs_exempt_from_3d_bound": far_rungs,
        }
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _build_mjwarp_backend(project: Any, device: str) -> Any:
    from rl_vla_bootstrapping.simulation.cdpr_backend import (
        CDPRBackendConfig,
        create_cdpr_backend,
    )

    simulator = project.simulator
    xml_path = project.resolve_path(simulator.fixed_scene_xml)
    if xml_path is None:
        raise SystemExit("The config does not define simulator.fixed_scene_xml.")
    scales = dict(
        getattr(project.embodiment.action_adapter, "controller_scales", {})
        or {}
    )
    config = CDPRBackendConfig(
        backend="mjlab_mjwarp",
        worlds_per_rank=int(simulator.worlds_per_rank),
        groups_per_rank=int(simulator.groups_per_rank),
        grpo_group_size=int(
            int(simulator.worlds_per_rank) // max(int(simulator.groups_per_rank), 1)
        ),
        action_step_xyz=float(scales.get("x", 0.015)),
        action_step_yaw=float(scales.get("yaw", 0.08)),
        action_step_gripper=float(scales.get("gripper", 0.05)),
        render_width=int(simulator.render_width),
        render_height=int(simulator.render_height),
        object_slots=int(simulator.object_slots),
        nconmax=int(simulator.nconmax),
        njmax=int(simulator.njmax),
        device=str(device),
        xml_path=Path(xml_path),
    )
    config.validate()
    print(
        f"[p0] allocating {config.worlds_per_rank} MJWarp worlds on {device}",
        flush=True,
    )
    return create_cdpr_backend(config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("fake", "mjwarp"),
        default="fake",
        help=(
            "fake: the start the resetter COMMANDS, CPU, seconds. mjwarp: the "
            "start the simulator HOLDS, GPU. Run both -- only the pair rules "
            "out a correct reset that the sim then moves."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="Resets per rung. Instruction sampling is random per group, so a "
        "rung with several instructions needs enough rounds to fill each.",
    )
    parser.add_argument("--worlds", type=int, default=0)
    parser.add_argument("--groups", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=20260817)
    parser.add_argument("--caps", type=float, nargs="*", default=[])
    parser.add_argument("--metadata-override", action="append", default=[])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the pre-registered verdict fails.",
    )
    args = parser.parse_args(argv)

    import torch

    from rl_vla_bootstrapping.core.config import load_project_config
    from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
        BatchedReverseFrontierResetter,
        RankLocalCurriculum,
    )
    from rl_vla_bootstrapping.policy.rank_local_grpo import RankLocalGroupLayout
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
        INSTRUCTION_TO_ID,
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    project = load_project_config(args.config.expanduser().resolve())
    metadata = _apply_overrides(
        dict(project.task.metadata or {}), args.metadata_override
    )
    instructions = tuple(project.task.instruction_types or ())
    objects = tuple(project.task.target_objects or ())
    if not instructions:
        raise SystemExit("The config lists no task.instruction_types.")

    worlds = int(args.worlds or project.simulator.worlds_per_rank)
    groups = int(args.groups or project.simulator.groups_per_rank)
    layout = RankLocalGroupLayout(
        worlds_per_rank=worlds,
        groups_per_rank=groups,
        group_size=worlds // max(groups, 1),
    )
    layout.validate()

    # Stated up front, because each of these silently voids the measurement.
    # random_workspace_gripper_start false skips the entire cap block; the z
    # flag decides whether the cap bounds the 3-D distance the reward measures
    # or only its XY shadow, and therefore which over_cap number is the real
    # one.
    preconditions = {
        "random_workspace_gripper_start": bool(
            metadata.get("random_workspace_gripper_start", False)
        ),
        "curriculum_cap_includes_z": bool(
            metadata.get("curriculum_cap_includes_z", False)
        ),
        "random_workspace_min_goal_xy_distance": float(
            metadata.get("random_workspace_min_goal_xy_distance", 0.0)
        ),
        "placement_caught_object_fraction": float(
            metadata.get("placement_caught_object_fraction", 1.0)
        ),
        "pick_up_prelifted_group_fraction": float(
            metadata.get("pick_up_prelifted_group_fraction", 0.0)
        ),
        "pick_up_aligned_group_fraction": float(
            metadata.get("pick_up_aligned_group_fraction", 0.0)
        ),
        "curriculum_horizon_coupling_enabled": bool(
            metadata.get("curriculum_horizon_coupling_enabled", False)
        ),
        "placement_far_rung_min_cap": float(
            metadata.get("placement_far_rung_min_cap", float("inf"))
        ),
    }
    print(f"[p0] preconditions {json.dumps(preconditions, sort_keys=True)}", flush=True)
    if not preconditions["random_workspace_gripper_start"]:
        raise SystemExit(
            "random_workspace_gripper_start is false, so the approach-cap block "
            "(mjwarp_rank_local_collector.py:1579) never runs and every number "
            "this probe would print is unrelated to the cap. Fix the config "
            "rather than reading the output."
        )

    ladders = _ladders(metadata, instructions)
    if args.caps:
        ladders = {name: list(args.caps) for name in instructions}
    print(f"[p0] rungs {json.dumps(ladders, sort_keys=True)}", flush=True)

    if args.backend == "mjwarp":
        backend = _build_mjwarp_backend(project, args.device)
        read_back = True
    else:
        backend = build_recording_backend(
            torch, object_slots=int(project.simulator.object_slots)
        )
        read_back = False

    grasp_offset = float(metadata.get("pick_grasp_height_offset", 0.0075))
    # Every rung of every instruction is visited, and the OTHER instructions are
    # pinned at their own first rung meanwhile. Sweeping all of them together
    # would confound "this instruction's cap moved" with "the mix moved".
    rungs_by_instruction: dict[str, list[dict[str, Any]]] = {
        name: [] for name in instructions
    }
    for name in instructions:
        for cap in ladders[name]:
            caps_by_id = {
                int(INSTRUCTION_TO_ID[other]): (
                    float(cap)
                    if other == name
                    else float(ladders[other][0])
                )
                for other in instructions
            }
            resetter = BatchedReverseFrontierResetter(
                backend=backend,
                layout=layout,
                curriculum=RankLocalCurriculum(device=backend.device),
                rank=0,
                base_seed=int(args.base_seed),
                instruction_types=instructions,
                allowed_objects=objects,
                frontier_probability=1.0,
                rehearsal_probability=0.0,
                balanced_target_catalogs=True,
                task_metadata=metadata,
            )
            measured = measure_rung(
                torch=torch,
                resetter=resetter,
                backend=backend,
                caps_by_id=caps_by_id,
                rounds=int(args.rounds),
                grasp_offset=grasp_offset,
                read_back=read_back,
            )
            row = measured.get(name)
            if row is None:
                print(
                    f"[p0] {name} cap={cap}: no worlds sampled this "
                    "instruction; raise --rounds",
                    flush=True,
                )
                continue
            rungs_by_instruction[name].append(row)
            source = row["ee_source"]
            print(
                f"[p0] {name:<16} cap={cap:<6.3f} n={row['n']:<5} "
                f"median={row['planar_median']:.4f} "
                f"p05={row['planar_p05']:.4f} p95={row['planar_p95']:.4f} "
                f"max={row['planar_max']:.4f} "
                f"over_cap={row['over_cap_fraction']:.4f} "
                f"(3d {row['over_cap_fraction_3d']:.4f})  "
                f"ee: curr={source['curriculum']:.2f} "
                f"caught={source['caught']:.2f} "
                f"prelift={source['prelifted']:.2f} "
                f"aligned={source['aligned']:.2f}",
                flush=True,
            )

    rungs_by_instruction = {
        name: rows for name, rows in rungs_by_instruction.items() if rows
    }
    decision = verdict(
        rungs_by_instruction,
        includes_z=bool(preconditions["curriculum_cap_includes_z"]),
        far_rung_min_cap=float(preconditions["placement_far_rung_min_cap"]),
    )
    report = {
        "config": str(args.config),
        "backend": args.backend,
        "worlds_per_rank": worlds,
        "groups_per_rank": groups,
        "rounds_per_rung": int(args.rounds),
        "base_seed": int(args.base_seed),
        "preconditions": preconditions,
        "ladders": ladders,
        "rungs": rungs_by_instruction,
        "verdict": decision,
    }
    (output / "start_distance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    failed = [name for name, item in decision.items() if not item["pass"]]
    for name, item in sorted(decision.items()):
        state = "PASS" if item["pass"] else "FAIL"
        detail = "" if item["pass"] else f"  <- {', '.join(item['failures'])}"
        print(
            f"[p0] {state} {name}: medians "
            f"{[round(value, 4) for value in item['medians']]} "
            f"against caps {item['caps']}{detail}",
            flush=True,
        )
        for far in item["far_rungs_exempt_from_3d_bound"]:
            print(
                f"[p0]      cap {far['cap']:.3f} is the far rung: the 3-D "
                "bound is released by design, measured overshoot "
                f"{far['over_cap_fraction_3d']:.4f} of worlds, max "
                f"{far['spatial_max']:.4f} m",
                flush=True,
            )
    print(
        f"[p0] wrote {output / 'start_distance_report.json'}",
        flush=True,
    )
    if failed and bool(args.strict):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
