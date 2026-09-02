#!/usr/bin/env python3
"""Split placement failures into the predicate terms that caused them.

The composed `put_into` task has a CEILING that is not the policy's. The
scripted oracle -- which is handed ground truth and drives the reference phase
chain -- scores plate 0.909 and bowl **0.455** pooled over 8192 worlds. No
policy can exceed its own oracle, so "70% on bowl" is unreachable by training
until something about the task, the release, or the predicate changes. Which of
those it is has never been measured: the campaign has the oracle's success rate
and nothing about the shape of its 55% of failures.

This measures that shape, and it does it WITHOUT GPU time. Every quantity the
container predicate reads is already stored in a recording:

    object_xyz          [T, W, slots, 3]   target and reference positions
    gripper_opening     [T, W]             the release test's input
    caught_target       [T, W]             what latches ever_grasped
    active              [T, W]             which steps happened
    support_surface_z   [W]                the settle test's datum
    target_rest_height  [W]                ... and its offset
    release_threshold   [W]                the per-world release bar

So the twelve oracle harvest rounds under `runs/phase4_bank/o6_*` can be
decomposed as they stand, and so can any `sil_record --mode record` eval. Costs
seconds on a CPU.

WHY A FUNNEL AND NOT A CLASSIFIER
---------------------------------
`container_ok` is a seven-term conjunction, and a failed episode usually misses
more than one. Reporting "the reason" for each failure would be a choice
dressed as a measurement. The funnel instead reports, for each term, how many
episodes EVER satisfied it at an active step -- so the term that collapses is
visible without deciding which single term to blame. The mutually exclusive
`blocking_stage` column is derived from the funnel afterwards, in causal order,
and is offered as a convenience rather than as the finding.

THE DUPLICATE-PREDICATE HAZARD, AND THE GUARD AGAINST IT
--------------------------------------------------------
This file recomputes terms that `evaluate_active_sparse_tasks` owns, which is
exactly the duplication this campaign has already paid for twice. It cannot
call through: the predicate is a stateful torch function that latches inside a
live `BatchedTaskState`, and there is no such state beside a stored npz.

So the recomputation is CHECKED rather than trusted. The final conjunction is
compared against the recording's own latched `success`, world by world, and a
disagreement is reported first and loudly. `--max-predicate-disagreement`
(default 0) makes it fatal. A decomposition that does not reproduce the verdict
it is decomposing describes nothing, and the failure mode this guards against
is the silent one: a threshold read from the wrong metadata key produces a
plausible table that is about a different task.

WHAT THE OUTPUT IS FOR
----------------------
Three questions, in the order they should be asked of a placement family:

1. Does the object reach the receptacle at all? -> `ever_xy_ok` in the funnel.
   Below the grasp rate, the composed prefix is the binding constraint and
   nothing about the release matters yet.
2. Does it arrive and then leave? -> `released_inside_but_failed`, whose
   `bounce_m` is the XY the object travelled between the release and its
   resting place. A bowl is concave; if bounce is large this is an object
   landing in the target and coming back out, and the fix is the release
   height, not the policy.
3. Does it arrive, stay, and still fail? -> `not_settled` and `z_miss`. Those
   are predicate geometry, and the fix is in the config.

`release_clearance_buckets` answers 2 quantitatively: success rate against the
height the object was let go from, with the bucket counts kept visible so a
thin bucket cannot be read as a trend.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MUJOCO_GL", "disable")

import argparse  # noqa: E402
import glob as _glob  # noqa: E402
import json  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from tools.audit.sil_record import (  # noqa: E402
    _Recording,
    _instruction_name,
    _write_csv,
)

CONTAINER_INSTRUCTIONS = ("put_into_plate", "put_into_bowl")

# Causal order, and each stage is the previous stage's conjunction with one
# more term AT THE SAME ENV STEP. That nesting is the whole correctness of this
# file, and it was got wrong first: asking each term independently with
# `.any()` over the episode lets an episode satisfy "the object was over the
# receptacle" at a step where the object is still in the gripper, mid-carry,
# 10 cm up. An object released over the centre of a bowl and bounced out then
# reads as `not_settled` -- true of the moment it was inside, useless as a
# cause -- when the finding is that it came to rest in the wrong place.
#
# `settled` comes BEFORE the geometry on purpose. "Did it land in the
# receptacle" is only a question once the object has been let go and stopped
# moving; asked earlier it describes the carry, not the placement.
FUNNEL_STAGES = (
    "ever_grasped",
    "ever_released",
    "ever_settled_after_release",
    "ever_z_ok_at_rest",
    "ever_xy_ok_at_rest",
    "success",
)
BLOCKING_LABELS = {
    "ever_grasped": "no_grasp",
    "ever_released": "no_release",
    "ever_settled_after_release": "not_settled",
    "ever_z_ok_at_rest": "z_miss",
    "ever_xy_ok_at_rest": "xy_miss",
    "success": "release_height_gate",
}


class _Thresholds:
    """The container predicate's constants, read where production reads them.

    Defaults duplicated from `BatchedTaskThresholds` and
    `BatchedCatchReleaseDenseReward`; the metadata keys are the ones
    `catch_release_dense_reward_from_metadata` uses. Both are printed, because
    a decomposition run against the wrong config is the most likely way to get
    a confident wrong answer here -- a plate radius applied to bowl episodes
    moves the funnel's collapse point by a whole stage.
    """

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        def number(key: str, default: float) -> float:
            try:
                return float(metadata.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        self.plate_radius = max(number("put_plate_xy_tolerance", 0.091), 1e-6)
        self.bowl_radius = max(number("put_bowl_xy_tolerance", 0.057), 1e-6)
        # The predicate ANDs TWO z tests -- `container_z <= cfg.container_z` and
        # `container_z <= container_z_tolerance` -- and they are not independent
        # knobs. `RankLocalMJWarpGRPOCollector._task_thresholds` builds
        # `cfg.container_z` FROM `catch_release.container_z_tolerance`, so in
        # production both sides read the same metadata key and the conjunction
        # is one test. Read once here for the same reason: inventing a second
        # key would fall back to a default that happens to equal the config's
        # value and would be right by accident until someone changed it.
        self.container_z_tolerance = max(
            number("put_container_z_tolerance", 0.12), 0.0
        )
        self.container_z = self.container_z_tolerance
        self.settle_margin = max(
            number("placement_wrong_drop_settle_margin", 0.025), 0.0
        )
        # NOT metadata-driven. `BatchedTaskThresholds.release_opening` is a
        # dataclass default and `_task_thresholds` does not override it, so the
        # floor under every world's release bar is 0.55 in every config. The
        # per-world part is `release_threshold`, which the recording stores.
        self.release_opening = 0.55
        # The composed scene's object spawn, so an episode whose object is not
        # actually inside it can be COUNTED rather than inferred from a mean
        # sitting above its own p90.
        self.spawn_min = max(number("placement_grasp_object_min_distance", 0.06), 0.0)
        self.spawn_max = max(
            number("placement_grasp_object_max_distance", 0.10), self.spawn_min
        )
        shared = max(number("put_release_max_height", 0.0), 0.0)
        self.plate_release_max = (
            max(number("put_plate_release_max_height", 0.0), 0.0) or shared
        )
        self.bowl_release_max = (
            max(number("put_bowl_release_max_height", 0.0), 0.0) or shared
        )

    def radius_for(self, instruction: str) -> float:
        return self.bowl_radius if instruction == "put_into_bowl" else self.plate_radius

    def release_max_for(self, instruction: str) -> float:
        limit = (
            self.bowl_release_max
            if instruction == "put_into_bowl"
            else self.plate_release_max
        )
        return float(limit) if limit > 0.0 else float("inf")

    def as_dict(self) -> dict[str, Any]:
        return {
            "plate_radius_m": round(self.plate_radius, 5),
            "bowl_radius_m": round(self.bowl_radius, 5),
            "container_z_m": round(self.container_z, 5),
            "container_z_tolerance_m": round(self.container_z_tolerance, 5),
            "settle_margin_m": round(self.settle_margin, 5),
            "release_opening": round(self.release_opening, 5),
            "plate_release_max_height_m": (
                None if self.plate_release_max <= 0.0 else round(self.plate_release_max, 5)
            ),
            "bowl_release_max_height_m": (
                None if self.bowl_release_max <= 0.0 else round(self.bowl_release_max, 5)
            ),
            "spawn_min_m": round(self.spawn_min, 5),
            "spawn_max_m": round(self.spawn_max, 5),
            "release_height_gate_armed": bool(
                self.plate_release_max > 0.0 or self.bowl_release_max > 0.0
            ),
        }


def _percentiles(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": round(float(finite.mean()), 5),
        "p10": round(float(np.percentile(finite, 10)), 5),
        "p50": round(float(np.percentile(finite, 50)), 5),
        "p90": round(float(np.percentile(finite, 90)), 5),
    }


def _episode_terms(
    recording: _Recording, thresholds: _Thresholds
) -> list[dict[str, Any]]:
    """One row per container world, with every predicate term recomputed.

    Vectorised over [T, W] and then reduced per world, because the predicate is
    latched with `success.any()` over the episode: a term satisfied at one step
    and lost at the next still counted, and a per-step snapshot at termination
    would call those failures.
    """

    worlds = recording.worlds
    steps = recording.active.shape[0]
    world_index = np.arange(worlds)

    target = recording.object_xyz[:, world_index, recording.target_slots, :]
    reference = recording.object_xyz[:, world_index, recording.reference_slots, :]
    container_xy = np.linalg.norm(target[:, :, :2] - reference[:, :, :2], axis=-1)
    container_z = np.abs(target[:, :, 2] - reference[:, :, 2])
    # The predicate's `clearance_now`: SIGNED, and against the receptacle rather
    # than the desk. A negative value is an object below the receptacle origin,
    # which is what being inside a bowl looks like.
    clearance = target[:, :, 2] - reference[:, :, 2]

    active = recording.active.astype(bool)
    caught = recording.caught_target.astype(bool)
    # Latched exactly as `state.ever_grasped.logical_or_` latches it: once true,
    # true for every later step of the episode.
    ever_grasped_t = np.logical_or.accumulate(caught & active, axis=0)
    release_bar = np.maximum(
        recording.release_threshold.astype(np.float64),
        float(thresholds.release_opening),
    )
    released = recording.gripper_opening.astype(np.float64) >= release_bar[None, :]

    settled = target[:, :, 2] <= (
        recording.support_surface_z[None, :]
        + recording.target_rest_height[None, :]
        + float(thresholds.settle_margin)
    )

    instructions = [_instruction_name(i) for i in recording.instruction_ids]
    radii = np.array(
        [thresholds.radius_for(name) for name in instructions], dtype=np.float64
    )
    z_limit = min(thresholds.container_z, thresholds.container_z_tolerance)

    xy_ok = container_xy <= radii[None, :]
    z_ok = container_z <= z_limit
    release_now = released & ever_grasped_t & active

    rows: list[dict[str, Any]] = []
    for world in range(worlds):
        name = instructions[world]
        if name not in CONTAINER_INSTRUCTIONS:
            continue
        live = active[:, world]
        if not live.any():
            continue
        last = int(np.flatnonzero(live)[-1])

        # WHEN the grasp happened, and whether the episode ran out of budget.
        #
        # These separate the two live explanations for `no_release`, which is
        # 25-33% of every failure population measured so far. Either the
        # episode grasped so late that no budget remained to carry and open --
        # a horizon problem, fixable with placement_grasp_horizon_min_decisions
        # -- or it grasped in good time and the release never fired, which is a
        # policy/controller problem and a completely different piece of work.
        grasp_steps = np.flatnonzero(caught[:, world] & live)
        first_grasp = int(grasp_steps[0]) if grasp_steps.size else -1
        budget = int(recording.horizons[world]) * int(
            recording.actions_per_decision
        )
        steps_active = int(live.sum())
        # An episode terminates on success, on a settled wrong drop, or on
        # timeout, so "used the whole budget" means nothing terminated it.
        timed_out = steps_active >= budget
        # How far the object actually sat from its receptacle at the start.
        #
        # MEASURED FROM object_xyz[0], NOT FROM initial_target_xyz, and the
        # difference is not cosmetic. `initial_target_positions` is captured
        # BEFORE the placement repositioning: the resetter updates it for
        # `held_group` and for `grasp_learning` and NOT for
        # `uncaught_container`, so a composed episode's entry still holds the
        # pre-repositioning lattice point somewhere in the workspace. Reading
        # it here reported medians of 0.26-0.35 m against a spawn the config
        # fixes at 0.06-0.10 m (placement_grasp_object_min/max_distance), which
        # is how the bug was caught: a plausible-looking number that disagreed
        # with the knob that produces it.
        #
        # object_xyz[0] is one env step of physics after the reset. For an
        # object at rest on the desk that is a sub-millimetre difference.
        reset_separation = float(
            np.linalg.norm(target[0, world, :2] - reference[0, world, :2])
        )

        # Did the episode END holding the object, or did it lose it?
        #
        # This splits `no_release` a second way, orthogonally to `timed_out`.
        # An episode that ran the whole budget still holding never got to the
        # release; one that terminated early while NOT holding dropped the
        # object and was terminated by `wrong_place_settled` -- it grasped, lost
        # grip without ever opening past the release bar, and the object settled
        # outside the receptacle. Those are a horizon problem and a grip
        # retention problem, and they want opposite work.
        held_now = caught[:, world] & (
            recording.gripper_opening[:, world] <= 0.94
        )
        ended_holding = bool(held_now[last])
        lost_steps = np.flatnonzero(
            (~caught[:, world]) & live & ever_grasped_t[:, world]
        )
        lost_grip = int(lost_steps[0]) if lost_steps.size else -1

        # The release the predicate latches on: the FIRST one, so a policy that
        # opens, re-closes and opens again is scored on the height it first let
        # go from -- which is the one that decided the object's trajectory.
        release_steps = np.flatnonzero(release_now[:, world])
        first_release = int(release_steps[0]) if release_steps.size else -1
        release_clearance = (
            float(clearance[first_release, world]) if first_release >= 0 else float("nan")
        )
        xy_at_release = (
            float(container_xy[first_release, world]) if first_release >= 0 else float("nan")
        )
        limit = thresholds.release_max_for(name)
        # NaN compares false, matching the predicate's own comment: a world that
        # never released fails the gate without a separate isnan branch.
        release_height_ok = bool(release_clearance <= limit)

        # Nested, one term added per stage, all read at the SAME step.
        s_grasped = caught[:, world] & live
        s_released = release_now[:, world]
        s_settled = s_released & settled[:, world]
        s_z = s_settled & z_ok[:, world]
        s_xy = s_z & xy_ok[:, world]
        stage_reached = {
            "ever_grasped": bool(s_grasped.any()),
            "ever_released": bool(s_released.any()),
            "ever_settled_after_release": bool(s_settled.any()),
            "ever_z_ok_at_rest": bool(s_z.any()),
            "ever_xy_ok_at_rest": bool(s_xy.any()),
        }
        # `s_xy` is `container_ok` without the height gate, and the gate is a
        # per-episode latch rather than a per-step term -- so it multiplies the
        # episode verdict rather than joining the conjunction above.
        recomputed = bool(s_xy.any() and release_height_ok)
        stage_reached["success"] = recomputed

        # How close the object came to REST from the receptacle, which is a
        # different question from how close it ever passed while being carried.
        rest_steps = np.flatnonzero(s_settled)
        min_xy_at_rest = (
            float(container_xy[rest_steps, world].min())
            if rest_steps.size
            else float("nan")
        )

        blocking = ""
        if not recomputed:
            for stage in FUNNEL_STAGES:
                if not stage_reached[stage]:
                    blocking = BLOCKING_LABELS[stage]
                    break

        xy_at_rest = float(container_xy[last, world])
        rows.append(
            {
                "world": world,
                "instruction": name,
                "recorded_success": bool(recording.episode_success[world]),
                "recomputed_success": recomputed,
                "blocking_stage": blocking,
                "grasped_at_reset": bool(recording.physical_grasp_at_reset[world]),
                "first_grasp_env_step": first_grasp,
                "first_release_env_step": first_release,
                "timed_out": timed_out,
                "ended_holding": ended_holding,
                "lost_grip_env_step": lost_grip,
                "env_step_budget": budget,
                "object_to_receptacle_at_reset_m": round(reset_separation, 5),
                "release_clearance_m": round(release_clearance, 5),
                "xy_at_release_m": round(xy_at_release, 5),
                "min_xy_m": round(float(container_xy[live, world].min()), 5),
                "min_xy_at_rest_m": round(min_xy_at_rest, 5),
                "xy_at_rest_m": round(xy_at_rest, 5),
                # Positive = the object left the place it was released over.
                # This is the number that says whether a bowl is bouncing its
                # contents back out, and it is only meaningful once released.
                "bounce_m": (
                    round(xy_at_rest - xy_at_release, 5)
                    if first_release >= 0
                    else float("nan")
                ),
                "z_at_rest_m": round(float(container_z[last, world]), 5),
                "settle_gap_m": round(
                    float(
                        target[last, world, 2]
                        - recording.support_surface_z[world]
                        - recording.target_rest_height[world]
                        - float(thresholds.settle_margin)
                    ),
                    5,
                ),
                "radius_m": round(float(radii[world]), 5),
                "env_steps_active": int(live.sum()),
                "horizon_decisions": int(recording.horizons[world]),
                "instruction_text": str(recording.instructions[world]),
            }
        )
    return rows


def _bucket_release_clearance(
    rows: Sequence[Mapping[str, Any]], edges: Sequence[float]
) -> list[dict[str, Any]]:
    """Success rate against the height the object was let go from.

    Only episodes that actually released are counted; the rest have no height.
    Counts are reported beside every rate on purpose -- the campaign's own rule
    is that a rate without its denominator is not a result, and the tails here
    are thin by construction because the oracle releases at a scripted height.
    """

    released = [
        row for row in rows if np.isfinite(float(row["release_clearance_m"]))
    ]
    out: list[dict[str, Any]] = []
    bounds = list(edges)
    for index in range(len(bounds) + 1):
        low = -float("inf") if index == 0 else bounds[index - 1]
        high = float("inf") if index == len(bounds) else bounds[index]
        subset = [
            row
            for row in released
            if low <= float(row["release_clearance_m"]) < high
        ]
        if not subset:
            continue
        successes = sum(1 for row in subset if row["recomputed_success"])
        out.append(
            {
                "clearance_low_m": None if index == 0 else round(low, 4),
                "clearance_high_m": None if index == len(bounds) else round(high, 4),
                "episodes": len(subset),
                "successes": successes,
                "success_rate": round(successes / len(subset), 4),
                "bounce_m": _percentiles(
                    np.array([row["bounce_m"] for row in subset], dtype=np.float64)
                ),
            }
        )
    return out


def _summarize(
    rows: Sequence[Mapping[str, Any]], thresholds: _Thresholds, edges: Sequence[float]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in CONTAINER_INSTRUCTIONS:
        subset = [row for row in rows if row["instruction"] == name]
        if not subset:
            continue
        total = len(subset)
        successes = [row for row in subset if row["recomputed_success"]]
        failures = [row for row in subset if not row["recomputed_success"]]

        funnel: dict[str, Any] = {}
        # Recomputed from the mutually exclusive blocking stage: an episode
        # reached a stage iff nothing earlier blocked it. Derived this way so
        # the funnel and the taxonomy cannot disagree with each other.
        order = list(FUNNEL_STAGES)
        for position, stage in enumerate(order):
            blocked_earlier = {BLOCKING_LABELS[s] for s in order[: position + 1]}
            reached = sum(
                1 for row in subset if row["blocking_stage"] not in blocked_earlier
            )
            funnel[stage] = {
                "episodes": reached,
                "fraction_of_all": round(reached / total, 4),
            }

        taxonomy: dict[str, Any] = {}
        for label in BLOCKING_LABELS.values():
            count = sum(1 for row in failures if row["blocking_stage"] == label)
            if count:
                taxonomy[label] = {
                    "episodes": count,
                    "fraction_of_failures": round(count / max(len(failures), 1), 4),
                }

        # The question the bowl ceiling turns on: the object was released with
        # its XY already inside the receptacle radius, and the episode still
        # failed. Whatever happened, happened AFTER a correct release.
        arrived_then_failed = [
            row
            for row in failures
            if np.isfinite(float(row["xy_at_release_m"]))
            and float(row["xy_at_release_m"]) <= float(row["radius_m"])
        ]
        # The conditional rates. A funnel read as fractions-of-all hides where
        # the loss is: bowl and plate both lose most of their episodes at the
        # grasp, but bowl ALSO loses a quarter of its grasps before the release
        # and plate loses 2%. Those are different problems and the marginal
        # numbers make them look like one.
        stages = [total] + [
            funnel[stage]["episodes"] for stage in FUNNEL_STAGES
        ]
        conditional = {}
        for index, stage in enumerate(FUNNEL_STAGES):
            denominator = stages[index]
            conditional[stage] = (
                round(stages[index + 1] / denominator, 4)
                if denominator
                else None
            )

        # Why `no_release` happened, which is the one term the funnel cannot
        # explain on its own.
        no_release = [row for row in failures if row["blocking_stage"] == "no_release"]
        timed_out = [row for row in no_release if row["timed_out"]]
        grasped_rows = [
            row for row in subset if int(row["first_grasp_env_step"]) >= 0
        ]
        summary[name] = {
            "episodes": total,
            "successes": len(successes),
            "success_rate": round(len(successes) / total, 4),
            "radius_m": round(thresholds.radius_for(name), 5),
            "funnel": funnel,
            "funnel_conditional": conditional,
            "failure_taxonomy": taxonomy,
            "no_release_diagnosis": {
                "episodes": len(no_release),
                # Ran the whole budget: nothing terminated them, so the carry
                # and the open had no room left. A high fraction here is a
                # HORIZON finding and points at
                # placement_grasp_horizon_min_decisions.
                "timed_out": len(timed_out),
                "timed_out_fraction": round(
                    len(timed_out) / max(len(no_release), 1), 4
                ),
                # Ended still holding: never reached the release. Ended NOT
                # holding without having released: dropped it. The oracle's
                # no_release is the first; the policy's is mostly the second.
                "ended_holding": sum(
                    1 for row in no_release if row["ended_holding"]
                ),
                "lost_grip": sum(
                    1
                    for row in no_release
                    if int(row["lost_grip_env_step"]) >= 0
                    and not row["ended_holding"]
                ),
                "first_grasp_env_step": _percentiles(
                    np.array(
                        [row["first_grasp_env_step"] for row in no_release],
                        dtype=np.float64,
                    )
                ),
                "env_step_budget": _percentiles(
                    np.array(
                        [row["env_step_budget"] for row in no_release],
                        dtype=np.float64,
                    )
                ),
            },
            "first_grasp_env_step": {
                "successes": _percentiles(
                    np.array(
                        [row["first_grasp_env_step"] for row in successes],
                        dtype=np.float64,
                    )
                ),
                "all_that_grasped": _percentiles(
                    np.array(
                        [row["first_grasp_env_step"] for row in grasped_rows],
                        dtype=np.float64,
                    )
                ),
            },
            # Objects that are not where the resetter put them.
            #
            # The spawn is drawn in [min, max] and then CLAMPED to the workspace
            # bounds, so a receptacle near an edge legitimately produces a
            # closer object -- that is the low tail. A spawn ABOVE the maximum
            # cannot come from the draw at all: the object has been moved by
            # physics before the first observation, which for a CONCAVE
            # receptacle is what an object spawned intersecting its wall looks
            # like once the solver resolves the penetration.
            "spawn_outside_configured_range": {
                "range_m": [
                    round(thresholds.spawn_min, 5),
                    round(thresholds.spawn_max, 5),
                ],
                "below_min": sum(
                    1
                    for row in subset
                    if float(row["object_to_receptacle_at_reset_m"])
                    < thresholds.spawn_min
                ),
                "above_max": sum(
                    1
                    for row in subset
                    if float(row["object_to_receptacle_at_reset_m"])
                    > thresholds.spawn_max
                ),
                "above_max_never_grasped": sum(
                    1
                    for row in subset
                    if float(row["object_to_receptacle_at_reset_m"])
                    > thresholds.spawn_max
                    and int(row["first_grasp_env_step"]) < 0
                ),
                "above_max_distance_m": _percentiles(
                    np.array(
                        [
                            row["object_to_receptacle_at_reset_m"]
                            for row in subset
                            if float(row["object_to_receptacle_at_reset_m"])
                            > thresholds.spawn_max
                        ],
                        dtype=np.float64,
                    )
                ),
            },
            # Split by whether the grasp happened at all. If the bowl's wall is
            # obstructing, ungrasped episodes concentrate at the NEAR end of the
            # 0.06-0.10 m spawn range and grasped ones at the far end.
            "object_to_receptacle_at_reset_m": {
                "grasped": _percentiles(
                    np.array(
                        [
                            row["object_to_receptacle_at_reset_m"]
                            for row in grasped_rows
                        ],
                        dtype=np.float64,
                    )
                ),
                "never_grasped": _percentiles(
                    np.array(
                        [
                            row["object_to_receptacle_at_reset_m"]
                            for row in subset
                            if int(row["first_grasp_env_step"]) < 0
                        ],
                        dtype=np.float64,
                    )
                ),
            },
            "released_inside_but_failed": {
                "episodes": len(arrived_then_failed),
                "fraction_of_failures": round(
                    len(arrived_then_failed) / max(len(failures), 1), 4
                ),
                "bounce_m": _percentiles(
                    np.array(
                        [row["bounce_m"] for row in arrived_then_failed],
                        dtype=np.float64,
                    )
                ),
                "release_clearance_m": _percentiles(
                    np.array(
                        [row["release_clearance_m"] for row in arrived_then_failed],
                        dtype=np.float64,
                    )
                ),
                "z_at_rest_m": _percentiles(
                    np.array(
                        [row["z_at_rest_m"] for row in arrived_then_failed],
                        dtype=np.float64,
                    )
                ),
                "settle_gap_m": _percentiles(
                    np.array(
                        [row["settle_gap_m"] for row in arrived_then_failed],
                        dtype=np.float64,
                    )
                ),
            },
            "release_clearance_m": {
                "successes": _percentiles(
                    np.array(
                        [row["release_clearance_m"] for row in successes],
                        dtype=np.float64,
                    )
                ),
                "failures": _percentiles(
                    np.array(
                        [row["release_clearance_m"] for row in failures],
                        dtype=np.float64,
                    )
                ),
            },
            "bounce_m": {
                "successes": _percentiles(
                    np.array([row["bounce_m"] for row in successes], dtype=np.float64)
                ),
                "failures": _percentiles(
                    np.array([row["bounce_m"] for row in failures], dtype=np.float64)
                ),
            },
            "min_xy_m": {
                # Carried-past distance versus came-to-rest distance. A family
                # whose object passes over the receptacle and rests far from it
                # has a release problem; one that never passes close has an
                # approach problem, and they need opposite fixes.
                "failures_any_step": _percentiles(
                    np.array([row["min_xy_m"] for row in failures], dtype=np.float64)
                ),
                "failures_at_rest": _percentiles(
                    np.array(
                        [row["min_xy_at_rest_m"] for row in failures],
                        dtype=np.float64,
                    )
                ),
            },
            "release_clearance_buckets": _bucket_release_clearance(subset, edges),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose put_into_* failures into the predicate terms that "
            "caused them. CPU only; reads recordings already on disk."
        )
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        required=True,
        help="record_*.npz or replay_*.npz paths; globs are expanded.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "The task config the recordings were produced under. Supplies the "
            "receptacle radii, the z tolerances and the settle margin, so the "
            "recomputed predicate matches the one that scored them."
        ),
    )
    parser.add_argument(
        "--metadata-override",
        nargs="*",
        default=(),
        help="KEY=VALUE, applied as in sil_record/xy_approach_probe.",
    )
    parser.add_argument(
        "--release-clearance-edges",
        nargs="*",
        type=float,
        default=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14),
        help=(
            "Bucket edges in metres for the release-height sweep. The default "
            "brackets the 0.10 the shaping currently hovers at and the ~0.042 "
            "resting clearance."
        ),
    )
    parser.add_argument(
        "--max-predicate-disagreement",
        type=int,
        default=0,
        help=(
            "How many worlds may disagree with the recording's own latched "
            "success before this refuses to report. Default 0: a "
            "decomposition that does not reproduce the verdict it is "
            "decomposing describes a different task."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    from rl_vla_bootstrapping.core.config import load_project_config

    paths: list[Path] = []
    for pattern in args.recordings:
        expanded = sorted(_glob.glob(pattern))
        if not expanded:
            raise SystemExit(f"No recordings matched {pattern!r}.")
        paths.extend(Path(p) for p in expanded)

    project = load_project_config(Path(args.config))
    metadata = dict(project.task.metadata or {})
    for override in args.metadata_override or ():
        key, _, raw = str(override).partition("=")
        if not key or not raw:
            raise SystemExit(f"--metadata-override expects KEY=VALUE, got {override!r}")
        lowered = raw.strip().lower()
        if lowered in {"true", "false"}:
            metadata[key] = lowered == "true"
        else:
            try:
                metadata[key] = float(raw)
            except ValueError:
                metadata[key] = raw
        print(f"[decomp] metadata override {key}={metadata[key]!r}", flush=True)

    thresholds = _Thresholds(metadata)
    print(f"[decomp] thresholds {json.dumps(thresholds.as_dict())}", flush=True)

    rows: list[dict[str, Any]] = []
    for path in paths:
        recording = _Recording.from_npz(path)
        found = _episode_terms(recording, thresholds)
        for row in found:
            row["recording"] = path.name
            row["start_distance_cap"] = (
                None
                if not np.isfinite(recording.start_distance_cap)
                else round(float(recording.start_distance_cap), 4)
            )
        rows.extend(found)
        print(
            f"[decomp] {path.name}: {len(found)} container episodes",
            flush=True,
        )

    if not rows:
        raise SystemExit(
            "No put_into_plate / put_into_bowl episodes in these recordings. "
            "Point --recordings at a placement or composed harvest."
        )

    disagreements = [
        row for row in rows if row["recorded_success"] != row["recomputed_success"]
    ]
    if disagreements:
        print(
            f"[decomp] PREDICATE DISAGREEMENT on {len(disagreements)} of "
            f"{len(rows)} worlds. The terms recomputed here do not reproduce "
            "the verdict the trainer latched. Most likely --config is not the "
            "one these recordings were scored under, or a threshold key has "
            "been renamed. First five:",
            flush=True,
        )
        for row in disagreements[:5]:
            print(
                f"[decomp]   {row['recording']} world {row['world']} "
                f"{row['instruction']}: recorded={row['recorded_success']} "
                f"recomputed={row['recomputed_success']} "
                f"min_xy={row['min_xy_m']} z_at_rest={row['z_at_rest_m']} "
                f"release_clearance={row['release_clearance_m']}",
                flush=True,
            )
        if len(disagreements) > int(args.max_predicate_disagreement):
            raise SystemExit(
                f"{len(disagreements)} disagreements exceeds "
                f"--max-predicate-disagreement {args.max_predicate_disagreement}."
            )

    summary = {
        "recordings": [str(path) for path in paths],
        "config": str(args.config),
        "thresholds": thresholds.as_dict(),
        "container_episodes": len(rows),
        "predicate_disagreements": len(disagreements),
        "by_instruction": _summarize(rows, thresholds, args.release_clearance_edges),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "failure_decomposition.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "container_episodes.csv", rows)

    for name, entry in summary["by_instruction"].items():
        print(f"\n[decomp] {name}: {entry['successes']}/{entry['episodes']} = "
              f"{entry['success_rate']:.4f} (radius {entry['radius_m']} m)", flush=True)
        for stage, value in entry["funnel"].items():
            print(
                f"[decomp]   {stage:<18} {value['episodes']:>6} "
                f"({value['fraction_of_all']:.4f})",
                flush=True,
            )
        if entry["failure_taxonomy"]:
            print("[decomp]   failures by first blocking term:", flush=True)
            for label, value in entry["failure_taxonomy"].items():
                print(
                    f"[decomp]     {label:<20} {value['episodes']:>6} "
                    f"({value['fraction_of_failures']:.4f})",
                    flush=True,
                )
        print("[decomp]   conditional (each stage given the one above):", flush=True)
        for stage, value in entry["funnel_conditional"].items():
            print(f"[decomp]     {stage:<28} {value}", flush=True)
        diag = entry["no_release_diagnosis"]
        if diag["episodes"]:
            print(
                f"[decomp]   no_release: {diag['episodes']} episodes, "
                f"{diag['timed_out']} ran the whole budget "
                f"({diag['timed_out_fraction']:.4f}), "
                f"{diag['ended_holding']} ended still holding, "
                f"{diag['lost_grip']} dropped it; first grasp at "
                f"{diag['first_grasp_env_step']} of budget "
                f"{diag['env_step_budget']}",
                flush=True,
            )
        ejected = entry["spawn_outside_configured_range"]
        print(
            f"[decomp]   spawn outside {ejected['range_m']}: "
            f"{ejected['below_min']} below (workspace clamp), "
            f"{ejected['above_max']} ABOVE "
            f"({ejected['above_max_never_grasped']} never grasped) "
            f"at {ejected['above_max_distance_m']}",
            flush=True,
        )
        reset_split = entry["object_to_receptacle_at_reset_m"]
        print(
            f"[decomp]   spawn distance grasped={reset_split['grasped']} "
            f"never_grasped={reset_split['never_grasped']}",
            flush=True,
        )
        arrived = entry["released_inside_but_failed"]
        print(
            f"[decomp]   released INSIDE the radius and still failed: "
            f"{arrived['episodes']} ({arrived['fraction_of_failures']:.4f} of "
            f"failures); bounce {arrived['bounce_m']}",
            flush=True,
        )

    print(f"\n[decomp] wrote {output / 'failure_decomposition.json'}", flush=True)
    print(f"[decomp] wrote {output / 'container_episodes.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
