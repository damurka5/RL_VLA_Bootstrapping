#!/usr/bin/env python3
"""Was the object dropped, or did the grasp DETECTOR unlatch?

Phase 7's remaining loss is one transition: 140 of plate's 147 `no_release`
failures lose the grasp mid-carry without ever opening the gripper, and giving
plate the oracle's `release|grasp` alone puts composed plate at 0.7622. Four
hypotheses about that transition are dead (release height, horizon, grasp
speed, more composed demonstrations). This file attacks the fifth, which has
never been tested because the word "drop" has been doing two jobs at once.

WHAT `caught_target` ACTUALLY IS
-------------------------------
`caught_target` in a recording is not a latch. It is the LIVE per-step
`physical_grasp` that `_update_physical_grasp` writes and hands straight to the
success predicate, and `physical_grasp` is a five-term conjunction::

    grasp_eligible & active & bilateral_contact
                   & (left_normal_force  >= 0.05 N)
                   & (right_normal_force >= 0.05 N)
                   & stable_relative_pose      # <= 8 mm AND <= 0.15 rad
                                               # PER ENV STEP
    persisted >= 2 consecutive steps, and & ~release_open

So `lost_grip_env_step` -- already computed per episode by
`placement_failure_decomposition` -- fires on four physically distinct events:

    D1  bilateral_contact lost .......... the object really left the pads
    D2  contact kept, force < 0.05 N .... grazing / rolling contact
    D3  contact and force fine, but the
        object moved > 8 mm relative to
        the end effector IN ONE STEP .... the detector rejected a real grasp
    D4  the persistence counter zeroed .. the 2-step aftermath of any of those

**D3 is not a drop.** It is the grasp detector unlatching on a fast hand, and
it is a live candidate here for a reason that is already in the ledger: this
policy grasps and carries about five times faster than the scripted oracle
(first grasp p50 19-21 env steps against 90-104). `relative_position_slip` is a
raw per-env-step position difference, so carry speed enters it linearly. The
8 mm bar was calibrated on the CPU reference engine against a SCRIPTED lift --
0.46-3.52 mm, "expected reading is a reject rate at or near zero" -- which is
to say it was validated against the slow arm and never against the fast one.

If D3 dominates, composed plate is already near 0.76 and has been mis-scored,
and no amount of training touches it. That is worth knowing before any GPU is
spent, which is why this costs none.

WHY THIS NEEDS NO NEW INSTRUMENTATION
-------------------------------------
The five terms above are computed per step per world in production and then
immediately summed into run-level scalars, so they are not in any recording.
But the CONSEQUENCE of each is, because a recording already stores, per env
step: `caught_target`, `object_xyz` for every slot, `ee_xyz`,
`gripper_opening`, `actions`, and `active`. From those:

  * re-latch    -- does `caught_target` come back within a few steps with no
                   intervening release? Re-latching costs 2 consecutive
                   persistent steps, so a 2-4 step gap followed by recovery IS
                   the signature of a single slip-test trip. This one test can
                   confirm D3 outright.
  * separation  -- `d(t) = ||object - ee||` after the loss. Flat at its held
                   value means the object never left the hand.
  * fall        -- does the object descend to `support_surface_z +
                   target_rest_height`? An object that never falls was never
                   dropped.
  * slip proxy  -- `||r(t) - r(t-1)||` IS `relative_position_slip` minus its
                   orientation term. Its distribution over held steps can be
                   compared directly against the 8 mm bar.
  * carry speed -- end-effector displacement and commanded XYZ magnitude, read
                   AT the transition rather than as an episode-level average.
                   The dead grasp-speed hypothesis was an episode-level
                   correlate; this is the same variable asked at the moment it
                   would have to act.

Run it on the oracle's harvest rounds too (`--arm`). The oracle's
`release|grasp` is 0.9831 against the policy's 0.6449; if its held-step slip
sits an order of magnitude under the bar while the policy's straddles it, D3 is
confirmed as a speed-dependent property of the detector rather than a
behaviour of the policy.

WHAT THIS FILE DOES NOT RECOMPUTE
---------------------------------
The predicate. `blocking_stage`, `first_grasp_env_step` and
`lost_grip_env_step` are imported from `placement_failure_decomposition` rather
than re-derived, so the population analysed here is exactly the population that
tool reports and the two cannot drift apart. That file's own predicate guard
(`--max-predicate-disagreement`) runs here as well, for the same reason it
exists there: an analysis that does not reproduce the verdict it is analysing
describes a different task.

Conditioning on `no_release` is what makes the taxonomy safe. Within those
episodes the gripper never crossed the release bar for the whole episode, so a
`caught_target` False step cannot be a normal release seen from the side.

Usage::

    python tools/audit/grasp_loss_forensics.py \\
        --config configs/examples/cdpr_smolvla_phase7_sparse_joint.yaml \\
        --arm policy=runs/phase7_eval/record_*.npz \\
        --arm oracle=runs/phase4_bank/o6_*/record_*.npz \\
        --output runs/phase7_eval/grasp_loss
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

from tools.audit.sil_record import _Recording, _write_csv  # noqa: E402
from tools.audit.placement_failure_decomposition import (  # noqa: E402
    CONTAINER_INSTRUCTIONS,
    _Thresholds,
    _episode_terms,
    _percentiles,
)

# Duplicated from `mjwarp_rank_local_collector`, and CHECKED against it below
# rather than trusted. A slip bar read from the wrong place would move every
# "over the bar" count in this file without changing anything else about the
# output, which is the silent way to be confidently wrong here.
_FALLBACK_SLIP_BAR_M = 0.008
_FALLBACK_PERSISTENCE_STEPS = 2
_FALLBACK_MIN_PAD_FORCE_N = 0.05


def _collector_constants() -> tuple[float, int, float, str]:
    """The production grasp constants, imported when the import is possible.

    The collector pulls in torch and the mjwarp backend, which a CPU-only audit
    box may not have. Falling back is fine; falling back SILENTLY is not, so
    the provenance travels into the summary either way.
    """

    try:
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            _GRASP_MAX_RELATIVE_POSITION_SLIP_M,
            _GRASP_MIN_PAD_FORCE_N,
            _GRASP_PERSISTENCE_STEPS,
        )
    except Exception as error:  # pragma: no cover - depends on host env
        return (
            _FALLBACK_SLIP_BAR_M,
            _FALLBACK_PERSISTENCE_STEPS,
            _FALLBACK_MIN_PAD_FORCE_N,
            f"duplicated literals ({type(error).__name__}: {error})",
        )
    return (
        float(_GRASP_MAX_RELATIVE_POSITION_SLIP_M),
        int(_GRASP_PERSISTENCE_STEPS),
        float(_GRASP_MIN_PAD_FORCE_N),
        "imported from mjwarp_rank_local_collector",
    )


# Causal order. The first label that applies wins, and the order is not
# arbitrary: `relatched` comes first because a grasp that comes back is
# evidence about the DETECTOR that survives whatever happens later in the
# episode, and `censored` comes last so that positive evidence of separation or
# of a fall is never discarded just because the episode ended soon after.
FIRST_LOSS_LABELS = (
    "relatched",
    "separated_and_fell",
    "separated_no_fall",
    "held_no_relatch",
    "censored",
)

# What the episode's object was doing when the recording stopped. Orthogonal to
# the label above, which is about the FIRST loss: an episode can chatter once,
# re-latch, and still put the object on the desk later.
OUTCOME_LABELS = ("ended_holding", "object_at_rest_height", "object_above_rest")


class _Params:
    """The four judgement calls, all exposed and all reported."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.relatch_steps = int(args.relatch_steps)
        self.window_steps = int(args.window_steps)
        self.separation_m = float(args.separation_m)
        self.min_steps_after = int(args.min_steps_after)
        self.slip_bar_m = float(args.slip_bar)

    def as_dict(self) -> dict[str, Any]:
        return {
            "relatch_steps": self.relatch_steps,
            "window_steps": self.window_steps,
            "separation_m": self.separation_m,
            "min_steps_after": self.min_steps_after,
            "slip_bar_m": self.slip_bar_m,
        }


def _runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end) runs of True. Used to count grasp chatter."""

    idx = np.flatnonzero(mask.astype(bool))
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _loss_rows(
    recording: _Recording,
    thresholds: _Thresholds,
    params: _Params,
    base_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One row per container episode that ever grasped.

    Every quantity is derived from arrays the recording already holds. Nothing
    here re-decides whether an episode succeeded or which stage blocked it --
    those come in on `base_rows` from the decomposition.
    """

    worlds = recording.worlds
    world_index = np.arange(worlds)
    target_all = recording.object_xyz[:, world_index, recording.target_slots, :]
    active = recording.active.astype(bool)
    caught = recording.caught_target.astype(bool)
    ever_grasped_t = np.logical_or.accumulate(caught & active, axis=0)
    release_bar = np.maximum(
        recording.release_threshold.astype(np.float64),
        float(thresholds.release_opening),
    )
    opening = recording.gripper_opening.astype(np.float64)
    desk = (
        recording.support_surface_z.astype(np.float64)
        + recording.target_rest_height.astype(np.float64)
    )

    rows: list[dict[str, Any]] = []
    for base in base_rows:
        world = int(base["world"])
        first_grasp = int(base["first_grasp_env_step"])
        if first_grasp < 0:
            continue

        live = active[:, world]
        live_idx = np.flatnonzero(live)
        if live_idx.size == 0:
            continue
        last = int(live_idx[-1])
        # WHY the episode stopped, which is what `censored` cannot say on its
        # own. `terminated` is stored per step, so its value at the last live
        # step separates "the predicate ended this episode" from "the budget
        # ran out" -- and those want opposite work. Carried rather than
        # re-derived: `timed_out` is the decomposition's own column.
        terminated_at_end = bool(recording.terminated[last, world])

        target = target_all[:, world, :].astype(np.float64)
        ee = recording.ee_xyz[:, world, :].astype(np.float64)
        # The object's offset from the end effector. Its STEP-TO-STEP change is
        # `relative_position_slip` without the orientation term -- the same
        # quantity the 8 mm stability test reads, recomputed from stored
        # positions rather than guessed at.
        relative = target - ee
        distance = np.linalg.norm(relative, axis=-1)
        slip = np.full(distance.shape, np.nan, dtype=np.float64)
        slip[1:] = np.linalg.norm(np.diff(relative, axis=0), axis=-1)
        ee_speed = np.full(distance.shape, np.nan, dtype=np.float64)
        ee_speed[1:] = np.linalg.norm(np.diff(ee, axis=0), axis=-1)
        # `actions[t]` is appended BEFORE `backend.step` and the predicate row
        # AFTER it, so `actions[t]` is the action that produced the state at
        # index t. Channel 4 is the gripper delta; 0..2 are the XYZ command.
        command = np.linalg.norm(
            recording.actions[:, world, 0:3].astype(np.float64), axis=-1
        )
        grip_command = recording.actions[:, world, 4].astype(np.float64)
        height = target[:, 2] - float(desk[world])

        held_mask = caught[:, world] & live
        loss_mask = (~caught[:, world]) & live & ever_grasped_t[:, world]
        loss_runs = _runs_of_true(loss_mask)
        held_runs = _runs_of_true(held_mask)

        first_loss = int(loss_runs[0][0]) if loss_runs else -1
        final_loss = int(loss_runs[-1][0]) if loss_runs else -1

        # The held window used for the slip baseline stops at the first loss:
        # steps after it are a different regime and would blur the very
        # comparison this is for.
        hold_idx = np.flatnonzero(held_mask)
        if first_loss >= 0:
            baseline_idx = hold_idx[hold_idx < first_loss]
        else:
            baseline_idx = hold_idx
        if baseline_idx.size == 0:
            baseline_idx = np.array([first_grasp], dtype=np.int64)
        distance_held = float(np.median(distance[baseline_idx]))
        slip_held = slip[baseline_idx]
        slip_held = slip_held[np.isfinite(slip_held)]
        speed_held = ee_speed[baseline_idx]
        speed_held = speed_held[np.isfinite(speed_held)]

        def _pct(values: np.ndarray, q: float) -> float:
            return (
                float(np.percentile(values, q)) if values.size else float("nan")
            )

        row: dict[str, Any] = {
            "world": world,
            "instruction": base["instruction"],
            "blocking_stage": base["blocking_stage"],
            "recomputed_success": base["recomputed_success"],
            "grasped_at_reset": base["grasped_at_reset"],
            "first_grasp_env_step": first_grasp,
            "first_loss_env_step": first_loss,
            "final_loss_env_step": final_loss,
            "loss_events": len(loss_runs),
            "held_runs": len(held_runs),
            "held_steps": int(held_mask.sum()),
            "env_steps_active": int(live.sum()),
            "last_active_env_step": last,
            "terminated_at_end": terminated_at_end,
            "timed_out": bool(base["timed_out"]),
            "ended_holding": bool(base["ended_holding"]),
            "env_step_budget": int(base["env_step_budget"]),
            # The slip distribution while genuinely holding, against the bar
            # the detector applies to it. This is the D3 evidence.
            "slip_held_p50_m": round(_pct(slip_held, 50), 6),
            "slip_held_p90_m": round(_pct(slip_held, 90), 6),
            "slip_held_max_m": round(
                float(slip_held.max()) if slip_held.size else float("nan"), 6
            ),
            "held_steps_over_slip_bar": int(
                (slip_held > params.slip_bar_m).sum()
            ),
            "held_steps_measured": int(slip_held.size),
            "ee_speed_held_p50_m": round(_pct(speed_held, 50), 6),
            "distance_held_m": round(distance_held, 6),
        }

        if first_loss < 0:
            row.update(
                {
                    "first_loss_class": "",
                    "relatch_gap_steps": -1,
                    "slip_at_loss_m": float("nan"),
                    "ee_speed_at_loss_m": float("nan"),
                    "command_xyz_at_loss": float("nan"),
                    "gripper_command_at_loss": float("nan"),
                    "distance_excess_m": float("nan"),
                    "min_height_after_loss_m": float("nan"),
                    "steps_live_after_loss": 0,
                    "steps_loss_to_end": -1,
                    "height_at_loss_m": float("nan"),
                    "episode_outcome": _outcome(
                        held_mask, height, last, thresholds
                    ),
                }
            )
            rows.append(row)
            continue

        # Did it come back? A re-latch needs `_GRASP_PERSISTENCE_STEPS`
        # consecutive good steps, so the gap is bounded below by that and a
        # short gap is the slip test tripping once. Only counted when the
        # gripper never crossed the release bar in between -- otherwise the
        # "re-latch" would be a re-grasp after a deliberate open.
        window_end = min(first_loss + params.relatch_steps + 1, last + 1)
        relatch_gap = -1
        for step in range(first_loss + 1, window_end):
            if not live[step]:
                break
            if opening[step, world] >= release_bar[world]:
                break
            if caught[step, world]:
                relatch_gap = int(step - first_loss)
                break

        after_end = min(first_loss + params.window_steps + 1, last + 1)
        after_idx = np.flatnonzero(live[first_loss:after_end]) + first_loss
        steps_after = int(max(after_end - first_loss - 1, 0))
        distance_excess = (
            float(distance[after_idx].max()) - distance_held
            if after_idx.size
            else float("nan")
        )
        # Whether the object ever reached resting height, measured to the END
        # of the episode rather than to the end of the window: `wrong_place_
        # settled` terminates an episode shortly after the object lands, so a
        # window-bounded fall test would systematically miss real drops.
        tail_idx = np.arange(first_loss, last + 1)[live[first_loss : last + 1]]
        min_height_after = (
            float(height[tail_idx].min()) if tail_idx.size else float("nan")
        )
        fell = bool(
            np.isfinite(min_height_after)
            and min_height_after <= float(thresholds.settle_margin)
        )
        separated = bool(
            np.isfinite(distance_excess)
            and distance_excess > params.separation_m
        )

        if relatch_gap >= 0:
            label = "relatched"
        elif separated and fell:
            label = "separated_and_fell"
        elif separated:
            label = "separated_no_fall"
        elif steps_after >= params.min_steps_after:
            label = "held_no_relatch"
        else:
            label = "censored"

        row.update(
            {
                "first_loss_class": label,
                "relatch_gap_steps": relatch_gap,
                "slip_at_loss_m": round(float(slip[first_loss]), 6),
                "ee_speed_at_loss_m": round(float(ee_speed[first_loss]), 6),
                "command_xyz_at_loss": round(float(command[first_loss]), 6),
                "gripper_command_at_loss": round(
                    float(grip_command[first_loss]), 6
                ),
                "distance_excess_m": round(distance_excess, 6),
                "min_height_after_loss_m": round(min_height_after, 6),
                "steps_live_after_loss": steps_after,
                # Unclipped by --window-steps, so a censored episode reports how
                # much episode was actually left rather than the window's floor.
                "steps_loss_to_end": int(last - first_loss),
                "height_at_loss_m": round(float(height[first_loss]), 6),
                "episode_outcome": _outcome(held_mask, height, last, thresholds),
            }
        )
        rows.append(row)
    return rows


def _outcome(
    held_mask: np.ndarray,
    height: np.ndarray,
    last: int,
    thresholds: _Thresholds,
) -> str:
    """Where the object was when the recording stopped.

    Deliberately orthogonal to `first_loss_class`: an episode can chatter once,
    re-latch, and still put the object on the desk twenty steps later, and
    collapsing the two would hide exactly that case.
    """

    if bool(held_mask[last]):
        return "ended_holding"
    if float(height[last]) <= float(thresholds.settle_margin):
        return "object_at_rest_height"
    return "object_above_rest"


def _taxonomy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    losses = [row for row in rows if int(row["first_loss_env_step"]) >= 0]
    out: dict[str, Any] = {
        "grasped_episodes": len(rows),
        "episodes_with_grasp_loss": len(losses),
        "loss_rate_given_grasp": (
            round(len(losses) / len(rows), 4) if rows else None
        ),
    }
    classes: dict[str, Any] = {}
    for label in FIRST_LOSS_LABELS:
        count = sum(1 for row in losses if row["first_loss_class"] == label)
        classes[label] = {
            "episodes": count,
            "fraction_of_losses": (
                round(count / len(losses), 4) if losses else None
            ),
        }
    out["first_loss_class"] = classes
    outcomes: dict[str, Any] = {}
    for label in OUTCOME_LABELS:
        count = sum(1 for row in rows if row["episode_outcome"] == label)
        outcomes[label] = {
            "episodes": count,
            "fraction_of_grasped": (
                round(count / len(rows), 4) if rows else None
            ),
        }
    out["episode_outcome"] = outcomes
    # Chatter, independent of how any single loss is labelled: an episode with
    # several loss events and several held runs did not "drop the object", it
    # lost and regained a detector latch repeatedly.
    multi = [row for row in losses if int(row["loss_events"]) > 1]
    out["multiple_loss_events"] = {
        "episodes": len(multi),
        "fraction_of_losses": (
            round(len(multi) / len(losses), 4) if losses else None
        ),
        "loss_events": _percentiles(
            np.array([row["loss_events"] for row in losses], dtype=np.float64)
        ),
    }
    out["relatch_gap_steps"] = _percentiles(
        np.array(
            [
                row["relatch_gap_steps"]
                for row in losses
                if int(row["relatch_gap_steps"]) >= 0
            ],
            dtype=np.float64,
        )
    )
    # Which population each distribution is over, stated rather than inferred
    # from the field name: a "slip while held" percentile is about every
    # grasped episode, and a "slip at the loss step" percentile only exists for
    # the ones that lost the latch. Mixing the two denominators would make the
    # oracle contrast unreadable.
    over_grasped = (
        "slip_held_p50_m",
        "slip_held_p90_m",
        "slip_held_max_m",
        "ee_speed_held_p50_m",
        "distance_held_m",
        "held_steps",
        "loss_events",
    )
    over_losses = (
        "slip_at_loss_m",
        "ee_speed_at_loss_m",
        "command_xyz_at_loss",
        "gripper_command_at_loss",
        "distance_excess_m",
        "min_height_after_loss_m",
        "height_at_loss_m",
        "steps_live_after_loss",
        "steps_loss_to_end",
        "first_loss_env_step",
    )
    for field, source in (
        [(name, rows) for name in over_grasped]
        + [(name, losses) for name in over_losses]
    ):
        out[field] = _percentiles(
            np.array([float(row[field]) for row in source], dtype=np.float64)
        )
    # What ended the episodes the separation test could not see. `censored`
    # is not a mechanism, it is a measurement that ran out of episode -- so the
    # only useful thing to report about it is why the episode stopped.
    censored = [row for row in losses if row["first_loss_class"] == "censored"]
    out["censored_diagnosis"] = {
        "episodes": len(censored),
        "terminated_at_end": sum(
            1 for row in censored if row["terminated_at_end"]
        ),
        "timed_out": sum(1 for row in censored if row["timed_out"]),
        "ended_holding": sum(1 for row in censored if row["ended_holding"]),
        "neither": sum(
            1
            for row in censored
            if not row["terminated_at_end"] and not row["timed_out"]
        ),
        "steps_loss_to_end": _percentiles(
            np.array(
                [float(row["steps_loss_to_end"]) for row in censored],
                dtype=np.float64,
            )
        ),
        "height_at_loss_m": _percentiles(
            np.array(
                [float(row["height_at_loss_m"]) for row in censored],
                dtype=np.float64,
            )
        ),
    }
    out["terminated_at_end"] = {
        "episodes": sum(1 for row in rows if row["terminated_at_end"]),
        "of_grasped": len(rows),
    }
    # The single number the D3 question turns on: how much of the time a
    # genuinely-held step is already over the bar the detector applies to it.
    measured = sum(int(row["held_steps_measured"]) for row in rows)
    over = sum(int(row["held_steps_over_slip_bar"]) for row in rows)
    out["held_steps_over_slip_bar"] = {
        "held_steps_measured": measured,
        "over_bar": over,
        "fraction": round(over / measured, 5) if measured else None,
    }
    return out


def _scene_fingerprint(recording: _Recording, world: int) -> tuple:
    """What makes two episodes in different arms the SAME scene.

    The reset is a pure function of `base_seed/rank/update_index/round_index`,
    so matching round index and world index SHOULD mean matching scene -- but
    "should" is how a paired comparison silently pairs the wrong episodes when
    an arm was run at a different cap or composed fraction. The object layout
    one env step after reset is the physical check on that assumption.
    """

    return (
        int(recording.round_index),
        int(world),
        int(recording.instruction_ids[world]),
        int(recording.target_slots[world]),
        tuple(np.round(recording.object_xyz[0, world].ravel(), 4).tolist()),
    )


def _paired(arms: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Per-arm statistics restricted to scenes every arm actually ran.

    This is the comparison the question wants -- the oracle's grasps against
    the policy's ON THE SAME SCENES -- and it is reported separately from the
    marginal tables because an unmatched comparison between a policy run and an
    oracle harvest is a comparison of two different start distributions.
    """

    names = list(arms)
    if len(names) < 2:
        return {"arms": names, "note": "paired table needs at least two arms"}
    keyed = {
        name: {row["_scene"]: row for row in rows} for name, rows in arms.items()
    }
    shared = set(keyed[names[0]])
    for name in names[1:]:
        shared &= set(keyed[name])
    out: dict[str, Any] = {
        "arms": names,
        "scenes_per_arm": {name: len(keyed[name]) for name in names},
        "scenes_shared": len(shared),
    }
    if not shared:
        out["note"] = (
            "No scene matched across arms. Same --round-index, --worlds, "
            "--start-distance-cap and composed fraction are required for the "
            "reset to be the same function of its seed."
        )
        return out
    per_arm: dict[str, Any] = {}
    for name in names:
        subset = [keyed[name][key] for key in sorted(shared)]
        losses = [r for r in subset if int(r["first_loss_env_step"]) >= 0]
        per_arm[name] = {
            "grasped_episodes": len(subset),
            "episodes_with_grasp_loss": len(losses),
            "loss_rate_given_grasp": (
                round(len(losses) / len(subset), 4) if subset else None
            ),
            "relatched": sum(
                1 for r in losses if r["first_loss_class"] == "relatched"
            ),
            "separated_and_fell": sum(
                1 for r in losses if r["first_loss_class"] == "separated_and_fell"
            ),
            "slip_held_p50_m": _percentiles(
                np.array(
                    [float(r["slip_held_p50_m"]) for r in subset],
                    dtype=np.float64,
                )
            ),
            "slip_held_p90_m": _percentiles(
                np.array(
                    [float(r["slip_held_p90_m"]) for r in subset],
                    dtype=np.float64,
                )
            ),
            "ee_speed_held_p50_m": _percentiles(
                np.array(
                    [float(r["ee_speed_held_p50_m"]) for r in subset],
                    dtype=np.float64,
                )
            ),
        }
    out["by_arm"] = per_arm
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=GLOB",
        help=(
            "Repeatable. LABEL=GLOB, e.g. policy=runs/eval/record_*.npz and "
            "oracle=runs/phase4_bank/o6_*/record_*.npz. Two arms at the same "
            "--round-index give the same scenes and enable the paired table."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Project config these recordings were scored under.",
    )
    parser.add_argument(
        "--metadata-override",
        nargs="*",
        default=(),
        help="KEY=VALUE, applied as in sil_record/placement_failure_decomposition.",
    )
    parser.add_argument(
        "--relatch-steps",
        type=int,
        default=10,
        help=(
            "How long after the first loss a returning grasp still counts as a "
            "re-latch rather than a fresh grasp. A re-latch costs at least "
            "_GRASP_PERSISTENCE_STEPS, so the informative range starts at 2."
        ),
    )
    parser.add_argument(
        "--window-steps",
        type=int,
        default=16,
        help="Steps after the first loss used for the separation test.",
    )
    parser.add_argument(
        "--separation-m",
        type=float,
        default=0.02,
        help=(
            "How far above its held value the object-to-end-effector distance "
            "must rise to count as the object having left the hand. The raw "
            "distance_excess_m is in the CSV so this can be re-chosen without "
            "re-running."
        ),
    )
    parser.add_argument(
        "--min-steps-after",
        type=int,
        default=4,
        help=(
            "Live steps needed after the loss before `held_no_relatch` may be "
            "asserted. Below this the episode is `censored`: terminating on "
            "wrong_place_settled right after a drop is common, and calling "
            "those held would invent the finding."
        ),
    )
    parser.add_argument(
        "--slip-bar",
        type=float,
        default=None,
        help=(
            "Override the stability bar the held-step slip is counted against. "
            "Defaults to the collector's _GRASP_MAX_RELATIVE_POSITION_SLIP_M."
        ),
    )
    parser.add_argument(
        "--max-predicate-disagreement",
        type=int,
        default=0,
        help=(
            "Worlds allowed to disagree with the recording's own latched "
            "success before this refuses to report."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    slip_bar, persistence, min_pad_force, provenance = _collector_constants()
    if args.slip_bar is not None:
        slip_bar = float(args.slip_bar)
        provenance = f"{provenance}; overridden by --slip-bar"
    args.slip_bar = slip_bar
    params = _Params(args)

    from rl_vla_bootstrapping.core.config import load_project_config

    project = load_project_config(Path(args.config))
    metadata = dict(project.task.metadata or {})
    for override in args.metadata_override or ():
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
        print(f"[grasploss] metadata override {key}={metadata[key]!r}", flush=True)

    thresholds = _Thresholds(metadata)
    print(
        f"[grasploss] grasp constants {provenance}: slip bar {slip_bar} m, "
        f"persistence {persistence} steps, min pad force {min_pad_force} N",
        flush=True,
    )
    print(f"[grasploss] params {json.dumps(params.as_dict())}", flush=True)
    # The settle margin is the fall test's bar, so a height percentile printed
    # without it beside it cannot be read.
    print(
        f"[grasploss] thresholds {json.dumps(thresholds.as_dict())}", flush=True
    )

    arms: dict[str, list[dict[str, Any]]] = {}
    arm_files: dict[str, list[str]] = {}
    # Per arm, not pooled. Arms are usually different runs scored under
    # different configs -- a policy eval beside an oracle harvest -- so one
    # mismatched arm pooled into a single counter kills the whole invocation
    # without saying which arm is wrong, and the natural next move is to raise
    # the allowance, which switches the guard off for the arms that were fine.
    disagreements: dict[str, int] = {}
    rows_seen: dict[str, int] = {}
    for entry in args.arm:
        label, sep, pattern = str(entry).partition("=")
        if not sep or not label or not pattern:
            raise SystemExit(f"--arm expects LABEL=GLOB, got {entry!r}")
        paths: list[Path] = []
        for part in pattern.split(","):
            expanded = sorted(_glob.glob(part.strip()))
            if not expanded:
                raise SystemExit(f"No recordings matched {part.strip()!r}.")
            paths.extend(Path(p) for p in expanded)
        collected: list[dict[str, Any]] = []
        disagreements.setdefault(label, 0)
        rows_seen.setdefault(label, 0)
        for path in paths:
            recording = _Recording.from_npz(path)
            base = _episode_terms(recording, thresholds)
            rows_seen[label] += len(base)
            bad = [
                row
                for row in base
                if row["recorded_success"] != row["recomputed_success"]
            ]
            disagreements[label] += len(bad)
            found = _loss_rows(recording, thresholds, params, base)
            fingerprints = {
                int(row["world"]): _scene_fingerprint(recording, int(row["world"]))
                for row in found
            }
            for row in found:
                row["arm"] = label
                row["recording"] = path.name
                row["round_index"] = int(recording.round_index)
                row["start_distance_cap"] = (
                    None
                    if not np.isfinite(recording.start_distance_cap)
                    else round(float(recording.start_distance_cap), 4)
                )
                row["_scene"] = fingerprints[int(row["world"])]
            collected.extend(found)
            print(
                f"[grasploss] {label}/{path.name}: {len(base)} container "
                f"episodes, {len(found)} ever grasped",
                flush=True,
            )
        if not collected:
            raise SystemExit(
                f"Arm {label!r} has no container episode that ever grasped. "
                "Point it at a composed put_into harvest."
            )
        arms[label] = collected
        arm_files[label] = [str(p) for p in paths]

    failed = {
        label: count
        for label, count in disagreements.items()
        if count > int(args.max_predicate_disagreement)
    }
    for label, count in disagreements.items():
        if not count:
            continue
        print(
            f"[grasploss] PREDICATE DISAGREEMENT in arm {label!r}: {count} of "
            f"{rows_seen[label]} container worlds. The terms recomputed here "
            "do not reproduce the verdict that arm latched, so --config is "
            "most likely not the one THAT arm was scored under.",
            flush=True,
        )
    if failed:
        raise SystemExit(
            "Arms over --max-predicate-disagreement "
            f"{args.max_predicate_disagreement}: "
            + ", ".join(f"{label} ({count})" for label, count in failed.items())
            + ". Run each of those alone under the config its own "
            "failure_decomposition.json names; the arms not listed here "
            "reproduced their verdicts and are fine."
        )

    summary: dict[str, Any] = {
        "config": str(args.config),
        "arms": arm_files,
        "thresholds": thresholds.as_dict(),
        "grasp_constants": {
            "provenance": provenance,
            "slip_bar_m": slip_bar,
            "persistence_steps": persistence,
            "min_pad_force_n": min_pad_force,
        },
        "params": params.as_dict(),
        "predicate_disagreements": dict(disagreements),
        "container_episodes": dict(rows_seen),
        "by_arm": {},
    }
    for label, rows in arms.items():
        entry: dict[str, Any] = {}
        for name in CONTAINER_INSTRUCTIONS:
            subset = [row for row in rows if row["instruction"] == name]
            if not subset:
                continue
            no_release = [
                row for row in subset if row["blocking_stage"] == "no_release"
            ]
            entry[name] = {
                "all_grasped": _taxonomy(subset),
                # The ledger's population: 140 of plate's 147 no_release
                # failures. Conditioning here is what makes a caught_target
                # False step unambiguous -- the gripper never crossed the
                # release bar in these episodes at all.
                "no_release_failures": (
                    _taxonomy(no_release) if no_release else None
                ),
            }
        summary["by_arm"][label] = entry
    summary["paired_on_shared_scenes"] = _paired(arms)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    flat = [
        {key: value for key, value in row.items() if key != "_scene"}
        for rows in arms.values()
        for row in rows
    ]
    _write_csv(output / "grasp_loss_episodes.csv", flat)
    (output / "grasp_loss_forensics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for label, entry in summary["by_arm"].items():
        for name, blocks in entry.items():
            for scope, block in blocks.items():
                if not block:
                    continue
                print(
                    f"\n[grasploss] {label} / {name} / {scope}: "
                    f"{block['episodes_with_grasp_loss']} grasp losses in "
                    f"{block['grasped_episodes']} grasped episodes "
                    f"(rate {block['loss_rate_given_grasp']})",
                    flush=True,
                )
                for cls, value in block["first_loss_class"].items():
                    print(
                        f"[grasploss]   {cls:<20} {value['episodes']:>6} "
                        f"({value['fraction_of_losses']})",
                        flush=True,
                    )
                multi = block["multiple_loss_events"]
                print(
                    f"[grasploss]   chatter: {multi['episodes']} episodes lost "
                    f"the latch more than once ({multi['fraction_of_losses']}); "
                    f"relatch gap {block['relatch_gap_steps']}",
                    flush=True,
                )
                cens = block["censored_diagnosis"]
                if cens["episodes"]:
                    print(
                        f"[grasploss]   censored: {cens['episodes']} episodes "
                        f"ended within the window -- "
                        f"{cens['terminated_at_end']} terminated by the "
                        f"predicate, {cens['timed_out']} ran the whole budget, "
                        f"{cens['ended_holding']} ended still holding, "
                        f"{cens['neither']} neither; steps from loss to end "
                        f"{cens['steps_loss_to_end']}; object height at the "
                        f"loss {cens['height_at_loss_m']}",
                        flush=True,
                    )
                bar = block["held_steps_over_slip_bar"]
                print(
                    f"[grasploss]   held steps over the {slip_bar} m bar: "
                    f"{bar['over_bar']}/{bar['held_steps_measured']} "
                    f"({bar['fraction']})",
                    flush=True,
                )
                print(
                    f"[grasploss]   slip while held p50 "
                    f"{block['slip_held_p50_m']}",
                    flush=True,
                )
                print(
                    f"[grasploss]   slip at the loss step "
                    f"{block['slip_at_loss_m']}",
                    flush=True,
                )
                print(
                    f"[grasploss]   distance excess after the loss "
                    f"{block['distance_excess_m']}",
                    flush=True,
                )
                print(
                    f"[grasploss]   object height at rest afterwards "
                    f"{block['min_height_after_loss_m']}",
                    flush=True,
                )

    paired = summary["paired_on_shared_scenes"]
    print(
        f"\n[grasploss] paired: {paired.get('scenes_shared')} scenes shared by "
        f"{paired.get('arms')}",
        flush=True,
    )
    if paired.get("note"):
        print(f"[grasploss]   {paired['note']}", flush=True)
    for label, block in (paired.get("by_arm") or {}).items():
        print(
            f"[grasploss]   {label:<10} loss rate "
            f"{block['loss_rate_given_grasp']}, relatched {block['relatched']}, "
            f"separated_and_fell {block['separated_and_fell']}, "
            f"slip_held_p50 {block['slip_held_p50_m']}",
            flush=True,
        )

    print(f"\n[grasploss] wrote {output / 'grasp_loss_forensics.json'}", flush=True)
    print(f"[grasploss] wrote {output / 'grasp_loss_episodes.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
