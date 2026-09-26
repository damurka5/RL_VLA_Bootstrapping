#!/usr/bin/env python3
"""Harvest the policy's own strict, end-to-end ``put_into`` successes.

This is deliberately not the staged demonstration collector. One checkpoint
receives one final ``put_into_plate`` or ``put_into_bowl`` prompt at reset and
runs continuously from the empty-hand scene until the production predicate
terminates it or the decision budget expires. There is no teacher, stage
switch, servo, pose restore, settle window, or simulator recovery.

Every round writes three independently auditable artifacts:

``attempts_<stem>.npz``
    Lightweight outcomes for every distinct scene attempted.
``record_<stem>.npz``
    States, priors, executed actions, masks, event steps, and provenance for
    strict successes only.
``frames_<stem>.npz``
    The exact pre-decision overview and wrist images for those successes,
    keyed by episode id. Omit only with ``--no-frames``.
``retention_<stem>.npz`` / ``retention_frames_<stem>.npz``
    Only with ``--retention-rows-per-stage K``. For every episode that is NOT
    a strict success but did grasp: up to K evenly spaced decisions of each
    stage it completed -- the approach if it grasped, the pickup if it also
    lifted -- with the pictures for exactly those decisions. This is retention
    material recorded under the same controller, prompt and action contract as
    the strict bank. Placement rows come only from strict episodes (surplus
    ones, at build time): a placement that was not strict is a slip or a drop,
    not behaviour worth preserving.

The collector does not choose the training subset. Pool shards and enforce an
exact object x destination quota with
``build_cdpr_full_put_into_dataset.py``; separating collection from selection
keeps an incomplete cell from being silently filled with an easier one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.evaluate_cdpr_full_put_into import run_unassisted  # noqa: E402
from tools.audit.xy_approach_probe import _build_world  # noqa: E402
from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (  # noqa: E402
    destination_instruction_id,
    object_label,
    student_instruction_text,
)
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (  # noqa: E402
    FullTaskSceneResetter,
)
from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import (  # noqa: E402
    SPLIT_NAMES,
    read_manifest,
    select_split,
)


SCHEMA = "cdpr_full_put_into_policy_round/v1"
RETENTION_SCHEMA = "cdpr_full_put_into_retention_round/v1"


def spread_evenly(values: np.ndarray, count: int) -> np.ndarray:
    """At most ``count`` entries of ``values``, evenly spaced, ends included."""

    values = np.asarray(values, dtype=np.int64)
    if int(count) <= 0 or values.size == 0:
        return values[:0]
    if values.size <= int(count):
        return values
    index = np.linspace(0, values.size - 1, int(count)).round().astype(np.int64)
    return np.unique(values[index])


def stage_decisions(
    *,
    active_decisions: int,
    grasp_step: int,
    lift_step: int,
    per: int,
    include_placement: bool,
) -> dict[int, np.ndarray]:
    """Decisions of each COMPLETED stage, by the dataset builder's boundaries.

    move_to (0) is every decision before the grasp decision; pick_up (1) runs
    from the grasp decision through the lift decision; placement (2) is the
    rest. A stage is returned only when it was completed: no grasp, no stages
    at all; no lift, the approach only. Placement is returned only on request,
    because only a strict episode's placement is worth preserving.
    """

    out: dict[int, np.ndarray] = {}
    active = int(active_decisions)
    if int(grasp_step) < 0 or active <= 0:
        return out
    grasp_decision = int(grasp_step) // int(per)
    out[0] = np.arange(0, min(grasp_decision, active), dtype=np.int64)
    if int(lift_step) >= 0:
        lift_decision = int(lift_step) // int(per)
        out[1] = np.arange(grasp_decision, min(lift_decision + 1, active), dtype=np.int64)
        if include_placement:
            out[2] = np.arange(lift_decision + 1, active, dtype=np.int64)
    return {stage: rows for stage, rows in out.items() if rows.size}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_batches(
    scenes: Sequence[Any],
    *,
    worlds: int,
    rounds: int,
    shard: int,
    num_shards: int,
    scene_offset: int = 0,
) -> list[list[Any]]:
    """Assign each manifest scene to at most one shard and one episode."""

    if int(worlds) < 2 or int(worlds) % 2:
        raise ValueError("--worlds must be an even integer >= 2.")
    if int(rounds) < 1:
        raise ValueError("--rounds must be positive.")
    if int(num_shards) < 1 or not 0 <= int(shard) < int(num_shards):
        raise ValueError(f"Invalid shard {shard}/{num_shards}.")
    if int(scene_offset) < 0:
        raise ValueError("--scene-offset must be non-negative.")

    # Offset before sharding, so a continuation consumes the same global scene
    # prefix regardless of how many GPUs it uses.
    remaining = list(scenes)[int(scene_offset) :]
    mine = [
        scene
        for index, scene in enumerate(remaining)
        if index % int(num_shards) == int(shard)
    ]
    needed = int(worlds) * int(rounds)
    if len(mine) < needed:
        raise ValueError(
            f"Shard {shard} has {len(mine)} scenes after offset "
            f"{scene_offset}, but {needed} are required. Lower --rounds or "
            "generate a larger manifest."
        )
    chosen = mine[:needed]
    uids = [str(scene.scene_uid) for scene in chosen]
    if len(set(uids)) != len(uids):
        raise ValueError("The planned batch repeats a scene_uid.")
    return [
        chosen[start : start + int(worlds)]
        for start in range(0, needed, int(worlds))
    ]


class FullTaskTrace:
    """Preallocated host capture for one unassisted rollout round."""

    def __init__(
        self,
        *,
        decisions: int,
        worlds: int,
        actions_per_decision: int,
        record_frames: bool,
    ) -> None:
        self.decisions = int(decisions)
        self.worlds = int(worlds)
        self.actions_per_decision = int(actions_per_decision)
        self.record_frames = bool(record_frames)
        self.used_decisions = 0

        self.state: np.ndarray | None = None
        self.prior: np.ndarray | None = None
        self.overview: np.ndarray | None = None
        self.wrist: np.ndarray | None = None
        self.decision_active = np.zeros(
            (self.decisions, self.worlds), dtype=bool
        )
        self.action = np.zeros(
            (
                self.decisions,
                self.worlds,
                self.actions_per_decision,
                5,
            ),
            dtype=np.float32,
        )
        self.action_mask = np.zeros(
            (self.decisions, self.worlds, self.actions_per_decision),
            dtype=bool,
        )
        self.first_grasp_step = np.full(self.worlds, -1, dtype=np.int64)
        self.first_lift_step = np.full(self.worlds, -1, dtype=np.int64)
        self.first_release_step = np.full(self.worlds, -1, dtype=np.int64)
        self.first_strict_step = np.full(self.worlds, -1, dtype=np.int64)

    @staticmethod
    def _host(value: Any, *, dtype: Any | None = None) -> np.ndarray:
        array = value.detach().cpu().numpy()
        return np.asarray(array, dtype=dtype)

    @staticmethod
    def _frame(value: Any) -> np.ndarray:
        image = (value.clamp(0.0, 1.0) * 255.0).round()
        return np.asarray(
            image.permute(0, 2, 3, 1).float().cpu().numpy(),
            dtype=np.uint8,
        )

    def record_decision(
        self,
        *,
        decision_index: int,
        cameras: Any,
        state: Any,
        prior: Any,
        active: Any,
    ) -> None:
        decision = int(decision_index)
        if not 0 <= decision < self.decisions:
            raise IndexError(f"Decision {decision} outside {self.decisions}.")
        if self.state is None:
            self.state = np.zeros(
                (self.decisions, self.worlds, int(state.shape[-1])),
                dtype=np.float32,
            )
            self.prior = np.zeros(
                (self.decisions, self.worlds, *tuple(prior.shape[1:])),
                dtype=np.float32,
            )
            if self.record_frames:
                height, width = int(cameras.overview.shape[-2]), int(
                    cameras.overview.shape[-1]
                )
                shape = (
                    self.decisions,
                    self.worlds,
                    height,
                    width,
                    3,
                )
                self.overview = np.zeros(shape, dtype=np.uint8)
                self.wrist = np.zeros(shape, dtype=np.uint8)
        assert self.state is not None and self.prior is not None
        self.state[decision] = self._host(state, dtype=np.float32)
        self.prior[decision] = self._host(prior, dtype=np.float32)
        self.decision_active[decision] = self._host(active, dtype=bool)
        if self.record_frames:
            assert self.overview is not None and self.wrist is not None
            self.overview[decision] = self._frame(cameras.overview)
            self.wrist[decision] = self._frame(cameras.wrist)
        self.used_decisions = max(self.used_decisions, decision + 1)

    def _mark_first(self, target: np.ndarray, fired: Any, step: int) -> None:
        mask = self._host(fired, dtype=bool) & (target < 0)
        target[mask] = int(step)

    def record_step(
        self,
        *,
        decision_index: int,
        action_index: int,
        action: Any,
        active: Any,
        physical_grasp: Any,
        held_lift: Any,
        released: Any,
        native: Any,
        strict: Any,
    ) -> None:
        decision, within = int(decision_index), int(action_index)
        self.action[decision, :, within] = self._host(
            action, dtype=np.float32
        )
        live = self._host(active, dtype=bool)
        self.action_mask[decision, :, within] = live
        step = decision * self.actions_per_decision + within
        self._mark_first(self.first_grasp_step, physical_grasp & active, step)
        self._mark_first(self.first_lift_step, held_lift & active, step)
        self._mark_first(self.first_release_step, released & active, step)
        self._mark_first(self.first_strict_step, strict & native & active, step)

    def selected_payload(self, selected: np.ndarray) -> dict[str, np.ndarray]:
        if self.state is None or self.prior is None:
            raise ValueError("The rollout captured no policy decisions.")
        rows = np.asarray(selected, dtype=np.int64)
        end = int(self.used_decisions)
        return {
            "state": self.state[:end, rows],
            "prior": self.prior[:end, rows],
            "action": self.action[:end, rows],
            "action_mask": self.action_mask[:end, rows],
            "decision_active": self.decision_active[:end, rows],
            "first_grasp_step": self.first_grasp_step[rows],
            "first_lift_step": self.first_lift_step[rows],
            "first_release_step": self.first_release_step[rows],
            "first_strict_step": self.first_strict_step[rows],
        }

    def frame_payload(
        self,
        selected: np.ndarray,
        *,
        episode_uid: Sequence[str],
    ) -> dict[str, np.ndarray]:
        if self.overview is None or self.wrist is None:
            raise ValueError("Frame recording was disabled.")
        rows = np.asarray(selected, dtype=np.int64)
        end = int(self.used_decisions)
        return {
            "overview": self.overview[:end, rows],
            "wrist": self.wrist[:end, rows],
            "world_index": rows.astype(np.int64),
            "episode_uid": np.asarray(list(episode_uid), dtype="U160"),
            "decisions": np.asarray(end, dtype=np.int64),
        }


def _cell_counts(
    destinations: Sequence[Any],
    catalogs: Sequence[Any],
    mask: np.ndarray | None = None,
) -> dict[str, int]:
    use = np.ones(len(destinations), dtype=bool) if mask is None else mask
    counts: dict[str, int] = {}
    for destination, catalog in zip(
        np.asarray(destinations)[use], np.asarray(catalogs)[use]
    ):
        key = f"{catalog}/{destination}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_round(
    *,
    output: Path,
    stem: str,
    scenes: Sequence[Any],
    trace: FullTaskTrace,
    result: Mapping[str, np.ndarray],
    checkpoint: Path,
    checkpoint_sha256: str,
    config: Path,
    manifest_sha256: str | None,
    split: str,
    shard: int,
    round_index: int,
    record_frames: bool,
    retention_rows_per_stage: int = 0,
) -> dict[str, Any]:
    strict = np.asarray(result["strict"], dtype=bool)
    selected = np.flatnonzero(strict)
    episode_all = np.asarray(
        [f"{stem}/r{round_index}w{world}" for world in range(len(scenes))],
        dtype="U160",
    )
    scene_uid = np.asarray([str(scene.scene_uid) for scene in scenes], dtype="U96")
    destination = np.asarray([str(scene.destination) for scene in scenes], dtype="U16")
    target_catalog = np.asarray(
        [str(scene.target_catalog) for scene in scenes], dtype="U64"
    )
    instruction_text = np.asarray(
        [
            student_instruction_text(
                object_label(scene.target_catalog), scene.destination
            )
            for scene in scenes
        ],
        dtype="U256",
    )
    instruction_id = np.asarray(
        [destination_instruction_id(scene.destination) for scene in scenes],
        dtype=np.int64,
    )

    common = {
        "episode_uid": episode_all,
        "scene_uid": scene_uid,
        "destination": destination,
        "target_catalog": target_catalog,
        "instruction_id": instruction_id,
        "instruction_text": instruction_text,
        "native": np.asarray(result["native"], dtype=bool),
        "strict": strict,
        "approached": np.asarray(result["approached"], dtype=bool),
        "grasped": np.asarray(result["grasped"], dtype=bool),
        "lifted": np.asarray(result["lifted"], dtype=bool),
        "released": np.asarray(result["released"], dtype=bool),
        "carry_slip": np.asarray(result["carry_slip"], dtype=bool),
        "wrong_place": np.asarray(result["wrong_place"], dtype=bool),
        "non_finite": np.asarray(result["non_finite"], dtype=bool),
    }
    attempts_path = output / f"attempts_{stem}.npz"
    np.savez_compressed(attempts_path, schema=np.asarray(SCHEMA), **common)

    files: dict[str, Any] = {"attempts": str(attempts_path)}
    if selected.size:
        payload = trace.selected_payload(selected)
        payload.update(
            {
                key: value[selected]
                for key, value in common.items()
            }
        )
        payload.update(
            {
                "schema": np.asarray(SCHEMA),
                "world_index": selected.astype(np.int64),
                "round_index": np.asarray(round_index, dtype=np.int64),
                "shard": np.asarray(shard, dtype=np.int64),
                "split": np.asarray(split),
                "checkpoint": np.asarray(str(checkpoint)),
                "checkpoint_sha256": np.asarray(checkpoint_sha256),
                "config": np.asarray(str(config)),
                "scene_manifest_sha256": np.asarray(
                    "" if manifest_sha256 is None else manifest_sha256
                ),
                "starts_grasped": np.zeros(selected.size, dtype=bool),
            }
        )
        record_path = output / f"record_{stem}.npz"
        np.savez_compressed(record_path, **payload)
        files["record"] = str(record_path)

        if record_frames:
            frames_path = output / f"frames_{stem}.npz"
            np.savez_compressed(
                frames_path,
                **trace.frame_payload(
                    selected, episode_uid=episode_all[selected].tolist()
                ),
            )
            files["frames"] = str(frames_path)
            files["frames_bytes"] = int(frames_path.stat().st_size)

    retention_rows = 0
    retention_episodes = 0
    if int(retention_rows_per_stage) > 0:
        end = int(trace.used_decisions)
        per = int(trace.actions_per_decision)
        candidates = np.flatnonzero(
            ~strict & common["grasped"] & ~common["non_finite"]
        )
        kept: list[int] = []
        masks: list[np.ndarray] = []
        for world in candidates.tolist():
            chosen = stage_decisions(
                active_decisions=int(trace.decision_active[:end, world].sum()),
                grasp_step=int(trace.first_grasp_step[world]),
                lift_step=int(trace.first_lift_step[world]),
                per=per,
                include_placement=False,
            )
            mask = np.zeros(end, dtype=bool)
            for rows in chosen.values():
                mask[spread_evenly(rows, int(retention_rows_per_stage))] = True
            if mask.any():
                kept.append(int(world))
                masks.append(mask)
        if kept:
            worlds = np.asarray(kept, dtype=np.int64)
            decision_mask = np.stack(masks, axis=1)
            payload = trace.selected_payload(worlds)
            payload.update({key: value[worlds] for key, value in common.items()})
            payload.update(
                {
                    "schema": np.asarray(RETENTION_SCHEMA),
                    "retention_decision_mask": decision_mask,
                    "retention_rows_per_stage": np.asarray(
                        int(retention_rows_per_stage), dtype=np.int64
                    ),
                    "world_index": worlds,
                    "round_index": np.asarray(round_index, dtype=np.int64),
                    "shard": np.asarray(shard, dtype=np.int64),
                    "split": np.asarray(split),
                    "checkpoint": np.asarray(str(checkpoint)),
                    "checkpoint_sha256": np.asarray(checkpoint_sha256),
                    "config": np.asarray(str(config)),
                    "scene_manifest_sha256": np.asarray(
                        "" if manifest_sha256 is None else manifest_sha256
                    ),
                    "starts_grasped": np.zeros(worlds.size, dtype=bool),
                }
            )
            retention_path = output / f"retention_{stem}.npz"
            np.savez_compressed(retention_path, **payload)
            files["retention"] = str(retention_path)
            if record_frames:
                assert trace.overview is not None and trace.wrist is not None
                # Zero pictures everywhere but the kept decisions: np.zeros is
                # lazily backed and compresses to nothing, so only the rows
                # that will be trained on cost memory or disk.
                shape = (end, worlds.size, *trace.overview.shape[2:])
                overview = np.zeros(shape, dtype=np.uint8)
                wrist = np.zeros(shape, dtype=np.uint8)
                for column, world in enumerate(worlds.tolist()):
                    rows = np.flatnonzero(decision_mask[:, column])
                    overview[rows, column] = trace.overview[rows, world]
                    wrist[rows, column] = trace.wrist[rows, world]
                retention_frames = output / f"retention_frames_{stem}.npz"
                np.savez_compressed(
                    retention_frames,
                    overview=overview,
                    wrist=wrist,
                    world_index=worlds,
                    episode_uid=episode_all[worlds],
                    decisions=np.asarray(end, dtype=np.int64),
                )
                files["retention_frames"] = str(retention_frames)
            retention_rows = int(decision_mask.sum())
            retention_episodes = int(worlds.size)

    return {
        "round": int(round_index),
        "attempts": len(scenes),
        "retention_episodes": retention_episodes,
        "retention_rows": retention_rows,
        "strict_successes": int(selected.size),
        "strict_rate": round(float(strict.mean()), 5),
        # Ended by a divergence reset; never strict, never harvested.
        "non_finite_episodes": int(common["non_finite"].sum()),
        "attempts_by_cell": _cell_counts(destination, target_catalog),
        "strict_by_cell": _cell_counts(
            destination, target_catalog, mask=strict
        ),
        "files": files,
    }


def _sum_tables(rows: Sequence[Mapping[str, int]], key: str) -> dict[str, int]:
    pooled: dict[str, int] = {}
    for row in rows:
        for name, count in row.get(key, {}).items():
            pooled[name] = pooled.get(name, 0) + int(count)
    return dict(sorted(pooled.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=SPLIT_NAMES, default="collection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--decisions", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--tag", default="strict_policy")
    parser.add_argument("--seed-torch", type=int, default=20260922)
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Action-only audit arm; the resulting bank cannot train vision.",
    )
    parser.add_argument(
        "--retention-rows-per-stage",
        type=int,
        default=0,
        help=(
            "Also keep up to this many evenly spaced decisions per completed "
            "stage of every non-strict episode that grasped, as retention "
            "material (retention_<stem>.npz). 0 keeps the strict-only harvest."
        ),
    )
    args = parser.parse_args(argv)
    if int(args.retention_rows_per_stage) < 0:
        parser.error("--retention-rows-per-stage must be non-negative.")

    checkpoint = args.checkpoint.expanduser().resolve()
    config = args.config.expanduser().resolve()
    scene_manifest = args.scene_manifest.expanduser().resolve()
    for name, path in (
        ("checkpoint", checkpoint),
        ("config", config),
        ("scene manifest", scene_manifest),
    ):
        if not path.is_file():
            raise SystemExit(f"The {name} does not exist: {path}")
    if int(args.decisions) < 1:
        parser.error("--decisions must be positive.")

    scenes, manifest = read_manifest(scene_manifest)
    selected = select_split(scenes, str(args.split))
    try:
        batches = plan_batches(
            selected,
            worlds=int(args.worlds),
            rounds=int(args.rounds),
            shard=int(args.shard),
            num_shards=int(args.num_shards),
            scene_offset=int(args.scene_offset),
        )
    except ValueError as exc:
        parser.error(str(exc))

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = sha256_file(checkpoint)
    world = _build_world(
        controller_workspace_from_config=True,
        checkpoint=checkpoint,
        config_path=config,
        device_str=str(args.device),
        worlds=int(args.worlds),
        group_size=2,
        microbatch=int(args.microbatch),
        load_policy=True,
        run_dir=output,
    )
    resetter = FullTaskSceneResetter(
        backend=world.backend,
        worlds_per_rank=int(args.worlds),
        support_surface_z=float(
            world.task_metadata.get("support_surface_z", 0.15)
        ),
        task_metadata=world.task_metadata,
    )
    vision_dim = (
        int(getattr(world.args, "residual_vision_dim", 0))
        if bool(getattr(world.args, "residual_vision_features", False))
        else 0
    )
    per = int(world.args.replan_every)
    record_frames = not bool(args.no_frames)
    if record_frames:
        estimated = (
            int(args.worlds)
            * int(args.decisions)
            * 2
            * 240
            * 320
            * 3
            / 1e9
        )
        print(
            f"[collect] frame buffer bound is ~{estimated:.2f} GB per round",
            flush=True,
        )

    import torch

    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for local_round, batch in enumerate(batches):
        # The offset itself is part of the identity. This keeps a continuation
        # collision-free even when its world or shard count differs from the
        # first harvest.
        global_round = int(args.scene_offset) + local_round
        torch.manual_seed(
            int(args.seed_torch)
            + global_round * 1_000_003
            + int(args.shard) * 10_000_019
        )
        trace = FullTaskTrace(
            decisions=int(args.decisions),
            worlds=int(args.worlds),
            actions_per_decision=per,
            record_frames=record_frames,
        )
        round_started = time.perf_counter()
        result = run_unassisted(
            world=world,
            resetter=resetter,
            scenes=batch,
            decisions=int(args.decisions),
            settle_decisions=0,
            assisted_yaw=None,
            stochastic_generator=None,
            vision_dim=vision_dim,
            video=None,
            round_index=global_round,
            trace=trace,
        )
        stem = f"{args.tag}_s{int(args.shard)}_r{global_round:04d}"
        report = _write_round(
            output=output,
            stem=stem,
            scenes=batch,
            trace=trace,
            result=result,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            config=config,
            manifest_sha256=manifest.get("manifest_sha256"),
            split=str(args.split),
            shard=int(args.shard),
            round_index=global_round,
            record_frames=record_frames,
            retention_rows_per_stage=int(args.retention_rows_per_stage),
        )
        report["wall_seconds"] = round(time.perf_counter() - round_started, 1)
        reports.append(report)
        print(
            f"[collect] shard {args.shard} round {local_round + 1}/"
            f"{args.rounds}: strict {report['strict_successes']}/"
            f"{report['attempts']} ({report['strict_rate']:.3f}), "
            f"non-finite {report['non_finite_episodes']}, "
            f"retention rows {report['retention_rows']}, "
            f"by cell {report['strict_by_cell']}",
            flush=True,
        )
        # Drop the multi-GB frame allocation before the next round.
        del trace

    attempts = sum(int(row["attempts"]) for row in reports)
    successes = sum(int(row["strict_successes"]) for row in reports)
    collection = {
        "schema": SCHEMA,
        "tool": "collect_cdpr_full_put_into.py",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "config": str(config),
        "scene_manifest": str(scene_manifest),
        "scene_manifest_sha256": manifest.get("manifest_sha256"),
        "split": str(args.split),
        "shard": int(args.shard),
        "num_shards": int(args.num_shards),
        "scene_offset": int(args.scene_offset),
        "worlds": int(args.worlds),
        "rounds": int(args.rounds),
        "decisions": int(args.decisions),
        "settle_decisions": 0,
        "record_frames": record_frames,
        "attempts": attempts,
        "strict_successes": successes,
        "strict_rate": round(successes / max(attempts, 1), 5),
        "non_finite_episodes": sum(
            int(row["non_finite_episodes"]) for row in reports
        ),
        "retention_rows_per_stage": int(args.retention_rows_per_stage),
        "retention_episodes": sum(int(row["retention_episodes"]) for row in reports),
        "retention_rows": sum(int(row["retention_rows"]) for row in reports),
        "attempts_by_cell": _sum_tables(reports, "attempts_by_cell"),
        "strict_by_cell": _sum_tables(reports, "strict_by_cell"),
        "rounds_detail": reports,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "contract": {
            "empty_start": True,
            "one_final_instruction": True,
            "continuous_single_policy": True,
            "teachers": False,
            "stage_switches": False,
            "servo": False,
            "state_restores": False,
            "simulator_recovery": False,
            "distinct_scenes": True,
            "accepted_outcome": "FullTaskOutcome.strict",
        },
    }
    report_path = output / "collection.json"
    report_path.write_text(
        json.dumps(collection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[collect] wrote {report_path}: {successes}/{attempts} strict, "
        f"by cell {collection['strict_by_cell']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
