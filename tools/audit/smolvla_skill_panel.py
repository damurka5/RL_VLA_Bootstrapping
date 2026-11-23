#!/usr/bin/env python3
"""Capture matched first-decision scenes and define disjoint probe holdouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from smolvla_skill_interventions import digest, write_json

QUADRANTS = ("SW", "SE", "NW", "NE")
MIN_DISTANCE_M = 0.015
META = ("uid", "skill", "target", "receptacle", "delta_xy", "start_xyz", "goal_xyz")


def strata(delta: np.ndarray, target: np.ndarray, receptacle: np.ndarray):
    quadrant = np.select(
        [(delta[:, 0] < 0) & (delta[:, 1] < 0), delta[:, 1] < 0, delta[:, 0] < 0],
        QUADRANTS[:3], default="NE",
    )
    return quadrant, np.asarray([f"{obj}:{dest}" if dest else "" for obj, dest in zip(target, receptacle)])


def split_scenes(meta: dict, holdout_quadrants: list[str], holdout_pairs: list[str],
                 seed: int) -> np.ndarray:
    quadrant, pair = strata(meta["delta_xy"], meta["target"], meta["receptacle"])
    bearing_mask = np.isin(quadrant, holdout_quadrants)
    pair_mask = np.isin(pair, holdout_pairs)
    for label, observed, requested in (("quadrant", quadrant, holdout_quadrants),
                                       ("pair", pair, holdout_pairs)):
        missing = set(requested) - set(observed)
        if missing:
            raise ValueError(f"Missing requested {label} holdout: {sorted(missing)}")

    # If a physical scene is queried twice, keep both prompts in the same partition.
    memberships: dict[str, tuple[bool, bool]] = {}
    for uid, bearing, combo in zip(meta["uid"], bearing_mask, pair_mask):
        before = memberships.get(uid, (False, False))
        memberships[uid] = (before[0] or bool(bearing), before[1] or bool(combo))

    def label(uid: str) -> str:
        bearing, combo = memberships[uid]
        if bearing and combo:
            return "joint_holdout"
        if bearing:
            return "bearing_holdout"
        if combo:
            return "pair_holdout"
        coin = hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0]
        return "ordinary_test" if coin < 51 else "probe_train"

    return np.asarray([label(uid) for uid in meta["uid"]])


def _geometry_csv(path: Path) -> tuple[dict, Counter]:
    metadata = {key: [] for key in META}
    excluded = Counter()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            audit = row["audit"]
            if audit not in ("move_to", "placement_held", "put_into_held", "put_into_caught",
                             "placement", "put_into"):
                excluded[f"other_audit:{audit}"] += 1
                continue
            placement = audit != "move_to"
            if placement and row["caught_start"] != "True":
                excluded["placement_not_caught"] += 1
                continue
            role = "container" if placement else "target"
            if row["reach_all"] != "True":
                excluded["reach_check_failed"] += 1
                continue
            if not any(float(row.get(f"{camera}_{role}_pixels_visible") or 0) > 0
                       for camera in ("overview", "wrist")):
                excluded["goal_has_no_visible_pixels"] += 1
                continue
            ee = [float(row[f"ee_{axis}"]) for axis in "xyz"]
            goal = [float(row[f"{role}_{axis}"]) for axis in "xy"]
            delta = [b - a for a, b in zip(ee, goal)]
            if np.linalg.norm(delta) < MIN_DISTANCE_M:
                excluded["goal_xy_too_close"] += 1
                continue
            metadata["uid"].append(f"{audit}:{row['round']}:{row['scene']}")
            metadata["skill"].append("placement" if placement else "move_to")
            metadata["target"].append(row["target"])
            metadata["receptacle"].append(row["instruction"].removeprefix("put_into_") if placement else "")
            metadata["delta_xy"].append(delta)
            metadata["start_xyz"].append(ee)
            metadata["goal_xyz"].append([*goal, float(row["target_z"]) if not placement else np.nan])
    if not metadata["uid"]:
        raise ValueError(f"No suitable scenes in {path}")
    return {key: np.asarray(value) for key, value in metadata.items()}, excluded


def plan(args) -> None:
    if args.geometry_csv:
        meta, excluded = _geometry_csv(args.geometry_csv)
    else:
        panels = []
        for path in args.panel:
            with np.load(path, allow_pickle=False) as data:
                panels.append({key: data[key] for key in META})
        meta = {key: np.concatenate([panel[key] for panel in panels]) for key in META}
        excluded = {}
    quadrant, pair = strata(meta["delta_xy"], meta["target"], meta["receptacle"])
    split = split_scenes(meta, args.holdout_quadrant, args.holdout_pair, args.seed)
    counts = Counter(zip(meta["skill"], split, quadrant, pair))
    if args.assignments:
        args.assignments.parent.mkdir(parents=True, exist_ok=True)
        with args.assignments.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("scene_uid", "skill", "partition", "quadrant", "target", "receptacle"))
            writer.writerows(zip(meta["uid"], meta["skill"], split, quadrant,
                                 meta["target"], meta["receptacle"]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {"seed": args.seed, "excluded": dict(excluded),
                                 "holdout_quadrant": args.holdout_quadrant,
                                 "holdout_pair": args.holdout_pair,
                                 "rows": [{"skill": s, "split": f, "quadrant": q, "pair": p, "n": n}
                                          for (s, f, q, p), n in sorted(counts.items())]})
    for (skill, partition), count in sorted(Counter(zip(meta["skill"], split)).items()):
        print(f"{skill:12} {partition:18} {count:5}")
    for skill in np.unique(meta["skill"]):
        mask = meta["skill"] == skill
        print(skill, "quadrants", dict(sorted(Counter(quadrant[mask]).items())))
        print(skill, "pairs", dict(sorted(Counter(pair[mask]).items())))
    if excluded:
        print("Excluded:", dict(excluded))


def capture(args) -> None:
    from xy_approach_probe import _build_world
    from sil_record import _instruction_name
    from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import goal_slots_for_reset
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import build_smolvla_state_tensor
    from rl_vla_bootstrapping.simulation.cdpr_object_catalog import ACTIVE_CDPR_CATALOGS

    args.output.mkdir(parents=True, exist_ok=False)
    world = _build_world(
        checkpoint=args.checkpoint, config_path=args.config, device_str=args.device,
        worlds=args.worlds, group_size=2, microbatch=args.worlds, load_policy=False,
        run_dir=args.output, start_distance_cap=args.cap,
        metadata_overrides=args.metadata_override or (),
        controller_workspace_from_config=not args.controller_workspace_from_checkpoint,
    )
    if getattr(world.args, "residual_relative_target", False):
        raise ValueError("Privileged target coordinates are enabled in the checkpoint")
    vision_dim = int(world.args.residual_vision_dim) if world.args.residual_vision_features else 0
    proprio_width = int(world.payload["state_dim"]) - vision_dim
    if proprio_width <= 0:
        raise ValueError("Invalid proprioceptive width")
    saved = 0
    exclusions = Counter()
    torch = world.torch
    with torch.no_grad():
        for round_index in range(args.round_index, args.round_index + args.rounds):
            reset = world.resetter.reset(update_index=0, round_index=round_index, allow_prelifted=False)
            low = world.backend.low_dim_observations()
            idx = torch.arange(args.worlds, device=world.device)
            goal_slots = goal_slots_for_reset(torch, reset)
            goals = low.object_positions[idx, goal_slots]
            start_xyz, goal_xyz = (t.detach().float().cpu().numpy() for t in (low.ee_position, goals))
            delta = goal_xyz[:, :2] - start_xyz[:, :2]
            proprio = build_smolvla_state_tensor(
                ee_position=low.ee_position, ee_yaw=low.ee_yaw, gripper_opening=low.gripper_opening,
                object_positions=low.object_positions, target_slots=reset.task_state.target_slots,
                state_dim=proprio_width,
            ).cpu().numpy()
            ids = reset.task_state.instruction_ids.cpu().tolist()
            names = [_instruction_name(i) for i in ids]
            held = reset.task_state.grasped.cpu().numpy()
            valid_task = np.asarray([
                name.startswith("move_to") if args.skill == "move_to"
                else name in ("put_into_plate", "put_into_bowl") and holding
                for name, holding in zip(names, held)
            ])
            valid_geometry = (np.linalg.norm(delta, axis=1) >= MIN_DISTANCE_M) & np.isfinite(goal_xyz).all(axis=1)
            leader = np.arange(args.worlds) % 2 == 0
            keep = leader & valid_task & valid_geometry
            exclusions["different_task_or_uncaught"] += int((leader & ~valid_task).sum())
            exclusions["invalid_or_short_goal"] += int((leader & valid_task & ~valid_geometry).sum())
            if not keep.any():
                continue
            cameras = world.backend.render_policy_cameras()
            catalogs = reset.group_target_catalog_ids.cpu().numpy().reshape(-1)
            if len(catalogs) == args.worlds // 2:
                catalogs = np.repeat(catalogs, 2)
            targets = np.asarray([ACTIVE_CDPR_CATALOGS[i] for i in catalogs])
            receptacle = np.asarray([n.removeprefix("put_into_") if args.skill == "placement" else "" for n in names])
            poses = np.column_stack((start_xyz, low.ee_yaw.cpu().numpy().reshape(-1, 1),
                                     low.object_positions.cpu().numpy().reshape(args.worlds, -1)))
            uid = np.full(args.worlds, "", dtype="U64")
            uid[keep] = [hashlib.sha256(np.rint(row * 1000).astype("<i8").tobytes()).hexdigest()
                         for row in poses[keep]]
            arrays = dict(
                uid=uid, skill=np.full(args.worlds, args.skill), target=targets,
                receptacle=receptacle, delta_xy=delta, start_xyz=start_xyz, goal_xyz=goal_xyz,
                instruction=np.asarray(reset.instructions), proprio=proprio,
                overview=cameras.overview.float().cpu().numpy(),
                wrist=cameras.wrist.float().cpu().numpy(),
            )
            path = args.output / f"panel_{round_index:05d}.npz"
            np.savez_compressed(path, **{key: value[keep] for key, value in arrays.items()})
            saved += int(keep.sum())
            print(f"{path}: {int(keep.sum())} scenes", flush=True)
    if not saved:
        raise RuntimeError("No suitable scenes captured; verify the skill and reset config")
    write_json(args.output / "panel.json", {
        "checkpoint_sha256": digest(args.checkpoint), "config": str(args.config),
        "config_sha256": digest(args.config), "cap_m": args.cap, "skill": args.skill,
        "device": args.device, "rows": saved, "excluded": dict(exclusions),
        "metadata_override": args.metadata_override,
        "round_index": args.round_index, "rounds": args.rounds, "worlds": args.worlds,
    })


def holdout_arguments(parser) -> None:
    parser.add_argument("--holdout-quadrant", action="append", choices=QUADRANTS, default=[])
    parser.add_argument("--holdout-pair", action="append", default=[], metavar="CATALOG:RECEPTACLE")
    parser.add_argument("--seed", type=int, default=1729)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    take = sub.add_parser("capture")
    for name in ("checkpoint", "config", "output"):
        take.add_argument(f"--{name}", type=Path, required=True)
    take.add_argument("--skill", choices=("move_to", "placement"), required=True)
    take.add_argument("--cap", type=float, required=True)
    take.add_argument("--controller-workspace-from-checkpoint", action="store_true",
                      help="Use saved controller bounds for configs without explicit bounds")
    take.add_argument("--rounds", type=int, default=4)
    take.add_argument("--round-index", type=int, default=90)
    take.add_argument("--worlds", type=int, default=64)
    take.add_argument("--device", default="cuda:0")
    take.add_argument("--metadata-override", nargs="+")
    take.set_defaults(run=capture)

    report = sub.add_parser("plan")
    inputs = report.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--geometry-csv", type=Path)
    inputs.add_argument("--panel", type=Path, nargs="+")
    report.add_argument("--output", type=Path)
    report.add_argument("--assignments", type=Path)
    holdout_arguments(report)
    report.set_defaults(run=plan)
    args = parser.parse_args()
    if args.run == capture and (args.worlds < 2 or args.worlds % 2 or args.rounds < 1 or args.cap <= 0):
        parser.error("Capture needs a positive even world count, positive rounds and cap")
    args.run(args)


if __name__ == "__main__":
    main()
