#!/usr/bin/env python3
"""Film successful AND failed live policy episodes under a pipeline eval config.

Uses sil_record's exact world builder/validate_round recorder, without replay or
smoothing. Records all worlds' telemetry; cameras are copied only for a random
subset of distinct GRPO scenes. Videos are qualitative samples, not rate estimates.
Set RLVLA_EMBODIMENT=widowx200 before imports for WidowX scenes/object geometry.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def pipeline_selection(pipeline: Path, evaluation: str, stage: str, policy_kind: str) -> dict[str, Any]:
    """Resolve original final-eval settings, including sharded summary metadata."""
    state = json.loads((pipeline / "pipeline_state.json").read_text())
    folder = pipeline / "eval" / "final" / evaluation
    wrapper = json.loads((folder / "summary.json").read_text())
    if wrapper.get("failed_shards"):
        raise ValueError(f"Incomplete final evaluation: {wrapper['failed_shards']}")
    summaries = [wrapper]
    summaries.extend(json.loads((folder / name).read_text())
                     for name in wrapper.get("shard_summaries", []))
    configs = {str(row["config"]) for row in summaries if row.get("config")}
    caps = {float(row["start_distance_cap"]) for row in summaries if row.get("start_distance_cap") is not None}
    if len(configs) != 1 or len(caps) != 1:
        raise ValueError(f"Cannot resolve one original eval config/cap in {folder}: configs={configs}, caps={caps}")
    if stage == "final":
        policy = json.loads((pipeline / "final_report.json").read_text())["policy"]
    else:
        record = state["stages"][stage]
        policy = (record.get("selected_checkpoint") or record.get("best_checkpoint")) if policy_kind == "rl" else record.get("policy")
    if not policy:
        raise ValueError(f"Missing {stage} {policy_kind} checkpoint")
    return {"checkpoint": Path(policy), "config": Path(next(iter(configs))),
            "start_distance_cap": next(iter(caps))}


def tracked_worlds(worlds: int, group_size: int, scenes: int, seed: int) -> list[int]:
    """One representative per randomly chosen reset group; avoid stride aliasing."""
    if worlds <= 0 or group_size <= 0 or worlds % group_size or scenes <= 0:
        raise ValueError("Need positive worlds/group-size/track-scenes and worlds divisible by group-size")
    groups = np.random.default_rng(seed).choice(
        worlds // group_size, size=min(scenes, worlds // group_size), replace=False
    )
    return sorted(int(group) * group_size for group in groups)


def outcome_rows(recording: Any) -> list[dict[str, Any]]:
    from tools.audit.sil_record import _episode_rows
    from rl_vla_bootstrapping.simulation.cdpr_object_catalog import ACTIVE_CDPR_CATALOGS

    rows = _episode_rows(recording)
    reset = recording.reset_object_xyz
    if reset is None:
        raise ValueError("Missing true reset positions; cannot label spawn locations")
    for index, row in enumerate(rows):
        target = int(recording.target_slots[index])
        reference = int(recording.reference_slots[index])
        row["round"] = int(recording.round_index)
        catalog = None if recording.target_catalog_ids is None else int(recording.target_catalog_ids[index])
        row["object"] = (ACTIVE_CDPR_CATALOGS[catalog]
                         if catalog is not None and 0 <= catalog < len(ACTIVE_CDPR_CATALOGS) else "unknown")
        row["diverged"] = (None if recording.diverged_world_mask is None
                           else bool(recording.diverged_world_mask[index]))
        for role, slot in (("target", target), ("container", reference)):
            for axis, name in enumerate("xyz"):
                row[f"{role}_reset_{name}"] = None if slot < 0 else float(reset[index, slot, axis])
        for axis, name in enumerate("xyz"):
            row[f"ee_reset_{name}"] = float(recording.reset_ee_xyz[index, axis])
        row["start_distance_cap"] = float(recording.start_distance_cap)
    return rows


def sliced_rates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Keep scene count beside world count: GRPO copies share a reset."""
    result = {}
    for name, fields in (
        ("by_instruction", ("instruction",)),
        ("by_object", ("instruction", "object")),
        ("by_spawn", ("instruction", "object", "target_reset_x", "target_reset_y")),
    ):
        groups: dict[tuple, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            key = tuple(round(row[field], 3) if field.startswith("target_reset_") else row[field]
                        for field in fields)
            groups[key].append(row)
        result[name] = []
        for key, items in sorted(groups.items()):
            result[name].append({
                **dict(zip(fields, key)),
                "episodes": len(items), "successes": sum(r["success"] for r in items),
                "rate": sum(r["success"] for r in items) / len(items),
                "ever_grasped_rate": sum(r["ever_grasped"] for r in items) / len(items),
                "distinct_reset_groups": len({(r["round"], r["scene"]) for r in items}),
            })
    return result


def episode_frames(frames: list[Any], recording: Any, world: int) -> list[Any]:
    """Drop frozen decision frames and append the separately rendered final state."""
    decisions = math.ceil(int(recording.episode_length[world]) / recording.actions_per_decision)
    return frames[:decisions] + frames[-1:] if decisions and frames else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, help="Resolve checkpoint and original eval settings from this completed pipeline")
    parser.add_argument("--evaluation", choices=("move_to", "pick_up", "put_into", "put_into_composed"), default="put_into_composed")
    parser.add_argument("--stage", default="final", help="final or a pipeline stage name")
    parser.add_argument("--policy-kind", choices=("rl", "handoff"), default="handoff", help="For a stage: selected RL before SIL, or actual handoff after SIL")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, help="Generated eval YAML, not the training YAML")
    parser.add_argument("--start-distance-cap", type=float)
    parser.add_argument("--output", type=Path, required=True, help="New directory; refuses to overwrite")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--smolvla-microbatch", type=int, default=256)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--track-scenes", type=int, default=16, help="Camera samples per round, one world per GRPO group")
    parser.add_argument("--per-outcome", type=int, default=8, help="Video cap per instruction and outcome across rounds")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback FPS; one frame per decision plus a final-state hold")
    parser.add_argument("--seed-torch", type=int, default=0)
    parser.add_argument("--video-seed", type=int, default=0)
    parser.add_argument("--horizon-decisions", type=int, default=0, help="0 preserves the task; positive values are a budget ablation")
    parser.add_argument("--oracle", action="store_true", help="Diagnostic scripted controller in place of the policy")
    args = parser.parse_args(argv)
    if args.pipeline_dir:
        try:
            selection = pipeline_selection(args.pipeline_dir.resolve(), args.evaluation, args.stage, args.policy_kind)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        for key, value in selection.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    if args.checkpoint is None or args.config is None or args.start_distance_cap is None:
        parser.error("Supply --pipeline-dir, or --checkpoint, --config and --start-distance-cap")
    if os.environ.get("RLVLA_EMBODIMENT", "").lower() != "widowx200":
        parser.error("Set RLVLA_EMBODIMENT=widowx200 before launching")
    if args.rounds <= 0 or args.per_outcome <= 0 or args.fps <= 0 or args.horizon_decisions < 0:
        parser.error("rounds, per-outcome, fps must be positive; horizon-decisions must be nonnegative")
    if not math.isfinite(args.start_distance_cap) or args.start_distance_cap <= 0:
        parser.error("start-distance-cap must be positive and finite")
    try:
        tracked_worlds(args.worlds, args.group_size, args.track_scenes, args.video_seed)
    except ValueError as error:
        parser.error(str(error))
    for path in (args.checkpoint, args.config):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg must be on PATH")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    # Import after argument validation so --help requires neither CUDA nor torch.
    from tools.audit.sil_record import _build_world, _RoundRecorder, _OracleActionSource, _write_csv
    from tools.audit.success_episode_videos import _FrameTap, _Mp4

    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest.update(video_source="live_policy" if not args.oracle else "scripted_oracle",
                    controller_workspace_from_config=True,
                    video_unit="one policy decision plus post-rollout final state",
                    episode_unit="world; copies within a GRPO scene are not independent resets")
    manifest["eval_config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    shutil.copy2(args.config, output / "source_eval_config.yaml")
    try:
        manifest["audit_git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["audit_git_commit"] = None
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    world = _build_world(
        checkpoint=args.checkpoint.resolve(), config_path=args.config.resolve(),
        device_str=args.device, worlds=args.worlds, group_size=args.group_size,
        microbatch=args.smolvla_microbatch, load_policy=True, run_dir=output,
        start_distance_cap=args.start_distance_cap, controller_workspace_from_config=True,
    )
    manifest.update(
        grasp_contract="contact_v2" if getattr(world.resetter, "contact_based_grasp", False) else "opening_v1",
        open_grasp_hover_height_m=getattr(world.resetter, "open_grasp_hover_height", None),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    kept: collections.Counter = collections.Counter()
    all_rows, videos = [], []
    for round_index in range(args.round_index, args.round_index + args.rounds):
        tracked = tracked_worlds(args.worlds, args.group_size, args.track_scenes, args.video_seed + round_index)
        with _FrameTap(world.backend, tracked, both_cameras=True) as tap:
            with _RoundRecorder(world, torch_seed=args.seed_torch, horizon_override=args.horizon_decisions,
                                action_source=_OracleActionSource() if args.oracle else None) as recorder:
                recording = recorder.run(round_index=round_index)
            # One extra render shows the terminal state; it is not a policy observation.
            with world.torch.inference_mode():
                world.backend.render_policy_cameras()
        recording.start_distance_cap = args.start_distance_cap
        recording.to_npz(output / f"record_{round_index:02d}.npz")
        rows = outcome_rows(recording)
        for row in rows:
            row["scene"] = int(row["world"]) // args.group_size
        all_rows.extend(rows)
        _write_csv(output / f"episodes_{round_index:02d}.csv", rows)
        for index in tracked:
            row = rows[index]
            outcome = "success" if row["success"] else "failed"
            key = (row["instruction"], outcome)
            if kept[key] >= args.per_outcome:
                continue
            frames = episode_frames(tap.frames[index], recording, index)
            if not frames:
                continue
            name = f"r{round_index:03d}_w{index:03d}_{row['instruction']}_{row['object']}_{outcome}"
            path = output / "videos" / f"{name}.mp4"
            writer = _Mp4(path, fps=args.fps, height=frames[0].shape[0], width=frames[0].shape[1])
            try:
                for frame in frames:
                    writer.write(frame)
                for _ in range(max(1, round(args.fps))):
                    writer.write(frames[-1])
            finally:
                writer.close()
            info = {**row, "video": str(path.relative_to(output)), "fps": args.fps,
                    "decision_frames": len(frames) - 1, "terminal_frame_is_extra_render": True}
            path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
            videos.append(info)
            kept[key] += 1
        summary = {**manifest, **sliced_rates(all_rows), "videos": videos,
                   "determinism": recorder.determinism}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "videos.json").write_text(json.dumps(videos, indent=2) + "\n")
        for entry in summary["by_instruction"]:
            print(f"[outcomes] {entry['instruction']}: {entry['successes']}/{entry['episodes']} "
                  f"success; ever grasped {entry['ever_grasped_rate']:.3f}; "
                  f"{entry['distinct_reset_groups']} reset groups", flush=True)
        # Release camera arrays before the next round, including the trimmed list.
        tap.frames.clear()
        frames = []
    print(f"[outcomes] {len(videos)} videos; report: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
