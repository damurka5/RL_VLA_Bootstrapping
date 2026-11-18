#!/usr/bin/env python3
"""Record the production grasp gates and compare them with scoring and motion.

Run with --run-dir on the GPU server, or analyze downloaded --recordings on CPU.
Object motion is a suspicion flag, never a replacement grasp/success label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import glob
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLUMNS = (
    "active", "eligible", "left_contact", "right_contact", "left_force_n", "right_force_n",
    "position_slip_m", "orientation_slip_rad", "stable_pose", "persistence_steps",
    "physical_grasp", "scored_grasp", "release_open", "loaded_finger_contact",
    "raw_left_force_n", "raw_right_force_n", "left_contact_distance_m", "right_contact_distance_m",
    "left_distance_rejected_force_n", "right_distance_rejected_force_n",
)


def raw_pad_metrics(backend, target_slots, limit):
    """Observe reported target/pad contacts before the distance filter.

    Infinity means no contact entry, not a general surface-distance query.
    The solved forces are already available from the original detector call.
    """
    torch = backend.torch
    world = backend._contact_worldid.to(torch.int64)
    safe = world.clamp(0, backend.worlds_per_rank - 1)
    valid = ((backend._contact_indices < backend._nacon.reshape(-1)[0])
             & (world >= 0) & (world < backend.worlds_per_rank))
    target_geoms = backend._slot_geom_ids.index_select(0, target_slots.index_select(0, safe))
    first, second = backend._contact_geom[:, 0], backend._contact_geom[:, 1]
    first_target = (first[:, None] == target_geoms).any(dim=1)
    second_target = (second[:, None] == target_geoms).any(dim=1)
    forces, distance = backend._contact_forces[:, 0].abs(), backend._contact_dist
    result = {}
    for side, pad in zip(("left", "right"), backend._finger_pad_geom_ids_tensor):
        matched = valid & ((first_target & (second == pad)) | (second_target & (first == pad)))
        summed = torch.zeros(backend.worlds_per_rank, dtype=forces.dtype, device=forces.device)
        summed.scatter_add_(0, safe, torch.where(matched, forces, 0.0))
        closest = torch.full_like(summed, float("inf"))
        closest.scatter_reduce_(0, safe, torch.where(matched, distance, float("inf")), reduce="amin", include_self=True)
        rejected = torch.zeros_like(summed)
        rejected.scatter_add_(0, safe, torch.where(matched & (distance > limit), forces, 0.0))
        result.update({f"raw_{side}_force_n": summed, f"{side}_contact_distance_m": closest,
                       f"{side}_distance_rejected_force_n": rejected})
    return result


class GraspTraceCapture:
    """Observe existing calls; never call the solver again or change its result."""
    def __init__(self, world):
        self.world = world
        self.rows = []
        self.pending = None
        self.stack = ExitStack()

    def patch(self, owner, name, replacement):
        original = getattr(owner, name)
        owned = name in vars(owner)
        def restore():
            if owned:
                setattr(owner, name, original)
            else:
                vars(owner).pop(name, None)
        self.stack.callback(restore)
        setattr(owner, name, replacement)
        return original

    def __enter__(self):
        import rl_vla_bootstrapping.policy.mjwarp_rank_local_collector as module
        from rl_vla_bootstrapping.simulation.mjlab_mjwarp_backend import FINGER_CONTACT_MAX_DISTANCE_M
        backend, collector, torch = self.world.backend, self.world.collector, self.world.torch
        native_contact = backend.finger_object_contact_metrics
        native_grasp = collector._update_physical_grasp
        native_score = module.evaluate_active_sparse_tasks
        self.metadata = {
            "schema": 1, "min_force_n": module._GRASP_MIN_PAD_FORCE_N,
            "persistence_steps": module._GRASP_PERSISTENCE_STEPS,
            "max_position_slip_m": module._GRASP_MAX_RELATIVE_POSITION_SLIP_M,
            "max_orientation_slip_rad": module._GRASP_MAX_RELATIVE_ORIENTATION_SLIP_RAD,
            "catalogs": list(module.ACTIVE_CDPR_CATALOGS),
            "instructions": list(module.ACTIVE_INSTRUCTION_TYPES),
            "group_size": int(self.world.layout.group_size),
            "max_contact_distance_m": FINGER_CONTACT_MAX_DISTANCE_M,
            "raw_contact_available": hasattr(backend, "_contact_geom"),
            "collector_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            "force_source_sha256": hashlib.sha256(Path(inspect.getsourcefile(native_contact)).read_bytes()).hexdigest(),
        }
        def contact(*args, **kwargs):
            result = native_contact(*args, **kwargs)
            self.contacts = result
            slots = args[0] if args else kwargs["target_slots"]
            self.raw = raw_pad_metrics(backend, slots, FINGER_CONTACT_MAX_DISTANCE_M) if self.metadata["raw_contact_available"] else {}
            return result
        def grasp(reset, low, active):
            if not self.rows:
                self.initial_persistence = reset.bilateral_contact_steps.detach().cpu().numpy().copy()
                self.metadata["contact_based_grasp"] = bool(reset.task_state.contact_based_grasp)
            result = native_grasp(reset, low, active)
            diagnostics = result[2]
            self.pending = {
                "active": active, "eligible": reset.grasp_eligible,
                "left_contact": self.contacts.left_contact, "right_contact": self.contacts.right_contact,
                "left_force_n": self.contacts.left_normal_force, "right_force_n": self.contacts.right_normal_force,
                "position_slip_m": diagnostics["relative_position_slip_m"],
                "orientation_slip_rad": diagnostics["relative_orientation_slip_rad"],
                "stable_pose": diagnostics["stable_relative_pose"],
                "persistence_steps": reset.bilateral_contact_steps, "physical_grasp": result[1],
                "release_open": low.gripper_opening >= reset.task_state.release_threshold,
                "loaded_finger_contact": diagnostics["loaded_finger_contact"],
                **{key: self.raw[key] if key in self.raw else torch.full_like(self.contacts.left_normal_force, float("nan"))
                   for key in COLUMNS if key.startswith("raw_") or "contact_distance" in key or "distance_rejected" in key},
            }
            return result
        def score(**kwargs):
            result = native_score(**kwargs)
            if self.pending is None:
                raise RuntimeError("Grasp trace has no detector call for this scoring step.")
            self.pending["scored_grasp"] = result.diagnostics["grasped"]
            values = torch.stack([self.pending[key].to(torch.float32) for key in COLUMNS], dim=-1)
            self.rows.append(values.detach().cpu().numpy().copy())
            self.pending = None
            return result
        try:
            self.patch(backend, "finger_object_contact_metrics", contact)
            self.patch(collector, "_update_physical_grasp", grasp)
            self.patch(module, "evaluate_active_sparse_tasks", score)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        self.stack.close()

    def save(self, path, recording, recording_path):
        self.values = np.stack(self.rows)
        if self.values.shape[:2] != recording.active.shape or self.pending is not None:
            raise RuntimeError("Detector trace and recorded environment steps are misaligned.")
        actual = self.values[..., COLUMNS.index("physical_grasp")].astype(bool)
        if not np.array_equal(actual, recording.caught_target):
            raise RuntimeError("Detector trace and the grasp supplied to scoring disagree.")
        np.savez_compressed(path, values=self.values, columns=np.asarray(COLUMNS),
                            initial_persistence=self.initial_persistence,
                            metadata_json=np.asarray(json.dumps(self.metadata)),
                            recording_sha256=np.asarray(hashlib.sha256(Path(recording_path).read_bytes()).hexdigest()))


def diagnose(record, values, columns, metadata, initial_persistence):
    data = {key: values[..., i] for i, key in enumerate(columns)}
    active = data["active"].astype(bool)
    actual = data["physical_grasp"].astype(bool)
    scored = data["scored_grasp"].astype(bool)
    if active.shape != record["active"].shape or not np.array_equal(actual, record["caught_target"]):
        raise ValueError("Recording/trace shape or physical-grasp alignment differs.")
    # A trace's active mask is AFTER the production divergence guard; the
    # recording's execution mask is BEFORE it. Idle later divergences do not
    # invalidate an already completed episode.
    live_diverged = (record["active"].astype(bool) & ~active).any(axis=0)
    usable = active & ~live_diverged[None, :]
    both = data["left_contact"].astype(bool) & data["right_contact"].astype(bool)
    finite_force = np.isfinite(data["left_force_n"]) & np.isfinite(data["right_force_n"])
    loaded = (data["left_force_n"] >= metadata["min_force_n"]) & (data["right_force_n"] >= metadata["min_force_n"])
    stable = ((data["position_slip_m"] <= metadata["max_position_slip_m"])
              & (data["orientation_slip_rad"] <= metadata["max_orientation_slip_rad"]))
    candidate = active & data["eligible"].astype(bool) & both & loaded & stable
    persistence = np.asarray(initial_persistence, dtype=np.int64).copy()
    expected = np.zeros_like(actual)
    counter_mismatch = np.zeros_like(actual)
    for step in range(len(active)):
        persistence = np.where(candidate[step], persistence + 1, 0)
        counter_mismatch[step] = persistence != data["persistence_steps"][step]
        expected[step] = candidate[step] & (persistence >= metadata["persistence_steps"])
    expected_v2 = expected.copy()
    if not metadata["contact_based_grasp"]:
        expected &= ~data["release_open"].astype(bool)
    expected_score = actual if metadata["contact_based_grasp"] else actual & (record["gripper_opening"] <= .94)
    reasons = np.full(active.shape, "accepted", dtype="U32")
    distance_rejected = ((data["left_distance_rejected_force_n"] >= metadata["min_force_n"])
                         | (data["right_distance_rejected_force_n"] >= metadata["min_force_n"]))
    # First failing gate; the underlying numeric values are always retained.
    rules = [
        (actual != expected, "detector_mismatch"), (actual & ~scored, "scoring_rejected"),
        (expected_v2 & ~actual & data["release_open"].astype(bool), "legacy_opening_veto"),
        (data["persistence_steps"] < metadata["persistence_steps"], "persistence"),
        (data["orientation_slip_rad"] > metadata["max_orientation_slip_rad"], "orientation_slip"),
        (data["position_slip_m"] > metadata["max_position_slip_m"], "position_slip"),
        (~loaded, "force_below_threshold"), (~both, "missing_bilateral_contact"),
        (distance_rejected & (~both | ~loaded) & bool(metadata.get("raw_contact_available")), "contact_distance_filter"),
        (~finite_force, "nonfinite_force"), (~data["eligible"].astype(bool), "ineligible"),
        (~active, "inactive"),
    ]
    for mask, label in rules:
        reasons[mask] = label
    worlds = actual.shape[1]
    target = record["object_xyz"][:, np.arange(worlds), record["target_slots"]]
    relative = target - record["ee_xyz"]
    initial_relative = (record["reset_object_xyz"][np.arange(worlds), record["target_slots"]]
                        - record["reset_ee_xyz"])
    previous = np.concatenate([initial_relative[None], relative[:-1]], axis=0)
    lift = target[..., 2] - record["initial_target_xyz"][None, :, 2]
    motion = (usable & np.isfinite(relative).all(axis=-1) & (lift >= .05)
              & (np.linalg.norm(relative, axis=-1) <= .04)
              & (np.linalg.norm(relative - previous, axis=-1) <= metadata["max_position_slip_m"]))
    duration = np.zeros(worlds, dtype=np.int64)
    proxy = np.zeros_like(actual)
    for step in range(len(active)):
        duration = np.where(motion[step], duration + 1, 0)
        proxy[step] = duration >= 3
    suspicious = proxy & ~actual
    round_index = int(record["round_index"])
    rows = []
    for world in range(worlds):
        mask = usable[:, world]
        catalog = int(record["target_catalog_ids"][world]) if "target_catalog_ids" in record else -1
        rows.append({
            "round": round_index, "world": world,
            "instruction": metadata["instructions"][int(record["instruction_ids"][world])],
            "object": metadata["catalogs"][catalog] if 0 <= catalog < len(metadata["catalogs"]) else "unknown",
            "success": bool((record["success"][:, world] & record["active"][:, world]).any()),
            "live_diverged": bool(live_diverged[world]), "active_steps": int(active[:, world].sum()),
            "audited_steps": int(mask.sum()),
            "ever_counted_grasp": bool((actual[:, world] & active[:, world]).any()),
            "detector_mismatch_steps": int(((actual != expected)[:, world] & mask).sum()),
            "all_gates_passed_but_uncounted_steps": int((expected[:, world] & ~actual[:, world] & mask).sum()),
            "grasped_but_not_scored_steps": int((actual[:, world] & ~scored[:, world] & mask).sum()),
            "legacy_opening_veto_steps": int((expected_v2[:, world] & ~actual[:, world]
                                              & data["release_open"][:, world].astype(bool) & mask).sum()) if not metadata["contact_based_grasp"] else 0,
            "scoring_mismatch_steps": int(((scored != expected_score)[:, world] & mask).sum()),
            "persistence_mismatch_steps": int((counter_mismatch[:, world] & mask).sum()),
            "nonfinite_force_steps": int((~finite_force[:, world] & mask).sum()),
            "nonfinite_raw_pad_force_steps": int((mask
                & (~np.isfinite(data["raw_left_force_n"][:, world])
                   | ~np.isfinite(data["raw_right_force_n"][:, world]))).sum()) if metadata.get("raw_contact_available") else 0,
            "counted_on_nonfinite_force_steps": int((actual[:, world] & ~finite_force[:, world] & mask).sum()),
            "motion_proxy_unrecognized_steps": int(suspicious[:, world].sum()),
            "wide_loaded_unrecognized_steps": int((mask & both[:, world] & loaded[:, world]
                                                    & finite_force[:, world] & ~actual[:, world]
                                                    & (data["persistence_steps"][:, world] >= metadata["persistence_steps"])
                                                    & (record["gripper_opening"][:, world] > .94)).sum()),
            "wide_loaded_counted_steps": int((mask & actual[:, world] & finite_force[:, world]
                                              & (record["gripper_opening"][:, world] > .94)).sum()),
            "opening_v1_would_reject_steps": int((mask & actual[:, world]
                                                  & (data["release_open"][:, world].astype(bool)
                                                     | (record["gripper_opening"][:, world] > .94))).sum()),
            "loaded_contacts_outside_distance_window_steps": int((mask & ~actual[:, world]
                & ((data["left_distance_rejected_force_n"][:, world] >= metadata["min_force_n"])
                   | (data["right_distance_rejected_force_n"][:, world] >= metadata["min_force_n"]))).sum()) if metadata.get("raw_contact_available") else 0,
            "suspect_rejection_reasons": dict(Counter(reasons[suspicious[:, world], world].tolist())),
        })
    return rows, data, reasons, suspicious


def analyze_files(pattern, output):
    paths = sorted(Path(path) for path in glob.glob(str(pattern)))
    if not paths:
        raise ValueError(f"No recordings match {pattern}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes = []
    sources = []
    for path in paths:
        trace = path.with_name(path.name.replace("record_", "grasp_trace_", 1))
        if not trace.is_file():
            raise ValueError(f"Missing {trace}; re-record with --grasp-audit for actual contact forces.")
        with np.load(path, allow_pickle=False) as archive:
            record = dict(archive)
        with np.load(trace, allow_pickle=False) as archive:
            if str(archive["recording_sha256"].item()) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"Trace fingerprint differs from {path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("schema") != 1 or set(archive["columns"].tolist()) != set(COLUMNS):
                raise ValueError(f"Unsupported grasp trace schema in {trace}")
            sources.append({"recording": str(path), "trace": str(trace), **metadata})
            rows, data, reasons, suspicious = diagnose(record, archive["values"], archive["columns"].tolist(),
                                                       metadata, archive["initial_persistence"])
        episodes.extend(rows)
        # Save complete step traces for up to 16 suspect episodes per round.
        suspects = sorted(rows, key=lambda row: (row["detector_mismatch_steps"] + row["scoring_mismatch_steps"],
                                                row["motion_proxy_unrecognized_steps"], row["nonfinite_force_steps"] + row["nonfinite_raw_pad_force_steps"],
                                                row["loaded_contacts_outside_distance_window_steps"]), reverse=True)
        for row in suspects[:16]:
            if not (row["motion_proxy_unrecognized_steps"] or row["detector_mismatch_steps"]
                    or row["scoring_mismatch_steps"] or row["nonfinite_force_steps"] or row["nonfinite_raw_pad_force_steps"]):
                if not row["loaded_contacts_outside_distance_window_steps"]:
                    continue
            world = row["world"]
            target = output / f"trace_r{row['round']}_w{world}.csv"
            fields = ["env_step", *COLUMNS, "opening", "rejection", "motion_proxy_unrecognized"]
            with target.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for step in np.flatnonzero(record["active"][:, world]):
                    writer.writerow({"env_step": int(step + 1), **{key: data[key][step, world] for key in COLUMNS},
                                     "opening": record["gripper_opening"][step, world],
                                     "rejection": reasons[step, world], "motion_proxy_unrecognized": suspicious[step, world]})
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows({**row, "suspect_rejection_reasons": json.dumps(row["suspect_rejection_reasons"])} for row in episodes)
    groups = defaultdict(list)
    for row in episodes:
        groups[(row["instruction"], row["object"])].append(row)
    summary = []
    for (instruction, obj), rows in sorted(groups.items()):
        valid = [row for row in rows if not row["live_diverged"]]
        summary.append({"instruction": instruction, "object": obj, "episodes": len(rows),
                        "live_diverged": len(rows) - len(valid),
                        **{key: sum(row[key] for row in valid) for key in rows[0] if key.endswith("_steps")},
                        "never_counted_but_motion_proxy": sum(not row["ever_counted_grasp"] and row["motion_proxy_unrecognized_steps"] > 0 for row in valid)})
    report = {"by_object": summary, "episodes": len(episodes), "sources": sources,
              "cohort": "Gate/motion evidence excludes live-divergent episodes. Live divergences are counted separately; idle later divergences do not invalidate completed episodes.",
              "interpretation": "Motion without a counted grasp is a suspicion flag. Check actual contacts/forces and video; motion alone does not prove a valid bilateral grasp."}
    (output / "grasp_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in summary:
        print("[grasp-audit] " + json.dumps(row), flush=True)
    return report


def annotate_frames(frames, recording, capture, world):
    from PIL import Image, ImageDraw
    result = []
    for index, frame in enumerate(frames):
        canvas = Image.fromarray(np.pad(frame, ((0, 96), (0, 0), (0, 0))))
        draw = ImageDraw.Draw(canvas)
        top = frame.shape[0]
        if index == 0:
            lines = ["RESET: before first action; contact measurements begin after physics steps."]
        elif index == len(frames) - 1:
            lines = ["POST-ROLLOUT VIEW: later than episode termination; no gate comparison here.",
                     f"Recorded episode success: {bool(recording.episode_success[world])}"]
        else:
            step = index * recording.actions_per_decision - 1
            row = dict(zip(COLUMNS, capture.values[step, world]))
            lines = [
                f"After env step {step+1}: physical={int(row['physical_grasp'])} scored={int(row['scored_grasp'])} opening={recording.gripper_opening[step, world]:.3f}",
                f"Pad contact L/R={int(row['left_contact'])}/{int(row['right_contact'])}; force L/R={row['left_force_n']:.3g}/{row['right_force_n']:.3g} N (min {capture.metadata['min_force_n']:.3g} N)",
                f"Relative slip={row['position_slip_m']*1000:.2f} mm, {row['orientation_slip_rad']:.3f} rad; persistence={int(row['persistence_steps'])}/{capture.metadata['persistence_steps']}; eligible={int(row['eligible'])}",
            ]
            if capture.metadata.get("raw_contact_available"):
                def gap(side):
                    value = row[f"{side}_contact_distance_m"]
                    return f"{value*1000:.2f}" if np.isfinite(value) else "--"
                lines.append(f"Buffer gap L/R={gap('left')}/{gap('right')}mm; force rejected by distance filter={row['left_distance_rejected_force_n']:.3g}/{row['right_distance_rejected_force_n']:.3g}N")
        for line_number, line in enumerate(lines):
            draw.text((5, top + 5 + line_number * 20), line, fill="white")
        result.append(np.asarray(canvas))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Record the trained run's latest policy on the GPU")
    source.add_argument("--recordings", help="Analyze existing record_*.npz and paired grasp_trace_*.npz")
    parser.add_argument("--checkpoint", type=Path, help="Optional override, for example the best checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--round-index", type=int, default=1000)
    parser.add_argument("--horizon-decisions", type=int, default=0,
                        help="0 keeps the original budget; positive values test a longer rollout")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.horizon_decisions < 0:
        parser.error("horizon-decisions must be nonnegative")
    if args.recordings:
        analyze_files(args.recordings, args.output)
        return 0
    from tools.train.continue_widowx200_stage7 import configure_huggingface_env
    run = args.run_dir.expanduser().resolve()
    lineage = json.loads((run / "continuation.json").read_text())
    checkpoint = (args.checkpoint or run / "rl/latest.pt").resolve()
    if not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")
    env = dict(os.environ, RLVLA_EMBODIMENT="widowx200", PYTHONUNBUFFERED="1")
    configure_huggingface_env(env)
    env.setdefault("MUJOCO_GL", "egl")
    command = [sys.executable, str(ROOT / "tools/audit/policy_outcome_videos.py"),
               "--pipeline-dir", lineage["source_pipeline"], "--evaluation", "put_into_composed",
               "--checkpoint", str(checkpoint), "--device", args.device, "--grasp-audit",
               "--round-index", str(args.round_index), "--rounds", str(args.rounds),
               "--horizon-decisions", str(args.horizon_decisions),
               "--track-scenes", "32", "--output", str(args.output.resolve())]
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
