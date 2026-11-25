#!/usr/bin/env python3
"""Build evaluation-only SmolVLA checkpoints and run paired native rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

EXPERT_LAYER = re.compile(r"\.lm_expert\.layers\.(\d+)\.")
PROJECTIONS = ("13:gate_proj", "14:gate_proj", "14:q_proj")
CONTRACT = (
    "policy_type", "base_checkpoint", "state_dim", "action_dim", "chunk_size",
    "residual_scale",
)
ARG_CONTRACT = (
    "lora_rank", "lora_alpha", "image_size", "image_feature_keys", "state_dim",
    "include_wrist", "include_aux_camera", "mask_empty_aux_camera",
    "smolvla_action_indices", "smolvla_action_normalization", "smolvla_model_image_size",
    "residual_vision_dim", "residual_vision_pooling", "residual_vision_features",
    "residual_relative_target", "replan_every",
)


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() if hasattr(
            hashlib, "file_digest"
        ) else _legacy_digest(file)


def _legacy_digest(file) -> str:
    sha = hashlib.sha256()
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        sha.update(chunk)
    return sha.hexdigest()


def load_checkpoint(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def prepared_payload(payload: dict, vision: str) -> dict:
    """Keep exactly the supported LoRA families and discard resume-only state."""
    from compare_smolvla_checkpoint_weights import lora_factors, lora_group

    args = dict(payload["args"])
    factors = lora_factors(payload)
    unknown = [name for name in factors if lora_group(name) == "other"]
    if unknown:
        raise ValueError(f"Unsupported LoRA modules: {unknown[:4]}")
    has_vision = any(lora_group(name) == "vision_encoder" for name in factors)
    args["train_vla_vision_lora"] = vision == "saved" and has_vision
    weights = {
        key: value for key, value in (payload.get("vla_lora") or {}).items()
        if args["train_vla_vision_lora"] or ".vision_model." not in key
    }
    # Eval needs the curriculum and the checkpoint's original runtime contract.
    return {
        **{key: value for key, value in payload.items() if "optimizer" not in key},
        "args": args,
        "vla_lora": weights,
    }


def effective_change(left, right, alpha: float) -> float:
    """Frobenius norm of alpha/r * (B2 A2 - B1 A1) without a dense projection."""
    import torch

    a1, b1 = left
    a2, b2 = right
    if torch.equal(a1, a2) and torch.equal(b1, b2):
        return 0.0
    left_factor = torch.cat((b2.double(), -b1.double()), dim=1)
    right_factor = torch.cat((a2.double(), a1.double()), dim=0)
    _, r = torch.linalg.qr(left_factor, mode="reduced")
    return float(torch.linalg.vector_norm(r @ right_factor)) * alpha / a1.shape[0]


def check_compatible(before: dict, after: dict) -> None:
    differences = [key for key in CONTRACT if before.get(key) != after.get(key)]
    differences += [
        key for key in ARG_CONTRACT if before["args"].get(key) != after["args"].get(key)
    ]
    for key, default in (("policy_architecture", "bounded_residual_v0"),
                         ("action_likelihood", "clipped_action_v0")):
        if (before.get(key) or default) != (after.get(key) or default):
            differences.append(key)
    if differences:
        raise ValueError(f"Incompatible checkpoint contracts: {', '.join(differences)}")
    for field in ("policy", "vla_lora"):
        before_shapes = {key: tuple(value.shape) for key, value in before[field].items()}
        after_shapes = {key: tuple(value.shape) for key, value in after[field].items()}
        if before_shapes != after_shapes:
            raise ValueError(f"Different {field} tensor keys or dimensions")


def checkpoint_in_reference_world(reference: dict, source: dict) -> dict:
    return {**reference, "policy": source["policy"], "vla_lora": source["vla_lora"],
            "lora_sha256": source.get("lora_sha256")}


def make_series(args) -> None:
    import torch

    with args.selection.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["series"] == args.series]
    ids = {row["id"] for row in rows}
    if not rows or len(ids) != len(rows) or set(args.omit) - ids:
        raise ValueError("Unknown series, duplicate IDs or unknown --omit IDs")
    rows = sorted((row for row in rows if row["id"] not in args.omit),
                  key=lambda row: int(row["order"]))
    if len(rows) < 2 or len({row["robot"] for row in rows}) != 1:
        raise ValueError("A series needs at least two checkpoints for one robot")

    paths = {}
    for row in rows:
        label = row["id"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", label) or label == "recipient":
            raise ValueError(f"Invalid checkpoint ID: {label}")
        path = (args.repo_root / row["checkpoint_path"]).resolve()
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Checkpoint size differs from the inventory: {path}")
        paths[label] = path
    reference_id = args.reference or rows[0]["id"]
    if reference_id not in paths:
        raise ValueError("--reference must be included in the selected series")
    reference = prepared_payload(load_checkpoint(paths[reference_id]), args.vision)

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".smolvla_series_") as temporary:
        stage = Path(temporary) / "series"
        stage.mkdir()
        arms = {}
        for row in rows:
            label = row["id"]
            path = paths[label]
            source = reference if label == reference_id else prepared_payload(load_checkpoint(path), args.vision)
            try:
                check_compatible(reference, source)
            except ValueError as error:
                raise ValueError(f"{label} vs {reference_id}: {error}") from error
            arm = "recipient" if label == reference_id else label
            filename = f"{arm}.pt"
            torch.save(checkpoint_in_reference_world(reference, source), stage / filename)
            arms[arm] = {**row, "source_path": str(path), "source_sha256": digest(path),
                         "source_global_step": source.get("global_step"), "file": filename}
        write_json(stage / "manifest.json", {"series": args.series, "reference": reference_id,
                                            "vision": args.vision, "arms": arms})
        stage.rename(output)
    print(output / "manifest.json")


def make(args) -> None:
    import torch

    from compare_smolvla_checkpoint_weights import lora_factors

    before = prepared_payload(load_checkpoint(args.recipient), args.vision)
    after = prepared_payload(load_checkpoint(args.donor), args.vision)
    check_compatible(before, after)
    bef, aft = lora_factors(before), lora_factors(after)
    by_layer = {name: int(match[1]) for name in bef if (match := EXPERT_LAYER.search(name))}
    if not by_layer:
        raise ValueError("No matching action-expert LoRA in these checkpoints")
    layers = sorted(set(by_layer.values()))
    alpha = float(before["args"]["lora_alpha"])
    norms = {name: effective_change(bef[name], aft[name], alpha) for name in by_layer}
    planned = {"all_expert": set(by_layer)}
    if args.experiment in ("projection", "both"):
        for projection in args.projection or PROJECTIONS:
            layer, family = projection.split(":", 1)
            names = {name for name, index in by_layer.items()
                     if index == int(layer) and name.rsplit(".", 1)[-1] == family}
            if len(names) != 1:
                raise ValueError(f"{projection}: expected one expert projection; got {sorted(names)}")
            planned[f"layer_{int(layer):02d}_{family}"] = names
    if args.experiment in ("depth", "both"):
        for depth in sorted(set(args.depth)):
            if not 0 < depth <= len(layers) // 2:
                raise ValueError(f"Depth {depth} would overlap the early and late sets: {layers}")
            for label, block in (("early", layers[:depth]), ("late", layers[-depth:])):
                planned[f"{label}_{depth:02d}"] = {
                    name for name, index in by_layer.items() if index in block
                }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arms = {}
    # Evaluate every arm with identical recipient reset/curriculum settings.
    donor_weights = checkpoint_in_reference_world(before, after)
    donor_residual = {**before, "policy": after["policy"]}
    for label, payload in (("recipient", before), ("donor_full", donor_weights),
                           ("donor_residual", donor_residual)):
        filename = f"{label}.pt"
        torch.save(payload, output / filename)
        arms[label] = {"file": filename}

    for label, names in planned.items():
        changes = sorted(name for name in names if norms[name] > args.noop_threshold)
        arm = {
            "selected": sorted(names),
            "effective_delta_norm": math.sqrt(sum(norms[name] ** 2 for name in names)),
            "changed": changes,
        }
        if changes:
            weights = dict(before["vla_lora"])
            for name in names:
                for factor in ("lora_a", "lora_b"):
                    key = f"{name}.{factor}"
                    weights[key] = after["vla_lora"][key]
            filename = f"{label}.pt"
            # Provenance for an intervention comes from the manifest.
            torch.save({**before, "vla_lora": weights, "lora_sha256": None}, output / filename)
            arm["file"] = filename
        else:
            arm["status"] = "effective_noop"
        arms[label] = arm

    manifest = {
        "recipient": {"path": str(args.recipient.resolve()), "sha256": digest(args.recipient),
                      "step": before.get("global_step")},
        "donor": {"path": str(args.donor.resolve()), "sha256": digest(args.donor),
                  "step": after.get("global_step")},
        "vision": args.vision,
        "expert_layers": layers,
        "noop_threshold": args.noop_threshold,
        "expert_projection_delta_norms": norms,
        "arms": arms,
    }
    write_json(output / "manifest.json", manifest)
    print(output / "manifest.json")


def paired_stats(rows: np.ndarray, group_size: int, seed: int) -> dict:
    """Cluster bootstrap over independent reset groups, keeping verdicts paired."""
    differences = rows[:, :, 1] - rows[:, :, 0]
    clusters = differences.reshape(-1, group_size).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = rng.choice(clusters, size=(2000, len(clusters)), replace=True).mean(axis=1)
    return {
        "episodes": int(differences.size), "reset_groups": len(clusters),
        "recipient_success": float(rows[:, :, 0].mean()),
        "arm_success": float(rows[:, :, 1].mean()),
        "delta_pp": 100 * float(clusters.mean()),
        "cluster_bootstrap_ci95_pp": (100 * np.quantile(boot, [0.025, 0.975])).tolist(),
        "rescued": int(np.count_nonzero(differences > 0)),
        "lost": int(np.count_nonzero(differences < 0)),
    }


def verify_reset(first, second) -> None:
    for field in ("instruction_ids", "instructions", "target_slots", "reference_slots",
                  "horizons", "physical_grasp_at_reset", "target_catalog_ids", "contact_based_grasp"):
        if not np.array_equal(getattr(first, field), getattr(second, field)):
            raise ValueError(f"Unpaired reset, round {first.round_index}: {field}")
    for field in ("reset_object_xyz", "reset_ee_xyz", "reset_object_quaternions",
                  "reset_ee_quaternion"):
        a, b = getattr(first, field), getattr(second, field)
        if a is None or b is None or a.shape != b.shape or not np.allclose(a, b, atol=1e-6, rtol=0):
            raise ValueError(f"Unpaired or missing reset field, round {first.round_index}: {field}")


def series_report(manifest: dict, arm_rows: dict, baseline: np.ndarray,
                  labels: np.ndarray, group_size: int, seed: int,
                  holdout_quadrants=(), holdout_pairs=()) -> dict:
    scores = {"recipient": baseline, **{arm: rows for arm, (rows, _) in arm_rows.items()}}
    instructions = {name: labels[:, 1] == name for name in np.unique(labels[:, 1])}
    by_id, timeline = {}, []
    for arm, spec in manifest["arms"].items():
        if arm not in scores:
            continue
        label = spec["id"]
        by_id[label] = scores[arm]
        timeline.append({
            "id": label, "order": int(spec["order"]), "role": spec["role"],
            "source_global_step": spec["source_global_step"],
            "success": {name: float(scores[arm][mask].mean()) for name, mask in instructions.items()},
        })
    strata = {}
    if holdout_quadrants or holdout_pairs:
        bearing = np.isin(labels[:, 2], holdout_quadrants)
        pair = np.isin(labels[:, 3], holdout_pairs)
        strata = {"ordinary": ~(bearing | pair), "sector_only": bearing & ~pair,
                  "pair_only": pair & ~bearing, "sector_and_pair": bearing & pair}
    transitions = []
    for spec in manifest["arms"].values():
        before, after = spec["previous_id"], spec["id"]
        if before not in by_id or after not in by_id:
            continue
        paired = np.stack((by_id[before], by_id[after]), axis=-1)
        transition = {
            "before": before, "after": after, "evidence": spec["evidence"],
            "instruction": {name: paired_stats(paired[mask], group_size, seed)
                            for name, mask in instructions.items()},
        }
        if strata:
            transition["scene_group"] = {
                name: {group: paired_stats(paired[mask & part], group_size, seed)
                       for group, part in strata.items() if np.any(mask & part)}
                for name, mask in instructions.items()
            }
        transitions.append(transition)
    return {"series": manifest["series"], "reference": manifest["reference"],
            "timeline": timeline, "adjacent_transitions": transitions}


def evaluate(args) -> None:
    from sil_record import _Recording, _instruction_name
    from smolvla_skill_panel import strata
    from rl_vla_bootstrapping.simulation.cdpr_object_catalog import ACTIVE_CDPR_CATALOGS

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    arms = {arm: spec for arm, spec in manifest["arms"].items() if "file" in spec}
    if args.arm:
        chosen = {"recipient", *args.arm}
        unknown = chosen - arms.keys()
        if unknown:
            raise ValueError(f"Unavailable arms (see manifest): {sorted(unknown)}")
        arms = {arm: spec for arm, spec in arms.items() if arm in chosen}
    if not args.no_null:
        arms["recipient_repeat"] = arms["recipient"]
    paths = {}
    for arm, spec in arms.items():
        directory = args.output / arm
        command = [
            sys.executable, str(Path(__file__).with_name("sil_record.py")),
            "--mode", "record", "--checkpoint", str(args.manifest.parent / spec["file"]),
            "--config", str(args.config), "--output", str(directory),
            "--round-index", str(args.round_index), "--rounds", str(args.rounds),
            "--worlds", str(args.worlds), "--group-size", str(args.group_size),
            "--smolvla-microbatch", str(args.microbatch), "--seed-torch", str(args.seed),
            "--start-distance-cap", str(args.cap), "--horizon-decisions", str(args.horizon),
        ]
        if not args.controller_workspace_from_checkpoint:
            command.append("--controller-workspace-from-config")
        if args.metadata_override:
            command.extend(("--metadata-override", *args.metadata_override))
        command.extend(("--devices", args.devices) if args.devices else ("--device", args.device))
        subprocess.run(command, check=True)
        paths[arm] = sorted(directory.glob("record_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
        if len(paths[arm]) != args.rounds:
            raise RuntimeError(f"{arm}: expected {args.rounds} recordings, found {len(paths[arm])}")

    report, arm_rows = {}, {}
    baseline = labels = None
    for arm, recordings in paths.items():
        if arm == "recipient":
            continue
        pair_rows, group_labels = [], []
        for base_path, arm_path in zip(paths["recipient"], recordings):
            base, test = (_Recording.from_npz(path) for path in (base_path, arm_path))
            if base.round_index != test.round_index:
                raise ValueError("Round indexing diverged between arms")
            verify_reset(base, test)
            episodes = np.stack((base.episode_success, test.episode_success), axis=-1)
            pair_rows.append(episodes.reshape(-1, args.group_size, 2).astype(float))
            picks = np.arange(0, args.worlds, args.group_size)
            names = [_instruction_name(i) for i in base.instruction_ids[picks]]
            held = base.physical_grasp_at_reset[picks]
            skills = np.asarray([
                "placement" if name.startswith("put_into") and caught else
                "composed" if name.startswith("put_into") else "move_to"
                if name.startswith("move_to") else name for name, caught in zip(names, held)
            ])
            slot = np.where(skills == "placement", base.reference_slots[picks], base.target_slots[picks])
            delta = base.reset_object_xyz[picks, slot, :2] - base.reset_ee_xyz[picks, :2]
            targets = np.asarray([ACTIVE_CDPR_CATALOGS[i] for i in base.target_catalog_ids[picks]])
            receptacles = np.asarray([name.removeprefix("put_into_") if
                                       skill == "placement" else "" for name, skill in zip(names, skills)])
            bearings, pairs = strata(delta, targets, receptacles)
            group_labels.extend(zip(skills, names, bearings, pairs))
        all_rows = np.concatenate(pair_rows)
        labels = np.asarray(group_labels)
        baseline = all_rows[:, :, 0]
        arm_report = {"all": paired_stats(all_rows, args.group_size, args.seed)}
        for column, name in enumerate(("skill", "instruction", "bearing", "pair")):
            arm_report[name] = {
                label: paired_stats(all_rows[labels[:, column] == label], args.group_size, args.seed)
                for label in np.unique(labels[:, column]) if label
            }
        report[arm] = arm_report
        arm_rows[arm] = all_rows[:, :, 1], labels
    depth_contrasts, depth_by_instruction = {}, {}
    for early in sorted(name for name in arm_rows if name.startswith("early_")):
        late = early.replace("early_", "late_")
        if late not in arm_rows:
            continue
        early_scores, labels = arm_rows[early]
        late_scores, late_labels = arm_rows[late]
        if not np.array_equal(labels, late_labels):
            raise ValueError(f"Non-matching labels in {early} vs {late}")
        scores = np.stack((early_scores, late_scores), axis=-1)
        depth_contrasts[f"{late}_vs_{early}"] = {
            skill: paired_stats(scores[labels[:, 0] == skill], args.group_size, args.seed)
            for skill in np.unique(labels[:, 0])
        }
        depth_by_instruction[f"{late}_vs_{early}"] = {
            name: paired_stats(scores[labels[:, 1] == name], args.group_size, args.seed)
            for name in np.unique(labels[:, 1])
        }
    timeline = {}
    if "series" in manifest and baseline is not None:
        timeline = {"series_summary": series_report(manifest, arm_rows, baseline, labels,
                                                    args.group_size, args.seed,
                                                    args.holdout_quadrant, args.holdout_pair)}
    write_json(args.output / "paired.json", {
        "config": str(args.config), "config_sha256": digest(args.config),
        "manifest": str(args.manifest), "manifest_sha256": digest(args.manifest),
        "rollout": {key: value for key, value in vars(args).items()
                    if isinstance(value, (str, float, int, bool, list, type(None)))},
        "arms": report, "late_minus_early": depth_contrasts,
        "late_minus_early_by_instruction": depth_by_instruction, **timeline,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    build = sub.add_parser("make")
    build.add_argument("--recipient", type=Path, required=True)
    build.add_argument("--donor", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--experiment", choices=("projection", "depth", "both"), default="both")
    build.add_argument("--projection", action="append", metavar="LAYER:LEAF")
    build.add_argument("--depth", nargs="+", type=int, default=[1, 2, 4, 8])
    build.add_argument("--vision", choices=("exclude", "saved"), default="exclude")
    build.add_argument("--noop-threshold", type=float, default=1e-8)
    build.set_defaults(run=make)

    series = sub.add_parser("make-series", help="Build paired full-policy arms from a series CSV")
    series.add_argument("--selection", type=Path, required=True)
    series.add_argument("--series", required=True)
    series.add_argument("--repo-root", type=Path, default=Path.cwd())
    series.add_argument("--reference")
    series.add_argument("--omit", action="append", default=[])
    series.add_argument("--output", type=Path, required=True)
    series.add_argument("--vision", choices=("exclude", "saved"), default="exclude")
    series.set_defaults(run=make_series)

    roll = sub.add_parser("evaluate")
    for name in ("manifest", "config", "output"):
        roll.add_argument(f"--{name}", type=Path, required=True)
    roll.add_argument("--cap", type=float, required=True)
    roll.add_argument("--holdout-quadrant", action="append", choices=("NE", "NW", "SE", "SW"), default=[])
    roll.add_argument("--holdout-pair", action="append", default=[], metavar="CATALOG:RECEPTACLE")
    roll.add_argument("--controller-workspace-from-checkpoint", action="store_true",
                      help="Use saved reference controller bounds for configs without explicit bounds")
    roll.add_argument("--arm", action="append")
    roll.add_argument("--no-null", action="store_true")
    roll.add_argument("--metadata-override", nargs="+")
    roll.add_argument("--device", default="cuda:0")
    roll.add_argument("--devices")
    for name, default in (("rounds", 1), ("round-index", 0), ("worlds", 32), ("group-size", 2),
                          ("microbatch", 16), ("seed", 1729), ("horizon", 0)):
        roll.add_argument(f"--{name}", type=int, default=default)
    roll.set_defaults(run=evaluate)
    args = parser.parse_args()
    if args.run == evaluate and (min(args.worlds, args.rounds, args.microbatch) < 1 or args.group_size < 2
                                 or args.worlds % args.group_size or args.cap <= 0):
        parser.error("Use positive dimensions, a divisible world count, and a positive cap")
    if args.run == make and args.noop_threshold < 0:
        parser.error("--noop-threshold must be nonnegative")
    args.run(args)


if __name__ == "__main__":
    main()
