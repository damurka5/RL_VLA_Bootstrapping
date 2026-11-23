#!/usr/bin/env python3
"""Read out goal bearing from real SmolVLA layers and test cross-skill transfer."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from smolvla_skill_interventions import digest, load_checkpoint, prepared_payload, write_json
from smolvla_skill_panel import META, holdout_arguments, split_scenes

SKILLS = ("move_to", "placement")
FEATURE = "feature__"


def extract(args) -> None:
    import torch

    from sil_sft import build_runtime_and_trainer, recompute_state_and_prior

    if args.output.exists():
        raise FileExistsError(args.output)
    payload = prepared_payload(load_checkpoint(args.checkpoint), args.vision)
    if payload["args"].get("residual_relative_target"):
        raise ValueError("The policy consumes privileged direction in its state")
    if args.device == "cpu":
        payload["args"]["mixed_precision"] = "none"
    runtime, trainer, config = build_runtime_and_trainer(
        payload, checkpoint=args.checkpoint, device=args.device, evaluation_only=True,
    )
    expert = runtime.policy.model.vlm_with_expert.lm_expert
    indices = args.layer or list(range(len(expert.layers)))
    if len(indices) != len(set(indices)) or any(i < 0 or i >= len(expert.layers) for i in indices):
        raise ValueError(f"Invalid expert layer selection: {indices}")
    vision_dim = int(config.residual_vision_dim) if config.residual_vision_features else 0
    proprio_width = int(payload["state_dim"]) - vision_dim
    metadata, features = defaultdict(list), defaultdict(list)
    observed_steps = set()
    batch_index = 0
    first, last, calls = {}, {}, {}

    def hook(index):
        def receive(_module, _inputs, output):
            # The LeRobot expert executes blocks manually. Module hooks on the
            # individual expert MLPs see the activation before residual addition.
            value = output[:, 0].detach().float().clone()
            if index not in first:
                first[index] = value
            last[index] = value
            calls[index] = calls.get(index, 0) + 1
        return receive

    handles = [expert.layers[i].mlp.register_forward_hook(hook(i)) for i in indices]
    try:
        with torch.inference_mode():
            for path in args.panel:
                with np.load(path, allow_pickle=False) as source:
                    panel = {key: source[key] for key in (*META, "instruction", "proprio", "overview", "wrist")}
                rows = len(panel["uid"])
                if panel["proprio"].shape != (rows, proprio_width):
                    raise ValueError(f"Incompatible proprioception in {path}")
                for key in (*META, "instruction"):
                    metadata[key].append(panel[key])
                image_rows = np.arange(rows)
                if args.shuffle_images:
                    # Keep the joint camera distribution and prompt/object
                    # category. Singleton categories are rejected as controls.
                    groups = defaultdict(list)
                    for i, condition in enumerate(zip(panel["skill"], panel["target"],
                                                      panel["receptacle"], panel["instruction"])):
                        groups[condition].append(i)
                    rng = np.random.default_rng(args.noise_seed)
                    for group in groups.values():
                        if len(group) == 1:
                            raise ValueError(f"Singleton camera-control group in {path}; increase --worlds")
                        order = rng.permutation(group)
                        image_rows[order] = np.roll(order, 1)
                for start in range(0, rows, args.batch_size):
                    stop = min(rows, start + args.batch_size)
                    selected = image_rows[start:stop]
                    first.clear()
                    last.clear()
                    calls.clear()
                    torch.manual_seed(args.noise_seed + batch_index)
                    batch_index += 1
                    kwargs = {
                        "overview": torch.from_numpy(panel["overview"][selected]).to(trainer.device),
                        "wrist": torch.from_numpy(panel["wrist"][selected]).to(trainer.device),
                        "proprio": torch.from_numpy(panel["proprio"][start:stop]).to(trainer.device),
                        "instructions": panel["instruction"][start:stop].tolist(),
                    }
                    if vision_dim:
                        prior, state = recompute_state_and_prior(
                            runtime, torch, **kwargs, vision_dim=vision_dim, enable_grad=False,
                        )
                        features["pooled_vision"].append(state[:, proprio_width:].float().cpu().numpy())
                    else:
                        prior = runtime.sample_cdpr_chunks_from_tensors(
                            primary_images=kwargs["overview"], wrist_images=kwargs["wrist"],
                            states=kwargs["proprio"], instructions=kwargs["instructions"],
                            microbatch_size=0,
                        )
                        state = kwargs["proprio"]
                    if set(calls) != set(indices) or len(set(calls.values())) != 1:
                        raise RuntimeError(f"Expert hooks did not capture a common denoising schedule: {calls}")
                    observed_steps.update(calls.values())
                    action = trainer.deterministic_action_chunks_tensor(
                        states=state, priors=prior, action_count=1,
                    )
                    features["prior_xy"].append(prior[:, 0, :2].float().cpu().numpy())
                    features["policy_xy"].append(action[:, 0, :2].float().cpu().numpy())
                    features["proprio"].append(panel["proprio"][start:stop])
                    for label, values in (("first", first), ("last", last)):
                        for i in indices:
                            features[f"mlp_{i:02d}_{label}"].append(values[i].cpu().numpy())
                print(f"{path}: {rows} rows, expert calls {sorted(observed_steps)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    info = json.dumps({
        "checkpoint": str(args.checkpoint), "sha256": digest(args.checkpoint),
        "panels": [digest(path) for path in args.panel], "denoising_calls": sorted(observed_steps),
        "vision": args.vision, "noise_seed": args.noise_seed, "batch_size": args.batch_size,
        "shuffle_images": args.shuffle_images,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream, metadata=np.asarray(info),
            **{key: np.concatenate(parts) for key, parts in metadata.items()},
            **{f"{FEATURE}{key}": np.concatenate(parts) for key, parts in features.items()},
        )


def readout(x_train, y_train, x_test, ridge):
    center, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    x_train, x_test = (x_train - center) / scale, (x_test - center) / scale
    bias = y_train.mean(axis=0)
    # Solve the smaller of primal/dual systems, with strength per fit example.
    penalty = ridge * len(x_train)
    if x_train.shape[1] <= len(x_train):
        matrix = x_train.T @ x_train
        matrix.flat[:: len(matrix) + 1] += penalty
        return x_test @ np.linalg.solve(matrix, x_train.T @ (y_train - bias)) + bias
    matrix = x_train @ x_train.T
    matrix.flat[:: len(matrix) + 1] += penalty
    return (x_test @ x_train.T) @ np.linalg.solve(matrix, y_train - bias) + bias


def cosine(predicted, target):
    length = np.linalg.norm(predicted, axis=1, keepdims=True)
    return np.clip(np.sum(predicted * target, axis=1) / np.maximum(length[:, 0], 1e-12), -1, 1)


def score(predicted, target):
    alignment = cosine(predicted, target)
    return {"mean_cosine": float(alignment.mean()),
            "mean_angular_error_deg": float(np.degrees(np.arccos(alignment)).mean()),
            "zero_prediction_fraction": float((np.linalg.norm(predicted, axis=1) < 1e-12).mean())}


def paired_ci(uids, feature_cosine, baseline_cosine, seed):
    delta = feature_cosine - baseline_cosine
    _, groups = np.unique(uids, return_inverse=True)
    totals, counts = np.bincount(groups, weights=delta), np.bincount(groups)
    picks = np.random.default_rng(seed).integers(len(totals), size=(1000, len(totals)))
    bootstrap = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return np.quantile(bootstrap, [0.025, 0.975]).tolist()


def analyze(args) -> None:
    metadata, arrays, extraction = load_features(args.features)
    test_arrays, test_extraction = arrays, extraction
    if args.test_features:
        other_metadata, test_arrays, test_extraction = load_features(args.test_features)
        if set(arrays) != set(test_arrays) or any(
            not np.array_equal(value, other_metadata[key]) for key, value in metadata.items()
        ):
            raise ValueError("Train and test archives must have identical scene order and feature keys")
        for field in ("sha256", "panels", "noise_seed", "batch_size", "denoising_calls", "vision"):
            if extraction.get(field) != test_extraction.get(field):
                raise ValueError(f"Non-comparable camera-control setting: {field}")
    fit_split = split_scenes(metadata, args.holdout_quadrant, args.holdout_pair, args.seed)
    delta = metadata["delta_xy"].astype(float)
    lengths = np.linalg.norm(delta, axis=1, keepdims=True)
    if not np.isfinite(delta).all() or (lengths <= 0).any():
        raise ValueError("Feature panel contains invalid goal directions")
    unit = delta / lengths
    count_rows = defaultdict(int)
    for skill, label in zip(metadata["skill"], fit_split):
        count_rows[f"{skill}:{label}"] += 1

    report = []
    for dst_skill in SKILLS:
        for src_skill in SKILLS:
            train = (metadata["skill"] == src_skill) & (fit_split == "probe_train")
            for split in sorted(set(fit_split) - {"probe_train"}):
                test = (metadata["skill"] == dst_skill) & (fit_split == split)
                if not test.any():
                    continue
                n_train, n_test = (len(np.unique(metadata["uid"][m])) for m in (train, test))
                row = {"fit_skill": src_skill, "eval_skill": dst_skill, "split": split,
                       "train_scenes": n_train, "test_scenes": n_test}
                if min(n_train, n_test) < args.min_scenes:
                    report.append({**row, "status": "insufficient_scenes"})
                    continue
                constant = np.broadcast_to(unit[train].mean(axis=0), unit[test].shape)
                predictions = {
                    label: readout(feature[train], unit[train], test_arrays[label][test], args.ridge)
                    for label, feature in arrays.items()
                }
                base_scores = cosine(predictions["proprio"], unit[test])
                uid = metadata["uid"][test]
                metrics = {
                    label: {**score(predictions[label], unit[test]),
                            "mean_cosine_delta_vs_proprio_ci95": paired_ci(
                                uid, cosine(predictions[label], unit[test]), base_scores, args.seed,
                            )}
                    for label in sorted(predictions)
                }
                report.append({
                    **row, "constant": score(constant, unit[test]), "readouts": metrics,
                    "observed_action": {key: score(test_arrays[key][test], unit[test])
                                        for key in ("policy_xy", "prior_xy") if key in arrays},
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {
        "extraction": extraction, "test_extraction": test_extraction,
        "split_seed": args.seed, "ridge": args.ridge,
        "holdout_quadrant": args.holdout_quadrant, "holdout_pair": args.holdout_pair,
        "counts": dict(count_rows), "comparisons": report,
    })
    if not any("readouts" in row for row in report):
        raise ValueError(f"No analyzable probe comparisons. See counts in {args.output}")
    print(args.output)


def load_features(path):
    with np.load(path, allow_pickle=False) as data:
        meta = {key: data[key] for key in META}
        features = {key.removeprefix(FEATURE): data[key].astype(float)
                    for key in data.files if key.startswith(FEATURE)}
        info = json.loads(str(data["metadata"]))
    if "proprio" not in features or any(len(v) != len(meta["uid"]) or v.ndim != 2 or
                                       not np.isfinite(v).all() for v in features.values()):
        raise ValueError(f"Invalid feature arrays in {path}")
    return meta, features, info


def linear_cka(x, y):
    a, b = x - x.mean(axis=0), y - y.mean(axis=0)
    aa, bb = a @ a.T, b @ b.T
    norm = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.clip(np.sum(aa * bb) / norm, 0, 1)) if norm else None


def cka(args) -> None:
    left, a, info_a = load_features(args.before)
    right, b, info_b = load_features(args.after)
    for field in ("panels", "noise_seed", "batch_size", "shuffle_images", "denoising_calls", "vision"):
        if info_a.get(field) != info_b.get(field):
            raise ValueError(f"Non-comparable extraction setting: {field}")
    if any(not np.array_equal(left[key], right[key]) for key in META):
        raise ValueError("CKA requires the exact same scenes in the same order")
    result = {}
    for skill in SKILLS:
        mask = left["skill"] == skill
        if int(mask.sum()) >= args.min_scenes:
            result[skill] = {key: linear_cka(a[key][mask], b[key][mask])
                             for key in sorted(a.keys() & b.keys())}
    if not result:
        raise ValueError("Insufficient matched scenes for CKA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"before": info_a, "after": info_b, "linear_cka": result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    capture = sub.add_parser("extract")
    capture.add_argument("--checkpoint", type=Path, required=True)
    capture.add_argument("--panel", type=Path, nargs="+", required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--layer", type=int, action="append")
    capture.add_argument("--device", default="cuda:0")
    capture.add_argument("--batch-size", type=int, default=16)
    capture.add_argument("--noise-seed", type=int, default=1729)
    capture.add_argument("--vision", choices=("exclude", "saved"), default="exclude")
    capture.add_argument("--shuffle-images", action="store_true")
    capture.set_defaults(run=extract)

    fit = sub.add_parser("analyze")
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--test-features", type=Path, help="Matched archive from the shuffled-camera control")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--ridge", type=float, default=1.0)
    fit.add_argument("--min-scenes", type=int, default=12)
    holdout_arguments(fit)
    fit.set_defaults(run=analyze)

    compare = sub.add_parser("cka")
    for name in ("before", "after", "output"):
        compare.add_argument(f"--{name}", type=Path, required=True)
    compare.add_argument("--min-scenes", type=int, default=12)
    compare.set_defaults(run=cka)
    args = parser.parse_args()
    if args.run == analyze and (args.ridge <= 0 or args.min_scenes < 2):
        parser.error("Analyze needs a positive ridge and at least two scenes per comparison")
    if args.run == extract and args.batch_size < 1:
        parser.error("Batch size must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
