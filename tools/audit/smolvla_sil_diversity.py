#!/usr/bin/env python3
"""Match SIL exposure while varying the sectors represented in training."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from sil_record import _Recording, _build_dataset, _instruction_name
from smolvla_skill_interventions import check_compatible, digest, load_checkpoint, prepared_payload
from smolvla_skill_panel import strata
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import ACTIVE_CDPR_CATALOGS

PAIRS = ("robocasa_apple:plate", "robocasa_orange:bowl")
SECTORS = ("SW", "NW", "SE")


def annotate(rec, task, dataset):
    ids = np.asarray([int(uid.rsplit("w", 1)[1]) for uid in dataset["episode_uid"]])
    slot = rec.target_slots if task == "move_to" else rec.reference_slots
    starts = rec.reset_ee_xyz
    delta = rec.reset_object_xyz[np.arange(rec.worlds), slot, :2] - starts[:, :2]
    objects = np.asarray([ACTIVE_CDPR_CATALOGS[i] for i in rec.target_catalog_ids])
    destinations = np.asarray([_instruction_name(i).removeprefix("put_into_") if task == "placement"
                               else "" for i in rec.instruction_ids])
    sector, pairs = strata(delta, objects, destinations)
    poses = np.column_stack((starts, rec.reset_object_xyz.reshape(rec.worlds, -1), rec.target_catalog_ids))
    scenes = np.asarray([sha256(np.rint(p * 1000).astype("<i8").tobytes()).hexdigest() for p in poses])
    dataset["scene_uid"] = scenes[ids]
    dataset["target_catalog"] = objects[ids]
    dataset["destination"] = destinations[ids]
    dataset["sector"] = sector[ids]
    dataset["pair"] = pairs[ids]
    dataset["distance_bin"] = (np.linalg.norm(delta[ids], axis=1) / 0.05).astype(int)
    return dataset


def collect(run, initializer, manifest_path):
    import torch

    manifest = json.loads(manifest_path.read_text())
    if digest(initializer) != manifest["recipient"]["sha256"]:
        raise ValueError("Initializer does not match the recorded recipient")
    teacher_arm = "donor_residual"
    alias = Path(manifest["arms"][teacher_arm]["file"]).name
    initial = prepared_payload(load_checkpoint(initializer), manifest["vision"])
    teacher = load_checkpoint(manifest_path.parent / alias)
    check_compatible(initial, teacher)
    if any(not torch.equal(v, teacher["vla_lora"][k]) for k, v in initial["vla_lora"].items()):
        raise ValueError("Teacher LoRA differs from the initializer, so cached priors need refreshing")
    if all(torch.equal(v, teacher["policy"][k]) for k, v in initial["policy"].items()):
        raise ValueError("The teacher is the deterministic initializer and supplies no new targets")
    manifest_hash = digest(manifest_path)
    parts = []
    for task in ("move_to", "placement"):
        folder = run / task / teacher_arm
        paired = json.loads((folder.parent / "paired.json").read_text())
        if paired["manifest_sha256"] != manifest_hash:
            raise ValueError(f"{folder}: source manifest does not match the recordings")
        summary = json.loads((folder / "summary.json").read_text())
        embedded = summary.get("checkpoint")
        if summary.get("sharded"):
            embedded = json.loads((folder / summary["shard_summaries"][0]).read_text())["checkpoint"]
        if not embedded or Path(embedded).name != alias:
            raise ValueError(f"{folder}: expected recordings of {alias}")
        files = sorted(folder.glob("record_*.npz"))
        if not files:
            raise FileNotFoundError(folder)
        for path in files:
            rec = _Recording.from_npz(path)
            if not rec.has_observations or rec.diverged_world_mask is None or rec.target_catalog_ids is None:
                raise ValueError(f"Missing supervised inputs or metadata: {path}")
            if task == "placement" and not rec.physical_grasp_at_reset.all():
                raise ValueError(f"Expected held placement: {path}")
            success = rec.success & ~rec.diverged_world_mask[None, :]
            if success.any():
                rec = replace(rec, success=success)
                data, _ = _build_dataset([rec], keys=[f"{task}-{rec.round_index}"])
                parts.append(annotate(rec, task, data))
    return {key: np.concatenate([batch[key] for batch in parts]) for key in parts[0]}


def select(data, seed):
    masks = data["action_mask"].sum(axis=1)
    held = (data["sector"] == "NE") | np.isin(data["pair"], PAIRS)
    validation_scenes = set(data["scene_uid"][held])
    validation_scenes.update(uid for uid in np.unique(data["scene_uid"])
                             if sha256(f"{seed}/{uid}".encode()).digest()[0] < 51)
    validation = np.isin(data["scene_uid"], list(validation_scenes))
    cells = defaultdict(lambda: defaultdict(list))
    columns = (data["instruction_id"], data["target_catalog"], data["distance_bin"],
               data["decision_index"] // 4, masks)
    for i, cell in enumerate(zip(*columns)):
        if not validation[i]:
            cells[cell][data["sector"][i]].append(i)
    restricted, broad = [], []
    rng = np.random.default_rng(seed)
    for cell in sorted(cells):
        pools = cells[cell]
        sw, nw, se = (rng.permutation(pools[sector]).tolist() for sector in SECTORS)
        count = min(len(sw), 2 * (len(nw) + len(se)))
        if not count:
            continue
        restricted.extend(sw[:count])
        broad.extend(sw[:count // 2])
        for i in range(count - count // 2):
            preferred, other = (nw, se) if i % 2 == 0 else (se, nw)
            broad.append((preferred or other).pop())
    if not restricted or len(restricted) != len(broad):
        raise ValueError("The recordings do not support matched broad/restricted cells")
    ref, test = map(np.asarray, (restricted, broad))
    if set(data["instruction_id"][ref]) != set(data["instruction_id"]):
        raise ValueError("At least one instruction has no matched training cell")
    for column in columns:
        if Counter(column[ref]) != Counter(column[test]):
            raise RuntimeError("A matched exposure marginal changed")
    return {"restricted": ref, "diverse": test}, np.flatnonzero(validation)


def build(args):
    data = collect(args.recordings, args.initializer, args.source_manifest)
    arms, val = select(data, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for name, train in arms.items():
        all_rows = np.concatenate((train, val))
        arrays = {key: value[all_rows] for key, value in data.items()}
        arrays["sft_validation"] = np.arange(len(all_rows)) >= len(train)
        np.savez_compressed(args.output / f"{name}.npz", **arrays)
        summary[name] = {"train_decisions": len(train), "train_valid_actions": int(data["action_mask"][train].sum()),
                         "train_scenes": len(set(data["scene_uid"][train])),
                         "train_sectors": dict(Counter(data["sector"][train])),
                         "train_by_instruction": dict(Counter(map(str, data["instruction_id"][train]))),
                         "validation_decisions": len(val)}
    report = {"recordings": str(args.recordings), "initializer": str(args.initializer),
              "initializer_sha256": digest(args.initializer), "source_manifest_sha256": digest(args.source_manifest), "seed": args.seed,
              "teacher_arm": "donor_residual", "excluded_sector": "NE",
              "excluded_pairs": list(PAIRS), "arms": summary}
    (args.output / "design.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def selection(args):
    design = json.loads((args.training / "data" / "design.json").read_text())
    if digest(args.initializer) != design["initializer_sha256"]:
        raise ValueError("SFT initializer differs from the dataset design")
    checkpoints = [args.initializer, *[args.training / arm / "sil_final_residual.pt"
                                        for arm in ("restricted", "diverse")]]
    runs = [json.loads((args.training / arm / "sft_report.json").read_text())
            for arm in ("restricted", "diverse")]
    for key in ("optimizer_updates", "supervised_actions_sampled", "training_settings"):
        if runs[0][key] != runs[1][key]:
            raise ValueError(f"SFT budgets differ: {key}")
    with args.output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("series", "robot", "order", "id", "role", "previous_id", "evidence",
                         "checkpoint_path", "size_bytes"))
        previous = ""
        for order, (label, path) in enumerate(zip(("B", "restricted", "diverse"), checkpoints), 1):
            writer.writerow(("sil_diversity", "cdpr", order, label, label, previous, "matched_budget",
                             path.resolve(), path.stat().st_size))
            previous = label



def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from sil_sft import _build_actor, _evaluate, _load_dataset

    data = _load_dataset(args.training / "data" / "restricted.npz")
    other = _load_dataset(args.training / "data" / "diverse.npz")
    val = data["sft_validation"]
    if any(not np.array_equal(data[k][val], other[k][other["sft_validation"]]) for k in data):
        raise ValueError("SIL branches must use identical validation rows")
    ne = data["sector"] == "NE"
    pair = np.isin(data["pair"], PAIRS)
    folds = {"all": np.ones(len(val), dtype=bool), "ordinary": ~(ne | pair),
             "NE_only": ne & ~pair, "pair_only": pair & ~ne, "NE_and_pair": ne & pair}
    instructions = {"all": np.ones(len(val), dtype=bool)}
    instructions.update({_instruction_name(i): data["instruction_id"] == i
                         for i in np.unique(data["instruction_id"])})
    device = torch.device("cpu")
    summaries, rows = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for arm in ("restricted", "diverse"):
        summary = json.loads((args.training / arm / "sft_report.json").read_text())
        summaries.append((arm, summary))
        actor = _build_actor(load_checkpoint(args.training / arm / "sil_final_residual.pt"), device).eval()
        for name, skill in instructions.items():
            for fold, member in folds.items():
                keep = val & skill & member
                if not keep.any():
                    continue
                tensors = {k: torch.from_numpy(data[k][keep]) for k in ("state", "prior", "action")}
                stats = _evaluate(actor, torch, **tensors,
                                  mask=torch.from_numpy(data["action_mask"][keep]), batch_size=128)
                rows.append({"arm": arm, "instruction": name, "fold": fold, "decisions": int(keep.sum()),
                             "scenes": len(set(data["scene_uid"][keep])), **stats})
    with (args.output / "offline_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, summary in summaries:
        history = summary["history"]
        ax.plot([x["epoch"] + 1 for x in history], [x["val_mse"] for x in history],
                marker="o", markersize=3, label=arm.capitalize())
    ax.set(title="CDPR SIL · validation on shared held-out scenes", xlabel="Epoch",
           ylabel="Masked action MSE")
    ax.grid(axis="y", alpha=.25)
    ax.legend(title="Training scene coverage")
    fig.tight_layout()
    fig.savefig(args.output / "offline_validation.png", dpi=180)
    plt.close(fig)
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    data = sub.add_parser("build")
    data.add_argument("--recordings", type=Path, required=True)
    data.add_argument("--source-manifest", type=Path, required=True)
    data.add_argument("--seed", type=int, default=1729)
    data.set_defaults(run=build)
    manifest = sub.add_parser("selection")
    manifest.add_argument("--training", type=Path, required=True)
    manifest.set_defaults(run=selection)
    offline = sub.add_parser("report")
    offline.add_argument("--training", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    offline.set_defaults(run=report)
    for cli in (data, manifest):
        cli.add_argument("--initializer", type=Path, required=True)
        cli.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
