#!/usr/bin/env python3
"""Audit scene strata and reconstruct first-decision residual activations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from types import SimpleNamespace
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sil_record import _instruction_name
from smolvla_direction_probe import cosine, linear_cka, paired_ci, readout
from smolvla_projection_report import ARMS, TASKS, TITLES, table_row
from smolvla_skill_interventions import digest, load_checkpoint, paired_stats, verify_reset
from smolvla_skill_panel import split_scenes, strata
from rl_vla_bootstrapping.policy.octo_finetune_cdpr import ResidualChunkActor
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import ACTIVE_CDPR_CATALOGS

PAIRS = ("robocasa_apple:plate", "robocasa_orange:bowl")
FOLDS = ("ordinary", "NE_only", "pair_only", "NE_and_pair")
FEATURES = ("proprio", "vision_pool", "prior", "hidden_1", "hidden_2", "bounded_residual")
RESET = ("instruction_ids", "instructions", "target_slots", "reference_slots",
         "second_reference_slots", "horizons", "physical_grasp_at_reset",
         "target_catalog_ids", "contact_based_grasp", "reset_object_xyz",
         "reset_ee_xyz", "reset_object_quaternions", "reset_ee_quaternion")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_actor(path):
    ckpt = load_checkpoint(path)
    if ckpt.get("policy_architecture", "bounded_residual_v0") != "bounded_residual_v0":
        raise ValueError("This reconstruction needs the legacy bounded residual policy")
    policy = ckpt["policy"]
    actor = ResidualChunkActor(
        state_dim=ckpt["state_dim"], chunk_size=ckpt["chunk_size"], action_dim=ckpt["action_dim"],
        hidden_dim=policy["actor.net.net.0.weight"].shape[0], residual_scale=ckpt["residual_scale"],
    )
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in policy.items()
                           if k.startswith("actor.")})
    return actor.eval(), int(ckpt["args"]["state_dim"])


@torch.inference_mode()
def activations(actor, states, priors, proprio_width):
    layers = []
    hooks = [layer.register_forward_hook(lambda _, _x, y: layers.append(y.numpy().copy()))
             for layer in actor.net.net if isinstance(layer, torch.nn.ReLU)]
    try:
        residual, logits = actor.reference_terms(torch.from_numpy(states), torch.from_numpy(priors))
    finally:
        for hook in hooks:
            hook.remove()
    blocks = (states[:, :proprio_width], states[:, proprio_width:], priors.reshape(len(states), -1),
              *layers, residual.numpy().reshape(len(states), -1))
    return dict(zip(FEATURES, blocks, strict=True)), torch.tanh(logits).numpy()


def read_records(run, swaps):
    models = {key: load_actor(swaps / f"{key}.pt") for key in ("recipient", "donor_full")}
    scene_rows, cka_rows = [], []
    errors, divergence = {}, {}
    meta = defaultdict(list)
    transfer = {key: defaultdict(list) for key in models}
    for task, instructions in TASKS.items():
        notes = json.loads((run / task / "paired.json").read_text())
        if digest(swaps / "manifest.json") != notes["manifest_sha256"]:
            raise ValueError("The supplied swaps do not match the recording manifest")
        cfg = Path(__file__).resolve().parents[2].joinpath(*Path(notes["config"]).parts[-3:])
        if digest(cfg) != notes["config_sha256"]:
            raise ValueError(f"Configuration hash mismatch: {cfg}")
        setup = notes["rollout"]
        group = setup["group_size"]
        groups = defaultdict(list)
        outcomes = defaultdict(list)
        scope = {key: defaultdict(list) for key in ("recipient", "donor_full", "donor_common_input")}
        for rnd in range(setup["round_index"], setup["round_index"] + setup["rounds"]):
            filename = f"record_{rnd:02d}.npz"
            observations = {}
            baseline = None
            for arm in ("recipient", *ARMS):
                with np.load(run / task / arm / filename, allow_pickle=False) as record:
                    reset = SimpleNamespace(**{k: record[k] for k in RESET}, round_index=int(record["round_index"]))
                    if reset.round_index != rnd:
                        raise ValueError(f"Unexpected round in {filename}")
                    if baseline is None:
                        baseline = reset
                    verify_reset(baseline, reset)
                    if not np.array_equal(baseline.second_reference_slots, reset.second_reference_slots):
                        raise ValueError(f"Reference slots differ in {filename}")
                    outcomes[arm].append(record["success"].any(axis=0).reshape(-1, group))
                    divergence[task, arm] = divergence.get((task, arm), 0) + int(record["diverged_worlds"])
                    if arm in models:
                        observations[arm] = record["states"][0], record["priors"][0]
                        actor, proprio_width = models[arm]
                        features, actions = activations(actor, *observations[arm], proprio_width)
                        executed = int(record["actions_per_decision"])
                        logged = record["actions"][:executed].transpose(1, 0, 2)
                        max_error = float(np.max(np.abs(actions[:, :executed] - logged)))
                        if max_error > 1e-4:
                            raise ValueError(f"Reconstruction error {max_error:g} in {task}/{arm}/{rnd}")
                        errors[task, arm] = max(errors.get((task, arm), 0), max_error)
                        for name, values in features.items():
                            scope[arm][name].append(values[::group])
            control, _ = activations(models["donor_full"][0], *observations["recipient"], models["donor_full"][1])
            for name, values in control.items():
                scope["donor_common_input"][name].append(values[::group])

            idx = np.arange(0, len(baseline.instruction_ids), group)
            ids = baseline.instruction_ids.reshape(-1, group)
            if not np.equal(ids, ids[:, :1]).all():
                raise ValueError("A reset group contains different instructions")
            names = np.asarray([_instruction_name(i) for i in ids[:, 0]])
            slots = baseline.reference_slots if task == "placement" else baseline.target_slots
            target = np.asarray([ACTIVE_CDPR_CATALOGS[i] for i in baseline.target_catalog_ids[idx]])
            receptacle = np.asarray([i.removeprefix("put_into_") if task == "placement" else "" for i in names])
            goal = baseline.reset_object_xyz[idx, slots[idx]]
            start = baseline.reset_ee_xyz[idx]
            delta = goal[:, :2] - start[:, :2]
            bearing, pairs = strata(delta, target, receptacle)
            pair = np.isin(pairs, PAIRS)
            is_ne = bearing == "NE"
            folds = np.select([is_ne & pair, is_ne, pair], [FOLDS[3], FOLDS[1], FOLDS[2]],
                              default=FOLDS[0])
            groups["instruction"].extend(names)
            groups["fold"].extend(folds)
            groups["bearing"].extend(bearing)
            groups["pair"].extend(pairs)
            poses = np.column_stack((baseline.reset_object_xyz[idx].reshape(len(idx), -1), start))
            meta["uid"].extend(hashlib.sha256(np.rint(p * 1000).astype("<i8").tobytes()).hexdigest() for p in poses)
            meta["skill"].extend([task] * len(idx))
            meta["delta_xy"].extend(delta)
            meta["target"].extend(target)
            meta["receptacle"].extend(receptacle)

        labels = {key: np.asarray(values) for key, values in groups.items()}
        outcomes = {arm: np.concatenate(parts) for arm, parts in outcomes.items()}
        feats = {arm: {name: np.concatenate(blocks) for name, blocks in values.items()}
                 for arm, values in scope.items()}
        for arm in transfer:
            for feature, values in feats[arm].items():
                transfer[arm][feature].append(values)
        contrasts = {arm: ("recipient", arm) for arm in ARMS}
        contrasts["donor_full_vs_donor_residual"] = ("donor_residual", "donor_full")
        for instruction in instructions:
            skill = labels["instruction"] == instruction
            for arm in ARMS:
                expected = notes["arms"][arm]["instruction"][instruction]
                for kind, member in (("recipient", "recipient"), ("arm", arm)):
                    if not np.isclose(outcomes[member][skill].mean(), expected[f"{kind}_success"]):
                        raise ValueError(f"Raw success does not reproduce paired.json: {instruction}/{arm}")
            for arm, (a, b) in contrasts.items():
                scores = np.stack((outcomes[a], outcomes[b]), axis=-1).astype(float)
                for axis in ("fold", "bearing", "pair"):
                    for category in np.unique(labels[axis][skill]):
                        if not category:
                            continue
                        mask = skill & (labels[axis] == category)
                        stats = paired_stats(scores[mask], group, setup["seed"])
                        scene_rows.append({**table_row("Confirmation", instruction, arm, stats),
                                           "stratum_type": axis, "stratum": category})
            for arm in ("donor_full", "donor_common_input"):
                for feature in FEATURES:
                    cka_rows.append({"instruction": instruction, "comparison": arm, "feature": feature,
                                     "reset_groups": int(skill.sum()),
                                     "cka": linear_cka(feats["recipient"][feature][skill].astype(float),
                                                       feats[arm][feature][skill].astype(float))})
    integrity = [{"task": task, "arm": arm, "diverged_worlds": n,
                  "first_chunk_max_error": errors.get((task, arm))}
                 for (task, arm), n in divergence.items()]
    return scene_rows, cka_rows, integrity, meta, transfer


def transfer_rows(meta, transfer):
    data = {key: np.asarray(rows) for key, rows in meta.items()}
    split = split_scenes(data, ["NE"], list(PAIRS), 1729)
    delta = data["delta_xy"].astype(float)
    lengths = np.linalg.norm(delta, axis=1, keepdims=True)
    if not (lengths > 0).all():
        raise ValueError("Zero XY distance in recorded resets")
    unit = delta / lengths
    results = []
    for checkpoint, parts in transfer.items():
        features = {key: np.concatenate(value).astype(float) for key, value in parts.items()}
        for source in TASKS:
            train = (data["skill"] == source) & (split == "probe_train")
            for target in TASKS:
                for fold in sorted(set(split) - {"probe_train"}):
                    test = (data["skill"] == target) & (split == fold)
                    if test.sum() < 12:
                        continue
                    baseline = cosine(readout(features["proprio"][train], unit[train],
                                              features["proprio"][test], 1.0), unit[test])
                    vision = cosine(readout(features["vision_pool"][train], unit[train],
                                            features["vision_pool"][test], 1.0), unit[test])
                    constant = cosine(np.broadcast_to(unit[train].mean(axis=0), unit[test].shape), unit[test])
                    for label, x in features.items():
                        scores = cosine(readout(x[train], unit[train], x[test], 1.0), unit[test])
                        lo, hi = paired_ci(data["uid"][test], scores, baseline, 1729)
                        vision_lo, vision_hi = paired_ci(data["uid"][test], scores, vision, 1729)
                        results.append({"checkpoint": checkpoint, "train_skill": source, "test_skill": target,
                                        "fold": fold, "feature": label, "train_groups": int(train.sum()),
                                        "test_groups": int(test.sum()), "mean_cosine": scores.mean(),
                                        "constant_mean_cosine": constant.mean(),
                                        "delta_vs_proprio": (scores - baseline).mean(),
                                        "delta_ci95_low": lo, "delta_ci95_high": hi,
                                        "delta_vs_vision_pool": (scores - vision).mean(),
                                        "vision_delta_ci95_low": vision_lo, "vision_delta_ci95_high": vision_hi})
    return results


def plots(output, scene, cka, transfer):
    names = ("Other scenes", "NE only", "Target pair only", "NE + target pair")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharex=True, sharey=True)
    for ax, (instruction, label) in zip(axes, TITLES.items()):
        rows = [r for r in scene if r["instruction"] == instruction and r["stratum_type"] == "fold"]
        for arm, y_shift in (("recipient_repeat", -.24), ("donor_full", 0), ("donor_residual", .24)):
            for y, fold in enumerate(FOLDS):
                for row in rows:
                    if row["stratum"] != fold or row["arm"] != arm:
                        continue
                    color = {"recipient_repeat": "#6d7781", "donor_full": "#246fb1", "donor_residual": "#d55e00"}[arm]
                    ax.hlines(y + y_shift, row["ci95_low_pp"], row["ci95_high_pp"], color=color)
                    ax.scatter(row["delta_pp"], y + y_shift, color=color, s=21)
                    if arm == "donor_full":
                        ax.text(.98, y, f'n={row["reset_groups"]}', transform=ax.get_yaxis_transform(),
                                ha="right", va="center", color="#5b6774", fontsize=8)
        ax.axvline(0, color="#a3a9b0", lw=1)
        ax.grid(axis="x", alpha=.25)
        ax.set_title(label)
    low, high = axes[0].get_xlim()
    axes[0].set_xlim(low, high + 0.12 * (high - low))
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([0], [0], marker="o", color=c, label=label)
               for c, label in (("#6d7781", "Repeat B"), ("#246fb1", "Full SIL"),
                                ("#d55e00", "Residual only"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .065))
    fig.suptitle("CDPR · effects across evaluation scene strata", fontweight="bold")
    fig.supxlabel("Success difference vs. B (percentage points), 95% cluster intervals", y=.03)
    fig.tight_layout(rect=(0, .15, 1, .90))
    fig.savefig(output / "scene_strata.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    layer_names = ("Proprio", "Vision pool", "Prior", "Hidden 1", "Hidden 2", "Residual")
    for instruction, label in TITLES.items():
        for comp, sty in (("donor_full", "-"), ("donor_common_input", "--")):
            values = {r["feature"]: r["cka"] for r in cka
                      if r["instruction"] == instruction and r["comparison"] == comp}
            color = {"Move to": "#246fb1", "Bowl": "#16877b", "Plate": "#d55e00"}[label]
            axes[0].plot(layer_names, [values[key] for key in FEATURES], sty, color=color,
                         marker="o", markersize=3, label=label if sty == "-" else None)
    axes[0].set(title="B → C · first-decision representation", ylabel="Linear CKA", ylim=(0, 1.04))
    axes[0].legend(title="Solid: recorded inputs\nDashed: same B inputs", fontsize=8)

    for ax, (fold, title) in zip(axes[1:], (("ordinary_test", "Ordinary test scenes"),
                                          ("bearing_holdout", "NE sector excluded from probe fit"))):
        for source, target, color in (("move_to", "placement", "#246fb1"),
                                      ("placement", "move_to", "#d55e00")):
            for arm, sty in (("recipient", "--"), ("donor_full", "-")):
                values = {r["feature"]: r["mean_cosine"] for r in transfer if r["checkpoint"] == arm
                          and r["train_skill"] == source and r["test_skill"] == target and r["fold"] == fold}
                ax.plot(layer_names, [values[key] for key in FEATURES], sty, color=color, marker="o",
                        markersize=3, label=f"{source} → {target}" if sty == "-" else None)
        ax.axhline(0, color="#a3a9b0", lw=1)
        ax.set(title=title, ylabel="Mean cosine" if ax == axes[1] else "", ylim=(-1.02, .75))
    axes[1].legend(title="Linear XY readout\nDashed: B · solid: C", fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=22)
        ax.grid(axis="y", alpha=.24)
    fig.tight_layout()
    fig.savefig(output / "residual_representations.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "swaps", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene, cka, integrity, meta, transfer = read_records(args.run, args.swaps)
    readouts = transfer_rows(meta, transfer)
    for name, rows in (("scene_strata", scene), ("residual_cka", cka), ("direction_readouts", readouts),
                       ("integrity", integrity)):
        save_csv(args.output / f"{name}.csv", rows)
    plots(args.output, scene, cka, readouts)
    print(args.output)


if __name__ == "__main__":
    main()
