#!/usr/bin/env python3
"""Fit the residual actor to the self-imitation demonstrations.

Part 4. Reads a ``demonstrations.npz`` written by ``sil_record.py --mode
dataset`` and trains the residual MLP to reproduce the recorded actions,
starting from the RL checkpoint that generated them.


What this trains, and what it cannot
------------------------------------

The policy is a frozen SmolVLA prior, a trainable residual MLP, and LoRA on
the action expert. This trains the **residual only**. The LoRA lives on the
SmolVLA runtime and updating it needs gradients through the vision tower,
which needs the 256x256 camera frames -- and the dataset stores the pooled
512-wide vision feature, not the images (~37 MB per round against ~5.0 GB).

That is not a silent omission. The source checkpoint's ``vla_lora`` tensors
are copied verbatim into the output, because the trainer that writes these
checkpoints warns that dropping them makes a resumed phase restart from a
zero adapter and throw away every step of VLA adaptation. The saved file is a
complete policy: new residual, original LoRA.


The algebra, which decides the loss
-----------------------------------

``ResidualChunkActor`` computes::

    features = cat([state, prior.flatten()])          # 518 + 8*5 = 558
    residual = tanh(net(features))                    # bounded to +-1
    action   = tanh(prior + residual_scale * residual)

Two consequences that are easy to get wrong.

The net sees the PRIOR, not just the state. The prior is a fresh noise draw
on every forward, so a residual that saw only the state would face a moving
target for a fixed state and could at best learn the mean correction. It sees
the draw, so the supervised problem is well posed as stated.

The reachable action set is therefore ``[tanh(p - s), tanh(p + s)]`` for
prior ``p`` and ``residual_scale`` ``s`` -- a bounded interval, not the whole
range. A target outside it cannot be fitted by any weights, and the loss will
sit at a floor that looks like underfitting and is not. The reachability
fraction is measured before the first step and reported, because "the loss
will not go down" is otherwise a week of tuning an optimizer that was never
the problem.

The actor emits ``chunk_size`` (8) actions and the plant executes
``replan_every`` (4) of them, so slots 4-7 have no recorded target. They are
left unsupervised rather than regularized to something invented, because
``deterministic_action_chunks_tensor`` slices ``[:, :count]`` with count 4 and
never reads them at inference.


The two numbers to read before believing any of it
--------------------------------------------------

The loss of the UNTRAINED actor on this dataset. The demonstrations came from
this same checkpoint, so before smoothing it would reproduce them exactly and
the initial loss measures only what smoothing changed. It is the null: if it
is already near zero there is nothing for SFT to learn, and any improvement
reported against it is noise.

The reachable fraction. See above.


Splitting
---------

Held out by EPISODE, never by row. Decisions from one episode share an
observation history and are near-duplicates of each other; a random row split
puts step 3 of an episode in train and step 4 in validation, and the
validation loss then measures memorization rather than generalization.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.xy_approach_probe import _load_checkpoint  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (  # noqa: E402
    ACTIVE_INSTRUCTION_TYPES,
)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        dataset = {key: data[key] for key in data.files}
    required = {"state", "prior", "action", "action_mask", "episode_uid"}
    missing = sorted(required.difference(dataset))
    if missing:
        raise ValueError(f"{path} is missing {missing}.")
    return dataset


def _filter_instructions(
    dataset: Mapping[str, np.ndarray], names: Sequence[str]
) -> np.ndarray:
    """Row mask for a whitelist of instruction names."""

    if not names:
        return np.ones((dataset["state"].shape[0],), dtype=bool)
    wanted = []
    for name in names:
        if name not in ACTIVE_INSTRUCTION_TYPES:
            raise SystemExit(
                f"Unknown instruction {name!r}. Known: "
                f"{list(ACTIVE_INSTRUCTION_TYPES)}"
            )
        wanted.append(ACTIVE_INSTRUCTION_TYPES.index(name))
    return np.isin(dataset["instruction_id"], wanted)


def _episode_split(
    episode_uid: np.ndarray, *, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole episodes.

    Splitting rows would put consecutive decisions of one episode on both
    sides. They share an observation history and differ by one env step, so
    the validation loss would be measuring memorization.
    """

    episodes = np.unique(episode_uid)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    held = max(1, int(round(len(shuffled) * float(val_fraction))))
    validation = set(shuffled[:held].tolist())
    is_val = np.array(
        [uid in validation for uid in episode_uid.tolist()], dtype=bool
    )
    return ~is_val, is_val


def _reachability(
    prior: np.ndarray,
    action: np.ndarray,
    mask: np.ndarray,
    *,
    residual_scale: float,
) -> dict[str, Any]:
    """What fraction of targets the residual can express at all.

    ``action = tanh(prior + scale * u)`` with ``u`` in [-1, 1], so the
    reachable set is the closed interval ``[tanh(p - s), tanh(p + s)]``.
    Computed directly rather than through ``atanh``, which diverges for the
    saturated targets a tanh policy produces constantly.
    """

    slots = action.shape[1]
    p = prior[:, :slots]
    low = np.tanh(p - float(residual_scale))
    high = np.tanh(p + float(residual_scale))
    reachable = (action >= low - 1e-6) & (action <= high + 1e-6)
    live = mask[..., None] & np.ones_like(reachable, dtype=bool)
    if not live.any():
        return {"reachable_fraction": None, "supervised_values": 0}
    shortfall = np.maximum(
        np.maximum(low - action, action - high), 0.0
    )[live]
    return {
        "reachable_fraction": round(float(reachable[live].mean()), 5),
        "supervised_values": int(live.sum()),
        "mean_shortfall": round(float(shortfall.mean()), 6),
        "max_shortfall": round(float(shortfall.max()), 6),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def _build_actor(payload: Mapping[str, Any], device: Any) -> Any:
    from rl_vla_bootstrapping.policy.octo_finetune_cdpr import (
        ResidualChunkActor,
    )

    args = dict(payload["args"])
    actor = ResidualChunkActor(
        state_dim=int(payload["state_dim"]),
        chunk_size=int(payload["chunk_size"]),
        action_dim=int(payload["action_dim"]),
        hidden_dim=int(payload.get("hidden_dim", args.get("hidden_dim", 1024))),
        residual_scale=float(
            payload.get("residual_scale", args.get("residual_scale", 1.0))
        ),
    ).to(device)
    # The saved state dict is SmolVLAGRPOPolicy's, whose actor is nested under
    # "actor." and which also carries log_std. Strict loading on the bare
    # ResidualChunkActor would fail on both, so the prefix is stripped and the
    # remainder must match exactly -- a silently partial load would leave a
    # randomly initialised residual wearing the checkpoint's name.
    state = payload["policy"]
    nested = {
        key[len("actor.") :]: value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    actor.load_state_dict(nested if nested else state, strict=True)
    return actor


def _evaluate(
    actor: Any,
    torch: Any,
    *,
    state: Any,
    prior: Any,
    action: Any,
    mask: Any,
    batch_size: int,
) -> dict[str, float]:
    slots = int(action.shape[1])
    total_sq = 0.0
    total_abs = 0.0
    total = 0.0
    actor.eval()
    with torch.no_grad():
        for start in range(0, int(state.shape[0]), batch_size):
            stop = start + batch_size
            out = actor(state[start:stop], prior[start:stop])[:, :slots]
            weight = mask[start:stop].unsqueeze(-1).float()
            error = (out - action[start:stop]) * weight
            total_sq += float((error**2).sum().item())
            total_abs += float(error.abs().sum().item())
            total += float(weight.sum().item()) * float(action.shape[-1])
    actor.train()
    if total <= 0.0:
        return {"mse": float("nan"), "mae": float("nan")}
    return {
        "mse": round(total_sq / total, 8),
        "mae": round(total_abs / total, 8),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "The RL adapter the demonstrations came from. Training starts "
            "from its residual, and its vla_lora is copied into the output."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--instructions",
        nargs="*",
        default=[],
        help=(
            "Train on these instructions only. Use it to drop a slice whose "
            "source success rate makes it mostly luck -- the rate is in the "
            "dataset.json beside the npz."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=("mse", "l1"),
        default="mse",
        help=(
            "Action-space loss. Deliberately not computed in pre-tanh space: "
            "the targets are tanh outputs and sit arbitrarily close to +-1, "
            "where atanh diverges."
        ),
    )
    args = parser.parse_args(argv)

    import torch

    device = torch.device(str(args.device))
    torch.manual_seed(int(args.seed))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(args.dataset.expanduser().resolve())
    rows = _filter_instructions(dataset, list(args.instructions or []))
    if not rows.any():
        raise SystemExit("The instruction filter selected no rows.")
    dataset = {key: value[rows] for key, value in dataset.items()}

    payload = _load_checkpoint(args.checkpoint.expanduser().resolve())
    residual_scale = float(
        payload.get(
            "residual_scale", dict(payload["args"]).get("residual_scale", 1.0)
        )
    )
    state_dim = int(payload["state_dim"])
    if int(dataset["state"].shape[-1]) != state_dim:
        raise SystemExit(
            f"The dataset carries {dataset['state'].shape[-1]}-wide states "
            f"and the checkpoint expects {state_dim}. These were recorded "
            "under different residual_vision_dim / state_dim settings and "
            "cannot be mixed."
        )

    reach = _reachability(
        dataset["prior"],
        dataset["action"],
        dataset["action_mask"],
        residual_scale=residual_scale,
    )

    train_rows, val_rows = _episode_split(
        dataset["episode_uid"],
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )

    def tensors(mask: np.ndarray) -> tuple[Any, Any, Any, Any]:
        return (
            torch.as_tensor(dataset["state"][mask], dtype=torch.float32,
                            device=device),
            torch.as_tensor(dataset["prior"][mask], dtype=torch.float32,
                            device=device),
            torch.as_tensor(dataset["action"][mask], dtype=torch.float32,
                            device=device),
            torch.as_tensor(dataset["action_mask"][mask], device=device),
        )

    tr_state, tr_prior, tr_action, tr_mask = tensors(train_rows)
    va_state, va_prior, va_action, va_mask = tensors(val_rows)
    slots = int(tr_action.shape[1])

    actor = _build_actor(payload, device)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    # The null. These demonstrations came from this checkpoint, so before
    # smoothing it reproduced them exactly; this measures only what smoothing
    # changed. If it is already near zero there is nothing to learn here.
    baseline_train = _evaluate(
        actor, torch, state=tr_state, prior=tr_prior, action=tr_action,
        mask=tr_mask, batch_size=int(args.batch_size),
    )
    baseline_val = _evaluate(
        actor, torch, state=va_state, prior=va_prior, action=va_action,
        mask=va_mask, batch_size=int(args.batch_size),
    )
    print(
        f"[sft] rows train={int(train_rows.sum())} val={int(val_rows.sum())} "
        f"episodes={len(np.unique(dataset['episode_uid']))}",
        flush=True,
    )
    print(
        f"[sft] reachable={reach['reachable_fraction']} "
        f"(max shortfall {reach.get('max_shortfall')})",
        flush=True,
    )
    print(
        f"[sft] untrained baseline: train mse={baseline_train['mse']} "
        f"val mse={baseline_val['mse']}",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    best = float("inf")
    best_epoch = -1
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    started = time.perf_counter()

    for epoch in range(int(args.epochs)):
        order = torch.randperm(
            int(tr_state.shape[0]), generator=generator
        ).to(device)
        running = 0.0
        batches = 0
        for start in range(0, int(order.numel()), int(args.batch_size)):
            index = order[start : start + int(args.batch_size)]
            out = actor(
                tr_state.index_select(0, index),
                tr_prior.index_select(0, index),
            )[:, :slots]
            target = tr_action.index_select(0, index)
            weight = tr_mask.index_select(0, index).unsqueeze(-1).float()
            residual = (out - target) * weight
            denominator = weight.sum().clamp_min(1.0) * float(target.shape[-1])
            if str(args.loss) == "l1":
                loss = residual.abs().sum() / denominator
            else:
                loss = (residual**2).sum() / denominator
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1

        train_metrics = _evaluate(
            actor, torch, state=tr_state, prior=tr_prior, action=tr_action,
            mask=tr_mask, batch_size=int(args.batch_size),
        )
        val_metrics = _evaluate(
            actor, torch, state=va_state, prior=va_prior, action=va_action,
            mask=va_mask, batch_size=int(args.batch_size),
        )
        history.append(
            {
                "epoch": epoch,
                "loss": round(running / max(batches, 1), 8),
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
            }
        )
        print(
            f"[sft] epoch {epoch:3d} loss={running / max(batches, 1):.6f} "
            f"train_mse={train_metrics['mse']:.6f} "
            f"val_mse={val_metrics['mse']:.6f}",
            flush=True,
        )
        if val_metrics["mse"] < best:
            best = val_metrics["mse"]
            best_epoch = epoch
            # Same payload shape the RL trainer writes, so the result loads in
            # xy_approach_probe and sil_record without a special case. vla_lora
            # is carried over untouched: this run never saw an image and has no
            # business changing the action expert, and dropping it would leave
            # a checkpoint that silently restarts from a zero adapter.
            saved = dict(payload)
            saved["policy"] = {
                f"actor.{key}": value.detach().cpu()
                for key, value in actor.state_dict().items()
            }
            if "log_std" in payload["policy"]:
                saved["policy"]["log_std"] = payload["policy"]["log_std"]
            saved["optimizer"] = optimizer.state_dict()
            saved["sil_sft"] = {
                "dataset": str(args.dataset),
                "source_checkpoint": str(args.checkpoint),
                "epoch": epoch,
                "val_mse": val_metrics["mse"],
                "trained": "residual_only",
            }
            torch.save(saved, output / "sil_sft_adapter.pt")

    report = {
        "dataset": str(args.dataset),
        "source_checkpoint": str(args.checkpoint),
        "instructions": list(args.instructions or []),
        "residual_scale": residual_scale,
        "state_dim": state_dim,
        "chunk_slots_emitted": int(payload["chunk_size"]),
        "chunk_slots_supervised": slots,
        "rows_train": int(train_rows.sum()),
        "rows_val": int(val_rows.sum()),
        "episodes": int(len(np.unique(dataset["episode_uid"]))),
        "reachability": reach,
        "baseline_untrained": {
            "train": baseline_train,
            "val": baseline_val,
        },
        "best_epoch": best_epoch,
        "best_val_mse": None if best == float("inf") else best,
        "history": history,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "trained_parameters": "residual actor only; vla_lora copied verbatim",
    }
    (output / "sft_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[sft] best val mse {best:.6f} at epoch {best_epoch}; wrote "
        f"{output / 'sil_sft_adapter.pt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
