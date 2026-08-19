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


# --------------------------------------------------------------------------
# Frames: joining pictures back to demonstration rows
# --------------------------------------------------------------------------


def frame_join_key(name: str) -> str:
    """Reduce either side's name to the identity the two share.

    The two writers spell the same episode source differently, and the first
    version of this join compared them raw and matched nothing at all --
    0 of 33102 rows, after the whole harvest had already been paid for.

    ``sil_record --mode replay`` writes ``replay_<X>.npz`` and
    ``frames_<X>.npz`` side by side, so ``<X>`` is the shared identity.
    ``--mode dataset`` then keys episodes by ``<parent>/<stem>`` of the replay
    -- "Directory AND stem. Neither alone is unique across both layouts this
    tool produces" -- so its half arrives as ``replay/replay_<X>``.

    Taking the basename and stripping either prefix leaves ``<X>`` on both
    sides. ``<X>`` is itself ``<rung dir>_<record stem>``, which is what makes
    it unique: a harvest writes record_00..NN per rung so stems repeat across
    rungs, and replays of a whole harvest land in one directory so parents
    repeat there.
    """

    tail = str(name).rsplit("/", 1)[-1]
    for prefix in ("frames_", "replay_"):
        if tail.startswith(prefix):
            return tail[len(prefix) :]
    return tail


def load_frame_index(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Open every frames_<X>.npz and key it by the identity it shares with the
    replay it was written beside. See frame_join_key."""

    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        name = Path(path).stem
        if not name.startswith("frames_"):
            raise SystemExit(
                f"{path} is not a frames_<stem>.npz written by sil_record."
            )
        stem = frame_join_key(name)
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        column = {
            int(world): position
            for position, world in enumerate(payload["world_index"])
        }
        index[stem] = {
            "path": str(path),
            "overview": payload["overview"],
            "wrist": payload["wrist"],
            "world_column": column,
            "decisions": int(payload["overview"].shape[0]),
        }
    return index


def resolve_frame_rows(
    episode_uid: np.ndarray,
    decision_index: np.ndarray,
    frames: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Map each demonstration row to (stem, decision, world column).

    Returns a mask of the rows that FOUND a frame and the lookups for them.
    Rows are dropped rather than filled: a missing frame means the replay that
    produced the row kept no pictures for that world (``--frame-worlds`` capped
    it, or the world failed the replay), and inventing one would train the
    vision path on a picture from a different episode.
    """

    keep = np.zeros(episode_uid.shape[0], dtype=bool)
    lookups: list[tuple[str, int, int]] = []
    for row, (uid, decision) in enumerate(zip(episode_uid, decision_index)):
        raw_stem, _, tail = str(uid).rpartition("/")
        # The NORMALISED key travels in the lookup, because frames_for_rows
        # indexes the frame index with it. Carrying the raw uid prefix here
        # would resolve the row and then raise a KeyError on the gather.
        stem = frame_join_key(raw_stem)
        entry = frames.get(stem)
        if entry is None or not tail.startswith("r"):
            continue
        world_part = tail.split("w", 1)
        if len(world_part) != 2:
            continue
        column = entry["world_column"].get(int(world_part[1]))
        if column is None or int(decision) >= int(entry["decisions"]):
            continue
        keep[row] = True
        lookups.append((stem, int(decision), int(column)))
    return keep, lookups


def frames_for_rows(
    lookups: Sequence[tuple[str, int, int]],
    frames: Mapping[str, Mapping[str, Any]],
    rows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Gather (overview, wrist) uint8 batches for a set of resolved rows."""

    overview = np.stack(
        [
            frames[lookups[row][0]]["overview"][lookups[row][1], lookups[row][2]]
            for row in rows
        ]
    )
    wrist = np.stack(
        [
            frames[lookups[row][0]]["wrist"][lookups[row][1], lookups[row][2]]
            for row in rows
        ]
    )
    return overview, wrist


# --------------------------------------------------------------------------
# The checkpoint that goes back into RL
# --------------------------------------------------------------------------


def build_resume_payload(
    source: Mapping[str, Any],
    *,
    policy_state: Mapping[str, Any],
    lora_state: Mapping[str, Any] | None,
    note: Mapping[str, Any],
) -> dict[str, Any]:
    """The SFT result, shaped so ``--resume-checkpoint`` accepts it.

    Neither existing loader does the right thing on its own. ``trainer.load``
    restores the optimizer, and an SFT AdamW carries moments taken from a
    different loss at a different scale. ``load_weights_only`` throws away
    ``extra_state`` -- which is where the approach-curriculum caps live, so the
    cap would drop to the first rung and undo the iteration that earned it.

    So: the source payload verbatim, with the weights replaced and BOTH
    optimizer states removed. ``load`` tolerates their absence and rebuilds
    them fresh, and the curriculum, global step, simulator metadata and args all
    survive untouched.
    """

    # The written policy must occupy exactly the key space of the one it
    # replaces. Two different modules are handed to this function -- a bare
    # ResidualChunkActor (keys "net.net.*") and the trainer's SmolVLAGRPOPolicy
    # (keys "log_std", "actor.net.net.*") -- and the caller is responsible for
    # presenting either in the checkpoint's spelling. Getting it wrong is
    # invisible here and surfaces two tools later as a load_state_dict error
    # listing forty keys, after a whole harvest and an SFT have been paid for.
    expected = set(dict(source.get("policy") or {}))
    written = set(policy_state)
    if expected and written != expected:
        raise ValueError(
            "The SFT policy state does not match the checkpoint's key space.\n"
            f"  missing:    {sorted(expected - written)[:6]}\n"
            f"  unexpected: {sorted(written - expected)[:6]}"
        )
    payload = dict(source)
    payload["policy"] = {
        key: value.detach().cpu() if hasattr(value, "detach") else value
        for key, value in policy_state.items()
    }
    if lora_state is not None:
        payload["vla_lora"] = {
            key: value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in lora_state.items()
        }
    payload.pop("optimizer", None)
    payload.pop("vla_lora_optimizer", None)
    payload["sil_sft"] = dict(note)
    return payload


# --------------------------------------------------------------------------
# Stage (b): LoRA from frames
# --------------------------------------------------------------------------


def build_runtime_and_trainer(
    payload: Mapping[str, Any],
    *,
    checkpoint: Path,
    device: str,
    train_vision_lora: bool = False,
) -> tuple[Any, Any, Any]:
    """The RL run's own SmolVLA runtime, LoRA and residual, rebuilt verbatim.

    Everything comes from the checkpoint's saved ``args``, and the LoRA is
    attached by ``SmolVLAGRPOTrainer.attach_vla_lora`` rather than by a copy of
    it here. A second LoRA attach would be a second place for the vision leaf
    names to be wrong, and those fail by matching nothing rather than by
    raising anywhere near the mistake.
    """

    from argparse import Namespace

    from rl_vla_bootstrapping.policy.smolvla_cdpr import load_smolvla_runtime
    from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
        SmolVLAGRPOTrainer,
    )

    values = dict(payload["args"])
    values.update(
        {
            "device": str(device),
            "distributed": False,
            "smolvla_compile_model": False,
            "resume_checkpoint": None,
        }
    )
    if train_vision_lora:
        # The design turns the vision tower on at the FIRST SFT, not during RL:
        # in RL the adapter sees 128 decision-0 records per update through a
        # PPO objective through a ten-step flow expert, and here it sees every
        # sampled decision under an MSE. But attach_vla_lora reads the flag off
        # the checkpoint's args, and those come from the RL run that wrote it --
        # where it is deliberately off. So it is set here, with the leaf names
        # defaulted, because the tower uses out_proj and fc1/fc2 rather than the
        # expert's o_proj and gate/up/down and reusing the expert list matches
        # almost nothing.
        values["train_vla_lora"] = True
        values["train_vla_vision_lora"] = True
        values.setdefault("lora_vision_name_contains", "vision")
        values.setdefault(
            "lora_vision_leaf_names", "q_proj,k_proj,v_proj,out_proj,fc1,fc2"
        )
    args = Namespace(**values)
    runtime = load_smolvla_runtime(
        checkpoint=str(args.base_checkpoint),
        device=str(device),
        mixed_precision=str(args.mixed_precision),
        image_size=int(args.image_size),
        state_dim=int(args.state_dim),
        image_feature_keys=(
            None
            if getattr(args, "image_feature_keys", None) is None
            else tuple(args.image_feature_keys)
        ),
        include_wrist=bool(args.include_wrist),
        include_aux_camera=bool(args.include_aux_camera),
        mask_empty_aux_camera=bool(
            getattr(args, "mask_empty_aux_camera", False)
        ),
        chunk_size=int(args.chunk_size),
        action_dim=int(args.action_dim),
        action_indices=(
            None
            if getattr(args, "smolvla_action_indices", None) is None
            else tuple(int(v) for v in args.smolvla_action_indices)
        ),
        action_normalization=str(args.smolvla_action_normalization),
        model_image_size=(
            None
            if int(getattr(args, "smolvla_model_image_size", 0)) <= 0
            else int(args.smolvla_model_image_size)
        ),
        compile_model=False,
        compile_mode=str(args.smolvla_compile_mode),
        vision_pooling=str(
            getattr(args, "residual_vision_pooling", "flat_random")
        ),
    )
    trainer = SmolVLAGRPOTrainer(
        args=args,
        state_dim=int(payload["state_dim"]),
        action_dim=int(payload["action_dim"]),
        chunk_size=int(payload["chunk_size"]),
        run_dir=Path(checkpoint).parent,
        device=device,
        distributed=None,
    )
    info = trainer.attach_vla_lora(runtime)
    print(
        f"[sft] LoRA attached: expert {info['vla_lora/modules']:.0f} modules, "
        f"vision {info['vla_lora/vision_modules']:.0f} modules, "
        f"{info['vla_lora/trainable_params']:.0f} trainable params",
        flush=True,
    )
    if info["vla_lora/vision_modules"] <= 0:
        print(
            "[sft] NOTE: the vision tower is NOT adapted in this checkpoint's "
            "args (train_vla_vision_lora is off). Phase 4 turns it on at the "
            "first SFT; set it in the config the harvest checkpoint came from.",
            flush=True,
        )
    trainer.load_weights_only(Path(checkpoint))
    return runtime, trainer, args


def recompute_state_and_prior(
    runtime: Any,
    torch: Any,
    *,
    overview: Any,
    wrist: Any,
    proprio: Any,
    instructions: Sequence[str],
    vision_dim: int,
    enable_grad: bool,
) -> tuple[Any, Any]:
    """One grad-carrying SmolVLA forward: (prior, residual state).

    The recorded ``state`` cannot be reused once the vision LoRA moves, because
    its vision block was pooled from the OLD adapter's connector tokens -- at
    deployment the residual would then see a different input than it trained
    on, and the gap would grow with every iteration. Recomputing costs nothing
    extra: the same forward that produces the grad-carrying prior also returns
    the pooled feature.

    The prior carries gradient; the vision block does not and cannot -- pooling
    runs under an unconditional ``no_grad`` behind a fixed random projection.
    So the vision LoRA reaches the action only through the prior.
    """

    prior, vision = runtime.sample_cdpr_chunks_and_vision_from_tensors(
        primary_images=overview,
        wrist_images=wrist,
        states=proprio,
        instructions=list(instructions),
        vision_dim=int(vision_dim),
        microbatch_size=0,
        enable_grad=bool(enable_grad),
    )
    state = torch.cat(
        [proprio, vision.to(dtype=proprio.dtype).detach()], dim=-1
    )
    return prior, state


def check_recomputed_vision(
    recomputed_state: Any,
    recorded_state: Any,
    *,
    vision_dim: int,
    torch: Any,
    control_state: Any | None = None,
) -> dict[str, float]:
    """M5: the pipeline check that costs one forward and catches everything.

    Only the VISION block is compared. The proprio block is copied through, so
    it is trivially equal, and the PRIOR cannot be compared at all -- LeRobot's
    ``sample_noise`` is a bare ``torch.normal`` with no generator, so the prior
    is a fresh draw on every forward (phase-3 report §2).

    A mismatch here means the frames are not the ones the policy was given: the
    tap fired at the wrong point, the cameras are swapped, or the pooling mode
    differs from the run's. All three train the vision path on the wrong
    pictures while every loss curve looks normal.

    ``control_state`` is the SAME rows recomputed a second time, at a different
    batch size, from the same stored frames. Without it the headline number is
    uninterpretable: this recompute differs from the rollout in two ways that
    are expected and harmless -- the frames went through a uint8 round trip,
    and the batch is a handful of rows against the rollout's hundreds, which
    selects different bf16 kernels -- and in one way that is fatal, the frames
    being the wrong pictures. The control shares the first two and not the
    third, so it is the instrument's own noise floor and the headline only
    means something as a multiple of it.

    Phase 3 spent a week on two nulls that each certified a policy knowing
    nothing. A lone difference with no control is the same mistake.
    """

    if int(vision_dim) <= 0:
        return {"vision_max_abs_diff": 0.0, "vision_dim": 0.0}
    a = recomputed_state[:, -int(vision_dim) :]
    b = recorded_state[:, -int(vision_dim) :]
    diff = (a - b).abs()
    scale = b.abs().mean().clamp_min(1.0e-6)
    out = {
        "vision_max_abs_diff": float(diff.max().item()),
        "vision_mean_abs_diff": float(diff.mean().item()),
        "vision_relative_mean_abs_diff": float((diff.mean() / scale).item()),
        "vision_dim": float(vision_dim),
    }
    if control_state is None:
        out["verdict"] = "no control -- the number above is uninterpretable"
        return out
    control = (control_state[:, -int(vision_dim) :] - a).abs()
    floor = float(control.mean().item())
    out["control_mean_abs_diff"] = floor
    out["control_max_abs_diff"] = float(control.max().item())
    # A headline within a few times the floor is the round trip and the
    # kernels; an order of magnitude above it is the pictures.
    ratio = out["vision_mean_abs_diff"] / max(floor, 1.0e-12)
    out["headline_over_control"] = round(ratio, 3)
    out["verdict"] = (
        "consistent with the uint8 round trip and batch-size numerics"
        if ratio <= 5.0
        else "NOT explained by the round trip -- these are probably not the "
        "frames the policy was given"
    )
    return out


def train_lora_stage(
    *,
    torch: Any,
    runtime: Any,
    trainer: Any,
    actor: Any,
    dataset: Mapping[str, np.ndarray],
    frames: Mapping[str, Mapping[str, Any]],
    lookups: Sequence[tuple[str, int, int]],
    rows_train: np.ndarray,
    rows_val: np.ndarray,
    vision_dim: int,
    device: Any,
    epochs: int,
    lr: float,
    kl_coef: float,
    microbatch: int,
    row_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Fit LoRA + residual through a grad-carrying SmolVLA forward.

    The gradient path is the RL update's, with MSE where PPO was: images ->
    SmolVLA (expert LoRA, and the vision tower when adapted) -> prior ->
    residual (which takes the prior as an input) -> action. The residual's own
    vision channel is NOT on that path and never will be.
    """

    proprio_dim = int(dataset["state"].shape[-1]) - int(vision_dim)
    slots = int(dataset["action"].shape[1])
    lora_params = list(trainer.vla_lora_params)
    optimizer = torch.optim.AdamW(
        lora_params + list(actor.parameters()), lr=float(lr)
    )
    generator = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []

    def batch_loss(rows: Sequence[int], *, grad: bool) -> tuple[Any, Any, int]:
        overview_np, wrist_np = frames_for_rows(lookups, frames, rows)
        # uint8 -> float32 in [0, 1] and NCHW, which is what the backend hands
        # the runtime at rollout time.
        def images(array: np.ndarray) -> Any:
            tensor = torch.as_tensor(array, device=device).to(torch.float32)
            return (tensor / 255.0).permute(0, 3, 1, 2).contiguous()

        index = np.asarray(rows, dtype=np.int64)
        proprio = torch.as_tensor(
            dataset["state"][index, :proprio_dim], dtype=torch.float32,
            device=device,
        )
        prior_ref = torch.as_tensor(
            dataset["prior"][index], dtype=torch.float32, device=device
        )
        target = torch.as_tensor(
            dataset["action"][index], dtype=torch.float32, device=device
        )
        mask = torch.as_tensor(dataset["action_mask"][index], device=device)
        instructions = [str(t) for t in dataset["instruction_text"][index]]
        prior, state = recompute_state_and_prior(
            runtime, torch,
            overview=images(overview_np), wrist=images(wrist_np),
            proprio=proprio, instructions=instructions,
            vision_dim=int(vision_dim), enable_grad=grad,
        )
        out = actor(state, prior)[:, :slots]
        weight = mask.unsqueeze(-1).float()
        denominator = weight.sum().clamp_min(1.0) * float(target.shape[-1])
        mse = (((out - target) * weight) ** 2).sum() / denominator
        # Anchored on the RECORDED prior, which is this iteration's starting
        # point -- the same reference the RL update uses.
        kl = ((prior - prior_ref.reshape_as(prior)) ** 2).mean()
        return mse, kl, len(rows)

    def evaluate(rows: np.ndarray) -> dict[str, float]:
        picked = np.flatnonzero(rows)
        if picked.size == 0:
            return {"mse": float("nan"), "kl": float("nan")}
        total_mse = 0.0
        total_kl = 0.0
        counted = 0
        actor.eval()
        with torch.no_grad():
            for start in range(0, picked.size, int(microbatch)):
                chunk = picked[start : start + int(microbatch)]
                mse, kl, n = batch_loss(list(chunk), grad=False)
                total_mse += float(mse.item()) * n
                total_kl += float(kl.item()) * n
                counted += n
        actor.train()
        return {
            "mse": round(total_mse / max(counted, 1), 8),
            "kl": round(total_kl / max(counted, 1), 8),
        }

    baseline = evaluate(rows_val)
    print(f"[sft][lora] untrained baseline on frames: {baseline}", flush=True)

    train_rows = np.flatnonzero(rows_train)
    per_epoch = max(1, int(round(train_rows.size * float(row_fraction))))
    for epoch in range(int(epochs)):
        picked = generator.permutation(train_rows)[:per_epoch]
        running = 0.0
        batches = 0
        for start in range(0, picked.size, int(microbatch)):
            chunk = picked[start : start + int(microbatch)]
            mse, kl, _ = batch_loss(list(chunk), grad=True)
            loss = mse + float(kl_coef) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1
        metrics = evaluate(rows_val)
        history.append(
            {"epoch": epoch, "loss": round(running / max(batches, 1), 8), **metrics}
        )
        print(
            f"[sft][lora] epoch {epoch:3d} loss={running / max(batches, 1):.6f} "
            f"val_mse={metrics['mse']:.6f} val_kl={metrics['kl']:.6f}",
            flush=True,
        )
    return {
        "rows_per_epoch": int(per_epoch),
        "rows_train": int(train_rows.size),
        "baseline": baseline,
        "history": history,
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
        "--frames",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "frames_<stem>.npz files from sil_record --record-frames. Given "
            "these, a second stage trains the SmolVLA LoRA through a "
            "grad-carrying forward from the pictures, which the dataset's "
            "512-wide vision feature cannot do: it is pooled under an "
            "unconditional no_grad behind a fixed random projection."
        ),
    )
    parser.add_argument(
        "--train-vision-lora",
        action="store_true",
        help=(
            "Adapt the SmolVLA vision tower during the LoRA stage. Off in the "
            "RL config on purpose and turned on here: in RL the adapter sees "
            "128 decision-0 records per update through PPO through a ten-step "
            "flow expert, and here it sees every sampled decision under an "
            "MSE. Read vla_lora/vision_modules in the attach line to confirm "
            "it matched something -- a wrong leaf name matches nothing rather "
            "than raising."
        ),
    )
    parser.add_argument("--lora-epochs", type=int, default=8)
    parser.add_argument("--lora-lr", type=float, default=0.0,
                        help="0 = take vla_lr from the checkpoint's args.")
    parser.add_argument(
        "--lora-kl-coef",
        type=float,
        default=0.1,
        help=(
            "Weight on ||prior_new - prior_recorded||^2. The RL update carries "
            "the same anchor; without it, epochs of fitting the policy's own "
            "outputs drag the prior wherever the residual finds convenient, "
            "which is the closed-loop drift this phase is supposed to watch "
            "for rather than cause."
        ),
    )
    parser.add_argument(
        "--lora-microbatch",
        type=int,
        default=4,
        help=(
            "Rows per grad-through-VLA forward. Small because the graph spans "
            "the flow-matching denoise loop and, with vision LoRA, the VLM "
            "prefix too. The RL trainer uses 4-16; raise it until it OOMs."
        ),
    )
    parser.add_argument(
        "--lora-row-fraction",
        type=float,
        default=0.3,
        help=(
            "Share of rows the LoRA stage sees per epoch. The residual stage "
            "is free and uses every row; this one costs a VLA forward and "
            "backward per row, so it is the term that decides the iteration's "
            "wall clock."
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
    best_policy_state: dict[str, Any] | None = None
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
            policy_state = {
                f"actor.{key}": value
                for key, value in actor.state_dict().items()
            }
            if "log_std" in payload["policy"]:
                policy_state["log_std"] = payload["policy"]["log_std"]
            best_policy_state = {
                key: value.detach().cpu().clone()
                for key, value in policy_state.items()
            }
            # No optimizer state, on purpose -- see build_resume_payload. The
            # previous version wrote this run's AdamW into the slot the RL
            # trainer reads, handing a resumed GRPO run moments taken from a
            # supervised loss at a different scale.
            torch.save(
                build_resume_payload(
                    payload,
                    policy_state=best_policy_state,
                    lora_state=None,
                    note={
                        "dataset": str(args.dataset),
                        "source_checkpoint": str(args.checkpoint),
                        "epoch": epoch,
                        "val_mse": val_metrics["mse"],
                        "trained": "residual_only",
                    },
                ),
                output / "sil_sft_adapter.pt",
            )

    lora_report: dict[str, Any] | None = None
    if args.frames:
        # Stage (b). The residual above is now fitted on the RECORDED priors;
        # this stage re-derives them from the pictures with the adapter in the
        # loop, so the two stages are not independent and this one runs second
        # on purpose: it starts from a residual that already reproduces the
        # demonstrations, and only has to keep doing so while the prior moves.
        if best_policy_state is None:
            raise SystemExit(
                "The residual stage never improved on its baseline, so there "
                "is no residual to hand to the LoRA stage."
            )
        vision_dim = int(
            dict(payload["args"]).get("residual_vision_dim", 0)
            if bool(dict(payload["args"]).get("residual_vision_features", False))
            else 0
        )
        frames = load_frame_index([Path(f) for f in args.frames])
        found, lookups = resolve_frame_rows(
            dataset["episode_uid"], dataset["decision_index"], frames
        )
        print(
            f"[sft][lora] {int(found.sum())}/{found.shape[0]} rows found a "
            f"frame across {len(frames)} files",
            flush=True,
        )
        if not found.any():
            example_uid = str(dataset["episode_uid"][0])
            raise SystemExit(
                "No demonstration row matched a frame.\n"
                f"  dataset episode_uid[0] = {example_uid!r} -> join key "
                f"{frame_join_key(example_uid.rpartition('/')[0])!r}\n"
                f"  frame join keys        = {sorted(frames)[:3]}\n"
                "Either these frames come from a different harvest than this "
                "dataset, or the two naming conventions have drifted apart "
                "again -- see frame_join_key."
            )
        runtime, trainer, _ = build_runtime_and_trainer(
            payload, checkpoint=args.checkpoint.expanduser().resolve(),
            device=str(args.device),
            train_vision_lora=bool(args.train_vision_lora),
        )
        base = trainer._unwrap(trainer.actor)
        # best_policy_state is ALREADY in SmolVLAGRPOPolicy's key space --
        # "log_std" plus "actor.net.net.*" -- because the residual stage put it
        # there when it wrote its checkpoint. The first version stripped the
        # "actor." prefix before loading, so nothing matched at all, and
        # strict=False turned that into silence: the LoRA stage was starting
        # from an UNTRAINED residual and there was no way to tell from the loss.
        base.load_state_dict(best_policy_state, strict=True)
        # M5, before a single gradient step: the vision block recomputed from
        # the frames must match the block the dataset recorded. Anything else
        # means these are not the pictures the policy was given.
        probe_rows = [
            int(row) for row in np.flatnonzero(found)[: int(args.lora_microbatch)]
        ]
        probe_positions = {
            int(row): position
            for position, row in enumerate(np.flatnonzero(found))
        }
        overview_np, wrist_np = frames_for_rows(
            lookups, frames, [probe_positions[row] for row in probe_rows]
        )
        proprio_dim = int(dataset["state"].shape[-1]) - vision_dim
        with torch.no_grad():
            _, recomputed = recompute_state_and_prior(
                runtime, torch,
                overview=(
                    torch.as_tensor(overview_np, device=args.device)
                    .to(torch.float32).div_(255.0).permute(0, 3, 1, 2).contiguous()
                ),
                wrist=(
                    torch.as_tensor(wrist_np, device=args.device)
                    .to(torch.float32).div_(255.0).permute(0, 3, 1, 2).contiguous()
                ),
                proprio=torch.as_tensor(
                    dataset["state"][probe_rows, :proprio_dim],
                    dtype=torch.float32, device=args.device,
                ),
                instructions=[
                    str(t) for t in dataset["instruction_text"][probe_rows]
                ],
                vision_dim=vision_dim, enable_grad=False,
            )
        recorded = torch.as_tensor(
            dataset["state"][probe_rows], dtype=torch.float32,
            device=args.device,
        )
        # The control: same rows, same stored frames, a different batch size.
        with torch.no_grad():
            _, control = recompute_state_and_prior(
                runtime, torch,
                overview=(
                    torch.as_tensor(overview_np[:1], device=args.device)
                    .to(torch.float32).div_(255.0).permute(0, 3, 1, 2).contiguous()
                ),
                wrist=(
                    torch.as_tensor(wrist_np[:1], device=args.device)
                    .to(torch.float32).div_(255.0).permute(0, 3, 1, 2).contiguous()
                ),
                proprio=torch.as_tensor(
                    dataset["state"][probe_rows[:1], :proprio_dim],
                    dtype=torch.float32, device=args.device,
                ),
                instructions=[
                    str(t) for t in dataset["instruction_text"][probe_rows[:1]]
                ],
                vision_dim=vision_dim, enable_grad=False,
            )
        integrity = check_recomputed_vision(
            recomputed[:1], recorded[:1], vision_dim=vision_dim, torch=torch,
            control_state=control,
        )
        print(f"[sft][lora] frame/state integrity: {integrity}", flush=True)

        kept = np.flatnonzero(found)
        lora_report = train_lora_stage(
            torch=torch, runtime=runtime, trainer=trainer, actor=base,
            dataset={
                key: value[kept] if getattr(value, "shape", None) else value
                for key, value in dataset.items()
            },
            frames=frames, lookups=lookups,
            rows_train=train_rows[kept], rows_val=val_rows[kept],
            vision_dim=vision_dim, device=torch.device(str(args.device)),
            epochs=int(args.lora_epochs),
            lr=float(args.lora_lr)
            or float(dict(payload["args"]).get("vla_lr", 1.0e-5)),
            kl_coef=float(args.lora_kl_coef),
            microbatch=int(args.lora_microbatch),
            row_fraction=float(args.lora_row_fraction),
            seed=int(args.seed),
        )
        lora_report["frame_state_integrity"] = integrity
        lora_report["rows_with_frames"] = int(found.sum())
        torch.save(
            build_resume_payload(
                payload,
                # No prefix: base IS the policy, so its state_dict already
                # reads "log_std" / "actor.net.net.*". Prefixing again produced
                # "actor.log_std" / "actor.actor.net.net.*", which loaded
                # nowhere.
                policy_state=base.state_dict(),
                lora_state=trainer._vla_lora_state_dict(),
                note={
                    "dataset": str(args.dataset),
                    "source_checkpoint": str(args.checkpoint),
                    "trained": "residual+vla_lora",
                    "lora_epochs": int(args.lora_epochs),
                    "kl_coef": float(args.lora_kl_coef),
                },
            ),
            output / "sil_sft_adapter.pt",
        )
        print(
            f"[sft][lora] wrote {output / 'sil_sft_adapter.pt'} "
            "(residual + LoRA, optimizer states dropped for a clean resume)",
            flush=True,
        )

    report = {
        "dataset": str(args.dataset),
        "source_checkpoint": str(args.checkpoint),
        "lora": lora_report,
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
