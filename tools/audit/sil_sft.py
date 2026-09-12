#!/usr/bin/env python3
"""Fit the residual actor -- and optionally the action-expert LoRA -- to a
demonstration bank.

Reads a ``demonstrations.npz`` written by ``sil_record.py --mode dataset`` or
by ``build_cdpr_staged_sft_dataset.py``, and trains the residual MLP to
reproduce the recorded actions, starting from a chosen initialization.


What this trains
----------------

The policy is a frozen SmolVLA prior, a trainable residual MLP, and LoRA on the
action expert (and, when asked, on the vision tower). Stage (a) trains the
**residual only**, from the stored ``state``/``prior`` columns. Stage (b) runs
only when ``--frames`` is given: it re-derives the prior from the pictures
through a grad-carrying SmolVLA forward and trains the LoRA with the residual
tracking it. The module header used to say "residual only" flatly; that has
been stale since ``train_lora_stage`` was added, and the distinction decides
whether the vision tower moves.

When no LoRA stage runs, the source checkpoint's ``vla_lora`` tensors are
copied verbatim into the output, because dropping them makes a resumed phase
restart from a zero adapter and throw away every step of VLA adaptation. The
saved file is a complete policy either way.


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

For a bank whose episodes share SCENES -- retries of one start, several rollout
seeds of it, the same chain viewed under two labels -- episode splitting is not
enough. Two episodes of one scene differ by the policy's sampling noise and
almost nothing else, so one in train and one in validation is the same
memorization with extra steps. ``--split-by scene`` holds out whole
``scene_uid`` groups instead, and is the correct setting for any bank built by
``build_cdpr_staged_sft_dataset.py``.


Sampling
--------

``--sampler natural`` visits every training row once per epoch, which weights
each stage by how long it happens to take: a placement carry is ~25 decisions
against a pickup's ~9, so the carry gets three times the gradient purely from
duration. ``--sampler balanced`` instead draws destination uniformly, then
semantic stage uniformly, then object, then a row -- with replacement, so a
smaller stratum is revisited rather than padded with fake rows.

Balanced sampling changes EXPOSURE, never the recorded episodes: no trajectory
is truncated and no action is invented. Because it draws with replacement, an
"epoch" is a number of sampled rows rather than a pass over the data, so the
report carries the optimizer updates and the supervised action count that
actually ran.
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


try:  # noqa: E402
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - present in the env lock, not required
    _tqdm = None


def progress_enabled(mode: str = "auto", stream: Any = None) -> bool:
    """Whether to draw bars. Off by default whenever stdout is not a terminal.

    Every long run in this campaign is launched under tee, a redirect or nohup,
    and a carriage-return bar in a log file is thousands of unreadable lines --
    so "auto" means "only when a human is watching". The epoch lines have to
    make the same call as the bars (they are written THROUGH tqdm when it is
    active, so they do not collide with it), which is why this is one predicate
    both consult rather than tqdm's own disable=None.
    """

    if str(mode) == "never":
        return False
    if _tqdm is None:
        return False
    if str(mode) == "always":
        return True
    handle = sys.stdout if stream is None else stream
    try:
        return bool(handle.isatty())
    except Exception:
        # A stream with no isatty (a capture object, a closed pipe) is not a
        # terminal for our purposes.
        return False


def progress_iter(
    iterable: Any,
    *,
    total: int | None,
    desc: str,
    leave: bool,
    enabled: bool,
) -> Any:
    """Wrap an iterable in a bar, or hand it back untouched when disabled."""

    if not enabled or _tqdm is None:
        return iterable
    return _tqdm(
        iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True
    )


def progress_write(text: str, *, enabled: bool) -> None:
    """Emit a line that must survive: above the bars, or as a plain print.

    The per-epoch lines are the durable record -- they are what gets pasted
    into a report -- so they are never replaced by the bar, only relocated.
    """

    if enabled and _tqdm is not None:
        _tqdm.write(text)
    else:
        print(text, flush=True)


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




def _scene_split(
    scene_uid: np.ndarray, *, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole SCENES, not whole episodes.

    ``_episode_split`` groups individual episode ids, which is right for a bank
    where each episode is an independent start. It is not enough for a staged
    bank: retries, extra rollout seeds and the relabelled views of one chain all
    share a ``scene_uid``, differ by the policy's sampling noise, and would
    otherwise land on both sides of the line. The validation number would then
    be measuring memorization of a start it has already seen.
    """

    scenes = np.unique(scene_uid)
    if scenes.size < 2:
        raise SystemExit(
            f"The bank holds {scenes.size} distinct scene(s); a scene-level "
            "split cannot hold anything out. Collect more scenes or pass "
            "--split-by episode deliberately."
        )
    generator = np.random.default_rng(int(seed))
    shuffled = generator.permutation(scenes)
    held = max(1, int(round(len(shuffled) * float(val_fraction))))
    validation = set(shuffled[:held].tolist())
    is_val = np.array(
        [uid in validation for uid in scene_uid.tolist()], dtype=bool
    )
    return ~is_val, is_val


class BalancedRowSampler:
    """Equal exposure by destination, then stage, then object -- not by length.

    The user's intuition, made precise: draw the destination uniformly, then the
    semantic stage uniformly, then the object with bounded imbalance, then a
    valid decision inside that cell. With replacement, so a stratum with fewer
    rows is revisited instead of padded.

    What this deliberately does NOT do is change the recorded data. The design's
    rule is that a chain with 12 move, 9 pickup and 25 placement decisions keeps
    all 46 -- truncating placement to nine, repeating the last frame, or padding
    pickup with invented actions would all "balance" the bank by corrupting it.
    Repeated draws are exposure, not new demonstrations, and the report keeps
    the raw and effective counts apart.

    An empty stratum is an ERROR. Silently replacing a missing (bowl, pick_up)
    cell with plate rows produces a run that trains on one destination while
    every log says two.
    """

    def __init__(
        self,
        dataset: Mapping[str, np.ndarray],
        rows: np.ndarray,
        *,
        seed: int,
        stage_column: str = "stage_name",
        destination_column: str = "destination",
        object_column: str = "target_catalog",
    ) -> None:
        missing = [
            name
            for name in (stage_column, destination_column, object_column)
            if name not in dataset
        ]
        if missing:
            raise SystemExit(
                f"--sampler balanced needs the columns {missing}, which this "
                "bank does not carry. It is written by "
                "build_cdpr_staged_sft_dataset.py; an older demonstrations.npz "
                "must use --sampler natural."
            )
        self.generator = np.random.default_rng(int(seed))
        index = np.flatnonzero(np.asarray(rows, dtype=bool))
        if index.size == 0:
            raise SystemExit("The balanced sampler was given no rows.")
        destinations = np.unique(dataset[destination_column][index])
        stages = np.unique(dataset[stage_column][index])
        self.cells: list[tuple[str, str, list[np.ndarray]]] = []
        self.census: dict[str, int] = {}
        empty: list[str] = []
        for destination in destinations:
            for stage in stages:
                mask = (dataset[destination_column][index] == destination) & (
                    dataset[stage_column][index] == stage
                )
                cell = index[mask]
                name = f"{destination}/{stage}"
                self.census[name] = int(cell.size)
                if cell.size == 0:
                    empty.append(name)
                    continue
                objects = [
                    cell[dataset[object_column][cell] == value]
                    for value in np.unique(dataset[object_column][cell])
                ]
                self.cells.append((str(destination), str(stage), objects))
        if empty:
            raise SystemExit(
                f"Empty strata {empty}. A balanced sampler cannot draw from a "
                "cell with no rows, and substituting another cell would train "
                "on a composition nothing reports. Collect the missing "
                "material or narrow --sampler-destinations / --sampler-stages."
            )
        self.rows = index
        self.draw_counts = np.zeros((len(self.cells),), dtype=np.int64)

    def draw(self, count: int) -> np.ndarray:
        cell_choice = self.generator.integers(len(self.cells), size=int(count))
        self.draw_counts += np.bincount(
            cell_choice, minlength=len(self.cells)
        ).astype(np.int64)
        picked = np.empty((int(count),), dtype=np.int64)
        for position, cell_index in enumerate(cell_choice.tolist()):
            _, _, objects = self.cells[cell_index]
            bucket = objects[self.generator.integers(len(objects))]
            picked[position] = int(
                bucket[self.generator.integers(bucket.size)]
            )
        return picked

    def report(self) -> dict[str, Any]:
        rows_by_stage: dict[str, int] = {}
        rows_by_destination: dict[str, int] = {}
        draws_by_stage: dict[str, int] = {}
        draws_by_destination: dict[str, int] = {}
        draws_by_cell: dict[str, int] = {}
        for cell_index, (destination, stage, _objects) in enumerate(self.cells):
            name = f"{destination}/{stage}"
            rows = int(self.census[name])
            draws = int(self.draw_counts[cell_index])
            rows_by_stage[stage] = rows_by_stage.get(stage, 0) + rows
            rows_by_destination[destination] = (
                rows_by_destination.get(destination, 0) + rows
            )
            draws_by_stage[stage] = draws_by_stage.get(stage, 0) + draws
            draws_by_destination[destination] = (
                draws_by_destination.get(destination, 0) + draws
            )
            draws_by_cell[name] = draws
        return {
            "cells": len(self.cells),
            "rows_available": int(self.rows.size),
            "rows_by_cell": dict(sorted(self.census.items())),
            "rows_by_stage": dict(sorted(rows_by_stage.items())),
            "rows_by_destination": dict(sorted(rows_by_destination.items())),
            "draws_total": int(self.draw_counts.sum()),
            "draws_by_cell": dict(sorted(draws_by_cell.items())),
            "draws_by_stage": dict(sorted(draws_by_stage.items())),
            "draws_by_destination": dict(sorted(draws_by_destination.items())),
        }


class RetentionMixer:
    """A declared share of each batch drawn from the original-label bank.

    Retention is a REQUIREMENT here, not a regularizer: the campaign's four
    instruction families are served by one policy, and a full-task SFT that
    saw only relabelled ``put_into`` rows would erase the others. Phase 3
    measured exactly that -- pick_up went to 0.000, not "degraded", because it
    was excluded from the SFT mix.

    The fraction is a starting experiment and is reported as realized counts,
    because a mixer that silently cannot supply its share is the same class of
    bug as the composed-fraction sweep that realized 0.981 three times.
    """

    def __init__(
        self,
        retention: Mapping[str, np.ndarray] | None,
        *,
        fraction: float,
        seed: int,
    ) -> None:
        self.retention = retention
        self.fraction = min(1.0, max(0.0, float(fraction)))
        self.generator = np.random.default_rng(int(seed) + 977)
        if self.retention is not None and self.fraction <= 0.0:
            raise SystemExit(
                "A retention bank was supplied with --retention-fraction 0. "
                "Either drop the bank or ask for a share of it."
            )
        if self.retention is None and self.fraction > 0.0:
            raise SystemExit(
                "--retention-fraction is positive but no --retention-dataset "
                "was given. Retention cannot come from the full-task bank: "
                "every row of it carries a put_into label."
            )
        self.drawn = 0

    @property
    def active(self) -> bool:
        return self.retention is not None and self.fraction > 0.0

    def split_counts(self, batch: int) -> tuple[int, int]:
        if not self.active:
            return int(batch), 0
        retained = int(round(int(batch) * self.fraction))
        return int(batch) - retained, retained

    def draw(self, count: int) -> np.ndarray:
        if count <= 0 or not self.active:
            return np.empty((0,), dtype=np.int64)
        total = int(self.retention["state"].shape[0])
        self.drawn += int(count)
        return self.generator.integers(total, size=int(count)).astype(np.int64)


def per_group_metrics(
    actor: Any,
    torch: Any,
    *,
    dataset: Mapping[str, np.ndarray],
    rows: np.ndarray,
    state: Any,
    prior: Any,
    action: Any,
    mask: Any,
    batch_size: int,
    column: str,
) -> dict[str, Any]:
    """Validation loss broken out by a categorical column.

    A single validation MSE cannot say WHERE a bank is failing to fit, and the
    three stages of a chain have very different action statistics: the approach
    is smooth XY, the grasp is a gripper step, the carry is a long traverse with
    a release at the end. A pooled number that improves while the release gets
    worse looks exactly like a number that improves.
    """

    if column not in dataset:
        return {}
    labels = dataset[column][rows]
    out: dict[str, Any] = {}
    for name in np.unique(labels):
        selected = np.flatnonzero(labels == name)
        if selected.size == 0:
            continue
        index = torch.as_tensor(selected, dtype=torch.int64, device=state.device)
        metrics = _evaluate(
            actor,
            torch,
            state=state.index_select(0, index),
            prior=prior.index_select(0, index),
            action=action.index_select(0, index),
            mask=mask.index_select(0, index),
            batch_size=int(batch_size),
        )
        out[str(name)] = {"rows": int(selected.size), **metrics}
    return out


def per_group_reachability(
    dataset: Mapping[str, np.ndarray],
    rows: np.ndarray,
    *,
    residual_scale: float,
    column: str,
) -> dict[str, Any]:
    """Target reachability by group, and by action axis inside each group.

    A relabelled prefix can sit outside the residual's bounded correction range
    even when the pooled fraction looks fine: the move-to prefix is now
    conditioned on a placement prompt, and the prior it gets under that prompt
    is a different prediction than the one that produced the recorded action.
    Reported per axis because "unreachable" almost always means one axis --
    typically the gripper or Z -- rather than a diffuse gap.
    """

    if column not in dataset:
        return {}
    labels = dataset[column][rows]
    prior = dataset["prior"][rows]
    action = dataset["action"][rows]
    mask = dataset["action_mask"][rows]
    out: dict[str, Any] = {}
    axes = ("x", "y", "z", "yaw", "gripper")
    for name in np.unique(labels):
        selected = labels == name
        entry = _reachability(
            prior[selected],
            action[selected],
            mask[selected],
            residual_scale=residual_scale,
        )
        slots = action.shape[1]
        low = np.tanh(prior[selected][:, :slots] - float(residual_scale))
        high = np.tanh(prior[selected][:, :slots] + float(residual_scale))
        reachable = (
            action[selected] >= low - 1e-6
        ) & (action[selected] <= high + 1e-6)
        live = mask[selected]
        entry["by_axis"] = {
            axes[axis]: round(
                float(reachable[..., axis][live].mean()), 5
            )
            for axis in range(min(len(axes), action.shape[-1]))
            if live.any()
        }
        out[str(name)] = entry
    return out


def refuse_stale_priors(dataset_path: Path, *, allow: bool) -> dict[str, Any]:
    """A relabelled bank whose priors were never refreshed is not trainable.

    ``build_cdpr_staged_sft_dataset.py`` rewrites the instruction on every row
    of a chain, including its move-to prefix, but ``state`` and ``prior`` were
    computed under the TEACHERS' prompts and adapters. Training on that pair
    fits the residual to correct a prediction the student will never make, and
    nothing about the loss curve says so. ``sil_refresh_priors.py`` clears the
    marker; this refuses the run until it has.
    """

    report = dataset_path.parent / "dataset.json"
    if not report.is_file():
        return {}
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not bool(payload.get("priors_stale")):
        return payload
    if allow:
        print(
            "[sft] WARNING: this bank is marked priors_stale and "
            "--allow-stale-priors was passed. The residual is being fitted "
            "against priors drawn under the teachers' prompts.",
            flush=True,
        )
        return payload
    raise SystemExit(
        f"{report} marks this bank priors_stale: "
        f"{payload.get('priors_stale_reason')}\n"
        "Run tools/audit/sil_refresh_priors.py with the student "
        "initialization first, or pass --allow-stale-priors to train on the "
        "teachers' priors deliberately."
    )


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


def _npz_member_shape(path: Path, member: str) -> tuple[int, ...]:
    """The shape of one npz member, without decompressing the array.

    An npz is a zip of .npy files, and a .npy begins with a header giving the
    shape. Reading the first kilobyte of the member's decompressed stream is
    enough, which matters here: the arrays are gigabytes and all that is wanted
    is how many decisions they hold.
    """

    import zipfile

    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{member}.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version[0] == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(stream)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(stream)
    return tuple(int(value) for value in shape)


def load_frame_meta(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Index the frames files WITHOUT loading a single picture.

    The first version of this loaded every file's overview and wrist arrays up
    front. One iteration's harvest -- nine rungs, four rounds, uncapped worlds
    -- is 81 GB uncompressed, and the process was killed by the kernel's OOM
    killer after the residual stage had already finished. The arithmetic was
    available (a full round is 236 MB per decision at 512 worlds and two
    cameras, and it is written in the frame tap's own docstring); nobody
    multiplied it by thirty-six files at the point where it mattered.

    So the index carries only what the join needs -- which worlds a file holds
    and how many decisions -- and the pictures are fetched later, for a bounded
    set of rows.
    """

    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        name = Path(path).stem
        if not name.startswith("frames_"):
            raise SystemExit(
                f"{path} is not a frames_<stem>.npz written by sil_record."
            )
        with np.load(path, allow_pickle=False) as data:
            worlds = np.asarray(data["world_index"])
            decisions = (
                int(data["decisions"]) if "decisions" in data.files else None
            )
        if decisions is None:
            decisions = int(_npz_member_shape(Path(path), "overview")[0])
        index[frame_join_key(name)] = {
            "path": str(path),
            "world_column": {
                int(world): position for position, world in enumerate(worlds)
            },
            "decisions": int(decisions),
        }
    return index


def resolve_frame_rows(
    episode_uid: np.ndarray,
    decision_index: np.ndarray,
    frames: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Map each demonstration row to (join key, decision, world column).

    Returns a mask of the rows that FOUND a frame and the lookups for them.
    Rows are dropped rather than filled: a missing frame means the replay that
    produced the row kept no pictures for that world (``--frame-worlds`` capped
    it, or the world failed the replay), and inventing one would train the
    vision path on a picture from a different episode.

    Takes only the metadata index, so nothing is read from disk here.
    """

    keep = np.zeros(episode_uid.shape[0], dtype=bool)
    lookups: list[tuple[str, int, int]] = []
    for row, (uid, decision) in enumerate(zip(episode_uid, decision_index)):
        raw_stem, _, tail = str(uid).rpartition("/")
        # The NORMALISED key travels in the lookup, because materialize_frames
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


def load_frame_meta_by_uid(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Index staged frames files by the EPISODE ID they carry.

    The positional convention -- parse a world number out of a uid, look it up
    in ``world_index``, hope the file stem matches -- worked, but its failure
    mode is silence: the first version of that join matched 0 of 33 102 rows
    after a whole harvest had been paid for, because two writers spelled the
    same episode differently. A staged frames file carries the episode ids
    themselves, so the join is an identity comparison and a miss is a miss
    rather than a naming disagreement.

    Nothing is decompressed here; only the id array and the decision count.
    """

    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "episode_uid" not in data.files:
                raise SystemExit(
                    f"{path} carries no episode_uid array, so it cannot be "
                    "joined by explicit id. Use the positional join "
                    "(load_frame_meta) for frames written by sil_record."
                )
            uids = [str(value) for value in data["episode_uid"]]
            decisions = (
                int(data["decisions"])
                if "decisions" in data.files
                else int(_npz_member_shape(Path(path), "overview")[0])
            )
        for column, uid in enumerate(uids):
            if uid in index:
                raise SystemExit(
                    f"Episode {uid!r} appears in two frames files "
                    f"({index[uid]['path']} and {path}). One of them would "
                    "silently win and half the bank would be paired with the "
                    "wrong pictures."
                )
            index[uid] = {
                "path": str(path),
                "column": int(column),
                "decisions": int(decisions),
            }
    return index


def resolve_frame_rows_by_uid(
    episode_uid: np.ndarray,
    decision_index: np.ndarray,
    frames: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Map each row to (frames path, decision, world column) by explicit id.

    Returns the same shape ``resolve_frame_rows`` does -- a keep mask and the
    lookups for the kept rows -- so ``materialize_frames`` is unchanged. The
    key in each lookup is the FILE PATH rather than a derived stem, because
    there is no stem convention left to get wrong.
    """

    keep = np.zeros(episode_uid.shape[0], dtype=bool)
    lookups: list[tuple[str, int, int]] = []
    for row, (uid, decision) in enumerate(zip(episode_uid, decision_index)):
        entry = frames.get(str(uid))
        if entry is None or int(decision) >= int(entry["decisions"]):
            continue
        keep[row] = True
        lookups.append((str(entry["path"]), int(decision), int(entry["column"])))
    return keep, lookups


def materialize_frames_by_path(
    lookups: Sequence[tuple[str, int, int]], rows: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """``materialize_frames`` for lookups keyed by file path.

    Same one-pass-per-file discipline: a file is decompressed once, every
    selected row it holds is taken, and it is released. Peak host memory is one
    file plus the result.
    """

    if not rows:
        empty = np.zeros((0, 0, 0, 3), dtype=np.uint8)
        return empty, empty.copy()
    wanted: dict[str, list[tuple[int, int, int]]] = {}
    for position, row in enumerate(rows):
        path, decision, column = lookups[row]
        wanted.setdefault(path, []).append((position, decision, column))
    overview: np.ndarray | None = None
    wrist: np.ndarray | None = None
    for path, items in wanted.items():
        with np.load(path, allow_pickle=False) as data:
            source_overview = data["overview"]
            source_wrist = data["wrist"]
            if overview is None:
                shape = (len(rows),) + tuple(source_overview.shape[2:])
                overview = np.empty(shape, dtype=source_overview.dtype)
                wrist = np.empty(shape, dtype=source_wrist.dtype)
            for position, decision, column in items:
                overview[position] = source_overview[decision, column]
                wrist[position] = source_wrist[decision, column]
            del source_overview, source_wrist
    return overview, wrist


def projected_frame_bytes(
    meta: Mapping[str, Mapping[str, Any]], height: int = 240, width: int = 320
) -> int:
    """What loading every frame in the index would cost in RAM."""

    total = 0
    for entry in meta.values():
        total += (
            int(entry["decisions"])
            * len(entry["world_column"])
            * 2
            * int(height)
            * int(width)
            * 3
        )
    return total


def materialize_frames(
    meta: Mapping[str, Mapping[str, Any]],
    lookups: Sequence[tuple[str, int, int]],
    rows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the pictures for a BOUNDED set of resolved rows and nothing else.

    One pass per file that any selected row lands in: decompress it once, take
    the slices, release it. Peak memory is one file plus the result, instead of
    every file at once.
    """

    wanted: dict[str, list[tuple[int, int, int]]] = {}
    for position, row in enumerate(rows):
        key, decision, column = lookups[row]
        wanted.setdefault(key, []).append((position, decision, column))
    if not rows:
        empty = np.zeros((0, 0, 0, 3), dtype=np.uint8)
        return empty, empty.copy()

    overview: np.ndarray | None = None
    wrist: np.ndarray | None = None
    for key, items in wanted.items():
        with np.load(meta[key]["path"], allow_pickle=False) as data:
            source_overview = data["overview"]
            source_wrist = data["wrist"]
            if overview is None:
                shape = (len(rows),) + tuple(source_overview.shape[2:])
                overview = np.empty(shape, dtype=source_overview.dtype)
                wrist = np.empty(shape, dtype=source_wrist.dtype)
            for position, decision, column in items:
                overview[position] = source_overview[decision, column]
                wrist[position] = source_wrist[decision, column]
            del source_overview, source_wrist
    return overview, wrist


def select_frame_budget(
    found: np.ndarray, *, budget: int, seed: int
) -> np.ndarray:
    """Which resolved rows the LoRA stage will actually see, chosen once.

    A budget in ROWS, not a fraction of the harvest, because it is the only
    knob that bounds memory: the pictures for the selected rows are held for
    the whole stage, at 2 x 240 x 320 x 3 bytes each. A fraction scales with
    the harvest and the harvest scales with the ladder, which is how one
    iteration came to ask for 81 GB.

    Drawn once and reused across epochs rather than resampled per epoch, so the
    frames are loaded once. That costs some sample diversity and buys a bound;
    at 8192 rows against ~120k resolved, the alternative is loading the lot.
    """

    resolved = np.flatnonzero(found)
    if budget <= 0 or resolved.size <= int(budget):
        return resolved
    generator = np.random.default_rng(int(seed))
    return np.sort(generator.choice(resolved, size=int(budget), replace=False))


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
    overview_all: np.ndarray,
    wrist_all: np.ndarray,
    rows_train: np.ndarray,
    rows_val: np.ndarray,
    vision_dim: int,
    device: Any,
    epochs: int,
    lr: float,
    kl_coef: float,
    microbatch: int,
    seed: int,
    actor_lr: float,
    show_progress: bool = False,
    sampler_mode: str = "natural",
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
    # Two groups, two rates. The adapter and the residual are at different
    # points in their lives here: the residual has just been fitted to
    # convergence, and the adapter is about to take 512 times the optimizer
    # steps of a single RL update (8192 rows x 8 epochs / microbatch 4 = 16384
    # steps, against 128 records / 4 = 32 per update). Running both at RL's
    # per-update rate is what made the first attempt diverge -- train loss ROSE
    # from epoch 1 and validation went 64% worse than its own starting point.
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(lr)},
            {"params": list(actor.parameters()), "lr": float(actor_lr)},
        ]
    )
    generator = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []

    def batch_loss(rows: Sequence[int], *, grad: bool) -> tuple[Any, Any, int]:
        # rows index the MATERIALISED budget, so this is a gather from an
        # array already in memory rather than a read from thirty-six files.
        picked = np.asarray(rows, dtype=np.int64)
        overview_np = overview_all[picked]
        wrist_np = wrist_all[picked]
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
    # The starting point is a real candidate: if no epoch beats it, the right
    # answer is to apply no LoRA at all rather than the last thing computed.
    best = float(baseline["mse"])
    best_epoch = -1
    best_state: dict[str, Any] | None = None

    train_rows = np.flatnonzero(rows_train)
    # The SAME sampling rule as the residual stage. Two stages of one run
    # drawing from two different distributions would make the second stage's
    # contribution impossible to attribute: the design's requirement is
    # identical sampling logic in both training paths.
    stage_sampler = (
        BalancedRowSampler(dataset, rows_train, seed=int(seed))
        if str(sampler_mode) == "balanced"
        else None
    )
    if stage_sampler is not None:
        print(
            f"[sft][lora] balanced sampler over the frame budget: "
            f"{stage_sampler.report()}",
            flush=True,
        )
    lora_bar = progress_iter(
        range(int(epochs)),
        total=int(epochs),
        desc="sft lora",
        leave=True,
        enabled=show_progress,
    )
    for epoch in lora_bar:
        picked = (
            generator.permutation(train_rows)
            if stage_sampler is None
            else stage_sampler.draw(int(train_rows.size))
        )
        running = 0.0
        batches = 0
        lora_starts = list(range(0, picked.size, int(microbatch)))
        for start in progress_iter(
            lora_starts,
            total=len(lora_starts),
            desc=f"lora epoch {epoch}",
            leave=False,
            enabled=show_progress,
        ):
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
        improved = float(metrics["mse"]) < best
        if improved:
            best = float(metrics["mse"])
            best_epoch = epoch
            best_state = {
                "actor": {
                    key: value.detach().cpu().clone()
                    for key, value in actor.state_dict().items()
                },
                "lora": {
                    key: value.detach().cpu().clone()
                    for key, value in
                    (trainer._vla_lora_state_dict() or {}).items()
                },
            }
        progress_write(
            f"[sft][lora] epoch {epoch:3d} loss={running / max(batches, 1):.6f} "
            f"val_mse={metrics['mse']:.6f} val_kl={metrics['kl']:.6f}"
            f"{'  <- best' if improved else ''}",
            enabled=show_progress,
        )
        if show_progress and hasattr(lora_bar, "set_postfix"):
            lora_bar.set_postfix(
                val_mse=f"{metrics['mse']:.6f}",
                val_kl=f"{metrics['kl']:.2e}",
                refresh=False,
            )
    if best_state is None:
        print(
            "[sft][lora] NO epoch beat the starting point "
            f"({baseline['mse']}). The adapter is not applied and the "
            "residual-only checkpoint stands. Lower --lora-lr or --lora-epochs "
            "before reading this as 'LoRA does not help' -- a rising TRAIN "
            "loss means the step size is wrong, not that the data is.",
            flush=True,
        )
    else:
        print(
            f"[sft][lora] best epoch {best_epoch} at val_mse {best}",
            flush=True,
        )
    stage_sampler_report = (
        None if stage_sampler is None else stage_sampler.report()
    )
    if stage_sampler_report is not None:
        print(
            "[sft][lora] realized balanced exposure: "
            f"{stage_sampler_report['draws_by_stage']} by stage, "
            f"{stage_sampler_report['draws_by_destination']} by destination",
            flush=True,
        )
    return {
        "best_epoch": best_epoch,
        "best_val_mse": best,
        "applied": best_state is not None,
        "state": best_state,
        "rows_per_epoch": int(train_rows.size),
        "rows_train": int(train_rows.size),
        "baseline": baseline,
        "history": history,
        "sampler_report": stage_sampler_report,
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
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Draw tqdm bars for the epoch and batch loops. 'auto' (the "
            "default) draws them only when stdout is a terminal, because these "
            "runs are normally launched under tee or a redirect and a "
            "carriage-return bar in a log file is thousands of unreadable "
            "lines. The per-epoch lines are printed either way -- when a bar "
            "is active they go through tqdm.write so they scroll above it "
            "instead of colliding with it -- so a redirected run's log is "
            "byte-identical to one from before this existed. 'always' forces "
            "bars on (a tmux pipe someone is watching); 'never' forces them "
            "off."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--split-by",
        choices=("episode", "scene"),
        default="episode",
        help=(
            "What the held-out unit is. 'episode' is right when each episode "
            "is an independent start. 'scene' is REQUIRED for a bank whose "
            "episodes share starts -- retries, extra rollout seeds, relabelled "
            "views -- because two rollouts of one scene differ by sampling "
            "noise and splitting between them measures memorization."
        ),
    )
    parser.add_argument(
        "--sampler",
        choices=("natural", "balanced"),
        default="natural",
        help=(
            "'natural' visits every training row once per epoch, which weights "
            "each stage by how long it takes. 'balanced' draws destination, "
            "then semantic stage, then object uniformly, with replacement. "
            "Balancing changes EXPOSURE only: no episode is truncated and no "
            "action is invented."
        ),
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help=(
            "Optimizer steps per epoch under --sampler balanced. 0 means as "
            "many as a natural pass would take, so the two samplers are "
            "compared at a matched optimizer budget."
        ),
    )
    parser.add_argument(
        "--retention-dataset",
        type=Path,
        default=None,
        help=(
            "Original-label move/pickup/placement successes, mixed in at "
            "--retention-fraction. Retention cannot come from the full-task "
            "bank: every row of that carries a put_into label, and a run "
            "trained on it alone erases the other instructions -- measured, "
            "pick_up went to exactly 0.000 when it was left out of a mix."
        ),
    )
    parser.add_argument(
        "--retention-fraction",
        type=float,
        default=0.0,
        help=(
            "Share of each batch drawn from --retention-dataset. The design's "
            "starting point is 0.2 against 0.8 full-task; it is an experiment "
            "to report, not a proven optimum."
        ),
    )
    parser.add_argument(
        "--allow-stale-priors",
        action="store_true",
        help=(
            "Train on a bank whose dataset.json still marks its priors stale. "
            "A relabelled bank's state/prior were computed under the teachers' "
            "prompts; sil_refresh_priors.py exists to fix that."
        ),
    )
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
    parser.add_argument(
        "--lora-lr",
        type=float,
        default=0.0,
        help=(
            "0 = one TENTH of the checkpoint's vla_lr. Not vla_lr itself: that "
            "rate is calibrated for 32 optimizer steps per RL update under a "
            "PPO objective with a KL term holding it, and this stage takes "
            "16384 under an MSE. At the full rate the first run's TRAIN loss "
            "rose from epoch 1 and validation ended 64% above its own starting "
            "point. A starting point to be measured, not a tuned value."
        ),
    )
    parser.add_argument(
        "--lora-actor-lr",
        type=float,
        default=0.0,
        help=(
            "Residual learning rate DURING the LoRA stage; 0 = one tenth of "
            "--lr. The residual arrives here converged, and its job in this "
            "stage is to track a moving prior rather than to refit."
        ),
    )
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
        "--lora-rows",
        type=int,
        default=8192,
        help=(
            "Hard budget, in ROWS, on what the LoRA stage sees -- and the only "
            "knob that bounds memory, since those rows' pictures are held for "
            "the whole stage at 2x240x320x3 bytes each (8192 rows ~ 3.8 GB). "
            "A FRACTION of the harvest was the previous spelling and it scales "
            "with the ladder: one iteration's nine rungs at four rounds asked "
            "for 81 GB and the kernel killed the process. 0 = every resolved "
            "row, which is only safe on a small harvest."
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

    dataset_path = args.dataset.expanduser().resolve()
    bank_report = refuse_stale_priors(
        dataset_path, allow=bool(args.allow_stale_priors)
    )
    dataset = _load_dataset(dataset_path)
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

    if str(args.split_by) == "scene":
        if "scene_uid" not in dataset:
            raise SystemExit(
                "--split-by scene needs a scene_uid column. It is written by "
                "build_cdpr_staged_sft_dataset.py; an older bank has only "
                "episode ids and must use --split-by episode."
            )
        train_rows, val_rows = _scene_split(
            dataset["scene_uid"],
            val_fraction=float(args.val_fraction),
            seed=int(args.seed),
        )
    else:
        train_rows, val_rows = _episode_split(
            dataset["episode_uid"],
            val_fraction=float(args.val_fraction),
            seed=int(args.seed),
        )
    if "scene_uid" in dataset:
        # Explicit, because it is the failure this split exists to prevent and
        # a silent overlap looks like an unusually good validation curve.
        shared = set(dataset["scene_uid"][train_rows].tolist()) & set(
            dataset["scene_uid"][val_rows].tolist()
        )
        if shared and str(args.split_by) == "scene":
            raise SystemExit(
                f"{len(shared)} scenes appear on both sides of a scene split."
            )
        scene_leakage = len(shared)
    else:
        scene_leakage = None

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

    retention = None
    if args.retention_dataset is not None:
        retention = _load_dataset(args.retention_dataset.expanduser().resolve())
        if int(retention["state"].shape[-1]) != state_dim:
            raise SystemExit(
                f"The retention bank carries "
                f"{retention['state'].shape[-1]}-wide states against the "
                f"checkpoint's {state_dim}. The two banks were recorded under "
                "different observation layouts and cannot be mixed."
            )
        if int(retention["action"].shape[1]) != slots:
            raise SystemExit(
                "The retention bank supervises "
                f"{retention['action'].shape[1]} action slots against the "
                f"full-task bank's {slots}."
            )
        if "scene_uid" in retention and "scene_uid" in dataset:
            overlap = set(retention["scene_uid"].tolist()) & set(
                dataset["scene_uid"].tolist()
            )
            if overlap:
                raise SystemExit(
                    f"{len(overlap)} scenes appear in BOTH the full-task bank "
                    "and the retention bank. A shared scene must stay on one "
                    "side of the split across both views."
                )
    mixer = RetentionMixer(
        retention, fraction=float(args.retention_fraction), seed=int(args.seed)
    )
    if mixer.active:
        ret_state = torch.as_tensor(
            retention["state"], dtype=torch.float32, device=device
        )
        ret_prior = torch.as_tensor(
            retention["prior"], dtype=torch.float32, device=device
        )
        ret_action = torch.as_tensor(
            retention["action"], dtype=torch.float32, device=device
        )
        ret_mask = torch.as_tensor(retention["action_mask"], device=device)

    # Dataset row -> position in the training tensors, so a sampler that thinks
    # in dataset rows can index the tensors that were gathered from them.
    train_positions = np.full((int(train_rows.shape[0]),), -1, dtype=np.int64)
    train_positions[np.flatnonzero(train_rows)] = np.arange(
        int(train_rows.sum()), dtype=np.int64
    )
    sampler = None
    sampler_report: dict[str, Any] | None = None
    if str(args.sampler) == "balanced":
        sampler = BalancedRowSampler(dataset, train_rows, seed=int(args.seed))
        sampler_report = sampler.report()
        print(f"[sft] balanced sampler: {sampler_report}", flush=True)

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

    show_progress = progress_enabled(str(args.progress))
    epoch_bar = progress_iter(
        range(int(args.epochs)),
        total=int(args.epochs),
        desc="sft residual",
        leave=True,
        enabled=show_progress,
    )
    # An "epoch" under replacement sampling is a number of SAMPLED rows, not a
    # pass over the data, so the two counters below are reported rather than
    # inferred from the epoch count.
    natural_batches = max(
        1,
        (int(tr_state.shape[0]) + int(args.batch_size) - 1)
        // int(args.batch_size),
    )
    batches_per_epoch = (
        natural_batches
        if sampler is None or int(args.steps_per_epoch) <= 0
        else int(args.steps_per_epoch)
    )
    optimizer_updates = 0
    supervised_actions = 0
    retention_rows_drawn = 0
    for epoch in epoch_bar:
        order = torch.randperm(
            int(tr_state.shape[0]), generator=generator
        ).to(device)
        running = 0.0
        batches = 0
        starts = list(range(0, int(order.numel()), int(args.batch_size)))
        if sampler is not None:
            starts = list(range(batches_per_epoch))
        for start in progress_iter(
            starts,
            total=len(starts),
            desc=f"epoch {epoch}",
            leave=False,
            enabled=show_progress,
        ):
            main_count, retained_count = mixer.split_counts(
                int(args.batch_size)
            )
            if sampler is None:
                index = order[start : start + main_count]
            else:
                drawn = sampler.draw(main_count)
                positions = train_positions[drawn]
                index = torch.as_tensor(
                    positions, dtype=torch.int64, device=device
                )
            batch_state = tr_state.index_select(0, index)
            batch_prior = tr_prior.index_select(0, index)
            target = tr_action.index_select(0, index)
            weight_mask = tr_mask.index_select(0, index)
            if retained_count > 0:
                # Concatenated rather than alternated, so every optimizer step
                # sees both distributions. Alternating batches would let the
                # last batch of an epoch decide which one the step ends on.
                retained = mixer.draw(retained_count)
                retention_rows_drawn += int(retained.size)
                keep = torch.as_tensor(
                    retained, dtype=torch.int64, device=device
                )
                batch_state = torch.cat(
                    [batch_state, ret_state.index_select(0, keep)], dim=0
                )
                batch_prior = torch.cat(
                    [batch_prior, ret_prior.index_select(0, keep)], dim=0
                )
                target = torch.cat(
                    [target, ret_action.index_select(0, keep)], dim=0
                )
                weight_mask = torch.cat(
                    [weight_mask, ret_mask.index_select(0, keep)], dim=0
                )
            out = actor(batch_state, batch_prior)[:, :slots]
            weight = weight_mask.unsqueeze(-1).float()
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
            optimizer_updates += 1
            supervised_actions += int(weight_mask.sum().item())

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
        progress_write(
            f"[sft] epoch {epoch:3d} loss={running / max(batches, 1):.6f} "
            f"train_mse={train_metrics['mse']:.6f} "
            f"val_mse={val_metrics['mse']:.6f}",
            enabled=show_progress,
        )
        if show_progress and hasattr(epoch_bar, "set_postfix"):
            # The two numbers worth watching over 60+ epochs: where val_mse is
            # now, and whether it is still falling.
            epoch_bar.set_postfix(
                val_mse=f"{val_metrics['mse']:.6f}",
                best=f"{min(best, val_metrics['mse']):.6f}",
                refresh=False,
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
        # Metadata only -- no pictures are read here. See load_frame_meta.
        frames = load_frame_meta([Path(f) for f in args.frames])
        found, lookups = resolve_frame_rows(
            dataset["episode_uid"], dataset["decision_index"], frames
        )
        whole = projected_frame_bytes(frames) / 1e9
        print(
            f"[sft][lora] {int(found.sum())}/{found.shape[0]} rows found a "
            f"frame across {len(frames)} files "
            f"({whole:.1f} GB if every frame were loaded)",
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
        # The bounded set of rows this stage will see, and the only pictures
        # that are ever held. Chosen before anything is read.
        budget_rows = select_frame_budget(
            found, budget=int(args.lora_rows), seed=int(args.seed)
        )
        held = budget_rows.size * 2 * 240 * 320 * 3 / 1e9
        print(
            f"[sft][lora] budget {budget_rows.size} rows of "
            f"{int(found.sum())} resolved -> ~{held:.2f} GB of frames held",
            flush=True,
        )
        budget_lookup = {int(row): position for position, row in enumerate(
            np.flatnonzero(found)
        )}
        overview_all, wrist_all = materialize_frames(
            frames, lookups, [budget_lookup[int(row)] for row in budget_rows]
        )
        # Row -> position in the materialised budget.
        position_of = {int(row): i for i, row in enumerate(budget_rows)}
        # M5, before a single gradient step: the vision block recomputed from
        # the frames must match the block the dataset recorded. Anything else
        # means these are not the pictures the policy was given.
        probe_rows = [
            int(row) for row in budget_rows[: int(args.lora_microbatch)]
        ]
        probe_slice = np.asarray(
            [position_of[row] for row in probe_rows], dtype=np.int64
        )
        overview_np = overview_all[probe_slice]
        wrist_np = wrist_all[probe_slice]
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

        kept = budget_rows
        lora_report = train_lora_stage(
            show_progress=show_progress,
            torch=torch, runtime=runtime, trainer=trainer, actor=base,
            dataset={
                key: value[kept] if getattr(value, "shape", None) else value
                for key, value in dataset.items()
            },
            overview_all=overview_all, wrist_all=wrist_all,
            rows_train=train_rows[kept], rows_val=val_rows[kept],
            vision_dim=vision_dim, device=torch.device(str(args.device)),
            epochs=int(args.lora_epochs),
            lr=float(args.lora_lr)
            or float(dict(payload["args"]).get("vla_lr", 1.0e-5)) / 10.0,
            kl_coef=float(args.lora_kl_coef),
            microbatch=int(args.lora_microbatch),
            seed=int(args.seed),
            actor_lr=float(args.lora_actor_lr) or float(args.lr) / 10.0,
            sampler_mode=str(args.sampler),
        )
        lora_report["frame_state_integrity"] = integrity
        lora_report["rows_with_frames"] = int(found.sum())
        lora_report["rows_in_budget"] = int(budget_rows.size)
        lora_report["frames_held_gb"] = round(held, 3)
        best_state = lora_report.pop("state", None)
        if best_state is not None:
            base.load_state_dict(best_state["actor"], strict=True)
            if best_state["lora"]:
                runtime.policy.load_state_dict(best_state["lora"], strict=False)
        if lora_report["applied"]:
            torch.save(
                build_resume_payload(
                    payload,
                    # No prefix: base IS the policy, so its state_dict already
                    # reads "log_std" / "actor.net.net.*".
                    policy_state=base.state_dict(),
                    lora_state=trainer._vla_lora_state_dict(),
                    note={
                        "dataset": str(args.dataset),
                        "source_checkpoint": str(args.checkpoint),
                        "trained": "residual+vla_lora",
                        "lora_epochs": int(args.lora_epochs),
                        "lora_best_epoch": lora_report["best_epoch"],
                        "kl_coef": float(args.lora_kl_coef),
                    },
                ),
                output / "sil_sft_adapter.pt",
            )
            print(
                f"[sft][lora] wrote {output / 'sil_sft_adapter.pt'} "
                f"(residual + LoRA from epoch {lora_report['best_epoch']}, "
                "optimizer states dropped for a clean resume)",
                flush=True,
            )
        else:
            # The residual-only checkpoint written by the stage above is left
            # exactly where it is. Overwriting it with a diverged adapter is
            # what the first run did, and the file left on disk was that
            # stage's WORST epoch.
            print(
                f"[sft][lora] left {output / 'sil_sft_adapter.pt'} as the "
                "residual-only checkpoint; no adapter was applied.",
                flush=True,
            )


    # Per-stage and per-destination validation, taken on the BEST residual
    # rather than on whatever the last epoch left behind. A pooled MSE that
    # improves while the release gets worse looks exactly like a pooled MSE
    # that improves, and the three stages of a chain have very different action
    # statistics.
    per_stage: dict[str, Any] = {}
    if best_policy_state is not None:
        restored = _build_actor(payload, device)
        restored.load_state_dict(
            {
                key[len("actor.") :]: value
                for key, value in best_policy_state.items()
                if key.startswith("actor.")
            },
            strict=True,
        )
        val_index = np.flatnonzero(val_rows)
        for column in ("stage_name", "destination", "target_catalog"):
            per_stage[f"val_by_{column}"] = per_group_metrics(
                restored,
                torch,
                dataset=dataset,
                rows=val_index,
                state=va_state,
                prior=va_prior,
                action=va_action,
                mask=va_mask,
                batch_size=int(args.batch_size),
                column=column,
            )
    reach_by_group = {
        f"reachability_by_{column}": per_group_reachability(
            dataset,
            np.flatnonzero(train_rows),
            residual_scale=residual_scale,
            column=column,
        )
        for column in ("stage_name", "destination")
    }
    if sampler is not None:
        sampler_report = sampler.report()
        print(
            "[sft] realized balanced exposure: "
            f"{sampler_report['draws_by_stage']} by stage, "
            f"{sampler_report['draws_by_destination']} by destination",
            flush=True,
        )

    report = {
        "dataset": str(args.dataset),
        "source_checkpoint": str(args.checkpoint),
        "lora": lora_report,
        "bank_report": bank_report or None,
        "split_by": str(args.split_by),
        "scene_leakage": scene_leakage,
        "sampler": str(args.sampler),
        "sampler_report": sampler_report,
        "batches_per_epoch": int(batches_per_epoch),
        "optimizer_updates": int(optimizer_updates),
        "supervised_actions_sampled": int(supervised_actions),
        "retention": {
            "dataset": (
                None
                if args.retention_dataset is None
                else str(args.retention_dataset)
            ),
            "requested_fraction": float(args.retention_fraction),
            "rows_drawn": int(retention_rows_drawn),
            "realized_fraction": (
                round(
                    float(retention_rows_drawn)
                    / max(
                        float(
                            optimizer_updates * int(args.batch_size)
                        ),
                        1.0,
                    ),
                    4,
                )
                if retention_rows_drawn
                else 0.0
            ),
        },
        "per_stage": per_stage,
        **reach_by_group,
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
    print(
        f"[sft] {optimizer_updates} optimizer updates over "
        f"{supervised_actions} sampled supervised actions "
        f"(sampler={args.sampler}, split_by={args.split_by})",
        flush=True,
    )
    if per_stage.get("val_by_stage_name"):
        print(
            f"[sft] val by stage: {per_stage['val_by_stage_name']}", flush=True
        )
    if reach_by_group.get("reachability_by_stage_name"):
        print(
            "[sft] reachability by stage: "
            f"{ {name: entry['reachable_fraction'] for name, entry in reach_by_group['reachability_by_stage_name'].items()} }",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
