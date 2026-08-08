"""Can the residual see that it is holding the object?

The trainable residual's only view of grasp state is the frozen 512-d fixed
random projection of SmolVLA's connector tokens (``residual_vision_features``).
If that feature cannot linearly decode "the pads are loaded", then the residual
cannot condition its z command on having grasped, and the best it can do after
closing is the average over grasped and un-grasped worlds -- which, given the
loaded plant's dead zone below a_z ~ 0.15-0.20, is a command that lifts nothing.

The probe that matters needs HARD NEGATIVES. Over a whole episode, "holding" is
trivially predictable from gripper height and time, so a probe fit on all steps
scores well by reading the pose and says nothing. So a configurable fraction of
episodes are driven to MISS: the oracle's grasp point is displaced laterally, the
gripper descends to the same height and closes on air. The decisive number is the
probe's accuracy on the matched subset -- fingers closed, end-effector at grasp
height -- where pose is uninformative and only "is there an object between the
pads" separates the classes.

Two questions, one capture. **Does it know it is holding the object** (a
classification probe, below) and **does it know where the object is** (a
regression probe on the EE->target XY vector). The 7.5M-step run made the second
the live one: the deterministic policy holds the correct height -- terminal
end-effector z 0.2006 m against grasp points at 0.19-0.21, ceiling-pinned rate
0.000 -- and misses by 0.40 m *horizontally*. Its action-to-target cosine is
0.11, the frozen prior's 0.05. So the policy is blind in XY, and the question is
whether the feature it reads carries the answer at all.

Three feature sets are probed, because they call for different fixes:

  proprio (6-d)      the control. If this alone decodes grasp state, the
                     question is moot and the residual already has the signal.
  vision 512-d       what the residual is actually fed.
  connector 30720-d  the un-projected tokens. If this decodes and the 512-d
                     projection does not, the fixed random projection is
                     throwing the signal away and the fix is the projection
                     (learn it, or widen it). If neither decodes, the frozen
                     connector does not represent grasp state at all and no head
                     reading it can help -- the encoder itself has to adapt,
                     which is what the action-expert LoRA would have to do.

Method follows the house probe in ``evaluate_cdpr_smolvla_mjwarp_videos``: dual
(kernel) ridge so the cost is O(N) in samples rather than in the 30720 feature
dims, an alpha sweep, and a label-shuffled control to calibrate chance. Splits
are BY EPISODE -- consecutive steps within an episode are near-duplicates, and a
step-level split leaks them across the fold and reports ~1.0 for anything.

Usage::

    python tools/audit/grasp_feature_probe.py --physics mjlab_mjwarp \\
        --episodes 40 --miss-fraction 0.5 --output runs/grasp_feature_probe
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence


def _configure_public_hf_models() -> None:
    """Strip an inherited credential before anything imports huggingface_hub.

    The Python mirror of ``configure_huggingface_public_models`` in
    ``scripts/huggingface_public_models.sh``, which every training launcher
    sources. Both ``lerobot/smolvla_base`` and its ``SmolVLM2-500M-Video-Instruct``
    backbone are public, but a stale HF_TOKEN in the remote shell (or one cached
    by huggingface_hub) turns the anonymous processor fetch into a 401
    RepositoryNotFoundError -- which reads as a missing model and is not one. The
    weights load first, so the failure lands well after the download bar, on
    AutoProcessor.from_pretrained.

    Called at import, before every other import here, because huggingface_hub
    reads HF_HUB_DISABLE_IMPLICIT_TOKEN into a module constant the first time it
    is imported; setting it later would be silently too late.

    Same escape hatch as the shell helper: RLVLA_HF_PUBLIC_MODELS_ONLY=0 leaves
    the environment alone for a genuinely private or gated checkpoint.
    """

    setting = os.environ.get("RLVLA_HF_PUBLIC_MODELS_ONLY", "1")
    if setting == "0":
        return
    if setting != "1":
        raise SystemExit("RLVLA_HF_PUBLIC_MODELS_ONLY must be 0 or 1.")
    removed = [
        name
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        if os.environ.pop(name, None)
    ]
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    if removed:
        print(f"[huggingface] ignoring inherited {', '.join(removed)}")


_configure_public_hf_models()

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (  # noqa: E402
    build_smolvla_state_tensor,
)

HARNESS = ROOT / "scripts" / "render_cdpr_task_reference_episodes.py"
DEFAULT_CATALOGS = (
    "robocasa_apple",
    "robocasa_tomato",
    "robocasa_orange",
    "robocasa_potato",
)


class _ConnectorReached(Exception):
    """Raised from the connector hook to abandon the rest of the forward."""


def _load_harness() -> Any:
    spec = importlib.util.spec_from_file_location(
        "cdpr_oracle_reference_harness", HARNESS
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the oracle harness at {HARNESS}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- data capture


class _Capture:
    """Records (features, grasp label) at every env step of every episode.

    Hooks ``_GraspShim.update``, which the harness calls once per env step with
    the backend in hand -- so this sees exactly the frame the policy would have
    seen and exactly the ``physical_grasp`` the trainer would have computed, with
    no second implementation of either.
    """

    def __init__(
        self,
        harness: Any,
        *,
        runtime: Any,
        vision_dim: int,
        state_dim: int,
        keep_connector: bool,
        capture_every: int = 1,
    ) -> None:
        self._harness = harness
        self._runtime = runtime
        self._vision_dim = int(vision_dim)
        self._state_dim = int(state_dim)
        self._keep_connector = bool(keep_connector)
        self._capture_every = max(1, int(capture_every))
        self._step_index = 0
        self._original = harness._GraspShim.update
        self.episode = -1
        self.missed = False
        self.rows: list[dict[str, Any]] = []

    def start_episode(self, *, missed: bool) -> None:
        self.episode += 1
        self.missed = bool(missed)
        # Restart the stride each episode so every episode is sampled the same
        # way, rather than the phase drifting with episode length.
        self._step_index = 0

    def install(self) -> None:
        capture = self

        def update(shim: Any, reset: Any, low_dim: Any, active: Any) -> Any:
            result = capture._original(shim, reset, low_dim, active)
            capture._record(shim, reset, low_dim, result)
            return result

        self._harness._GraspShim.update = update

    def remove(self) -> None:
        self._harness._GraspShim.update = self._original

    def _record(self, shim: Any, reset: Any, low_dim: Any, result: Any) -> None:
        """Store the frame and the labels. SmolVLA runs later, in batches.

        The first version ran one SmolVLA forward per env step at batch 2 --
        6240 forwards for 60 episodes, 35 minutes, with the GPU almost idle.
        Two things were wrong. Every step was featurized even though the probes
        subsample to ~24 per episode and throw ~75% away, and each forward was
        batch 2 because the harness clones two worlds and only world 0 is read.

        So capture is now cheap: render, keep the frame as fp16 on the host,
        record the labels, and skip the step entirely unless it is one of the
        ones that will be probed. The SmolVLA forwards happen afterwards in
        `compute_features`, batched.
        """

        import torch

        self._step_index += 1
        if (self._step_index - 1) % self._capture_every:
            return

        _low_dim, physical_grasp, diagnostics = result
        world = 0
        cameras = shim.backend.render_policy_cameras()
        state = build_smolvla_state_tensor(
            ee_position=low_dim.ee_position,
            ee_yaw=low_dim.ee_yaw,
            gripper_opening=low_dim.gripper_opening,
            object_positions=low_dim.object_positions,
            target_slots=reset.task_state.target_slots,
            state_dim=self._state_dim,
        )
        target_position = low_dim.object_positions[
            torch.arange(
                low_dim.object_positions.shape[0],
                device=low_dim.object_positions.device,
            ),
            reset.task_state.target_slots,
        ]
        row: dict[str, Any] = {
            "episode": int(self.episode),
            "missed": bool(self.missed),
            "physical_grasp": bool(physical_grasp[world].item()),
            "contact_loaded": bool(diagnostics["contact_loaded"][world].item()),
            "gripper_opening": float(low_dim.gripper_opening[world].item()),
            "ee_z": float(low_dim.ee_position[world, 2].item()),
            # Absolute object XY and end-effector XY, kept separately so the
            # raw location can be probed and so the regression control can swap
            # one episode's object for another's while keeping the trajectory.
            "target_xy": target_position[world, :2]
            .detach()
            .float()
            .cpu()
            .numpy()
            .copy(),
            "ee_xy": low_dim.ee_position[world, :2]
            .detach()
            .float()
            .cpu()
            .numpy()
            .copy(),
            "proprio": state[world].detach().float().cpu().numpy().copy(),
            "instruction": str(reset.instructions[world]),
            # fp16 on the host: at ~26 kept steps per episode these are a few
            # hundred MB, against several GB if every step were kept.
            "_overview": cameras.overview[world].detach().half().cpu(),
            "_wrist": cameras.wrist[world].detach().half().cpu(),
            "_state": state[world].detach().float().cpu(),
        }
        self.rows.append(row)

    def compute_features(
        self,
        batch_size: int,
        *,
        skip_action_expert: bool = True,
        verify_fast_path: bool = True,
    ) -> None:
        """Run SmolVLA over the captured frames in batches, then drop them.

        One forward per batch instead of one per env step.

        ``skip_action_expert`` abandons each forward the moment the connector
        has produced its tokens, by raising out of the forward hook. The
        connector runs in the prefix pass; everything after it is the
        flow-matching action expert integrating a chunk of actions this probe
        throws away. Aborting there is faithful by construction -- same model,
        same preprocessing, same module, same tensor, just stopped early -- and
        it needs no reimplementation of the model's internals, so it cannot
        drift when they change.

        The projected feature is then produced by the runtime's OWN
        ``_pool_vision``, which is the code the residual is fed by, rather than
        a copy of it here.

        ``verify_fast_path`` spends one full forward on the first batch to check
        the two paths agree before trusting the fast one. It costs a few seconds
        and it is the difference between believing this is equivalent and
        knowing it.
        """

        import torch

        if not self.rows:
            return
        module = self._runtime._vision_connector()
        device = getattr(self._runtime, "device", "cpu")
        total = len(self.rows)
        started = time.perf_counter()
        expected_calls: int | None = None
        verified = not verify_fast_path

        def run(overview, wrist, states, instructions, *, abort_after):
            """Return (captured_on_device, projected_or_None)."""
            captured: list[Any] = []

            def hook(_module, _inputs, output):
                captured.append(output.detach())
                if abort_after is not None and len(captured) >= abort_after:
                    raise _ConnectorReached
                return output

            handle = module.register_forward_hook(hook)
            projected = None
            try:
                with torch.inference_mode():
                    _prior, projected = (
                        self._runtime.sample_cdpr_chunks_and_vision_from_tensors(
                            primary_images=overview,
                            wrist_images=wrist,
                            states=states,
                            instructions=instructions,
                            vision_dim=self._vision_dim,
                            microbatch_size=0,
                        )
                    )
            except _ConnectorReached:
                projected = None
            finally:
                handle.remove()
            return captured, projected

        for begin in range(0, total, max(1, int(batch_size))):
            chunk = self.rows[begin : begin + max(1, int(batch_size))]
            overview = torch.stack([r.pop("_overview") for r in chunk]).to(
                device=device, dtype=torch.float32
            )
            wrist = torch.stack([r.pop("_wrist") for r in chunk]).to(
                device=device, dtype=torch.float32
            )
            states = torch.stack([r.pop("_state") for r in chunk]).to(
                device=device
            )
            instructions = [r["instruction"] for r in chunk]

            if expected_calls is None or not skip_action_expert:
                # Full forward: also tells us how many times the connector fires
                # per batch, which is what the fast path aborts on.
                captured, projected = run(
                    overview, wrist, states, instructions, abort_after=None
                )
                expected_calls = len(captured)
                vision = projected
                if skip_action_expert and not verified:
                    fast_captured, _ = run(
                        overview,
                        wrist,
                        states,
                        instructions,
                        abort_after=expected_calls,
                    )
                    fast_vision = self._runtime._pool_vision(
                        fast_captured, self._vision_dim
                    )
                    delta = float(
                        (fast_vision.float() - vision.float()).abs().max().item()
                    )
                    if delta > 1e-3:
                        raise RuntimeError(
                            "Fast path disagrees with the full forward "
                            f"(max |delta| = {delta:.3e}). Rerun with "
                            "--no-skip-action-expert."
                        )
                    print(
                        f"[probe] fast path verified against the full forward "
                        f"(max |delta| = {delta:.2e}); skipping the action "
                        f"expert from here"
                    )
                    verified = True
            else:
                captured, _ = run(
                    overview,
                    wrist,
                    states,
                    instructions,
                    abort_after=expected_calls,
                )
                vision = self._runtime._pool_vision(captured, self._vision_dim)

            vision_cpu = vision.detach().float().cpu().numpy()
            for index, row in enumerate(chunk):
                row["vision"] = vision_cpu[index].copy()
            if self._keep_connector and captured:
                # Overview + wrist only, matching _pool_vision; never the masked
                # aux. Kept as [cameras*tokens, channels] rather than flattened:
                # the 16 tokens per camera are a ~4x4 spatial grid, so WHERE the
                # object is lives in WHICH token carries it, and a reduction
                # that respects that axis can keep position where one that
                # flattens first cannot.
                cams = captured[: max(1, min(2, len(captured)))]
                stacked = np.concatenate(
                    [cam.float().cpu().numpy() for cam in cams], axis=1
                ).astype(np.float32)
                for index, row in enumerate(chunk):
                    row["connector"] = stacked[index].copy()
            print(
                f"[probe] features {min(begin + len(chunk), total)}/{total}",
                end="\r",
                flush=True,
            )
        elapsed = time.perf_counter() - started
        print(
            f"[probe] featurized {total} steps in {elapsed:.0f}s "
            f"({total / max(elapsed, 1e-6):.1f}/s)"
        )


def _block_shuffled_labels(
    labels: np.ndarray, episodes: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    """Permute labels BETWEEN episodes, keeping them constant within one.

    A step-level shuffle is the wrong null here. Grasp state is near-constant
    inside an episode, so any feature that merely identifies the episode -- the
    object's colour, where it sits on the desk, the exact end-effector pose --
    predicts the label. A step-level shuffle destroys that block structure, so
    the control comes out at chance and the identity shortcut passes as signal.
    Measured on a stub run: proprioception alone scored 0.967 against a
    step-shuffled control of 0.522, which is entirely episode identity.

    This permutes which episode carries which label and leaves the block
    structure intact, so beating it means predicting WHICH episodes grasped,
    not merely which episode a step came from.
    """

    unique = np.unique(episodes)
    per_episode = {
        episode: float(labels[episodes == episode].mean() > 0.5)
        for episode in unique
    }
    permuted = rng.permutation([per_episode[e] for e in unique])
    mapping = dict(zip(unique.tolist(), permuted.tolist()))
    return np.array([mapping[episode] for episode in episodes])


def _dual_ridge_scores(
    features: np.ndarray,
    labels: np.ndarray,
    episodes: np.ndarray,
    *,
    seed: int = 0,
    folds: int = 5,
) -> dict[str, float]:
    """K-fold linear probe, folded BY EPISODE, against a block-shuffled control.

    Dual (kernel) ridge on +-1 labels: cost is O(N^2) in samples and independent
    of feature width, which is what makes the 30720-d connector affordable.

    Two things this deliberately does not do. It does not split at step level --
    consecutive steps are near-duplicates and would leak across the fold. And it
    does not use a step-level shuffled control -- see _block_shuffled_labels.
    The alpha sweep picks the least regularization whose control is still at
    chance, so D >> N overfitting cannot pass as signal.

    The honest sample size is the number of EPISODES, not steps, because the
    label barely varies within one. ``episodes`` is reported alongside so the
    result is read with that in mind.
    """

    unique = np.unique(episodes)
    nan = {
        "accuracy": float("nan"),
        "control": float("nan"),
        "margin": float("nan"),
        "spread": float("nan"),
        "alpha": 0.0,
        "episodes": int(unique.size),
    }
    if unique.size < 6 or np.unique(labels > 0.5).size < 2:
        return nan
    rng = np.random.RandomState(seed)
    signed = np.where(labels > 0.5, 1.0, -1.0)
    control_signed = np.where(
        _block_shuffled_labels(labels, episodes, rng) > 0.5, 1.0, -1.0
    )

    order = rng.permutation(unique.size)
    fold_count = max(2, min(int(folds), unique.size))
    assignments = np.array_split(unique[order], fold_count)

    # Precompute each fold ONCE. The Gram depends only on the features and the
    # fold's training statistics -- not on alpha and not on which label set is
    # being scored -- so building it inside the sweep rebuilt a 30720-wide
    # standardized copy ninety times (9 alphas x {real, control} x 5 folds) for
    # a result that never changed.
    #
    # Eigendecomposing K_train once then makes the whole alpha sweep almost
    # free: with K = Q diag(lam) Q^T, the dual solution is
    # Q diag(1/(lam + alpha)) Q^T y, so every extra alpha is a division and two
    # small matrix-vector products rather than another O(n^3) solve.
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    features32 = np.ascontiguousarray(features, dtype=np.float32)
    for held in assignments:
        test = np.isin(episodes, held)
        train = ~test
        if train.sum() < 2 or test.sum() < 2:
            continue
        centred = features32 - features32[train].mean(axis=0, keepdims=True)
        centred /= np.maximum(
            features32[train].std(axis=0, keepdims=True), 1.0e-8
        )
        # float32 for the outer product (the dominant cost), float64 for the
        # decomposition: a probe does not need more precision than that, and
        # the matmul is where all the time goes.
        gram = (centred @ centred.T).astype(np.float64) + 1.0
        eigenvalues, vectors = np.linalg.eigh(gram[np.ix_(train, train)])
        prepared.append(
            (train, test, vectors, eigenvalues, gram[np.ix_(test, train)] @ vectors)
        )
    del features32

    def score(target: np.ndarray, alpha: float) -> tuple[float, float]:
        per_fold: list[float] = []
        for train, test, vectors, eigenvalues, projected in prepared:
            if np.unique(target[test] > 0).size < 2:
                # A fold with one class cannot produce a balanced accuracy that
                # means anything; skipping is honest, inventing 0.5 is not.
                continue
            weights = (vectors.T @ target[train]) / (eigenvalues + alpha)
            predicted = (projected @ weights) > 0.0
            truth = target[test] > 0
            halves = [
                float((predicted[truth == value] == value).mean())
                for value in (True, False)
                if (truth == value).sum()
            ]
            per_fold.append(float(np.mean(halves)))
        if not per_fold:
            return float("nan"), float("nan")
        return float(np.mean(per_fold)), float(np.std(per_fold))

    best: tuple[float, float, float, float] | None = None
    fallback: tuple[float, float, float, float] | None = None
    for alpha in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        control, _ = score(control_signed, alpha)
        real, spread = score(signed, alpha)
        if np.isnan(control) or np.isnan(real):
            continue
        if fallback is None or abs(control - 0.5) < abs(fallback[1] - 0.5):
            fallback = (real, control, spread, alpha)
        if abs(control - 0.5) <= 0.05:
            best = (real, control, spread, alpha)
            break
    chosen = best or fallback
    if chosen is None:
        return nan
    return {
        "accuracy": chosen[0],
        "control": chosen[1],
        # The margin over the block-shuffled control is the number that means
        # something; the raw accuracy on its own does not.
        "margin": chosen[0] - chosen[1],
        "spread": chosen[2],
        "alpha": chosen[3],
        "episodes": int(unique.size),
    }


def _dual_ridge_r2(
    features: np.ndarray,
    targets: np.ndarray,
    control_targets: np.ndarray,
    episodes: np.ndarray,
    *,
    seed: int = 0,
    folds: int = 5,
) -> dict[str, float]:
    """Held-out R^2 of a linear map feature -> 2-D target, folded by episode.

    Same dual-ridge machinery as the classification probe, with two differences
    that matter for a vector target.

    The control is a TARGET SWAP, not a label shuffle: each episode keeps its
    real end-effector trajectory and is given another episode's object. That
    destroys the feature-to-object association while preserving everything about
    the motion, which a plain shuffle would also destroy and so score too easily.

    ``direction_cosine`` is reported next to R^2 because it is the quantity that
    decides whether a policy can servo. A probe can carry real R^2 while
    pointing the wrong way, and only the direction moves the end-effector toward
    the object.
    """

    unique = np.unique(episodes)
    nan = {
        "r2": float("nan"),
        "control_r2": float("nan"),
        "direction_cosine": float("nan"),
        "direction_spread": float("nan"),
        "alpha": 0.0,
        "episodes": int(unique.size),
    }
    if unique.size < 6:
        return nan
    rng = np.random.RandomState(seed)
    order = rng.permutation(unique.size)
    fold_count = max(2, min(int(folds), unique.size))
    assignments = np.array_split(unique[order], fold_count)

    prepared = []
    features32 = np.ascontiguousarray(features, dtype=np.float32)
    for held in assignments:
        test = np.isin(episodes, held)
        train = ~test
        if train.sum() < 2 or test.sum() < 2:
            continue
        centred = features32 - features32[train].mean(axis=0, keepdims=True)
        centred /= np.maximum(
            features32[train].std(axis=0, keepdims=True), 1.0e-8
        )
        gram = (centred @ centred.T).astype(np.float64) + 1.0
        eigenvalues, vectors = np.linalg.eigh(gram[np.ix_(train, train)])
        prepared.append(
            (train, test, vectors, eigenvalues, gram[np.ix_(test, train)] @ vectors)
        )
    del features32
    if not prepared:
        return nan

    def score(values: np.ndarray, alpha: float) -> tuple[float, float]:
        r2s: list[float] = []
        cosines: list[float] = []
        for train, test, vectors, eigenvalues, projected in prepared:
            centre = values[train].mean(axis=0, keepdims=True)
            weights = (vectors.T @ (values[train] - centre)) / (
                eigenvalues + alpha
            )[:, None]
            predicted = projected @ weights + centre
            truth = values[test]
            residual = ((truth - predicted) ** 2).sum()
            total = ((truth - centre) ** 2).sum()
            r2s.append(float(1.0 - residual / max(total, 1.0e-12)))
            norms = np.linalg.norm(predicted, axis=1) * np.linalg.norm(
                truth, axis=1
            )
            keep = norms > 1.0e-9
            if keep.any():
                cosines.append(
                    float(
                        ((predicted[keep] * truth[keep]).sum(axis=1) / norms[keep]).mean()
                    )
                )
        if not r2s:
            return float("nan"), float("nan"), float("nan")
        return (
            float(np.mean(r2s)),
            float(np.mean(cosines) if cosines else np.nan),
            float(np.std(cosines) if cosines else np.nan),
        )

    best = None
    fallback = None
    for alpha in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        control, _, _ = score(control_targets, alpha)
        real, cosine, spread = score(targets, alpha)
        if np.isnan(control) or np.isnan(real):
            continue
        if fallback is None or abs(control) < abs(fallback[1]):
            fallback = (real, control, cosine, spread, alpha)
        if abs(control) <= 0.05:
            best = (real, control, cosine, spread, alpha)
            break
    chosen = best or fallback
    if chosen is None:
        return nan
    return {
        "r2": chosen[0],
        "control_r2": chosen[1],
        "direction_cosine": chosen[2],
        # Across folds. A ranking whose gaps are inside this is not a ranking.
        "direction_spread": chosen[3],
        "alpha": chosen[4],
        "episodes": int(unique.size),
    }


def _mlp_r2(
    features: np.ndarray,
    targets: np.ndarray,
    control_targets: np.ndarray,
    episodes: np.ndarray,
    *,
    seed: int = 0,
    folds: int = 5,
    hidden: int = 1024,
    epochs: int = 2000,
    weight_decay: float = 1.0e-4,
) -> dict[str, float]:
    """Held-out R^2 of a NONLINEAR map feature -> 2-D target, folded by episode.

    Same folds, same episode-swap control and same reported quantities as
    ``_dual_ridge_r2``, so the two are directly comparable; the only difference
    is the hypothesis class. That matters because every localization number in
    this campaign is from a LINEAR probe, and the thing that has to learn the
    map is a two-hidden-layer MLP -- so "+0.389 linear" and "the residual can
    learn it" are different claims, and a 5.2M-step run has now failed to move
    the residual's aim off 0.055 while being fed that feature.

    The architecture mirrors the residual's own first three layers (two hidden
    layers of ``hidden``, ReLU) so a failure here is a statement about what the
    residual could learn, not about a probe that was too small.

    The control is trained IDENTICALLY -- same width, same epochs, same
    schedule -- on the swapped targets. An MLP with 1024 hidden units on a few
    thousand samples can memorize, and a control that scores well is how that
    shows up. Read the control before the score.
    """

    import torch

    unique = np.unique(episodes)
    nan = {
        "r2": float("nan"),
        "control_r2": float("nan"),
        "direction_cosine": float("nan"),
        "direction_spread": float("nan"),
        "train_r2": float("nan"),
        "episodes": int(unique.size),
    }
    if unique.size < 6:
        return nan
    rng = np.random.RandomState(seed)
    order = rng.permutation(unique.size)
    fold_count = max(2, min(int(folds), unique.size))
    assignments = np.array_split(unique[order], fold_count)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # One copy, on the device, and every fold's standardization happens there.
    #
    # The host version of this OOM-killed a 180-episode capture: the connector
    # is 4320 x 30720 float32 (~530 MB), and building `(train - mean) / scale`
    # per fold made another one of those per fold on top of the capture, the
    # stacked copy and the ridge probe's N x N gram. A 44 GB card does not
    # notice 530 MB; host RAM did.
    features_t = torch.as_tensor(
        np.ascontiguousarray(features, dtype=np.float32), device=device
    )

    def fit(values: np.ndarray) -> tuple[list[float], list[float], list[float]]:
        r2s: list[float] = []
        train_r2s: list[float] = []
        cosines: list[float] = []
        for index, held in enumerate(assignments):
            test = np.isin(episodes, held)
            train = ~test
            if train.sum() < 8 or test.sum() < 2:
                continue
            train_idx = torch.as_tensor(
                np.flatnonzero(train), dtype=torch.long, device=device
            )
            test_idx = torch.as_tensor(
                np.flatnonzero(test), dtype=torch.long, device=device
            )
            raw_train = features_t.index_select(0, train_idx)
            mean = raw_train.mean(dim=0, keepdim=True)
            scale = raw_train.std(dim=0, keepdim=True).clamp_min(1e-8)
            x_train = (raw_train - mean) / scale
            del raw_train
            x_test = (features_t.index_select(0, test_idx) - mean) / scale
            centre = values[train].mean(axis=0, keepdims=True)
            y_train = torch.as_tensor(
                (values[train] - centre).astype(np.float32), device=device
            )
            y_test_np = values[test]

            torch.manual_seed(int(seed) + index)
            model = torch.nn.Sequential(
                torch.nn.Linear(x_train.shape[1], hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, 2),
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1.0e-3, weight_decay=weight_decay
            )
            model.train()
            for _ in range(int(epochs)):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(x_train), y_train)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                predicted = model(x_test).cpu().numpy() + centre
                fitted = model(x_train).cpu().numpy() + centre
            del x_train, x_test, model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

            truth = y_test_np
            residual = ((truth - predicted) ** 2).sum()
            total = ((truth - centre) ** 2).sum()
            r2s.append(float(1.0 - residual / max(total, 1.0e-12)))
            train_truth = values[train]
            train_residual = ((train_truth - fitted) ** 2).sum()
            train_total = ((train_truth - centre) ** 2).sum()
            train_r2s.append(
                float(1.0 - train_residual / max(train_total, 1.0e-12))
            )
            norms = np.linalg.norm(predicted, axis=1) * np.linalg.norm(
                truth, axis=1
            )
            keep = norms > 1.0e-9
            if keep.any():
                cosines.append(
                    float(
                        (
                            (predicted[keep] * truth[keep]).sum(axis=1)
                            / norms[keep]
                        ).mean()
                    )
                )
        return r2s, train_r2s, cosines

    real, train_real, cosines = fit(targets)
    control, _, _ = fit(control_targets)
    del features_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not real:
        return nan
    return {
        "r2": float(np.mean(real)),
        "control_r2": float(np.mean(control)) if control else float("nan"),
        "direction_cosine": float(np.mean(cosines)) if cosines else float("nan"),
        "direction_spread": float(np.std(cosines)) if cosines else float("nan"),
        # Held-out far below train is the MLP memorizing rather than learning
        # the map; the control catches it too, and both are printed.
        "train_r2": float(np.mean(train_real)),
        "episodes": int(unique.size),
    }


def _swap_targets_between_episodes(
    rows: Sequence[dict[str, Any]], rng: np.random.RandomState
) -> np.ndarray:
    """Each episode keeps its trajectory and gets another episode's object."""

    episodes = sorted({row["episode"] for row in rows})
    per_episode = {
        episode: next(
            row["target_xy"] for row in rows if row["episode"] == episode
        )
        for episode in episodes
    }
    permuted = rng.permutation(len(episodes))
    mapping = {
        episode: per_episode[episodes[permuted[index]]]
        for index, episode in enumerate(episodes)
    }
    return np.stack([mapping[row["episode"]] for row in rows])


def _stack(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    stacked = np.stack([np.asarray(row[key], dtype=np.float32) for row in rows])
    # The connector is stored [N, cameras*tokens, channels]; probes want it flat.
    return stacked.reshape(stacked.shape[0], -1) if stacked.ndim > 2 else stacked


def _connector_reductions(
    tokens: np.ndarray, seed: int, shipped: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Candidate replacements for the fixed random projection.

    ``tokens`` is [N, cameras*tokens, channels]. The question these answer is
    whether the projection fails because it is too NARROW or because it is
    structurally wrong.

    A fixed random projection retains only ``d_out/d_in`` of any linearly
    decodable signal -- computed exactly, 512 of 30720 is 1.7%, and widening to
    8192 still only reaches 27%. So width alone cannot fix it. The spatial
    reductions instead keep the token axis, where position actually lives, and
    reduce only channels.
    """

    rng = np.random.RandomState(seed)
    count, places, channels = tokens.shape
    flat = tokens.reshape(count, -1)

    def random_projection(width: int) -> np.ndarray:
        matrix = rng.randn(flat.shape[1], width).astype(np.float32)
        matrix /= np.sqrt(flat.shape[1])
        return flat @ matrix

    def per_token_projection(width: int) -> np.ndarray:
        # One channel reduction SHARED across tokens, applied per token, so the
        # spatial grid survives intact and only the 960 channels are mixed.
        matrix = rng.randn(channels, width).astype(np.float32)
        matrix /= np.sqrt(channels)
        return (tokens @ matrix).reshape(count, -1)

    reductions: dict[str, np.ndarray] = {}
    if shipped is not None:
        # The projection the residual is ACTUALLY fed, not a re-draw of the same
        # kind. At 1.7% retention the outcome depends heavily on which matrix
        # was drawn, so a re-draw and the real one can disagree by more than
        # their fold spreads -- which is itself an argument against a random
        # projection: its usefulness is a lottery ticket, fixed at seed time.
        reductions["SHIPPED (real _pool_vision)"] = shipped
    reductions.update({
        "random 512 (re-draw)": random_projection(512),
        "random 2048": random_projection(2048),
        f"channel-mean per token ({places})": tokens.mean(axis=2),
        f"per-token random x4 ({places * 4})": per_token_projection(4),
        f"per-token random x8 ({places * 8})": per_token_projection(8),
        # 32 places x 16 = 512: the SAME width the residual takes today, so this
        # one is a drop-in swap that keeps every tensor shape and lets a warm
        # start still load.
        f"per-token random x16 ({places * 16}) DROP-IN": per_token_projection(16),
        f"per-token random x32 ({places * 32})": per_token_projection(32),
        f"un-projected ({flat.shape[1]})": flat,
    })
    return reductions


def _report_reductions(
    rows: Sequence[dict[str, Any]],
    *,
    max_steps_per_episode: int,
    seed: int,
) -> dict[str, Any]:
    """Which reduction of the connector keeps the object's position?"""

    rows = _subsample_per_episode(rows, int(max_steps_per_episode))
    episodes = np.array([row["episode"] for row in rows])
    if len(set(episodes.tolist())) < 6 or "connector" not in rows[0]:
        return {}
    tokens = np.stack(
        [np.asarray(row["connector"], dtype=np.float32) for row in rows]
    )
    target = _stack(rows, "target_xy")
    swapped = _swap_targets_between_episodes(
        rows, np.random.RandomState(int(seed))
    )
    print("\n" + "=" * 74)
    print("What should replace the fixed random projection?")
    print("=" * 74)
    print(
        f"\n  approach steps {len(rows)}, "
        f"episodes {len(set(episodes.tolist()))}\n"
    )
    out: dict[str, Any] = {}
    shipped = _stack(rows, "vision") if "vision" in rows[0] else None
    for label, reduced in _connector_reductions(
        tokens, int(seed), shipped
    ).items():
        scores = _dual_ridge_r2(
            reduced, target, swapped, episodes, seed=int(seed)
        )
        out[label] = scores
        if np.isnan(scores["r2"]):
            continue
        print(
            f"    {label:<32} R2 {scores['r2']:+.3f}"
            f"   (control {scores['control_r2']:+.3f})"
            f"   dir cos {scores['direction_cosine']:+.3f}"
            f" +-{scores['direction_spread']:.3f}"
        )
    print(
        "\n  If `random 2048` is no better than `random 512`, width is not the\n"
        "  problem and the projection is structurally wrong -- a fixed random\n"
        "  map keeps only d_out/d_in of any linear signal, 1.7% at the shipped\n"
        "  512 of 30720. If a per-token or channel-mean reduction approaches\n"
        "  `un-projected`, the fix is to stop flattening the spatial grid before\n"
        "  reducing, which costs far fewer dimensions than widening.\n"
        "\n"
        "  Read the +- spread before the ranking: it is the standard deviation\n"
        "  across folds, and gaps smaller than it are not real. The DROP-IN row\n"
        "  is 512 wide, exactly what the residual takes today, so adopting it\n"
        "  changes no tensor shape and a warm start still loads.\n"
        "\n"
        "  SHIPPED is the real projection; `random 512 (re-draw)` is another\n"
        "  draw of the same kind. If those two disagree by more than their\n"
        "  spreads, that is the point: at 1.7%% retention the random projection\n"
        "  is a lottery fixed at seed time, and a per-token reduction reaches\n"
        "  the un-projected ceiling at the same width without the gamble."
    )
    return out


def _subsample_per_episode(
    rows: Sequence[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Keep at most ``limit`` evenly spaced steps per episode.

    Consecutive steps are near-duplicates and the label barely varies inside an
    episode, so the hundredth step of an episode adds essentially nothing to a
    probe whose effective sample size is the episode count. It does, however,
    add its full share to an O(N^2) Gram -- at 104 steps x 60 episodes the
    30720-wide connector arm builds a 6240x6240 kernel to answer a question
    with 60 independent units in it.

    Evenly spaced rather than the first ``limit``: the early steps of an episode
    are all approach, and taking a prefix would drop the grasp entirely.
    """

    if limit <= 0:
        return list(rows)
    kept: list[dict[str, Any]] = []
    for episode in sorted({row["episode"] for row in rows}):
        block = [row for row in rows if row["episode"] == episode]
        if len(block) <= limit:
            kept.extend(block)
            continue
        indices = np.linspace(0, len(block) - 1, limit).round().astype(int)
        kept.extend(block[index] for index in sorted(set(indices.tolist())))
    return kept


def _report_subset(
    name: str,
    rows: Sequence[dict[str, Any]],
    label_key: str,
    *,
    keep_connector: bool,
    max_steps_per_episode: int = 0,
) -> dict[str, Any]:
    rows = _subsample_per_episode(rows, int(max_steps_per_episode))
    labels = np.array([1.0 if row[label_key] else 0.0 for row in rows])
    episodes = np.array([row["episode"] for row in rows])
    positive = float(labels.mean()) if labels.size else float("nan")
    print(f"\n  {name}")
    print(
        f"    steps {len(rows)}   episodes {len(set(episodes.tolist()))}   "
        f"positive rate {positive:.3f}"
    )
    if labels.size == 0 or np.unique(labels).size < 2:
        print("    only one class present -- probe not identifiable here")
        return {"steps": len(rows), "positive_rate": positive}
    out: dict[str, Any] = {"steps": len(rows), "positive_rate": positive}
    feature_sets = [("proprio (6)", "proprio"), ("vision (512)", "vision")]
    if keep_connector and "connector" in rows[0]:
        feature_sets.append(
            (f"connector ({rows[0]['connector'].size})", "connector")
        )
    for label, key in feature_sets:
        scores = _dual_ridge_scores(_stack(rows, key), labels, episodes)
        out[key] = scores
        if np.isnan(scores["accuracy"]):
            print(f"    {label:<24} not identifiable (too few episodes/classes)")
            continue
        print(
            f"    {label:<24} acc {scores['accuracy']:.3f} +-{scores['spread']:.3f}"
            f"   control {scores['control']:.3f}"
            f"   MARGIN {scores['margin']:+.3f}   (alpha {scores['alpha']:g})"
        )
    return out


# ---------------------------------------------------------------------- main


def _report_localization(
    name: str,
    rows: Sequence[dict[str, Any]],
    *,
    keep_connector: bool,
    max_steps_per_episode: int,
    seed: int,
    pooling: str = "flat_random",
    mlp: bool = False,
    mlp_hidden: int = 1024,
    mlp_epochs: int = 2000,
    mlp_max_features: int = 4096,
) -> dict[str, Any]:
    """Can the feature say WHERE the object is, relative to the gripper?"""

    rows = _subsample_per_episode(rows, int(max_steps_per_episode))
    episodes = np.array([row["episode"] for row in rows])
    print(f"\n  {name}")
    print(f"    steps {len(rows)}   episodes {len(set(episodes.tolist()))}")
    if len(rows) < 12 or len(set(episodes.tolist())) < 6:
        print("    too few episodes to fold")
        return {"steps": len(rows)}
    ee = _stack(rows, "ee_xy")
    target = _stack(rows, "target_xy")
    episode_count = len(set(episodes.tolist()))
    if mlp and episode_count < 100:
        # The object is fixed within an episode, so its steps are one training
        # point wearing many hats and the effective sample size is the EPISODE
        # count. On a synthetic task an MLP scores 0.55 at 30 episodes with a
        # train R2 of 1.00 -- pure memorization -- and 0.94 at 120. A negative
        # result from too few episodes says nothing about the feature.
        print(
            f"    WARNING: {episode_count} episodes is too few for the MLP "
            "probe; it will memorize. Re-run with --episodes 150 or more "
            "before reading the MLP rows as evidence."
        )
    rng = np.random.RandomState(int(seed))
    swapped = _swap_targets_between_episodes(rows, rng)
    out: dict[str, Any] = {"steps": len(rows)}
    # The target is the object's ABSOLUTE XY, not the EE->target vector, and
    # that is deliberate. The swap control cannot null the relative vector: the
    # real label (target - ee) and the control (swapped - ee) share the -ee
    # term, ee is in proprioception, so a model predicts a chunk of both and the
    # alpha sweep regularizes the real signal away chasing a control that will
    # not fall. Verified on synthetic data -- a feature carrying the object
    # position scores R2 0.61 / dir cos 0.89 on absolute XY and 0.008 on the
    # relative vector, which is the control failing, not the feature.
    #
    # Absolute XY answers the question anyway: if the feature does not say where
    # the object is, nothing downstream can servo to it.
    vision = _stack(rows, "vision")
    proprio = _stack(rows, "proprio")
    matrices = {
        "proprio (6)": proprio,
        f"vision ({vision.shape[1]})": vision,
        "proprio+vision": np.concatenate([proprio, vision], axis=1),
    }
    if str(pooling) == "dual_random" and vision.shape[1] % 2 == 0:
        # The two halves, scored separately on the SAME episodes and folds.
        # This is the comparison the campaign has only ever made across
        # separate captures: flat_random keeps its columns first, so the split
        # is exactly where the loader puts it.
        half = vision.shape[1] // 2
        matrices[f"  ^ flat half ({half})"] = vision[:, :half]
        matrices[f"  ^ per_token half ({half})"] = vision[:, half:]
    if keep_connector and "connector" in rows[0]:
        matrices[f"connector ({rows[0]['connector'].size})"] = _stack(
            rows, "connector"
        )
    def score_block(suffix: str, values: np.ndarray, control: np.ndarray) -> None:
        for label, matrix in matrices.items():
            scores = _dual_ridge_r2(
                matrix, values, control, episodes, seed=int(seed)
            )
            out[f"{label}{suffix}"] = scores
            if np.isnan(scores["r2"]):
                continue
            print(
                f"    {label:<26} linear  R2 {scores['r2']:+.3f}"
                f"   (control {scores['control_r2']:+.3f})"
                f"   dir cos {scores['direction_cosine']:+.3f}"
                f" +-{scores['direction_spread']:.3f}"
            )
            if not mlp:
                continue
            if matrix.shape[1] > int(mlp_max_features):
                print(
                    f"    {'':<26} MLP     skipped: {matrix.shape[1]} features "
                    f"> --mlp-max-features {int(mlp_max_features)}"
                )
                continue
            nonlinear = _mlp_r2(
                matrix,
                values,
                control,
                episodes,
                seed=int(seed),
                hidden=int(mlp_hidden),
                epochs=int(mlp_epochs),
            )
            out[f"{label}{suffix} [mlp]"] = nonlinear
            if np.isnan(nonlinear["r2"]):
                continue
            print(
                f"    {'':<26} MLP     R2 {nonlinear['r2']:+.3f}"
                f"   (control {nonlinear['control_r2']:+.3f})"
                f"   dir cos {nonlinear['direction_cosine']:+.3f}"
                f" +-{nonlinear['direction_spread']:.3f}"
                f"   [train R2 {nonlinear['train_r2']:+.3f}]"
            )

    print("    -- absolute object XY --")
    score_block("", target, swapped)
    # The RELATIVE vector, and it is the one that decides whether a policy can
    # servo. Absolute XY is decodable from the end-effector pose alone whenever
    # the start cap is small -- the object is within ~5 cm of a gripper that
    # roams +-0.25 m -- so "the object is where the gripper is" scores ~0.9 R2
    # while carrying nothing about WHICH WAY to move. A linear probe is held
    # back from that shortcut by the alpha sweep; an MLP takes it every time,
    # which is why proprioception (six numbers containing no object at all)
    # scored 0.847 direction cosine here, ABOVE the un-projected connector.
    #
    # On (object - ee) that shortcut is worthless: the reset places the object
    # in a roughly uniform direction within the cap, so the end-effector pose
    # predicts the offset at ~0. Proprioception is therefore the interpretable
    # null on these rows -- if it does not collapse, the block is not measuring
    # what it claims and nothing below it can be read.
    print("\n    -- relative (object - ee): the servo direction --")
    score_block("_relative", target - ee, swapped - ee)
    # How much of any score is just the start cap. Starts are capped 5 cm from
    # the object, so the end-effector pose alone predicts the object position
    # well -- proprio is not a null here, it is the number the vision rows have
    # to beat.
    out["ee_to_object_median_m"] = float(
        np.median(np.linalg.norm(target - ee, axis=1))
    )
    print(
        f"    (median EE-to-object distance over these steps: "
        f"{out['ee_to_object_median_m']*1000:.0f} mm -- proprio scores highly "
        f"whenever this is small)"
    )
    return out


def _probe_and_report(
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    *,
    keep_connector: bool,
    features_path: Path | None,
) -> int:
    """Save the captured features, then probe them and print the verdict."""

    label_key = str(args.label)
    args.output.mkdir(parents=True, exist_ok=True)

    # Save BEFORE probing. The rollout is the expensive half -- a SmolVLA
    # forward per env step -- and it must not be lost to a probe that is slow,
    # runs out of memory, or is simply interrupted. --from-features re-probes
    # this file without touching the simulator or the VLA.
    if features_path is not None:
        np.savez_compressed(
            features_path,
            vision=_stack(rows, "vision"),
            proprio=_stack(rows, "proprio"),
            physical_grasp=np.array(
                [1.0 if row["physical_grasp"] else 0.0 for row in rows]
            ),
            contact_loaded=np.array(
                [1.0 if row["contact_loaded"] else 0.0 for row in rows]
            ),
            episode=np.array([row["episode"] for row in rows]),
            missed=np.array([1.0 if row["missed"] else 0.0 for row in rows]),
            gripper_opening=np.array(
                [row["gripper_opening"] for row in rows]
            ),
            ee_z=np.array([row["ee_z"] for row in rows]),
            target_xy=_stack(rows, "target_xy"),
            ee_xy=_stack(rows, "ee_xy"),
            **(
                {
                    "connector": np.stack(
                        [
                            np.asarray(row["connector"], dtype=np.float32)
                            for row in rows
                        ]
                    )
                }
                if "connector" in rows[0]
                else {}
            ),
        )
        print(f"\n[probe] wrote {features_path} before probing")
        if getattr(args, "capture_only", False):
            print(
                "[probe] --capture-only: stopping here. Merge the shards with\n"
                f"        --from-features <shard>/features.npz ..."
            )
            return 0

    print("\n" + "=" * 74)
    print(f"Linear probe: frozen features -> {label_key}")
    print("=" * 74)
    all_episodes = {row["episode"] for row in rows}
    grasped_episodes = {row["episode"] for row in rows if row[label_key]}
    missed_episodes = {row["episode"] for row in rows if row["missed"]}
    print(
        f"\n  {len(rows)} steps over {len(all_episodes)} episodes; "
        f"{len(grasped_episodes)} ever reached {label_key}; "
        f"{len(missed_episodes)} were driven to close on air"
    )

    results: dict[str, Any] = {"label": label_key}
    results["all_steps"] = _report_subset(
        "ALL STEPS  (pose is informative here -- expect a high score that means "
        "little)",
        rows,
        label_key,
        keep_connector=keep_connector,
        max_steps_per_episode=int(args.max_steps_per_episode),
    )
    matched = [
        row
        for row in rows
        if row["gripper_opening"] <= float(args.closed_gripper_below)
    ]
    results["matched_closed_gripper"] = _report_subset(
        "MATCHED SUBSET  (fingers closed -- object present or not is what is "
        "left to decode)",
        matched,
        label_key,
        keep_connector=keep_connector,
        max_steps_per_episode=int(args.max_steps_per_episode),
    )

    # Localization. The grasp probe above answers "does it know it is holding";
    # this answers "does it know where to go", which the 7.5M-step run showed to
    # be the live failure: the deterministic policy holds the correct height and
    # misses by 0.40 m horizontally.
    print("\n" + "=" * 74)
    print("Probe: frozen features -> object location (XY)")
    print("=" * 74)
    approach = [
        row
        for row in rows
        if not row["physical_grasp"] and not row["contact_loaded"]
    ]
    results["localization_approach"] = _report_localization(
        "APPROACH STEPS  (free space, before any contact -- where servoing "
        "happens)",
        approach,
        keep_connector=keep_connector,
        max_steps_per_episode=int(args.max_steps_per_episode),
        seed=int(args.seed),
        pooling=str(getattr(args, "vision_pooling", "flat_random")),
        mlp=bool(getattr(args, "mlp_probe", False)),
        mlp_hidden=int(getattr(args, "mlp_hidden", 1024)),
        mlp_epochs=int(getattr(args, "mlp_epochs", 2000)),
        mlp_max_features=int(getattr(args, "mlp_max_features", 4096)),
    )
    results["localization_all"] = _report_localization(
        "ALL STEPS",
        rows,
        keep_connector=keep_connector,
        max_steps_per_episode=int(args.max_steps_per_episode),
        seed=int(args.seed),
        pooling=str(getattr(args, "vision_pooling", "flat_random")),
        mlp=bool(getattr(args, "mlp_probe", False)),
        mlp_hidden=int(getattr(args, "mlp_hidden", 1024)),
        mlp_epochs=int(getattr(args, "mlp_epochs", 2000)),
        mlp_max_features=int(getattr(args, "mlp_max_features", 4096)),
    )
    print(
        "\n"
        + "-" * 74
        + "\n  Read VISION against PROPRIO, on approach steps, and read dir cos.\n"
        "\n"
        "  proprio is NOT a null. Starts are capped 5 cm from the object, so the\n"
        "  gripper pose alone predicts the object position well and proprio will\n"
        "  score high. It is the bar the vision rows have to clear, not zero.\n"
        "\n"
        "  vision alone scores well: the object IS localizable from what the\n"
        "    residual is fed, so the servoing failure is RL rather than\n"
        "    perception -- the information is there and the policy ignores it.\n"
        "  vision ~ 0 but connector scores: the fixed random projection is\n"
        "    destroying the object's position. Learn it, or widen it.\n"
        "  both ~ 0: the frozen encoder does not localize the object at all, no\n"
        "    head reading it can servo, and the encoder itself has to adapt --\n"
        "    work for the action-expert LoRA, not for the residual.\n"
        "\n"
        "  proprio+vision usually reads BELOW proprio alone, and that is the\n"
        "  kernel, not evidence that vision hurts: a dot-product similarity over\n"
        "  512 mostly-uninformative dimensions swamps six informative ones. The\n"
        "  residual is an MLP and does not have that problem. Read the row as a\n"
        "  lower bound.\n"
        "\n"
        "  For scale: the trained policy's own action-to-target cosine is 0.11\n"
        "  and the frozen prior's is 0.05, against a deterministic miss of\n"
        "  0.40 m horizontally."
    )

    if keep_connector and "connector" in rows[0]:
        results["reductions"] = _report_reductions(
            approach,
            max_steps_per_episode=int(args.max_steps_per_episode),
            seed=int(args.seed),
        )

    summary = args.output / "grasp_feature_probe.json"
    summary.write_text(json.dumps(results, indent=2, default=float))
    print(
        "\n"
        + "-" * 74
        + "\n  Read MARGIN on the MATCHED subset. Nothing else here is evidence.\n"
        "\n"
        "  The raw accuracy is inflated by episode identity: grasp state is\n"
        "  near-constant within an episode, so any feature that says WHICH\n"
        "  episode a step came from predicts the label. The control is a\n"
        "  between-episode label shuffle that keeps that structure, so the\n"
        "  margin over it is the part attributable to seeing the grasp.\n"
        "\n"
        "  proprio margin already large: the residual can ALREADY tell. The\n"
        "    6-d state carries gripper_opening, and a hit stops the fingers at\n"
        "    the object's width while a miss closes them fully -- the same\n"
        "    physical signal a real parallel gripper reports. Observability is\n"
        "    then not the explanation for anything, and no new input is needed.\n"
        "  vision margin >> proprio margin: the frozen feature carries grasp\n"
        "    state, the residual can condition on it, and the lift failure is\n"
        "    not an observability problem.\n"
        "  vision margin ~ 0 but connector margin > 0: the fixed random\n"
        "    projection is discarding it. Learn or widen the projection.\n"
        "  both margins ~ 0: the frozen connector does not represent grasp\n"
        "    state, so no head reading it can help and the encoder itself must\n"
        "    adapt -- which is work for the action-expert LoRA, not a new input.\n"
        "\n"
        "  The effective sample size is EPISODES, not steps. Under ~20 matched\n"
        "  episodes a margin of a few points is noise; rerun with --episodes\n"
        "  high enough that the fold spread is small next to the margin.\n"
        f"\n  wrote {summary}"
    )
    return 0


def _build_runtime(harness: Any, config_path: Path, device: str) -> tuple[Any, int, int]:
    import yaml

    from rl_vla_bootstrapping.policy.smolvla_cdpr import load_smolvla_runtime

    raw = yaml.safe_load(config_path.read_text()) or {}
    policy = raw.get("policy", {}) or {}
    rl_args = dict(
        ((raw.get("training", {}) or {}).get("rl", {}) or {}).get("args", {}) or {}
    )
    state_dim = int(rl_args.get("state_dim", 6))
    vision_dim = int(rl_args.get("residual_vision_dim", 512))
    if not bool(rl_args.get("residual_vision_features", False)):
        print(
            "[probe] WARNING: residual_vision_features is false in this config, "
            "so the residual is NOT fed the vision feature at all. Probing it "
            "anyway, but the result describes a feature the run does not use."
        )
    runtime = load_smolvla_runtime(
        checkpoint=str(policy.get("base_checkpoint", "lerobot/smolvla_base")),
        device=str(device),
        mixed_precision=str(rl_args.get("mixed_precision", "bf16")),
        image_size=int(rl_args.get("image_size", 256)),
        state_dim=state_dim,
        include_wrist=bool(rl_args.get("include_wrist", True)),
        include_aux_camera=bool(rl_args.get("include_aux_camera", True)),
        mask_empty_aux_camera=bool(rl_args.get("mask_empty_aux_camera", True)),
        chunk_size=int(rl_args.get("chunk_size", 8)),
        action_dim=int(rl_args.get("action_dim", 5)),
        action_normalization=str(
            rl_args.get("smolvla_action_normalization", "tanh")
        ),
        model_image_size=(
            int(rl_args["smolvla_model_image_size"])
            if int(rl_args.get("smolvla_model_image_size", 0)) > 0
            else None
        ),
        # No torch.compile: this runs a handful of single-frame forwards, so
        # compilation would cost more than it saves.
        compile_model=False,
        # Without this the probe scored whatever load_smolvla_runtime defaults
        # to (flat_random) no matter what the config asked for -- so a run
        # configured for per_token_random or dual_random was measured on the
        # feature it had replaced.
        vision_pooling=str(
            rl_args.get("residual_vision_pooling", "flat_random")
        ),
    )
    return runtime, vision_dim, state_dim, str(
        rl_args.get("residual_vision_pooling", "flat_random")
    )


def rows_from_feature_files(paths: Sequence[Any]) -> tuple[list[dict[str, Any]], bool]:
    """Rebuild probe rows from saved captures, decompressing each array ONCE.

    np.load on an .npz is lazy and every ``saved[key]`` decompresses the whole
    array again. This used to index inside a per-row loop, so each field was
    decompressed once per row -- 4320 rows against a 4320 x 30720 connector is
    thousands of full 530 MB decompressions, and it OOM-killed the box twice.
    On a small capture the same code only looks slow, which is why the property
    is tested by counting lookups rather than by watching memory.

    The row values are VIEWS into the materialized arrays, not copies.
    """

    rows: list[dict[str, Any]] = []
    offset = 0
    has_connector = True
    for path in paths:
        saved = np.load(path)
        arrays = {key: saved[key] for key in saved.files}
        count = int(arrays["episode"].shape[0])
        local = [
            {
                # Offset so two shards' episode 0 do not merge into one
                # episode -- which would silently break every fold split.
                "episode": int(arrays["episode"][index]) + offset,
                "missed": bool(arrays["missed"][index]),
                "physical_grasp": bool(arrays["physical_grasp"][index]),
                "contact_loaded": bool(arrays["contact_loaded"][index]),
                "gripper_opening": float(arrays["gripper_opening"][index]),
                "ee_z": float(arrays["ee_z"][index]),
                "proprio": arrays["proprio"][index],
                "vision": arrays["vision"][index],
                "target_xy": arrays["target_xy"][index],
                "ee_xy": arrays["ee_xy"][index],
            }
            for index in range(count)
        ]
        if "connector" in arrays:
            connector = arrays["connector"]
            for index, row in enumerate(local):
                row["connector"] = connector[index]
        else:
            has_connector = False
        rows.extend(local)
        offset = max(row["episode"] for row in rows) + 1
        print(f"[probe] loaded {count} steps from {path}")
    return rows, has_connector


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "examples"
        / "cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml",
    )
    parser.add_argument(
        "--physics",
        choices=("auto", "mjlab_mjwarp", "mujoco_cpu"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument(
        "--miss-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of episodes driven to close on AIR at grasp height. These "
            "are the hard negatives; without them the probe just reads pose."
        ),
    )
    parser.add_argument(
        "--miss-offset-m",
        type=float,
        default=0.05,
        help="Lateral displacement of the grasp point on a missed episode.",
    )
    parser.add_argument(
        "--closed-gripper-below",
        type=float,
        default=0.35,
        help="Matched-subset gate: normalized gripper opening at or below this.",
    )
    parser.add_argument(
        "--label",
        choices=("physical_grasp", "contact_loaded"),
        default="physical_grasp",
        help=(
            "physical_grasp is what gates the reward. contact_loaded is the "
            "visually similar precursor, useful as a sanity comparison."
        ),
    )
    parser.add_argument(
        "--no-connector",
        action="store_true",
        help="Skip the 30720-d un-projected probe (saves memory and time).",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=24,
        help=(
            "Evenly spaced steps kept per episode before probing. The effective "
            "sample size is the EPISODE count, so extra steps within one add "
            "almost no information while adding their full share to an O(N^2) "
            "kernel. 0 keeps every step."
        ),
    )
    _bool_flag = parser.add_argument
    _bool_flag(
        "--skip-action-expert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Abandon each forward once the connector has produced its tokens. "
            "Everything after it is the flow-matching action expert computing a "
            "chunk this probe discards. Verified against a full forward on the "
            "first batch before it is trusted."
        ),
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help=(
            "Write features.npz and stop, without probing. For running several "
            "shards in parallel; probe the merged set afterwards by passing "
            "every shard's file to --from-features."
        ),
    )
    parser.add_argument(
        "--capture-every",
        type=int,
        default=4,
        help=(
            "Featurize every Nth env step. The probes subsample to "
            "--max-steps-per-episode anyway, so featurizing all ~104 steps of "
            "an episode computes ~75%% of them to be discarded. 1 restores the "
            "original behaviour."
        ),
    )
    parser.add_argument(
        "--feature-batch",
        type=int,
        default=64,
        help=(
            "SmolVLA batch for the featurization pass. The rollout harness "
            "clones two worlds and only world 0 is read, so a per-step forward "
            "runs at batch 2 and leaves the GPU idle. Lower this if it OOMs."
        ),
    )
    parser.add_argument(
        "--from-features",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Re-probe a features.npz written by an earlier run instead of "
            "rolling out again. Skips the simulator and SmolVLA entirely. The "
            "Accepts several files, which are concatenated with their episode "
            "ids offset so shards stay distinct -- that is how parallel "
            "captures are merged."
        ),
    )
    parser.add_argument(
        "--mlp-probe",
        action="store_true",
        help=(
            "Also fit a NONLINEAR probe -- two hidden layers of --mlp-hidden, "
            "the residual's own shape -- on the same folds with the same "
            "episode-swap control. Every localization number in this campaign "
            "is linear, and the thing that has to learn the map is an MLP, so "
            "'+0.389 linear' and 'the residual can learn it' are different "
            "claims. A 5.2M-step run has now failed to move the residual's aim "
            "off 0.055 while being fed that feature, which makes the "
            "difference worth measuring rather than assuming."
        ),
    )
    parser.add_argument("--mlp-hidden", type=int, default=1024)
    parser.add_argument(
        "--mlp-max-features",
        type=int,
        default=4096,
        help=(
            "Skip the MLP probe for feature blocks wider than this. The 30720-d "
            "connector needs ~530 MB per copy and OOM-killed a 180-episode "
            "capture; it is a ceiling reference whose linear row already "
            "answers the question, while the rows that matter -- the "
            "residual's actual 1024-d input and the 6-d proprioception null -- "
            "are far below the cap. Raise it only with the host RAM to spare."
        ),
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=2000,
        help=(
            "Full-batch AdamW steps per fold. NOT a free parameter: on a "
            "synthetic task whose answer is known, the probe reads direction "
            "cosine 0.81 at 400 steps, 0.87 at 1200 and 0.91 at 3000 -- so a "
            "short budget produces a false negative that looks exactly like a "
            "feature carrying nothing. Train R^2 is printed beside every score; "
            "if it is not close to 1, the probe is under-trained and the "
            "held-out number means nothing yet."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "grasp_feature_probe"
    )
    parser.add_argument("--start-distance-cap", type=float, default=0.20)
    parser.add_argument(
        "--target-catalogs", nargs="+", default=list(DEFAULT_CATALOGS)
    )
    args = parser.parse_args(argv)

    if args.from_features is not None:
        rows, has_connector = rows_from_feature_files(args.from_features)
        print(
            f"[probe] re-probing {len(rows)} steps over "
            f"{len({row['episode'] for row in rows})} episodes"
        )
        return _probe_and_report(
            rows,
            args,
            keep_connector=has_connector and not args.no_connector,
            features_path=None,
        )

    harness = _load_harness()
    runtime, vision_dim, state_dim, pooling = _build_runtime(
        harness, args.config, args.device
    )
    args.vision_pooling = pooling
    print(f"[probe] vision pooling from the config: {pooling} ({vision_dim}-d)")
    capture = _Capture(
        harness,
        runtime=runtime,
        vision_dim=vision_dim,
        state_dim=state_dim,
        keep_connector=not args.no_connector,
        capture_every=int(args.capture_every),
    )

    # Displace the grasp point on a chosen fraction of episodes so the gripper
    # descends to the right height and closes on nothing. oracle_phases is
    # called exactly once per episode, which makes it the episode boundary.
    rng = np.random.RandomState(int(args.seed))
    original_phases = harness.oracle_phases
    original_grasp_point = harness._grasp_point
    displacement = {"vector": np.zeros(3)}

    def grasp_point(ctx: Any, obs: dict[str, Any]) -> np.ndarray:
        return original_grasp_point(ctx, obs) + displacement["vector"]

    def oracle_phases(instruction_type: str) -> Any:
        missed = bool(rng.random_sample() < float(args.miss_fraction))
        angle = rng.random_sample() * 2.0 * np.pi
        displacement["vector"] = (
            np.array(
                (
                    float(args.miss_offset_m) * np.cos(angle),
                    float(args.miss_offset_m) * np.sin(angle),
                    0.0,
                )
            )
            if missed
            else np.zeros(3)
        )
        capture.start_episode(missed=missed)
        return original_phases(instruction_type)

    harness._grasp_point = grasp_point
    harness.oracle_phases = oracle_phases
    capture.install()

    saved_argv = sys.argv
    sys.argv = [
        str(HARNESS),
        "--instructions", "pick_up",
        "--episodes-per-instruction", str(int(args.episodes)),
        "--physics", str(args.physics),
        "--device", str(args.device),
        "--no-video",
        # The probe reads the cameras itself, so it needs the renderer to exist
        # but not a single encoded frame written to disk.
        "--force-renderer",
        "--continue-after-terminal",
        "--start-distance-cap", str(float(args.start_distance_cap)),
        "--target-catalogs", *args.target_catalogs,
        "--output", str(args.output / "rollouts"),
    ]
    try:
        status = harness.main()
    finally:
        sys.argv = saved_argv
        capture.remove()
        harness.oracle_phases = original_phases
        harness._grasp_point = original_grasp_point
    if int(status or 0) != 0:
        return int(status or 0)

    rows = capture.rows
    if not rows:
        print("[probe] captured no steps")
        return 1
    capture.compute_features(
        int(args.feature_batch),
        skip_action_expert=bool(args.skip_action_expert),
    )
    return _probe_and_report(
        rows,
        args,
        keep_connector=not args.no_connector,
        features_path=args.output / "features.npz",
    )


if __name__ == "__main__":
    raise SystemExit(main())
