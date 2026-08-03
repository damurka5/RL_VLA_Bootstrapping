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
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

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
    ) -> None:
        self._harness = harness
        self._runtime = runtime
        self._vision_dim = int(vision_dim)
        self._state_dim = int(state_dim)
        self._keep_connector = bool(keep_connector)
        self._original = harness._GraspShim.update
        self.episode = -1
        self.missed = False
        self.rows: list[dict[str, Any]] = []

    def start_episode(self, *, missed: bool) -> None:
        self.episode += 1
        self.missed = bool(missed)

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
        import torch

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
        connector_flat: list[Any] = []
        hook = None
        if self._keep_connector:
            module = self._runtime._vision_connector()
            hook = module.register_forward_hook(
                lambda _m, _i, out: connector_flat.append(
                    out.detach().float().cpu()
                )
            )
        try:
            with torch.inference_mode():
                _prior, vision = (
                    self._runtime.sample_cdpr_chunks_and_vision_from_tensors(
                        primary_images=cameras.overview,
                        wrist_images=cameras.wrist,
                        states=state,
                        instructions=reset.instructions,
                        vision_dim=self._vision_dim,
                        microbatch_size=0,
                    )
                )
        finally:
            if hook is not None:
                hook.remove()

        row: dict[str, Any] = {
            "episode": int(self.episode),
            "missed": bool(self.missed),
            "physical_grasp": bool(physical_grasp[world].item()),
            "contact_loaded": bool(diagnostics["contact_loaded"][world].item()),
            "gripper_opening": float(low_dim.gripper_opening[world].item()),
            "ee_z": float(low_dim.ee_position[world, 2].item()),
            "proprio": state[world].detach().float().cpu().numpy().copy(),
            "vision": vision[world].detach().float().cpu().numpy().copy(),
        }
        if self._keep_connector and connector_flat:
            # Overview + wrist only, matching _pool_vision; never the masked aux.
            cams = connector_flat[: max(1, min(2, len(connector_flat)))]
            row["connector"] = np.concatenate(
                [cam[world].reshape(-1).numpy() for cam in cams]
            ).astype(np.float32)
        self.rows.append(row)


# ---------------------------------------------------------------- probe maths


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

    def score(target: np.ndarray, alpha: float) -> tuple[float, float]:
        per_fold: list[float] = []
        for held in assignments:
            test = np.isin(episodes, held)
            train = ~test
            if train.sum() < 2 or test.sum() < 2:
                continue
            if np.unique(target[test] > 0).size < 2:
                # A fold with one class cannot produce a balanced accuracy that
                # means anything; skipping is honest, inventing 0.5 is not.
                continue
            matrix = features.astype(np.float64)
            matrix = matrix - matrix[train].mean(axis=0, keepdims=True)
            matrix = matrix / np.maximum(
                matrix[train].std(axis=0, keepdims=True), 1.0e-8
            )
            gram = matrix @ matrix.T + 1.0
            dual = np.linalg.solve(
                gram[np.ix_(train, train)] + alpha * np.eye(int(train.sum())),
                target[train],
            )
            predicted = (gram[np.ix_(test, train)] @ dual) > 0.0
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


def _stack(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.stack([np.asarray(row[key], dtype=np.float32) for row in rows])


def _report_subset(
    name: str,
    rows: Sequence[dict[str, Any]],
    label_key: str,
    *,
    keep_connector: bool,
) -> dict[str, Any]:
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
    )
    return runtime, vision_dim, state_dim


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
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "grasp_feature_probe"
    )
    parser.add_argument("--start-distance-cap", type=float, default=0.20)
    parser.add_argument(
        "--target-catalogs", nargs="+", default=list(DEFAULT_CATALOGS)
    )
    args = parser.parse_args(argv)

    harness = _load_harness()
    runtime, vision_dim, state_dim = _build_runtime(
        harness, args.config, args.device
    )
    capture = _Capture(
        harness,
        runtime=runtime,
        vision_dim=vision_dim,
        state_dim=state_dim,
        keep_connector=not args.no_connector,
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
    label_key = str(args.label)
    keep_connector = not args.no_connector

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
    )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / "grasp_feature_probe.json"
    summary.write_text(json.dumps(results, indent=2, default=float))
    np.savez_compressed(
        args.output / "features.npz",
        vision=_stack(rows, "vision"),
        proprio=_stack(rows, "proprio"),
        label=np.array([1.0 if row[label_key] else 0.0 for row in rows]),
        episode=np.array([row["episode"] for row in rows]),
        gripper_opening=np.array([row["gripper_opening"] for row in rows]),
    )

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


if __name__ == "__main__":
    raise SystemExit(main())
