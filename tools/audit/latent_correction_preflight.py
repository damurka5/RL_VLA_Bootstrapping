#!/usr/bin/env python3
"""GPU preflight for the zero-init correction / latent-likelihood pilot.

Spec: docs/reports/campaign/CDPR_ZERO_INIT_CORRECTION_IMPLEMENTATION.md, section 9.

Runs before any optimization, on the real legacy checkpoint:

1. Records the source checkpoint's SHA-256, the config and manifest hashes, the
   Git revision, runtime versions, and the resolved optimizer of both arms.
2. Rolls the LEGACY checkpoint out through the evaluator's own loader and
   rollout (deterministic mean, prior noise as configured), capturing the real
   ``(state, prior)`` inputs of live worlds at every ``--capture-every``-th
   decision, then tags them by final strict outcome and decision bucket.
3. On those SAME tensors -- one prior computation shared by every actor, so
   SmolVLA's prior noise cannot masquerade as an actor difference -- compares
   the legacy mean with the converted candidate's and the control's, checks
   the correction is exactly zero, and that the reference branch and LoRA are
   bit-identical to the source.
4. With fixed latent noise and fixed offsets, compares the executed commands
   of the legacy sampler with both latent samplers.
5. Saves the zero-update candidate, reloads it through the resume path, and
   repeats 3-4.
6. Optionally (``--candidate-checkpoint``, e.g. after a one-update smoke run),
   loads a trained pilot checkpoint through the evaluator's loader and shows
   its evaluation actions include the learned correction.

Exactness is fixed-input equality. Independent MJWarp episode verdicts are NOT
compared: repeat noise makes them a performance question, not an identity one.

Usage::

    python3 tools/audit/latent_correction_preflight.py \\
        --config configs/examples/cdpr_smolvla_three_stage_put_into_latent_correction.yaml \\
        --checkpoint runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006/smolvla_grpo_adapter.pt \\
        --scene-manifest runs/three_stage/scenes_8192.json \\
        --output runs/latent_pilot_preflight_<date>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from argparse import Namespace  # noqa: E402
from typing import Any, Sequence  # noqa: E402

from tools.audit.evaluate_cdpr_full_put_into import run_unassisted  # noqa: E402
from tools.audit.xy_approach_probe import _build_world, _load_checkpoint  # noqa: E402
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import FullTaskSceneResetter  # noqa: E402
from rl_vla_bootstrapping.policy.latent_correction_policy import (  # noqa: E402
    ACTION_LIKELIHOOD_LATENT,
    POLICY_ARCHITECTURE_CORRECTION,
    POLICY_ARCHITECTURE_LEGACY,
    tensor_fingerprint,
)
from rl_vla_bootstrapping.policy.smolvla_cdpr import prior_noise_scale_from_env  # noqa: E402
from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import read_manifest, select_split  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(module: str) -> str:
    try:
        return str(__import__(module).__version__)
    except Exception as error:  # pragma: no cover - probe only
        return f"unavailable: {type(error).__name__}"


class InputCapture:
    """The evaluator's ``trace`` hook: keeps live (state, prior) rows."""

    def __init__(self, every: int) -> None:
        self.every = max(1, int(every))
        self.rows: list[tuple[int, Any, Any, Any]] = []

    def record_decision(self, *, decision_index, cameras, state, prior, active) -> None:  # noqa: ARG002
        if int(decision_index) % self.every:
            return
        live = active.nonzero(as_tuple=False).flatten()
        if live.numel():
            self.rows.append((int(decision_index), live.clone(), state[live].detach().clone(),
                              prior[live].detach().clone()))

    def record_step(self, **_: Any) -> None:
        return None


def _latent_trainer(world: Any, payload: dict, architecture: str, run_dir: Path, lr: float) -> Any:
    from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import SmolVLAGRPOTrainer

    args = Namespace(**vars(world.args))
    args.policy_architecture = architecture
    args.action_likelihood = ACTION_LIKELIHOOD_LATENT
    args.optimizer_lr_override = float(lr)
    args.vla_lora_updates_enabled = False
    args.legacy_init_checkpoint = None
    trainer = SmolVLAGRPOTrainer(
        args=args, state_dim=int(payload["state_dim"]), action_dim=int(payload["action_dim"]),
        chunk_size=int(payload["chunk_size"]), run_dir=run_dir, device=world.device,
    )
    # The runtime already carries the source LoRA (restored by _build_world);
    # the conversion reloads it from the checkpoint, verifies it, and freezes it.
    trainer.vla_runtime = world.runtime
    return trainer


def _optimizer_summary(trainer: Any) -> dict[str, Any]:
    groups = trainer.optimizer.param_groups
    return {
        "type": type(trainer.optimizer).__name__,
        "lr": [float(g["lr"]) for g in groups],
        "eps": [float(g["eps"]) for g in groups],
        "weight_decay": [float(g["weight_decay"]) for g in groups],
        "trainable_parameters": trainer.trainable_parameter_count(),
        "state_entries": len(trainer.optimizer.state),
    }


def _equivalence(torch: Any, legacy: Any, others: dict[str, Any], batches: Sequence[dict]) -> dict[str, Any]:
    """Max |mean difference| per group of captured inputs, and the correction."""

    out: dict[str, Any] = {}
    with torch.inference_mode():
        for batch in batches:
            reference = legacy._unwrap(legacy.actor)(batch["state"], batch["prior"])
            for name, trainer in others.items():
                mean = trainer._unwrap(trainer.actor)(batch["state"], batch["prior"])
                parts = trainer.action_components_tensor(
                    states=batch["state"], priors=batch["prior"], action_count=int(trainer.chunk_size))
                key = f"{name}/{batch['group']}"
                entry = out.setdefault(key, {"rows": 0, "max_abs_mean_diff": 0.0,
                                             "max_abs_correction": 0.0, "bitwise_equal": True})
                diff = (mean - reference).abs()
                entry["rows"] += int(mean.shape[0])
                entry["max_abs_mean_diff"] = max(entry["max_abs_mean_diff"], float(diff.max()))
                entry["max_abs_correction"] = max(entry["max_abs_correction"], float(parts["correction"].abs().max()))
                entry["bitwise_equal"] = entry["bitwise_equal"] and bool(torch.equal(mean, reference))
    return out


def _sampled_commands(torch: Any, legacy: Any, others: dict[str, Any], batch: dict, seed: int) -> dict[str, Any]:
    """Same generator state, same realized offsets -> same executed commands."""

    worlds = int(batch["state"].shape[0])
    action_dim = int(legacy.action_dim)
    g = torch.Generator(device=legacy.device).manual_seed(seed)
    offsets = torch.randn((worlds, action_dim), generator=g, device=legacy.device) * legacy.episode_offset_std
    gate = (torch.arange(worlds, device=legacy.device) % 2 == 0).unsqueeze(-1).to(offsets.dtype)
    offsets = offsets * gate
    offset_std = legacy.episode_offset_std.unsqueeze(0).expand(worlds, -1) * gate
    reference, _, _ = legacy.sample_action_chunks_tensor(
        states=batch["state"], priors=batch["prior"], action_count=int(legacy.args.replan_every),
        generator=torch.Generator(device=legacy.device).manual_seed(seed + 1),
        mean_offset=offsets, offset_std=offset_std)
    out = {}
    for name, trainer in others.items():
        sample = trainer.sample_latent_action_chunks_tensor(
            states=batch["state"], priors=batch["prior"], action_count=int(legacy.args.replan_every),
            generator=torch.Generator(device=legacy.device).manual_seed(seed + 1), mean_offset=offsets)
        out[name] = {
            "executed_equal": bool(torch.equal(sample["executed_action"], reference)),
            "max_abs_executed_diff": float((sample["executed_action"] - reference).abs().max()),
            "latent_outside_bounds_fraction": float((sample["policy_sample"].abs() > 1.0).float().mean()),
            "offset_recorded_equal": bool(torch.equal(sample["behavior_mean_offset"], offsets)),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="The LEGACY source adapter.")
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--split", default="student_validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--decisions", type=int, default=128)
    parser.add_argument("--capture-every", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--expected-source-sha256", default="")
    parser.add_argument("--candidate-checkpoint", type=Path, default=None,
                        help="A trained pilot checkpoint whose evaluation must use its correction.")
    args = parser.parse_args(argv)

    import torch

    started = time.perf_counter()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.expanduser().resolve()
    config = args.config.expanduser().resolve()
    manifest_path = args.scene_manifest.expanduser().resolve()
    source_sha = _sha256(checkpoint)
    failures: list[str] = []
    if args.expected_source_sha256 and source_sha != args.expected_source_sha256:
        failures.append(f"source sha256 {source_sha} != expected {args.expected_source_sha256}")
    payload = _load_checkpoint(checkpoint)
    report: dict[str, Any] = {
        "tool": "latent_correction_preflight.py",
        "source_checkpoint": str(checkpoint),
        "source_sha256": source_sha,
        "source_global_step": int(payload.get("global_step", 0)),
        "config": str(config), "config_sha256": _sha256(config),
        "scene_manifest": str(manifest_path), "scene_manifest_sha256": _sha256(manifest_path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=ROOT).strip()),
        "runtime_versions": {"python": platform.python_version(), "torch": _version("torch"),
                             "mujoco": _version("mujoco"), "warp": _version("warp"),
                             "cuda": torch.version.cuda},
        "prior_noise_scale": prior_noise_scale_from_env(),
    }

    world = _build_world(
        controller_workspace_from_config=True, checkpoint=checkpoint, config_path=config,
        device_str=str(args.device), worlds=int(args.worlds), group_size=2,
        microbatch=int(args.microbatch), load_policy=True, run_dir=output,
    )
    legacy = world.trainer
    if legacy.policy_architecture != POLICY_ARCHITECTURE_LEGACY:
        raise SystemExit(f"--checkpoint must be a legacy adapter; it is {legacy.policy_architecture}.")

    # --- real inputs from the legacy policy's own rollout ----------------------
    scenes, _ = read_manifest(manifest_path)
    batch_scenes = select_split(scenes, str(args.split))[: int(args.worlds)]
    resetter = FullTaskSceneResetter(
        backend=world.backend, worlds_per_rank=int(args.worlds),
        support_surface_z=float(world.task_metadata.get("support_surface_z", 0.15)),
        task_metadata=world.task_metadata,
    )
    vision_dim = (int(getattr(world.args, "residual_vision_dim", 0))
                  if bool(getattr(world.args, "residual_vision_features", False)) else 0)
    capture = InputCapture(int(args.capture_every))
    rollout = run_unassisted(
        world=world, resetter=resetter, scenes=batch_scenes, decisions=int(args.decisions),
        settle_decisions=0, assisted_yaw=None, vision_dim=vision_dim, trace=capture,
    )
    strict = torch.as_tensor(rollout["strict"], device=world.device).to(torch.bool)
    groups: dict[str, dict[str, list]] = {}
    for decision, live, state, prior in capture.rows:
        bucket = "early" if decision < 32 else ("middle" if decision < 64 else "late")
        for outcome, mask in (("strict", strict[live]), ("failed", ~strict[live])):
            if bool(mask.any()):
                entry = groups.setdefault(f"{outcome}_{bucket}", {"state": [], "prior": []})
                entry["state"].append(state[mask])
                entry["prior"].append(prior[mask])
    batches = [{"group": name, "state": torch.cat(v["state"]), "prior": torch.cat(v["prior"])}
               for name, v in sorted(groups.items())]
    report["captured_inputs"] = {b["group"]: int(b["state"].shape[0]) for b in batches}
    report["capture_rollout_strict"] = f"{int(strict.sum())}/{int(strict.numel())}"
    if not any(name.startswith("strict") for name in groups) or not any(name.startswith("failed") for name in groups):
        failures.append("captured inputs do not span both successful and failed episodes; raise --worlds")

    # --- conversion -------------------------------------------------------------
    trainers = {}
    lineages = {}
    for name, architecture in (("candidate", POLICY_ARCHITECTURE_CORRECTION), ("control", POLICY_ARCHITECTURE_LEGACY)):
        trainer = _latent_trainer(world, payload, architecture, output / name, args.lr)
        lineages[name] = trainer.initialize_from_legacy_checkpoint(checkpoint)
        trainers[name] = trainer
    report["lineage"] = lineages["candidate"]
    report["optimizers"] = {name: _optimizer_summary(t) for name, t in trainers.items()}
    for name, summary in report["optimizers"].items():
        if any(lr != float(args.lr) for lr in summary["lr"]) or summary["state_entries"]:
            failures.append(f"{name} optimizer is not fresh at lr {args.lr}: {summary}")
    candidate = trainers["candidate"]
    legacy_actor_state = {k[len("actor.net."):]: v for k, v in payload["policy"].items() if k.startswith("actor.net.")}
    report["integrity"] = {
        "reference_equals_source": tensor_fingerprint(candidate._reference_named_parameters())
        == tensor_fingerprint({k: v.to(world.device) for k, v in legacy_actor_state.items()}),
        "lora_equals_source": lineages["candidate"]["source_lora_sha256"]
        == tensor_fingerprint(dict(payload["vla_lora"])) if payload.get("vla_lora") else None,
        "lora_frozen": not any(p.requires_grad for p in candidate._lora_named_parameters().values()),
        "reference_frozen": not any(p.requires_grad for p in candidate._reference_named_parameters().values()),
    }
    for key, value in report["integrity"].items():
        if value is False:
            failures.append(f"integrity check failed: {key}")

    def check_equivalence(label: str, actors: dict[str, Any]) -> None:
        eq = _equivalence(torch, legacy, actors, batches)
        report[f"equivalence_{label}"] = eq
        worst = max((e["max_abs_mean_diff"] for e in eq.values()), default=float("nan"))
        worst_corr = max((e["max_abs_correction"] for e in eq.values()), default=float("nan"))
        if not all(e["bitwise_equal"] for e in eq.values()) or worst_corr != 0.0:
            failures.append(f"{label}: mean max error {worst:g}, correction max {worst_corr:g}")
        sampled = _sampled_commands(torch, legacy, actors, batches[0], seed=1234)
        report[f"sampled_commands_{label}"] = sampled
        for name, entry in sampled.items():
            if not (entry["executed_equal"] and entry["offset_recorded_equal"]):
                failures.append(f"{label}: sampled commands differ for {name}: {entry}")
        print(f"[preflight] {label}: max |mean diff| {worst:.3e}, max |correction| {worst_corr:.3e}, "
              f"sampled commands equal: {all(e['executed_equal'] for e in sampled.values())}", flush=True)

    check_equivalence("converted", trainers)

    # --- save / reload through the resume path ---------------------------------
    zero_path = candidate.save(global_step=0, args=candidate.args, extra_state={})
    reloaded = _latent_trainer(world, payload, POLICY_ARCHITECTURE_CORRECTION, output / "reloaded", args.lr)
    reloaded.load(zero_path)
    report["zero_update_checkpoint"] = {"path": str(zero_path), "sha256": _sha256(zero_path)}
    check_equivalence("reloaded", {"candidate_reloaded": reloaded})

    # --- a trained pilot checkpoint, through the evaluator's loader -----------
    if args.candidate_checkpoint is not None:
        trained_path = args.candidate_checkpoint.expanduser().resolve()
        trained_world = _build_world(
            controller_workspace_from_config=True, checkpoint=trained_path, config_path=config,
            device_str=str(args.device), worlds=int(args.worlds), group_size=2,
            microbatch=int(args.microbatch), load_policy=True, run_dir=output / "trained_eval",
        )
        trained = trained_world.trainer
        stats = {"policy_architecture": trained.policy_architecture,
                 "sha256": _sha256(trained_path), "groups": {}}
        with torch.inference_mode():
            for batch in batches:
                chunk = trained.deterministic_action_chunks_tensor(
                    states=batch["state"], priors=batch["prior"], action_count=int(trained.args.replan_every))
                parts = trained.action_components_tensor(
                    states=batch["state"], priors=batch["prior"], action_count=int(trained.args.replan_every))
                reference = torch.tanh(parts["reference_logit"])
                stats["groups"][batch["group"]] = {
                    "abs_correction_mean": float(parts["correction"].abs().mean()),
                    "abs_correction_max": float(parts["correction"].abs().max()),
                    "max_abs_eval_minus_reference": float((chunk - reference).abs().max()),
                    "eval_equals_tanh_combined": bool(torch.equal(chunk, torch.tanh(parts["logit"]).clamp(-1, 1))),
                }
        report["trained_candidate"] = stats
        if trained.policy_architecture != POLICY_ARCHITECTURE_CORRECTION:
            failures.append("trained checkpoint did not load as the correction architecture")
        if max(g["abs_correction_max"] for g in stats["groups"].values()) == 0.0:
            failures.append("trained checkpoint's correction is exactly zero (no update reached it?)")

    report["failures"] = failures
    report["passed"] = not failures
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (output / "latent_correction_preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("source_sha256", "captured_inputs", "integrity", "passed", "failures")},
                     indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
