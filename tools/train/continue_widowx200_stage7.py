#!/usr/bin/env python3
"""Continue only WidowX stage 7 with phased milestone rewards and contact-v2.

Without --execute, prepare a new config/run and print the server commands.
With --execute, run the CUDA preflight first, then start joint RL.
--pipeline-dir starts from policy weights; --resume-run restores training state.
The source pipeline, checkpoints and completed-stage markers are never edited.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
STAGE = "s7_joint_staged"
INSTRUCTIONS = {"move_to_object", "pick_up", "put_into_plate", "put_into_bowl"}
# Match the pipeline driver's isolation from previous smoke/pilot launches.
STALE_OVERRIDES = (
    "RLVLA_SMOLVLA_RESUME_CHECKPOINT", "RLVLA_SMOLVLA_WARMSTART_CHECKPOINT",
    "RLVLA_SMOLVLA_LEGACY_INIT_CHECKPOINT", "RLVLA_SMOLVLA_POLICY_ARCHITECTURE",
    "RLVLA_SMOLVLA_RESET_LOG_STD", "RLVLA_SMOLVLA_MAX_TRAIN_STEPS",
    "RLVLA_SMOLVLA_MJWARP_MAX_UPDATES", "RLVLA_SMOLVLA_PPO_EPOCHS",
    "RLVLA_SMOLVLA_OPTIMIZER_LR_OVERRIDE", "RLVLA_SMOLVLA_PRIOR_NOISE_SCALE",
    "RLVLA_SMOLVLA_REFERENCE_ANCHOR_BANK", "RLVLA_SMOLVLA_NPROC_PER_NODE",
    "RLVLA_CDPR_THREE_STAGE_SCENE_MANIFEST", "RLVLA_CDPR_DEMO_BANK",
    "RLVLA_TASK_METADATA_JSON", "RLVLA_MJWARP_ALLOW_CPU",
    "RLVLA_MJWARP_WORLDS_PER_RANK", "RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE",
    "RLVLA_MEMORY_REPORT",
)


def configure_huggingface_env(env: dict[str, str]) -> None:
    """Set public/cache-only access before the child imports HF libraries.

    Mirror scripts/huggingface_public_models.sh. The token-file override also
    prevents older Hub clients from trying to refresh an unused cached OAuth
    token. Model-cache paths and the user's real token files are untouched.
    """
    public_only = env.get("RLVLA_HF_PUBLIC_MODELS_ONLY", "1")
    offline = env.get("RLVLA_HF_OFFLINE", "0")
    if public_only not in {"0", "1"} or offline not in {"0", "1"}:
        raise ValueError("RLVLA_HF_PUBLIC_MODELS_ONLY and RLVLA_HF_OFFLINE must be 0 or 1.")
    if public_only == "1":
        env.pop("HF_TOKEN", None)
        env.pop("HUGGING_FACE_HUB_TOKEN", None)
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        env["HF_TOKEN_PATH"] = os.devnull
    if offline == "1":
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", RLVLA_HF_PREFLIGHT="0")


def configure_saved_huggingface_env(env: dict[str, str]) -> None:
    # The explicit launch mode also applies to env overrides in the saved YAML.
    for key in ("RLVLA_HF_PUBLIC_MODELS_ONLY", "RLVLA_HF_OFFLINE"):
        if key in os.environ:
            env[key] = os.environ[key]
    configure_huggingface_env(env)


def prepare(pipeline_dir: Path | None, run_name: str, *, checkpoint: Path | None = None,
            steps: int | None = None, composed_horizon_decisions: int | None = None,
            no_early_stop: bool = False, resume_run: Path | None = None) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("Run name must be a plain directory name.")
    if steps is not None and steps <= 0:
        raise ValueError("--steps must be positive.")
    if composed_horizon_decisions is not None and composed_horizon_decisions <= 0:
        raise ValueError("--composed-horizon-decisions must be positive.")
    lineage = None
    if resume_run is not None:
        resume_run = resume_run.expanduser().resolve()
        lineage = json.loads((resume_run / "continuation.json").read_text())
        if (Path(lineage["run_dir"]).resolve() != resume_run
                or lineage.get("source_stage") != STAGE
                or lineage.get("grasp_contract") != "contact_v2"
                or lineage.get("reward") != "staged_milestone"):
            raise ValueError("--resume-run requires a contact-v2 phased-reward stage-7 run.")
        if pipeline_dir is not None:
            raise ValueError("Choose --pipeline-dir or --resume-run, not both.")
        pipeline_dir = Path(lineage["source_pipeline"])
        if checkpoint is None:
            checkpoint = resume_run / "rl" / "latest.pt"
    if pipeline_dir is None:
        raise ValueError("Supply --pipeline-dir or --resume-run.")
    pipeline_dir = pipeline_dir.expanduser().resolve()
    state = json.loads((pipeline_dir / "pipeline_state.json").read_text())
    stage = state["stages"][STAGE]
    source = Path(lineage["config"] if lineage else stage["config"]).expanduser()
    if not source.is_absolute():
        source = ROOT / source
    source = source.resolve()
    config = yaml.safe_load(source.read_text())
    if set(config["task"]["instruction_types"]) != INSTRUCTIONS:
        raise ValueError("The saved stage-7 config must contain all four joint instructions.")
    if not str(config["embodiment"]["name"]).startswith("widowx200"):
        raise ValueError("The saved config is not a WidowX-200 config.")
    if lineage and not config["task"].get("metadata", {}).get("staged_milestone_reward"):
        raise ValueError("The resumed run's saved config no longer uses phased milestone rewards.")
    if checkpoint is None:
        checkpoint = stage.get("policy") or stage.get("selected_checkpoint") or stage.get("best_checkpoint")
        if not checkpoint:
            raise ValueError("No stage-7 policy recorded; supply --checkpoint.")
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Missing initializer checkpoint: {checkpoint}")

    # Keep the new YAML beside the original so all relative assets still resolve.
    destination = source.with_name(f"{run_name}.yaml")
    run_dir = ROOT / "runs" / run_name
    if run_dir.exists() or destination.exists():
        raise ValueError(f"Choose a new --run-name; {run_dir} or {destination} already exists.")
    config = copy.deepcopy(config)
    config.setdefault("project", {}).update(name=run_name, output_root=str(ROOT / "runs"))
    metadata = config["task"].setdefault("metadata", {})
    metadata.update(staged_milestone_reward=True, sparse_binary_reward=False,
                    open_grasp_hover_height=0.04)
    if composed_horizon_decisions is not None:
        # The same metadata builds the training and composed-validation
        # resetters; their uncaught placements must get the same budget.
        metadata["placement_grasp_horizon_min_decisions"] = composed_horizon_decisions
    rl_args = config["training"]["rl"]["args"]
    rl_args.pop("resume_checkpoint", None)
    rl_args.pop("legacy_init_checkpoint", None)
    rl_args.update(three_stage_sparse_credit=False, split_credit_at_grasp=False,
                   early_stop_metric="min_by_instruction_composed")
    if lineage:
        rl_args["resume_checkpoint"] = str(checkpoint)
    if no_early_stop:
        rl_args.update(early_stop_target=0.0, early_stop_plateau_validations=0)
    if int(rl_args.get("composed_validation_episodes_per_instruction", 0)) <= 0:
        rl_args["composed_validation_episodes_per_instruction"] = 256
    if steps is not None:
        rl_args["max_train_steps"] = steps
    if int(rl_args["max_train_steps"]) <= 0:
        raise ValueError("Stage-7 training budget must be positive.")
    # Explicitly record the embodiment in the stage subprocess environment too.
    remote_env = config.setdefault("remote", {}).setdefault("env_vars", {})
    for saved_env in (remote_env, config["project"].get("env", {})):
        for key in STALE_OVERRIDES:
            saved_env.pop(key, None)
        configure_saved_huggingface_env(saved_env)
    remote_env["RLVLA_EMBODIMENT"] = "widowx200"
    if lineage:
        remote_env["RLVLA_SMOLVLA_RESUME_CHECKPOINT"] = str(checkpoint)
    else:
        remote_env.update(RLVLA_SMOLVLA_WARMSTART_CHECKPOINT=str(checkpoint),
                          RLVLA_SMOLVLA_RESET_LOG_STD="1")

    run_dir.mkdir(parents=True, exist_ok=False)
    with destination.open("x") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    commands = [
        [sys.executable, str(ROOT / "scripts/preflight_widowx200_pipeline.py"),
         "--config", str(destination), "--skills", "joint", "--device", "cuda:0",
         "--worlds", str(config["simulator"]["worlds_per_rank"]),
         "--output", str(run_dir / "preflight")],
        [sys.executable, "-m", "rl_vla_bootstrapping.cli.train", "--config", str(destination),
         "--stage", "rl", "--run-name", run_name, "--execute"],
    ]
    payload = {
        "source_pipeline": str(pipeline_dir), "source_stage": STAGE,
        "source_config": str(source), "config": str(destination), "run_dir": str(run_dir),
        "initialization": "full_state_resume" if lineage else "weights_only",
        "fresh_optimizer": not bool(lineage), "fresh_curriculum": not bool(lineage),
        "global_step_start": None if lineage else 0,
        "reward": "staged_milestone", "grasp_contract": "contact_v2",
        "max_train_steps": int(rl_args["max_train_steps"]),
        "ranks": int(config["training"]["rl"].get("launcher_args", {}).get("nproc_per_node", 1)),
        "early_stop_metric": rl_args["early_stop_metric"], "commands": commands,
        "composed_horizon_min_decisions": int(metadata.get("placement_grasp_horizon_min_decisions", 32)),
        "early_stop_target": float(rl_args.get("early_stop_target", 0.0)),
        "early_stop_plateau_validations": int(rl_args.get("early_stop_plateau_validations", 0)),
        "config_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    if lineage:
        payload.update(source_run=str(resume_run), resume_checkpoint=str(checkpoint))
    else:
        payload["warmstart"] = str(checkpoint)
    (run_dir / "continuation.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def continuation_env(payload: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    for key in STALE_OVERRIDES:
        env.pop(key, None)
    env.update(RLVLA_EMBODIMENT="widowx200", PYTHONUNBUFFERED="1",
               RLVLA_SMOLVLA_NPROC_PER_NODE=str(payload["ranks"]))
    if payload["initialization"] == "full_state_resume":
        env["RLVLA_SMOLVLA_RESUME_CHECKPOINT"] = payload["resume_checkpoint"]
    else:
        env.update(RLVLA_SMOLVLA_WARMSTART_CHECKPOINT=payload["warmstart"],
                   RLVLA_SMOLVLA_RESET_LOG_STD="1")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    configure_huggingface_env(env)
    return env


def retry_startup(run_dir: Path) -> dict[str, Any]:
    """Retry a prepared run that failed before saving its first checkpoint."""
    run_dir = run_dir.expanduser().resolve()
    payload = json.loads((run_dir / "continuation.json").read_text())
    if (Path(payload["run_dir"]).resolve() != run_dir
            or payload.get("grasp_contract") != "contact_v2"
            or payload.get("reward") != "staged_milestone"):
        raise ValueError("--retry-run requires a prepared contact-v2 stage-7 run.")
    if any((run_dir / "rl").rglob("*.pt")):
        raise ValueError("This run already has checkpoints; resume its latest.pt instead of retrying startup.")
    config = Path(payload["config"])
    raw = yaml.safe_load(config.read_text())
    if not raw["task"]["metadata"].get("staged_milestone_reward"):
        raise ValueError("The prepared config no longer enables phased milestone rewards.")
    initializer = payload.get("resume_checkpoint") or payload["warmstart"]
    if not Path(initializer).is_file():
        raise ValueError(f"Missing initializer: {initializer}")
    checksum = payload.get("config_sha256")
    if checksum and hashlib.sha256(config.read_bytes()).hexdigest() != checksum:
        raise ValueError("Prepared config changed; prepare a fresh continuation to rerun its preflight.")
    # Saved project/remote env overrides the shell in cli.train. Apply the HF
    # repair there as well, without touching task, optimizer or simulator args.
    repaired = copy.deepcopy(raw)
    for saved_env in (repaired.setdefault("remote", {}).setdefault("env_vars", {}),
                      repaired.setdefault("project", {}).get("env", {})):
        configure_saved_huggingface_env(saved_env)
    if repaired != raw:
        config.write_text(yaml.safe_dump(repaired, sort_keys=False))
    payload["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    payload["startup_retry"] = True
    payload["commands"][0][0] = sys.executable
    payload["commands"][1][0] = sys.executable
    (run_dir / "continuation.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run_logged(command: list[str], env: dict[str, str], path: Path) -> None:
    with path.open("a") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pipeline-dir", type=Path)
    source.add_argument("--resume-run", type=Path, help="Resume latest.pt and training state from a prior contact-v2 phased stage-7 run")
    source.add_argument("--retry-run", type=Path, help="Retry model-loading startup in an existing prepared run")
    parser.add_argument("--checkpoint", type=Path, help="Override the selected policy, or the full checkpoint when resuming")
    parser.add_argument("--run-name", default="widowx200_s7_contact_v2_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--steps", type=int, help="Total selected-action step cap, including restored steps on resume; default preserves the saved cap")
    parser.add_argument("--composed-horizon-decisions", type=int, help="Uncaught placement horizon floor in training and composed validation")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable validation target and plateau stopping; keep validation and checkpoints")
    parser.add_argument("--execute", action="store_true", help="Run CUDA preflight and then joint RL")
    args = parser.parse_args(argv)
    try:
        if args.retry_run:
            if (args.checkpoint is not None or args.steps is not None
                    or args.composed_horizon_decisions is not None or args.no_early_stop):
                raise ValueError("--retry-run preserves its initializer, budget and rules; omit overrides.")
            payload = retry_startup(args.retry_run)
        else:
            payload = prepare(args.pipeline_dir, args.run_name, checkpoint=args.checkpoint, steps=args.steps,
                              composed_horizon_decisions=args.composed_horizon_decisions,
                              no_early_stop=args.no_early_stop, resume_run=args.resume_run)
        env = continuation_env(payload)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    resuming = payload["initialization"] == "full_state_resume"
    initializer = payload.get("resume_checkpoint") or payload["warmstart"]
    print(f"[stage7] {'full-state resume' if resuming else 'weights-only initializer'}: {initializer}", flush=True)
    print("[stage7] phased reward: approach -> grasp -> lift -> carry -> placement", flush=True)
    state = "restored optimizer/curriculum/step count" if resuming else "fresh optimizer/curriculum; step 0"
    print(f"[stage7] contact_v2; {state}; total cap {payload['max_train_steps']} selected-action steps", flush=True)
    if "composed_horizon_min_decisions" in payload:
        print(f"[stage7] uncaught placement horizon: >= {payload['composed_horizon_min_decisions']} decisions (training + composed validation)", flush=True)
        print(f"[stage7] validation stopping: target={payload['early_stop_target']:g}, plateau window={payload['early_stop_plateau_validations']} (0 disables each)", flush=True)
    print(f"[stage7] run directory: {payload['run_dir']}", flush=True)
    print("[stage7] source pipeline and earlier stages preserved", flush=True)
    if env.get("RLVLA_HF_PUBLIC_MODELS_ONLY", "1") == "1":
        print("[stage7] Hugging Face: anonymous public models; cached credentials ignored", flush=True)
    if env.get("HF_HUB_OFFLINE") == "1":
        print("[stage7] Hugging Face: local model cache only", flush=True)
    for command in payload["commands"]:
        print("[stage7] " + shlex.join(command), flush=True)
    if not args.execute:
        print("[stage7] Prepared only. To execute these commands, set:", flush=True)
        mode_keys = (("RLVLA_SMOLVLA_RESUME_CHECKPOINT",) if resuming else
                     ("RLVLA_SMOLVLA_WARMSTART_CHECKPOINT", "RLVLA_SMOLVLA_RESET_LOG_STD"))
        for key in ("RLVLA_EMBODIMENT", *mode_keys):
            print(f"export {key}={shlex.quote(env[key])}")
        opposite = ("RLVLA_SMOLVLA_WARMSTART_CHECKPOINT RLVLA_SMOLVLA_RESET_LOG_STD" if resuming else
                    "RLVLA_SMOLVLA_RESUME_CHECKPOINT")
        print(f"unset {opposite} RLVLA_SMOLVLA_LEGACY_INIT_CHECKPOINT")
        return 0
    try:
        run_dir = Path(payload["run_dir"])
        preflight = run_dir / "preflight" / "preflight.json"
        passed = bool(payload.get("startup_retry") and preflight.is_file()
                      and json.loads(preflight.read_text()).get("passed") is True)
        if passed:
            print("[stage7] reusing passed preflight for this model-loading startup retry", flush=True)
        else:
            run_logged(payload["commands"][0], env, run_dir / "preflight.log")
        run_logged(payload["commands"][1], env, run_dir / "training.log")
    except subprocess.CalledProcessError as error:
        print(f"[stage7] Child failed with code {error.returncode}; see logs in {payload['run_dir']}", file=sys.stderr)
        return error.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
