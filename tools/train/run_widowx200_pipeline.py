#!/usr/bin/env python3
"""Run the whole WidowX-200 SmolVLA RL pipeline, stage by stage, unattended.

Reads ``configs/examples/widowx200_smolvla_pipeline.yaml``: a complete base
config plus a ``widowx200_pipeline:`` section that lists the stages. For each
stage this driver

1. writes the merged stage config to ``configs/widowx200_pipeline/`` (two
   levels below the repository, like ``configs/examples``, so every relative
   path in it resolves exactly as in the base file);
2. trains it with the production GRPO trainer, warm-started WEIGHTS-ONLY from
   the previous stage's policy (fresh optimizer, curriculum and step counter),
   and lets the trainer's own early-stop rule end it on its validation;
3. picks the stage's best validated checkpoint;
4. harvests a self-imitation bank for the skill from that checkpoint (executed
   actions, smoothed and replayed, with the camera frames), which also sets the
   skill's reference success;
5. measures every previously learned instruction on the new policy and, when
   one has degraded, runs self-imitation cycles: the pooled bank, re-primed
   under the current policy, is fitted by residual SFT, and the candidate is
   kept only if it repairs the degraded instructions without new damage.

All state lives in ``<pipeline-dir>/pipeline_state.json``. Re-running the same
command on the same ``--pipeline-dir`` resumes: finished stages are skipped and
an interrupted training stage continues from its ``latest.pt``.

``--smoke`` runs every stage and one forced SIL cycle at toy scale (64 worlds
per rank, two updates per stage) so the whole chain is exercised in about an
hour before the real run commits days to it.

Usage (on the GPU box, inside the cdpr-mjlab environment)::

    python tools/train/run_widowx200_pipeline.py \\
        --config configs/examples/widowx200_smolvla_pipeline.yaml \\
        --pipeline-dir runs/widowx200_pipeline_20261022

The launcher scripts/train_widowx200_pipeline_remote.sh wraps this with the
environment, the GPU preflight and the smoke pass.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.sil_artifacts import completed_replay, valid_npz

PIPELINE_KEY = "widowx200_pipeline"
GENERATED_CONFIG_DIR = ROOT / "configs" / "widowx200_pipeline"
EMBODIMENT = "widowx200"


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def log(message: str) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[pipeline {stamp}] {message}", flush=True)


def deep_merge(base: Any, override: Any) -> Any:
    """``override`` onto ``base``: mappings merge recursively, all else replaces."""

    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            merged[key] = deep_merge(merged.get(key), value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a mapping.")
    return data


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(payload), sort_keys=False, default_flow_style=None, width=100)
    path.write_text(text, encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover - not a checkout
        return "unknown"


class ChildFailed(RuntimeError):
    pass


_ACTIVE_CHILD: subprocess.Popen | None = None


def _forward_signal(signum: int, _frame: Any) -> None:
    child = _ACTIVE_CHILD
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + int(signum))


def run_command(
    command: Sequence[str],
    *,
    log_path: Path,
    env: Mapping[str, str],
    dry_run: bool = False,
) -> None:
    """Run one child to completion, teeing its output into ``log_path``.

    The child gets its own process group so an interrupt of this driver takes
    torchrun and both ranks down with it rather than orphaning GPU workers.
    """

    global _ACTIVE_CHILD
    printable = " ".join(str(part) for part in command)
    log(f"$ {printable}")
    log(f"  log: {log_path}")
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as sink:
        sink.write(f"\n===== {_dt.datetime.now().isoformat()} $ {printable}\n")
        sink.flush()
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        _ACTIVE_CHILD = process
        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sink.write(line)
            code = process.wait()
        finally:
            _ACTIVE_CHILD = None
    if code != 0:
        raise ChildFailed(f"exit code {code}: {printable} (see {log_path})")


# --------------------------------------------------------------------------
# Pipeline definition
# --------------------------------------------------------------------------


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config_path = Path(args.config).expanduser().resolve()
        raw = load_yaml(self.config_path)
        if PIPELINE_KEY not in raw:
            raise SystemExit(f"{self.config_path} has no `{PIPELINE_KEY}:` section.")
        self.spec: dict[str, Any] = raw.pop(PIPELINE_KEY)
        self.base: dict[str, Any] = raw
        self.dir = Path(args.pipeline_dir).expanduser().resolve()
        self.name = self.dir.name
        self.state_path = self.dir / "pipeline_state.json"
        self.smoke = bool(args.smoke)
        self.dry_run = bool(args.dry_run)
        self.python = sys.executable
        self.devices = [
            name.strip() for name in str(args.devices).split(",") if name.strip()
        ]
        self.sil = dict(self.spec.get("sil") or {})
        if self.smoke:
            self.sil.update(
                harvest_rounds=1,
                eval_rounds=1,
                frame_worlds=8,
                rows_per_instruction=400,
                sft_epochs=1,
                max_cycles=1,
            )
        self.state = self._load_state()

    # -- state -------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if bool(state.get("smoke")) != self.smoke:
                raise SystemExit("Smoke and production must use separate pipeline directories.")
            log(f"resuming {self.state_path}")
            if state.get("config") != str(self.config_path):
                log(
                    f"WARNING: state was created from {state.get('config')}, "
                    f"now running {self.config_path}"
                )
            return state
        return {
            "config": str(self.config_path),
            "git_commit_at_start": git_commit(),
            "created": _dt.datetime.now().isoformat(),
            "smoke": self.smoke,
            "current_policy": None,
            "references": {},
            "banks": {},
            "stages": {},
            "events": [],
        }

    def save_state(self) -> None:
        if self.dry_run:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def event(self, kind: str, **payload: Any) -> None:
        entry = {"time": _dt.datetime.now().isoformat(), "kind": kind, **payload}
        self.state.setdefault("events", []).append(entry)
        self.save_state()

    # -- environment ----------------------------------------------------------

    def child_env(self, **extra: str) -> dict[str, str]:
        env = dict(os.environ)
        env["RLVLA_EMBODIMENT"] = EMBODIMENT
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), *[part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]]
        )
        # Nothing from an earlier launch may leak into a stage.
        for key in (
            "RLVLA_SMOLVLA_RESUME_CHECKPOINT",
            "RLVLA_SMOLVLA_WARMSTART_CHECKPOINT",
            "RLVLA_SMOLVLA_LEGACY_INIT_CHECKPOINT",
            "RLVLA_SMOLVLA_POLICY_ARCHITECTURE",
            "RLVLA_SMOLVLA_RESET_LOG_STD",
            "RLVLA_SMOLVLA_MAX_TRAIN_STEPS",
            "RLVLA_SMOLVLA_MJWARP_MAX_UPDATES",
            "RLVLA_SMOLVLA_PPO_EPOCHS",
            "RLVLA_SMOLVLA_OPTIMIZER_LR_OVERRIDE",
            "RLVLA_SMOLVLA_PRIOR_NOISE_SCALE",
            "RLVLA_SMOLVLA_REFERENCE_ANCHOR_BANK",
            "RLVLA_CDPR_THREE_STAGE_SCENE_MANIFEST",
            "RLVLA_CDPR_DEMO_BANK",
            "RLVLA_TASK_METADATA_JSON",
            "RLVLA_MJWARP_ALLOW_CPU",
            "RLVLA_MJWARP_WORLDS_PER_RANK",
            "RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE",
            "RLVLA_MEMORY_REPORT",
        ):
            env.pop(key, None)
        env["RLVLA_SMOLVLA_NPROC_PER_NODE"] = str(max(1, len(self.devices)))
        if self.smoke:
            env["RLVLA_MJWARP_WORLDS_PER_RANK"] = "64"
            env["RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE"] = "64"
            env["RLVLA_SMOLVLA_MJWARP_MAX_UPDATES"] = "2"
        env.update({key: str(value) for key, value in extra.items()})
        return env

    # -- configs ---------------------------------------------------------------

    def skill(self, name: str) -> dict[str, Any]:
        skills = self.spec.get("skills") or {}
        if name not in skills:
            raise SystemExit(f"Unknown skill {name!r}; defined: {sorted(skills)}")
        return dict(skills[name])

    def validation_caps(self, skill_name: str) -> dict[str, float]:
        """Each instruction is evaluated at its skill's full distance."""

        owners = {
            instruction: float(spec.get("eval_cap", 0.2))
            for name, spec in self.spec["skills"].items() if name != "joint"
            for instruction in spec["instructions"]
        }
        return {name: owners[name] for name in self.skill(skill_name)["instructions"]}

    def phase(self, name: str) -> dict[str, Any]:
        phases = self.spec.get("phases") or {}
        if name not in phases:
            raise SystemExit(f"Unknown phase {name!r}; defined: {sorted(phases)}")
        return dict(phases[name] or {})

    def _smoke_overrides(self) -> dict[str, Any]:
        return {
            "training": {
                "rl": {
                    "args": {
                        "smolvla_compile_model": False,
                        "validation_every_steps": 1,
                        "validation_episodes_per_instruction": 16,
                        "composed_validation_episodes_per_instruction": 16,
                        "save_every_steps": 1,
                        "grpo_target_records_per_update": 128,
                        "grpo_max_groups_per_update": 16,
                        "grpo_target_informative_groups": 0,
                        "grpo_max_collection_seconds_per_update": 180,
                        "vla_update_max_records": 16,
                    }
                }
            }
        }

    def stage_config(self, stage: Mapping[str, Any]) -> dict[str, Any]:
        skill = self.skill(str(stage["skill"]))
        phase = self.phase(str(stage["phase"]))
        early = dict(stage.get("early_stop") or {})
        budget = int(stage.get("budget_steps", 10_000_000))
        stage_args = {
            "max_train_steps": budget,
            "validation_start_distance_caps": [
                f"{name}={cap:g}" for name, cap in self.validation_caps(str(stage["skill"])).items()
            ],
            "early_stop_metric": str(early.get("metric", "validation/success_rate")),
            "early_stop_target": float(early.get("target", 0.0)),
            "early_stop_patience": int(early.get("patience", 2)),
            "early_stop_plateau_validations": int(early.get("plateau", 0)),
            "early_stop_min_delta": float(early.get("min_delta", 0.02)),
            "early_stop_min_steps": int(early.get("min_steps", 0)),
        }
        if self.smoke:
            stage_args.update(early_stop_target=0.0, early_stop_plateau_validations=0, early_stop_min_steps=0)
        merged = deep_merge(self.base, skill.get("overrides") or {})
        merged = deep_merge(merged, phase.get("overrides") or {})
        merged = deep_merge(merged, {"training": {"rl": {"args": stage_args}}})
        merged = deep_merge(
            merged, {"project": {"name": f"{self.name}__{stage['name']}"}}
        )
        if self.smoke:
            merged = deep_merge(merged, self._smoke_overrides())
        return merged

    def eval_config(self, skill_name: str, *, composed: bool = False) -> dict[str, Any]:
        """A skill's task with every training aid off: the task itself.

        The success predicates do not depend on the reward phase, so one
        evaluation config per skill serves dense, binary and staged policies,
        which is what makes their rates comparable.
        """

        skill = self.skill(skill_name)
        merged = deep_merge(self.base, skill.get("overrides") or {})
        metadata = {
            "pick_up_prelifted_group_fraction": 0.0,
            "pick_up_aligned_group_fraction": 0.0,
            "sparse_binary_reward": False,
            "staged_milestone_reward": False,
        }
        if skill_name == "put_into":
            if composed:
                metadata.update(
                    placement_caught_object_fraction=0.0,
                    placement_caught_curriculum_enabled=False,
                    placement_uncaught_object_at_cell=True,
                    placement_grasp_horizon_min_decisions=40,
                    placement_wrong_drop_requires_lift=True,
                )
            else:
                metadata.update(placement_caught_object_fraction=1.0)
        merged = deep_merge(merged, {"task": {"metadata": metadata}})
        if self.smoke:
            merged = deep_merge(merged, self._smoke_overrides())
        return merged

    def write_config(self, label: str, payload: Mapping[str, Any]) -> Path:
        path = GENERATED_CONFIG_DIR / f"{self.name}__{label}.yaml"
        header = (
            f"# GENERATED by tools/train/run_widowx200_pipeline.py from\n"
            f"# {self.config_path.relative_to(ROOT) if self.config_path.is_relative_to(ROOT) else self.config_path}\n"
            f"# for pipeline {self.name}, {label}. Edit the source, not this file.\n"
        )
        if not self.dry_run or not path.exists():
            write_yaml(path, payload)
            path.write_text(header + path.read_text(encoding="utf-8"), encoding="utf-8")
        return path

    # -- evaluation helpers --------------------------------------------------

    def record(
        self,
        *,
        checkpoint: Path,
        config: Path,
        cap: float,
        rounds: int,
        output: Path,
        log_name: str,
        first_round: int = 0,
    ) -> dict[str, dict[str, float]]:
        """Deterministic record-mode rollouts; returns per-instruction rates."""

        from tools.audit.sil_artifacts import completed_recording

        if self.dry_run or not completed_recording(output, first_round=int(first_round), rounds=int(rounds)):
            command = [
                self.python, str(ROOT / "tools" / "audit" / "sil_record.py"),
                "--mode", "record",
                "--controller-workspace-from-config",
                "--checkpoint", str(checkpoint),
                "--config", str(config),
                "--start-distance-cap", f"{float(cap):.4f}",
                "--rounds", str(int(rounds)),
                "--round-index", str(int(first_round)),
                "--output", str(output),
            ]
            if self.sil.get("seed_torch") is not None:
                command += ["--seed-torch", str(int(self.sil["seed_torch"]))]
            if self.smoke:
                command += ["--worlds", "64", "--smolvla-microbatch", "64"]
            if len(self.devices) > 1 and int(rounds) > 1:
                command += ["--devices", ",".join(self.devices)]
            else:
                command += ["--device", self.devices[0] if self.devices else "cuda:0"]
            run_command(command, log_path=self.dir / "logs" / f"{log_name}.log", env=self.child_env(), dry_run=self.dry_run)
        if self.dry_run:
            return {}
        if not completed_recording(output, first_round=int(first_round), rounds=int(rounds)):
            raise ChildFailed(f"Incomplete recording under {output}: expected {rounds} rounds starting at {first_round}")
        return read_record_rates(output)

    # -- training --------------------------------------------------------------

    def train(self, stage: Mapping[str, Any], config: Path, init: Path | None) -> Path:
        name = str(stage["name"])
        run_name = f"{self.name}__{name}"
        run_dir = ROOT / "runs" / run_name
        latest = run_dir / "rl" / "latest.pt"
        complete = run_dir / "rl" / "TRAINING_COMPLETE"
        if complete.is_file():
            # Finished (early-stopped or at budget) before this driver could
            # record it. Resuming would train past the stop the run chose.
            log(f"{name}: training already complete ({complete})")
            return run_dir / "rl"
        extra: dict[str, str] = {}
        if getattr(self.args, "memory_check", False):
            extra.update(RLVLA_SMOLVLA_MJWARP_MAX_UPDATES="2", RLVLA_MEMORY_REPORT="1")
        phase = self.phase(str(stage["phase"]))
        if latest.is_file() and not self.dry_run:
            # An interrupted stage: continue it. MAX_TRAIN_STEPS is a total.
            log(f"{name}: resuming interrupted training from {latest}")
            extra["RLVLA_SMOLVLA_RESUME_CHECKPOINT"] = str(latest)
        elif init is not None:
            extra["RLVLA_SMOLVLA_WARMSTART_CHECKPOINT"] = str(init)
            if bool(phase.get("reset_log_std", False)):
                extra["RLVLA_SMOLVLA_RESET_LOG_STD"] = "1"
        command = [
            self.python, "-m", "rl_vla_bootstrapping.cli.train",
            "--config", str(config),
            "--stage", "rl",
            "--run-name", run_name,
            "--execute",
        ]
        run_command(command, log_path=self.dir / "logs" / f"{name}.log", env=self.child_env(**extra), dry_run=self.dry_run)
        if not self.dry_run:
            complete.write_text(_dt.datetime.now().isoformat() + "\n", encoding="utf-8")
        return run_dir / "rl"

    def memory_check_config(self) -> dict[str, Any]:
        """Production model/batches, with the largest rollout and refill budget."""

        joint = next(stage for stage in self.spec["stages"] if stage["skill"] == "joint")
        configs = [self.stage_config(stage) for stage in self.spec["stages"]]
        horizon = max(
            int(config["task"]["metadata"].get(key, 0))
            for config in configs
            for key in ("random_workspace_horizon_high", "curriculum_horizon_max", "placement_grasp_horizon_min_decisions")
        )
        groups = max(int(config["training"]["rl"]["args"]["grpo_max_groups_per_update"]) for config in configs)
        return deep_merge(self.stage_config(joint), {
            "task": {"metadata": {
                "curriculum_horizon_min": horizon, "curriculum_horizon_max": horizon,
                "random_workspace_horizon_low": horizon, "random_workspace_horizon_high": horizon,
                "placement_caught_curriculum_enabled": False, "placement_caught_object_fraction": 0.0,
            }},
            "training": {"rl": {"args": {
                "max_train_steps": 10**12, "early_stop_target": 0.0,
                "early_stop_plateau_validations": 0, "validation_every_steps": 1,
                "save_every_steps": 1, "grpo_max_groups_per_update": groups,
                "grpo_target_informative_groups": 10**9,
                "grpo_max_collection_seconds_per_update": 0,
            }}},
        })

    def run_memory_check(self) -> None:
        if self.smoke:
            raise SystemExit("The memory check must use production sizes, without --smoke.")
        config = self.memory_check_config()
        path = self.write_config("memory_check", config)
        stage = {"name": "memory_check", "phase": "staged"}
        rl_dir = self.train(stage, path, None)
        if self.dry_run:
            return
        args = config["training"]["rl"]["args"]
        expected_rounds = math.ceil(int(args["grpo_max_groups_per_update"]) / int(config["simulator"]["groups_per_rank"]))
        reports = []
        for rank in range(len(self.devices)):
            report_path = rl_dir / f"memory_rank{rank}.jsonl"
            rows = [json.loads(line) for line in report_path.read_text().splitlines() if line.strip()]
            if len(rows) < 2 or any(int(row["rounds_collected"]) < expected_rounds for row in rows[-2:]):
                raise ChildFailed(f"Memory check rank {rank} did not exercise two maximum-refill updates")
            if args.get("train_vla_lora") and any(int(row["vla_records"]) < int(args["vla_update_max_records"]) for row in rows[-2:]):
                raise ChildFailed(f"Memory check rank {rank} did not exercise a full LoRA update")
            reports.append({"rank": rank, "updates": rows[-2:]})
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "memory_report.json").write_text(json.dumps({
            "config": str(path), "git_commit": git_commit(), "ranks": reports,
            "horizon_decisions": config["task"]["metadata"]["curriculum_horizon_max"],
            "worlds_per_rank": config["simulator"]["worlds_per_rank"],
        }, indent=2), encoding="utf-8")
        log(f"Production-size memory check passed; {self.dir / 'memory_report.json'}")

    # -- the stage loop --------------------------------------------------------

    def run(self) -> None:
        stages = list(self.spec.get("stages") or [])
        if not stages:
            raise SystemExit("No stages defined.")
        only = set(self.args.only_stage or [])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_state()
        log(f"pipeline {self.name}: {len(stages)} stages, smoke={self.smoke}, devices={self.devices}")
        for index, stage in enumerate(stages):
            name = str(stage["name"])
            if only and name not in only:
                continue
            record = self.state["stages"].setdefault(name, {"status": "pending"})
            if record.get("status") == "done":
                log(f"{name}: done already, skipping")
                continue
            self.run_stage(index, stage, record)
        self.final_report()

    def run_stage(self, index: int, stage: Mapping[str, Any], record: dict[str, Any]) -> None:
        name = str(stage["name"])
        skill_name = str(stage["skill"])
        log(f"=== stage {index + 1}: {name} (skill {skill_name}, phase {stage['phase']}) ===")
        config = self.write_config(name, self.stage_config(stage))
        record["config"] = str(config)

        init: Path | None = None
        if str(stage.get("init", "previous")) != "base":
            current = self.state.get("current_policy")
            if not current:
                raise SystemExit(
                    f"{name} warm-starts from the previous stage's policy, but no "
                    "stage has produced one. Run the earlier stages first."
                )
            init = Path(current)
        record.setdefault("init", None if init is None else str(init))

        if record.get("status") in (None, "pending", "training"):
            record["status"] = "training"
            record["started"] = record.get("started") or _dt.datetime.now().isoformat()
            self.save_state()
            rl_dir = self.train(stage, config, init)
            record["rl_dir"] = str(rl_dir)
            if self.dry_run:
                # Nothing was trained; a placeholder lets the dry run print the
                # rest of the chain's commands.
                self.state["current_policy"] = str(rl_dir / f"<best checkpoint of {name}>")
                return
            best = select_best_checkpoint(
                rl_dir, str((stage.get("early_stop") or {}).get("metric", "validation/success_rate")),
                validation_caps=self.validation_caps(skill_name),
            )
            record.update(best)
            early = rl_dir / "early_stop.json"
            record["early_stop"] = json.loads(early.read_text()) if early.is_file() else None
            record["status"] = "trained"
            record["policy"] = best["best_checkpoint"]
            self.save_state()
            log(
                f"{name}: best checkpoint {best['best_checkpoint']} "
                f"({best['metric']}={best['best_value']}, step {best['best_step']}); "
                f"early stop: {record['early_stop']}"
            )

        policy = Path(record["policy"])
        skill = self.skill(skill_name)
        if skill_name != "joint" and record.get("status") == "trained":
            self.harvest(stage, skill_name, policy, record)
            record["status"] = "harvested"
            self.save_state()

        if record.get("status") in ("trained", "harvested"):
            if bool(stage.get("sil", True)) and bool(self.sil.get("enabled", True)):
                policy = self.retention(stage, skill_name, policy, record)
            record["policy"] = str(policy)
            record["status"] = "done"
            record["finished"] = _dt.datetime.now().isoformat()
            self.state["current_policy"] = str(policy)
            self.event("stage_done", stage=name, policy=str(policy))
            log(f"{name}: done; current policy {policy}")

    # -- harvest and references ------------------------------------------------

    def harvest(self, stage: Mapping[str, Any], skill_name: str, policy: Path, record: dict[str, Any]) -> None:
        """Record + smoothed replay with frames; sets the skill's references."""

        name = str(stage["name"])
        skill = self.skill(skill_name)
        cap = float(skill.get("eval_cap", 0.2))
        eval_config = self.write_config(f"eval_{skill_name}", self.eval_config(skill_name))
        base = self.dir / "banks" / skill_name / name
        rates = self.record(
            checkpoint=policy,
            config=eval_config,
            cap=cap,
            rounds=int(self.sil.get("harvest_rounds", 4)),
            output=base / "record",
            log_name=f"{name}_harvest_record",
        )
        record["harvest_rates"] = rates
        log(f"{name}: harvest rates {format_rates(rates)}")

        replays, frames = self.harvest_replays(
            policy, eval_config, cap, base / "record", base / "replay", name,
        )
        provenance = "policy"
        if self.smoke and not replays:
            # Two updates need not teach a skill. Exercise the data plumbing
            # with a clearly labelled scripted fixture, isolated in the smoke
            # directory; production banks always contain policy experience.
            oracle_dir = base / "smoke_oracle"
            if not (oracle_dir / "summary.json").is_file() or not has_successful_recording(oracle_dir):
                run_command([
                    self.python, str(ROOT / "tools/audit/sil_record.py"),
                    "--mode", "oracle", "--checkpoint", str(policy),
                    "--config", str(eval_config), "--controller-workspace-from-config",
                    "--start-distance-cap", str(cap), "--rounds", "1",
                    "--worlds", "64", "--smolvla-microbatch", "64",
                    "--seed-torch", "0", "--device", self.devices[0],
                    "--output", str(oracle_dir),
                ], log_path=self.dir / "logs" / f"{name}_smoke_oracle.log", env=self.child_env(), dry_run=self.dry_run)
            replays, frames = self.harvest_replays(
                policy, eval_config, cap, oracle_dir, base / "smoke_fixture_replay", name,
                smooth_window=1,
            )
            provenance = "oracle_smoke_fixture"
            if not replays and not self.dry_run:
                raise ChildFailed(f"{name}: smoke oracle produced no usable frames; SIL plumbing was not exercised")

        mean_rate = _mean([rates[i]["rate"] for i in skill["instructions"] if i in rates])
        previous = self.state["banks"].get(skill_name)
        if replays and frames and (previous is None or mean_rate >= float(previous.get("mean_rate", -1.0))):
            self.state["banks"][skill_name] = {
                "stage": name, "checkpoint": str(policy), "cap": cap,
                "replays": replays, "frames": frames, "mean_rate": mean_rate,
                "rates": rates, "provenance": provenance,
            }
            log(f"{name}: bank for {skill_name} now from this stage ({provenance}, mean rate {mean_rate:.3f})")
        elif not replays:
            record["empty_harvest"] = True
            self.event("empty_harvest", stage=name, skill=skill_name)
            log(f"{name}: WARNING no usable successful frames; previous bank retained, if any")

        for instruction in skill["instructions"]:
            if instruction not in rates:
                continue
            reference = self.state["references"].get(instruction)
            value = float(rates[instruction]["rate"])
            if reference is None or value > float(reference["rate"]):
                self.state["references"][instruction] = {
                    "rate": value, "stage": name, "checkpoint": str(policy), "skill": skill_name,
                }
        self.save_state()

    def harvest_replays(
        self, policy: Path, eval_config: Path, cap: float,
        source_dir: Path, replay_dir: Path, name: str, *, smooth_window: int | None = None,
    ) -> tuple[list[str], list[str]]:
        import numpy as np

        replays, frames = [], []
        for source in sorted(source_dir.glob("record_*.npz")):
            with np.load(source, allow_pickle=False) as data:
                if not bool(data["success"].any()):
                    log(f"{name}: skip {source.name}: no successful episodes")
                    continue
            stem = f"{source.parent.name}_{source.stem}"
            replay_path = replay_dir / f"replay_{stem}.npz"
            frames_path = replay_dir / f"frames_{stem}.npz"
            completion = completed_replay(replay_path, frames_path)
            command = [
                self.python, str(ROOT / "tools" / "audit" / "sil_record.py"),
                "--mode", "replay",
                "--controller-workspace-from-config",
                "--checkpoint", str(policy),
                "--config", str(eval_config),
                "--actions", str(source),
                "--start-distance-cap", f"{cap:.4f}",
                "--smooth", "moving_average",
                "--smooth-window", str(smooth_window if smooth_window is not None else int(self.sil.get("smooth_window", 5))),
                "--record-frames",
                "--frame-worlds", str(int(self.sil.get("frame_worlds", 96))),
                "--device", self.devices[0] if self.devices else "cuda:0",
                "--output", str(replay_dir),
            ]
            if self.sil.get("seed_torch") is not None:
                command += ["--seed-torch", str(int(self.sil["seed_torch"]))]
            if self.smoke:
                command += ["--worlds", "64", "--smolvla-microbatch", "64"]
            if completion is None:
                run_command(command, log_path=self.dir / "logs" / f"{name}_harvest_replay.log", env=self.child_env(), dry_run=self.dry_run)
                completion = completed_replay(replay_path, frames_path)
            if completion is not None and int(completion["frame_successes"]) > 0:
                replays.append(str(replay_path))
                frames.append(str(frames_path))
        return replays, frames

    def raise_references(self, rates: Mapping[str, Mapping[str, float]], policy: Path, source: str) -> None:
        """A reference is the best rate any accepted policy has shown."""

        for instruction, entry in rates.items():
            reference = self.state["references"].get(instruction)
            if reference is not None and float(entry["rate"]) > float(reference["rate"]):
                reference.update(rate=float(entry["rate"]), stage=source, checkpoint=str(policy))
        self.save_state()

    def learned_skills(self) -> list[str]:
        skills = []
        for entry in self.state["references"].values():
            if entry["skill"] not in skills:
                skills.append(entry["skill"])
        return skills

    def evaluate_all(self, policy: Path, label: str) -> dict[str, dict[str, float]]:
        rates: dict[str, dict[str, float]] = {}
        for skill_name in self.learned_skills():
            skill = self.skill(skill_name)
            config = self.write_config(f"eval_{skill_name}", self.eval_config(skill_name))
            rates.update(
                self.record(
                    checkpoint=policy,
                    config=config,
                    cap=float(skill.get("eval_cap", 0.2)),
                    rounds=int(self.sil.get("eval_rounds", 2)),
                    output=self.dir / "eval" / label / skill_name,
                    log_name=f"eval_{label}_{skill_name}",
                )
            )
        return rates

    def degraded(self, rates: Mapping[str, Mapping[str, float]]) -> list[str]:
        out = []
        for instruction, reference in self.state["references"].items():
            if instruction not in rates:
                continue
            ref = float(reference["rate"])
            now = float(rates[instruction]["rate"])
            tolerance = max(float(self.sil.get("degrade_abs", 0.08)), float(self.sil.get("degrade_rel", 0.2)) * ref)
            if ref - now > tolerance:
                out.append(instruction)
        return out

    # -- self-imitation retention -----------------------------------------------

    def retention(self, stage: Mapping[str, Any], skill_name: str, policy: Path, record: dict[str, Any]) -> Path:
        name = str(stage["name"])
        if len(self.learned_skills()) < 2 and not self.smoke:
            log(f"{name}: one skill learned so far; nothing to retain")
            return policy
        rates = self.evaluate_all(policy, f"{name}_policy")
        record["retention_rates"] = rates
        degraded = self.degraded(rates)
        log(f"{name}: retention {format_rates(rates)}; degraded: {degraded or 'none'}")
        force = self.smoke and name == "s2_pick_up_dense"
        if not degraded and not force:
            self.save_state()
            return policy
        if not self.state["banks"]:
            log(f"{name}: degraded but no bank exists yet; continuing")
            return policy

        cycles = record.setdefault("sil_cycles", [])
        current = policy
        current_rates = rates
        for previous in cycles:  # a resumed stage continues from its last accepted cycle
            if previous.get("accepted"):
                current = Path(previous["candidate"])
                current_rates = previous["candidate_rates"]
                degraded = self.degraded(current_rates)
        if cycles and (not cycles[-1].get("accepted") or not degraded):
            record["retention_rates_after_sil"] = current_rates
            return current
        for cycle in range(len(cycles), int(self.sil.get("max_cycles", 2))):
            label = f"{name}_sil{cycle + 1}"
            result = self.sil_cycle(label, current, current_rates, degraded or list(current_rates))
            cycles.append(result)
            self.save_state()
            if result.get("accepted"):
                current = Path(result["candidate"])
                current_rates = result["candidate_rates"]
                self.raise_references(current_rates, current, label)
                degraded = self.degraded(current_rates)
                self.event("sil_accepted", stage=name, cycle=cycle + 1, candidate=str(current))
                if not degraded:
                    break
            else:
                self.event("sil_rejected", stage=name, cycle=cycle + 1, reason=result.get("reason"))
                break
        record["retention_rates_after_sil"] = current_rates
        return current

    def sil_cycle(
        self,
        label: str,
        policy: Path,
        policy_rates: Mapping[str, Mapping[str, float]],
        degraded: Sequence[str],
    ) -> dict[str, Any]:
        base = self.dir / "sil" / label
        banks = list(self.state["banks"].values())
        if not self.smoke and any(bank.get("provenance") == "oracle_smoke_fixture" for bank in banks):
            raise ChildFailed("Production SIL cannot consume scripted smoke fixtures")
        replays = [path for bank in banks for path in bank["replays"]]
        frames = [path for bank in banks for path in bank["frames"]]
        log(f"{label}: SIL over {len(banks)} banks ({len(replays)} replays) for {list(degraded)}")
        env = self.child_env()
        device = self.devices[0] if self.devices else "cuda:0"
        logs = self.dir / "logs"

        dataset = base / "dataset" / "demonstrations.npz"
        if not valid_npz(dataset, ("state", "prior", "action", "episode_uid")):
            run_command(
                [
                    self.python, str(ROOT / "tools" / "audit" / "sil_record.py"),
                    "--mode", "dataset",
                    "--inputs", *replays,
                    "--require-frames", *frames,
                    "--rows-per-instruction", str(int(self.sil.get("rows_per_instruction", 12000))),
                    "--output", str(dataset.parent),
                ],
                log_path=logs / f"{label}_dataset.log", env=env, dry_run=self.dry_run,
            )
        refreshed = base / "refreshed" / "demonstrations.npz"
        if not valid_npz(refreshed, ("state", "prior", "action", "episode_uid")):
            run_command(
                [
                    self.python, str(ROOT / "tools" / "audit" / "sil_refresh_priors.py"),
                    "--dataset", str(dataset),
                    "--frames", *frames,
                    "--checkpoint", str(policy),
                    # An original-label retention bank: every instruction keeps
                    # its own prompt.
                    "--final-prompt-prefix", "",
                    "--device", device,
                    "--output", str(refreshed.parent),
                ],
                log_path=logs / f"{label}_refresh.log", env=env, dry_run=self.dry_run,
            )
        adapter = base / "sft" / "sil_sft_adapter.pt"
        report = base / "sft" / "sft_report.json"
        if not report.is_file():
            run_command(
                [
                    self.python, str(ROOT / "tools" / "audit" / "sil_sft.py"),
                    "--dataset", str(refreshed),
                    "--checkpoint", str(policy),
                    "--epochs", str(int(self.sil.get("sft_epochs", 20))),
                    "--device", device,
                    "--output", str(adapter.parent),
                ],
                log_path=logs / f"{label}_sft.log", env=env, dry_run=self.dry_run,
            )
        if self.dry_run:
            return {"accepted": False, "reason": "dry run"}
        if not adapter.is_file():
            return {"accepted": False, "reason": "sil_sft kept the initializer (no epoch beat it)"}

        candidate_rates = self.evaluate_all(adapter, f"{label}_candidate")
        gains = {
            name: float(candidate_rates[name]["rate"]) - float(policy_rates[name]["rate"])
            for name in candidate_rates
            if name in policy_rates
        }
        degraded_gain = _mean([gains[name] for name in degraded if name in gains])
        collateral = {
            name: -gain
            for name, gain in gains.items()
            if name not in degraded and -gain > float(self.sil.get("max_collateral", 0.05))
        }
        accepted = (
            math.isfinite(degraded_gain)
            and degraded_gain >= float(self.sil.get("accept_min_gain", 0.02))
            and not collateral
        )
        reason = (
            f"degraded mean gain {degraded_gain:+.3f}; collateral losses {collateral or 'none'}"
        )
        log(f"{label}: {'ACCEPT' if accepted else 'REJECT'} -- {reason}; gains {json.dumps({k: round(v, 3) for k, v in gains.items()})}")
        return {
            "label": label,
            "candidate": str(adapter),
            "candidate_rates": candidate_rates,
            "policy_rates": dict(policy_rates),
            "gains": gains,
            "accepted": bool(accepted),
            "reason": reason,
        }

    # -- the end -----------------------------------------------------------------

    def final_report(self) -> None:
        if self.dry_run:
            return
        policy = self.state.get("current_policy")
        if not policy or self.state.get("final_report"):
            return
        stages = self.spec.get("stages") or []
        if any(self.state["stages"].get(str(s["name"]), {}).get("status") != "done" for s in stages):
            return
        policy = Path(policy)
        report: dict[str, Any] = {"policy": str(policy)}
        for skill_name, composed in (("move_to", False), ("pick_up", False), ("put_into", False), ("put_into", True)):
            skill = self.skill(skill_name)
            label = f"{skill_name}{'_composed' if composed else ''}"
            config = self.write_config(f"eval_{label}", self.eval_config(skill_name, composed=composed))
            report[label] = self.record(
                checkpoint=policy,
                config=config,
                cap=float(skill.get("eval_cap", 0.2)),
                rounds=max(2, int(self.sil.get("eval_rounds", 2))),
                output=self.dir / "eval" / "final" / label,
                log_name=f"eval_final_{label}",
            )
        final = self.dir / "final_policy.pt"
        if not final.exists():
            try:
                final.symlink_to(policy)
            except OSError:
                pass
        report["references"] = self.state["references"]
        (self.dir / "final_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        self.state["final_report"] = str(self.dir / "final_report.json")
        self.save_state()
        lines = [f"# WidowX-200 pipeline {self.name}", "", f"Final policy: `{policy}`", ""]
        lines.append("| evaluation | instruction | success | episodes |")
        lines.append("|---|---|---:|---:|")
        for label in ("move_to", "pick_up", "put_into", "put_into_composed"):
            for instruction, entry in sorted((report.get(label) or {}).items()):
                lines.append(f"| {label} | {instruction} | {entry['rate']:.3f} | {entry['episodes']} |")
        lines += ["", "## Stages", ""]
        for stage in stages:
            entry = self.state["stages"].get(str(stage["name"]), {})
            lines.append(
                f"- **{stage['name']}**: best {entry.get('metric')}={entry.get('best_value')} at step "
                f"{entry.get('best_step')}; early stop {(entry.get('early_stop') or {}).get('reason')}; "
                f"SIL cycles {len(entry.get('sil_cycles') or [])}; policy `{entry.get('policy')}`"
            )
        (self.dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"final report: {self.dir / 'final_report.md'}")
        print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------
# Reading results
# --------------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else float("nan")


def format_rates(rates: Mapping[str, Mapping[str, float]]) -> str:
    return ", ".join(f"{name}={entry['rate']:.3f}/{entry['episodes']}" for name, entry in sorted(rates.items()))


def has_successful_recording(directory: Path) -> bool:
    """A cached zero-success oracle run is not a usable smoke fixture."""

    import numpy as np
    import zipfile

    for path in sorted(directory.glob("record_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                if bool(data["success"].any()):
                    return True
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            continue
    return False


def read_record_rates(directory: Path) -> dict[str, dict[str, float]]:
    """Per-instruction success pooled on counts over every recorded round."""

    from tools.audit.sil_eval_table import _collect

    collected = _collect(str(directory))
    out: dict[str, dict[str, float]] = {}
    for name, bucket in collected["by_instruction"].items():
        episodes = int(bucket["episodes"])
        out[name] = {
            "rate": (int(bucket["successes"]) / episodes) if episodes else 0.0,
            "episodes": episodes,
            "successes": int(bucket["successes"]),
        }
    return out


def select_best_checkpoint(
    rl_dir: Path, metric: str, *, validation_caps: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """The validated checkpoint with the best value of the stage's metric.

    Every validation forces a checkpoint at the same global step, so each
    validation row names an adapter on disk. Ties go to the later step. A run
    with no validation (it stopped before the first one) falls back to the
    newest checkpoint.
    """

    from rl_vla_bootstrapping.policy.early_stopping import early_stop_metric_value, validation_matches_caps

    rl_dir = Path(rl_dir)
    best: tuple[float, int, Path] | None = None
    path = rl_dir / "validation.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if validation_caps is not None and not validation_matches_caps(row, validation_caps):
                continue
            value = early_stop_metric_value(row, metric)
            step = int(row.get("global_step", -1))
            checkpoint = rl_dir / f"step_{step:07d}" / "smolvla_grpo_adapter.pt"
            if not math.isfinite(value) or not checkpoint.is_file():
                continue
            if best is None or (value, step) >= (best[0], best[1]):
                best = (value, step, checkpoint)
    if best is None:
        if validation_caps is not None:
            raise ChildFailed(f"{rl_dir}: no checkpoint validated at the required fixed distances; cannot rank curriculum-only validation")
        steps = sorted(rl_dir.glob("step_*/smolvla_grpo_adapter.pt"), key=lambda p: int(p.parent.name.split("_", 1)[1]))
        if not steps:
            raise ChildFailed(f"{rl_dir} holds no checkpoint at all.")
        newest = steps[-1]
        return {
            "best_checkpoint": str(newest),
            "best_value": None,
            "best_step": int(newest.parent.name.split("_", 1)[1]),
            "metric": metric,
        }
    return {
        "best_checkpoint": str(best[2]),
        "best_value": round(best[0], 5),
        "best_step": best[1],
        "metric": metric,
    }


# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "examples" / "widowx200_smolvla_pipeline.yaml"))
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--smoke", action="store_true", help="Toy-scale pass over every stage and one SIL cycle.")
    parser.add_argument("--memory-check", action="store_true", help="Two production-size updates at the maximum horizon and refill budget, with per-rank memory reports.")
    parser.add_argument("--dry-run", action="store_true", help="Write the stage configs and print the commands only.")
    parser.add_argument("--only-stage", action="append", default=[], help="Run only these stages (repeatable).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)
    os.environ["RLVLA_EMBODIMENT"] = EMBODIMENT
    pipeline = Pipeline(args)
    try:
        if args.memory_check:
            pipeline.run_memory_check()
        else:
            pipeline.run()
    except ChildFailed as error:
        pipeline.event("failed", error=str(error))
        log(f"FAILED: {error}")
        log(f"Re-run the same command to resume from {pipeline.state_path}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
