from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from rl_vla_bootstrapping.core.commands import StagePlan, ensure_directory, run_stage
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.core.specs import ProjectConfig
from rl_vla_bootstrapping.embodiments.mujoco import MujocoEmbodiment
from rl_vla_bootstrapping.evaluation.external import build_benchmark_plan
from rl_vla_bootstrapping.policy.openvla_oft import (
    build_openvla_rl_plan,
    build_openvla_sft_plan,
)
from rl_vla_bootstrapping.simulation.preview import render_preview


def _stage_set(values: Iterable[str]) -> set[str]:
    out = {str(value).lower() for value in values}
    if "all" in out:
        return {"preview", "rl", "sft", "eval"}
    return out


class BootstrapPipeline:
    def __init__(self, config: ProjectConfig):
        self.config = config

    @classmethod
    def from_path(cls, config_path: str | Path) -> "BootstrapPipeline":
        return cls(load_project_config(config_path))

    def validate(self) -> list[str]:
        errors = []
        embodiment = MujocoEmbodiment(config=self.config, spec=self.config.embodiment)
        errors.extend(embodiment.validate())

        codec = self.config.build_action_codec()
        if codec.action_dim != len(self.config.embodiment.action_adapter.common_action_keys):
            errors.append("Action codec dimensionality does not match embodiment action keys.")

        for raw in (
            self.config.repos.openvla_oft,
            self.config.repos.dataset_repo,
            self.config.repos.embodiment_repo,
            self.config.policy.repo_path,
            self.config.policy.rl_script,
            self.config.policy.sft_script,
            self.config.simulation.catalog_path,
        ):
            path = self.config.resolve_path(raw)
            if raw and path is not None and not path.exists():
                errors.append(f"Configured path does not exist: {path}")

        return errors

    def make_run_dir(self, run_name: str | None = None) -> Path:
        root = ensure_directory(self.config.resolve_path(self.config.project.output_root) or Path("runs"))
        if run_name:
            final_name = run_name
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_name = f"{self.config.project.name}_{stamp}"
        return ensure_directory(root / final_name)

    def export_manifests(self, run_dir: Path) -> dict[str, Path]:
        manifest_dir = ensure_directory(run_dir / "manifests")
        codec_path = self.config.build_action_codec().export_manifest(
            manifest_dir / self.config.policy.action_codec.export_filename
        )

        payload = asdict(self.config)
        payload["config_path"] = self.config.config_path.as_posix()
        resolved_path = manifest_dir / "project_config_resolved.json"
        resolved_path.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True), encoding="utf-8")
        return {
            "action_codec": codec_path,
            "project_config": resolved_path,
        }

    def build_stage_plans(self, run_dir: Path, requested_stages: Iterable[str]) -> list[StagePlan]:
        stages = _stage_set(requested_stages)
        plans: list[StagePlan] = []

        if "preview" in stages or ("rl" in stages and self.config.training.preview_before_rl):
            plans.append(
                StagePlan(
                    name="preview",
                    kind="internal_preview",
                    command=None,
                    cwd=str(run_dir),
                    notes=["Preview stage renders the robot and configured cameras before training."],
                    artifact_paths=[str(run_dir / "preview")],
                )
            )

        if "rl" in stages and self.config.training.rl.enabled:
            if self.config.policy.type == "openvla_oft":
                plans.append(build_openvla_rl_plan(self.config, run_dir))
            else:
                raise ValueError(f"Unsupported policy type for RL stage: {self.config.policy.type}")

        if "sft" in stages and self.config.training.sft.enabled:
            if self.config.policy.type == "openvla_oft":
                plans.append(build_openvla_sft_plan(self.config, run_dir))
            else:
                raise ValueError(f"Unsupported policy type for SFT stage: {self.config.policy.type}")

        if "eval" in stages:
            for benchmark in self.config.evaluation.benchmarks:
                if benchmark.enabled:
                    plans.append(build_benchmark_plan(self.config, run_dir, benchmark))

        return plans

    def execute_stage(self, plan: StagePlan, run_dir: Path) -> int:
        if plan.kind == "internal_preview":
            try:
                render_preview(self.config, run_dir / "preview")
            except ModuleNotFoundError as exc:
                missing = exc.name or str(exc)
                raise RuntimeError(
                    "Preview stage is missing a simulator dependency. "
                    f"Install `{missing}` in the runtime environment for this embodiment."
                ) from exc
            return 0
        return run_stage(plan)

    def execute(self, run_dir: Path, requested_stages: Iterable[str]) -> list[tuple[StagePlan, int]]:
        results: list[tuple[StagePlan, int]] = []
        for plan in self.build_stage_plans(run_dir, requested_stages):
            code = self.execute_stage(plan, run_dir)
            results.append((plan, code))
            if code != 0:
                break
        return results
