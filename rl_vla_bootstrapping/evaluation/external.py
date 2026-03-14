from __future__ import annotations

from pathlib import Path

from rl_vla_bootstrapping.core.commands import StagePlan, append_cli_arg
from rl_vla_bootstrapping.core.specs import BenchmarkSpec, ProjectConfig


def build_benchmark_plan(config: ProjectConfig, run_dir: Path, benchmark: BenchmarkSpec) -> StagePlan:
    if not benchmark.script_path:
        raise ValueError(f"Benchmark `{benchmark.name}` does not define `script_path`.")

    script_path = config.resolve_path(benchmark.script_path)
    if script_path is None:
        raise ValueError(f"Could not resolve script path for benchmark `{benchmark.name}`.")

    argv = [config.project.python_executable, str(script_path)]
    args = dict(benchmark.args)
    args.setdefault("run_root_dir", str(run_dir / "eval" / benchmark.name))
    args.setdefault("checkpoint_dir", str(run_dir / "rl"))
    args.setdefault(
        "action_codec",
        str(run_dir / "manifests" / config.policy.action_codec.export_filename),
    )
    args.setdefault("robot_config", str(config.config_path))
    for key, value in args.items():
        append_cli_arg(argv, key, value)

    env = dict(config.project.env)
    env.update(config.remote.env_vars)
    env.update(benchmark.env)
    return StagePlan(
        name=f"eval:{benchmark.name}",
        kind="external_python",
        command=argv,
        cwd=str(script_path.parent),
        env=env,
        notes=[f"External benchmark stage for `{benchmark.name}`."],
        artifact_paths=[str(run_dir / "eval" / benchmark.name)],
    )
