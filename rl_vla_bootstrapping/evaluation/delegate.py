from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def _resolve_repo_path(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()


def parse_common_args(prog: str, benchmark_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog, description=f"Delegate {benchmark_name} benchmark runs.")
    parser.add_argument("--benchmark-repo", required=True, help="Path to the benchmark repository root.")
    parser.add_argument("--entry-script", required=True, help="Path relative to the benchmark repo or absolute.")
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--run-root-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--action-codec", default=None)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument("--task-suite", default="all")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--forward-arg",
        action="append",
        default=[],
        help="Extra KEY=VALUE pair forwarded as `--KEY VALUE` to the benchmark entry script.",
    )
    return parser.parse_args()


def run_delegate(args: argparse.Namespace, benchmark_name: str) -> int:
    repo_root = _resolve_repo_path(args.benchmark_repo)
    if not repo_root.exists():
        raise FileNotFoundError(
            f"{benchmark_name} repository not found: {repo_root}. "
            "Stage it with `rl_vla_bootstrapping.cli.assets` or update the config path."
        )

    entry_path = Path(args.entry_script).expanduser()
    if not entry_path.is_absolute():
        entry_path = (repo_root / entry_path).resolve()
    if not entry_path.exists():
        raise FileNotFoundError(f"{benchmark_name} entry script not found: {entry_path}")

    argv = [args.python_executable, str(entry_path), "--task-suite", str(args.task_suite), "--episodes", str(args.episodes)]
    if args.checkpoint_dir:
        argv.extend(["--checkpoint-dir", str(args.checkpoint_dir)])
    if args.action_codec:
        argv.extend(["--action-codec", str(args.action_codec)])
    if args.robot_config:
        argv.extend(["--robot-config", str(args.robot_config)])

    for item in args.forward_arg:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE in --forward-arg, got: {item}")
        key, value = item.split("=", 1)
        argv.extend([f"--{key.replace('_', '-')}", value])

    print("Benchmark command:")
    print(" ".join(shlex.quote(part) for part in argv))
    if args.dry_run:
        return 0

    completed = subprocess.run(argv, cwd=str(repo_root), env=dict(os.environ), check=False)
    return int(completed.returncode)
