from __future__ import annotations

import shlex
import subprocess
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def option_name(key: str, *, preserve_underscores: bool = False) -> str:
    if preserve_underscores:
        return "--" + key
    return "--" + key.replace("_", "-")


def append_cli_arg(
    argv: list[str],
    key: str,
    value: Any,
    *,
    preserve_underscores: bool = False,
) -> None:
    if value is None:
        return

    flag = option_name(key, preserve_underscores=preserve_underscores)
    if isinstance(value, bool):
        negative_suffix = key if preserve_underscores else key.replace("_", "-")
        argv.append(flag if value else f"--no-{negative_suffix}")
        return

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return
        argv.append(flag)
        argv.extend(str(item) for item in value)
        return

    argv.extend([flag, str(value)])


def format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


@dataclass
class StagePlan:
    name: str
    kind: str
    command: list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    notes: list[str] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)

    def pretty_command(self) -> str:
        if not self.command:
            return ""
        return format_command(self.command)


def run_stage(plan: StagePlan) -> int:
    if not plan.command:
        return 0
    child_env = dict(os.environ)
    child_env.update(plan.env)
    completed = subprocess.run(
        plan.command,
        cwd=plan.cwd,
        env=child_env,
        check=False,
    )
    return int(completed.returncode)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
