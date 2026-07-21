#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_PROJECT = "nurtdinov-team/CDPR"
DEFAULT_TASK_NAME = "cdpr-rl-a100-smoke"
DEFAULT_CONFIG = "configs/examples/cdpr_openvla_clearml_a100_smoke.yaml"
DEFAULT_OPENVLA_REPO = "https://github.com/damurka5/openvla-oft.git"
DEFAULT_OPENVLA_BRANCH = "a40"
DEFAULT_OPENVLA_PATH = "/root/repo/openvla-oft"


def _expand_key_value_args(argv: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for item in argv:
        if not item.startswith("-") and "=" in item:
            key, value = item.split("=", 1)
            expanded.extend(["--" + key.replace("_", "-"), value])
            continue
        expanded.append(item)
    return expanded


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ClearML launcher for a short one-GPU CDPR OpenVLA RL smoke run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-name", "--project_name", default=DEFAULT_PROJECT)
    parser.add_argument("--task-name", "--task_name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", default="rl")
    parser.add_argument("--run-name", "--run_name", default=None)
    parser.add_argument("--asset-dataset-id", "--asset_dataset_id", default=None)
    parser.add_argument("--asset-dataset-name", "--asset_dataset_name", default=None)
    parser.add_argument("--asset-dataset-project", "--asset_dataset_project", default=DEFAULT_PROJECT)
    parser.add_argument("--openvla-repo-url", "--openvla_repo_url", default=DEFAULT_OPENVLA_REPO)
    parser.add_argument("--openvla-branch", "--openvla_branch", default=DEFAULT_OPENVLA_BRANCH)
    parser.add_argument("--openvla-commit", "--openvla_commit", default=None)
    parser.add_argument("--openvla-path", "--openvla_path", default=DEFAULT_OPENVLA_PATH)
    parser.add_argument(
        "--artifact-upload-interval-minutes",
        "--artifact_upload_interval_minutes",
        type=float,
        default=30.0,
        help="Periodically upload newly created checkpoint directories while training is still running.",
    )
    parser.add_argument(
        "--disable-periodic-artifact-upload",
        "--disable_periodic_artifact_upload",
        action="store_true",
        help="Only upload run artifacts at process shutdown.",
    )
    parser.add_argument(
        "--skip-assets",
        "--skip_assets",
        action="store_true",
        help="Use already staged assets under assets/externals instead of downloading a ClearML Dataset.",
    )
    args, unknown = parser.parse_known_args(_expand_key_value_args(argv))
    if unknown:
        print(f"[clearml-cdpr] Ignoring unknown launcher args: {unknown}", flush=True)
    return args


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    prefix = f"(cd {cwd} &&) " if cwd else ""
    print(f"[clearml-cdpr] $ {prefix}{shlex.join([str(part) for part in cmd])}", flush=True)
    completed = subprocess.run([str(part) for part in cmd], cwd=str(cwd) if cwd else None, env=env, check=False)
    return int(completed.returncode)


def _check_call(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    code = _run(cmd, cwd=cwd)
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {shlex.join([str(part) for part in cmd])}")


def _load_clearml() -> tuple[Any, Any]:
    try:
        from clearml import Dataset, Task
    except ImportError as exc:
        raise SystemExit("clearml is not installed. Use the ClearML requirements file for this task.") from exc
    return Dataset, Task


def _connected_args(task: Any, args: argparse.Namespace) -> argparse.Namespace:
    params = vars(args).copy()
    try:
        task.connect(params, name="clearml_cdpr_rl")
    except TypeError:
        task.connect(params)
    return argparse.Namespace(**params)


def _ensure_compat_repo_link(repo_root: Path) -> None:
    compat = Path("/root/repo/RL_VLA_Bootstrapping")
    if repo_root == compat:
        return
    try:
        compat.parent.mkdir(parents=True, exist_ok=True)
        if compat.is_symlink() and compat.resolve() == repo_root:
            return
        if compat.exists():
            return
        os.symlink(repo_root, compat, target_is_directory=True)
        print(f"[clearml-cdpr] Linked {compat} -> {repo_root}", flush=True)
    except OSError as exc:
        print(f"[clearml-cdpr] WARNING: could not create compatibility repo link: {exc}", flush=True)


def ensure_openvla_repo(path: Path, *, repo_url: str, branch: str | None, commit: str | None) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        if not (path / ".git").exists():
            raise RuntimeError(f"OpenVLA path exists but is not a git checkout: {path}")
        _check_call(["git", "fetch", "--all", "--tags", "--prune"], cwd=path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = ["git", "clone"]
        if branch and not commit:
            clone_cmd.extend(["--branch", branch, "--depth", "1"])
        clone_cmd.extend([repo_url, str(path)])
        _check_call(clone_cmd)

    if branch:
        _check_call(["git", "fetch", "origin", branch], cwd=path)
    if commit:
        _check_call(["git", "checkout", commit], cwd=path)
    elif branch:
        _check_call(["git", "checkout", branch], cwd=path)
        _check_call(["git", "pull", "--ff-only", "origin", branch], cwd=path)

    code = _run(["git", "rev-parse", "--short", "HEAD"], cwd=path)
    if code != 0:
        raise RuntimeError(f"Could not inspect OpenVLA checkout at {path}")


def _looks_like_ycb(path: Path) -> bool:
    return (path / "apple").is_dir() and (path / "banana").is_dir()


def _looks_like_libero_assets(path: Path) -> bool:
    return (path / "textures").is_dir() and (
        (path / "stable_hope_objects").is_dir() or (path / "stable_scanned_objects").is_dir()
    )


def _find_asset_dir(root: Path, candidates: Sequence[Path], predicate: Callable[[Path], bool], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists() and predicate(candidate):
            return candidate.resolve()

    for candidate in root.rglob("*"):
        if candidate.is_dir() and predicate(candidate):
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find {label} assets under ClearML dataset root: {root}")


def _link_dir(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        try:
            if target.resolve() == source.resolve():
                return
        except OSError:
            pass
        try:
            target.rmdir()
        except OSError as exc:
            raise RuntimeError(f"Refusing to replace non-empty asset directory: {target}") from exc

    os.symlink(source, target, target_is_directory=True)
    print(f"[clearml-cdpr] Linked {target} -> {source}", flush=True)


def _asset_targets_are_ready(repo_root: Path) -> bool:
    return _looks_like_ycb(repo_root / "assets" / "externals" / "ycb") and _looks_like_libero_assets(
        repo_root / "assets" / "externals" / "libero"
    )


def stage_assets(repo_root: Path, Dataset: Any, args: argparse.Namespace) -> None:
    if args.skip_assets:
        if not _asset_targets_are_ready(repo_root):
            raise RuntimeError("--skip-assets was set, but assets/externals is not staged.")
        print("[clearml-cdpr] Using already staged local assets.", flush=True)
        return

    if args.asset_dataset_id:
        dataset = Dataset.get(dataset_id=args.asset_dataset_id)
    elif args.asset_dataset_name:
        dataset = Dataset.get(dataset_name=args.asset_dataset_name, dataset_project=args.asset_dataset_project)
    elif _asset_targets_are_ready(repo_root):
        print("[clearml-cdpr] Asset Dataset was not provided; using already staged local assets.", flush=True)
        return
    else:
        raise RuntimeError(
            "Pass --asset-dataset-id or --asset-dataset-name. "
            "The ClearML agent starts in a clean container and needs YCB/LIBERO assets."
        )

    asset_root = Path(dataset.get_local_copy()).expanduser().resolve()
    print(f"[clearml-cdpr] ClearML asset dataset root: {asset_root}", flush=True)

    ycb = _find_asset_dir(
        asset_root,
        candidates=[
            asset_root / "assets" / "externals" / "ycb",
            asset_root / "externals" / "ycb",
            asset_root / "ycb",
        ],
        predicate=_looks_like_ycb,
        label="YCB",
    )
    libero = _find_asset_dir(
        asset_root,
        candidates=[
            asset_root / "assets" / "externals" / "libero",
            asset_root / "externals" / "libero",
            asset_root / "libero",
            asset_root / "assets",
        ],
        predicate=_looks_like_libero_assets,
        label="LIBERO",
    )

    _link_dir(ycb, repo_root / "assets" / "externals" / "ycb")
    _link_dir(libero, repo_root / "assets" / "externals" / "libero")


def _upload_artifact(task: Any, name: str, path: Path) -> None:
    if not path.exists():
        print(f"[clearml-cdpr] Artifact path not found, skipping {name}: {path}", flush=True)
        return
    print(f"[clearml-cdpr] Uploading artifact `{name}`: {path}", flush=True)
    try:
        task.upload_artifact(name=name, artifact_object=str(path), wait_on_upload=True)
    except TypeError:
        task.upload_artifact(name=name, artifact_object=str(path))


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("step_"))
    except ValueError:
        return -1


def _discover_checkpoint_dirs(run_dir: Path) -> list[Path]:
    candidates = []
    for path in run_dir.rglob("step_*"):
        if not path.is_dir():
            continue
        if (path / "action_head_cdpr.pt").is_file() or (path / "vla_cdpr_adapter").is_dir():
            candidates.append(path)
    return sorted(candidates, key=lambda p: (_checkpoint_step(p), str(p)))


def upload_outputs(task: Any, run_dir: Path) -> None:
    _upload_artifact(task, "cdpr_rl_run_dir", run_dir)
    _upload_artifact(task, "cdpr_rl_tensorboard", run_dir / "rl" / "tensorboard")
    _upload_artifact(task, "cdpr_rl_preview", run_dir / "preview")
    checkpoint_dirs = _discover_checkpoint_dirs(run_dir)
    if checkpoint_dirs:
        _upload_artifact(task, "cdpr_rl_latest_checkpoint", checkpoint_dirs[-1])
    try:
        task.flush(wait_for_uploads=True)
    except TypeError:
        task.flush()


class TensorboardClearMLMirror:
    def __init__(self, task: Any, tensorboard_dir: Path, *, interval_s: float = 30.0) -> None:
        self.task = task
        self.tensorboard_dir = tensorboard_dir
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_steps: dict[str, int] = {}

    def start(self) -> None:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: F401
        except Exception as exc:
            print(f"[clearml-cdpr] TensorBoard scalar mirror disabled: {exc}", flush=True)
            return
        self._thread = threading.Thread(target=self._loop, name="tensorboard-clearml-mirror", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.sync_once()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sync_once()

    def sync_once(self) -> None:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except Exception:
            return
        if not self.tensorboard_dir.exists():
            return

        event_dirs = sorted({path.parent for path in self.tensorboard_dir.rglob("events.out.tfevents*")})
        logger = self.task.get_logger()
        for event_dir in event_dirs:
            try:
                accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
                accumulator.Reload()
            except Exception as exc:
                print(f"[clearml-cdpr] WARNING: could not read TensorBoard events in {event_dir}: {exc}", flush=True)
                continue

            for tag in accumulator.Tags().get("scalars", []):
                key = f"{event_dir}:{tag}"
                last_step = self._last_steps.get(key, -1)
                max_seen = last_step
                parts = tag.split("/", 1)
                title = f"tensorboard/{parts[0]}"
                series = parts[1] if len(parts) == 2 else parts[0]
                try:
                    events = accumulator.Scalars(tag)
                except Exception:
                    continue
                for event in events:
                    step = int(event.step)
                    if step <= last_step:
                        continue
                    logger.report_scalar(
                        title=title,
                        series=series,
                        value=float(event.value),
                        iteration=step,
                    )
                    max_seen = max(max_seen, step)
                self._last_steps[key] = max_seen


class PeriodicArtifactUploader:
    def __init__(self, task: Any, run_dir: Path, *, interval_s: float) -> None:
        self.task = task
        self.run_dir = run_dir
        self.interval_s = max(60.0, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._uploaded_checkpoints: set[Path] = set()
        self._tensorboard_upload_count = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="clearml-artifact-uploader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.sync_once()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sync_once()

    def sync_once(self) -> None:
        try:
            for checkpoint_dir in _discover_checkpoint_dirs(self.run_dir):
                resolved = checkpoint_dir.resolve()
                if resolved in self._uploaded_checkpoints:
                    continue
                step = checkpoint_dir.name.removeprefix("step_")
                _upload_artifact(self.task, f"cdpr_rl_checkpoint_{step}", checkpoint_dir)
                self._uploaded_checkpoints.add(resolved)

            tensorboard_dir = self.run_dir / "rl" / "tensorboard"
            if tensorboard_dir.exists() and any(tensorboard_dir.rglob("events.out.tfevents*")):
                self._tensorboard_upload_count += 1
                _upload_artifact(self.task, f"cdpr_rl_tensorboard_periodic_{self._tensorboard_upload_count:04d}", tensorboard_dir)
        except Exception as exc:
            print(f"[clearml-cdpr] WARNING: periodic artifact upload failed: {exc}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    Dataset, Task = _load_clearml()
    task = Task.current_task() or Task.init(project_name=args.project_name, task_name=args.task_name)
    args = _connected_args(task, args)

    repo_root = _repo_root()
    _ensure_compat_repo_link(repo_root)
    ensure_openvla_repo(
        Path(args.openvla_path),
        repo_url=str(args.openvla_repo_url),
        branch=str(args.openvla_branch) if args.openvla_branch else None,
        commit=str(args.openvla_commit) if args.openvla_commit else None,
    )
    stage_assets(repo_root, Dataset, args)

    run_name = args.run_name or f"cdpr_rl_clearml_smoke_{task.id[:8]}"
    run_dir = repo_root / "runs" / run_name
    task.set_parameter("General/run_name", run_name)
    task.set_parameter("General/run_dir", str(run_dir))
    task.set_parameter("General/tensorboard_dir", str(run_dir / "rl" / "tensorboard"))

    env = dict(os.environ)
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_API_KEY", "dummy")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    cmd = [
        sys.executable,
        "-m",
        "rl_vla_bootstrapping.cli.train",
        "--config",
        str(repo_root / args.config),
        "--stage",
        str(args.stage),
        "--execute",
        "--run-name",
        run_name,
    ]

    tensorboard_mirror = TensorboardClearMLMirror(task, run_dir / "rl" / "tensorboard")
    tensorboard_mirror.start()
    artifact_uploader = None
    if not args.disable_periodic_artifact_upload:
        artifact_uploader = PeriodicArtifactUploader(
            task,
            run_dir,
            interval_s=float(args.artifact_upload_interval_minutes) * 60.0,
        )
        artifact_uploader.start()
    ret = 1
    try:
        ret = _run(cmd, cwd=repo_root, env=env)
    finally:
        tensorboard_mirror.stop()
        if artifact_uploader is not None:
            artifact_uploader.stop()
        upload_outputs(task, run_dir)
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
