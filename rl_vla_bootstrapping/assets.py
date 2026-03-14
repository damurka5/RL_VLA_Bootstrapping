from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from rl_vla_bootstrapping.core.specs import AssetBundleSpec, ProjectConfig


@dataclass(frozen=True)
class AssetStatus:
    name: str
    source_path: str | None
    target_path: str
    exists: bool
    linked: bool
    required: bool
    kind: str
    description: str


def _default_target(bundle: AssetBundleSpec) -> str:
    if bundle.target_path:
        return bundle.target_path
    parent = "benchmarks/externals" if bundle.kind == "benchmark_repo" else "assets/externals"
    return f"{parent}/{bundle.name}"


def _resolve_without_following_symlinks(config: ProjectConfig, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return Path(os.path.abspath(str(config.root_dir / path)))


def resolve_asset_paths(config: ProjectConfig, bundle: AssetBundleSpec) -> tuple[Path | None, Path]:
    source = config.resolve_path(bundle.source_path) if bundle.source_path else None
    target = _resolve_without_following_symlinks(config, _default_target(bundle))
    return source, target


def collect_asset_status(config: ProjectConfig) -> list[AssetStatus]:
    statuses: list[AssetStatus] = []
    for bundle in config.assets.bundles:
        source, target = resolve_asset_paths(config, bundle)
        exists = target.exists()
        linked = target.is_symlink() if exists else False
        statuses.append(
            AssetStatus(
                name=bundle.name,
                source_path=source.as_posix() if source is not None else None,
                target_path=target.as_posix(),
                exists=exists,
                linked=linked,
                required=bundle.required,
                kind=bundle.kind,
                description=bundle.description,
            )
        )
    return statuses


def stage_asset_bundles(
    config: ProjectConfig,
    *,
    mode: str = "symlink",
    force: bool = False,
) -> list[AssetStatus]:
    statuses: list[AssetStatus] = []
    for bundle in config.assets.bundles:
        source, target = resolve_asset_paths(config, bundle)
        if source is None or not source.exists():
            statuses.append(
                AssetStatus(
                    name=bundle.name,
                    source_path=source.as_posix() if source is not None else None,
                    target_path=target.as_posix(),
                    exists=False,
                    linked=False,
                    required=bundle.required,
                    kind=bundle.kind,
                    description=bundle.description,
                )
            )
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if not force:
                statuses.append(
                    AssetStatus(
                        name=bundle.name,
                        source_path=source.as_posix(),
                        target_path=target.as_posix(),
                        exists=True,
                        linked=target.is_symlink(),
                        required=bundle.required,
                        kind=bundle.kind,
                        description=bundle.description,
                    )
                )
                continue
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)

        if mode == "copy":
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            linked = False
        else:
            os.symlink(source, target, target_is_directory=source.is_dir())
            linked = True

        statuses.append(
            AssetStatus(
                name=bundle.name,
                source_path=source.as_posix(),
                target_path=target.as_posix(),
                exists=True,
                linked=linked,
                required=bundle.required,
                kind=bundle.kind,
                description=bundle.description,
            )
        )
    return statuses


def export_asset_manifest(config: ProjectConfig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [status.__dict__ for status in collect_asset_status(config)]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
