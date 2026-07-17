#!/usr/bin/env python3
"""Invalidate generated CDPR wrappers when the source robot XML changes."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


ROBOT_XML_RELATIVE = Path("robots/cdpr/cdpr_mujoco/cdpr.xml")
WRAPPER_CACHE_RELATIVE = Path("robots/cdpr/cdpr_dataset/wrappers")
FINGERPRINT_MARKER = ".cdpr_robot_xml.sha256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_wrapper_cache(repo_root: Path, *, mode: str = "auto") -> bool:
    """Refresh generated wrappers and return whether stale entries were removed."""

    if mode not in {"auto", "force", "off"}:
        raise ValueError(f"Unsupported refresh mode: {mode!r}")
    if mode == "off":
        print("[wrapper-cache] refresh disabled")
        return False

    repo_root = Path(repo_root).expanduser().resolve()
    robot_xml = repo_root / ROBOT_XML_RELATIVE
    wrapper_cache = repo_root / WRAPPER_CACHE_RELATIVE
    if not robot_xml.is_file():
        raise FileNotFoundError(f"CDPR source XML not found: {robot_xml}")

    current_fingerprint = _sha256(robot_xml)
    wrapper_cache.mkdir(parents=True, exist_ok=True)
    marker = wrapper_cache / FINGERPRINT_MARKER
    previous_fingerprint = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""

    if mode == "auto" and previous_fingerprint == current_fingerprint:
        print(f"[wrapper-cache] current for robot XML {current_fingerprint[:12]}")
        return False

    removed = 0
    for entry in wrapper_cache.iterdir():
        if entry.name == FINGERPRINT_MARKER:
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)
        removed += 1

    marker.write_text(f"{current_fingerprint}\n", encoding="utf-8")
    reason = "forced refresh" if mode == "force" else "robot XML changed"
    print(
        f"[wrapper-cache] removed {removed} stale entr{'y' if removed == 1 else 'ies'} "
        f"({reason}); new fingerprint={current_fingerprint[:12]}"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("auto", "force", "off"), default="auto")
    args = parser.parse_args()
    refresh_wrapper_cache(args.repo_root, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
