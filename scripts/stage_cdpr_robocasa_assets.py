#!/usr/bin/env python3
"""Stage the small RoboCasa visual subset used by the CDPR MJ-Lab backend.

The official RoboCasa Objaverse object archive is 2.16 GB. This downloader
uses HTTP byte ranges to read its ZIP directory and fetch only the model XML
and visual files selected by ``cdpr_object_catalog``. Collision files are
deliberately omitted because MJWarp uses the fixed native-primitive colliders
defined in the CDPR scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
    ACTIVE_CDPR_CATALOGS,
    OBJECT_VARIANTS,
)


ARCHIVE_URL = (
    "https://huggingface.co/datasets/jianzhang96/robocasa-assets/"
    "resolve/main/objaverse.zip"
)
ARCHIVE_SIZE = 2_164_109_015
ARCHIVE_SHA256 = "4ecba30e93e3c600d5dcfac1395ba9ae720aa67abd6b3c734e9c18768d890b84"
ARCHIVE_UPSTREAM = (
    "https://utexas.box.com/shared/static/"
    "03eionyo8fk3a9dsksq9jb8du5lqfw8h.zip"
)
EOCD_SEARCH_BYTES = 65_557


@dataclass(frozen=True)
class ZipMember:
    name: str
    compression: int
    crc32: int
    compressed_size: int
    size: int
    local_header_offset: int


def _fetch_range(url: str, start: int, end: int, *, retries: int = 3) -> bytes:
    expected = end - start + 1
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "RL-VLA-Bootstrapping-RoboCasa-Stager/1",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
                content_range = response.headers.get("Content-Range", "")
            if len(content) != expected or not content_range.startswith("bytes "):
                raise RuntimeError(
                    f"Server did not honor byte range {start}-{end}: "
                    f"received {len(content)} bytes, Content-Range={content_range!r}."
                )
            return content
        except Exception as exc:  # pragma: no cover - network-specific
            error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch archive bytes {start}-{end}: {error}") from error


def _read_zip_index(url: str) -> dict[str, ZipMember]:
    tail_start = ARCHIVE_SIZE - EOCD_SEARCH_BYTES
    tail = _fetch_range(url, tail_start, ARCHIVE_SIZE - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("ZIP end-of-central-directory record was not found.")
    (
        signature,
        disk,
        central_disk,
        entries_on_disk,
        entries_total,
        central_size,
        central_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", tail, eocd)
    if (
        signature != b"PK\x05\x06"
        or disk != 0
        or central_disk != 0
        or entries_on_disk != entries_total
        or comment_size != 0
    ):
        raise RuntimeError("Unsupported multi-disk or commented ZIP archive.")
    central = _fetch_range(
        url, central_offset, central_offset + central_size - 1
    )
    members: dict[str, ZipMember] = {}
    cursor = 0
    for _ in range(entries_total):
        if central[cursor : cursor + 4] != b"PK\x01\x02":
            raise RuntimeError(f"Invalid central-directory entry at byte {cursor}.")
        fields = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        name_size, extra_size, member_comment_size = fields[10:13]
        name = central[cursor + 46 : cursor + 46 + name_size].decode("utf-8")
        members[name] = ZipMember(
            name=name,
            compression=fields[4],
            crc32=fields[7],
            compressed_size=fields[8],
            size=fields[9],
            local_header_offset=fields[16],
        )
        cursor += 46 + name_size + extra_size + member_comment_size
    return members


def _extract_member(url: str, member: ZipMember) -> bytes:
    header = _fetch_range(
        url, member.local_header_offset, member.local_header_offset + 29
    )
    (
        signature,
        _version,
        _flags,
        compression,
        _time,
        _date,
        _crc,
        _compressed_size,
        _size,
        name_size,
        extra_size,
    ) = struct.unpack("<4s5H3L2H", header)
    if signature != b"PK\x03\x04" or compression != member.compression:
        raise RuntimeError(f"Invalid local header for {member.name!r}.")
    data_start = member.local_header_offset + 30 + name_size + extra_size
    compressed = _fetch_range(
        url, data_start, data_start + member.compressed_size - 1
    )
    if member.compression == 0:
        content = compressed
    elif member.compression == 8:
        content = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(
            f"Unsupported ZIP compression {member.compression} for {member.name!r}."
        )
    if len(content) != member.size:
        raise RuntimeError(
            f"Size mismatch for {member.name!r}: {len(content)} != {member.size}."
        )
    crc = zlib.crc32(content) & 0xFFFFFFFF
    if crc != member.crc32:
        raise RuntimeError(
            f"CRC mismatch for {member.name!r}: {crc:08x} != {member.crc32:08x}."
        )
    return content


def _selected_members() -> Iterable[tuple[str, str]]:
    for catalog in ACTIVE_CDPR_CATALOGS:
        variant = OBJECT_VARIANTS[catalog]
        relative = Path(variant.asset_directory)
        if relative.parts[:2] != ("objects", "objaverse"):
            raise RuntimeError(
                f"Unexpected RoboCasa asset directory: {variant.asset_directory!r}."
            )
        archive_prefix = Path(*relative.parts[1:])
        for filename in variant.asset_files:
            yield catalog, (archive_prefix / filename).as_posix()


def _local_path(target: Path, archive_member: str) -> Path:
    return target / "objects" / archive_member


def _verify_staged_manifest(target: Path) -> int:
    manifest_path = target / "cdpr_subset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Staged RoboCasa manifest does not exist: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("archive_sha256") != ARCHIVE_SHA256:
        raise RuntimeError("Staged manifest references an unexpected archive.")
    if tuple(manifest.get("catalogs", ())) != ACTIVE_CDPR_CATALOGS:
        raise RuntimeError("Staged manifest catalog order does not match the backend.")
    expected_paths = {
        _local_path(target, name).relative_to(target).as_posix()
        for _, name in _selected_members()
    }
    records = manifest.get("files", ())
    if {str(record.get("path", "")) for record in records} != expected_paths:
        raise RuntimeError("Staged manifest file set does not match the backend.")
    for record in records:
        path = target / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing staged RoboCasa member: {path}")
        content = path.read_bytes()
        if len(content) != int(record["bytes"]):
            raise RuntimeError(f"Size mismatch for staged RoboCasa member: {path}")
        if hashlib.sha256(content).hexdigest() != str(record["sha256"]):
            raise RuntimeError(f"SHA-256 mismatch for staged RoboCasa member: {path}")
    print(
        f"Verified {len(records)} RoboCasa files for "
        f"{len(ACTIVE_CDPR_CATALOGS)} catalogs under {target}."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the curated RoboCasa visual subset for CDPR MJ-Lab."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT / "assets" / "externals" / "robocasa",
    )
    parser.add_argument("--url", default=ARCHIVE_URL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    if args.verify_only:
        return _verify_staged_manifest(target)
    requested = tuple(_selected_members())
    index = _read_zip_index(str(args.url))
    missing_from_archive = [name for _, name in requested if name not in index]
    if missing_from_archive:
        raise RuntimeError(
            "Selected RoboCasa members are missing from the archive: "
            + ", ".join(missing_from_archive)
        )

    records: list[dict[str, object]] = []
    for position, (catalog, name) in enumerate(requested, start=1):
        member = index[name]
        destination = _local_path(target, name)
        valid_existing = False
        if destination.is_file() and not args.force:
            content = destination.read_bytes()
            valid_existing = (
                len(content) == member.size
                and (zlib.crc32(content) & 0xFFFFFFFF) == member.crc32
            )
        if not valid_existing:
            print(f"[{position:02d}/{len(requested):02d}] {name}", flush=True)
            content = _extract_member(str(args.url), member)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        else:
            content = destination.read_bytes()
        records.append(
            {
                "catalog": catalog,
                "archive_member": name,
                "path": destination.relative_to(target).as_posix(),
                "bytes": len(content),
                "crc32": f"{member.crc32:08x}",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    manifest = {
        "format": 1,
        "source": "RoboCasa Objaverse objects",
        "license": "CC BY 4.0",
        "upstream_archive_url": ARCHIVE_UPSTREAM,
        "download_url": str(args.url),
        "archive_bytes": ARCHIVE_SIZE,
        "archive_sha256": ARCHIVE_SHA256,
        "collision_policy": "CDPR fixed native primitives; RoboCasa collisions omitted",
        "catalogs": list(ACTIVE_CDPR_CATALOGS),
        "files": records,
    }
    manifest_path = target / "cdpr_subset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Verified {len(records)} RoboCasa files for "
        f"{len(ACTIVE_CDPR_CATALOGS)} catalogs under {target}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
