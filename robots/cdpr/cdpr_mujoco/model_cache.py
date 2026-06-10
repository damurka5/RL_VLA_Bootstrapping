"""Process-local MuJoCo compiled-model cache for CDPR audits and rollouts."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco as mj


FILE_ATTRS = {"file"}


@dataclass(frozen=True)
class CompiledModelCacheKey:
    """Stable cache key for a compiled MuJoCo model."""

    digest: str
    xml_path: str
    semantic_key: tuple[tuple[str, str], ...]
    file_signature_hash: str
    timestep: float
    offscreen_width: int
    offscreen_height: int
    offscreen_samples: str

    def short(self) -> str:
        return self.digest[:12]


@dataclass
class CompiledModelCacheEvent:
    enabled: bool
    hit: bool
    key: str
    key_short: str
    compile_time_s: float
    cache_size: int
    rss_mb: float
    cache_max_size: int = 0
    evictions: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "hit": bool(self.hit),
            "miss": bool(self.enabled and not self.hit),
            "key": self.key,
            "key_short": self.key_short,
            "compile_time_s": float(self.compile_time_s),
            "cache_size": int(self.cache_size),
            "cache_max_size": int(self.cache_max_size),
            "evictions": int(self.evictions),
            "rss_mb": float(self.rss_mb),
            "reason": str(self.reason),
        }


_MODEL_CACHE: OrderedDict[CompiledModelCacheKey, mj.MjModel] = OrderedDict()
_CACHE_HITS = 0
_CACHE_MISSES = 0
_CACHE_EVICTIONS = 0
_LAST_EVENT: CompiledModelCacheEvent | None = None


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    if usage > 1024 * 1024 * 8:
        return float(usage / (1024 * 1024))
    return float(usage / 1024)


def _cache_max_size_from_env() -> int:
    raw = os.environ.get("RLVLA_CDPR_COMPILED_MODEL_CACHE_MAX_SIZE")
    if raw is None or str(raw).strip() == "":
        return 32
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 32
    return max(0, int(value))


def _evict_lru_models_if_needed(max_size: int) -> None:
    global _CACHE_EVICTIONS
    limit = max(0, int(max_size))
    if limit <= 0:
        return
    while len(_MODEL_CACHE) > limit:
        _MODEL_CACHE.popitem(last=False)
        _CACHE_EVICTIONS += 1


def _resolve_path(raw: str, base: Path) -> Path:
    candidate = Path(os.path.expanduser(str(raw)))
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def _iter_file_refs(xml_path: Path) -> Iterable[Path]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return
    for elem in root.iter():
        for attr, raw in elem.attrib.items():
            if attr in FILE_ATTRS and raw:
                yield _resolve_path(raw, xml_path.parent)


def _collect_file_signature(xml_path: Path) -> tuple[tuple[str, int, int], ...]:
    """Return a lightweight signature for the wrapper/include graph."""

    resolved = Path(xml_path).expanduser().resolve()
    seen: set[Path] = set()
    stack = [resolved]
    out: list[tuple[str, int, int]] = []
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and path.is_file():
            stat = path.stat()
            out.append((path.as_posix(), int(stat.st_size), int(stat.st_mtime_ns)))
            if path.suffix.lower() in {".xml", ".mjcf"}:
                for ref in _iter_file_refs(path):
                    if ref not in seen:
                        stack.append(ref)
        else:
            out.append((path.as_posix(), -1, -1))
    return tuple(sorted(out))


def _normalize_semantic_key(semantic_key: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not semantic_key:
        return ()
    normalized: list[tuple[str, str]] = []
    for key, value in sorted(dict(semantic_key).items(), key=lambda item: str(item[0])):
        if isinstance(value, (list, tuple, set)):
            value_str = ",".join(str(x) for x in sorted(value))
        else:
            value_str = str(value)
        normalized.append((str(key), value_str))
    return tuple(normalized)


def build_cache_key(
    xml_path: str | Path,
    *,
    timestep: float,
    offscreen_width: int,
    offscreen_height: int,
    offscreen_samples: str,
    semantic_key: dict[str, Any] | None = None,
) -> CompiledModelCacheKey:
    resolved = Path(xml_path).expanduser().resolve()
    file_signature = _collect_file_signature(resolved)
    file_signature_json = json.dumps(file_signature, sort_keys=True, separators=(",", ":"))
    file_signature_hash = hashlib.sha256(file_signature_json.encode("utf-8")).hexdigest()
    semantic = _normalize_semantic_key(semantic_key)
    payload = {
        "xml_path": resolved.as_posix(),
        "semantic_key": semantic,
        "file_signature_hash": file_signature_hash,
        "timestep": float(timestep),
        "offscreen_width": int(offscreen_width),
        "offscreen_height": int(offscreen_height),
        "offscreen_samples": str(offscreen_samples),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return CompiledModelCacheKey(
        digest=digest,
        xml_path=resolved.as_posix(),
        semantic_key=semantic,
        file_signature_hash=file_signature_hash,
        timestep=float(timestep),
        offscreen_width=int(offscreen_width),
        offscreen_height=int(offscreen_height),
        offscreen_samples=str(offscreen_samples),
    )


def get_compiled_model(
    xml_path: str | Path,
    *,
    enabled: bool,
    timestep: float,
    offscreen_width: int,
    offscreen_height: int,
    offscreen_samples: str,
    semantic_key: dict[str, Any] | None = None,
) -> tuple[mj.MjModel, CompiledModelCacheEvent]:
    """Return a compiled model and cache event for instrumentation."""

    global _CACHE_HITS, _CACHE_MISSES, _LAST_EVENT
    cache_max_size = _cache_max_size_from_env()
    key = build_cache_key(
        xml_path,
        timestep=timestep,
        offscreen_width=offscreen_width,
        offscreen_height=offscreen_height,
        offscreen_samples=offscreen_samples,
        semantic_key=semantic_key,
    )

    if enabled and key in _MODEL_CACHE:
        _CACHE_HITS += 1
        model = _MODEL_CACHE.pop(key)
        _MODEL_CACHE[key] = model
        event = CompiledModelCacheEvent(
            enabled=True,
            hit=True,
            key=key.digest,
            key_short=key.short(),
            compile_time_s=0.0,
            cache_size=len(_MODEL_CACHE),
            cache_max_size=cache_max_size,
            evictions=_CACHE_EVICTIONS,
            rss_mb=_rss_mb(),
        )
        _LAST_EVENT = event
        return model, event

    start = time.perf_counter()
    model = mj.MjModel.from_xml_path(str(Path(xml_path).expanduser().resolve()))
    model.opt.timestep = float(timestep)
    compile_time = time.perf_counter() - start

    if enabled:
        _MODEL_CACHE[key] = model
        _evict_lru_models_if_needed(cache_max_size)
        _CACHE_MISSES += 1
    event = CompiledModelCacheEvent(
        enabled=bool(enabled),
        hit=False,
        key=key.digest,
        key_short=key.short(),
        compile_time_s=float(compile_time),
        cache_size=len(_MODEL_CACHE),
        cache_max_size=cache_max_size,
        evictions=_CACHE_EVICTIONS,
        rss_mb=_rss_mb(),
        reason="" if enabled else "disabled",
    )
    _LAST_EVENT = event
    return model, event


def cache_stats() -> dict[str, Any]:
    return {
        "size": len(_MODEL_CACHE),
        "hits": int(_CACHE_HITS),
        "misses": int(_CACHE_MISSES),
        "evictions": int(_CACHE_EVICTIONS),
        "max_size": int(_cache_max_size_from_env()),
        "hit_rate": float(_CACHE_HITS / max(1, _CACHE_HITS + _CACHE_MISSES)),
        "last_event": None if _LAST_EVENT is None else _LAST_EVENT.as_dict(),
    }


def clear_cache() -> None:
    global _CACHE_HITS, _CACHE_MISSES, _CACHE_EVICTIONS, _LAST_EVENT
    _MODEL_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0
    _CACHE_EVICTIONS = 0
    _LAST_EVENT = None
