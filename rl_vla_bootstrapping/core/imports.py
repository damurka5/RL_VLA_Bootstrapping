from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


class ImportResolutionError(RuntimeError):
    """Raised when a configured entrypoint cannot be imported."""


@contextmanager
def prepend_sys_path(paths: Iterable[Path]):
    resolved = [str(Path(p).resolve()) for p in paths if p]
    original = list(sys.path)
    try:
        for path in reversed(resolved):
            if path not in sys.path:
                sys.path.insert(0, path)
        yield
    finally:
        sys.path[:] = original


def _module_name_from_root(file_path: Path, root: Path) -> tuple[str, bool] | None:
    try:
        relative = file_path.relative_to(root)
    except ValueError:
        return None

    if relative.suffix != ".py":
        return None

    if relative.name == "__init__.py":
        module_parts = list(relative.parts[:-1])
    else:
        module_parts = [*relative.parts[:-1], relative.stem]

    if not module_parts or not all(part.isidentifier() for part in module_parts):
        return None

    return ".".join(module_parts), len(module_parts) > 1


def _infer_package_root(file_path: Path) -> Path | None:
    package_root = file_path.parent
    package_found = False

    while (package_root / "__init__.py").exists():
        package_found = True
        package_root = package_root.parent

    if not package_found:
        return None
    return package_root


def _resolve_file_module(file_path: Path, python_paths: Iterable[Path]) -> tuple[str, list[Path]] | None:
    candidates: list[tuple[bool, int, int, str, Path]] = []
    seen: set[tuple[str, Path]] = set()

    roots = [(Path(path).resolve(), 0) for path in python_paths if path]
    inferred_root = _infer_package_root(file_path)
    if inferred_root is not None:
        roots.append((inferred_root.resolve(), 1))

    for root, origin in roots:
        resolved = _module_name_from_root(file_path, root)
        if resolved is None:
            continue
        module_name, package_aware = resolved
        key = (module_name, root)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((package_aware, origin, module_name.count("."), module_name, root))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-int(item[0]), item[1], item[2], item[3]))
    _, _, _, module_name, root = candidates[0]
    return module_name, [root]


def import_object(
    *,
    file_path: Path | None,
    module_name: str | None,
    attribute: str,
    python_paths: Iterable[Path] = (),
) -> Any:
    with prepend_sys_path(python_paths):
        if file_path is not None:
            file_path = Path(file_path).resolve()
            if not file_path.exists():
                raise ImportResolutionError(f"Entrypoint file does not exist: {file_path}")
            resolved_file_module = _resolve_file_module(file_path, python_paths)
            if resolved_file_module is not None:
                resolved_module_name, extra_paths = resolved_file_module
                with prepend_sys_path(extra_paths):
                    module = importlib.import_module(resolved_module_name)
            else:
                spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
                if spec is None or spec.loader is None:
                    raise ImportResolutionError(f"Could not create import spec for: {file_path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        elif module_name:
            module = importlib.import_module(module_name)
        else:
            raise ImportResolutionError("Entrypoint needs either `file` or `module`.")

    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        src = str(file_path) if file_path is not None else str(module_name)
        raise ImportResolutionError(f"Could not resolve `{attribute}` from `{src}`.") from exc


def call_with_supported_kwargs(func: Any, **candidate_kwargs: Any) -> Any:
    signature = inspect.signature(func)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    filtered: dict[str, Any] = {}
    for name, value in candidate_kwargs.items():
        if accepts_kwargs or name in signature.parameters:
            filtered[name] = value
    return func(**filtered)
