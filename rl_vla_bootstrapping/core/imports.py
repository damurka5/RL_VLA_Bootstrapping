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
