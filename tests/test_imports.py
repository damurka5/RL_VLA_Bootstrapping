from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rl_vla_bootstrapping.core.imports import import_object


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class ImportObjectTests(unittest.TestCase):
    def test_import_object_supports_relative_imports_from_file_entrypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "pkg_simple" / "__init__.py", "")
            _write(root / "pkg_simple" / "helper.py", "VALUE = 7\n")
            module_path = _write(
                root / "pkg_simple" / "entrypoint.py",
                "from .helper import VALUE\n"
                "def build_value():\n"
                "    return VALUE\n",
            )

            build_value = import_object(
                file_path=module_path,
                module_name=None,
                attribute="build_value",
                python_paths=[root],
            )

            self.assertEqual(build_value(), 7)

    def test_import_object_prefers_package_context_over_narrow_python_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "pkg_preferred" / "__init__.py", "")
            _write(root / "pkg_preferred" / "helper.py", "VALUE = 11\n")
            module_path = _write(
                root / "pkg_preferred" / "entrypoint.py",
                "from .helper import VALUE\n"
                "def build_value():\n"
                "    return VALUE\n",
            )

            build_value = import_object(
                file_path=module_path,
                module_name=None,
                attribute="build_value",
                python_paths=[root / "pkg_preferred", root],
            )

            self.assertEqual(build_value(), 11)


if __name__ == "__main__":
    unittest.main()
