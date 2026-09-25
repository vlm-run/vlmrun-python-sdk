"""Tests for verifying optional dependency handling."""

from __future__ import annotations

import builtins
import sys

import pytest

from vlmrun.client.exceptions import DependencyError
from vlmrun.common import dependencies


def _block_import(monkeypatch, module_name: str) -> None:
    for key in list(sys.modules):
        if key == module_name or key.startswith(f"{module_name}."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    real_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        blocked = (
            name == module_name
            or name.startswith(f"{module_name}.")
            or (fromlist and module_name in fromlist)
        )
        if blocked:
            raise ImportError(f"No module named '{module_name}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)


def test_require_openai_suggestion(monkeypatch):
    """OpenAI is a core dependency; errors should point at base install."""
    _block_import(monkeypatch, "openai")
    with pytest.raises(DependencyError) as exc_info:
        dependencies.require_openai()
    assert "pip install vlmrun" in exc_info.value.suggestion
    assert "[openai]" not in exc_info.value.suggestion


@pytest.mark.parametrize(
    ("require_fn", "module_name", "extra"),
    [
        (dependencies.require_pandas, "pandas", "all"),
        (dependencies.require_numpy, "numpy", "video"),
        (dependencies.require_cv2, "cv2", "video"),
        (dependencies.require_ipython_html, "IPython", "all"),
        (dependencies.require_pypdfium2, "pypdfium2", "doc"),
    ],
)
def test_optional_dependency_errors(require_fn, module_name, extra, monkeypatch):
    """Missing optional deps should raise DependencyError with install hints."""
    _block_import(monkeypatch, module_name)
    with pytest.raises(DependencyError) as exc_info:
        require_fn()
    assert f"vlmrun[{extra}]" in exc_info.value.suggestion


def test_markdown_table_to_dataframe_requires_pandas(monkeypatch):
    """MarkdownTable.to_dataframe should lazy-load pandas."""

    def _raise_pandas():
        raise DependencyError(
            message="pandas is not installed",
            suggestion="Install it with `pip install vlmrun[all]`",
        )

    monkeypatch.setattr("vlmrun.client.types.require_pandas", _raise_pandas)
    from vlmrun.client.types import MarkdownTable, TableHeader

    table = MarkdownTable(
        headers=[TableHeader(id="col1", column=0, name="Column 1")],
        data=[{"col1": "value"}],
    )
    with pytest.raises(DependencyError):
        table.to_dataframe()


def test_require_websockets_returns_the_module():
    """The ws transport's guard, which only a websocket session should trip."""
    from vlmrun.common.dependencies import require_websockets

    assert require_websockets().__name__ == "websockets"


def test_require_websockets_names_the_ws_extra(monkeypatch):
    """A missing install points at the extra that provides it."""
    import builtins

    from vlmrun.client.exceptions import DependencyError
    from vlmrun.common.dependencies import require_websockets

    real_import = builtins.__import__

    def fail_websockets(name, *args, **kwargs):
        if name == "websockets":
            raise ImportError("no websockets")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_websockets)
    with pytest.raises(DependencyError) as caught:
        require_websockets()
    assert "vlmrun[ws]" in caught.value.suggestion


def test_websockets_is_not_imported_at_module_scope():
    """An install without the ws extra must not break importing the SDK.

    Asserted structurally rather than by uninstalling: no module under vlmrun
    may import websockets except through the guard.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "vlmrun"
    offenders = []
    for path in root.rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.match(r"\s*(import websockets|from websockets)", line):
                # The guard itself is the one place allowed to import it.
                if path.name != "dependencies.py":
                    offenders.append(f"{path.relative_to(root)}:{number}")
    assert not offenders, f"websockets imported outside the guard: {offenders}"


@pytest.mark.parametrize(
    "first_import",
    ["vlmrun.common.dependencies", "vlmrun.common.video", "vlmrun.common.image"],
)
def test_common_modules_import_standalone(first_import):
    """Each module must import without a sibling having been imported first.

    ``_dependency_error`` needs ``vlmrun.client.exceptions``, and importing that
    runs ``vlmrun.client.__init__``, which comes back here for its own guards. A
    module-scope import therefore breaks whichever of these is the first
    ``vlmrun`` import in a process — which a test inside this suite cannot see,
    because pytest has already imported the package. Hence the subprocess.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", f"import {first_import}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
