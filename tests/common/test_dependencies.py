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
