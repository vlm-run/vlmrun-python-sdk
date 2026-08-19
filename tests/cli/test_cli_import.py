"""Tests for CLI import behavior."""

from __future__ import annotations

import builtins
import sys

from typer.testing import CliRunner

from tests.common.test_dependencies import _block_import


def test_cli_imports_without_openai(monkeypatch):
    """The CLI entrypoint should import even when openai is not installed."""
    _block_import(monkeypatch, "openai")

    for key in list(sys.modules):
        if key == "vlmrun.cli.cli" or key.startswith("vlmrun.cli."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    from vlmrun.cli.cli import app

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "vlmrun version:" in result.stdout
