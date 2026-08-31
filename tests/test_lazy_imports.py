"""Regression tests for deferred heavy dependency imports."""

from __future__ import annotations

import builtins
import sys

import pytest

from vlmrun.client.types import SchemaResponse


def _purge_vlmrun_modules() -> None:
    for key in list(sys.modules):
        if key == "vlmrun" or key.startswith("vlmrun."):
            del sys.modules[key]


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


def test_types_import_does_not_load_vlmrun_hub():
    """Importing client types must not pull in vlmrun-hub (datamodel-code-generator)."""
    _purge_vlmrun_modules()
    import vlmrun.client.types  # noqa: F401

    assert "vlmrun.hub" not in sys.modules
    assert "datamodel_code_generator" not in sys.modules


def test_vlmrun_client_import_does_not_load_vlmrun_hub():
    """VLMRun client import must not eagerly load vlmrun-hub."""
    _purge_vlmrun_modules()
    from vlmrun.client import VLMRun  # noqa: F401

    assert "vlmrun.hub" not in sys.modules


def test_cli_import_does_not_load_vlmrun_hub():
    """CLI entry import must not eagerly load vlmrun-hub (gw chat spin-up)."""
    _purge_vlmrun_modules()
    import vlmrun.cli.cli  # noqa: F401

    assert "vlmrun.hub" not in sys.modules


def test_schema_response_model_lazy_loads_vlmrun_hub():
    """response_model should import vlmrun-hub only when accessed."""
    _purge_vlmrun_modules()
    schema = SchemaResponse(
        domain="document.invoice",
        schema_version="1",
        schema_hash="abc",
        gql_stmt="",
        json_schema={
            "title": "Invoice",
            "type": "object",
            "properties": {"total": {"type": "number"}},
        },
    )

    assert "vlmrun.hub" not in sys.modules

    model = schema.response_model

    assert "vlmrun.hub" in sys.modules
    assert model.__name__ == "Invoice"


def test_schema_response_model_requires_vlmrun_hub(monkeypatch):
    """Missing vlmrun-hub should surface only when response_model is used."""
    _purge_vlmrun_modules()
    _block_import(monkeypatch, "vlmrun.hub")

    schema = SchemaResponse(
        domain="document.invoice",
        schema_version="1",
        schema_hash="abc",
        gql_stmt="",
        json_schema={"title": "Invoice", "type": "object", "properties": {}},
    )

    with pytest.raises(ImportError, match="vlmrun.hub"):
        _ = schema.response_model
