"""Test models command."""

import json

from vlmrun.cli.cli import app
from tests.conftest import strip_ansi
from tests.test_gateway import patched_cli


def test_models_lists_gateway_catalog(runner, patched_cli):
    """Test top-level models command lists the gateway catalog."""
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "zai-org/glm-ocr" in out
    assert "paddleocr/pp-ocrv6" in out
    assert "INPUTS" in out
    assert "LIMITS" in out
    assert "image_url" in out
    assert "Gateway Models" in out


def test_models_detail_by_alias(runner, patched_cli):
    """Test models detail view for a single model."""
    result = runner.invoke(app, ["models", "pp-ocrv6"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "paddleocr/pp-ocrv6" in out
    assert "detect" in out


def test_models_json(runner, patched_cli):
    """Test models --json returns the raw catalog."""
    result = runner.invoke(app, ["models", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    ids = {m["id"] for m in data}
    assert ids == {"zai-org/glm-ocr", "paddleocr/pp-ocrv6"}


def test_models_list_subcommand_removed(runner, patched_cli):
    """The old `models list` subcommand is no longer registered."""
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 1
    assert "not found" in strip_ansi(result.stdout).lower()
