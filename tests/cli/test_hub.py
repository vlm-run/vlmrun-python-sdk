"""Test hub subcommand."""
import pytest
from typer.testing import CliRunner

from vlmrun.cli.cli import app


def test_hub_version(runner, mock_client):
    """Test hub version command."""
    result = runner.invoke(app, ["hub", "version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_hub_list(runner, mock_client):
    """Test hub list command."""
    result = runner.invoke(app, ["hub", "list"])
    assert result.exit_code == 0
    assert "item1" in result.stdout
    assert "Test Item" in result.stdout


def test_hub_submit(runner, mock_client, tmp_path):
    """Test hub submit command."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    result = runner.invoke(app, ["hub", "submit", str(test_file), "--name", "test", "--version", "1.0.0"])
    assert result.exit_code == 0
    assert "item1" in result.stdout
