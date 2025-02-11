"""Test config subcommand."""

import pytest
from vlmrun.cli.cli import app


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """Create a temporary config file."""
    config_path = tmp_path / ".vlmrun.toml"
    monkeypatch.setattr("vlmrun.cli._cli.config.CONFIG_FILE", config_path)
    return config_path


def test_show_empty_config(runner, config_file):
    """Test showing config when no values are set."""
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "No configuration values set" in result.stdout


def test_set_and_show_config(runner, config_file):
    """Test setting and showing config values."""
    result = runner.invoke(app, ["config", "set", "--api-key", "test-key"])
    assert result.exit_code == 0
    assert "Set API key" in result.stdout

    result = runner.invoke(app, ["config", "set", "--base-url", "https://test.vlm.run"])
    assert result.exit_code == 0
    assert "Set base URL" in result.stdout

    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "api_key:" in result.stdout
    assert "base_url: https://test.vlm.run" in result.stdout


def test_unset_config(runner, config_file):
    """Test unsetting config values."""
    runner.invoke(app, ["config", "set", "--api-key", "test-key"])
    runner.invoke(app, ["config", "set", "--base-url", "https://test.vlm.run"])

    result = runner.invoke(app, ["config", "unset", "--api-key"])
    assert result.exit_code == 0
    assert "Removed API key" in result.stdout

    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "api_key" not in result.stdout
    assert "base_url: https://test.vlm.run" in result.stdout


def test_set_no_values(runner, config_file):
    """Test set command with no values."""
    result = runner.invoke(app, ["config", "set"])
    assert result.exit_code == 0
    assert "No values provided to set" in result.stdout


def test_unset_no_values(runner, config_file):
    """Test unset command with no values."""
    result = runner.invoke(app, ["config", "unset"])
    assert result.exit_code == 0
    assert "No values specified to unset" in result.stdout


def test_config_file_permissions(runner, tmp_path, monkeypatch):
    """Test handling of permission errors."""
    config_path = tmp_path / ".vlmrun.toml"
    config_path.touch(mode=0o444)
    monkeypatch.setattr("vlmrun.cli._cli.config.CONFIG_FILE", config_path)

    result = runner.invoke(app, ["config", "set", "--api-key", "test-key"])
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_invalid_toml_handling(runner, config_file):
    """Test handling of invalid TOML format."""
    config_file.write_text("invalid [ toml")

    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "No configuration values set" in result.stdout
