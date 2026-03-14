"""Test models subcommand."""

from vlmrun.cli.cli import app
from tests.conftest import strip_ansi


def test_list_models(runner, mock_client, config_file):
    """Test list models command."""
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "model1" in out
    assert "test-domain" in out
    assert "Models" in out


def test_list_models_with_filter(runner, mock_client, config_file):
    """Test list models command with domain filter."""
    result = runner.invoke(app, ["models", "list", "--domain", "test-domain"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "model1" in out
    assert "test-domain" in out

    result = runner.invoke(app, ["models", "list", "--domain", "nonexistent"])
    assert result.exit_code == 0
    assert "model1" not in strip_ansi(result.stdout)


def test_list_models_formatting(runner, mock_client, config_file):
    """Test that list models output is properly formatted."""
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "CATEGORY" in out
    assert "MODEL" in out
    assert "DOMAIN" in out
