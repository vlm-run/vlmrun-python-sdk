"""Test models subcommand."""

from vlmrun.cli.cli import app


def test_list_models(runner, mock_client):
    """Test list models command."""
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "model1" in result.stdout
    assert "test-domain" in result.stdout
    assert "Available Models" in result.stdout


def test_list_models_with_filter(runner, mock_client):
    """Test list models command with domain filter."""
    result = runner.invoke(app, ["models", "list", "--domain", "test-domain"])
    assert result.exit_code == 0
    assert "model1" in result.stdout
    assert "test-domain" in result.stdout

    result = runner.invoke(app, ["models", "list", "--domain", "nonexistent"])
    assert result.exit_code == 0
    assert "model1" not in result.stdout


def test_list_models_formatting(runner, mock_client):
    """Test that list models output is properly formatted."""
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "Category" in result.stdout
    assert "Model" in result.stdout
    assert "Domain" in result.stdout
