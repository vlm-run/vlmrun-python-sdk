"""Test models subcommand."""

from vlmrun.cli.cli import app


def test_list_models(runner, mock_client):
    """Test list models command."""
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "model1" in result.stdout
    assert "Test Model" in result.stdout
