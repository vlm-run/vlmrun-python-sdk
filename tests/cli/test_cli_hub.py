"""Test hub subcommand."""

from vlmrun.cli.cli import app


def test_hub_version(runner, mock_client):
    """Test hub version command."""
    client = mock_client
    result = runner.invoke(app, ["hub", "version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
