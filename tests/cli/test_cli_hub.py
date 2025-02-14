"""Test hub subcommand."""

from vlmrun.cli.cli import app


def test_hub_version(runner, mock_client, config_file):
    """Test hub version command."""
    result = runner.invoke(app, ["hub", "version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
    assert "github.com/vlm-run/vlmrun-hub" in result.stdout


def test_hub_list_domains(runner, mock_client, config_file):
    """Test listing hub domains."""
    result = runner.invoke(app, ["hub", "list"])
    assert result.exit_code == 0
    assert "document.invoice" in result.stdout
    assert "document.receipt" in result.stdout
    assert "document.utility_bill" in result.stdout


def test_hub_list_domains_with_filter(runner, mock_client, config_file):
    """Test listing hub domains with filter."""
    result = runner.invoke(app, ["hub", "list", "--domain", "document.invoice"])
    assert result.exit_code == 0
    assert "document.invoice" in result.stdout
    assert "document.receipt" not in result.stdout


def test_hub_schema(runner, mock_client, config_file):
    """Test getting schema for a domain."""
    result = runner.invoke(app, ["hub", "schema", "document.invoice"])
    assert result.exit_code == 0
    assert "Schema Information" in result.stdout
    assert "invoice_number" in result.stdout
    assert "total_amount" in result.stdout
