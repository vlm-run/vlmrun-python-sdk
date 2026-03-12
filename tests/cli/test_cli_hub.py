"""Test hub subcommand."""

from vlmrun.cli.cli import app
from tests.conftest import strip_ansi


def test_hub_version(runner, mock_client, config_file):
    """Test hub version command."""
    result = runner.invoke(app, ["hub", "version"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "0.1.0" in out
    assert "github.com/vlm-run/vlmrun-hub" in out


def test_hub_list_domains(runner, mock_client, config_file):
    """Test listing hub domains."""
    result = runner.invoke(app, ["hub", "list"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "document.invoice" in out
    assert "document.receipt" in out
    assert "document.utility_bill" in out


def test_hub_list_domains_with_filter(runner, mock_client, config_file):
    """Test listing hub domains with filter."""
    result = runner.invoke(app, ["hub", "list", "--domain", "document.invoice"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "document.invoice" in out
    assert "document.receipt" not in out


def test_hub_schema(runner, mock_client, config_file):
    """Test getting schema for a domain."""
    result = runner.invoke(app, ["hub", "schema", "document.invoice"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "Schema Information" in out
    assert "invoice_number" in out
    assert "total_amount" in out
