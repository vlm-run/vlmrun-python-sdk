"""Test predictions subcommand."""

from vlmrun.cli.cli import app
from tests.conftest import strip_ansi


def test_list_predictions(runner, mock_client, config_file, monkeypatch):
    """Test list predictions command."""
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["predictions", "list"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "prediction1" in out
    assert "running" in out
    assert "2024-01-01" in out


def test_list_predictions_with_status_filter(runner, mock_client, config_file):
    """Test list predictions with status filter."""
    result = runner.invoke(app, ["predictions", "list", "--status", "running"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "prediction1" in out
    assert "running" in out

    result = runner.invoke(app, ["predictions", "list", "--status", "completed"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "No predictions found" in out or out.strip() == ""


def test_get_prediction(runner, mock_client, config_file):
    """Test get prediction command."""
    result = runner.invoke(app, ["predictions", "get", "prediction1"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "prediction1" in out
    assert "Status" in out
    assert "running" in out
    assert "Created" in out
    assert "2024-01-01" in out


def test_get_prediction_with_wait(runner, mock_client, config_file):
    """Test get prediction command with wait flag."""
    result = runner.invoke(
        app, ["predictions", "get", "prediction1", "--wait", "--timeout", "5"]
    )
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "prediction1" in out
    assert "completed" in out
    assert "Credits Used: 100" in out
    assert "Response" in out
    assert "test" in out


def test_get_prediction_usage_display(runner, mock_client, config_file):
    """Test that prediction usage information is displayed correctly."""
    result = runner.invoke(app, ["predictions", "get", "prediction1", "--wait"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "Credits Used" in out
    assert "100" in out


def test_list_predictions_table_format(runner, mock_client, config_file, monkeypatch):
    """Test that list output is formatted correctly."""
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["predictions", "list"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "ID" in out
    assert "CREATED" in out
    assert "STATUS" in out


def test_list_predictions_since_filter(runner, mock_client, config_file):
    """predictions list --since filters out older predictions."""
    result = runner.invoke(app, ["predictions", "list", "--since", "2025-01-01"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "No predictions found" in out


def test_list_predictions_until_filter(runner, mock_client, config_file):
    """predictions list --until keeps older predictions."""
    result = runner.invoke(app, ["predictions", "list", "--until", "2025-01-01"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "prediction1" in out


def test_list_predictions_invalid_since(runner, mock_client, config_file):
    """predictions list --since with bad date format exits with error."""
    result = runner.invoke(app, ["predictions", "list", "--since", "01/01/2024"])
    assert result.exit_code == 1
    assert "Invalid date format" in result.stdout


def test_list_predictions_invalid_until(runner, mock_client, config_file):
    """predictions list --until with bad date format exits with error."""
    result = runner.invoke(app, ["predictions", "list", "--until", "not-a-date"])
    assert result.exit_code == 1
    assert "Invalid date format" in result.stdout


def test_list_predictions_limit(runner, mock_client, config_file):
    """predictions list --limit is accepted without error."""
    result = runner.invoke(app, ["predictions", "list", "--limit", "5"])
    assert result.exit_code == 0


def test_get_prediction_help(runner, mock_client, config_file):
    """predictions get --help shows documentation."""
    result = runner.invoke(app, ["predictions", "get", "--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "--wait" in out
    assert "--timeout" in out
