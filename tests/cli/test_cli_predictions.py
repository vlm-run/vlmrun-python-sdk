"""Test predictions subcommand."""

from vlmrun.cli.cli import app


def test_list_predictions(runner, mock_client, config_file, monkeypatch):
    """Test list predictions command."""
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["predictions", "list"])
    assert result.exit_code == 0
    assert "prediction1" in result.stdout
    assert "running" in result.stdout
    assert "2024-01-01" in result.stdout


def test_list_predictions_with_status_filter(runner, mock_client, config_file):
    """Test list predictions with status filter."""
    # Test with running status (should match mock data)
    result = runner.invoke(app, ["predictions", "list", "--status", "running"])
    assert result.exit_code == 0
    assert "prediction1" in result.stdout
    assert "running" in result.stdout

    result = runner.invoke(app, ["predictions", "list", "--status", "completed"])
    assert result.exit_code == 0
    assert "No predictions found" in result.stdout or result.stdout.strip() == ""


def test_get_prediction(runner, mock_client, config_file):
    """Test get prediction command."""
    result = runner.invoke(app, ["predictions", "get", "prediction1"])
    assert result.exit_code == 0
    assert "prediction1" in result.stdout
    assert "Status" in result.stdout
    assert "running" in result.stdout
    assert "Created" in result.stdout
    assert "2024-01-01" in result.stdout


def test_get_prediction_with_wait(runner, mock_client, config_file):
    """Test get prediction command with wait flag."""
    result = runner.invoke(
        app, ["predictions", "get", "prediction1", "--wait", "--timeout", "5"]
    )
    assert result.exit_code == 0
    assert "prediction1" in result.stdout
    assert "completed" in result.stdout
    assert "Credits Used: 100" in result.stdout
    assert "Response" in result.stdout
    assert "test" in result.stdout


def test_get_prediction_usage_display(runner, mock_client, config_file):
    """Test that prediction usage information is displayed correctly."""
    result = runner.invoke(app, ["predictions", "get", "prediction1", "--wait"])
    assert result.exit_code == 0
    assert "Credits Used" in result.stdout
    assert "100" in result.stdout


def test_list_predictions_table_format(runner, mock_client, config_file, monkeypatch):
    """Test that list output is formatted correctly."""
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["predictions", "list"])
    assert result.exit_code == 0
    assert "id" in result.stdout
    assert "reated_at" in result.stdout
    assert "status" in result.stdout
