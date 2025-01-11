"""Test fine-tuning subcommand."""

from vlmrun.cli.cli import app


def test_create_finetune(runner, mock_client):
    """Test create fine-tuning command."""
    result = runner.invoke(app, ["fine-tuning", "create", "file1", "test-model"])
    assert result.exit_code == 0
    assert "job1" in result.stdout


def test_list_finetune(runner, mock_client):
    """Test list fine-tuning command."""
    result = runner.invoke(app, ["fine-tuning", "list"])
    assert result.exit_code == 0
    assert "job1" in result.stdout
    assert "test-model" in result.stdout


def test_get_finetune(runner, mock_client):
    """Test get fine-tuning command."""
    result = runner.invoke(app, ["fine-tuning", "get", "job1"])
    assert result.exit_code == 0
    assert "running" in result.stdout


def test_cancel_finetune(runner, mock_client):
    """Test cancel fine-tuning command."""
    result = runner.invoke(app, ["fine-tuning", "cancel", "job1"])
    assert result.exit_code == 0


def test_status_finetune(runner, mock_client):
    """Test status fine-tuning command."""
    result = runner.invoke(app, ["fine-tuning", "status", "job1"])
    assert result.exit_code == 0
    assert "running" in result.stdout
