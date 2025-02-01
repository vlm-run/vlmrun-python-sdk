"""Test files subcommand."""

from vlmrun.cli.cli import app


def test_list_files(runner, mock_client):
    """Test list files command."""
    vlm = mock_client
    result = runner.invoke(app, ["files", "list"])
    assert result.exit_code == 0
    assert "file1" in result.stdout
    assert "test.txt" in result.stdout


def test_upload_file(runner, mock_client, tmp_path):
    """Test upload file command."""
    vlm = mock_client
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    result = runner.invoke(app, ["files", "upload", str(test_file)])
    assert result.exit_code == 0
    assert "file1" in result.stdout


def test_delete_file(runner, mock_client):
    """Test delete file command."""
    vlm = mock_client
    result = runner.invoke(app, ["files", "delete", "file1"])
    assert result.exit_code == 0


def test_get_file(runner, mock_client):
    """Test get file command."""
    vlm = mock_client
    result = runner.invoke(app, ["files", "get", "file1"])
    assert result.exit_code == 0
    assert "test content" in result.stdout
