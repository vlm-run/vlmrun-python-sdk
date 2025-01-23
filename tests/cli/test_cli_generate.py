"""Test generate subcommand."""

from vlmrun.cli.cli import app


def test_generate_image(runner, mock_client, tmp_path):
    """Test generate image command."""
    result = runner.invoke(app, ["generate", "image", "test prompt"])
    assert result.exit_code == 0


def test_generate_video(runner, mock_client, tmp_path):
    """Test generate video command."""
    result = runner.invoke(app, ["generate", "video", "test prompt"])
    assert result.exit_code == 0


def test_generate_document(runner, mock_client, tmp_path):
    """Test generate document command."""
    result = runner.invoke(app, ["generate", "document", "test prompt"])
    assert result.exit_code == 0
