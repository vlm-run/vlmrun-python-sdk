"""Test generate subcommand."""

from vlmrun.cli.cli import app


def test_generate_image(runner, mock_client, tmp_path):
    """Test generate image command."""
    client = mock_client
    test_image = tmp_path / "test.jpg"
    test_image.write_bytes(b"test image data")
    result = runner.invoke(app, ["generate", "image", str(test_image)])
    assert result.exit_code == 0


def test_generate_video(runner, mock_client, tmp_path):
    """Test generate video command."""
    client = mock_client
    result = runner.invoke(app, ["generate", "video", "test prompt"])
    assert result.exit_code == 0


def test_generate_document(runner, mock_client, tmp_path):
    """Test generate document command."""
    client = mock_client
    result = runner.invoke(app, ["generate", "document", "test prompt"])
    assert result.exit_code == 0
