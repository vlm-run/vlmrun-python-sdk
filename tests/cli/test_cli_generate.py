"""Test generate subcommand."""

from pathlib import Path

from vlmrun.cli.cli import app
from vlmrun.common.utils import download_artifact


def test_generate_image(runner, mock_client, config_file, tmp_path):
    """Test generate image command."""
    path: Path = download_artifact(
        "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/document.invoice/invoice_1.jpg",
        format="file",
    )
    result = runner.invoke(
        app, ["generate", "image", str(path), "--domain", "document.invoice"]
    )
    assert result.exit_code == 0


def test_generate_document(runner, mock_client, config_file, tmp_path):
    """Test generate document command."""
    path: Path = download_artifact(
        "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/document.bank-statement/lending_bankstatement.pdf",
        format="file",
    )
    result = runner.invoke(
        app, ["generate", "document", str(path), "--domain", "document.bank-statement"]
    )
    assert result.exit_code == 0
