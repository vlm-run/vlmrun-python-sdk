from pathlib import Path

from vlmrun.common.utils import download_artifact, create_archive

PDF_URL = "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/document.bank-statement/lending_bankstatement.pdf"


def test_download_artifact():
    """Test that download_artifact can download a PDF."""
    pdf = download_artifact(PDF_URL, "file")
    assert pdf.exists()


def test_create_archive():
    """Test that create_archive can create a tar.gz file."""
    import tarfile

    archive_path: Path = create_archive(
        Path(__file__).parent.parent / "test_data/image_dataset", "test_image_dataset"
    )
    assert archive_path.exists()
    assert archive_path.name.endswith(".tar.gz")

    # Unzip the archive and check if there is a folder with the same name as the stem
    stem = archive_path.name.replace(".tar.gz", "")
    with tarfile.open(archive_path, "r:gz") as tar:
        assert len(tar.getmembers()) == 4  # basedir + 3 images
        assert tar.getmembers()[0].name == stem
