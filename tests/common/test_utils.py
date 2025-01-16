from vlmrun.common.utils import download_artifact

PDF_URL = "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/document.bank-statement/lending_bankstatement.pdf"


def test_download_artifact():
    """Test that download_artifact can download a PDF."""
    pdf = download_artifact(PDF_URL, "file")
    assert pdf.exists()
