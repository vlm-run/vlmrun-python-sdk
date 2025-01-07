import requests
import pytest
from PIL import Image
from vlmrun.common.pdf import pdf_images
from loguru import logger


# URL of the PDF to be tested
PDF_URL = "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/document.bank-statement/lending_bankstatement.pdf"


@pytest.fixture
def pdf_file(tmp_path):
    """Download the PDF and save it to a temporary file."""
    response = requests.get(PDF_URL)
    response.raise_for_status()
    pdf_path = tmp_path / "test_document.pdf"
    pdf_path.write_bytes(response.content)
    return pdf_path


def test_pdf_images(pdf_file):
    """Test that pdf_images yields an iterator of valid images."""
    pages = pdf_images(pdf_file, dpi=72)
    for page in pages:
        assert isinstance(page.image, Image.Image)
        assert page.image.mode == "RGB"
        assert page.image.size[0] > 0 and page.image.size[1] > 0
        logger.debug(f"page={page}")
