"""Tests for image utilities."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Union

import pytest
import requests
from PIL import Image

from vlmrun.common.image import encode_image
from vlmrun.common.utils import download_image


def _validate_base64_image(data: Union[str, bytes], expected_format: str) -> None:
    """Helper to validate base64 encoded image data."""
    # Convert to string if needed
    data_str = data.decode() if isinstance(data, bytes) else str(data)

    # Validate prefix
    prefix = f"data:image/{expected_format.lower()};base64,"
    assert data_str.startswith(prefix), f"Data does not start with {prefix}"

    # Extract base64 content
    base64_str = data_str.split(",", 1)[1]

    # Validate base64 content
    try:
        base64.b64decode(base64_str.encode())
    except Exception as e:
        raise AssertionError(f"Invalid base64 content: {e}")


@pytest.fixture
def sample_image(tmp_path) -> Image.Image:
    """Create a sample image for testing."""
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test_image.png"
    img.save(img_path)
    return img


@pytest.fixture
def sample_image_path(sample_image, tmp_path) -> Path:
    """Save sample image to a file and return the path."""
    path = tmp_path / "test_image.png"
    sample_image.save(path)
    return path


def test_encode_image_from_pil(sample_image):
    """Test encoding a PIL Image."""
    # Test PNG encoding
    png_data = encode_image(sample_image, format="PNG")
    _validate_base64_image(png_data, "PNG")

    # Test JPEG encoding
    jpeg_data = encode_image(sample_image, format="JPEG")
    _validate_base64_image(jpeg_data, "JPEG")

    # Test binary format
    binary_data = encode_image(sample_image, format="binary")
    assert isinstance(binary_data, bytes)
    assert Image.open(BytesIO(binary_data)).mode == "RGB"


def test_encode_image_from_path(sample_image_path):
    """Test encoding an image from a file path."""
    # Test with string path
    str_path_data = encode_image(str(sample_image_path))
    _validate_base64_image(str_path_data, "PNG")

    # Test with Path object
    path_data = encode_image(sample_image_path)
    _validate_base64_image(path_data, "PNG")

    # Verify both methods produce same result
    assert str_path_data == path_data


def test_encode_image_invalid():
    """Test encoding with invalid inputs."""
    with pytest.raises(FileNotFoundError):
        encode_image("nonexistent.jpg")  # Invalid path


def test_download_image():
    """Test downloading an image from URL."""
    # Use a reliable test image URL
    url = "https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/hopper.png"

    # Download and verify image
    img = download_image(url)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size[0] > 0 and img.size[1] > 0


def test_download_image_invalid():
    """Test downloading with invalid URL."""
    with pytest.raises(requests.exceptions.RequestException):
        download_image("https://nonexistent.example.com/image.jpg")
