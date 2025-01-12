"""Image utilities for VLMRun."""

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Literal, Union

import requests
from PIL import Image


def encode_image(
    image: Union[Image.Image, str, Path],
    format: Literal["PNG", "JPEG", "binary"] = "PNG",
    quality: int = 90,
) -> Union[str, bytes]:
    """Convert an image to a base64 string or binary format.

    Args:
        image: PIL Image, path to image, or Path object
        format: Output format ("PNG", "JPEG", or "binary")
        quality: JPEG quality (1-100, default 90). Only used for JPEG format.

    Returns:
        Base64 encoded string or binary bytes

    Raises:
        FileNotFoundError: If image path doesn't exist
        ValueError: If image type is invalid or quality is out of range
    """
    if not (1 <= quality <= 100):
        raise ValueError(f"Quality must be between 1 and 100, got {quality}")

    if isinstance(image, (str, Path)):
        if not Path(image).exists():
            raise FileNotFoundError(f"File not found {image}")
        image = Image.open(str(image)).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise ValueError(f"Invalid image type: {type(image)}")

    buffered = BytesIO()
    if format == "binary":
        image.save(buffered, format="PNG")
        buffered.seek(0)
        return buffered.getvalue()

    image_format = image.format or format
    if image_format.upper() not in ["PNG", "JPEG"]:
        raise ValueError(f"Unsupported format: {image_format}")

    save_params = {
        "format": image_format,
        **({"quality": quality, "subsampling": 0} if image_format.upper() == "JPEG" else {})
    }
    
    try:
        image.save(buffered, **save_params)
        img_str = b64encode(buffered.getvalue()).decode()
        return f"data:image/{image_format.lower()};base64,{img_str}"
    except Exception as e:
        raise ValueError(f"Failed to save image in {image_format} format") from e


def remote_image(url: str | Path) -> Image.Image:
    """Load an image from a URL or local path.

    Args:
        url: URL or Path to the image

    Returns:
        PIL Image in RGB format

    Raises:
        ValueError: If URL/path is invalid or image cannot be loaded
    """
    _headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    if isinstance(url, Path) or (isinstance(url, str) and not url.startswith("http")):
        try:
            return Image.open(url).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image from path={url}") from e

    try:
        response = requests.get(url, headers=_headers, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download image from url={url}") from e
    except Exception as e:
        raise ValueError(f"Failed to process image from url={url}") from e


def download_image(url: str) -> Image.Image:
    """Download an image from a URL.

    Args:
        url: URL of the image to download

    Returns:
        PIL Image in RGB format

    Raises:
        ValueError: If URL is invalid or image cannot be downloaded
    """
    if not isinstance(url, str):
        raise ValueError(f"URL must be a string, got {type(url)}")
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL format: {url}")
    return remote_image(url)
