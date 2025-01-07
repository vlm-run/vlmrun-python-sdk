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
) -> Union[str, bytes]:
    """Convert an image to a base64 string or binary format.
    
    Args:
        image: PIL Image, path to image, or Path object
        format: Output format ("PNG", "JPEG", or "binary")
    
    Returns:
        Base64 encoded string or binary bytes
    
    Raises:
        FileNotFoundError: If image path doesn't exist
        ValueError: If image type is invalid
    """
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
    image.save(buffered, format=image_format)
    img_str = b64encode(buffered.getvalue()).decode()
    return f"data:image/{image_format.lower()};base64,{img_str}"


def download_image(url: str) -> Image.Image:
    """Download an image from a URL.
    
    Args:
        url: URL of the image to download
    
    Returns:
        PIL Image in RGB format
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
    bytes = requests.get(url, headers=_headers).content
    return Image.open(BytesIO(bytes)).convert("RGB")
