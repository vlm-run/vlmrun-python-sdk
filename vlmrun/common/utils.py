"""General utilities for VLMRun."""

from io import BytesIO
from pathlib import Path
from typing import Union, Literal
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

import requests
from PIL import Image
from vlmrun.constants import VLMRUN_CACHE_DIR


# HTTP request headers
_HEADERS = {
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


def remote_image(url: Union[str, Path]) -> Image.Image:
    """Load an image from a URL or local path.

    Args:
        url: URL or Path to the image

    Returns:
        PIL Image in RGB format

    Raises:
        ValueError: If URL/path is invalid
        requests.exceptions.RequestException: If there's an error downloading the image
        IOError: If there's an error processing the image
    """
    if isinstance(url, Path) or (isinstance(url, str) and not url.startswith("http")):
        try:
            return Image.open(url).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image from path={url}") from e

    try:
        response = requests.get(url, headers=_HEADERS, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException:
        # Let request exceptions propagate through
        raise
    except Exception as e:
        raise ValueError(f"Failed to process image from {url}") from e


def download_image(url: str) -> Image.Image:
    """Download an image from a URL.

    Args:
        url: URL of the image to download

    Returns:
        PIL Image in RGB format

    Raises:
        ValueError: If URL format is invalid
        requests.exceptions.RequestException: If there's an error downloading the image
    """
    if not isinstance(url, str):
        raise ValueError(f"URL must be a string, got {type(url)}")
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL format: {url}")
    return remote_image(url)


@retry(wait=wait_fixed(10), stop=stop_after_attempt(3), reraise=True)
def download_artifact(
    url: str, format: Literal["image", "json", "file"]
) -> Union[Image.Image, dict, Path]:
    if not url.startswith("http"):
        raise ValueError(f"Invalid URL: {url}")
    if format == "image":
        bytes = requests.get(url, headers=_HEADERS).content
        return Image.open(BytesIO(bytes)).convert("RGB")
    elif format == "json":
        return requests.get(url, headers=_HEADERS).json()
    elif format == "file":
        path = VLMRUN_CACHE_DIR / "downloads" / Path(url).name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with requests.get(url, headers=_HEADERS, stream=True) as r:
                r.raise_for_status()
                with path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        else:
            logger.debug(f"File already exists [path={path}]")
        return path
    else:
        raise ValueError(f"Invalid format: {format}")
