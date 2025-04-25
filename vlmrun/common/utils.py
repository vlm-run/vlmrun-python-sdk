"""General utilities for VLMRun."""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Union, Literal, Dict, Any
from loguru import logger

import tarfile
import requests
from PIL import Image, ImageOps
from vlmrun.constants import VLMRUN_TMP_DIR, VLMRUN_CACHE_DIR
from vlmrun.common.image import _open_image_with_exif


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
            return _open_image_with_exif(url)
        except Exception as e:
            raise ValueError(f"Failed to open image from path={url}") from e

    try:
        response = requests.get(url, headers=_HEADERS, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass
        return image.convert("RGB")
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


def create_archive(directory: Path, name: str) -> Path:
    """Create a tar.gz archive from a directory.

    Args:
        directory: Path to directory to archive
        name: Name of the archive
    Returns:
        str: Path to created archive file

    Raises:
        ValueError: If directory does not exist
    """
    if isinstance(directory, str):
        directory = Path(directory)

    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    # Create archive in temp directory
    logger.debug(
        f"Creating tar.gz file from directory [path={directory}, n_files={len(list(directory.iterdir()))}]"
    )
    date_str = datetime.now().strftime("%Y%m%d")
    archive_name = f"{name}_{date_str}"
    archive_path = VLMRUN_TMP_DIR / f"{archive_name}.tar.gz"

    # Check if tar.gz file already exists
    if archive_path.exists():
        logger.debug(f"Tar.gz file already exists [path={archive_path}]")
        return archive_path

    with tarfile.open(str(archive_path), "w:gz") as tar:
        tar.add(directory, arcname=archive_name)
    logger.debug(f"Created tar.gz file [path={archive_path}]")
    return archive_path


def download_artifact(
    url: str, format: Literal["image", "json", "file"]
) -> Union[Image.Image, Dict[str, Any], Path]:
    if not url.startswith("http"):
        raise ValueError(f"Invalid URL: {url}")
    if format == "image":
        bytes = requests.get(url, headers=_HEADERS).content
        image = Image.open(BytesIO(bytes))
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass
        return image.convert("RGB")
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
