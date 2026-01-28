from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

from PIL import Image

from vlmrun.common.logging import logger


@dataclass
class TiffPage:
    image: Image.Image
    """The image of the page."""
    page_number: int
    """The page number of the page."""


def tiff_images(
    path: Path,
) -> Iterable[Union[Image.Image, TiffPage]]:
    """Extract images from a multi-page TIFF file.

    Args:
        path: Path to the TIFF file

    Yields:
        TiffPage objects containing the image and page number
    """
    if path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Supported types are .tif, .tiff"
        )

    logger.debug(
        f"Extracting images from TIFF [path={path}, size={path.stat().st_size / 1024 / 1024:.2f} MB]"
    )

    img = Image.open(path)
    n_frames = getattr(img, "n_frames", 1)

    logger.debug(f"TIFF has {n_frames} frame(s)")

    for i in range(n_frames):
        img.seek(i)
        frame = img.copy().convert("RGB")
        yield TiffPage(image=frame, page_number=i)

    img.close()
    logger.debug(f"Closed TIFF file [path={path}]")
