from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Tuple, Optional, Union

from PIL import Image

from loguru import logger


@dataclass
class Page:
    image: Image.Image
    """The image of the page."""
    page_number: int
    """The page number of the page."""


def pdf_images(
    path: Path,
    dpi: int = 150,
    pages: Optional[Tuple[int]] = None,
    backend: Literal["pypdfium2"] = "pypdfium2",
) -> Iterable[Union[Image.Image, Page]]:
    """Extract images from a PDF file."""
    if path.suffix not in (".pdf",):
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Supported types are .pdf"
        )

    logger.debug(
        f"Extracting images from PDF [path={path}, dpi={dpi}, size={path.stat().st_size / 1024 / 1024:.2f} MB]"
    )

    if backend == "pypdfium2":
        import pypdfium2 as pdfium

        logger.debug(f"Opening PDF document [path={path}]")
        doc = pdfium.PdfDocument(str(path))

        if pages is None:
            pages = range(len(doc))

        def iterator():
            for idx in pages:
                page = doc[idx]
                bitmap = page.render(scale=dpi / 72)
                image: Image.Image = bitmap.to_pil()
                yield Page(image=image, page_number=idx)

        yield from iterator()

        logger.debug(f"Closing PDF document [path={path}]")
        doc.close()
    else:
        raise ValueError(f"Unsupported backend: {backend}")
