"""Generation API commands."""

from pathlib import Path
from typing import List

import typer
from PIL import Image
from rich import print as rprint

from vlmrun.client import VLMRun
from vlmrun.client.types import PredictionResponse
from vlmrun.common.image import _open_image_with_exif
from vlmrun.common.tiff import tiff_images


def _is_tiff_file(path: Path) -> bool:
    """Check if a file is a TIFF file."""
    return path.suffix.lower() in (".tif", ".tiff")


def _load_images_from_file(path: Path) -> List[Image.Image]:
    """Load images from a file, handling multi-page TIFFs.

    Args:
        path: Path to the image file

    Returns:
        List of PIL Image objects
    """
    if _is_tiff_file(path):
        return [page.image for page in tiff_images(path)]
    else:
        return [_open_image_with_exif(path)]


app = typer.Typer(help="Generation operations", no_args_is_help=True)


@app.command()
def image(
    ctx: typer.Context,
    image: Path = typer.Argument(
        ..., help="Input image file", exists=True, readable=True
    ),
    domain: str = typer.Option(
        ..., help="Domain to use for generation (e.g. `document.invoice`)"
    ),
) -> None:
    """Generate an image."""
    client: VLMRun = ctx.obj
    if not Path(image).is_file():
        raise typer.Abort(f"Image file does not exist: {image}")

    images: List[Image.Image] = _load_images_from_file(image)
    response: PredictionResponse = client.image.generate(images=images, domain=domain)
    rprint(response)


@app.command()
def document(
    ctx: typer.Context,
    path: Path = typer.Argument(
        ..., help="Path to the document file", exists=True, readable=True
    ),
    domain: str = typer.Option(
        ..., help="Domain to use for generation (e.g. `document.invoice`)"
    ),
) -> None:
    """Generate a document."""
    client: VLMRun = ctx.obj
    if not Path(path).is_file():
        raise typer.Abort(f"Document file does not exist: {path}")

    if _is_tiff_file(path):
        images: List[Image.Image] = [page.image for page in tiff_images(path)]
        response: PredictionResponse = client.image.generate(
            images=images, domain=domain
        )
    else:
        response = client.document.generate(file=path, domain=domain)
    rprint(response)
