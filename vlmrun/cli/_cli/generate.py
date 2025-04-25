"""Generation API commands."""

from pathlib import Path

import typer
from PIL import Image
from rich import print as rprint

from vlmrun.client import VLMRun
from vlmrun.client.types import PredictionResponse
from vlmrun.common.image import _open_image_with_exif

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

    img: Image.Image = _open_image_with_exif(image)
    response: PredictionResponse = client.image.generate(images=[img], domain=domain)
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

    response = client.document.generate(file=path, domain=domain)
    rprint(response)
