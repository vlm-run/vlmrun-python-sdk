"""Generation API commands."""

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from vlmrun.client import VLMRun

app = typer.Typer(help="Generation operations")


@app.command()
def image(
    ctx: typer.Context,
    image: Path = typer.Argument(
        ..., help="Input image file", exists=True, readable=True
    ),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
) -> None:
    """Generate an image."""
    client: VLMRun = ctx.obj
    response = client.image.generate(images=[image], model="vlm-1", domain="image")
    if output and response and hasattr(response, "response"):
        if isinstance(response.response, bytes):
            output.write_bytes(response.response)
            rprint(f"Image saved to {output}")
        else:
            rprint("Error: Response does not contain valid image data")
    else:
        rprint("Image data generated (use --output to save to file)")


@app.command()
def video(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Video generation prompt"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
) -> None:
    """Generate a video."""
    client: VLMRun = ctx.obj
    response = client.video.generate(
        file_or_url=prompt, model="vlm-1", domain="video"  # Using prompt as input text
    )
    if output and response and hasattr(response, "response"):
        if isinstance(response.response, bytes):
            output.write_bytes(response.response)
            rprint(f"Video saved to {output}")
        else:
            rprint("Error: Response does not contain valid video data")
    else:
        rprint("Video data generated (use --output to save to file)")


@app.command()
def document(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Document generation prompt"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
) -> None:
    """Generate a document."""
    client: VLMRun = ctx.obj
    response = client.document.generate(
        file_or_url=prompt,  # Using prompt as input text
        model="vlm-1",
        domain="document",
    )
    if output and response and hasattr(response, "response"):
        if isinstance(response.response, bytes):
            output.write_bytes(response.response)
            rprint(f"Document saved to {output}")
        else:
            rprint("Error: Response does not contain valid document data")
    else:
        rprint("Document data generated (use --output to save to file)")
