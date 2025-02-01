"""Generation API commands."""

from pathlib import Path
from typing import Optional, List

import typer
from rich import print as rprint

from vlmrun.client import VLMRun
from vlmrun.client.types import PredictionResponse

app = typer.Typer(help="Generation operations")


@app.command()
def image(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Image generation prompt"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
) -> None:
    """Generate an image."""
    vlm: VLMRun = ctx.obj
    response = vlm.image.generate(
        images=[],  # Empty list since we're using text-to-image
        model="default",
        domain="image",
        json_schema={"prompt": prompt}  # Pass prompt in json_schema
    )
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
    vlm: VLMRun = ctx.obj
    response = vlm.video.generate(
        file_or_url=prompt,  # Using prompt as input text
        model="default",
        domain="video"
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
    vlm: VLMRun = ctx.obj
    response = vlm.document.generate(
        file_or_url=prompt,  # Using prompt as input text
        model="default",
        domain="document"
    )
    if output and response and hasattr(response, "response"):
        if isinstance(response.response, bytes):
            output.write_bytes(response.response)
            rprint(f"Document saved to {output}")
        else:
            rprint("Error: Response does not contain valid document data")
    else:
        rprint("Document data generated (use --output to save to file)")
