"""Generation API commands."""
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

app = typer.Typer(help="Generation operations")

@app.command()
def image(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Image generation prompt"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
) -> None:
    """Generate an image."""
    client = ctx.obj
    image_data = client.generate_image(prompt)
    if output:
        output.write_bytes(image_data)
        rprint(f"Image saved to {output}")
    else:
        rprint("Image data generated (use --output to save to file)")

@app.command()
def video(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Video generation prompt"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
) -> None:
    """Generate a video."""
    client = ctx.obj
    video_data = client.generate_video(prompt)
    if output:
        output.write_bytes(video_data)
        rprint(f"Video saved to {output}")
    else:
        rprint("Video data generated (use --output to save to file)")

@app.command()
def document(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Document generation prompt"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
) -> None:
    """Generate a document."""
    client = ctx.obj
    document_data = client.generate_document(prompt)
    if output:
        output.write_bytes(document_data)
        rprint(f"Document saved to {output}")
    else:
        rprint("Document data generated (use --output to save to file)")
