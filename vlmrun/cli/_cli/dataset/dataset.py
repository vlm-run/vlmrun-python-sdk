import os
import typer
from typing import List
from pathlib import Path

from vlmrun.common.utils import create_archive

app = typer.Typer(help="Dataset operations")


@app.command()
def create(
    ctx: typer.Context,
    directory: Path = typer.Option(
        ...,
        help="Directory containing images, PDFs or videos",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    domain: str = typer.Option(..., help="Domain for the dataset"),
    dataset_name: str = typer.Option(..., help="Name of the dataset"),
    dataset_type: str = typer.Option(
        ..., help="Type of dataset ('images', 'pdfs', 'videos')"
    ),
) -> None:
    """Create a dataset from a directory of images.

    This command will:
    1. Create a tar.gz archive from the image directory
    2. Upload the archive using client.upload_file
    3. Create a dataset using the uploaded file
    """
    client = ctx.obj

    if dataset_type not in ("images", "pdfs", "videos"):
        typer.echo("Error: Invalid dataset type", err=True)
        raise typer.Exit(1)

    if dataset_type == "images":
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        files = [
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
    elif dataset_type == "pdfs":
        files = [
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() == ".pdf"
        ]

    elif dataset_type == "videos":
        video_extensions = {".mp4", ".webm"}
        files = [
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in video_extensions
        ]
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    if not len(files):
        typer.echo(
            f"Error: No files of type={dataset_type}] found in directory", err=True
        )
        raise typer.Exit(1)

    typer.echo(f"Available files [n={len(files)}, dataset_type={dataset_type}]")

    # Create archive
    try:
        archive_path = create_archive(directory)
        typer.echo(f"Created archive: {archive_path}")

        # Upload archive
        file_response = client.files.upload(archive_path, purpose="dataset")
        typer.echo(f"Uploaded archive with file ID: {file_response.id}")

        # Create dataset
        dataset_response = client.dataset.create(
            file_id=file_response.id,
            domain=domain,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
        )
        typer.echo(f"Dataset created successfully! ID: {dataset_response.dataset_id}")

        # Verify dataset creation
        dataset_info = client.dataset.get(dataset_response.dataset_id)
        typer.echo(f"Dataset file ID: {dataset_info.file_id}")

    finally:
        # Cleanup temporary archive
        if "archive_path" in locals() and os.path.exists(archive_path):
            os.unlink(archive_path)
            typer.echo("Cleaned up temporary archive")


@app.command()
def generate(
    ctx: typer.Context,
    domain: str = typer.Argument(..., help="Domain for dataset"),
    urls: List[str] = typer.Option(..., help="List of URLs"),
    url_type: str = typer.Option(
        ..., help="Type of URLs ('yt_playlist' or 'yt_video')", case_sensitive=False
    ),
    dataset_name: str = typer.Option(..., help="Name of the dataset"),
    dataset_format: str = typer.Option(
        "json", help="Format of the dataset ('json' or 'messages')"
    ),
    max_frames_per_video: int = typer.Option(
        ..., help="Maximum number of frames to generate per video", min=1
    ),
    max_samples: int = typer.Option(
        ...,
        help="Maximum number of samples to generate (between 10 and 100,000)",
        min=10,
        max=100_000,
    ),
) -> None:
    """
    Generate a dataset by calling /v1/dataset/generate.

    This command allows you to generate datasets from YouTube playlists or videos.
    The generated dataset will be saved according to the specified format and parameters.
    """
    client = ctx.obj

    # Validate url_type
    valid_url_types = ["yt_playlist", "yt_video"]
    if url_type.lower() not in [t.lower() for t in valid_url_types]:
        typer.echo(f"Error: url_type must be one of {valid_url_types}", err=True)
        raise typer.Exit(1)

    # Make the request to the endpoint
    response = client.api_request(
        method="POST",
        path="/v1/dataset/generate",
        json={
            "domain": domain,
            "urls": urls,
            "url_type": url_type,
            "dataset_name": dataset_name,
            "dataset_format": dataset_format,
            "max_frames_per_video": max_frames_per_video,
            "max_samples": max_samples,
        },
    )

    # Extract dataset_uri from response and display it
    if isinstance(response, dict) and "dataset_uri" in response:
        typer.echo("Dataset generated successfully!")
        typer.echo(f"Dataset URI: {response['dataset_uri']}")
        if "elapsed_s" in response:
            typer.echo(f"Generation time: {response['elapsed_s']:.2f} seconds")
    else:
        typer.echo(response)
