import os
import typer
from typing import List
from pathlib import Path

from vlmrun.common.utils import create_archive, list_image_files

app = typer.Typer(help="Dataset operations")


@app.command()
def create(
    ctx: typer.Context,
    directory: Path = typer.Option(..., help="Directory containing images", exists=True, dir_okay=True, file_okay=False),
    domain: str = typer.Option(..., help="Domain for the dataset"),
) -> None:
    """Create a dataset from a directory of images.

    This command will:
    1. Create a tar.gz archive from the image directory
    2. Upload the archive using client.upload_file
    3. Create a dataset using the uploaded file
    """
    client = ctx.obj

    # Verify directory contains images
    image_files = list_image_files(directory)
    if not image_files:
        typer.echo("Error: No image files found in directory", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(image_files)} image files")

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
            dataset_type="images"
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
