import typer
from typing import List
from pathlib import Path
from rich.table import Table
from rich.console import Console

from loguru import logger
from vlmrun.client import VLMRun
from vlmrun.client.types import DatasetResponse
from vlmrun.constants import VLMRUN_TMP_DIR

app = typer.Typer(
    help="Dataset operations",
    add_completion=False,
    no_args_is_help=True,
)


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
        ..., help="Type of dataset ('images', 'documents', 'videos')"
    ),
    wandb_base_url: str = typer.Option(
        "https://api.wandb.ai", help="Base URL for Weights & Biases"
    ),
    wandb_project_name: str = typer.Option(
        None, help="Project name for Weights & Biases"
    ),
    wandb_api_key: str = typer.Option(None, help="API key for Weights & Biases"),
) -> None:
    """Create a dataset from a directory of images.

    This command will:
    1. Create a tar.gz archive from the image directory
    2. Upload the archive
    3. Create a dataset using the uploaded file
    """
    client: VLMRun = ctx.obj

    valid_types = {"images": "images", "pdfs": "documents", "videos": "videos"}
    if dataset_type not in valid_types:
        typer.echo("Error: Invalid dataset type", err=True)
        raise typer.Exit(1)
    dataset_type = valid_types[dataset_type]

    if dataset_type == "images":
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        files = [
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
    elif dataset_type == "documents":
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

    # Create dataset
    try:
        dataset_response = client.datasets.create(
            domain=domain,
            dataset_directory=directory,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            wandb_base_url=wandb_base_url,
            wandb_project_name=wandb_project_name,
            wandb_api_key=wandb_api_key,
        )
        typer.echo(f"Dataset created successfully! ID: {dataset_response.id}")

        # Verify dataset creation
        dataset_info = client.datasets.get(dataset_response.id)
        typer.echo(f"Dataset created with ID: {dataset_info.id}")

    finally:
        logger.debug(
            f"Generated dataset [dataset_id={dataset_response.id}, tmp_dir={VLMRUN_TMP_DIR}]"
        )


@app.command()
def list(
    ctx: typer.Context,
    skip: int = typer.Option(0, help="Skip the first N datasets"),
    limit: int = typer.Option(10, help="Limit the number of datasets to list"),
) -> None:
    """List datasets."""
    client: VLMRun = ctx.obj
    datasets: List[DatasetResponse] = client.datasets.list(skip=skip, limit=limit)

    console = Console()
    table = Table(show_header=True)
    table.add_column("id")
    table.add_column("dataset_name")
    table.add_column("dataset_type")
    table.add_column("domain")
    table.add_column("created_at")
    table.add_column("completed_at")
    table.add_column("status")
    table.add_column("wandb_url")

    for dataset in datasets:
        table.add_row(
            dataset.id,
            dataset.dataset_name,
            dataset.dataset_type,
            dataset.domain,
            dataset.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            (
                dataset.completed_at.strftime("%Y-%m-%d %H:%M:%S")
                if dataset.completed_at
                else ""
            ),
            dataset.status,
            dataset.wandb_url,
        )
    console.print(table)
