"""Fine-tuning API commands."""

import typer
from typing import List
from rich.table import Table
from rich.console import Console
from rich import print as rprint

from vlmrun.client import VLMRun
from vlmrun.client.types import FinetuningResponse, FinetuningProvisionResponse

app = typer.Typer(
    help="Fine-tuning operations",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def create(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Base model name"),
    training_file: str = typer.Argument(..., help="Training file ID or URL"),
    validation_file: str = typer.Option(None, help="Validation file ID or URL"),
    num_epochs: int = typer.Option(1, help="Number of epochs"),
    batch_size: int = typer.Option(1, help="Batch size"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    suffix: str = typer.Option(None, help="Suffix for the fine-tuned model"),
    wandb_project_name: str = typer.Option(None, help="Weights & Biases project name"),
    wandb_api_key: str = typer.Option(None, help="Weights & Biases API key"),
    wandb_base_url: str = typer.Option(None, help="Weights & Biases base URL"),
) -> None:
    """Create a fine-tuning job."""
    client: VLMRun = ctx.obj
    result: FinetuningResponse = client.fine_tuning.create(
        model=model,
        suffix=suffix,
        training_file=training_file,
        validation_file=validation_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        wandb_project_name=wandb_project_name,
        wandb_api_key=wandb_api_key,
        wandb_base_url=wandb_base_url,
    )
    rprint(f"Created fine-tuning job with ID: {result.id}")


@app.command()
def list(ctx: typer.Context) -> None:
    """List all fine-tuning jobs."""
    client: VLMRun = ctx.obj
    jobs: List[FinetuningResponse] = client.fine_tuning.list()
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("id", min_width=40)
    table.add_column("model")
    table.add_column("status")
    table.add_column("created_at")
    table.add_column("completed_at")
    table.add_column("wandb_url")
    for job in jobs:
        table.add_row(
            job.id,
            job.model,
            job.status,
            job.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            job.completed_at.strftime("%Y-%m-%d %H:%M:%S") if job.completed_at else "",
            job.wandb_url,
        )

    console.print(table)


@app.command()
def provision(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Model to provision"),
    duration: int = typer.Option(10 * 60, help="Duration for the provisioned model"),
    concurrency: int = typer.Option(1, help="Concurrency for the provisioned model"),
) -> None:
    """Provision a fine-tuning model."""
    client: VLMRun = ctx.obj
    result: FinetuningProvisionResponse = client.fine_tuning.provision(
        model=model, duration=duration, concurrency=concurrency
    )
    rprint(f"Provisioned fine-tuning model\n{result}")


@app.command()
def get(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="ID of the fine-tuning job"),
) -> None:
    """Get fine-tuning job details."""
    client: VLMRun = ctx.obj
    job: FinetuningResponse = client.fine_tuning.get(job_id)
    rprint(job)


@app.command()
def cancel(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="ID of the fine-tuning job to cancel"),
) -> None:
    """Cancel a fine-tuning job."""
    client: VLMRun = ctx.obj
    client.fine_tuning.cancel(job_id)
    rprint(f"Cancelled fine-tuning job {job_id}")
