"""Fine-tuning API commands."""

import typer
from rich.table import Table
from rich.console import Console
from rich import print as rprint

from vlmrun.client import Client
from vlmrun.client.types import FinetuningResponse

app = typer.Typer(
    help="Fine-tuning operations",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def create(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Base model name"),
    training_file_id: str = typer.Argument(..., help="Training file ID"),
    validation_file_id: str = typer.Option(None, help="Validation file ID"),
    num_epochs: int = typer.Option(1, help="Number of epochs"),
    batch_size: int = typer.Option(1, help="Batch size"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    suffix: str = typer.Option(None, help="Suffix for the fine-tuned model"),
    wandb_project_name: str = typer.Option(None, help="Weights & Biases project name"),
) -> None:
    """Create a fine-tuning job."""
    client: Client = ctx.obj
    result: FinetuningResponse = client.fine_tuning.create(
        model=model,
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        wandb_project_name=wandb_project_name,
        suffix=suffix,
    )
    rprint(f"Created fine-tuning job with ID: {result.id}")


@app.command()
def list(ctx: typer.Context) -> None:
    """List all fine-tuning jobs."""
    client: Client = ctx.obj
    jobs: list[FinetuningResponse] = client.fine_tuning.list()
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Job ID")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Created At")

    for job in jobs:
        table.add_row(
            job.id,
            job.model,
            job.status,
            job.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        )

    console.print(table)


@app.command()
def get(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="ID of the fine-tuning job"),
) -> None:
    """Get fine-tuning job details."""
    client: Client = ctx.obj
    job: FinetuningResponse = client.fine_tuning.get(job_id)
    rprint(job)


@app.command()
def cancel(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="ID of the fine-tuning job to cancel"),
) -> None:
    """Cancel a fine-tuning job."""
    client: Client = ctx.obj
    client.fine_tuning.cancel(job_id)
    rprint(f"Cancelled fine-tuning job {job_id}")
