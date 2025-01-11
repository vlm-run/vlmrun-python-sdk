"""Fine-tuning API commands."""

import typer
from rich.table import Table
from rich.console import Console
from rich import print as rprint

app = typer.Typer(
    help="Fine-tuning operations",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def create(
    ctx: typer.Context,
    training_file: str = typer.Argument(..., help="Training file ID"),
    model: str = typer.Argument(..., help="Base model name"),
    n_epochs: int = typer.Option(1, help="Number of epochs"),
    batch_size: int = typer.Option(1, help="Batch size"),
    learning_rate: float = typer.Option(1e-5, help="Learning rate"),
) -> None:
    """Create a fine-tuning job."""
    client = ctx.obj
    result = client.create_fine_tuning_job(
        training_file=training_file,
        model=model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    rprint(f"Created fine-tuning job with ID: {result['id']}")


@app.command()
def list(ctx: typer.Context) -> None:
    """List all fine-tuning jobs."""
    client = ctx.obj
    jobs = client.list_fine_tuning_jobs()
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Job ID")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Created At")

    for job in jobs:
        table.add_row(
            job["id"],
            job["model"],
            job["status"],
            job["created_at"],
        )

    console.print(table)


@app.command()
def get(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="ID of the fine-tuning job"),
) -> None:
    """Get fine-tuning job details."""
    client = ctx.obj
    job = client.get_fine_tuning_job(job_id)
    rprint(job)


@app.command()
def cancel(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="ID of the fine-tuning job to cancel"),
) -> None:
    """Cancel a fine-tuning job."""
    client = ctx.obj
    client.cancel_fine_tuning_job(job_id)
    rprint(f"Cancelled fine-tuning job {job_id}")


@app.command()
def status(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="ID of the fine-tuning job"),
) -> None:
    """Get fine-tuning job status."""
    client = ctx.obj
    status = client.get_fine_tuning_job_status(job_id)
    rprint(f"Status for job {job_id}: {status['status']}")
