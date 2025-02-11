"""Predictions API commands."""

import typer
from rich.table import Table
from rich.console import Console
from rich import print as rprint
from datetime import datetime

from vlmrun.client import VLMRun

app = typer.Typer(
    help="Prediction operations",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def list(
    ctx: typer.Context,
    skip: int = typer.Option(0, help="Skip the first N predictions"),
    limit: int = typer.Option(10, help="Limit the number of predictions to list"),
    status: str = typer.Option(
        None,
        help="Filter by status (enqueued, pending, running, completed, failed, paused)",
    ),
    since: str = typer.Option(None, help="Show predictions since date (YYYY-MM-DD)"),
    until: str = typer.Option(None, help="Show predictions until date (YYYY-MM-DD)"),
) -> None:
    """List predictions."""
    client: VLMRun = ctx.obj
    predictions = client.predictions.list(skip=skip, limit=limit)

    if status:
        predictions = [p for p in predictions if p.status == status]

    if since:
        try:
            since_date = datetime.strptime(since, "%Y-%m-%d")
            predictions = [p for p in predictions if p.created_at >= since_date]
        except ValueError:
            rprint("[red]Error:[/] Invalid date format for --since. Use YYYY-MM-DD")
            raise typer.Exit(1)

    if until:
        try:
            until_date = datetime.strptime(until, "%Y-%m-%d")
            predictions = [p for p in predictions if p.created_at <= until_date]
        except ValueError:
            rprint("[red]Error:[/] Invalid date format for --until. Use YYYY-MM-DD")
            raise typer.Exit(1)

    if not predictions:
        rprint("[yellow]No predictions found[/]")
        return

    console = Console()
    table = Table(show_header=True, header_style="white", border_style="white")
    table.add_column("id", min_width=40)
    table.add_column("created_at")
    table.add_column("status")
    table.add_column("usage.elements_processed")
    table.add_column("usage.element_type")
    table.add_column("usage.credits_used")
    for prediction in predictions:
        table.add_row(
            prediction.id,
            prediction.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            str(prediction.status),
            str(prediction.usage.elements_processed),
            str(prediction.usage.element_type),
            str(prediction.usage.credits_used),
            style="white",
        )

    console.print(table)


@app.command()
def get(
    ctx: typer.Context,
    prediction_id: str = typer.Argument(..., help="ID of the prediction to retrieve"),
    wait: bool = typer.Option(False, help="Wait for prediction to complete"),
    timeout: int = typer.Option(60, help="Timeout in seconds when waiting"),
) -> None:
    """Get prediction details with optional wait functionality."""
    client: VLMRun = ctx.obj

    if wait:
        prediction = client.predictions.wait(prediction_id, timeout=timeout)
    else:
        prediction = client.predictions.get(prediction_id)

    console = Console()

    console.print("\nPrediction Details:\n", style="white")

    details = {
        "ID": prediction.id,
        "Status": prediction.status,
    }

    if prediction.created_at:
        details["Created"] = prediction.created_at.strftime("%Y-%m-%d %H:%M:%S")
    if prediction.completed_at:
        details["Completed"] = prediction.completed_at.strftime("%Y-%m-%d %H:%M:%S")

    if prediction.usage:
        if prediction.usage.elements_processed:
            details["Elements Processed"] = prediction.usage.elements_processed
        if prediction.usage.element_type:
            details["Element Type"] = prediction.usage.element_type
        if prediction.usage.credits_used:
            details["Credits Used"] = prediction.usage.credits_used

    for key, value in details.items():
        console.print(f"{key}: {value}", style="white")

    if prediction.response:
        console.print("\nResponse:", style="white")
        console.print(prediction.response, style="white")

    if prediction.status == "failed" and prediction.error:
        console.print("\nError:", style="white")
        console.print(prediction.error, style="white")
