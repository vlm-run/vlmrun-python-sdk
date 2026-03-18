"""Predictions API commands."""

from __future__ import annotations

import typer
from typing import TYPE_CHECKING

from rich import box
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from datetime import datetime

if TYPE_CHECKING:
    from vlmrun.client import VLMRun

app = typer.Typer(
    help="List and retrieve prediction results.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


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
            from datetime import timezone

            since_date = datetime.strptime(since, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            predictions = [p for p in predictions if p.created_at >= since_date]
        except ValueError:
            console.print(
                "[red]Error:[/] Invalid date format for --since. Use YYYY-MM-DD"
            )
            raise typer.Exit(1)

    if until:
        try:
            from datetime import timezone

            until_date = datetime.strptime(until, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            predictions = [p for p in predictions if p.created_at <= until_date]
        except ValueError:
            console.print(
                "[red]Error:[/] Invalid date format for --until. Use YYYY-MM-DD"
            )
            raise typer.Exit(1)

    if not predictions:
        console.print("[yellow]No predictions found[/]")
        return

    table = Table(
        show_header=True,
        box=box.SIMPLE_HEAVY,
        header_style="bold white",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("ID", style="bold cyan", min_width=40)
    table.add_column("CREATED", style="dim")
    table.add_column("STATUS")
    table.add_column("ELEMENTS", style="dim")
    table.add_column("TYPE", style="dim")
    table.add_column("CREDITS", style="dim")
    for prediction in predictions:
        table.add_row(
            prediction.id,
            prediction.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            str(prediction.status),
            str(prediction.usage.elements_processed),
            str(prediction.usage.element_type),
            str(prediction.usage.credits_used),
        )

    console.print(
        Panel(
            table,
            title="[bold]Predictions[/bold]",
            title_align="left",
            subtitle=f"[dim]{len(predictions)} prediction(s)[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 1),
        )
    )


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

    error = getattr(prediction, "error", None)
    if prediction.status == "failed" and error:
        console.print("\nError:", style="white")
        console.print(error, style="white")
