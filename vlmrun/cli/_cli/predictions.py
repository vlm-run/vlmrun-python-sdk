"""Predictions API commands."""

from __future__ import annotations

import typer
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from datetime import datetime

if TYPE_CHECKING:
    from vlmrun.client import VLMRun

app = typer.Typer(
    help="List and retrieve prediction results.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


def _status_style(status: str) -> str:
    return {
        "completed": "green",
        "running": "yellow",
        "failed": "red",
        "pending": "dim",
        "enqueued": "dim",
        "paused": "yellow",
    }.get(status, "white")


def _compute_duration(created_at, completed_at, usage) -> str:
    if usage and usage.duration_seconds:
        return f"{usage.duration_seconds}s"
    if completed_at and created_at:
        delta = (completed_at - created_at).total_seconds()
        if delta > 0:
            return f"{int(delta)}s"
    return "-"


def _format_row(
    id_val: str,
    domain_val: str,
    status_val: str,
    status_style: str,
    created_val: str,
    dur_val: str,
    domain_width: int,
) -> Text:
    domain_trunc = domain_val[:domain_width].ljust(domain_width)
    line = Text()
    line.append(f" {id_val}  ", style="bold cyan")
    line.append(f"{domain_trunc}  ", style="white")
    line.append(f"{status_val:<10s}  ", style=status_style)
    line.append(f"{created_val}  ", style="dim")
    line.append(f"{dur_val:>6s}", style="dim")
    return line


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

    # Layout: pad(1) + ID(36) + gap(2) + domain(flex) + gap(2) + status(10) + gap(2) + created(16) + gap(2) + dur(6)
    # Fixed cols = 1+36+2 + 2+10+2+16+2+6 = 77
    panel_w = min(console.width, 150)
    inner_w = panel_w - 2
    domain_w = max(inner_w - 85, 10)

    header = Text()
    header.append(f" {'ID':<36s}  ", style="bold")
    header.append(f"{'DOMAIN':<{domain_w}s}  ", style="bold")
    header.append(f"{'STATUS':<10s}  ", style="bold")
    header.append(f"{'CREATED':<16s}  ", style="bold")
    header.append(f"{'DURATION':>6s}", style="bold")

    sep = Text(f" {'─' * (inner_w - 1)}")

    rows: list[Text] = [header, sep]
    for prediction in predictions:
        usage = prediction.usage
        st = _status_style(prediction.status)
        dur = _compute_duration(
            prediction.created_at, prediction.completed_at, usage
        )
        rows.append(
            _format_row(
                prediction.id,
                prediction.domain or "-",
                prediction.status,
                st,
                prediction.created_at.strftime("%Y-%m-%d %H:%M"),
                dur,
                domain_w,
            )
        )

    console.print(
        Panel(
            Group(*rows),
            title="[bold]Predictions[/bold]",
            title_align="left",
            subtitle=f"[dim]{len(predictions)} prediction(s)[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 0),
            width=panel_w,
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
        if prediction.usage.elements_processed is not None:
            details["Elements Processed"] = prediction.usage.elements_processed
        if prediction.usage.element_type is not None:
            details["Element Type"] = prediction.usage.element_type
        if prediction.usage.credits_used is not None:
            details["Credits Used"] = prediction.usage.credits_used

    dur = _compute_duration(
        prediction.created_at, prediction.completed_at, prediction.usage
    )
    if dur != "-":
        details["Duration"] = dur

    for key, value in details.items():
        console.print(f"{key}: {value}", style="white")

    if prediction.response:
        console.print("\nResponse:", style="white")
        console.print(prediction.response, style="white")

    error = getattr(prediction, "error", None)
    if prediction.status == "failed" and error:
        console.print("\nError:", style="white")
        console.print(error, style="white")
