"""Predictions API commands."""

from typing import List

import typer
from rich.table import Table
from rich.console import Console
from vlmrun.client import VLMRun
from vlmrun.client.types import PredictionResponse

app = typer.Typer(
    help="Predictions API commands",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def list(ctx: typer.Context) -> None:
    """List all predictions."""
    client: VLMRun = ctx.obj

    predictions: List[PredictionResponse] = client.predictions.list()
    console = Console()
    table = Table(show_header=True)
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
        )

    console.print(table)
