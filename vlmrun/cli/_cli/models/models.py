"""Models API commands."""

from typing import List

import typer
from rich.table import Table
from rich.console import Console

from vlmrun.client import Client
from vlmrun.client.types import ModelInfoResponse

# Show the help message for the models command if no subcommand is provided
app = typer.Typer(
    help="Model operations",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def list(ctx: typer.Context) -> None:
    """List available models."""
    client: Client = ctx.obj
    models: List[ModelInfoResponse] = client.models.list()
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Domain")

    for model in models:
        table.add_row(
            model.model,
            model.domain,
        )

    console.print(table)
