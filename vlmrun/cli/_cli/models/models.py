"""Models API commands."""

import typer
from rich.table import Table
from rich.console import Console

app = typer.Typer(help="Model operations")


@app.command()
def list(ctx: typer.Context) -> None:
    """List available models."""
    client = ctx.obj
    models = client.list_models()
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model ID")
    table.add_column("Name")
    table.add_column("Description")

    for model in models:
        table.add_row(
            model["id"],
            model["name"],
            model["description"],
        )

    console.print(table)
