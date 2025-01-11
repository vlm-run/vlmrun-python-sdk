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
    table.add_column("Model")
    table.add_column("Domain")

    for model in models:
        table.add_row(
            model["model"],
            model["domain"],
        )

    console.print(table)
