"""Hub API commands."""
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich.console import Console
from rich import print as rprint

app = typer.Typer(help="Hub operations")

@app.command()
def version(ctx: typer.Context) -> None:
    """Get hub version."""
    client = ctx.obj
    version = client.get_hub_version()
    rprint(f"Hub version: {version}")

@app.command()
def list(ctx: typer.Context) -> None:
    """List hub items."""
    client = ctx.obj
    items = client.list_hub_items()
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Item ID")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Version")
    
    for item in items:
        table.add_row(
            item["id"],
            item["name"],
            item["type"],
            item["version"],
        )
    
    console.print(table)

@app.command()
def submit(
    ctx: typer.Context,
    path: Path = typer.Argument(..., help="Path to item to submit", exists=True),
    name: str = typer.Option(..., help="Name of the item"),
    version: str = typer.Option(..., help="Version of the item"),
) -> None:
    """Submit an item to the hub."""
    client = ctx.obj
    result = client.submit_hub_item(str(path), name=name, version=version)
    rprint(f"Submitted item with ID: {result['id']}")
