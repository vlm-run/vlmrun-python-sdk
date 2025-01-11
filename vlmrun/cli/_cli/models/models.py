"""Models API commands."""
import typer
from rich.table import Table
from rich.console import Console

app = typer.Typer(help="Model operations")

@app.command()
def list() -> None:
    """List available models."""
    ctx = typer.get_app_ctx()
    client = ctx.obj
    # TODO: Implement list functionality
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model ID")
    table.add_column("Name")
    table.add_column("Description")
    
    console.print(table)
