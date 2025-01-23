"""Hub API commands."""

import typer
from rich import print as rprint

app = typer.Typer(
    help="Hub operations",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def version(ctx: typer.Context) -> None:
    """Get hub version."""
    client = ctx.obj
    version = client.hub.version
    rprint(f"Hub version: {version}")
