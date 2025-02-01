"""Hub API commands."""

import typer
from rich import print as rprint

from vlmrun.client import VLMRun

app = typer.Typer(
    help="Hub operations",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def version(ctx: typer.Context) -> None:
    """Get hub version."""
    client: VLMRun = ctx.obj
    info = client.hub.info()
    version = info.version
    rprint(f"Hub version: {version}")
