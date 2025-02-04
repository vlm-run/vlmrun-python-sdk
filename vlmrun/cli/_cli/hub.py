"""Hub API commands."""

import typer
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from vlmrun.client import VLMRun
from vlmrun.client.types import HubDomainInfo
from typing import List

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


@app.command("list")
def list_domains(ctx: typer.Context) -> None:
    """List hub domains."""
    client: VLMRun = ctx.obj
    domains: List[HubDomainInfo] = client.hub.list_domains()
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("domain")

    for domain in domains:
        table.add_row(domain.domain)
    console.print(table)
