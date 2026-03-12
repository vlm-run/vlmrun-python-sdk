"""Hub API commands."""

from __future__ import annotations

import typer
import json
from typing import TYPE_CHECKING, List
from rich import box
from rich.table import Table
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

if TYPE_CHECKING:
    from vlmrun.client import VLMRun
    from vlmrun.client.types import HubDomainInfo, HubSchemaResponse

app = typer.Typer(
    help="Hub operations",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


@app.command()
def version(ctx: typer.Context) -> None:
    """Get hub version."""
    client: VLMRun = ctx.obj
    info = client.hub.info()
    console.print(f"Hub version: {info.version}", style="white")
    console.print(
        "\nVisit https://github.com/vlm-run/vlmrun-hub for more information",
        style="white",
    )


@app.command("list")
def list_domains(
    ctx: typer.Context,
    domain: str = typer.Option(
        None, help="Filter domains (e.g. 'document' or 'document.invoice')"
    ),
) -> None:
    """List hub domains."""
    client: VLMRun = ctx.obj
    domains: List[HubDomainInfo] = client.hub.list_domains()

    if domain:
        domains = [d for d in domains if domain in d.domain]

    table = Table(
        show_header=True,
        box=box.SIMPLE_HEAVY,
        header_style="bold white",
        padding=(0, 1),
        expand=True,
    )

    table.add_column("CATEGORY")
    table.add_column("DOMAIN", style="bold cyan")

    domain_groups = {}
    for domain in domains:
        category = domain.domain.split(".")[0]
        if category not in domain_groups:
            domain_groups[category] = []
        domain_groups[category].append(domain)

    for category in sorted(domain_groups.keys()):
        first_in_category = True
        for domain in sorted(domain_groups[category], key=lambda x: x.domain):
            if first_in_category:
                table.add_row(category, domain.domain)
                first_in_category = False
            else:
                table.add_row("", domain.domain)
        if category != sorted(domain_groups.keys())[-1]:
            table.add_row("", "")

    console.print(
        Panel(
            table,
            title="[bold]Domains[/bold]",
            title_align="left",
            subtitle=f"[dim]{len(domains)} domain(s)[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 1),
        )
    )


@app.command()
def schema(
    ctx: typer.Context,
    domain: str = typer.Argument(
        ..., help="Domain identifier (e.g. 'document.invoice')"
    ),
) -> None:
    """Get JSON schema for a domain."""
    client: VLMRun = ctx.obj
    response: HubSchemaResponse = client.hub.get_schema(domain)

    console.print("\nSchema Information:", style="white")
    meta_table = Table(show_header=False, box=None)
    meta_table.add_row("Domain:", domain, style="white")
    meta_table.add_row("Version:", response.schema_version, style="white")
    console.print(meta_table)

    console.print("\nJSON Schema:", style="white")
    json_str = json.dumps(response.json_schema, indent=2)
    syntax = Syntax(json_str, "json", theme="default", line_numbers=True)
    console.print(Panel(syntax, expand=False, border_style="white"))
