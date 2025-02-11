"""Hub API commands."""

import typer
import json
from rich.table import Table
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from vlmrun.client import VLMRun
from vlmrun.client.types import HubDomainInfo, HubSchemaResponse
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
    console = Console()
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

    console = Console()
    table = Table(
        show_header=True,
        header_style="white",
        title="Available Domains",
        title_style="white",
        border_style="white",
        expand=True,
    )

    table.add_column("Category")
    table.add_column("Domain")

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
                table.add_row(category, domain.domain, style="white")
                first_in_category = False
            else:
                table.add_row("", domain.domain, style="white")
        if category != sorted(domain_groups.keys())[-1]:
            table.add_row("", "", style="white")

    console.print(table)


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

    console = Console()

    console.print("\nSchema Information:", style="white")
    meta_table = Table(show_header=False, box=None)
    meta_table.add_row("Domain:", domain, style="white")
    meta_table.add_row("Version:", response.schema_version, style="white")
    console.print(meta_table)

    console.print("\nJSON Schema:", style="white")
    json_str = json.dumps(response.json_schema, indent=2)
    syntax = Syntax(json_str, "json", theme="default", line_numbers=True)
    console.print(Panel(syntax, expand=False, border_style="white"))
