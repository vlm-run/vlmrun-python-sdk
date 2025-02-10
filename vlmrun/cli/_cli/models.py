"""Models API commands."""

from typing import List

import typer
from rich.table import Table
from rich.console import Console

from vlmrun.client import VLMRun
from vlmrun.client.types import ModelInfoResponse

app = typer.Typer(
    help="Model operations",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def list(
    ctx: typer.Context,
    domain: str = typer.Option(
        None, help="Filter domains (e.g. 'document' or 'document.invoice')"
    ),
) -> None:
    """List available models."""
    client: VLMRun = ctx.obj
    models: List[ModelInfoResponse] = client.models.list()

    if domain:
        models = [m for m in models if domain in m.domain]

    console = Console()
    table = Table(
        show_header=True,
        header_style="bold cyan",
        title="Available Models",
        title_style="bold",
        border_style="blue",
        expand=True,
    )

    table.add_column("Category", style="bold yellow")
    table.add_column("Model", style="magenta")
    table.add_column("Domain", style="green")

    domain_groups = {}
    for model in models:
        category = model.domain.split(".")[0]
        if category not in domain_groups:
            domain_groups[category] = []
        domain_groups[category].append(model)

    for category in sorted(domain_groups.keys()):
        first_in_category = True
        for model in sorted(domain_groups[category], key=lambda x: x.domain):
            if first_in_category:
                table.add_row(category, model.model, model.domain)
                first_in_category = False
            else:
                table.add_row("", model.model, model.domain)
        if category != sorted(domain_groups.keys())[-1]:
            table.add_row("", "", "", style="dim")

    console.print(table)
