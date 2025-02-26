"""Models API commands."""

from typing import List

import typer
from rich.table import Table
from rich.console import Console

from vlmrun.client import VLMRun
from vlmrun.client.types import ModelInfo

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
    models: List[ModelInfo] = client.models.list()

    if domain:
        models = [m for m in models if domain in m.domain]

    console = Console()
    table = Table(
        show_header=True,
        header_style="white",
        border_style="white",
        title="Available Models",
        title_style="white",
        expand=True,
    )

    table.add_column("Category")
    table.add_column("Model")
    table.add_column("Domain")

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
                table.add_row(category, model.model, model.domain, style="white")
                first_in_category = False
            else:
                table.add_row("", model.model, model.domain, style="white")
        if category != sorted(domain_groups.keys())[-1]:
            table.add_row("", "", "", style="dim white")

    console.print(table)
