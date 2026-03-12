"""Models API commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from vlmrun.client import VLMRun
    from vlmrun.client.types import ModelInfo

app = typer.Typer(
    help="Model operations",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


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

    table = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
        expand=True,
    )

    table.add_column("CATEGORY")
    table.add_column("MODEL", style="bold cyan")
    table.add_column("DOMAIN", style="dim")

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
            table.add_row("", "", "")

    console.print(
        Panel(
            table,
            title="[bold]Models[/bold]",
            title_align="left",
            subtitle=f"[dim]{len(models)} model(s)[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 1),
        )
    )
