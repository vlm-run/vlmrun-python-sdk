"""Main CLI implementation for vlmrun."""

from __future__ import annotations

from typing import Optional
import os

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from vlmrun.client import VLMRun
from vlmrun.cli._cli.files import app as files_app
from vlmrun.cli._cli.fine_tuning import app as fine_tuning_app
from vlmrun.cli._cli.models import app as models_app
from vlmrun.cli._cli.generate import app as generate_app
from vlmrun.cli._cli.hub import app as hub_app
from vlmrun.cli._cli.datasets import app as dataset_app
from vlmrun.cli._cli.predictions import app as predictions_app
from vlmrun.cli._cli.config import app as config_app, get_config
from vlmrun.constants import DEFAULT_BASE_URL

app = typer.Typer(
    name="vlmrun",
    help="CLI for VLM Run (https://app.vlm.run)",
    add_completion=True,
    no_args_is_help=True,
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from vlmrun import version

        rprint(f"vlmrun version: {version.__version__}")
        raise typer.Exit()


def check_credentials(
    ctx: typer.Context, api_key: Optional[str], base_url: str
) -> None:
    """Check if API key is present and show helpful message if missing."""
    config = get_config()
    api_key = api_key or config.api_key
    base_url = base_url or config.base_url

    if not api_key:
        console.print("\n[red bold]Error:[/] API key not found! 🔑\n")
        console.print(
            Panel(
                Text.from_markup(
                    "To use the VLM Run CLI, you need to provide an API key. You can either:\n\n"
                    "1. Set it in your config:\n"
                    "   [green]vlmrun config set --api-key 'your-api-key'[/]\n\n"
                    "2. Set it as an environment variable:\n"
                    "   [green]export VLMRUN_API_KEY='your-api-key'[/]\n\n"
                    "3. Pass it directly as an argument:\n"
                    "   [green]vlmrun --api-key 'your-api-key' ...[/]\n\n"
                    "Get your API key at: [blue]https://app.vlm.run/dashboard[/]"
                ),
                title="API Key Required",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    is_default_url = base_url == DEFAULT_BASE_URL and not os.getenv("VLMRUN_BASE_URL")
    if is_default_url and not ctx.meta.get("has_shown_base_url_notice"):
        console.print(
            f"[yellow]Note:[/] Using default API endpoint: [blue]{base_url}[/]\n"
            "To use a different endpoint, set [green]VLMRUN_BASE_URL[/] or use [green]--base-url[/]"
        )
        ctx.meta["has_shown_base_url_notice"] = True


@app.callback()
def main(
    ctx: typer.Context,
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="VLMRUN_API_KEY",
        help="VLM Run API key. Can also be set via VLMRUN_API_KEY environment variable.",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        envvar="VLMRUN_BASE_URL",
        help="VLM Run API base URL. Can also be set via VLMRUN_BASE_URL environment variable.",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode.",
    ),
) -> None:
    """VLM Run CLI tool for interacting with the VLM Run API platform."""
    if ctx.invoked_subcommand is not None:
        check_credentials(ctx, api_key, base_url)
        ctx.obj = VLMRun(api_key=api_key, base_url=base_url)


# Add subcommands
app.add_typer(files_app, name="files")
app.add_typer(predictions_app, name="predictions")
app.add_typer(fine_tuning_app, name="fine-tuning")
app.add_typer(models_app, name="models")
app.add_typer(generate_app, name="generate")
app.add_typer(hub_app, name="hub")
app.add_typer(dataset_app, name="datasets")
app.add_typer(config_app, name="config")

if __name__ == "__main__":
    app()
