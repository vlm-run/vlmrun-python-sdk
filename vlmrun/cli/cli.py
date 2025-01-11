"""Main CLI implementation for vlmrun."""

from __future__ import annotations

from typing import Optional

import typer
from rich import print as rprint

from vlmrun.client import Client
from vlmrun.cli._cli.files import app as files_app
from vlmrun.cli._cli.finetune import app as finetune_app
from vlmrun.cli._cli.models import app as models_app
from vlmrun.cli._cli.generate import app as generate_app
from vlmrun.cli._cli.hub import app as hub_app

app = typer.Typer(
    name="vlmrun",
    help="CLI for VLM Run (https://app.vlm.run)",
    add_completion=True,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from vlmrun import version

        rprint(f"vlmrun version: {version.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    base_url: Optional[str] = typer.Option(
        None,
        help="Base URL. Defaults to environment variable VLMRUN_BASE_URL",
        envvar="VLMRUN_BASE_URL",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        help="API Key. Defaults to environment variable VLMRUN_API_KEY",
        envvar="VLMRUN_API_KEY",
    ),
    debug: Optional[bool] = typer.Option(
        False,
        "--debug",
        help="Enable debug mode",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """VLM Run CLI tool for interacting with the VLM Run API platform."""
    ctx.obj = Client(api_key=api_key, base_url=base_url)


# Add subcommands
app.add_typer(files_app, name="files")
app.add_typer(finetune_app, name="fine-tuning")
app.add_typer(models_app, name="models")
app.add_typer(generate_app, name="generate")
app.add_typer(hub_app, name="hub")

if __name__ == "__main__":
    app()
