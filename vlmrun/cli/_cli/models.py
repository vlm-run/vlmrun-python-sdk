"""Gateway model catalog command."""

from __future__ import annotations

from typing import Optional

import typer

from vlmrun.cli._cli.gateway import run_models

MODELS_HELP = """List gateway models, or detail one model.

\b
EXAMPLES:
  vlmrun models                        List every model with its methods.
  vlmrun models paddleocr/pp-ocrv6     Methods, params and examples for one model.
  vlmrun models --json                 Raw model catalog.
"""


def models(
    ctx: typer.Context,
    model: Optional[str] = typer.Argument(
        None,
        help="Model id or alias. Shows that model's methods, params and examples.",
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON."),
) -> None:
    """List gateway models, or detail one model."""
    run_models(ctx, model, output_json, command="vlmrun models")
