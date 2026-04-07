"""Unified generation command — auto-detects file type and routes to the right API."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from vlmrun.client import VLMRun
from vlmrun.client.types import (
    AgentSkill,
    AgentToolset,
    GenerationConfig,
    PredictionResponse,
)
from vlmrun.constants import (
    SUPPORTED_AUDIO_FILETYPES,
    SUPPORTED_DOCUMENT_FILETYPES,
    SUPPORTED_IMAGE_FILETYPES,
    SUPPORTED_INPUT_FILETYPES,
    SUPPORTED_VIDEO_FILETYPES,
)

console = Console()

AVAILABLE_TOOLSETS: List[str] = list(AgentToolset.__args__)

GENERATE_HELP = """Generate structured predictions for images, documents, videos, and audio.

\b
The file type is auto-detected from the extension and routed to the
appropriate API endpoint. Images and documents both use /v1/document/generate
(with batch=True) so they support skills and async processing.

\b
EXAMPLES:
  vlmrun generate -i invoice.pdf --domain document.invoice
  vlmrun generate -i photo.jpg --domain image.caption
  vlmrun generate -i meeting.mp4 --domain video.transcription
  vlmrun generate -i call.mp3 --domain audio.transcription
  vlmrun generate -i doc.pdf --domain document.invoice --skill ./my-skill
  vlmrun generate -i doc.pdf --skill-id invoice-extraction:latest
  vlmrun generate -i doc.pdf --schema schema.json

\b
SKILLS:
  --skill      Path to a local skill directory (inline)
  --skill-id   Server-side skill as <name>:<version> (e.g. my-skill:latest)
  Only one of --skill or --skill-id may be provided.
  When a skill is provided, --domain is optional.

\b
FILES: .jpg .png .gif .pdf .doc .docx .mp4 .mov .mp3 .wav (and more)
"""


def _build_inline_skill(directory: Path) -> AgentSkill:
    try:
        return AgentSkill.from_directory(directory)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1) from e


def _resolve_skills(
    skill_dirs: Optional[List[Path]],
    skill_ids: Optional[List[str]],
) -> Optional[List[AgentSkill]]:
    """Build AgentSkill list from --skill dirs or --skill-id references."""
    if skill_dirs and skill_ids:
        console.print(
            "[red]Error:[/] --skill and --skill-id are mutually exclusive."
        )
        raise typer.Exit(1)

    if skill_dirs:
        return [_build_inline_skill(d) for d in skill_dirs]

    if skill_ids:
        skills: List[AgentSkill] = []
        for sid in skill_ids:
            if ":" in sid:
                skill_name, skill_version = sid.rsplit(":", 1)
            else:
                skill_name, skill_version = sid, "latest"
            skills.append(
                AgentSkill(
                    type="skill_reference",
                    skill_id=skill_name,
                    version=skill_version,
                )
            )
        return skills

    return None


def _detect_media_type(path: Path) -> str:
    """Return one of 'image', 'document', 'video', 'audio' based on file extension."""
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_IMAGE_FILETYPES:
        return "image"
    if suffix in SUPPORTED_DOCUMENT_FILETYPES:
        return "document"
    if suffix in SUPPORTED_VIDEO_FILETYPES:
        return "video"
    if suffix in SUPPORTED_AUDIO_FILETYPES:
        return "audio"
    return "unknown"


def generate(
    ctx: typer.Context,
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input file (image/document/video/audio).",
        exists=True,
        readable=True,
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        "-d",
        help="Domain for generation (e.g. document.invoice, image.caption). Optional when a skill is provided.",
    ),
    skill_dirs: Optional[List[Path]] = typer.Option(
        None,
        "--skill",
        "-k",
        help="Path to a skill directory (must contain SKILL.md). Repeatable.",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    skill_ids: Optional[List[str]] = typer.Option(
        None,
        "--skill-id",
        help="Server-side skill as <name>:<version> (e.g. my-skill:latest). Repeatable.",
    ),
    json_schema_path: Optional[Path] = typer.Option(
        None,
        "--schema",
        help="Path to a JSON schema file for the response model.",
        exists=True,
        readable=True,
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Custom prompt to guide generation.",
    ),
    batch: bool = typer.Option(
        True,
        "--batch/--no-batch",
        help="Process in batch (async) mode. Default: batch.",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for the prediction to complete (when batch=True). Default: wait.",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        help="Timeout in seconds when waiting for prediction to complete.",
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Output format. Use 'json' for machine-readable JSON output.",
    ),
) -> None:
    """Generate structured predictions for images, documents, videos, and audio."""
    client: VLMRun = ctx.obj
    if client is None:
        console.print("[red]Error:[/] Client not initialized. Check your API key.")
        raise typer.Exit(1)

    output_json = False
    if output_format is not None:
        if output_format.lower() == "json":
            output_json = True
        else:
            console.print(f"[red]Error:[/] Unsupported output format '{output_format}'")
            raise typer.Exit(1)

    suffix = input_file.suffix.lower()
    if suffix not in SUPPORTED_INPUT_FILETYPES:
        console.print(f"[red]Error:[/] Unsupported file type: {suffix}")
        console.print(f"\nSupported types: {', '.join(SUPPORTED_INPUT_FILETYPES)}")
        raise typer.Exit(1)

    media_type = _detect_media_type(input_file)

    skills = _resolve_skills(skill_dirs, skill_ids)

    # domain is required unless a skill is provided
    if domain is None and skills is None:
        console.print(
            "[red]Error:[/] --domain is required unless a skill is provided via --skill or --skill-id."
        )
        raise typer.Exit(1)

    # Build GenerationConfig
    json_schema: Optional[Dict[str, Any]] = None
    if json_schema_path is not None:
        try:
            json_schema = json.loads(json_schema_path.read_text())
        except Exception as e:
            console.print(f"[red]Error:[/] Failed to load JSON schema: {e}")
            raise typer.Exit(1) from e

    config: Optional[GenerationConfig] = None
    if any([skills, json_schema, prompt]):
        config = GenerationConfig(
            skills=skills,
            json_schema=json_schema,
            prompt=prompt,
        )

    try:
        if not output_json:
            console.print(
                f"  [bold blue]Generating[/bold blue] [dim]{media_type}[/dim]"
                + (f" [dim]domain={domain}[/dim]" if domain else "")
            )

        # For images and documents, use the document endpoint (supports batch + skills).
        # For video/audio, use the respective endpoint.
        start_time = time.time()

        if media_type in ("image", "document"):
            with Status("Processing...", console=console, spinner="dots") if not output_json else _noop_ctx():
                response: PredictionResponse = client.document.generate(
                    file=input_file,
                    domain=domain,
                    batch=batch,
                    config=config,
                )
        elif media_type == "video":
            with Status("Processing...", console=console, spinner="dots") if not output_json else _noop_ctx():
                response = client.video.generate(
                    file=input_file,
                    domain=domain,
                    batch=batch,
                    config=config,
                )
        elif media_type == "audio":
            with Status("Processing...", console=console, spinner="dots") if not output_json else _noop_ctx():
                response = client.audio.generate(
                    file=input_file,
                    domain=domain,
                    batch=batch,
                    config=config,
                )
        else:
            console.print(f"[red]Error:[/] Could not determine media type for {input_file}")
            raise typer.Exit(1)

        # If batch mode and wait requested, poll until complete
        if batch and wait and response.status != "completed":
            if not output_json:
                with Status(
                    f"Waiting for prediction [cyan]{response.id}[/cyan]...",
                    console=console,
                    spinner="dots",
                ):
                    response = client.predictions.wait(response.id, timeout=timeout)
            else:
                response = client.predictions.wait(response.id, timeout=timeout)

        elapsed = time.time() - start_time

        if output_json:
            output = response.model_dump(mode="json")
            output["elapsed_s"] = round(elapsed, 2)
            print(json.dumps(output, indent=2, default=str))
        else:
            _print_prediction_result(response, elapsed)

    except TimeoutError as e:
        console.print(f"\n[yellow]Timeout:[/] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        raise typer.Exit(1) from e


class _noop_ctx:
    """No-op context manager for when we don't want a spinner."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _print_prediction_result(response: PredictionResponse, elapsed: float) -> None:
    status_style = "green" if response.status == "completed" else "yellow"
    stats = [f"[{status_style}]{response.status}[/{status_style}]"]
    if response.usage and response.usage.credits_used:
        stats.append(f"{response.usage.credits_used} credit(s)")
    stats.append(f"{elapsed:.1f}s")
    subtitle = " \u00b7 ".join(stats)

    if response.response:
        if isinstance(response.response, dict):
            body = json.dumps(response.response, indent=2, default=str)
        else:
            body = str(response.response)
    else:
        body = "[dim]No response body[/dim]"

    console.print(
        Panel(
            body,
            title=f"[bold]Prediction {response.id}[/bold]",
            title_align="left",
            subtitle=f"[dim]{subtitle}[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(1, 2),
        )
    )
