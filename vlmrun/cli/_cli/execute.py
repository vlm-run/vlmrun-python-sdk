"""Execute command for VLM Run CLI — submit agent executions."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.tree import Tree

from vlmrun.client import VLMRun
from vlmrun.client.types import (
    AgentExecutionConfig,
    AgentExecutionResponse,
    AgentSkill,
    AgentToolset,
)
from vlmrun.constants import SUPPORTED_INPUT_FILETYPES

console = Console()

AVAILABLE_MODELS = [
    "vlmrun-orion-1",
    "vlmrun-orion-1:lite",
    "vlmrun-orion-1:fast",
    "vlmrun-orion-1:auto",
    "vlmrun-orion-1:pro",
]

AVAILABLE_TOOLSETS: List[str] = list(AgentToolset.__args__)

DEFAULT_MODEL = "vlmrun-orion-1:auto"

EXECUTE_HELP = """Execute an agent via /v1/agent/execute.

\b
EXAMPLES:
  vlmrun execute -n my-agent:v1 -i invoice.pdf
  vlmrun execute -p "Extract invoice fields" -i doc.pdf --schema schema.json
  vlmrun execute -n my-agent:v1 -i img.jpg --skill ./my-skill
  vlmrun execute -n my-agent:v1 -i img.jpg --skill-id my-skill:latest
  vlmrun execute -p "Describe" -i photo.jpg --no-wait
  vlmrun execute -n my-agent:v1 -i a.jpg -i b.pdf -t image -t document

\b
AGENT NAME:
  Use <agent-name>:<agent-version> format (e.g. invoice-extractor:v2).
  If --name is omitted, the prompt is used to identify the agent.

\b
MODELS:
  vlmrun-orion-1:lite   Lightweight
  vlmrun-orion-1:fast   Speed-optimized
  vlmrun-orion-1:auto   Auto-select (default)
  vlmrun-orion-1:pro    Most capable

\b
SKILLS:
  --skill      Path to a local skill directory (inline)
  --skill-id   Server-side skill as <name>:<version>
  Only one of --skill or --skill-id may be provided.

\b
TOOLSETS:
  --toolset    Tool category to enable (repeatable).
               Available: core, image, image-gen, world-gen, viz, document, video, web

\b
FILES: .jpg .png .gif .mp4 .mov .pdf .doc .mp3 .wav (and more)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def _build_inline_skill(directory: Path) -> AgentSkill:
    try:
        return AgentSkill.from_directory(directory)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1) from e


def _resolve_skills(
    skill_dirs: Optional[List[Path]],
    skill_ids: Optional[List[str]],
    output_json: bool,
) -> Optional[List[AgentSkill]]:
    """Build AgentSkill list from --skill dirs or --skill-id references."""
    if skill_dirs and skill_ids:
        console.print(
            "[red]Error:[/] --skill and --skill-id are mutually exclusive. "
            "Provide one or the other, not both."
        )
        raise typer.Exit(1)

    if skill_dirs:
        skills = [_build_inline_skill(d) for d in skill_dirs]
        if not output_json:
            console.print(
                f"  [green]\u2713[/green] Loaded {len(skills)} skill(s) (inline)"
            )
        return skills

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
        if not output_json:
            console.print(
                f"  [green]\u2713[/green] Using {len(skills)} skill(s) (referenced)"
            )
        return skills

    return None


def _upload_files(
    client: VLMRun, files: List[Path], show_progress: bool = True
) -> List[Any]:
    """Upload files concurrently and return FileResponse objects."""
    file_responses: List[Any] = []

    if not show_progress:
        with ThreadPoolExecutor(max_workers=min(len(files), 4)) as executor:
            futures = {
                executor.submit(client.files.upload, file=f, purpose="assistants"): f
                for f in files
            }
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    file_responses.append(future.result())
                except Exception as e:
                    console.print(f"[red]Error uploading {file_path.name}:[/] {e}")
                    raise typer.Exit(1) from e
    else:
        with Status("Uploading...", console=console, spinner="dots") as status:
            with ThreadPoolExecutor(max_workers=min(len(files), 4)) as executor:
                futures = {
                    executor.submit(
                        client.files.upload, file=f, purpose="assistants"
                    ): f
                    for f in files
                }
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        file_responses.append(future.result())
                        status.update(f"Uploading {file_path.name}...")
                    except Exception as e:
                        console.print(
                            f"[red]Error uploading {file_path.name}:[/] {e}"
                        )
                        raise typer.Exit(1) from e

    return file_responses


def _print_execution_result(result: AgentExecutionResponse, elapsed: float) -> None:
    """Pretty-print a completed execution result."""
    status_style = "green" if result.status == "completed" else "red"

    stats = [f"[{status_style}]{result.status}[/{status_style}]"]
    if result.usage and result.usage.credits_used is not None:
        stats.append(f"{result.usage.credits_used} credit(s)")
    if result.usage and result.usage.steps is not None:
        stats.append(f"{result.usage.steps} step(s)")
    stats.append(f"{elapsed:.1f}s")
    subtitle = " \u00b7 ".join(stats)

    if result.response:
        body = json.dumps(result.response, indent=2, default=str)
    else:
        body = "[dim]No response body[/dim]"

    console.print(
        Panel(
            body,
            title=f"[bold]Execution {result.id}[/bold]",
            title_align="left",
            subtitle=f"[dim]{subtitle}[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


def execute(
    ctx: typer.Context,
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Agent name as <agent-name>:<agent-version> (e.g. my-agent:v1).",
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Prompt to guide the execution. Can be text or a path to a .txt/.md file.",
    ),
    input_files: Optional[List[Path]] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input file (image/video/document/audio). Repeatable.",
        exists=True,
        readable=True,
    ),
    json_schema_path: Optional[Path] = typer.Option(
        None,
        "--schema",
        help="Path to a JSON schema file for the response model.",
        exists=True,
        readable=True,
    ),
    skill_dirs: Optional[List[Path]] = typer.Option(
        None,
        "--skill",
        "-k",
        help=(
            "Path to a skill directory (must contain SKILL.md). Repeatable. "
            "Cannot be used together with --skill-id."
        ),
        exists=True,
        file_okay=False,
        readable=True,
    ),
    skill_ids: Optional[List[str]] = typer.Option(
        None,
        "--skill-id",
        help=(
            "Server-side skill reference as <skill-name>:<version> "
            "(e.g. my-skill:latest). Repeatable. "
            "Cannot be used together with --skill."
        ),
    ),
    toolsets: Optional[List[str]] = typer.Option(
        None,
        "--toolset",
        "-t",
        help=(
            "Tool category to enable (repeatable). "
            f"Available: {', '.join(list(AgentToolset.__args__))}."
        ),
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Model: vlmrun-orion-1[:lite|fast|auto|pro]",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for execution to complete (default: wait).",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        help="Timeout in seconds when waiting for execution to complete.",
    ),
    poll_interval: int = typer.Option(
        5,
        "--poll-interval",
        help="Seconds between status checks when waiting.",
    ),
    callback_url: Optional[str] = typer.Option(
        None,
        "--callback-url",
        help="URL to call when execution completes (webhook).",
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Output format. Use 'json' for JSON output.",
    ),
) -> None:
    """Execute an agent with files, skills, and structured output."""
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

    if model not in AVAILABLE_MODELS:
        console.print(f"[red]Error:[/] Invalid model '{model}'")
        console.print(f"\nAvailable models: {', '.join(AVAILABLE_MODELS)}")
        raise typer.Exit(1)

    if toolsets:
        for ts in toolsets:
            if ts not in AVAILABLE_TOOLSETS:
                console.print(f"[red]Error:[/] Invalid toolset '{ts}'")
                console.print(
                    f"\nAvailable toolsets: {', '.join(AVAILABLE_TOOLSETS)}"
                )
                raise typer.Exit(1)

    if input_files:
        for file_path in input_files:
            suffix = file_path.suffix.lower()
            if suffix not in SUPPORTED_INPUT_FILETYPES:
                console.print(f"[red]Error:[/] Unsupported file type: {suffix}")
                console.print(
                    f"\nSupported types: {', '.join(SUPPORTED_INPUT_FILETYPES)}"
                )
                raise typer.Exit(1)

    # Resolve prompt (may be a file path)
    final_prompt = None
    if prompt is not None:
        p = Path(prompt)
        if p.exists() and p.is_file():
            final_prompt = p.read_text().strip()
        else:
            final_prompt = prompt

    # Load JSON schema if provided
    json_schema: Optional[Dict[str, Any]] = None
    if json_schema_path is not None:
        try:
            json_schema = json.loads(json_schema_path.read_text())
        except Exception as e:
            console.print(f"[red]Error:[/] Failed to load JSON schema: {e}")
            raise typer.Exit(1) from e

    # Build skills
    skills = _resolve_skills(skill_dirs, skill_ids, output_json)

    try:
        # Upload files
        file_responses: List[Any] = []
        if input_files:
            file_responses = _upload_files(
                client, input_files, show_progress=not output_json
            )
            if not output_json:
                tree = Tree("", guide_style="dim", hide_root=True)
                for fp in input_files:
                    size_str = _format_file_size(fp.stat().st_size)
                    tree.add(f"{fp.name} [dim]({size_str})[/dim]")
                console.print(
                    Panel(
                        tree,
                        title=f"Uploaded {len(file_responses)} file(s)",
                        title_align="left",
                        border_style="dim",
                    )
                )

        # Build inputs dict from uploaded files
        inputs: Optional[Dict[str, Any]] = None
        if file_responses:
            inputs = {
                "files": [
                    {
                        "type": "file_url",
                        "file_url": {"url": fr.public_url},
                    }
                    for fr in file_responses
                ]
            }

        # Build config
        config = AgentExecutionConfig(
            prompt=final_prompt,
            json_schema=json_schema,
            skills=skills,
        )

        if not output_json:
            console.print(
                f"\n  [bold blue]Submitting execution[/bold blue]"
                + (f" [dim]({name})[/dim]" if name else "")
                + f" [dim]model={model}[/dim]"
            )

        # Execute
        response: AgentExecutionResponse = client.agent.execute(
            name=name,
            inputs=inputs,
            config=config,
            model=model,
            batch=True,
            callback_url=callback_url,
            toolsets=toolsets,
        )

        if not output_json:
            console.print(
                f"  [green]\u2713[/green] Execution submitted: [cyan]{response.id}[/cyan]"
                f"  status=[yellow]{response.status}[/yellow]"
            )

        if not wait:
            if output_json:
                print(
                    json.dumps(response.model_dump(mode="json"), indent=2, default=str)
                )
            else:
                console.print(
                    f"\n  [dim]Check status with:[/dim] vlmrun executions get {response.id}"
                )
            return

        # Poll until complete
        start_time = time.time()
        if not output_json:
            with Status(
                f"Waiting for execution [cyan]{response.id}[/cyan]...",
                console=console,
                spinner="dots",
            ):
                result = client.executions.wait(
                    response.id, timeout=timeout, sleep=poll_interval
                )
        else:
            result = client.executions.wait(
                response.id, timeout=timeout, sleep=poll_interval
            )

        elapsed = time.time() - start_time

        if output_json:
            output = result.model_dump(mode="json")
            output["elapsed_s"] = round(elapsed, 2)
            print(json.dumps(output, indent=2, default=str))
        else:
            _print_execution_result(result, elapsed)

    except TimeoutError as e:
        console.print(f"\n[yellow]Timeout:[/] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        raise typer.Exit(1) from e
