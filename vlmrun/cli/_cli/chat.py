"""Chat command for VLM Run CLI - Visual AI from your terminal."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.status import Status

from vlmrun.client import VLMRun
from vlmrun.client.completions import CompletionChunk, CompletionResponse
from vlmrun.constants import (
    VLMRUN_ARTIFACT_CACHE_DIR,
    SUPPORTED_INPUT_FILETYPES,
)

app = typer.Typer(
    help="Chat with VLM Run's Orion visual AI agent",
    no_args_is_help=False,
)

console = Console()

# Available models
AVAILABLE_MODELS = [
    "vlmrun-orion-1:fast",
    "vlmrun-orion-1:auto",
    "vlmrun-orion-1:pro",
]

DEFAULT_MODEL = "vlmrun-orion-1:auto"


def read_prompt_from_stdin() -> Optional[str]:
    """Read prompt from stdin if available (piped input)."""
    if not sys.stdin.isatty():
        try:
            return sys.stdin.read().strip()
        except Exception:
            return None
    return None


def read_prompt_from_file(file_path: str) -> str:
    """Read prompt from a file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    return path.read_text().strip()


def resolve_prompt_option(prompt_value: str) -> str:
    """Resolve the -p/--prompt option value.

    The -p option can be:
    - "stdin" -> read from stdin
    - A path to an existing file -> read from file
    - Any other string -> use as the prompt text directly
    """
    # Check for stdin
    if prompt_value.lower() == "stdin":
        stdin_content = read_prompt_from_stdin()
        if stdin_content:
            return stdin_content
        raise ValueError("No input provided on stdin")

    # Check if it's a file path
    path = Path(prompt_value)
    if path.exists() and path.is_file():
        return path.read_text().strip()

    # Otherwise, treat as a literal prompt string
    return prompt_value


def resolve_prompt(
    prompt_arg: Optional[str],
    prompt_option: Optional[str],
) -> Optional[str]:
    """Resolve the prompt from various sources.

    Priority:
    1. Command line argument (prompt_arg) - unless it's "-" indicating stdin
    2. -p option (prompt_option) - can be text, file path, or "stdin"
    3. Auto-detect piped stdin
    """
    # Check if prompt argument indicates stdin
    if prompt_arg == "-":
        stdin_content = read_prompt_from_stdin()
        if stdin_content:
            return stdin_content
        raise ValueError("No input provided on stdin")

    # Priority 1: Command line argument (positional)
    if prompt_arg:
        return prompt_arg

    # Priority 2: -p option (can be text, file, or "stdin")
    if prompt_option:
        return resolve_prompt_option(prompt_option)

    # Priority 3: Auto-detect piped stdin
    stdin_content = read_prompt_from_stdin()
    if stdin_content:
        return stdin_content

    return None


def display_response(response: CompletionResponse, output_json: bool = False) -> None:
    """Display the completion response to the console."""
    if output_json:
        # Output raw JSON
        output = {
            "id": response.id,
            "model": response.model,
            "status": response.status,
            "content": response.content,
            "response": response.response,
            "created_at": response.created_at,
            "completed_at": response.completed_at,
            "artifacts": [
                {
                    "id": a.id,
                    "url": a.url,
                    "filename": a.filename,
                }
                for a in response.artifacts
            ],
            "usage": {
                "credits_used": response.usage.credits_used,
                "elements_processed": response.usage.elements_processed,
                "element_type": response.usage.element_type,
            } if response.usage else None,
        }
        console.print_json(json.dumps(output, indent=2, default=str))
    else:
        # Display formatted response
        console.print()

        # Try to get text content
        text = response.text
        if text:
            console.print(Markdown(text))
        elif response.response:
            # If no text, print the response dict nicely
            console.print_json(json.dumps(response.response, indent=2, default=str))
        elif response.content:
            console.print(Markdown(response.content))
        else:
            console.print("[dim]No text response[/]")


@app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(
        None,
        help="The prompt to send to Orion. Use '-' to read from stdin.",
    ),
    prompt_file: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Prompt text, file path, or 'stdin' for piped input.",
    ),
    input_files: Optional[List[Path]] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input file(s): images, videos, or documents. Can be repeated.",
        exists=True,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory for artifacts (default: ~/.vlm/cache/artifacts/)",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Model to use for processing.",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output raw JSON response.",
    ),
    no_stream: bool = typer.Option(
        False,
        "--no-stream",
        "-ns",
        help="Do not stream the response in real-time.",
    ),
    no_download: bool = typer.Option(
        False,
        "--no-download",
        "-nd",
        help="Skip artifact download.",
    ),
) -> None:
    """Chat with VLM Run's Orion visual AI agent.

    Process images, videos, and documents with natural language.

    Prompt Sources (in order of precedence):

    1. Command line argument

    2. -p option (text, file path, or 'stdin')

    3. Auto-detected piped stdin

    Examples:

        vlmrun chat "What's in this image?" -i photo.jpg

        vlmrun chat -p "What's in this image?" -i photo.jpg

        vlmrun chat -p long_prompt.txt -i photo.jpg

        echo "Describe this image" | vlmrun chat - -i photo.jpg

        echo "Describe this image" | vlmrun chat -p stdin -i photo.jpg
    """
    # Get client from context
    client: VLMRun = ctx.obj

    if client is None:
        console.print("[red]Error:[/] Client not initialized. Check your API key.")
        raise typer.Exit(1)

    # Resolve prompt from various sources
    try:
        final_prompt = resolve_prompt(prompt, prompt_file)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)

    if not final_prompt:
        console.print("[red]Error:[/] No prompt provided.")
        console.print("\nUsage:")
        console.print("  vlmrun chat \"Your prompt here\" -i file.jpg")
        console.print("  vlmrun chat -p prompt.txt -i file.jpg")
        console.print("  echo \"Your prompt\" | vlmrun chat - -i file.jpg")
        console.print("\nRun [green]vlmrun chat --help[/] for more options.")
        raise typer.Exit(1)

    # Validate model
    if model not in AVAILABLE_MODELS:
        console.print(f"[red]Error:[/] Invalid model '{model}'")
        console.print(f"\nAvailable models:")
        for m in AVAILABLE_MODELS:
            default_marker = " (default)" if m == DEFAULT_MODEL else ""
            console.print(f"  - {m}{default_marker}")
        raise typer.Exit(1)

    # Validate input files if provided
    if input_files:
        for file_path in input_files:
            suffix = file_path.suffix.lower()
            if suffix not in SUPPORTED_INPUT_FILETYPES:
                console.print(f"[red]Error:[/] Unsupported file type: {suffix}")
                console.print(f"\nSupported types: {', '.join(SUPPORTED_INPUT_FILETYPES)}")
                raise typer.Exit(1)

    # Generate a session ID for artifact organization
    session_id = uuid.uuid4().hex[:12]

    # Set up output directory
    if output_dir:
        artifact_dir = output_dir
    else:
        artifact_dir = VLMRUN_ARTIFACT_CACHE_DIR / session_id

    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Show upload progress if there are files
        if input_files and not output_json:
            console.print(f"\n[dim]Uploading {len(input_files)} file(s)...[/]")

        # Use the completions API with streaming
        if no_stream or output_json:
            # Non-streaming mode
            if not output_json:
                with Status("[bold blue]Processing...", console=console, spinner="dots") as status:
                    response = client.agent.completions.create(
                        prompt=final_prompt,
                        files=[str(f) for f in input_files] if input_files else None,
                        model=model,
                        stream=False,
                    )
            else:
                response = client.agent.completions.create(
                    prompt=final_prompt,
                    files=[str(f) for f in input_files] if input_files else None,
                    model=model,
                    stream=False,
                )

            # Display the response
            display_response(response, output_json=output_json)

            # Download artifacts if requested
            if not no_download and response.artifacts:
                console.print(f"\n[dim]Downloading {len(response.artifacts)} artifact(s)...[/]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Downloading...", total=len(response.artifacts))

                    for artifact in response.artifacts:
                        try:
                            output_path = client.agent.completions.download_artifact(
                                artifact,
                                artifact_dir,
                            )
                            progress.update(task, description=f"Downloaded {output_path.name}")
                        except Exception as e:
                            progress.update(task, description=f"[red]Failed: {e}[/]")

                        progress.advance(task)

                console.print(f"[green]Artifacts saved to:[/] {artifact_dir}")

            # Show usage info (unless JSON output)
            if not output_json and response.usage and response.usage.credits_used:
                console.print(f"\n[dim]Credits used: {response.usage.credits_used}[/]")

        else:
            # Streaming mode
            if not output_json:
                console.print(f"[dim]Using model: {model}[/]")

            last_status = None
            response = None

            with Status("[bold blue]Starting...", console=console, spinner="dots") as status:
                stream = client.agent.completions.create(
                    prompt=final_prompt,
                    files=[str(f) for f in input_files] if input_files else None,
                    model=model,
                    stream=True,
                )

                # Iterate through the stream
                for chunk in stream:
                    if isinstance(chunk, CompletionChunk):
                        # Update status based on chunk
                        if chunk.status and chunk.status != last_status:
                            last_status = chunk.status
                            status_messages = {
                                "enqueued": "[yellow]Request queued...",
                                "pending": "[yellow]Waiting to start...",
                                "running": "[bold blue]Processing your request...",
                                "completed": "[green]Complete!",
                                "failed": "[red]Failed",
                                "paused": "[yellow]Paused",
                                "error": "[red]Error",
                            }
                            status.update(status_messages.get(chunk.status, f"[blue]{chunk.status}..."))

                        if chunk.finished and chunk.status == "failed":
                            console.print(f"\n[red]Error:[/] {chunk.delta or 'Execution failed'}")
                            raise typer.Exit(1)

                    elif isinstance(chunk, CompletionResponse):
                        # This is the final response
                        response = chunk

            # Display the final response
            if response:
                display_response(response, output_json=output_json)

                # Download artifacts if requested
                if not no_download and response.artifacts:
                    console.print(f"\n[dim]Downloading {len(response.artifacts)} artifact(s)...[/]")

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task("Downloading...", total=len(response.artifacts))

                        for artifact in response.artifacts:
                            try:
                                output_path = client.agent.completions.download_artifact(
                                    artifact,
                                    artifact_dir,
                                )
                                progress.update(task, description=f"Downloaded {output_path.name}")
                            except Exception as e:
                                progress.update(task, description=f"[red]Failed: {e}[/]")

                            progress.advance(task)

                    console.print(f"[green]Artifacts saved to:[/] {artifact_dir}")

                # Show usage info (unless JSON output)
                if not output_json and response.usage and response.usage.credits_used:
                    console.print(f"\n[dim]Credits used: {response.usage.credits_used}[/]")

    except TimeoutError as e:
        console.print(f"[red]Timeout:[/] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
