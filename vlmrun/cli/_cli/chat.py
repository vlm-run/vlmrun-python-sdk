"""Chat command for VLM Run CLI - Visual AI from your terminal."""

from __future__ import annotations

import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.status import Status
from rich.table import Table

from vlmrun.client import VLMRun
from vlmrun.client.types import FileResponse
from vlmrun.constants import (
    VLMRUN_ARTIFACT_CACHE_DIR,
    SUPPORTED_INPUT_FILETYPES,
)

CHAT_HELP = """Process images, videos, and documents with natural language.

\b
PROMPT SOURCES (precedence order):
  1. Argument   vlmrun chat "prompt" -i file.jpg
  2. -p option  vlmrun chat -p prompt.txt -i file.jpg
  3. Stdin      echo "prompt" | vlmrun chat - -i file.jpg

\b
EXAMPLES:
  vlmrun chat "Describe this" -i photo.jpg
  vlmrun chat "Compare" -i a.jpg -i b.jpg
  vlmrun chat -p prompt.txt -i doc.pdf --json
  echo "Summarize" | vlmrun chat -p stdin -i video.mp4

\b
MODELS:
  vlmrun-orion-1:fast  Speed-optimized
  vlmrun-orion-1:auto  Auto-select (default)
  vlmrun-orion-1:pro   Most capable

\b
OUTPUT:
  --simple    Plain text output (token-efficient for automation)
  --json      JSON output for programmatic use

\b
FILES: .jpg .png .gif .mp4 .mov .pdf .doc .mp3 .wav (and more)
"""

app = typer.Typer(
    help=CHAT_HELP,
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


def upload_single_file(client: VLMRun, file_path: Path) -> FileResponse:
    """Upload a single file and return the FileResponse."""
    return client.files.upload(file_path, purpose="assistants")


def upload_files_concurrent(
    client: VLMRun, files: List[Path], simple_output: bool = False
) -> List[FileResponse]:
    """Upload files concurrently and return their FileResponses."""
    file_responses: List[FileResponse] = []

    if simple_output:
        # Simple output: just upload without progress display
        with ThreadPoolExecutor(max_workers=min(len(files), 4)) as executor:
            futures = {
                executor.submit(upload_single_file, client, f): f for f in files
            }
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    response = future.result()
                    file_responses.append(response)
                except Exception as e:
                    raise typer.Exit(1) from e
    else:
        # Rich output with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Uploading files...", total=len(files))

            with ThreadPoolExecutor(max_workers=min(len(files), 4)) as executor:
                futures = {
                    executor.submit(upload_single_file, client, f): f for f in files
                }
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        response = future.result()
                        file_responses.append(response)
                        progress.update(
                            task, description=f"Uploaded {file_path.name}"
                        )
                    except Exception as e:
                        console.print(f"[red]Error uploading {file_path.name}:[/] {e}")
                        raise typer.Exit(1) from e
                    progress.advance(task)

    return file_responses


def build_messages(
    prompt: str, file_responses: Optional[List[FileResponse]] = None
) -> List[Dict[str, Any]]:
    """Build OpenAI-style messages with optional file attachments."""
    content: List[Dict[str, Any]] = []

    # Add files as image_url content parts using public URLs
    if file_responses:
        for file_resp in file_responses:
            url = file_resp.public_url
            if url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })

    # Add text prompt
    content.append({
        "type": "text",
        "text": prompt,
    })

    return [{"role": "user", "content": content}]


def extract_artifacts(response_content: str) -> List[Dict[str, str]]:
    """Extract artifact URLs from response content."""
    artifacts = []

    # Try to parse as JSON to find artifact URLs
    try:
        data = json.loads(response_content)
        if isinstance(data, dict):
            # Look for common artifact patterns
            for key in ["artifacts", "files", "outputs"]:
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict) and "url" in item:
                            artifacts.append({
                                "url": item["url"],
                                "filename": item.get("filename"),
                            })
                        elif isinstance(item, str) and item.startswith(("http://", "https://")):
                            artifacts.append({"url": item, "filename": None})
    except (json.JSONDecodeError, TypeError):
        pass

    return artifacts


def download_artifact(url: str, output_dir: Path, filename: Optional[str] = None) -> Path:
    """Download an artifact from a URL to the output directory."""
    # Determine filename from URL if not provided
    if not filename:
        parsed = urlparse(url)
        filename = Path(parsed.path).name or f"artifact_{uuid.uuid4().hex[:8]}"

    output_path = output_dir / filename

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return output_path


def format_usage_table(usage: Dict[str, Any], latency_ms: float) -> Table:
    """Create a usage statistics table."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    if usage:
        if usage.get("prompt_tokens"):
            table.add_row("Prompt tokens", str(usage["prompt_tokens"]))
        if usage.get("completion_tokens"):
            table.add_row("Completion tokens", str(usage["completion_tokens"]))
        if usage.get("total_tokens"):
            table.add_row("Total tokens", str(usage["total_tokens"]))

    table.add_row("Latency", f"{latency_ms:.0f}ms")
    return table


def print_simple_output(
    content: str,
    model: str,
    latency_ms: float,
    usage: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, str]]] = None,
    artifact_dir: Optional[Path] = None,
) -> None:
    """Print simple, token-efficient output for automation."""
    print(content)
    print()
    print(f"---")
    print(f"model: {model}")
    print(f"latency: {latency_ms:.0f}ms")
    if usage:
        tokens = usage.get("total_tokens")
        if tokens:
            print(f"tokens: {tokens}")
    if artifacts and artifact_dir:
        print(f"artifacts: {artifact_dir}")


def print_rich_output(
    content: str,
    model: str,
    latency_ms: float,
    usage: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, str]]] = None,
    artifact_dir: Optional[Path] = None,
) -> None:
    """Print rich-formatted output with panels."""
    # Main response panel
    console.print()
    console.print(
        Panel(
            Markdown(content),
            title="[bold blue]Response[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Footer with usage stats
    footer_parts = [f"[dim]Model:[/dim] {model}", f"[dim]Latency:[/dim] {latency_ms:.0f}ms"]

    if usage:
        tokens = usage.get("total_tokens")
        if tokens:
            footer_parts.append(f"[dim]Tokens:[/dim] {tokens}")

    if artifacts and artifact_dir:
        footer_parts.append(f"[dim]Artifacts:[/dim] {artifact_dir}")

    console.print(f"  {'  │  '.join(footer_parts)}")
    console.print()


@app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(
        None,
        help="Prompt text. Use '-' for stdin.",
    ),
    prompt_file: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Prompt: text string, file path, or 'stdin'.",
    ),
    input_files: Optional[List[Path]] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input file (image/video/document). Repeatable.",
        exists=True,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Artifact output directory. [default: ~/.vlm/cache/artifacts/<id>]",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Model: vlmrun-orion-1:fast|auto|pro",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output JSON instead of formatted text.",
    ),
    simple_output: bool = typer.Option(
        False,
        "--simple",
        "-s",
        help="Plain text output (token-efficient for automation).",
    ),
    no_stream: bool = typer.Option(
        False,
        "--no-stream",
        "-ns",
        help="Disable streaming (wait for complete response).",
    ),
    no_download: bool = typer.Option(
        False,
        "--no-download",
        "-nd",
        help="Skip artifact download.",
    ),
) -> None:
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
        # Upload input files concurrently if provided
        file_responses: List[FileResponse] = []
        if input_files:
            if not output_json and not simple_output:
                console.print(f"\n[dim]Uploading {len(input_files)} file(s)...[/]")

            file_responses = upload_files_concurrent(
                client, input_files, simple_output=simple_output or output_json
            )

            if not output_json and not simple_output:
                console.print(f"[green]Uploaded {len(file_responses)} file(s)[/]\n")

        # Build messages for the chat completion
        messages = build_messages(final_prompt, file_responses if file_responses else None)

        if not output_json and not simple_output and not no_stream:
            console.print(f"[dim]Using model: {model}[/]")

        # Call the OpenAI-compatible chat completions API
        openai_client = client.openai
        response_content = ""
        usage_data: Optional[Dict[str, Any]] = None

        start_time = time.time()

        if no_stream:
            # Non-streaming mode
            if not output_json and not simple_output:
                with Status("[bold blue]Processing...", console=console, spinner="dots"):
                    response = openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=False,
                    )
            else:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                )

            latency_ms = (time.time() - start_time) * 1000
            response_content = response.choices[0].message.content or ""

            if response.usage:
                usage_data = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            if output_json:
                output = {
                    "id": response.id,
                    "model": response.model,
                    "content": response_content,
                    "latency_ms": latency_ms,
                    "usage": usage_data,
                }
                console.print_json(json.dumps(output, indent=2, default=str))
            elif simple_output:
                print_simple_output(
                    response_content, model, latency_ms, usage_data
                )
            else:
                print_rich_output(
                    response_content, model, latency_ms, usage_data
                )

        else:
            # Streaming mode
            with Status("[bold blue]Processing...", console=console, spinner="dots") as status:
                stream = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )

                # Collect streaming content
                chunks = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)

                response_content = "".join(chunks)

            latency_ms = (time.time() - start_time) * 1000

            # Display the complete response
            if output_json:
                output = {
                    "content": response_content,
                    "latency_ms": latency_ms,
                }
                console.print_json(json.dumps(output, indent=2, default=str))
            elif simple_output:
                print_simple_output(response_content, model, latency_ms)
            else:
                print_rich_output(response_content, model, latency_ms)

        # Extract and download artifacts if present
        if not no_download:
            artifacts = extract_artifacts(response_content)
            if artifacts:
                if not output_json and not simple_output:
                    console.print(f"\n[dim]Downloading {len(artifacts)} artifact(s)...[/]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    disable=simple_output or output_json,
                ) as progress:
                    task = progress.add_task("Downloading...", total=len(artifacts))

                    for artifact in artifacts:
                        try:
                            output_path = download_artifact(
                                artifact["url"],
                                artifact_dir,
                                artifact.get("filename"),
                            )
                            progress.update(task, description=f"Downloaded {output_path.name}")
                        except Exception as e:
                            progress.update(task, description=f"[red]Failed: {e}[/]")

                        progress.advance(task)

                if not output_json and not simple_output:
                    console.print(f"[green]Artifacts saved to:[/] {artifact_dir}")
                elif simple_output:
                    print(f"artifacts: {artifact_dir}")

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
