"""Chat command for VLM Run CLI - Visual AI from your terminal."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.status import Status

from vlmrun.client import VLMRun
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


def upload_files(client: VLMRun, files: List[Path]) -> List[str]:
    """Upload files and return their public URLs."""
    urls = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Uploading files...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"Uploading {file_path.name}...")

            try:
                file_response = client.files.upload(file_path, purpose="assistants")

                # Get public URL
                public_url = file_response.public_url
                if not public_url and file_response.id:
                    public_url = client.files.get_public_url(file_response.id)

                if public_url:
                    urls.append(public_url)

            except Exception as e:
                console.print(f"[red]Error uploading {file_path.name}:[/] {e}")
                raise typer.Exit(1)

            progress.advance(task)

    return urls


def build_messages(prompt: str, file_urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Build OpenAI-style messages with optional file attachments."""
    content: List[Dict[str, Any]] = []

    # Add file URLs as image_url content parts
    if file_urls:
        for url in file_urls:
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
        # Upload input files if provided
        file_urls = []
        if input_files:
            if not output_json:
                console.print(f"\n[dim]Uploading {len(input_files)} file(s)...[/]")
            file_urls = upload_files(client, input_files)
            if not output_json:
                console.print(f"[green]Uploaded {len(file_urls)} file(s)[/]\n")

        # Build messages for the chat completion
        messages = build_messages(final_prompt, file_urls if file_urls else None)

        if not output_json and not no_stream:
            console.print(f"[dim]Using model: {model}[/]")

        # Call the OpenAI-compatible chat completions API
        openai_client = client.openai
        response_content = ""

        if no_stream:
            # Non-streaming mode
            if not output_json:
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

            response_content = response.choices[0].message.content or ""

            if output_json:
                output = {
                    "id": response.id,
                    "model": response.model,
                    "content": response_content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                        "completion_tokens": response.usage.completion_tokens if response.usage else None,
                        "total_tokens": response.usage.total_tokens if response.usage else None,
                    } if response.usage else None,
                }
                console.print_json(json.dumps(output, indent=2, default=str))
            else:
                console.print()
                console.print(Markdown(response_content))

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

            # Display the complete response
            if output_json:
                output = {
                    "content": response_content,
                }
                console.print_json(json.dumps(output, indent=2, default=str))
            else:
                console.print()
                console.print(Markdown(response_content))

        # Extract and download artifacts if present
        if not no_download:
            artifacts = extract_artifacts(response_content)
            if artifacts:
                console.print(f"\n[dim]Downloading {len(artifacts)} artifact(s)...[/]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
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

                console.print(f"[green]Artifacts saved to:[/] {artifact_dir}")

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
