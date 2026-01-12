"""Chat command for VLM Run CLI - Visual AI from your terminal."""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.tree import Tree

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
  --json      JSON output for programmatic use

\b
FILES: .jpg .png .gif .mp4 .mov .pdf .doc .mp3 .wav (and more)
"""

app = typer.Typer(
    help=CHAT_HELP,
    no_args_is_help=True,
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


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def upload_files_concurrent(
    client: VLMRun, files: List[Path], show_progress: bool = True
) -> List[FileResponse]:
    """Upload files concurrently and return their file responses."""
    file_responses: List[FileResponse] = []

    if not show_progress:
        # JSON output: upload without progress display
        with ThreadPoolExecutor(max_workers=min(len(files), 4)) as executor:
            futures = {
                executor.submit(client.files.upload, file=f, purpose="assistants"): f
                for f in files
            }
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    file_response = future.result()
                    file_responses.append(file_response)
                except Exception as e:
                    console.print(f"[red]Error uploading {file_path.name}:[/] {e}")
                    raise typer.Exit(1) from e
    else:
        # Rich output - upload with status spinner
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
                        file_response = future.result()
                        file_responses.append(file_response)
                        status.update(f"Uploading {file_path.name}...")
                    except Exception as e:
                        console.print(f"[red]Error uploading {file_path.name}:[/] {e}")
                        raise typer.Exit(1) from e

    return file_responses


def build_messages(
    prompt: str, file_responses: Optional[List[FileResponse]] = None
) -> List[Dict[str, Any]]:
    """Build OpenAI-style messages with optional file attachments."""
    content: List[Dict[str, Any]] = []

    # Add files using file IDs
    if file_responses:
        for file_response in file_responses:
            content.append({"type": "input_file", "file_id": file_response.id})

    # Add text prompt
    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


def extract_artifact_refs(response_content: str) -> List[str]:
    """Extract artifact reference IDs from response content.

    Looks for patterns like img_XXXXXX, aud_XXXXXX, vid_XXXXXX, doc_XXXXXX,
    recon_XXXXXX, arr_XXXXXX, url_XXXXXX in the response text.
    """
    # Reference patterns from vlmrun/types/refs.py
    patterns = [
        r"\bimg_\w{6}\b",  # ImageRef
        r"\baud_\w{6}\b",  # AudioRef
        r"\bvid_\w{6}\b",  # VideoRef
        r"\bdoc_\w{6}\b",  # DocumentRef
        r"\brecon_\w{6}\b",  # ReconRef
        r"\barr_\w{6}\b",  # ArrayRef
        r"\burl_\w{6}\b",  # UrlRef
    ]

    refs: Set[str] = set()
    for pattern in patterns:
        matches = re.findall(pattern, response_content)
        refs.update(matches)

    return sorted(list(refs))


def get_file_extension(ref_id: str) -> str:
    """Get appropriate file extension based on reference type."""
    prefix = ref_id.split("_")[0]
    extensions = {
        "img": ".png",
        "aud": ".mp3",
        "vid": ".mp4",
        "doc": ".pdf",
        "recon": ".obj",
        "arr": ".npy",
        "url": ".txt",
    }
    return extensions.get(prefix, ".bin")


def download_artifact(
    client: VLMRun, session_id: str, ref_id: str, output_dir: Path
) -> Path:
    """Download an artifact using the artifacts API.

    Args:
        client: VLMRun client instance
        session_id: The session ID from the chat response
        ref_id: The artifact reference ID (e.g., img_abc123)
        output_dir: Directory to save the artifact

    Returns:
        Path to the downloaded artifact file
    """
    from PIL import Image
    from pydantic import AnyHttpUrl
    import shutil

    # Get the artifact from the API
    artifact = client.artifacts.get(session_id=session_id, object_id=ref_id)

    # Determine filename
    extension = get_file_extension(ref_id)
    filename = f"{ref_id}{extension}"
    output_path = output_dir / filename

    # Handle different artifact types
    if isinstance(artifact, Image.Image):
        # Save PIL Image to file
        artifact.save(output_path, format="JPEG", quality=95)
    elif isinstance(artifact, Path):
        # Copy file from temp location to output directory
        shutil.copy2(artifact, output_path)
    elif isinstance(artifact, AnyHttpUrl):
        # Save URL to text file
        with open(output_path, "w") as f:
            f.write(str(artifact))
    else:
        raise TypeError(f"Unexpected artifact type: {type(artifact)}")

    return output_path


def format_time(seconds: float) -> str:
    """Format time in human-readable format (e.g., '0.45s', '3.12s', '1m 2s')."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"


class TimedStatus:
    """A status display that shows elapsed time."""

    def __init__(self, message: str, console: Console, spinner: str = "dots"):
        self.base_message = message
        self.console = console
        self.spinner = spinner
        self.start_time = None
        self.status = None
        self._stop_event = threading.Event()
        self._timer_thread = None

    def __enter__(self):
        self.start_time = time.time()
        self.status = Status(
            f"[bold blue]{self.base_message}",
            console=self.console,
            spinner=self.spinner,
        )
        self.status.start()

        # Start timer thread
        self._timer_thread = threading.Thread(target=self._update_timer, daemon=True)
        self._timer_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._timer_thread:
            self._timer_thread.join(timeout=0.5)
        if self.status:
            self.status.stop()
        return False

    def _update_timer(self):
        """Update the status message with elapsed time."""
        while not self._stop_event.is_set():
            if self.status and self.start_time:
                elapsed = time.time() - self.start_time
                time_str = format_time(elapsed)
                self.status.update(f"[bold blue]{self.base_message} ({time_str})")
            time.sleep(0.1)  # Update every 100ms


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


def print_rich_output(
    content: str,
    model: str,
    latency_ms: float,
    usage: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, str]]] = None,
    artifact_dir: Optional[Path] = None,
) -> None:
    """Print rich-formatted output with panels."""
    # Build subtitle with stats
    latency_formatted = format_time(latency_ms / 1000)
    stats = [model, latency_formatted]
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        if total_tokens:
            stats.append(
                f"P:{prompt_tokens} / C:{completion_tokens} / T:{total_tokens} tokens"
            )
        # Add credits if available
        credits = usage.get("credits_used")
        if credits is not None:
            stats.append(f"{credits} credits")

    subtitle = " · ".join(stats)

    # Main response panel
    console.print(
        Panel(
            Markdown(content),
            title="[bold]Response[/bold]",
            title_align="left",
            subtitle=f"[dim]{subtitle}[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(1, 2),
        )
    )

    if artifacts and artifact_dir:
        console.print(f"  [dim]Artifacts:[/dim] {artifact_dir}")


@app.command()
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
    """Process images, videos, and documents with natural language."""
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
        console.print('  vlmrun chat "Your prompt here" -i file.jpg')
        console.print("  vlmrun chat -p prompt.txt -i file.jpg")
        console.print('  echo "Your prompt" | vlmrun chat - -i file.jpg')
        console.print("\nRun [green]vlmrun chat --help[/] for more options.")
        raise typer.Exit(1)

    # Validate model
    if model not in AVAILABLE_MODELS:
        console.print(f"[red]Error:[/] Invalid model '{model}'")
        console.print("\nAvailable models:")
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
                console.print(
                    f"\nSupported types: {', '.join(SUPPORTED_INPUT_FILETYPES)}"
                )
                raise typer.Exit(1)

    try:
        # Upload input files concurrently if provided
        file_responses: List[FileResponse] = []
        if input_files:
            file_responses = upload_files_concurrent(
                client, input_files, show_progress=not output_json
            )

            if not output_json:
                # Create tree view of uploaded files
                tree = Tree("", guide_style="dim", hide_root=True)
                for file_path in input_files:
                    size_str = format_file_size(file_path.stat().st_size)
                    tree.add(f"{file_path.name} [dim]({size_str})[/dim]")
                console.print(
                    Panel(
                        tree,
                        title=f"Uploaded {len(file_responses)} file(s)",
                        title_align="left",
                        border_style="dim",
                    )
                )

        # Build messages for the chat completion
        messages = build_messages(
            final_prompt, file_responses if file_responses else None
        )

        # Call the OpenAI-compatible chat completions API
        response_content = ""
        usage_data: Optional[Dict[str, Any]] = None
        response_id: Optional[str] = None

        start_time = time.time()

        if no_stream:
            # Non-streaming mode
            if not output_json:
                with TimedStatus(
                    f"Processing ([bold]{model}[/bold])...", console=console, spinner="dots"
                ):
                    response = client.agent.completions.create(
                        model=model,
                        messages=messages,
                        stream=False,
                    )
            else:
                console.print(f"Processing ([bold]{model}[/bold])...")
                response = client.agent.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                )

            latency_ms = (time.time() - start_time) * 1000
            response_content = response.choices[0].message.content or ""
            response_id = response.session_id

            if response.usage:
                usage_data = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                # Add credits if available (may be in custom fields)
                if (
                    hasattr(response.usage, "credits_used")
                    and response.usage.credits_used is not None
                ):
                    usage_data["credits_used"] = response.usage.credits_used
                elif (
                    hasattr(response.usage, "credits")
                    and response.usage.credits is not None
                ):
                    usage_data["credits_used"] = response.usage.credits

            if output_json:
                output = {
                    "id": response.id,
                    "session_id": response.session_id,
                    "model": response.model,
                    "content": response_content,
                    "latency_ms": latency_ms,
                    "usage": usage_data,
                }
                console.print_json(json.dumps(output, indent=2, default=str))
            else:
                print_rich_output(response_content, model, latency_ms, usage_data)

        else:
            # Streaming mode
            with TimedStatus(f"Processing ([bold]{model}[/bold])...", console=console, spinner="dots"):
                stream = client.agent.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )

                # Collect streaming content
                chunks = []
                for chunk in stream:
                    # Capture session_id from first chunk
                    if not response_id and hasattr(chunk, "session_id"):
                        response_id = chunk.session_id
                    if (
                        chunk.choices
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content
                    ):
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
            else:
                print_rich_output(response_content, model, latency_ms)

        # Extract and download artifacts if present
        if not no_download:
            artifact_refs = extract_artifact_refs(response_content)
            if artifact_refs:
                # Use response_id as session_id if available
                if not response_id:
                    console.print(
                        "[yellow]Warning:[/] No session_id available, artifacts may not download correctly"
                    )
                    session_id = "unknown"
                else:
                    session_id = response_id

                # Set up output directory
                if output_dir:
                    artifact_dir = output_dir
                else:
                    artifact_dir = VLMRUN_ARTIFACT_CACHE_DIR / session_id
                artifact_dir.mkdir(parents=True, exist_ok=True)

                downloaded_files = []

                # Download artifacts with status spinner
                if not output_json:
                    with Status(
                        "Downloading artifacts...", console=console, spinner="dots"
                    ):
                        for ref_id in artifact_refs:
                            try:
                                output_path = download_artifact(
                                    client,
                                    session_id,
                                    ref_id,
                                    artifact_dir,
                                )
                                downloaded_files.append(output_path)
                            except Exception as e:
                                console.print(
                                    f"[red]Failed to download {ref_id}: {e}[/]"
                                )
                else:
                    # JSON output mode - download without progress
                    for ref_id in artifact_refs:
                        try:
                            output_path = download_artifact(
                                client,
                                session_id,
                                ref_id,
                                artifact_dir,
                            )
                            downloaded_files.append(output_path)
                        except Exception as e:
                            console.print(f"[red]Failed to download {ref_id}: {e}[/]")

                if not output_json and downloaded_files:
                    # Create elegant tree view of artifacts
                    tree = Tree(f"{artifact_dir}", guide_style="dim")
                    for file_path in sorted(downloaded_files):
                        # Get file size
                        size_str = format_file_size(file_path.stat().st_size)
                        tree.add(f"{file_path.name} [dim]({size_str})[/dim]")

                    console.print(
                        Panel(
                            tree,
                            title=f"Downloaded {len(downloaded_files)} artifact(s)",
                            title_align="left",
                            border_style="dim",
                        )
                    )

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
