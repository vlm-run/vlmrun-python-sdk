"""Gateway commands for the VLM Run CLI.

Talk to OpenAI-compatible OCR / VLM models hosted behind the VLM Run gateway
(``https://gateway.vlm.run/v1``), authenticating with the same
``VLMRUN_API_KEY`` used everywhere else.

Unlike ``vlmrun chat`` (which uploads to the Files API and calls the Orion
agent), the gateway is a raw passthrough to third-party models. Documents and
images are therefore inlined as base64 ``data:`` URLs in standard OpenAI
``image_url`` content parts, and most models (especially OCR models such as
``glm-ocr`` and ``paddle-ocrv6``) do not accept text-only input.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

from vlmrun.client import VLMRun
from vlmrun.cli._cli.chat import (
    TimedStatus,
    format_file_size,
    handle_api_errors,
)

console = Console()

CHAT_HELP = """Run OCR / VLM models on the VLM Run gateway.

\b
EXAMPLES:
  vlmrun gw chat doc.pdf -m glm-ocr
  vlmrun gw chat a.pdf b.pdf -m paddle-ocrv6
  vlmrun gw chat img.jpg -m paddle-ocrv6
  vlmrun gw chat img.jpg -p "describe this image" -m qwen3.6-0.8b
  vlmrun gw chat doc.pdf -m glm-ocr -e temperature=0 -e max_tokens=4096

\b
NOTES:
  Most gateway models (e.g. OCR models) require at least one input file and do
  not accept text-only prompts. Use -p only for models that support it.
"""

app = typer.Typer(
    help="Run OCR / VLM models on the OpenAI-compatible VLM Run gateway.",
    add_completion=False,
    no_args_is_help=True,
)


def _guess_mime(path: Path) -> str:
    """Best-effort MIME type for a local file."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _encode_file_part(path: Path) -> Dict[str, Any]:
    """Encode a local file as an OpenAI ``image_url`` data-URL content part.

    The gateway accepts documents and images inline as base64 ``data:`` URLs.
    """
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime = _guess_mime(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def _build_messages(files: List[Path], prompt: Optional[str]) -> List[Dict[str, Any]]:
    """Build a single OpenAI-style user message from files + optional prompt."""
    content: List[Dict[str, Any]] = [_encode_file_part(f) for f in files]
    if prompt:
        content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _parse_extra(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """Parse repeatable ``key=value`` options into create() kwargs.

    Values are parsed as JSON when possible (so ``temperature=0.2`` becomes a
    float and ``stop=["\\n"]`` becomes a list), else kept as strings.
    """
    extra: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            console.print(
                f"[red]Error:[/] Invalid --extra value '{pair}'. Use key=value."
            )
            raise typer.Exit(1)
        key, _, raw = pair.partition("=")
        key = key.strip()
        try:
            value: Any = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            value = raw
        extra[key] = value
    return extra


def _extract_pricing(model: Dict[str, Any]) -> tuple[str, str]:
    """Best-effort extraction of input/output price ($ per 1M tokens).

    Gateway model objects carry pricing metadata as extra fields. Field names
    vary, so this checks the common shapes and normalizes to per-1M-token USD.
    """

    def _fmt(value: Any, per_million: bool) -> Optional[str]:
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if not per_million:
            num *= 1_000_000
        return f"${num:.4f}".rstrip("0").rstrip(".")

    pricing = model.get("pricing") if isinstance(model.get("pricing"), dict) else {}

    # (candidate keys, whether the value is already per-1M-tokens)
    input_candidates = [
        (pricing.get("input"), True),
        (pricing.get("prompt"), True),
        (model.get("input_price_per_1m"), True),
        (model.get("input_cost_per_1m_tokens"), True),
        (model.get("input_price"), True),
        (model.get("input_cost_per_token"), False),
        (model.get("prompt_cost_per_token"), False),
    ]
    output_candidates = [
        (pricing.get("output"), True),
        (pricing.get("completion"), True),
        (model.get("output_price_per_1m"), True),
        (model.get("output_cost_per_1m_tokens"), True),
        (model.get("output_price"), True),
        (model.get("output_cost_per_token"), False),
        (model.get("completion_cost_per_token"), False),
    ]

    def _first(candidates: List[tuple[Any, bool]]) -> str:
        for value, per_million in candidates:
            formatted = _fmt(value, per_million)
            if formatted is not None:
                return formatted
        return "-"

    return _first(input_candidates), _first(output_candidates)


@app.command()
def health(ctx: typer.Context) -> None:
    """Check gateway health."""
    client: VLMRun = ctx.obj
    with TimedStatus("Checking gateway...", console=console):
        ok = client.gateway.health()

    if ok:
        console.print(
            Panel(
                f"[green]Gateway is healthy[/green]\n[dim]{client.gateway.base_url}[/dim]",
                title="[green]OK[/green]",
                title_align="left",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[red]Gateway is unreachable[/red]\n[dim]{client.gateway.base_url}[/dim]",
                title="[red]Unhealthy[/red]",
                title_align="left",
                border_style="red",
            )
        )
        raise typer.Exit(1)


@app.command()
def models(
    ctx: typer.Context,
    output_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON."),
) -> None:
    """List models available on the gateway with pricing info."""
    client: VLMRun = ctx.obj

    with handle_api_errors():
        model_objs = client.gateway.models()

    # Normalize to dicts (OpenAI Model objects are pydantic models).
    rows: List[Dict[str, Any]] = []
    for m in model_objs:
        if hasattr(m, "model_dump"):
            rows.append(m.model_dump())
        elif isinstance(m, dict):
            rows.append(m)
        else:
            rows.append({"id": str(m)})

    if output_json:
        print(json.dumps(rows, indent=2, default=str))
        return

    table = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("MODEL", style="bold cyan")
    table.add_column("OWNED BY", style="dim")
    table.add_column("INPUT $/1M", justify="right")
    table.add_column("OUTPUT $/1M", justify="right")

    for row in sorted(rows, key=lambda r: str(r.get("id", ""))):
        input_price, output_price = _extract_pricing(row)
        table.add_row(
            str(row.get("id", "-")),
            str(row.get("owned_by", "-")),
            input_price,
            output_price,
        )

    console.print(
        Panel(
            table,
            title="[bold]Gateway Models[/bold]",
            title_align="left",
            subtitle=f"[dim]{len(rows)} model(s)[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 1),
        )
    )


@app.command(help=CHAT_HELP, context_settings={"max_content_width": 120})
def chat(
    ctx: typer.Context,
    files: List[Path] = typer.Argument(
        None,
        help="Input document/image file(s) to process. Repeatable.",
        exists=True,
        readable=True,
    ),
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Gateway model id (e.g. glm-ocr, paddle-ocrv6, qwen3.6-0.8b).",
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Optional text prompt (only for models that support text input).",
    ),
    extra: Optional[List[str]] = typer.Option(
        None,
        "--extra",
        "-e",
        help="Extra create() kwarg as key=value (repeatable), e.g. -e temperature=0.",
    ),
    no_stream: bool = typer.Option(
        False, "--no-stream", "-ns", help="Disable streaming."
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON."),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", help="Request timeout in seconds."
    ),
) -> None:
    """Run a gateway model over one or more documents/images."""
    client: VLMRun = ctx.obj

    if not files and not prompt:
        console.print(
            "[red]Error:[/] Provide at least one input file. "
            "Most gateway models do not accept text-only input."
        )
        raise typer.Exit(1)

    files = files or []
    create_kwargs = _parse_extra(extra)
    if timeout is not None:
        create_kwargs["timeout"] = timeout

    # Show the files being processed.
    if files and not output_json:
        tree = Tree("", guide_style="dim", hide_root=True)
        for f in files:
            size_str = format_file_size(f.stat().st_size)
            tree.add(f"{f.name} [dim]({size_str})[/dim]")
        console.print(
            Panel(
                tree,
                title=f"Processing {len(files)} file(s) [dim]({model})[/dim]",
                title_align="left",
                border_style="dim",
            )
        )

    messages = _build_messages(files, prompt)
    start_time = time.time()
    status_msg = f"Processing ([bold]{model}[/bold])..."

    if no_stream:
        if output_json:
            with handle_api_errors():
                response = client.gateway.completions.create(
                    model=model, messages=messages, stream=False, **create_kwargs
                )
        else:
            with (
                TimedStatus(status_msg, console=console),
                handle_api_errors(),
            ):
                response = client.gateway.completions.create(
                    model=model, messages=messages, stream=False, **create_kwargs
                )
        latency_s = time.time() - start_time
        content = response.choices[0].message.content or ""
        usage = response.usage
    else:
        chunks: List[str] = []
        usage = None

        def _consume(stream) -> None:
            nonlocal usage
            for chunk in stream:
                if (
                    chunk.choices
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.content
                ):
                    chunks.append(chunk.choices[0].delta.content)
                if getattr(chunk, "usage", None):
                    usage = chunk.usage

        if output_json:
            with handle_api_errors():
                _consume(
                    client.gateway.completions.create(
                        model=model, messages=messages, stream=True, **create_kwargs
                    )
                )
        else:
            with (
                TimedStatus(status_msg, console=console),
                handle_api_errors(),
            ):
                _consume(
                    client.gateway.completions.create(
                        model=model, messages=messages, stream=True, **create_kwargs
                    )
                )
        content = "".join(chunks)
        latency_s = time.time() - start_time

    if output_json:
        out = {
            "model": model,
            "content": content,
            "latency_s": latency_s,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
        }
        print(json.dumps(out, indent=2, default=str))
        return

    _print_output(content, model, latency_s, usage)


def _print_output(content: str, model: str, latency_s: float, usage: Any) -> None:
    """Render the gateway response in a Rich panel."""
    stats = [model]
    if usage is not None:
        total = getattr(usage, "total_tokens", None)
        if total:
            prompt_toks = getattr(usage, "prompt_tokens", 0)
            completion_toks = getattr(usage, "completion_tokens", 0)
            stats.append(f"P:{prompt_toks} / C:{completion_toks} / T:{total} tokens")
    stats.append(f"{latency_s:.2f}s")

    console.print(
        Panel(
            Markdown(content) if content else "[dim](empty response)[/dim]",
            title="[bold]Response[/bold]",
            title_align="left",
            subtitle=f"[dim][white]{' · '.join(stats)}[/white][/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    app()
