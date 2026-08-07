"""Gateway commands for the VLM Run CLI.

Talk to OpenAI-compatible OCR / VLM models hosted behind the VLM Run gateway
(``https://gateway.vlm.run/v1``), authenticating with the same
``VLMRUN_API_KEY`` used everywhere else.

Unlike ``vlmrun chat`` (which uploads to the Files API and calls the Orion
agent), the gateway is a raw passthrough to third-party models. Inputs are
inlined as base64 ``data:`` URLs in the message content: documents use
``document_url`` content parts, images use ``image_url``, and ``file_url`` is
the fallback for anything unidentifiable. Most models (especially OCR models
such as ``zai-org/glm-ocr`` and ``paddleocr/pp-ocrv6``) do not accept
text-only input.

Commands: ``health``, ``models`` (list or detail one model), ``chat``,
``embed`` (embeddings) and ``transcribe`` (audio transcriptions).
"""

from __future__ import annotations

import base64
import inspect
import json
import mimetypes
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich import box

from vlmrun.client import VLMRun
from vlmrun.client.gateway import _require_openai
from vlmrun.cli._cli.chat import (
    TimedStatus,
    format_file_size,
    handle_api_errors,
)
from vlmrun.constants import (
    SUPPORTED_DOCUMENT_FILETYPES,
    SUPPORTED_VIDEO_FILETYPES,
)

console = Console()

CHAT_HELP = """Run OCR / VLM models on the VLM Run gateway.

\b
EXAMPLES:
  vlmrun gw chat doc.pdf -m zai-org/glm-ocr
  vlmrun gw chat a.pdf b.pdf -m paddleocr/pp-ocrv6
  vlmrun gw chat img.jpg -m paddleocr/pp-ocrv6
  vlmrun gw chat img.jpg -p "describe this image" -m qwen/qwen3.5-0.8b
  vlmrun gw chat https://example.com/scan.jpg -m paddleocr/pp-ocrv6
  vlmrun gw chat https://example.com/report.pdf -m zai-org/glm-ocr
  vlmrun gw chat doc.pdf -m zai-org/glm-ocr -e temperature=0 -e max_tokens=4096

\b
METHODS:
  Each model exposes methods with a default. Run `vlmrun gw models <model>` for
  its methods, params, and copy-pasteable example commands.
  vlmrun gw chat img.jpg -m paddleocr/pp-ocrv6 --method detect
  vlmrun gw chat img.jpg -m paddleocr/pp-ocrv6 --method ocr \\
      --method-params '{"lang": "en", "score_threshold": 0.5}'
  vlmrun gw chat img.jpg -m paddleocr/pp-ocrv6 --json-mode

\b
NOTES:
  Model ids are the full `<org>/<name>` shown by `vlmrun gw models`; short
  aliases (e.g. `glm-ocr`) also work.
  Inputs are local file paths or http(s) URLs (image, document or video).
  Most gateway models (e.g. OCR models) require at least one input and do not
  accept text-only prompts. Use -p only for models that support it.
"""

GATEWAY_HELP = """OCR, VLM, embedding and transcription models on the VLM Run gateway.

An OpenAI-compatible passthrough to third-party models, authenticated with the
same VLMRUN_API_KEY as the rest of the CLI. `vlmrun gateway` and `vlmrun gw`
are the same command.

\b
Start here:
  vlmrun gw models              See what is available (task + methods per model)
  vlmrun gw models <model>      Methods, params and copy-pasteable examples
"""

app = typer.Typer(
    help=GATEWAY_HELP,
    add_completion=False,
    no_args_is_help=True,
)


# Magic-byte signatures, checked before the filename extension. Extensions lie
# (a .jpg that is really WebP is common), and the gateway trusts the media type
# we declare in the data URL, so a wrong one makes it misroute the file.
_MAGIC_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"%PDF", "application/pdf"),
)


def _sniff_mime(data: bytes) -> Optional[str]:
    """MIME type from a file's magic bytes, or None if unrecognized."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "video/x-msvideo"
    if data[4:8] == b"ftyp":
        return "video/mp4"
    for signature, mime in _MAGIC_SIGNATURES:
        if data.startswith(signature):
            return mime
    return None


def _guess_mime(path: Path, data: Optional[bytes] = None) -> str:
    """Best-effort MIME type for a local file, preferring its actual content."""
    if data is None:
        try:
            with path.open("rb") as fh:
                data = fh.read(16)
        except OSError:
            data = b""
    sniffed = _sniff_mime(data)
    if sniffed:
        return sniffed
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _is_http_url(value: str) -> bool:
    """Return True if ``value`` looks like an http(s) URL."""
    return value.startswith(("http://", "https://"))


def _suffix_from_url(url: str) -> str:
    """File extension from a URL path, ignoring query strings."""
    return Path(urlparse(url).path).suffix.lower()


def _content_part_type_for_suffix(suffix: str, mime: Optional[str] = None) -> str:
    """Content-part type from a file extension and optional MIME type."""
    if suffix in SUPPORTED_DOCUMENT_FILETYPES:
        return "document_url"
    mime = mime or mimetypes.guess_type(f"name{suffix}")[0] or ""
    if mime == "application/pdf":
        return "document_url"
    if mime.startswith("video/") or suffix in SUPPORTED_VIDEO_FILETYPES:
        return "video_url"
    if mime.startswith("image/"):
        return "image_url"
    return "file_url"


def _content_part_type(path: Path, mime: Optional[str] = None) -> str:
    """Content-part type for a file.

    Documents (``.pdf``, ``.doc``, ``.docx``) are sent as ``document_url`` and
    images as ``image_url``. ``file_url`` is the fallback for anything we cannot
    identify: the gateway routes it through its document/PDF path, which fails
    outright on a plain image.
    """
    return _content_part_type_for_suffix(path.suffix.lower(), mime or _guess_mime(path))


def _content_part_type_from_url(url: str) -> str:
    """Content-part type for a remote http(s) URL."""
    suffix = _suffix_from_url(url)
    mime, _ = mimetypes.guess_type(urlparse(url).path)
    return _content_part_type_for_suffix(suffix, mime)


def _encode_file_part(path: Path) -> Dict[str, Any]:
    """Encode a local file as a gateway data-URL content part.

    The gateway accepts files inline as base64 ``data:`` URLs under a
    ``document_url`` (documents), ``image_url`` (images) or ``file_url``
    (anything else) content part.
    """
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime = _guess_mime(path, data)
    key = _content_part_type(path, mime)
    return {
        "type": key,
        key: {"url": f"data:{mime};base64,{b64}"},
    }


def _encode_url_part(url: str) -> Dict[str, Any]:
    """Build a gateway content part that references a remote http(s) URL."""
    key = _content_part_type_from_url(url)
    return {
        "type": key,
        key: {"url": url},
    }


def _encode_chat_input(raw: str) -> Dict[str, Any]:
    """Encode one chat input — a local file path or http(s) URL."""
    if _is_http_url(raw):
        return _encode_url_part(raw)
    path = Path(raw).expanduser()
    return _encode_file_part(path)


def _validate_chat_input(raw: str) -> None:
    """Ensure a non-URL chat input refers to a readable local file."""
    if _is_http_url(raw):
        return
    path = Path(raw).expanduser()
    if not path.is_file():
        console.print(
            f"[red]Error:[/] Input '{raw}' is not a file. "
            "Provide a local path or an http(s) URL."
        )
        raise typer.Exit(1)


def _parse_response_format(value: str) -> Dict[str, Any]:
    """Parse ``--response-format`` into an OpenAI ``response_format`` object.

    Accepts the shorthands ``text`` and ``json_object`` (with ``json`` as an
    alias), or a full JSON object for advanced cases (e.g. ``json_schema``).
    Exits with a clear message on anything else.
    """
    stripped = value.strip()
    aliases = {
        "text": {"type": "text"},
        "json": {"type": "json_object"},
        "json_object": {"type": "json_object"},
    }
    if stripped in aliases:
        return aliases[stripped]
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError) as e:
            console.print(f"[red]Error:[/] --response-format must be valid JSON: {e}")
            raise typer.Exit(1)
        if not isinstance(parsed, dict) or "type" not in parsed:
            console.print(
                "[red]Error:[/] --response-format JSON must be an object with a "
                "'type' key, e.g. '{\"type\":\"json_object\"}'."
            )
            raise typer.Exit(1)
        return parsed
    console.print(
        f"[red]Error:[/] Unknown --response-format '{value}'. Use 'text', "
        "'json_object', or a JSON object with a 'type' key."
    )
    raise typer.Exit(1)


def _build_messages(inputs: List[str], prompt: Optional[str]) -> List[Dict[str, Any]]:
    """Build a single OpenAI-style user message from file paths/URLs + prompt."""
    content: List[Dict[str, Any]] = [_encode_chat_input(raw) for raw in inputs]
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


@lru_cache(maxsize=1)
def _openai_create_params() -> frozenset:
    """Parameter names accepted by the OpenAI SDK's ``chat.completions.create()``.

    Introspected rather than hardcoded so the split below tracks whatever
    version of the ``openai`` package is installed.
    """
    # Route a missing dependency through the SDK's DependencyError (with install
    # hints) instead of surfacing a raw ImportError. Reuses _require_openai so
    # the install message lives in one place.
    _require_openai()
    from openai.resources.chat.completions import Completions

    sig = inspect.signature(Completions.create)
    names = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }
    return frozenset(names - {"self"})


def _split_create_kwargs(
    extra: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split user-supplied kwargs into OpenAI create() kwargs and extra_body.

    ``create()`` has an explicit signature and rejects unknown keywords, so
    gateway-specific fields (``method``, ``document_dpi``, ...) must travel in
    ``extra_body`` to reach the server as top-level request-body fields.
    """
    known = _openai_create_params()
    kwargs: Dict[str, Any] = {}
    body: Dict[str, Any] = {}
    for key, value in extra.items():
        if key in known:
            kwargs[key] = value
        else:
            body[key] = value

    # An explicit -e extra_body={...} merges with the routed fields.
    explicit = kwargs.pop("extra_body", None)
    if isinstance(explicit, dict):
        body = {**explicit, **body}
    return kwargs, body


def _format_methods(model: Dict[str, Any]) -> str:
    """Render a model's methods, marking the default with ``*``."""
    methods = model.get("methods") or []
    default = model.get("default_method") or ""
    if not methods:
        return "-"
    return ", ".join(f"[bold]{m}[/bold]*" if m == default else m for m in methods)


def _format_inputs(model: Dict[str, Any]) -> str:
    """Render the input types a model accepts, minus the ``_url`` noise."""
    caps = model.get("capabilities") or {}
    types = caps.get("supported_input_types") or []
    if not types:
        return "-"
    return ", ".join(t.removesuffix("_url") for t in types)


def _model_dicts(client: VLMRun) -> List[Dict[str, Any]]:
    """Fetch gateway models and normalize them to plain dicts."""
    with handle_api_errors():
        model_objs = client.gateway.models()

    rows: List[Dict[str, Any]] = []
    for m in model_objs:
        if hasattr(m, "model_dump"):
            rows.append(m.model_dump())
        elif isinstance(m, dict):
            rows.append(m)
        else:
            rows.append({"id": str(m)})
    return sorted(rows, key=lambda r: str(r.get("id", "")))


def _parse_extra_body_help(help_text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Split a model's ``extra_body_help`` into JSON examples and prose notes.

    The gateway packs this field with ``|``-separated fragments, some of which
    are JSON extra_body payloads (e.g. ``{"method":"ocr"}``) and some of which
    are free-text hints. Parsing it keeps the examples we print in sync with
    whatever the gateway currently advertises.
    """
    examples: List[Dict[str, Any]] = []
    notes: List[str] = []
    for fragment in (help_text or "").split("|"):
        fragment = fragment.strip()
        if not fragment:
            continue
        try:
            parsed = json.loads(fragment)
        except (json.JSONDecodeError, ValueError):
            notes.append(fragment)
            continue
        if isinstance(parsed, dict):
            examples.append(parsed)
        else:
            notes.append(fragment)
    return examples, notes


def _example_command(model_id: str, payload: Dict[str, Any], sample: str) -> str:
    """Render an extra_body example as a runnable `vlmrun gw chat` command."""
    parts = [f"vlmrun gw chat {sample} -m {model_id}"]
    method = payload.get("method")
    if method:
        parts.append(f"--method {method}")
    params = payload.get("method_params")
    if isinstance(params, dict):
        parts.append(f"--method-params '{json.dumps(params)}'")
    for key, value in payload.items():
        if key in ("method", "method_params"):
            continue
        parts.append(f"-e {key}={json.dumps(value)}")
    return " ".join(parts)


def _sample_input(model: Dict[str, Any]) -> str:
    """Pick a plausible sample filename for a model's example commands."""
    caps = model.get("capabilities") or {}
    types = caps.get("supported_input_types") or []
    if "document_url" in types:
        return "doc.pdf"
    if "image_url" in types:
        return "img.jpg"
    if "video_url" in types:
        return "clip.mp4"
    return "input.bin"


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


MODELS_HELP = """List gateway models, or detail one model.

\b
EXAMPLES:
  vlmrun gw models                        List every model with its methods.
  vlmrun gw models paddleocr/pp-ocrv6     Methods, params and examples for one model.
  vlmrun gw models --json                 Raw model catalog.
"""


def _model_detail(row: Dict[str, Any]) -> Panel:
    """Render one model's methods, params, notes and example commands."""
    model_id = str(row.get("id", "-"))
    examples, notes = _parse_extra_body_help(row.get("extra_body_help", ""))
    sample = _sample_input(row)

    tree = Tree("", guide_style="dim", hide_root=True)
    tree.add(f"[dim]task[/dim]      {row.get('task', '-')}")
    tree.add(f"[dim]methods[/dim]   {_format_methods(row)}")
    tree.add(f"[dim]inputs[/dim]    {_format_inputs(row)}")
    aliases = row.get("aliases") or []
    if aliases:
        tree.add(f"[dim]aliases[/dim]   {', '.join(aliases)}")
    for note in notes:
        tree.add(f"[dim]note[/dim]      {note}")
    if examples:
        branch = tree.add("[dim]examples[/dim]")
        for ex in examples:
            branch.add(Text(_example_command(model_id, ex, sample), style="cyan"))

    return Panel(
        tree,
        title=f"[bold cyan]{model_id}[/bold cyan]",
        title_align="left",
        border_style="blue",
        padding=(0, 1),
    )


def _model_detail_json(row: Dict[str, Any]) -> Dict[str, Any]:
    """JSON form of a model's detail view, including runnable commands."""
    examples, notes = _parse_extra_body_help(row.get("extra_body_help", ""))
    sample = _sample_input(row)
    return {
        "id": row.get("id"),
        "aliases": row.get("aliases") or [],
        "task": row.get("task"),
        "methods": row.get("methods") or [],
        "default_method": row.get("default_method") or None,
        "supported_input_types": (row.get("capabilities") or {}).get(
            "supported_input_types"
        )
        or [],
        "extra_body_examples": examples,
        "notes": notes,
        "commands": [
            _example_command(str(row.get("id")), ex, sample) for ex in examples
        ],
    }


@app.command(help=MODELS_HELP, context_settings={"max_content_width": 120})
def models(
    ctx: typer.Context,
    model: Optional[str] = typer.Argument(
        None,
        help="Model id or alias. Shows that model's methods, params and examples.",
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON."),
) -> None:
    """List gateway models, or detail one model."""
    client: VLMRun = ctx.obj
    rows = _model_dicts(client)

    if model:
        wanted = model.strip()
        match = [
            r
            for r in rows
            if wanted == str(r.get("id")) or wanted in (r.get("aliases") or [])
        ]
        if not match:
            console.print(
                f"[red]Error:[/] Model '{model}' not found on the gateway. "
                "Run `vlmrun gw models` to list available models."
            )
            raise typer.Exit(1)
        if output_json:
            print(json.dumps(_model_detail_json(match[0]), indent=2, default=str))
            return
        console.print(_model_detail(match[0]))
        return

    if output_json:
        print(json.dumps(rows, indent=2, default=str))
        return

    table = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
    )
    table.add_column("MODEL", style="bold cyan", no_wrap=True)
    table.add_column("TASK", style="dim", no_wrap=True)
    table.add_column("METHODS")

    for row in rows:
        # Aliases are omitted here to keep method names from truncating at 80
        # columns; the per-model detail view lists them.
        table.add_row(
            str(row.get("id", "-")),
            str(row.get("task", "-")),
            _format_methods(row),
        )

    console.print(
        Panel(
            table,
            title="[bold]Gateway Models[/bold]",
            title_align="left",
            subtitle=f"[dim]{len(rows)} model(s) · [bold]*[/bold] = default method · `vlmrun gw models <model>` for examples[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 1),
        )
    )


@app.command(help=CHAT_HELP, context_settings={"max_content_width": 120})
def chat(
    ctx: typer.Context,
    inputs: List[str] = typer.Argument(
        None,
        help="Input file path(s) or http(s) URL(s) (image, document or video). Repeatable.",
    ),
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Gateway model id, full <org>/<name> or alias (see `vlmrun gw models`).",
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Optional text prompt (only for models that support text input).",
    ),
    method: Optional[str] = typer.Option(
        None,
        "--method",
        "-M",
        help="Model method, e.g. ocr, detect, markdown. Defaults to the model's default_method.",
    ),
    method_params: Optional[str] = typer.Option(
        None,
        "--method-params",
        help='JSON object of method arguments, e.g. \'{"lang": "en"}\'.',
    ),
    json_mode: bool = typer.Option(
        False,
        "--json-mode",
        help=(
            "Enable JSON mode (response_format json_object). Mutually exclusive "
            "with --response-format."
        ),
    ),
    response_format: Optional[str] = typer.Option(
        None,
        "--response-format",
        help=(
            "Ask the MODEL to constrain its output: 'text', 'json_object' (JSON "
            'mode), or a JSON object like \'{"type":"json_schema",...}\'. '
            "Sent to the gateway as `response_format`; not yet honored server-side. "
            "(Distinct from --json, which formats the CLI's own output. Use "
            "--json-mode as a shorthand for json_object.)"
        ),
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

    inputs = list(inputs or [])
    if not inputs and not prompt:
        console.print(
            "[red]Error:[/] Provide at least one input file or URL. "
            "Most gateway models do not accept text-only input."
        )
        raise typer.Exit(1)

    for raw in inputs:
        _validate_chat_input(raw)

    create_kwargs, extra_body = _split_create_kwargs(_parse_extra(extra))
    if timeout is not None:
        create_kwargs["timeout"] = timeout

    if method:
        extra_body["method"] = method
    if method_params:
        try:
            parsed_params = json.loads(method_params)
        except (json.JSONDecodeError, ValueError) as e:
            console.print(f"[red]Error:[/] --method-params must be valid JSON: {e}")
            raise typer.Exit(1)
        if not isinstance(parsed_params, dict):
            console.print("[red]Error:[/] --method-params must be a JSON object.")
            raise typer.Exit(1)
        extra_body["method_params"] = parsed_params

    if json_mode and response_format:
        console.print(
            "[red]Error:[/] --json-mode and --response-format are mutually "
            "exclusive. Use one or the other."
        )
        raise typer.Exit(1)

    if json_mode:
        create_kwargs["response_format"] = {"type": "json_object"}
    elif response_format:
        # A standard OpenAI create() field, so it rides as a top-level kwarg.
        create_kwargs["response_format"] = _parse_response_format(response_format)

    if extra_body:
        create_kwargs["extra_body"] = extra_body

    # Show the inputs being processed.
    if inputs and not output_json:
        tree = Tree("", guide_style="dim", hide_root=True)
        for raw in inputs:
            if _is_http_url(raw):
                tree.add(raw)
            else:
                path = Path(raw).expanduser()
                size_str = format_file_size(path.stat().st_size)
                tree.add(f"{path.name} [dim]({size_str})[/dim]")
        console.print(
            Panel(
                tree,
                title=f"Processing {len(inputs)} input(s) [dim]({model})[/dim]",
                title_align="left",
                border_style="dim",
            )
        )

    messages = _build_messages(inputs, prompt)
    start_time = time.time()
    status_msg = f"Processing ([bold]{model}[/bold])..."

    # The gateway only streams document (PDF) requests: it emits one SSE chunk
    # per page. Image- and text-only requests ignore `stream` and return a
    # single plain chat.completion body still labelled text/event-stream, so an
    # SSE reader finds no events and yields empty content. Stream only when a
    # document is actually present.
    has_document = any(
        part.get("type") == "document_url"
        for message in messages
        for part in message["content"]
    )
    if not has_document or no_stream:
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

    error = _content_error(content)

    if output_json:
        out = {
            "model": model,
            "content": content,
            "latency_s": latency_s,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
        }
        print(json.dumps(out, indent=2, default=str))
        raise typer.Exit(1 if error else 0)

    if error:
        console.print(f"[red]Error:[/] {error}")
        raise typer.Exit(1)

    _print_output(content, model, latency_s, usage)


EMBED_HELP = """Embed text, images or video with a gateway embedding model.

\b
EXAMPLES:
  vlmrun gw embed -t "a blue parrot" -m qwen/qwen3-vl-embedding-2b
  vlmrun gw embed photo.jpg -m qwen/qwen3-vl-embedding-2b
  vlmrun gw embed a.jpg b.jpg -t "caption" -m qwen/qwen3-vl-embedding-2b
  vlmrun gw embed photo.jpg -t "caption" --join -m qwen/qwen3-vl-embedding-2b
  vlmrun gw embed -t "hi" -m qwen/qwen3-vl-embedding-2b --dimensions 64
  vlmrun gw embed photo.jpg -m qwen/qwen3-vl-embedding-2b --json  # full vectors

\b
NOTES:
  Every file and every -t/--text is embedded as its own vector. Use --join to
  embed them together as a single vector instead (e.g. an image plus its
  caption); models embed at most one image per vector, so --join accepts at
  most one file.
  Video is accepted by the API but is not currently backed by any embedding
  model: it returns the same vector regardless of the clip.
"""

TRANSCRIBE_HELP = """Transcribe audio with a gateway transcription model.

\b
EXAMPLES:
  vlmrun gw transcribe clip.mp3 -m nvidia/parakeet-tdt-0.6b-v3
  vlmrun gw transcribe clip.mp4 -m nvidia/parakeet-tdt-0.6b-v3   # video's audio track
  vlmrun gw transcribe clip.mp3 -m nvidia/parakeet-tdt-0.6b-v3 -f srt
  vlmrun gw transcribe clip.mp3 -m nvidia/parakeet-tdt-0.6b-v3 --language en
  vlmrun gw transcribe --url https://example.com/a.mp3 -m nvidia/parakeet-tdt-0.6b-v3

\b
NOTES:
  Accepts audio files, or a video file whose audio track is transcribed.
  Formats: json, text, verbose_json, srt, vtt.
"""

TRANSCRIBE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")


def _embed_part(path: Path) -> Dict[str, Any]:
    """Encode a local file as an embedding content part.

    Embedding models take images and video only; anything else (a PDF, a text
    file) is rejected here rather than sent as a mislabelled ``image_url`` that
    the gateway would fail on.
    """
    data = path.read_bytes()
    mime = _guess_mime(path, data)
    if mime.startswith("video/"):
        key = "video_url"
    elif mime.startswith("image/"):
        key = "image_url"
    else:
        console.print(
            f"[red]Error:[/] Cannot embed '{path.name}': unsupported type "
            f"'{mime}'. Embedding models accept images and video only "
            "(use --text for text)."
        )
        raise typer.Exit(1)
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": key, key: {"url": f"data:{mime};base64,{b64}"}}


def _content_error(content: str) -> Optional[str]:
    """Return the error message if the response body is a gateway error payload.

    On some 200 responses (e.g. an unknown ``--method``) the gateway ships an
    ``{"error": "..."}`` object as the message content instead of a real
    result. Detect that exact shape — a lone ``error`` key — so the CLI can fail
    loudly instead of rendering it as a successful response. Normal outputs are
    either ``<document>``-wrapped or ``{"text": ...}`` lines, so this never
    misfires.
    """
    stripped = content.strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict) and list(parsed.keys()) == ["error"]:
        return str(parsed["error"])
    return None


def _renderable(content: str):
    """Pick a Rich renderable for gateway output.

    OCR responses are wrapped in ``<document>``/``<page>`` tags, and detect/ocr
    methods emit JSON lines. Rich's Markdown renderer treats angle-bracket tags
    as HTML and drops them, blanking the output, so only render as Markdown when
    the payload does not start with markup or JSON.
    """
    stripped = content.lstrip()
    if stripped.startswith(("<", "{", "[")):
        return Text(content)
    return Markdown(content)


def _format_cost(cost: Any) -> Optional[str]:
    """Format the gateway's per-request ``usage.cost`` (USD), or None."""
    try:
        value = float(cost)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return "$0"
    if value >= 0.01:
        return f"${value:.4f}"
    # Sub-cent: show up to 6 decimals without scientific notation, but never
    # collapse a real cost to "$0".
    formatted = f"${value:.6f}".rstrip("0").rstrip(".")
    return formatted if formatted != "$0" else "<$0.000001"


def _print_output(content: str, model: str, latency_s: float, usage: Any) -> None:
    """Render the gateway response in a Rich panel."""
    stats = [model]
    if usage is not None:
        total = getattr(usage, "total_tokens", None)
        if total:
            prompt_toks = getattr(usage, "prompt_tokens", 0)
            completion_toks = getattr(usage, "completion_tokens", 0)
            stats.append(f"P:{prompt_toks} / C:{completion_toks} / T:{total} tokens")
        cost = _format_cost(getattr(usage, "cost", None))
        if cost:
            stats.append(cost)
    stats.append(f"{latency_s:.2f}s")

    console.print(
        Panel(
            _renderable(content) if content else "[dim](empty response)[/dim]",
            title="[bold]Response[/bold]",
            title_align="left",
            subtitle=f"[dim][white]{' · '.join(stats)}[/white][/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(1, 2),
        )
    )


@app.command(help=EMBED_HELP, context_settings={"max_content_width": 120})
def embed(
    ctx: typer.Context,
    files: List[Path] = typer.Argument(
        None,
        help="Image/video file(s) to embed. Repeatable.",
        exists=True,
        readable=True,
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Embedding model id (see `vlmrun gw models`)."
    ),
    text: Optional[List[str]] = typer.Option(
        None,
        "--text",
        "-t",
        help="Text to embed (repeatable). Its own vector unless --join is set.",
    ),
    join: bool = typer.Option(
        False,
        "--join",
        help="Embed all inputs together as one vector (max one file).",
    ),
    dimensions: Optional[int] = typer.Option(
        None, "--dimensions", "-d", help="Truncate vectors to this many dimensions."
    ),
    output_json: bool = typer.Option(
        False, "--json", "-j", help="Output raw JSON, including full vectors."
    ),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", help="Request timeout in seconds."
    ),
) -> None:
    """Embed text, images or video with a gateway embedding model."""
    client: VLMRun = ctx.obj
    files = files or []
    texts = list(text or [])

    if not files and not texts:
        console.print("[red]Error:[/] Provide at least one file or --text to embed.")
        raise typer.Exit(1)
    if join and len(files) > 1:
        console.print(
            "[red]Error:[/] --join accepts at most one file: embedding models take "
            "a single image per vector. Drop --join to embed each file separately."
        )
        raise typer.Exit(1)

    # `input` is a list whose items are each either a plain string or a *list*
    # of content parts; a flat list of parts is rejected by the API.
    inputs: List[Any] = []
    labels: List[str] = []
    if join:
        parts = [_embed_part(f) for f in files]
        parts += [{"type": "text", "text": t} for t in texts]
        inputs.append(parts)
        labels.append(" + ".join([f.name for f in files] + [f'"{t}"' for t in texts]))
    else:
        for f in files:
            inputs.append([_embed_part(f)])
            labels.append(f.name)
        for t in texts:
            inputs.append(t)
            labels.append(f'"{t}"')

    create_kwargs: Dict[str, Any] = {}
    if dimensions is not None:
        create_kwargs["dimensions"] = dimensions
    if timeout is not None:
        create_kwargs["timeout"] = timeout

    start_time = time.time()
    if output_json:
        with handle_api_errors():
            response = client.gateway.embeddings.create(
                model=model, input=inputs, **create_kwargs
            )
    else:
        with (
            TimedStatus(f"Embedding ([bold]{model}[/bold])...", console=console),
            handle_api_errors(),
        ):
            response = client.gateway.embeddings.create(
                model=model, input=inputs, **create_kwargs
            )
    latency_s = time.time() - start_time

    if output_json:
        print(
            json.dumps(
                response.model_dump() if hasattr(response, "model_dump") else response,
                indent=2,
                default=str,
            )
        )
        return

    table = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
    )
    table.add_column("INPUT", style="bold cyan")
    table.add_column("DIMS", justify="right")
    table.add_column("PREVIEW", style="dim")

    for i, item in enumerate(response.data):
        vector = item.embedding
        label = labels[i] if i < len(labels) else str(i)
        if isinstance(vector, str):  # encoding_format=base64
            preview, dims = f"{vector[:28]}...", "-"
        else:
            preview = "[" + ", ".join(f"{v:+.3f}" for v in vector[:4]) + ", ...]"
            dims = str(len(vector))
        table.add_row(label if len(label) <= 34 else label[:31] + "...", dims, preview)

    usage = getattr(response, "usage", None)
    stats = [model]
    total = getattr(usage, "total_tokens", None)
    if total:
        stats.append(f"T:{total} tokens")
    stats.append(f"{latency_s:.2f}s")

    console.print(
        Panel(
            table,
            title="[bold]Embeddings[/bold]",
            title_align="left",
            subtitle=f"[dim]{' · '.join(stats)}[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 1),
        )
    )


@app.command(help=TRANSCRIBE_HELP, context_settings={"max_content_width": 120})
def transcribe(
    ctx: typer.Context,
    file: Optional[Path] = typer.Argument(
        None,
        help="Audio file (or a video whose audio track is transcribed).",
        exists=True,
        readable=True,
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Transcription model id (see `vlmrun gw models`)."
    ),
    url: Optional[str] = typer.Option(
        None, "--url", help="Hosted audio URL instead of a local file."
    ),
    response_format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help=f"Response format: {', '.join(TRANSCRIBE_FORMATS)}.",
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l", help="ISO-639-1 language hint, e.g. en."
    ),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p", help="Context to bias transcription (proper nouns)."
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON."),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", help="Request timeout in seconds."
    ),
) -> None:
    """Transcribe audio with a gateway transcription model."""
    client: VLMRun = ctx.obj

    if not file and not url:
        console.print("[red]Error:[/] Provide an audio file or --url.")
        raise typer.Exit(1)
    if file and url:
        console.print("[red]Error:[/] Provide either a file or --url, not both.")
        raise typer.Exit(1)
    if response_format not in TRANSCRIBE_FORMATS:
        console.print(
            f"[red]Error:[/] Unknown --format '{response_format}'. "
            f"Choose from: {', '.join(TRANSCRIBE_FORMATS)}."
        )
        raise typer.Exit(1)

    create_kwargs: Dict[str, Any] = {"response_format": response_format}
    if language:
        create_kwargs["language"] = language
    if prompt:
        create_kwargs["prompt"] = prompt
    if timeout is not None:
        create_kwargs["timeout"] = timeout
    if url:
        # `url` is a gateway extension to the OpenAI transcription form.
        create_kwargs["extra_body"] = {"url": url}

    if file and not output_json:
        console.print(
            Panel(
                f"{file.name} [dim]({format_file_size(file.stat().st_size)})[/dim]",
                title=f"Transcribing [dim]({model})[/dim]",
                title_align="left",
                border_style="dim",
            )
        )

    start_time = time.time()

    def _create():
        if file:
            with file.open("rb") as fh:
                return client.gateway.transcriptions.create(
                    model=model, file=fh, **create_kwargs
                )
        # The OpenAI SDK requires a `file`; the gateway reads `url` instead.
        return client.gateway.transcriptions.create(
            model=model, file=("audio.mp3", b"", "audio/mpeg"), **create_kwargs
        )

    if output_json:
        with handle_api_errors():
            response = _create()
    else:
        with (
            TimedStatus(f"Transcribing ([bold]{model}[/bold])...", console=console),
            handle_api_errors(),
        ):
            response = _create()
    latency_s = time.time() - start_time

    text_out = response if isinstance(response, str) else getattr(response, "text", "")

    if output_json:
        if hasattr(response, "model_dump"):
            print(json.dumps(response.model_dump(), indent=2, default=str))
        else:
            print(
                json.dumps(
                    {"model": model, "text": text_out, "latency_s": latency_s},
                    indent=2,
                    default=str,
                )
            )
        return

    console.print(
        Panel(
            Text(text_out) if text_out else "[dim](empty transcript)[/dim]",
            title="[bold]Transcript[/bold]",
            title_align="left",
            subtitle=f"[dim]{model} · {response_format} · {latency_s:.2f}s[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    app()
