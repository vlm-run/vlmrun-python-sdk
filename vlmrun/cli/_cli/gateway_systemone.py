"""``vlmrun gw systemone`` — typed, calibrated decisions on the gateway.

A thin renderer over :class:`vlmrun.client.systemone.SystemOne`, which drives
the official ``typesafe-sdk`` client. The command is registered on the gateway
Typer app from :mod:`vlmrun.cli._cli.gateway`.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from vlmrun.client import VLMRun
from vlmrun.client.exceptions import DependencyError, InputError
from vlmrun.client.systemone import MAX_IMAGES, normalize_questions
from vlmrun.common.mime import guess_mime, is_http_url, mime_from_url, suffix_from_url

console = Console()

SYSTEMONE_HELP = """Typed, calibrated decisions over text, JSON, images and PDFs.

Answers are read off the model in a single denoise step: nothing is generated and
nothing is parsed, so an answer can never be off-schema, and the probabilities are
the model's own. A read costs one forward pass — use it to classify, route, gate or
score, not to write text.

\b
QUESTIONS:
  Every question has an id, a type, and optional instructions. The id never reaches
  the model; it is the key its answer comes back under.

\b
    noul    yes/no. Answer is P(yes), 0.0-1.0.
    choice  pick one of 2-128 named options. Answer is the option + per-option
            probabilities + confidence.
    score   ordered rubric of 2-10 levels. Answer is the expected level
            (0-indexed, fractional) + legend + probabilities + confidence.

\b
  -Q takes either a list (id carried per question) or the wire map (id as key).
  Both accept @file.json and - for stdin.

\b
  List form — write or generate this:
    -Q '{"questions": [
          {"id": "refund_requested", "type": "noul",
           "instructions": "Does the state request a refund?"},
          {"id": "request_type", "type": "choice",
           "instructions": "What is the main request in the state?",
           "options": [
             {"name": "refund", "description": "The customer wants money returned"},
             "rebooking",
             "information"]},
          {"id": "frustration", "type": "score",
           "instructions": "How frustrated does the customer appear?",
           "levels": ["Calm and neutral", "Concerned but civil", "Very angry"]}
        ]}'

\b
  The outer {"questions": ...} wrapper is optional; a bare [...] is the same.
  Answers render in list order.

\b
  Map form — the wire shape, sent as-is:
    -Q '{"refund_requested": {"type": "noul"},
         "request_type": {"type": "choice",
                          "criteria": {"refund": "Money returned",
                                       "rebooking": null, "information": null}}}'

\b
  Field aliases, accepted in either form:
    options  = criteria   (choice)  list of "name" or {"name","description"},
                                    or a map {name: description|null}
    levels   = criteria   (score)   ordered list of level descriptions
    criteria                        the wire value, verbatim

\b
  Inline flags for quick asks, repeatable, merged over -Q:
    --noul   ID[="question text"]
    --choice ID="a|b|c"             a label may carry :description
    --score  ID="level0|level1|..."

\b
STATE AND MEDIA:
  Positionals are resolved by what they are:

\b
    an image path or URL     -> image input (up to 8; remote ones are inlined)
    a .pdf path or URL       -> document input (one per read; first pages)
    .txt .md .json .yaml     -> the state (.json parsed into an object)
    anything else            -> literal state text (several are joined)

\b
  Use -s/--state to be explicit; then every positional is media.
  -s accepts text, @file, or - for stdin.

\b
ANSWERS:
  --json prints the response verbatim, keyed by your ids:

\b
    {"model": "...",
     "answers": {
       "refund_requested": {"type": "noul", "noul": 0.88},
       "request_type": {"type": "choice", "choice": "refund",
                        "probabilities": {"refund": 0.89, "rebooking": 0.07,
                                          "information": 0.04},
                        "confidence": 0.74},
       "frustration": {"type": "score", "score": 1.2,
                       "legend": {"0": "Calm and neutral", "1": "..."},
                       "probabilities": {"0": 0.14, "1": 0.52, "2": 0.34},
                       "confidence": 0.22}},
     "usage": {"input_tokens": 208, "output_tokens": 0}}

\b
  confidence = 1 - H(p)/ln K: 1.0 is certain, 0.0 is uniform over the options.
  output_tokens is always 0 — nothing is generated.

\b
LIMITS:
  1+ questions · choice 2-128 options · score 2-10 levels
  images <= 8 (JPEG/PNG/WebP/GIF, 5 MB each) · one PDF, first pages read
  steps 1-8 · samples 1-32 (every draw is billed)
  Unknown fields are rejected (422), so a typo fails loudly rather than being
  ignored.

\b
EXAMPLES:
  vlmrun gw systemone "Invoice #44 was charged twice" --noul is_urgent

\b
  vlmrun gw systemone ticket.txt \\
    --choice department="billing|technical|sales" \\
    --score frustration="Calm|Frustrated|Very angry"

\b
  vlmrun gw systemone ticket.txt -Q @triage.json --json

\b
  vlmrun gw systemone invoice.pdf --detail high \\
    --choice kind="invoice|receipt|contract"

\b
  vlmrun gw systemone -s "Does the scan match the claim?" scan.jpg claim.pdf \\
    --noul matches

\b
  vlmrun gw systemone --body '{"state": "...", "questions": [...], "samples": 4}'

\b
NOTES:
  Requires the typesafe extra: pip install vlmrun[typesafe]
  --detail is applied per request: one `high` input lifts the whole read.
"""

# Extensions read as the state rather than as media.
TEXT_SUFFIXES = frozenset(
    {".txt", ".md", ".markdown", ".json", ".jsonl", ".yaml", ".yml", ".csv", ".log"}
)


def _fail(message: str, suggestion: str | None = None) -> "typer.Exit":
    """Print an error (with an optional hint) and build the exit.

    Both strings are escaped: error text carries JSON and ``vlmrun[typesafe]``
    style install hints, which Rich would otherwise read as markup tags.
    """
    console.print(f"[red]Error:[/] {escape(message)}")
    if suggestion:
        console.print(f"[dim]{escape(suggestion)}[/dim]")
    return typer.Exit(1)


def _read_arg(value: str) -> str:
    """Resolve an argument that may be inline text, ``@file`` or ``-`` (stdin)."""
    if value == "-":
        return sys.stdin.read()
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        if not path.is_file():
            raise _fail(f"{path} is not a file.")
        return path.read_text()
    return value


def _load_json_arg(value: str, what: str) -> Any:
    """Parse a JSON argument given inline, as ``@file.json``, or on stdin."""
    raw = _read_arg(value)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise _fail(f"--{what} must be valid JSON: {e}")


def _split_id(value: str, flag: str) -> Tuple[str, Optional[str]]:
    """Split ``ID=rest`` into its id and remainder."""
    name, sep, rest = value.partition("=")
    name = name.strip()
    if not name:
        raise _fail(f"--{flag} needs a question id, e.g. --{flag} is_urgent")
    return name, (rest if sep else None)


def _labelled_options(rest: str, flag: str, name: str) -> List[Any]:
    """Parse ``a|b:description|c`` into the question's option list."""
    options: List[Any] = []
    for item in rest.split("|"):
        item = item.strip()
        if not item:
            continue
        label, sep, description = item.partition(":")
        label = label.strip()
        if not label:
            raise _fail(f"--{flag} {name}: empty label in {rest!r}")
        options.append(
            {"name": label, "description": description.strip()} if sep else label
        )
    return options


def _questions_from_flags(
    nouls: Optional[List[str]],
    choices: Optional[List[str]],
    scores: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Build question objects from the repeatable inline flags."""
    questions: List[Dict[str, Any]] = []
    for value in nouls or []:
        name, rest = _split_id(value, "noul")
        question: Dict[str, Any] = {"id": name, "type": "noul"}
        if rest:
            question["instructions"] = rest.strip()
        questions.append(question)
    for value in choices or []:
        name, rest = _split_id(value, "choice")
        if not rest:
            raise _fail(
                f'--choice {name}: needs options, e.g. --choice {name}="a|b|c"',
                'A label may carry a description: --choice kind="invoice:A bill|receipt"',
            )
        questions.append(
            {
                "id": name,
                "type": "choice",
                "options": _labelled_options(rest, "choice", name),
            }
        )
    for value in scores or []:
        name, rest = _split_id(value, "score")
        if not rest:
            raise _fail(
                f'--score {name}: needs levels, e.g. --score {name}="low|medium|high"',
                "Levels are ordered lowest first.",
            )
        levels = [item.strip() for item in rest.split("|") if item.strip()]
        questions.append({"id": name, "type": "score", "levels": levels})
    return questions


def _merge_questions(base: Any, overrides: List[Dict[str, Any]]) -> Any:
    """Merge inline-flag questions over a -Q/--body spec, flags winning."""
    if not overrides:
        return base
    if base is None:
        return overrides
    merged = normalize_questions(base)
    for question in overrides:
        body = {k: v for k, v in question.items() if k != "id"}
        merged[str(question["id"])] = body
    return merged


def _classify(raw: str) -> str:
    """Classify one positional as ``image``, ``document``, ``state_file`` or ``text``."""
    if is_http_url(raw):
        suffix, mime = suffix_from_url(raw), mime_from_url(raw)
        if suffix == ".pdf" or mime == "application/pdf":
            return "document"
        if mime.startswith("image/"):
            return "image"
        raise _fail(
            f"cannot tell what {raw} is from its URL.",
            "Use a URL ending in an image or .pdf extension, or download it first.",
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        return "text"
    suffix = path.suffix.lower()
    mime = guess_mime(path)
    if mime == "application/pdf":
        return "document"
    if mime.startswith("image/"):
        return "image"
    if suffix in TEXT_SUFFIXES or mime.startswith("text/"):
        return "state_file"
    raise _fail(
        f"{raw} is not a supported input.",
        "This route reads images (JPEG/PNG/WebP/GIF, up to 8) and one PDF; "
        "pass text with -s @file.",
    )


def _state_from_file(path: Path) -> Any:
    """Read a state file, parsing JSON so the state goes over as an object."""
    text = path.read_text()
    if path.suffix.lower() in (".json", ".jsonl"):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text
    return text


def _resolve_inputs(
    inputs: List[str], explicit_state: bool
) -> Tuple[List[Any], List[str], Optional[str]]:
    """Split positionals into state parts, images and one document."""
    state_parts: List[Any] = []
    images: List[str] = []
    document: Optional[str] = None

    for raw in inputs:
        kind = _classify(raw)
        if kind == "image":
            images.append(raw)
        elif kind == "document":
            if document is not None:
                raise _fail(
                    "one PDF per request — its first pages are the read.",
                    "Send a second request for the other document.",
                )
            document = raw
        elif explicit_state:
            raise _fail(
                f"{raw} is not media, but --state was given.",
                "With -s/--state every positional must be an image or a PDF.",
            )
        elif kind == "state_file":
            state_parts.append(_state_from_file(Path(raw).expanduser()))
        else:
            state_parts.append(raw)

    if len(images) > MAX_IMAGES:
        raise _fail(f"{len(images)} images; the limit is {MAX_IMAGES}.")
    return state_parts, images, document


def _bar(value: float, width: int = 9) -> str:
    """A unicode meter for a 0-1 value."""
    value = max(0.0, min(1.0, float(value)))
    full, remainder = divmod(value * width * 8, 8)
    eighths = " ▏▎▍▌▋▊▉"
    return (
        "█" * int(full) + (eighths[int(remainder)] if int(remainder) else "")
    ).ljust(width)


def _probabilities_line(
    probabilities: Dict[Any, float],
    legend: Dict[Any, Any] | None = None,
    *,
    by_key: bool = False,
) -> str:
    """One line of ``label prob`` pairs — score rubrics in level order, choices by likelihood."""
    items = sorted(
        probabilities.items(), key=(lambda kv: kv[0]) if by_key else (lambda kv: -kv[1])
    )
    parts = []
    for key, probability in items:
        label = f"{key} {legend[key]}" if legend and key in legend else str(key)
        parts.append(f"{label} {probability:.2f}")
    return " · ".join(parts)


def _answer_row(answer: Any) -> Tuple[str, float, str, str]:
    """``(shown answer, meter value, metric name, probabilities line)``."""
    kind = getattr(answer, "type", None)
    if kind == "noul":
        value = float(answer.noul)
        return ("yes" if value >= 0.5 else "no", value, "P(yes)", "")
    if kind == "choice":
        return (
            str(answer.choice),
            float(answer.confidence),
            "confidence",
            _probabilities_line(dict(answer.probabilities)),
        )
    if kind == "score":
        legend = {int(k): v for k, v in dict(answer.legend).items()}
        nearest = legend.get(int(round(float(answer.score))), "")
        shown = f"{float(answer.score):.1f}"
        if isinstance(nearest, str) and nearest:
            shown = f"{shown} {nearest}"
        return (
            shown,
            float(answer.confidence),
            "confidence",
            _probabilities_line(
                {int(k): v for k, v in dict(answer.probabilities).items()},
                legend,
                by_key=True,
            ),
        )
    return (str(answer), 0.0, "", "")


def _render(response: Any, latency_s: float) -> None:
    """Print the decisions, one line each, with a wrapped probabilities line below."""
    rows = [(name, *_answer_row(answer)) for name, answer in response.answers.items()]
    if not rows:
        console.print("[yellow]The response carried no answers.[/yellow]")
        return

    name_width = max(len(row[0]) for row in rows)
    answer_width = min(max(len(row[1]) for row in rows), 34)

    lines: List[Text] = []
    for name, shown, value, metric, probabilities in rows:
        line = Text()
        line.append(f"{name:<{name_width}}  ", style="bold")
        line.append(f"{shown[:answer_width]:<{answer_width}}  ", style="cyan")
        line.append(_bar(value), style="green")
        line.append(f"  {value:.2f}")
        line.append(f"  {metric}", style="dim")
        lines.append(line)
        if probabilities:
            indent = " " * (name_width + 4)
            width = max(20, console.width - len(indent) - 6)
            for chunk in textwrap.wrap(probabilities, width=width) or [probabilities]:
                lines.append(Text(indent + chunk, style="dim", no_wrap=True))

    stats = [f"{latency_s * 1000:.0f} ms"]
    input_tokens = getattr(response.usage, "input_tokens", None)
    if input_tokens:
        stats.insert(0, f"{input_tokens} input tokens")
    console.print(
        Panel(
            Group(*lines),
            title=f"Decisions [dim]({response.model})[/dim]",
            title_align="left",
            subtitle=f"[dim]{' · '.join(stats)}[/dim]",
            border_style="dim",
        )
    )


def _print_json(response: Any) -> None:
    """Print the server's response body verbatim."""
    try:
        console.print_json(response.raw_http_response.text)
    except Exception:
        console.print_json(response.model_dump_json())


def _render_api_error(exc: Any) -> None:
    """Render a TypeSafe API error, unpacking the 422 validation list."""
    status = getattr(exc, "status", None)
    body = getattr(exc, "body", None)
    detail = body.get("detail") if isinstance(body, dict) else None

    console.print(f"[red]Error:[/] The gateway rejected this request ({status}).")
    if isinstance(detail, list):
        for item in detail:
            if not isinstance(item, dict):
                console.print(f"  {item}")
                continue
            location = ".".join(
                str(part) for part in item.get("loc", []) if part != "body"
            )
            console.print(
                f"  [yellow]{escape(location or 'body')}[/yellow]  {escape(str(item.get('msg', '')))}"
            )
        console.print(
            "[dim]Unknown fields are rejected outright — check for typos in question keys.[/dim]"
        )
    elif isinstance(detail, dict):
        console.print(f"  {escape(str(detail.get('message', detail)))}")
    elif detail:
        console.print(f"  {escape(str(detail))}")
    else:
        console.print(f"  {escape(str(exc))}")

    request_id = getattr(exc, "request_id", None)
    if request_id:
        console.print(f"[dim]request id: {request_id}[/dim]")


def systemone(
    ctx: typer.Context,
    inputs: List[str] = typer.Argument(
        None,
        help="State text, or file paths / URLs (images, one PDF, or a text state file).",
    ),
    questions: Optional[str] = typer.Option(
        None,
        "--questions",
        "-Q",
        help="Questions as inline JSON (list or map), @file.json, or - for stdin.",
    ),
    noul: Optional[List[str]] = typer.Option(
        None,
        "--noul",
        help='Yes/no question: ID or ID="question text". Repeatable.',
    ),
    choice: Optional[List[str]] = typer.Option(
        None,
        "--choice",
        help='Choice question: ID="a|b|c". A label may carry :description. Repeatable.',
    ),
    score: Optional[List[str]] = typer.Option(
        None,
        "--score",
        help='Score question: ID="level0|level1|...". Repeatable.',
    ),
    state: Optional[str] = typer.Option(
        None,
        "--state",
        "-s",
        help="State as text, @file, or -. Every positional is then media.",
    ),
    body: Optional[str] = typer.Option(
        None,
        "--body",
        "-B",
        help="Full request body as JSON or @file.json; other flags override it.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model override (default: the served model)."
    ),
    detail: str = typer.Option(
        "auto",
        "--detail",
        help="Vision budget for this request: auto, low or high.",
    ),
    steps: Optional[int] = typer.Option(
        None, "--steps", min=1, max=8, help="Denoise steps per read (1-8)."
    ),
    samples: Optional[int] = typer.Option(
        None,
        "--samples",
        min=1,
        max=32,
        help="Noise draws to average (1-32). Every draw is billed.",
    ),
    output_json: bool = typer.Option(
        False, "--json", "-j", help="Print the raw response JSON."
    ),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", help="Request timeout in seconds."
    ),
) -> None:
    """Read typed decisions off the gateway's System One route."""
    client: VLMRun = ctx.obj

    if detail not in ("auto", "low", "high"):
        raise _fail(f"--detail must be auto, low or high; got {detail!r}")

    request_body: Dict[str, Any] = {}
    if body:
        parsed = _load_json_arg(body, "body")
        if not isinstance(parsed, dict):
            raise _fail("--body must be a JSON object.")
        request_body = dict(parsed)

    question_spec: Any = request_body.pop("questions", None)
    if questions:
        question_spec = _load_json_arg(questions, "questions")
    question_spec = _merge_questions(
        question_spec, _questions_from_flags(noul, choice, score)
    )
    if question_spec is None:
        raise _fail(
            "no questions were given.",
            'Add --noul ID, --choice ID="a|b|c", --score ID="low|high", or -Q \'{...}\'.',
        )

    state_parts, images, document = _resolve_inputs(
        list(inputs or []), state is not None
    )
    resolved_state: Any = request_body.pop("state", None)
    if state is not None:
        resolved_state = _read_arg(state)
    elif len(state_parts) == 1:
        resolved_state = state_parts[0]
    elif state_parts:
        resolved_state = " ".join(str(part) for part in state_parts)

    if resolved_state is None:
        if not images and not document:
            raise _fail(
                "no state was given.",
                "Pass text, a text file, -s @file, or at least one image or PDF.",
            )
        resolved_state = ""

    extra_body = {k: v for k, v in request_body.items() if k not in ("model",)}
    start = time.time()
    try:
        response = client.gateway.systemone.decide(
            resolved_state,
            question_spec,
            model=model or request_body.get("model"),
            images=images,
            document=document,
            detail=detail,  # type: ignore[arg-type]
            steps=steps,
            samples=samples,
            timeout=timeout,
            extra_body=extra_body or None,
        )
    except DependencyError as e:
        raise _fail(str(e.message), e.suggestion)
    except InputError as e:
        raise _fail(str(e.message), e.suggestion)
    except Exception as e:  # noqa: BLE001 - rendered below, re-raised when unknown
        typesafe_error = type(e).__name__.startswith("TypeSafe")
        if not typesafe_error:
            raise
        if hasattr(e, "status"):
            _render_api_error(e)
        else:
            console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
    latency_s = time.time() - start

    if output_json:
        _print_json(response)
    else:
        _render(response, latency_s)
