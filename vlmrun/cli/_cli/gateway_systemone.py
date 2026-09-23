"""``vlmrun gw systemone`` — typed, calibrated decisions on the gateway.

A thin renderer over :class:`vlmrun.client.systemone.SystemOne`, which drives
the official ``typesafe-sdk`` client. The command is registered on the gateway
Typer app from :mod:`vlmrun.cli._cli.gateway`.
"""

from __future__ import annotations

import json
import statistics
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from vlmrun.client import VLMRun
from vlmrun.client.exceptions import DependencyError, InputError
from vlmrun.client.systemone import (
    MAX_IMAGES,
    normalize_questions,
    timings_of,
    usage_of,
)
from vlmrun.common.mime import guess_mime, is_http_url, mime_from_url, suffix_from_url

console = Console()

SYSTEMONE_HELP = """Typed, calibrated decisions over text, JSON, images and PDFs.

`vlmrun gw s1` is a shorthand for this command.

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
    an image path or URL     -> image content part (up to 8; remote ones inlined)
    a .pdf path or URL       -> file content part (one per read; first pages)
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
  vlmrun gw s1 "Invoice #44 was charged twice" --noul is_urgent

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
GATES, REPEATS AND DRY RUNS:
  --gate asserts a condition and sets the exit code, so a read can guard a
  script. Repeatable; all must pass. Selectors:

\b
    id                      P(yes) for a noul, the score for a score,
                            the chosen label for a choice
    id.noul  id.score       the number, explicitly
    id.choice               the chosen label
    id.confidence           1 - H(p)/ln K
    id.probabilities.LABEL  one option's probability

\b
    vlmrun gw s1 report.pdf --noul is_safe --gate 'is_safe>0.9'
    vlmrun gw s1 ticket.txt --choice dept="billing|technical" --gate 'dept==billing' \\
      --gate 'dept.confidence>=0.6'

\b
  Exit codes: 0 every gate passed · 1 a gate failed · 2 the request failed.
  With --json the verdicts go to stderr so stdout stays a clean response body.

\b
  --repeat N sends the same request N times and reports mean, spread and every
  read — identical requests are not guaranteed to be identical reads, so this is
  how you see how much to trust one number. Gates then apply to the mean.
  (Distinct from --samples, which averages noise draws inside one read.)

\b
    vlmrun gw s1 ticket.txt --noul is_urgent --repeat 5

\b
  --dry-run prints the request body and exits without sending it. Piped output
  is the verbatim body; on a terminal, base64 payloads are abbreviated.

\b
    vlmrun gw s1 scan.jpg --noul signed --dry-run
    vlmrun gw s1 ticket.txt --noul a --dry-run | curl -sd @- \\
      "$VLMRUN_GATEWAY_URL/../typesafe/v1/systemone" -H "authorization: Bearer $VLMRUN_API_KEY"

\b
NOTES:
  Requires the typesafe extra: pip install vlmrun[typesafe]
  --detail is applied per request: one `high` input lifts the whole read.
  A gate is checked against the question spec before the request is sent, so a
  selector that cannot apply costs nothing.
  Remote images are fetched by this CLI (the route takes images only as data
  URLs) and must resolve to a public address; set VLMRUN_ALLOW_PRIVATE_URLS=1
  for an internal image host.
"""

# grep/diff convention: 1 means "the check did not pass", 2 means "it broke".
EXIT_GATE_FAILED = 1
EXIT_ERROR = 2

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
    return typer.Exit(EXIT_ERROR)


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


# Longest first, so ">=" is matched before ">".
GATE_OPERATORS = ("<=", ">=", "==", "!=", "<", ">")


def _aggregate(runs: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Fold one or more responses into per-question statistics.

    A single run aggregates to itself, so gates and rendering work off one
    shape whether ``--repeat`` was used or not.
    """
    folded: Dict[str, Dict[str, Any]] = {}
    for response in runs:
        for name, answer in response.answers.items():
            kind = getattr(answer, "type", None)
            entry = folded.setdefault(
                name,
                {
                    "kind": kind,
                    "values": [],
                    "labels": [],
                    "confidences": [],
                    "legend": {},
                },
            )
            if kind == "noul":
                entry["values"].append(float(answer.noul))
            elif kind == "choice":
                entry["labels"].append(str(answer.choice))
                entry["confidences"].append(float(answer.confidence))
                entry.setdefault("probabilities", []).append(dict(answer.probabilities))
            elif kind == "score":
                entry["values"].append(float(answer.score))
                entry["confidences"].append(float(answer.confidence))
                entry["legend"] = {int(k): v for k, v in dict(answer.legend).items()}
                entry.setdefault("probabilities", []).append(
                    {int(k): v for k, v in dict(answer.probabilities).items()}
                )
    for entry in folded.values():
        entry["mean"] = statistics.fmean(entry["values"]) if entry["values"] else None
        entry["stdev"] = (
            statistics.pstdev(entry["values"]) if len(entry["values"]) > 1 else 0.0
        )
        entry["confidence"] = (
            statistics.fmean(entry["confidences"]) if entry["confidences"] else None
        )
        entry["label"] = (
            Counter(entry["labels"]).most_common(1)[0][0] if entry["labels"] else None
        )
        probabilities = entry.get("probabilities") or []
        keys = {k for run in probabilities for k in run}
        entry["mean_probabilities"] = {
            key: statistics.fmean([run.get(key, 0.0) for run in probabilities])
            for key in keys
        }
    return folded


def _parse_gate(expression: str) -> Tuple[str, str, Any]:
    """Split ``dept.confidence>=0.7`` into its selector, operator and value."""
    for operator in GATE_OPERATORS:
        left, found, right = expression.partition(operator)
        if not found:
            continue
        selector, raw = left.strip(), right.strip()
        if not selector or not raw:
            break
        try:
            return selector, operator, float(raw)
        except ValueError:
            if operator not in ("==", "!="):
                raise _fail(
                    f"--gate {expression!r}: {raw!r} is not a number",
                    f"{operator} compares numbers; use == or != to compare a label.",
                )
            return selector, operator, raw.strip("\"'")
    raise _fail(
        f"--gate {expression!r} is not a comparison",
        "Write it as id>0.8, id.confidence>=0.5, id==label or "
        "id.probabilities.label>0.3.",
    )


# Which selectors mean anything for each answer type. A noul has no
# confidence and no options, so `approved.choice` is a malformed gate, not a
# gate that happens to be false.
GATE_FIELDS: Dict[str, frozenset] = {
    "noul": frozenset({"noul"}),
    "choice": frozenset({"choice", "confidence", "probabilities"}),
    "score": frozenset({"score", "confidence", "probabilities"}),
}


def _check_gate_selector(
    selector: str, kind: str | None, name: str, known: List[str]
) -> None:
    """Reject a selector the answer type cannot carry.

    Raises:
        typer.Exit: The question is absent, or the field does not apply to it.
    """
    if kind is None:
        raise _fail(
            f"--gate refers to {name!r}, which is not one of the questions asked",
            f"Questions in this request: {', '.join(known) or 'none'}.",
        )
    field = selector.partition(".")[2]
    if not field:
        return
    allowed = GATE_FIELDS.get(kind, frozenset())
    if field.partition(".")[0] not in allowed:
        raise _fail(
            f"--gate selector {selector!r} is not valid for a {kind} question",
            f"A {kind} question supports: {name}, "
            + ", ".join(f"{name}.{f}" for f in sorted(allowed))
            + ("." if kind == "noul" else " (probabilities takes a label)."),
        )


def _validate_gates(gates: List[Tuple[str, str, Any]], questions: Any) -> None:
    """Check every gate against the question spec, before a read is paid for.

    The types are known from the request, so a broken gate should cost nothing.
    :func:`_gate_value` repeats the check against the answers as a backstop.
    """
    types = {name: q.get("type") for name, q in normalize_questions(questions).items()}
    for selector, _, _ in gates:
        name = selector.partition(".")[0]
        _check_gate_selector(selector, types.get(name), name, list(types))


def _gate_value(folded: Dict[str, Dict[str, Any]], selector: str) -> Any:
    """Resolve a gate selector against the aggregated answers.

    Raises:
        typer.Exit: The question is absent, or the field is not one this answer
            type carries — a mismatch is a broken gate, and silently resolving
            it to None would let it pass.
    """
    name, _, field = selector.partition(".")
    entry = folded.get(name)
    _check_gate_selector(selector, entry and entry["kind"], name, list(folded))
    kind = entry["kind"]
    if not field:
        # The primary value of each type: P(yes), the chosen label, the score.
        return entry["label"] if kind == "choice" else entry["mean"]
    if field in ("noul", "score"):
        return entry["mean"]
    if field == "choice":
        return entry["label"]
    if field == "confidence":
        return entry["confidence"]
    _, _, label = field.partition(".")
    if not label:
        raise _fail(
            f"--gate selector {selector!r} needs an option",
            f"e.g. {name}.probabilities.<label>.",
        )
    key = int(label) if kind == "score" and label.isdigit() else label
    if key not in entry["mean_probabilities"]:
        raise _fail(
            f"--gate refers to {selector!r}, but {name!r} has no option {label!r}",
            f"Options: {', '.join(str(k) for k in entry['mean_probabilities'])}.",
        )
    return entry["mean_probabilities"][key]


def _compare(left: Any, operator: str, right: Any) -> bool:
    """Apply one gate operator, comparing labels as strings."""
    if left is None:
        # Unreachable once _gate_value validates the selector; a gate must
        # never pass because its left-hand side was missing.
        return False
    if isinstance(left, str) or isinstance(right, str):
        if operator == "==":
            return str(left) == str(right)
        if operator == "!=":
            return str(left) != str(right)
        raise _fail(
            f"cannot apply {operator} to the label {left!r}",
            "Numeric operators work on noul, score, confidence and probabilities.",
        )
    return {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }[operator](left, right)


def _evaluate_gates(
    gates: List[Tuple[str, str, Any]], folded: Dict[str, Dict[str, Any]]
) -> List[Tuple[bool, str, Any]]:
    """``(passed, expression, observed value)`` for each gate."""
    results = []
    for selector, operator, expected in gates:
        observed = _gate_value(folded, selector)
        shown = f"{observed:.2f}" if isinstance(observed, float) else str(observed)
        results.append(
            (
                _compare(observed, operator, expected),
                f"{selector}{operator}{expected}",
                shown,
            )
        )
    return results


def _render_gates(results: List[Tuple[bool, str, Any]], *, err: bool = False) -> None:
    """Print each gate's verdict, to stderr when stdout carries JSON."""
    out = Console(stderr=True) if err else console
    width = max(len(expression) for _, expression, _ in results)
    for index, (passed, expression, observed) in enumerate(results):
        mark = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        label = "gates" if index == 0 else "     "
        out.print(
            f"[dim]{label}[/dim]  {mark}  {escape(expression):<{width}}  [dim]{observed}[/dim]"
        )


def _render_repeat(
    folded: Dict[str, Dict[str, Any]], runs: List[Any], latency_s: float
) -> None:
    """Print mean, spread and the individual reads across repeated requests."""
    name_width = max(len(name) for name in folded)
    lines: List[Text] = []
    for name, entry in folded.items():
        line = Text()
        line.append(f"{name:<{name_width}}  ", style="bold")
        if entry["kind"] == "choice":
            counts = Counter(entry["labels"])
            top, hits = counts.most_common(1)[0]
            line.append(f"{top} {hits}/{len(entry['labels'])}  ", style="cyan")
            line.append(f"confidence {entry['confidence']:.2f}", style="dim")
            detail = " · ".join(f"{label} x{n}" for label, n in counts.most_common())
        else:
            line.append(f"{entry['mean']:.3f} ± {entry['stdev']:.3f}  ", style="cyan")
            line.append("P(yes)" if entry["kind"] == "noul" else "score", style="dim")
            detail = " ".join(f"{value:.3f}" for value in entry["values"])
        lines.append(line)
        lines.append(Text(" " * (name_width + 2) + detail, style="dim", no_wrap=True))

    console.print(
        Panel(
            Group(*lines),
            title=f"Decisions [dim]({runs[0].model}, {len(runs)} reads)[/dim]",
            title_align="left",
            subtitle=f"[dim]{' · '.join(_stats_parts(runs, latency_s))}[/dim]",
            border_style="dim",
        )
    )


def _stats_parts(runs: List[Any], wall_s: float) -> List[str]:
    """The footer's numbers: tokens, then where the time actually went.

    `api` is measured around the HTTP call. A decision is not streamed and its
    body is a few hundred bytes, so time-to-first-byte is normally the whole of
    it — `ttfb` is only broken out when the two diverge. `prep` is the local
    encoding, `wall` is everything this process did.
    """
    timings = [t for t in (timings_of(run) for run in runs) if t is not None]
    usages = [usage_of(run) for run in runs]
    tokens = sum(u.get("input_tokens") or 0 for u in usages)
    cached = sum(
        (u.get("input_tokens_details") or {}).get("cached_tokens") or 0 for u in usages
    )
    reads = sum(u.get("reads") or 0 for u in usages)
    cost = sum(u.get("cost") or 0.0 for u in usages)

    parts = []
    if tokens:
        parts.append(f"{tokens} tok" + (f" ({cached} cached)" if cached else ""))
    # `reads` only earns a column when it is not one per request — grouping or
    # `samples` multiplied the bill and nothing else in the footer would say so.
    if reads > len(runs):
        parts.append(f"{reads} reads")

    def summarize(values: List[float], label: str) -> str | None:
        if not values:
            return None
        if len(values) == 1:
            return f"{label} {values[0]:.0f} ms"
        ordered = sorted(values)
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        return f"{label} {statistics.median(ordered):.0f}/{p95:.0f} ms"

    api = [t.api_ms for t in timings if t.api_ms is not None]
    ttfb = [t.ttfb_ms for t in timings if t.ttfb_ms is not None]
    prep = [t.prep_ms for t in timings]
    transfer = [t.transfer_ms for t in timings if t.transfer_ms is not None]

    for value in (summarize(api, "api"),):
        if value:
            parts.append(value)
    # Only worth a column when the body took real time to come down.
    if transfer and max(transfer) >= 5.0:
        value = summarize(ttfb, "ttfb")
        if value:
            parts.append(value)
    if prep and max(prep) >= 1.0:
        value = summarize(prep, "prep")
        if value:
            parts.append(value)
    parts.append(f"wall {wall_s * 1000 / max(1, len(runs)):.0f} ms")
    if cost:
        parts.append(f"${cost:.6f}")
    if len(runs) > 1:
        parts.append("p50/p95")
    return parts


def _timings_json(runs: List[Any], wall_s: float) -> str:
    """The same numbers as JSON, for --json runs (written to stderr)."""
    rows = []
    for run in runs:
        t = timings_of(run)
        rows.append(
            {
                "prep_ms": round(t.prep_ms, 1) if t else None,
                "ttfb_ms": round(t.ttfb_ms, 1) if t and t.ttfb_ms is not None else None,
                "api_ms": round(t.api_ms, 1) if t and t.api_ms is not None else None,
                "total_ms": round(t.total_ms, 1) if t else None,
                "usage": usage_of(run),
            }
        )
    return json.dumps({"wall_ms": round(wall_s * 1000, 1), "reads": rows})


def _elide(value: Any) -> Any:
    """Shorten base64 payloads so a dry-run body stays readable on a terminal."""
    if isinstance(value, str) and value.startswith("data:") and len(value) > 96:
        header, _, payload = value.partition(",")
        return f"{header},<{len(payload)} base64 chars>"
    if isinstance(value, dict):
        return {k: _elide(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_elide(v) for v in value]
    return value


def _print_body(body: Dict[str, Any]) -> None:
    """Print the request body: verbatim when piped, abbreviated on a terminal."""
    if console.is_terminal:
        console.print_json(json.dumps(_elide(body)))
    else:
        print(json.dumps(body))


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

    stats = _stats_parts([response], latency_s)
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
    """Print the server's response body verbatim (a JSON array when repeated)."""
    if isinstance(response, list):
        console.print_json(json.dumps([_raw_body(run) for run in response]))
        return
    console.print_json(json.dumps(_raw_body(response)))


def _raw_body(response: Any) -> Any:
    """The response exactly as the server sent it, falling back to the model."""
    try:
        return json.loads(response.raw_http_response.text)
    except Exception:
        return json.loads(response.model_dump_json())


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
    gate: Optional[List[str]] = typer.Option(
        None,
        "--gate",
        "-g",
        help=(
            "Assert a condition, e.g. 'is_urgent>0.8', 'dept==billing', "
            "'dept.confidence>=0.5'. Repeatable. Exit 1 if any fails."
        ),
    ),
    repeat: int = typer.Option(
        1,
        "--repeat",
        "-r",
        min=1,
        max=64,
        help="Send the request N times and report mean and spread.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the request body and exit without sending it.",
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

    # Documented precedence is --body < -Q < flags, but extra_body is merged
    # last, so any field with a dedicated flag must be dropped from the body
    # once that flag is given — otherwise `--body '{"samples":32}' --samples 1`
    # silently bills 32 draws.
    overridden = {
        "model": True,
        "steps": steps is not None,
        "samples": samples is not None,
        "content": bool(images or document),
    }
    extra_body = {k: v for k, v in request_body.items() if not overridden.get(k, False)}
    gates = [_parse_gate(expression) for expression in (gate or [])]
    if gates:
        _validate_gates(gates, question_spec)
    system_one = client.gateway.systemone

    if dry_run:
        try:
            _print_body(
                system_one.build_request(
                    resolved_state,
                    question_spec,
                    model=model or request_body.get("model"),
                    images=images,
                    document=document,
                    detail=detail,  # type: ignore[arg-type]
                    steps=steps,
                    samples=samples,
                    extra_body=extra_body or None,
                )
            )
        except InputError as e:
            raise _fail(str(e.message), e.suggestion)
        return

    runs = []
    start = time.time()
    try:
        for _ in range(repeat):
            runs.append(
                system_one.decide(
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
            )
    except DependencyError as e:
        raise _fail(str(e.message), e.suggestion)
    except InputError as e:
        raise _fail(str(e.message), e.suggestion)
    except Exception as e:  # noqa: BLE001 - rendered below, re-raised when unknown
        if not type(e).__name__.startswith("TypeSafe"):
            raise
        if hasattr(e, "status"):
            _render_api_error(e)
        else:
            console.print(f"[red]Error:[/] {escape(str(e))}")
        raise typer.Exit(EXIT_ERROR)
    latency_s = time.time() - start

    folded = _aggregate(runs)
    if output_json:
        _print_json(runs if repeat > 1 else runs[0])
        Console(stderr=True).print(_timings_json(runs, latency_s), soft_wrap=True)
    elif repeat > 1:
        _render_repeat(folded, runs, latency_s)
    else:
        _render(runs[0], latency_s)

    if gates:
        results = _evaluate_gates(gates, folded)
        _render_gates(results, err=output_json)
        if not all(passed for passed, _, _ in results):
            raise typer.Exit(EXIT_GATE_FAILED)
