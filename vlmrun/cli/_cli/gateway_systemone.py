"""``vlmrun gw systemone`` — typed, calibrated decisions on the gateway.

A thin renderer over :class:`vlmrun.client.systemone.SystemOne`, which drives
the official ``typesafe-sdk`` client. The command is registered on the gateway
Typer app from :mod:`vlmrun.cli._cli.gateway`.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console, Group
from rich import box
from rich.markup import escape
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from vlmrun.client import VLMRun
from vlmrun.client.exceptions import DependencyError, InputError
from vlmrun.client.systemone import (
    MAX_IMAGES,
    REASONING_EFFORTS,
    SYSTEMONE_MODEL,
    FrameDecision,
    probe_url_mime,
    normalize_questions,
    timings_of,
    usage_of,
)
from vlmrun.common.mime import (
    guess_mime,
    is_http_url,
    mime_from_url,
    normalize_mime,
    suffix_from_url,
)
from vlmrun.constants import SUPPORTED_VIDEO_FILETYPES

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
    an image path, URL or data: URL -> image part (up to 8; remote ones inlined)
    a .pdf path, URL or data: URL   -> file part (one per read; first pages)
    a video path (.mp4 .mov .avi .mkv .webm) -> sampled into frames, one read each
    .txt .md .json .yaml     -> the state (.json parsed into an object)
    anything else            -> literal state text (several are joined)

\b
  A URL that carries no usable extension — a signed link, an object-store key —
  is probed for what it actually serves, so `gw s1 <url>` works the way
  `gw chat <url>` does.

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
  output_tokens is 0 on a diffusion engine, which emits the template it was
  seeded with, and non-zero on a generative one, which writes it.

\b
LIMITS:
  1+ questions · choice 2-128 options · score 2-10 levels
  images <= 8 (JPEG/PNG/WebP/GIF, 5 MB each) · one PDF, first pages read
  one video per command, sampled to <= --max-frames frames (default 60)
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
  vlmrun gw systemone door.mp4 --fps 2 --noul is_open

\b
  vlmrun gw systemone --body '{"state": "...", "questions": [...], "samples": 4}'

\b
  vlmrun gw s1 models

\b
MODELS AND REASONING:
  Several engines serve this route and there is no catch-all alias — name one
  with -m, or take the default. `vlmrun gw s1 models` lists what this gateway
  actually serves:

\b
    google/diffusiongemma-26b-a4b-it   default; one denoise step over a seeded canvas
    google/gemma-4-26b-a4b-it          generative; 4B active, cheap for its size
    qwen/qwen3.5-0.8b                  generative; smallest, less peaked answers
    qwen/qwen3.8-27b                   generative; the chat engine the OpenAI routes serve

\b
  --reasoning-effort lets a generative engine think first: none (default),
  minimal, low, medium or high — 0/32/64/128/256 tokens before the answer, all
  billed as output tokens. A diffusion engine refuses the field.

\b
    vlmrun gw s1 ticket.txt -m google/gemma-4-26b-a4b-it \\
      --reasoning-effort medium --choice dept="billing|tax|technical"

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
VIDEO:
  Pass a video and it is sampled into frames, each frame its own read.
  There is no video model here and no streaming: the timeline is built
  client-side out of single-frame decisions, which is what makes every
  frame's answer independent and comparable.

\b
    --fps N          frames read per second of video (default 1)
    --max-frames N   refuse to sample past N frames (default 60)
    --concurrency N  reads in flight at once (default 4)
    --ws             read the clip over one websocket session

\b
  --fps is a request, not a promise: it cannot exceed the video's own
  rate, and nothing is interpolated. Sampling follows the presentation
  clock, so a variable-rate video is read at the times asked for rather
  than every Nth frame.

\b
    vlmrun gw s1 door.mp4 --noul is_open --fps 2

\b
  Every frame is a billed read, so --fps and the clip's length set the
  bill: 2 fps over 30s is 60 reads. --max-frames is checked against the
  duration before anything is sent, so an expensive ask fails for free.

\b
  Answers render as a series over time, not as a mean: a sparkline per
  numeric question with its true peak and low, and a choice as the
  segments it held. A series longer than the terminal is averaged into
  the columns available — the peak and low beside it stay the real ones.

\b
    vlmrun gw s1 door.mp4 --choice state="open|closed|blocked" --fps 1
      state  closed 0.0s-3.0s · open 4.0s-11.0s · blocked 12.0s-14.0s

\b
  --json gives every frame, each read tagged with where it came from:

\b
    {"video": {"fps": 1.0, "frames": 15, "duration_s": 15.0,
               "native_fps": 30.0},
     "reads": [{"frame": 0, "timestamp_s": 0.0, "response": {...}}, ...]}

\b
  --gate-mode says how a gate reads the timeline:

\b
    any (default)  the condition held on at least one frame
    all            it held on every frame
    sustained:N    it held on N frames in a row
    mean           it holds of the average across frames

\b
  any is the default because a gate over footage is usually asking
  whether something ever happened. sustained:N is the debounced form —
  it ignores a single frame's misread, which any by construction cannot.

\b
    vlmrun gw s1 door.mp4 --noul is_open --fps 2 --gate 'is_open>0.9'
    vlmrun gw s1 door.mp4 --noul is_open --fps 4 --gate 'is_open>0.9' \\
      --gate-mode sustained:8     # two seconds open, not one bad frame

\b
  --ws reads the clip over one session on /typesafe/ws instead of a request
  per frame. The questions are sent once with the handshake and cached
  server-side, and the reads are pipelined over a single socket, which is
  faster than a request each: 60 frames of a 60s clip read in 0.9s against
  2.0s at --concurrency 8, for the same tokens and cost. --concurrency is
  requested as the session's in-flight limit, which the route caps at 8; the
  server's answer wins.
  Needs the ws extra: pip install vlmrun[ws]

\b
    vlmrun gw s1 door.mp4 --ws --fps 2 --noul is_open

\b
  --dry-run prints the first frame's body: every frame's request is that
  one with a different image. --repeat does not combine with a video —
  it measures the spread of repeated reads of one fixed input, and each
  frame here is a different input.

\b
NOTES:
  Requires the typesafe extra: pip install vlmrun[typesafe]
  --detail is applied per request: one `high` input lifts the whole read.
  A gate is checked against the question spec before the request is sent, so a
  selector that cannot apply costs nothing.
  Remote images are fetched by this CLI (the route takes images only as data
  URLs) and must resolve to a public address; set VLMRUN_ALLOW_PRIVATE_URLS=1
  for an internal image host.
  Video needs the video extra: pip install vlmrun[video]
  A video has to be a local file — frames are sampled here, not by the gateway,
  so a URL or data: URL has to be downloaded first.
"""

# grep/diff convention: 1 means "the check did not pass", 2 means "it broke".
EXIT_GATE_FAILED = 1
EXIT_ERROR = 2

# Extensions read as the state rather than as media.
TEXT_SUFFIXES = frozenset(
    {".txt", ".md", ".markdown", ".json", ".jsonl", ".yaml", ".yml", ".csv", ".log"}
)

VIDEO_SUFFIXES = frozenset(SUPPORTED_VIDEO_FILETYPES)

# One frame a second reads a clip at a useful resolution without turning a
# minute of footage into thousands of billed reads.
DEFAULT_VIDEO_FPS = 1.0

# A ceiling on what one command can spend. Every frame is a separate read, so
# an unbounded --fps on a long clip is an unbounded bill; this makes the user
# say out loud that they meant it.
DEFAULT_MAX_FRAMES = 60

# How a gate reads a timeline. `mean` matches --repeat, and is the odd one out:
# averaging across time hides the moment a condition held, which is usually the
# thing being asked about.
GATE_MODES = ("any", "all", "mean", "sustained")


def _render_models(client: VLMRun, output_json: bool) -> None:
    """Print the models this gateway actually serves on the route."""
    listing = client.gateway.systemone.models()
    models = list(listing.models)
    if output_json:
        console.print_json(
            json.dumps(
                [
                    m.model_dump() if hasattr(m, "model_dump") else dict(m)
                    for m in models
                ]
            )
        )
        return
    table = Table(
        box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="dim"
    )
    table.add_column("model", style="bold", no_wrap=True)
    table.add_column("released", style="dim", no_wrap=True)
    table.add_column("description")
    for m in models:
        name = m.name + (" [dim](default)[/dim]" if m.name == SYSTEMONE_MODEL else "")
        table.add_row(name, m.release_date, m.description)
    console.print(
        Panel(
            table,
            title=f"System One models [dim]({client.gateway.systemone.base_url})[/dim]",
            title_align="left",
            subtitle="[dim]there is no catch-all alias — name one of these with -m[/dim]",
            border_style="dim",
        )
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
    if raw.startswith("data:"):
        mime = normalize_mime(raw[5:].split(";", 1)[0])
        if mime == "application/pdf":
            return "document"
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("video/"):
            raise _fail(
                "a video has to be a file on disk, not a data URL.",
                "Frames are sampled locally; write it out and pass the path.",
            )
        raise _fail(
            f"data URL of type {mime!r} is not a supported input.",
            "This route reads images (JPEG/PNG/WebP/GIF) and one PDF.",
        )
    if is_http_url(raw):
        kind = _kind_for_mime(suffix_from_url(raw), mime_from_url(raw))
        if kind == "video":
            raise _fail(
                f"{raw} is a video, and frames are sampled locally.",
                "Download it and pass the path.",
            )
        if kind:
            return kind
        # Nothing in the URL says what it is — a signed link, an object-store
        # key, a name in a query string. Ask the server what it serves.
        try:
            probed = probe_url_mime(raw)
        except InputError as e:
            raise _fail(str(e.message), e.suggestion)
        except Exception as e:  # noqa: BLE001 - a fetch failure is the user's to see
            raise _fail(
                f"could not reach {raw}: {e}",
                "Check the URL, or download the file and pass the path.",
            )
        kind = _kind_for_mime("", probed or "")
        if kind == "video":
            raise _fail(
                f"{raw} serves {probed}, and frames are sampled locally.",
                "Download it and pass the path.",
            )
        if kind:
            return kind
        raise _fail(
            f"{raw} serves {probed or 'an unidentifiable type'}, which this route does not read.",
            "It reads images (JPEG/PNG/WebP/GIF, up to 8) and one PDF.",
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        return "text"
    suffix = path.suffix.lower()
    mime = guess_mime(path)
    kind = _kind_for_mime(suffix, mime)
    if kind:
        return kind
    if suffix in TEXT_SUFFIXES or mime.startswith("text/"):
        return "state_file"
    raise _fail(
        f"{raw} is not a supported input.",
        "This route reads images (JPEG/PNG/WebP/GIF, up to 8), one PDF, or one "
        "video sampled into frames; pass text with -s @file.",
    )


def _kind_for_mime(suffix: str, mime: str) -> str | None:
    """``image``, ``document``, ``video`` or None, from an extension and a MIME type."""
    if suffix == ".pdf" or mime == "application/pdf":
        return "document"
    if mime.startswith("image/"):
        return "image"
    if suffix in VIDEO_SUFFIXES or mime.startswith("video/"):
        return "video"
    return None


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
) -> Tuple[List[Any], List[str], Optional[str], Optional[str]]:
    """Split positionals into state parts, images, one document and one video."""
    state_parts: List[Any] = []
    images: List[str] = []
    document: Optional[str] = None
    video: Optional[str] = None

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
        elif kind == "video":
            if video is not None:
                raise _fail(
                    "one video per command — each frame is a separate read.",
                    "Run the command again for the other clip.",
                )
            video = raw
        elif explicit_state:
            raise _fail(
                f"{raw} is not media, but --state was given.",
                "With -s/--state every positional must be an image, a PDF or a video.",
            )
        elif kind == "state_file":
            state_parts.append(_state_from_file(Path(raw).expanduser()))
        else:
            state_parts.append(raw)

    if len(images) > MAX_IMAGES:
        raise _fail(f"{len(images)} images; the limit is {MAX_IMAGES}.")
    # A video is read frame by frame, each frame its own single-image request.
    # Mixing in a still or a PDF would make every frame's read a different
    # question, and the per-frame answers would no longer be comparable.
    if video is not None and (images or document):
        raise _fail(
            "a video cannot be combined with other media.",
            "Each frame is read on its own; send the image or PDF separately.",
        )
    return state_parts, images, document, video


def _video_scope(source: str, fps: float, max_frames: int) -> Tuple[float, float]:
    """Read a video's shape, refusing an ask that would cost more than intended.

    Opens the decoder only to read its header, so the refusal costs nothing.
    Cost policy lives here rather than in the SDK: the ceiling is about not
    surprising someone with a bill, which is a property of this command.

    Args:
        source (str): Path to the video.
        fps (float): Frames to read per second of video.
        max_frames (int): Frames this command is willing to read.

    Returns:
        Tuple[float, float]: Duration in seconds and native frame rate, each
            0.0 when the container does not report it.

    Raises:
        typer.Exit: If the video cannot be opened, or would sample past the
            ceiling.
    """
    from vlmrun.common.video import VideoReader

    path = Path(source).expanduser()
    try:
        reader = VideoReader(path)
    except DependencyError as e:
        raise _fail(str(e.message), e.suggestion)
    except (FileNotFoundError, RuntimeError) as e:
        raise _fail(
            f"could not read {source}: {e}",
            "Check the path, and that the file is a video this OpenCV build decodes.",
        )
    with reader:
        duration_s, native_fps = reader.duration_s, reader.fps

    if duration_s > 0:
        # Marks fall at 0, 1/fps, 2/fps …, so the count is the last mark plus
        # one — flooring the product alone undercounts a partial final second and
        # lets a clip past the ceiling only to truncate it. The last frame sits a
        # frame-interval short of the duration, so that is the real last instant;
        # without it, a whole-second clip would be over-counted and refused.
        last_instant = duration_s - (1.0 / native_fps if native_fps > 0 else 0.0)
        estimate = max(1, math.floor(max(0.0, last_instant) * fps) + 1)
        if estimate > max_frames:
            raise _fail(
                f"{fps:g} fps over {duration_s:.1f}s is up to {estimate} frames, "
                f"past the {max_frames}-frame ceiling.",
                f"Each frame is a billed read. Lower --fps, or raise "
                f"--max-frames to {estimate}.",
            )
    return duration_s, native_fps


def _first_frame_image(source: str) -> str:
    """The first frame of a video as a data URL, for ``--dry-run``.

    Raises:
        typer.Exit: If the video cannot be read.
    """
    from vlmrun.common.image import encode_frame
    from vlmrun.common.video import VideoReader

    try:
        with VideoReader(Path(source).expanduser()) as reader:
            for sampled in reader.frames(1.0, max_frames=1):
                return encode_frame(sampled.frame)
    except DependencyError as e:
        raise _fail(str(e.message), e.suggestion)
    except (FileNotFoundError, RuntimeError) as e:
        raise _fail(f"could not read {source}: {e}")
    raise _fail(f"no frames were read from {source}.")


def _collect_stream(
    decisions: Any, *, total: int | None, show_progress: bool
) -> List[Any]:
    """Drain a decision stream, drawing a progress bar as answers arrive.

    The stream yields lazily, so the bar tracks real progress rather than being
    a spinner over one opaque wait.

    Args:
        decisions: The iterator from :meth:`DecisionStream.map`.
        total (int | None): Expected frame count, when it can be estimated.
        show_progress (bool): Draw the bar.

    Returns:
        List[Any]: Every :class:`~vlmrun.client.systemone.FrameDecision`, in order.
    """
    if not show_progress:
        return list(decisions)
    collected = []
    with Progress(
        TextColumn("[dim]reading[/dim]"),
        BarColumn(bar_width=24),
        TextColumn("[dim]{task.completed}/{task.total} frames[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("reading", total=total)
        for decision in decisions:
            collected.append(decision)
            progress.advance(task)
    return collected


def _sparkline(values: List[float], lo: float, hi: float) -> str:
    """A one-character-per-value plot of a series over its own timeline."""
    blocks = "▁▂▃▄▅▆▇█"
    span = hi - lo
    if span <= 0:
        return blocks[0] * len(values)
    out = []
    for value in values:
        fraction = (value - lo) / span
        index = int(round(max(0.0, min(1.0, fraction)) * (len(blocks) - 1)))
        out.append(blocks[index])
    return "".join(out)


def _downsample(values: List[float], width: int) -> List[float]:
    """Average a series into at most ``width`` buckets, preserving its shape.

    A sparkline is one character per value, so a series longer than the terminal
    has to be condensed to be drawn at all. Bucket means smooth a lone spike,
    which is why the extremes are reported separately and exactly.
    """
    if width < 1 or len(values) <= width:
        return values
    buckets = []
    for position in range(width):
        start = position * len(values) // width
        end = max(start + 1, (position + 1) * len(values) // width)
        chunk = values[start:end]
        buckets.append(statistics.fmean(chunk) if chunk else 0.0)
    return buckets


def _runs_of(labels: List[str]) -> List[Tuple[str, int, int]]:
    """Collapse a label series into ``(label, first index, last index)`` runs."""
    segments: List[Tuple[str, int, int]] = []
    for position, label in enumerate(labels):
        if segments and segments[-1][0] == label:
            name, start, _ = segments[-1]
            segments[-1] = (name, start, position)
        else:
            segments.append((label, position, position))
    return segments


def _clock(seconds: float) -> str:
    """A compact timestamp: ``4.5s`` under a minute, ``1:04`` over."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds) // 60}:{seconds % 60:04.1f}"


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


def _render_timeline(
    folded: Dict[str, Dict[str, Any]],
    runs: List[Any],
    frames: List[FrameDecision],
    latency_s: float,
    *,
    fps: float,
    duration_s: float,
) -> None:
    """Print each question as a series over the video's timeline.

    Deliberately not :func:`_render_repeat`'s mean and spread. Those describe a
    scatter of reads of one fixed input; these frames are different inputs, and
    the shape over time — when a value rose, how long it held — is the answer
    being asked for. A mean would average it away.
    """
    name_width = max(len(name) for name in folded)
    indent = " " * (name_width + 2)
    # Leave room for the name column, the metric label and the panel's borders.
    plot_width = max(8, console.width - name_width - 16)
    drawn_width = min(len(frames), plot_width)
    lines: List[Text] = []
    plotted = 0

    for name, entry in folded.items():
        line = Text()
        line.append(f"{name:<{name_width}}  ", style="bold")

        if entry["kind"] == "choice":
            # A label series is a segmentation, not a curve: say what held when.
            segments = _runs_of(entry["labels"])
            line.append(
                " · ".join(
                    f"{label} {_clock(frames[start].timestamp_s)}"
                    + ("" if start == end else f"-{_clock(frames[end].timestamp_s)}")
                    for label, start, end in segments
                ),
                style="cyan",
            )
            line.no_wrap = True
            lines.append(line)
            counts = Counter(entry["labels"])
            detail = " · ".join(f"{label} x{n}" for label, n in counts.most_common())
            if entry["confidence"] is not None:
                detail += f" · confidence {entry['confidence']:.2f}"
            lines.append(Text(indent + detail, style="dim", no_wrap=True))
            continue

        values = entry["values"]
        if not values:
            lines.append(line)
            continue
        # Fixed scale, not the series' own range: an autoscaled flat series
        # looks like violent movement.
        if entry["kind"] == "score":
            hi = float(max(entry["legend"])) if entry["legend"] else max(values)
            metric = "score"
        else:
            hi, metric = 1.0, "P(yes)"
        line.append(
            _sparkline(_downsample(values, plot_width), 0.0, hi or 1.0), style="green"
        )
        line.append(f"  {metric}", style="dim")
        lines.append(line)
        # Extremes go on their own line: with the timestamps they need, they do
        # not fit beside a sparkline as wide as the frame count.
        peak = max(range(len(values)), key=values.__getitem__)
        trough = min(range(len(values)), key=values.__getitem__)
        lines.append(
            Text(
                f"{indent}mean {entry['mean']:.2f}"
                f" · peak {values[peak]:.2f} @ {_clock(frames[peak].timestamp_s)}"
                f" · low {values[trough]:.2f} @ {_clock(frames[trough].timestamp_s)}",
                style="dim",
                no_wrap=True,
            )
        )
        plotted += 1

    # One time axis under the sparklines, which all share the frame grid.
    if plotted and len(frames) > 1:
        first, last = _clock(frames[0].timestamp_s), _clock(frames[-1].timestamp_s)
        axis = (
            f"{first}{' ' * (drawn_width - len(first) - len(last))}{last}"
            if drawn_width >= len(first) + len(last) + 1
            else f"{first} → {last}"
        )
        lines.append(Text(indent + axis, style="dim", no_wrap=True))

    shown_fps = f"{fps:g} fps"
    scope = f"{len(frames)} frames @ {shown_fps}"
    if duration_s > 0:
        scope += f" of {_clock(duration_s)}"
    console.print(
        Panel(
            Group(*lines),
            title=f"Decisions [dim]({runs[0].model}, {scope})[/dim]",
            title_align="left",
            subtitle=f"[dim]{' · '.join(_stats_parts(runs, latency_s))}[/dim]",
            border_style="dim",
        )
    )


def _parse_gate_mode(raw: str) -> Tuple[str, int]:
    """Split ``sustained:3`` into its mode and frame count.

    Args:
        raw (str): ``any``, ``all``, ``mean`` or ``sustained:N``.

    Returns:
        Tuple[str, int]: The mode, and the run length ``sustained`` needs (1 otherwise).

    Raises:
        typer.Exit: If the mode is unknown or its count is malformed.
    """
    name, _, count = raw.partition(":")
    name = name.strip().lower()
    if name not in GATE_MODES:
        raise _fail(
            f"--gate-mode must be one of {', '.join(GATE_MODES)}; got {raw!r}",
            "sustained takes a frame count, e.g. --gate-mode sustained:3.",
        )
    if name != "sustained":
        if count:
            raise _fail(f"--gate-mode {name} takes no count; got {raw!r}")
        return name, 1
    if not count:
        raise _fail(
            "--gate-mode sustained needs a frame count.",
            "e.g. --gate-mode sustained:3 — the gate holds for 3 frames in a row.",
        )
    try:
        window = int(count)
    except ValueError:
        raise _fail(
            f"--gate-mode sustained:{count!r} — {count!r} is not a whole number"
        )
    if window < 1:
        raise _fail("--gate-mode sustained needs a count of 1 or more")
    return "sustained", window


def _longest_run(flags: List[bool]) -> int:
    """The longest stretch of consecutive ``True`` in a series."""
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _evaluate_gates_over_frames(
    gates: List[Tuple[str, str, Any]],
    per_frame: List[Dict[str, Dict[str, Any]]],
    frames: List[FrameDecision],
    mode: str,
    window: int,
) -> List[Tuple[bool, str, Any]]:
    """``(passed, expression, observed)`` for each gate, read across the timeline.

    Args:
        gates: Parsed gate triples.
        per_frame: Each frame's answers, aggregated to themselves.
        frames: The sampled frames, for timestamps.
        mode: ``any``, ``all`` or ``sustained``.
        window: Frames a ``sustained`` gate must hold for.

    Returns:
        One verdict per gate.
    """
    results = []
    total = len(per_frame)
    for selector, operator, expected in gates:
        hits = [
            _compare(_gate_value(folded, selector), operator, expected)
            for folded in per_frame
        ]
        count = sum(hits)
        expression = f"{selector}{operator}{expected}"
        if mode == "sustained":
            longest = _longest_run(hits)
            passed = longest >= window
            observed = f"longest run {longest}/{window}"
            if passed:
                # Where the run that satisfied it began. The default cannot be
                # reached while _longest_run agrees with hits; it keeps a drift
                # between them from surfacing as a bare StopIteration.
                start = next(
                    (
                        position
                        for position in range(total - longest + 1)
                        if all(hits[position : position + longest])
                    ),
                    0,
                )
                observed += f" from {_clock(frames[start].timestamp_s)}"
        else:
            passed = count > 0 if mode == "any" else count == total
            observed = f"{count}/{total} frames"
            if count and mode == "any":
                observed += f", first @ {_clock(frames[hits.index(True)].timestamp_s)}"
        results.append((passed, expression, observed))
    return results


def _frames_json(
    frames: List[FrameDecision],
    runs: List[Any],
    *,
    fps: float,
    duration_s: float,
    native_fps: float,
) -> str:
    """The per-frame responses as one JSON object, each read carrying its timestamp."""
    return json.dumps(
        {
            "video": {
                "fps": fps,
                "frames": len(frames),
                "duration_s": round(duration_s, 3) or None,
                "native_fps": native_fps or None,
            },
            "reads": [
                {
                    "frame": frame.index,
                    "timestamp_s": round(frame.timestamp_s, 3),
                    "response": _raw_body(run),
                }
                for frame, run in zip(frames, runs)
            ],
        }
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
    out_tokens = sum(u.get("output_tokens") or 0 for u in usages)
    cached = sum(
        (u.get("input_tokens_details") or {}).get("cached_tokens") or 0 for u in usages
    )
    reads = sum(u.get("reads") or 0 for u in usages)
    cost = sum(u.get("cost") or 0.0 for u in usages)

    parts = []
    if tokens:
        # A generative engine writes the answer template (and any thought), so
        # output tokens are only worth a column when they are not zero.
        shown = f"{tokens} tok"
        if out_tokens:
            shown += f" +{out_tokens} out"
        if cached:
            shown += f" ({cached} cached)"
        parts.append(shown)
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
        # Only an extra_forbidden is a typo; a value_error is a field that
        # exists but does not apply, and the typo hint misdirects there.
        if any(
            isinstance(item, dict) and item.get("type") == "extra_forbidden"
            for item in detail
        ):
            console.print(
                "[dim]Unknown fields are rejected outright — check for typos in question keys.[/dim]"
            )
    elif isinstance(detail, dict):
        console.print(f"  {escape(str(detail.get('message', detail)))}")
    elif detail:
        console.print(f"  {escape(str(detail))}")
    else:
        console.print(f"  {escape(str(exc))}")

    if isinstance(detail, list) and any(
        "reasoning_effort" in str(item.get("loc", ""))
        for item in detail
        if isinstance(item, dict)
    ):
        console.print(
            "[dim]Only generative engines reason; the default model is a diffusion "
            "engine. Run `vlmrun gw s1 models` and pick one with -m.[/dim]"
        )

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
    fps: Optional[float] = typer.Option(
        None,
        "--fps",
        help=(
            "Frames to read per second of video "
            f"(default {DEFAULT_VIDEO_FPS:g}). Each frame is one read."
        ),
    ),
    max_frames: int = typer.Option(
        DEFAULT_MAX_FRAMES,
        "--max-frames",
        min=1,
        help="Refuse to sample more frames than this from one video.",
    ),
    concurrency: int = typer.Option(
        4,
        "--concurrency",
        "-c",
        min=1,
        max=32,
        help="Frame reads in flight at once.",
    ),
    ws: bool = typer.Option(
        False,
        "--ws",
        help=(
            "Read a video over one websocket session instead of a request per "
            "frame. Needs vlmrun[ws]."
        ),
    ),
    gate_mode: Optional[str] = typer.Option(
        None,
        "--gate-mode",
        help=(
            "How a gate reads a video: any (default), all, mean, or sustained:N "
            "for N frames in a row."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the request body and exit without sending it.",
    ),
    reasoning_effort: Optional[str] = typer.Option(
        None,
        "--reasoning-effort",
        "-R",
        help=(
            "Think before answering: none, minimal, low, medium or high. "
            "Generative engines only — the default model is a diffusion engine."
        ),
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

    # `gw s1 models` lists the served set. To classify the literal word, use -s.
    if list(inputs or []) == ["models"] and not (
        questions or noul or choice or score or body
    ):
        try:
            _render_models(client, output_json)
        except DependencyError as e:
            raise _fail(str(e.message), e.suggestion)
        except Exception as e:  # noqa: BLE001 - rendered below, re-raised when unknown
            if not type(e).__name__.startswith("TypeSafe"):
                raise
            (
                _render_api_error(e)
                if hasattr(e, "status")
                else console.print(f"[red]Error:[/] {escape(str(e))}")
            )
            raise typer.Exit(EXIT_ERROR)
        return

    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
        raise _fail(
            f"--reasoning-effort must be one of {', '.join(REASONING_EFFORTS)}; "
            f"got {reasoning_effort!r}",
            "Only generative engines reason; run `vlmrun gw s1 models` to see them.",
        )

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

    state_parts, images, document, video = _resolve_inputs(
        list(inputs or []), state is not None
    )

    if video is None:
        if fps is not None:
            raise _fail(
                "--fps applies to a video, and none was given.",
                "Pass a video file, or drop --fps.",
            )
        if gate_mode is not None and _parse_gate_mode(gate_mode)[0] != "mean":
            raise _fail(
                f"--gate-mode {gate_mode!r} describes a timeline, and this is a "
                "single read.",
                "Gate modes apply to a video; without one a gate reads the "
                "single value (or, with --repeat, the mean).",
            )
        if ws:
            raise _fail(
                "--ws reads a video's frames over one session, and no video was given.",
                "Pass a video file, or drop --ws.",
            )
        frame_fps = 0.0
        gate_over_frames = ("mean", 1)
    else:
        # Each frame is its own read, so --repeat would multiply an already
        # per-frame bill and the two spreads would be indistinguishable.
        if repeat > 1:
            raise _fail(
                "--repeat and a video do not combine.",
                "A video is already many reads; --fps sets how many. To measure "
                "read-to-read spread, --repeat a single frame.",
            )
        frame_fps = DEFAULT_VIDEO_FPS if fps is None else fps
        if frame_fps <= 0:
            raise _fail(f"--fps must be positive; got {frame_fps:g}")
        gate_over_frames = _parse_gate_mode(gate_mode or "any")
    resolved_state: Any = request_body.pop("state", None)
    if state is not None:
        resolved_state = _read_arg(state)
    elif len(state_parts) == 1:
        resolved_state = state_parts[0]
    elif state_parts:
        resolved_state = " ".join(str(part) for part in state_parts)

    if resolved_state is None:
        if not images and not document and video is None:
            raise _fail(
                "no state was given.",
                "Pass text, a text file, -s @file, or at least one image, PDF or video.",
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
        "content": bool(images or document or video is not None),
        "reasoning_effort": reasoning_effort is not None,
    }
    extra_body = {k: v for k, v in request_body.items() if not overridden.get(k, False)}
    gates = [_parse_gate(expression) for expression in (gate or [])]
    if gates:
        _validate_gates(gates, question_spec)
    system_one = client.gateway.systemone

    if dry_run:
        # For a video, the body of the first frame's read: every frame's request
        # is this one with a different image.
        dry_images = images
        if video is not None:
            dry_images = [_first_frame_image(video)]
        try:
            _print_body(
                system_one.build_request(
                    resolved_state,
                    question_spec,
                    model=model or request_body.get("model"),
                    images=dry_images,
                    document=document,
                    detail=detail,  # type: ignore[arg-type]
                    steps=steps,
                    samples=samples,
                    reasoning_effort=reasoning_effort,
                    extra_body=extra_body or None,
                )
            )
        except InputError as e:
            raise _fail(str(e.message), e.suggestion)
        return

    duration_s = native_fps = 0.0
    if video is not None:
        duration_s, native_fps = _video_scope(video, frame_fps, max_frames)

    common = dict(
        model=model or request_body.get("model"),
        detail=detail,
        steps=steps,
        samples=samples,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        extra_body=extra_body or None,
    )

    decisions: List[FrameDecision] = []
    runs = []
    start = time.time()
    try:
        if video is not None:
            from vlmrun.common.video import VideoReader

            estimate = int(duration_s * frame_fps) if duration_s > 0 else None
            with (
                system_one.stream(
                    question_spec,
                    state=resolved_state,
                    concurrency=concurrency,
                    transport="ws" if ws else "http",
                    **common,  # type: ignore[arg-type]
                ) as stream,
                VideoReader(Path(video).expanduser()) as reader,
            ):
                decisions = _collect_stream(
                    stream.map(reader.frames(frame_fps, max_frames=max_frames)),
                    total=min(estimate, max_frames) if estimate else None,
                    show_progress=not output_json and console.is_terminal,
                )
            runs = [decision.response for decision in decisions]
            if not runs:
                raise _fail(
                    f"no frames were read from {video}.",
                    "The file may be empty or its codec unsupported by this "
                    "OpenCV build.",
                )
            if len(runs) >= max_frames:
                # Reaching the ceiling means the tail was never looked at, and a
                # partial timeline must not be presented as a whole one. Said
                # whether or not the duration was known, because the estimate is
                # an estimate.
                console.print(
                    f"[yellow]Stopped at the {max_frames}-frame ceiling; the rest "
                    f"of the video was not read.[/yellow] "
                    f"[dim]Raise --max-frames to go further.[/dim]"
                )
        else:
            for _ in range(repeat):
                runs.append(
                    system_one.decide(
                        resolved_state,
                        question_spec,
                        images=images,
                        document=document,
                        **common,  # type: ignore[arg-type]
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
        if video is not None:
            console.print_json(
                _frames_json(
                    decisions,
                    runs,
                    fps=frame_fps,
                    duration_s=duration_s,
                    native_fps=native_fps,
                )
            )
        else:
            _print_json(runs if repeat > 1 else runs[0])
        Console(stderr=True).print(_timings_json(runs, latency_s), soft_wrap=True)
    elif video is not None:
        _render_timeline(
            folded, runs, decisions, latency_s, fps=frame_fps, duration_s=duration_s
        )
    elif repeat > 1:
        _render_repeat(folded, runs, latency_s)
    else:
        _render(runs[0], latency_s)

    if gates:
        mode, window = gate_over_frames
        if video is not None and mode != "mean":
            results = _evaluate_gates_over_frames(
                gates, [_aggregate([run]) for run in runs], decisions, mode, window
            )
        else:
            results = _evaluate_gates(gates, folded)
        _render_gates(results, err=output_json)
        if not all(passed for passed, _, _ in results):
            raise typer.Exit(EXIT_GATE_FAILED)
