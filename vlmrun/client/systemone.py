"""System One — typed, calibrated decisions served on the VLM Run gateway.

``POST {gateway}/typesafe/v1/systemone`` answers named questions about text,
JSON, images and PDFs by reading the model's own distribution over each answer
slot in a single denoise step: nothing is generated and nothing is parsed, so
an answer can never be off-schema.

The route is wire-compatible with TypeSafe's Jev API, so this resource drives
the official :mod:`typesafe_sdk` client rather than re-implementing the
contract — the same way :class:`~vlmrun.client.gateway.Gateway` drives the
OpenAI SDK for chat completions. ``typesafe-sdk`` is an optional dependency::

    pip install vlmrun[typesafe]

Media, ``steps`` and ``samples`` are gateway extensions the TypeSafe SDK does
not model, so they ride its ``extra_body`` passthrough as ``content`` — the
OpenAI content parts a caller already sends to chat completions, read after the
state.

Example:
    ```python
    from vlmrun import VLMRun

    client = VLMRun()
    result = client.gateway.systemone.decide(
        state="Invoice #44 was charged twice, I need this fixed today",
        questions=[
            {"id": "is_urgent", "type": "noul", "instructions": "Is this time-sensitive?"},
            {"id": "department", "type": "choice", "options": ["billing", "technical", "sales"]},
        ],
    )
    result.nouls["is_urgent"].noul          # 0.91
    result.choices["department"].choice     # "billing"
    ```

    A series of inputs — a video's frames, a camera, a queue — goes through a
    :class:`DecisionStream`, which overlaps the reads and hands them back in
    order:

    ```python
    from vlmrun.common.video import VideoReader

    with client.gateway.systemone.stream(
        questions=[{"id": "is_open", "type": "noul"}],
        state="Is the door open?",
    ) as stream, VideoReader("door.mp4") as video:
        for decision in stream.map(video.frames(fps=2)):
            print(decision.timestamp_s, decision.response.nouls["is_open"].noul)
    ```
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import os
import re
import asyncio
import itertools
import socket
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    Sequence,
)
from urllib.parse import urlparse

from pydantic import BaseModel

from vlmrun.client.exceptions import APIError, InputError
from vlmrun.common.dependencies import require_typesafe, require_websockets
from vlmrun.common.mime import (
    IMAGE_MIME_TYPES,
    data_url,
    is_http_url,
    normalize_mime,
    sniff_mime,
)
from vlmrun.constants import TYPESAFE_BASE_URL_ENV, gateway_base_url
from vlmrun.types.abstract import VLMRunProtocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from typesafe_sdk import ListModelsResponse, SystemOneResponse

# The route's default engine. Several models are served — ``models()`` lists the
# set this gateway actually has, and there is no catch-all alias (``jev-latest``
# was removed when the route gained a second read strategy).
SYSTEMONE_MODEL = "google/diffusiongemma-26b-a4b-it"

# The websocket route, mounted beside the HTTP one under the same /typesafe root.
TYPESAFE_WEBSOCKET_PATH = "/ws"

# The gateway mounts TypeSafe's paths under this prefix, so the SDK's own
# /v1/systemone and /v1/models suffixes resolve with no mapping.
TYPESAFE_PREFIX = "typesafe"

# Images per read; a PDF's rasterised pages share this budget.
MAX_IMAGES = 8

MAX_IMAGE_BYTES = 5 * 1024 * 1024
# Only a PDF's first pages are read — enough to classify, route or gate one.
MAX_PDF_PAGES = 8

DEFAULT_READ_TIMEOUT = 120.0

MAX_CHOICE_OPTIONS = 128
MAX_SCORE_LEVELS = 10
MIN_CHOICE_OPTIONS = 2
MIN_SCORE_LEVELS = 2

QUESTION_TYPES = ("noul", "choice", "score")

# OpenAI's three values. `high` reads at 280 vision tokens, `auto` and `low` at
# 70. The budget is per request: one `high` input lifts them all.
# OpenAI's knob, as this route spells it. Only a generative engine can think
# before it answers; a diffusion engine seeds the answer template into a canvas
# and refuses the field.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")

ImageDetail = Literal["auto", "low", "high"]

# Friendly spellings accepted in place of the wire's ``criteria``.
_CRITERIA_ALIAS = {"choice": "options", "score": "levels"}
_NOUL_OUTCOME_ALIAS = {"yes": "true", "no": "false"}
_COMMON_KEYS = frozenset({"id", "type", "instructions", "criteria"})


def typesafe_base_url(gateway_url: str | None = None) -> str:
    """TypeSafe base URL derived from a gateway URL.

    ``https://gateway.vlm.run/v1`` -> ``https://gateway.vlm.run/typesafe``, which
    is what ``TYPESAFE_BASE_URL`` would be set to for the official SDK.

    Args:
        gateway_url: Gateway base URL; defaults to the SDK's gateway default.

    Returns:
        The ``/typesafe`` root, without a trailing slash.
    """
    root = gateway_base_url(gateway_url)
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root.rstrip('/')}/{TYPESAFE_PREFIX}"


def typesafe_websocket_url(
    base_url: str | None = None, *, gateway_url: str | None = None
) -> str:
    """The ``wss://`` URL of the session route, derived from the ``/typesafe`` root.

    The socket route is mounted beside the HTTP one under the same root, so its
    URL is the typesafe base URL with the scheme swapped and
    :data:`TYPESAFE_WEBSOCKET_PATH` appended. Derived rather than configured
    separately, so pointing the SDK at another deployment moves both.

    Args:
        base_url (str | None, optional): An explicit ``/typesafe`` root. Falls
            back to ``TYPESAFE_BASE_URL``, then to the root derived from the
            gateway URL.
        gateway_url (str | None, optional): Gateway URL the route is mounted
            beside, when ``base_url`` is not given.

    Returns:
        str: e.g. ``wss://gateway.vlm.run/typesafe/ws``.

    Example:
        ```python
        from vlmrun.client.systemone import typesafe_websocket_url

        typesafe_websocket_url()                                  # the default gateway
        typesafe_websocket_url(gateway_url="http://localhost:8000/v1")
        ```
    """
    root = (
        base_url or os.getenv(TYPESAFE_BASE_URL_ENV) or typesafe_base_url(gateway_url)
    ).rstrip("/")
    scheme, separator, rest = root.partition("://")
    if not separator:  # a bare host, with no scheme to swap
        return f"wss://{root}{TYPESAFE_WEBSOCKET_PATH}"
    return f"{'wss' if scheme == 'https' else 'ws'}://{rest}{TYPESAFE_WEBSOCKET_PATH}"


def _spec_error(message: str, *, suggestion: str) -> InputError:
    return InputError(
        message=message, suggestion=suggestion, error_type="question_spec"
    )


def _normalize_choice_criteria(raw: Any, where: str) -> dict[str, Any]:
    """Options as a list of labels/objects or a label->description mapping."""
    if isinstance(raw, Mapping):
        criteria = {str(k): v for k, v in raw.items()}
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        criteria = {}
        for index, option in enumerate(raw):
            if isinstance(option, str):
                criteria[option] = None
            elif isinstance(option, Mapping) and "name" in option:
                criteria[str(option["name"])] = option.get("description")
            else:
                raise _spec_error(
                    f"{where}: option {index} must be a label string or "
                    '{"name": ..., "description": ...}',
                    suggestion='Write options as ["a", "b"] or [{"name": "a", "description": "..."}]',
                )
    else:
        raise _spec_error(
            f"{where}: options must be a list of labels or a mapping of label to description",
            suggestion='e.g. "options": ["billing", "technical", "sales"]',
        )
    if not MIN_CHOICE_OPTIONS <= len(criteria) <= MAX_CHOICE_OPTIONS:
        raise _spec_error(
            f"{where}: a choice needs {MIN_CHOICE_OPTIONS}-{MAX_CHOICE_OPTIONS} "
            f"options; got {len(criteria)}",
            suggestion="Use a noul question for a yes/no decision.",
        )
    return criteria


def _normalize_score_criteria(raw: Any, where: str) -> list[Any]:
    """Levels as an ordered list of descriptions, lowest first."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise _spec_error(
            f"{where}: levels must be an ordered list of level descriptions",
            suggestion='e.g. "levels": ["Calm", "Frustrated", "Very angry"]',
        )
    levels = list(raw)
    if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
        raise _spec_error(
            f"{where}: a score needs {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels; "
            f"got {len(levels)}",
            suggestion="Collapse adjacent levels, or use a choice question.",
        )
    return levels


def _normalize_noul_criteria(raw: Any, where: str) -> dict[str, Any]:
    """Outcome descriptions, accepting ``yes``/``no`` for the wire's ``true``/``false``."""
    if not isinstance(raw, Mapping):
        raise _spec_error(
            f"{where}: criteria must be an object describing the true and false outcomes",
            suggestion='e.g. "criteria": {"true": "Needs action today", "false": "Can wait"}',
        )
    criteria: dict[str, Any] = {}
    for key, value in raw.items():
        name = _NOUL_OUTCOME_ALIAS.get(str(key).lower(), str(key).lower())
        if name not in ("true", "false"):
            raise _spec_error(
                f"{where}: unknown noul outcome {key!r}",
                suggestion='A noul describes only "true"/"yes" and "false"/"no".',
            )
        criteria[name] = value
    return criteria


def _normalize_question(raw: Any, name: str, where: str) -> dict[str, Any]:
    """One question, in either dialect, as the wire object the API expects."""
    if hasattr(raw, "model_dump"):  # a typesafe_sdk Noul/Choice/Score object
        return raw
    if not isinstance(raw, Mapping):
        raise _spec_error(
            f"{where}: a question must be an object", suggestion='e.g. {"type": "noul"}'
        )

    kind = raw.get("type")
    if kind not in QUESTION_TYPES:
        raise _spec_error(
            f"{where}: type must be one of {', '.join(QUESTION_TYPES)}; got {kind!r}",
            suggestion="noul is yes/no, choice picks a label, score reads an ordered rubric.",
        )

    alias = _CRITERIA_ALIAS.get(kind)
    allowed = _COMMON_KEYS | ({alias} if alias else set())
    unknown = set(raw) - allowed
    if unknown:
        wrong_alias = unknown & set(_CRITERIA_ALIAS.values())
        suggestion = (
            f'A {kind} question uses "{alias or "criteria"}".'
            if wrong_alias
            else f"Allowed keys: {', '.join(sorted(allowed))}."
        )
        raise _spec_error(
            f"{where}: unknown key(s) {', '.join(sorted(repr(k) for k in unknown))}",
            suggestion=suggestion,
        )

    question: dict[str, Any] = {"type": kind}
    if raw.get("instructions") is not None:
        question["instructions"] = raw["instructions"]

    criteria = raw.get("criteria")
    if alias and raw.get(alias) is not None:
        if criteria is not None:
            raise _spec_error(
                f'{where}: set either "{alias}" or "criteria", not both',
                suggestion=f'"{alias}" is the friendly spelling of "criteria".',
            )
        criteria = raw[alias]

    if kind == "choice":
        if criteria is None:
            raise _spec_error(
                f"{where}: a choice question needs options",
                suggestion='e.g. "options": ["billing", "technical", "sales"]',
            )
        question["criteria"] = _normalize_choice_criteria(criteria, where)
    elif kind == "score":
        if criteria is None:
            raise _spec_error(
                f"{where}: a score question needs levels",
                suggestion='e.g. "levels": ["Calm", "Frustrated", "Very angry"]',
            )
        question["criteria"] = _normalize_score_criteria(criteria, where)
    elif criteria is not None:
        question["criteria"] = _normalize_noul_criteria(criteria, where)

    return question


def normalize_questions(spec: Any) -> dict[str, Any]:
    """Normalize either question dialect into the wire mapping.

    Accepts a list of questions each carrying an ``id``, the wire mapping of id
    to question, either of those wrapped in ``{"questions": ...}``, and
    ``typesafe_sdk`` question objects. ``options`` (choice) and ``levels``
    (score) are accepted as the friendly spellings of ``criteria``.

    Args:
        spec: Questions in any accepted form.

    Returns:
        Mapping of question id to wire question object, in the given order.

    Raises:
        InputError: The spec is empty, malformed, or uses an unknown key.
    """
    if isinstance(spec, Mapping) and set(spec) == {"questions"}:
        spec = spec["questions"]

    questions: dict[str, Any] = {}
    if isinstance(spec, Mapping):
        for name, raw in spec.items():
            where = f"questions.{name}"
            if isinstance(raw, Mapping) and "id" in raw:
                if str(raw["id"]) != str(name):
                    raise _spec_error(
                        f'{where}: "id" is {raw["id"]!r} but the key is {name!r}',
                        suggestion="In the mapping form the key is the id; drop the id field.",
                    )
                raw = {k: v for k, v in raw.items() if k != "id"}
            questions[str(name)] = _normalize_question(raw, str(name), where)
    elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        for index, raw in enumerate(spec):
            where = f"questions[{index}]"
            if not isinstance(raw, Mapping) or not raw.get("id"):
                raise _spec_error(
                    f'{where}: every question in a list needs an "id"',
                    suggestion="The id is the key its answer comes back under.",
                )
            name = str(raw["id"])
            if name in questions:
                raise _spec_error(
                    f"{where}: duplicate question id {name!r}",
                    suggestion="Answers are keyed by id, so each must be unique.",
                )
            body = {k: v for k, v in raw.items() if k != "id"}
            questions[name] = _normalize_question(body, name, f'{where} "{name}"')
    else:
        raise _spec_error(
            "questions must be a list of questions or a mapping of id to question",
            suggestion='e.g. [{"id": "is_urgent", "type": "noul"}]',
        )

    if not questions:
        raise _spec_error(
            "no questions were given", suggestion="Ask at least one question."
        )
    return questions


MAX_REDIRECTS = 5

# Opt out of the private-address guard for a deployment whose images live on an
# internal host. Named for the SDK, not the route, because it governs our fetch.
ALLOW_PRIVATE_URLS_ENV = "VLMRUN_ALLOW_PRIVATE_URLS"


def _guard_fetchable(url: str) -> None:
    """Refuse to fetch a URL that resolves to a non-public address.

    We fetch image URLs ourselves, so a caller who does not control the URL —
    an agent acting on model output, a service taking one from a request — can
    otherwise aim this process at loopback, a private range, or a cloud
    metadata endpoint. Every resolved address must be public; set
    ``VLMRUN_ALLOW_PRIVATE_URLS=1`` for an internal image host.

    Args:
        url: The http(s) URL about to be fetched.

    Raises:
        InputError: The host is missing, does not resolve, or resolves to a
            loopback, link-local, private, reserved or multicast address.
    """
    if os.getenv(ALLOW_PRIVATE_URLS_ENV):
        return
    host = urlparse(url).hostname
    if not host:
        raise InputError(
            message=f"{url!r} has no host to fetch from",
            suggestion="Pass an absolute http(s) URL.",
        )
    try:
        resolved = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise InputError(
            message=f"cannot resolve {host!r}",
            suggestion="Check the URL, or pass the image as a local path.",
        ) from e
    for info in resolved:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_loopback
            or address.is_link_local
            or address.is_private
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise InputError(
                message=f"refusing to fetch {url!r}: {host} resolves to the non-public address {address}",
                suggestion=f"Set {ALLOW_PRIVATE_URLS_ENV}=1 to allow internal hosts, "
                "or pass the image as a local path.",
            )


@contextmanager
def _guarded_get(url: str, timeout: float) -> Any:
    """GET a URL with every redirect hop guarded; yields the streamed response.

    Redirects are followed by hand because a public URL that redirects to a
    metadata address would otherwise slip past a check made only on the URL the
    caller passed.

    Args:
        url: The http(s) URL to fetch.
        timeout: Per-request timeout in seconds.

    Yields:
        The streamed ``requests.Response`` for the final hop.

    Raises:
        InputError: The URL is not fetchable or redirects too many times.
    """
    import requests

    target = url
    for _ in range(MAX_REDIRECTS + 1):
        _guard_fetchable(target)
        response = requests.get(
            target, timeout=timeout, stream=True, allow_redirects=False
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            response.close()
            if not location:
                raise InputError(
                    message=f"{target!r} redirected without a location",
                    suggestion="Pass the file as a local path.",
                )
            target = requests.compat.urljoin(target, location)
            continue
        try:
            response.raise_for_status()
            yield response
        finally:
            response.close()
        return
    raise InputError(
        message=f"{url!r} redirected more than {MAX_REDIRECTS} times",
        suggestion="Pass the file as a local path.",
    )


def probe_url_mime(url: str, timeout: float = 15.0) -> str | None:
    """What a URL actually serves, from its first bytes.

    A URL often carries no usable extension — a signed link, an object-store key,
    an endpoint with the name in a query string — so the media type is read off
    the response instead. Magic bytes decide; ``content-type`` is the fallback,
    since servers mislabel freely.

    Args:
        url: The http(s) URL to probe.
        timeout: Per-request timeout in seconds.

    Returns:
        A normalized MIME type, or None when neither source identifies it.

    Raises:
        InputError: The URL is not fetchable.
    """
    with _guarded_get(url, timeout) as response:
        head = next(response.iter_content(64), b"")
        sniffed = sniff_mime(head)
        if sniffed:
            return normalize_mime(sniffed)
        declared = (response.headers.get("content-type") or "").split(";")[0].strip()
    return normalize_mime(declared) if declared else None


def _fetch_image(url: str, timeout: float) -> bytes:
    """Fetch an image, bounded in size, guarding every redirect hop.

    The body is read in chunks and abandoned the moment it exceeds the cap, so an
    oversized response costs a few chunks rather than its full length in memory.

    Args:
        url: Image URL to fetch.
        timeout: Per-request timeout in seconds.

    Returns:
        The image bytes.

    Raises:
        InputError: The URL is not fetchable or the body exceeds
            :data:`MAX_IMAGE_BYTES`.
    """

    def too_large() -> InputError:
        return InputError(
            message=f"image at {url!r} is larger than the {MAX_IMAGE_BYTES} byte limit",
            suggestion="Downscale the image before sending it.",
        )

    with _guarded_get(url, timeout) as response:
        # Content-Length is optional and untrusted; it only lets us refuse early.
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_IMAGE_BYTES:
            raise too_large()
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(64 * 1024):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise too_large()
            chunks.append(chunk)
        return b"".join(chunks)


def _decode_data_url(raw: str) -> bytes:
    """The bytes carried by a base64 ``data:`` URL."""
    header, _, payload = raw.partition(",")
    if ";base64" not in header:
        raise InputError(
            message="an image data URL must be base64-encoded",
            suggestion="Use data:image/...;base64,<payload>.",
        )
    try:
        return base64.b64decode(re.sub(r"\s+", "", payload), validate=True)
    except (binascii.Error, ValueError) as e:
        raise InputError(
            message="image data is not valid base64",
            suggestion="Re-encode the image, or pass it as a local path.",
        ) from e


def _read_image(source: str, timeout: float) -> bytes:
    """The bytes of one image, from a data URL, an http(s) URL or a local path."""
    if source.startswith("data:"):
        return _decode_data_url(source)
    if is_http_url(source):
        return _fetch_image(source, timeout)
    path = Path(source).expanduser()
    if not path.is_file():
        raise InputError(
            message=f"image {source!r} is not a file",
            suggestion="Pass a local path, an http(s) URL, or a data: URL.",
        )
    # Size first: reading a multi-gigabyte file to measure it is the failure
    # mode the limit exists to prevent.
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise InputError(
            message=f"image {source!r} is {path.stat().st_size} bytes; "
            f"the limit is {MAX_IMAGE_BYTES}",
            suggestion="Downscale the image before sending it.",
        )
    return path.read_bytes()


def _image_part(
    source: str | Path, detail: ImageDetail, *, timeout: float
) -> dict[str, Any]:
    """One ``image_url`` content part, always inlined as a data URL.

    The route accepts images only as ``data:image/...;base64,``, so a remote
    image is fetched here rather than handed to the gateway. Every source —
    data URL, http(s) URL or path — is validated the same way: decoded bytes
    against the size cap, and magic bytes against the supported types, since the
    route trusts the media type we declare.
    """
    raw_source = str(source)
    content = _read_image(raw_source, timeout)
    if len(content) > MAX_IMAGE_BYTES:
        raise InputError(
            message=f"image {raw_source!r} is {len(content)} bytes; the limit is {MAX_IMAGE_BYTES}",
            suggestion="Downscale the image before sending it.",
        )
    sniffed = sniff_mime(content[:16])
    if sniffed is None:
        raise InputError(
            message=f"{raw_source} is not a recognized image",
            suggestion="This route reads JPEG, PNG, WebP and GIF.",
        )
    mime = normalize_mime(sniffed)
    if mime not in IMAGE_MIME_TYPES:
        raise InputError(
            message=f"image type {mime!r} is not supported ({raw_source})",
            suggestion="This route reads JPEG, PNG, WebP and GIF.",
        )
    return {
        "type": "image_url",
        "image_url": {"url": data_url(content, mime), "detail": detail},
    }


def _file_part(source: str | Path, detail: ImageDetail) -> dict[str, Any]:
    """The one ``file`` content part: a PDF, inline or by URL.

    A remote PDF rides as its URL so the gateway fetches it; a local one is
    inlined, which skips the fetch entirely.
    """
    raw_source = str(source)
    if is_http_url(raw_source) or raw_source.startswith("data:"):
        return {
            "type": "file",
            "file": {"file_data": raw_source, "detail": detail},
        }
    path = Path(raw_source).expanduser()
    if not path.is_file():
        raise InputError(
            message=f"document {raw_source!r} is not a file",
            suggestion="Pass a local .pdf path or an http(s) URL.",
        )
    content = path.read_bytes()
    if not content.startswith(b"%PDF"):
        raise InputError(
            message=f"document {raw_source!r} is not a PDF",
            suggestion="This route reads one PDF; convert other formats first.",
        )
    return {
        "type": "file",
        "file": {
            "filename": path.name,
            "file_data": data_url(content, "application/pdf"),
            "detail": detail,
        },
    }


def build_content(
    *,
    images: Sequence[str | Path] | None = None,
    document: str | Path | None = None,
    text: str | Sequence[str] | None = None,
    detail: ImageDetail = "auto",
    timeout: float = 30.0,
) -> list[dict[str, Any]] | None:
    """Assemble the ``content`` parts that carry a read's media.

    These are the OpenAI content parts a caller already sends to
    ``/v1/openai/chat/completions`` — no role and no turn, because a read
    denoises one canvas over one state. Media leads, the state follows.

    Args:
        images: Image paths, http(s) URLs or data URLs (at most 8).
        document: One PDF path or http(s) URL; its first pages are read.
        text: Extra text parts, read after the state.
        detail: Vision budget applied to every part of this request.
        timeout: Seconds allowed for fetching a remote image.

    Returns:
        The ``content`` list, or None when there is no media.

    Raises:
        InputError: Too many images, an unsupported type, or a missing file.
    """
    parts: list[dict[str, Any]] = []
    image_list = list(images or [])
    if len(image_list) > MAX_IMAGES:
        raise InputError(
            message=f"{len(image_list)} images; the limit is {MAX_IMAGES}",
            suggestion="Split the inputs across requests.",
        )
    if document is not None:
        parts.append(_file_part(document, detail))
    parts.extend(_image_part(image, detail, timeout=timeout) for image in image_list)
    if isinstance(text, str):
        text = [text]
    parts.extend({"type": "text", "text": item} for item in (text or []) if item)

    return parts or None


class RequestTimings(BaseModel):
    """Where a read's wall-clock time went.

    ``api_ms`` is measured around the HTTP call itself, so it excludes the
    local work of encoding media and normalizing questions (``prep_ms``).
    ``ttfb_ms`` is the wait for the response headers — the read's own latency,
    since a decision is not streamed and its body is a few hundred bytes.
    """

    prep_ms: float = 0.0
    ttfb_ms: float | None = None
    api_ms: float | None = None
    total_ms: float = 0.0

    @property
    def transfer_ms(self) -> float | None:
        """Time spent reading the body after the headers arrived."""
        if self.api_ms is None or self.ttfb_ms is None:
            return None
        return max(0.0, self.api_ms - self.ttfb_ms)


def usage_of(response: Any) -> dict[str, Any]:
    """The response's ``usage`` object, including fields the SDK does not model.

    ``typesafe_sdk.Usage`` carries only Jev's ``input_tokens``/``output_tokens``
    and ignores the rest, so the gateway's additions — ``cost``, ``reads`` and
    the ``input_tokens_details`` split — are read back off the raw body. A
    websocket read has no HTTP body, so its session stashes the usage frame
    instead; either way the caller gets the same mapping.

    Args:
        response: A response returned by :meth:`SystemOne.decide`.

    Returns:
        The usage mapping, or the typed fields alone when the raw body is gone.
    """
    raw = getattr(response, "__dict__", {}).get("_raw")
    if raw is not None:
        try:
            return json.loads(raw.text).get("usage") or {}
        except (ValueError, AttributeError):
            pass
    stashed = getattr(response, "__dict__", {}).get("_vlmrun_usage")
    if stashed:
        return dict(stashed)
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    dump = getattr(usage, "model_dump", None)
    return dump() if dump is not None else dict(vars(usage))


def timings_of(response: Any) -> RequestTimings | None:
    """The :class:`RequestTimings` attached to a response by :meth:`SystemOne.decide`."""
    return getattr(response, "__dict__", {}).get("_vlmrun_timings")


def _timing_hooks() -> dict[str, list[Any]]:
    """httpx event hooks that stamp each request with its own timings.

    Stamps live on the request/response extensions rather than on the resource,
    so concurrent calls cannot read each other's numbers.
    """

    def on_request(request: Any) -> None:
        request.extensions["vlmrun_t0"] = time.perf_counter()

    def on_response(response: Any) -> None:
        started = response.request.extensions.get("vlmrun_t0")
        if started is not None:
            response.extensions["vlmrun_ttfb"] = (time.perf_counter() - started) * 1000

    return {"request": [on_request], "response": [on_response]}


# How many reads a stream keeps in flight when the caller does not say. Reads
# are independent, so this is purely about not opening an unbounded number of
# connections on a caller's behalf.
DEFAULT_STREAM_CONCURRENCY = 4


class FrameDecision(NamedTuple):
    """One read in a stream, with where its input came from.

    Attributes:
        index: Position in the stream, counting from 0. For a video this is the
            frame's index in the source, so it survives sampling.
        timestamp_s: Presentation time in seconds for a video frame, None for
            an input that has no place on a timeline.
        response: The read itself — the TypeSafe SDK's ``SystemOneResponse``.
    """

    index: int
    timestamp_s: float | None
    response: "SystemOneResponse"


class DecisionStream:
    """An open session that reads many decisions against one question spec.

    There is no streaming route on the gateway and no video model behind it: a
    decision is one request and one structured response. What is streamed is the
    *series* — inputs go out as they arrive, reads overlap, and results come back
    in input order, so a timeline can be consumed as it is produced rather than
    after the last frame.

    Held open because a stream owns things that need closing: a thread pool for
    the reads in flight, and, while iterating a video, the decoder. Use it as a
    context manager so both are released on the way out, including when the
    consumer stops early.

    The question spec is normalized once, when the stream is created, so a
    malformed question fails there rather than on the first frame.

    Example:
        ```python
        from vlmrun.common.video import VideoReader

        with client.gateway.systemone.stream(
            questions=[{"id": "is_open", "type": "noul"}],
            state="Is the door open?",
        ) as stream, VideoReader("door.mp4") as video:
            # Concurrent, ordered, lazy.
            for decision in stream.map(video.frames(fps=2)):
                print(decision.timestamp_s, decision.response.nouls["is_open"].noul)

            # Or one read at a time, the shape a live source wants.
            for frame in video.frames(fps=2):
                response = stream.send(frame)
        ```

    Attributes:
        count: Reads issued so far.
        is_open: Whether the stream can take reads right now.
    """

    def __init__(
        self,
        resource: "SystemOne",
        questions: Any,
        *,
        state: Any = "",
        model: str | None = None,
        detail: ImageDetail = "auto",
        steps: int | None = None,
        samples: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        concurrency: int = DEFAULT_STREAM_CONCURRENCY,
        quality: int = 85,
    ) -> None:
        """Open a decision stream.

        Args:
            resource: The :class:`SystemOne` resource the reads go through.
            questions: Questions in either dialect; normalized once here.
            state: State every read is about. Overridable per :meth:`send`.
            model: Model override for every read in this stream.
            detail: Vision budget per read.
            steps: Denoise steps per read.
            samples: Noise draws to average per read. Every draw is billed.
            reasoning_effort: Thinking budget before the answer.
            timeout: Per-read timeout in seconds.
            extra_body: Extra top-level body fields, merged last.
            concurrency: Reads in flight at once.
            quality: JPEG quality for frames encoded from arrays.

        Raises:
            InputError: If the questions are malformed, or concurrency is < 1.
        """
        if concurrency < 1:
            raise InputError(
                message=f"concurrency must be at least 1; got {concurrency}",
                suggestion="Pass 1 to read one at a time.",
            )
        self._resource = resource
        self._questions = normalize_questions(questions)
        self._state = state
        self._concurrency = concurrency
        self._quality = quality
        self._options: dict[str, Any] = {
            "model": model,
            "detail": detail,
            "steps": steps,
            "samples": samples,
            "reasoning_effort": reasoning_effort,
            "timeout": timeout,
            "extra_body": extra_body,
        }
        self._pool: ThreadPoolExecutor | None = None
        self._count = 0

    @property
    def count(self) -> int:
        """Reads issued on this stream so far."""
        return self._count

    @property
    def is_open(self) -> bool:
        """Whether the stream can take reads right now.

        False both before it is entered and after it is closed — a stream is
        usable only inside its ``with`` block.
        """
        return self._pool is not None

    @property
    def questions(self) -> dict[str, Any]:
        """The normalized question spec every read in this stream asks."""
        return dict(self._questions)

    def __enter__(self) -> "DecisionStream":
        """Start the worker pool.

        Returns:
            This stream.
        """
        self._pool = ThreadPoolExecutor(
            max_workers=self._concurrency, thread_name_prefix="vlmrun-systemone"
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the stream, dropping queued reads when leaving on an error."""
        self.close(cancel=exc_type is not None)

    def close(self, *, cancel: bool = False) -> None:
        """Release the worker pool.

        The resource's HTTP client is deliberately left alone: it is shared with
        every other read on the client, and a stream does not own it.

        Args:
            cancel: Drop reads that have not started instead of waiting for
                them. Used when the stream is abandoned rather than finished.
        """
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=not cancel, cancel_futures=cancel)

    def _check_open(self) -> None:
        """Raise unless the stream is open for reads.

        Raises:
            InputError: If the stream has not been entered, or was closed.
        """
        if not self.is_open:
            raise InputError(
                message="this decision stream is not open",
                suggestion="Use it as a context manager: `with resource.stream(...) as s:`.",
            )

    def _pool_or_raise(self) -> ThreadPoolExecutor:
        """The worker pool, or a pointed error when the stream is not open."""
        self._check_open()
        assert self._pool is not None  # narrowed by _check_open
        return self._pool

    def _submit(self, image: str | Path) -> Future:
        """Dispatch one read, returning a future for it.

        The seam between the two transports: this one hands the read to a worker
        pool, the websocket session pipelines it over one socket.
        """
        return self._pool_or_raise().submit(self._read, image)

    def _as_image(self, item: Any) -> str | Path:
        """Coerce one input into something a read accepts.

        Paths, URLs and data URLs go through untouched; an array is encoded
        here, and a :class:`~vlmrun.common.video.SampledFrame` contributes its
        frame.
        """
        # Strings and paths first: a str carries plenty of attributes that look
        # frame-like under getattr (``.index`` among them).
        if isinstance(item, (str, Path)):
            return item
        frame = getattr(item, "frame", None)
        from vlmrun.common.image import encode_frame

        return encode_frame(item if frame is None else frame, quality=self._quality)

    def _describe(
        self, item: Any, position: int
    ) -> tuple[int, float | None, str | Path]:
        """``(index, timestamp, image)`` for one input.

        A sampled frame keeps its own index and timestamp, so a timeline survives
        being mapped. Everything else is numbered by position: a string has an
        ``.index`` attribute of its own, and reading that as a frame number would
        be silently wrong rather than loudly wrong.
        """
        image = self._as_image(item)
        if isinstance(item, (str, Path)):
            return position, None, image
        index = getattr(item, "index", None)
        timestamp = getattr(item, "timestamp_s", None)
        return (
            index if isinstance(index, int) else position,
            timestamp if isinstance(timestamp, (int, float)) else None,
            image,
        )

    def send(self, image: Any, *, state: Any = None) -> "SystemOneResponse":
        """Read one input, blocking until the answer comes back.

        For a live source — a camera, a queue — where inputs are not known in
        advance. Use :meth:`map` or :meth:`video` to overlap reads.

        Args:
            image: A path, http(s) URL, data URL, RGB array, or ``SampledFrame``.
            state: State for this read only; defaults to the stream's.

        Returns:
            The read.

        Raises:
            InputError: If the image cannot be read or encoded.
            TypeSafeAPIError: If the gateway returns an unsuccessful response.
        """
        return self._read(self._as_image(image), state)

    def _read(self, image: str | Path, state: Any = None) -> "SystemOneResponse":
        """One read against this stream's fixed spec."""
        self._count += 1
        return self._resource.decide(
            self._state if state is None else state,
            self._questions,
            images=[image],
            **self._options,
        )

    def map(self, images: Iterable[Any]) -> Iterator[FrameDecision]:
        """Read many inputs lazily, overlapping the reads, yielding in input order.

        The concurrent counterpart to :meth:`send`. Inputs are consumed lazily
        and at most ``concurrency`` reads are in flight, so an iterator of a
        million frames costs bounded memory and stopping early stops the spend.
        Results are ordered by input, not by which answer arrived first.

        Returns an iterator: nothing is read until it is consumed, and it has
        to be consumed inside the ``with`` block.

        Pair it with :meth:`vlmrun.common.video.VideoReader.frames` to read a
        video::

            with VideoReader("door.mp4") as video:
                for decision in stream.map(video.frames(fps=2)):
                    ...

        Args:
            images: Paths, URLs, data URLs, arrays, or
                :class:`~vlmrun.common.video.SampledFrame` objects. A sampled
                frame contributes its own index and timestamp, so a timeline
                survives being mapped; anything else is numbered by position and
                carries no timestamp.

        Yields:
            FrameDecision: Each read, in input order.

        Raises:
            InputError: If the stream is not open, or an input cannot be read.
            TypeSafeAPIError: If the gateway returns an unsuccessful response.
        """

        def jobs() -> Iterator[tuple[int, float | None, str | Path]]:
            for position, item in enumerate(images):
                yield self._describe(item, position)

        return self._ordered(jobs())

    def _ordered(
        self, items: Iterable[tuple[int, float | None, str | Path]]
    ) -> Iterator[FrameDecision]:
        """Submit lazily and yield in submission order.

        A sliding window rather than ``Executor.map``: inputs are pulled only as
        slots free up, so memory does not scale with the length of the series,
        and a consumer that stops early cancels what has not started instead of
        paying for it.
        """
        self._check_open()
        pending: deque[tuple[int, float | None, Future]] = deque()
        try:
            for index, timestamp, image in items:
                while len(pending) >= self._concurrency:
                    done_index, done_at, future = pending.popleft()
                    yield FrameDecision(done_index, done_at, future.result())
                pending.append((index, timestamp, self._submit(image)))
            while pending:
                done_index, done_at, future = pending.popleft()
                yield FrameDecision(done_index, done_at, future.result())
        finally:
            # Abandoned mid-series. The window is what keeps the waste bounded —
            # only `concurrency` reads are ever submitted, and those have all
            # started, so this cancels nothing today. It is the backstop if the
            # window is ever allowed to run ahead of the workers.
            for _, _, future in pending:
                future.cancel()


# Seconds allowed for the upgrade and the session.created ack.
WEBSOCKET_OPEN_TIMEOUT = 30.0

# Seconds to wait for session.stats when closing before giving up on it.
WEBSOCKET_CLOSE_TIMEOUT = 5.0

# Most reads the route will hold in flight for one session. The server's own
# default is lower (2), so a session that wants concurrency has to ask for it;
# asking past this is a validation error, so the request is clamped here.
WEBSOCKET_MAX_INFLIGHT = 8


class WebSocketStream(DecisionStream):
    """A decision stream over the gateway's websocket session.

    Same surface as :class:`DecisionStream` — :meth:`send`, :meth:`map`, used as
    a context manager — over a different transport, so nothing above it changes.
    What changes underneath:

    * The question spec is sent once, in ``session.create``, instead of riding
      every request. The server reports it back as cached prompt tokens, so a
      long series is materially cheaper per read.
    * Reads are pipelined over one socket rather than run on parallel
      connections. ``concurrency`` is requested as the session's ``max_inflight``
      — the route's own default is lower than this SDK's, so a session that wants
      concurrency has to ask — and the ack's value is then authoritative, so the
      stream never outruns what the server actually granted.
    * The state is session-scoped. It is sent once when the session opens;
      changing it later is a barrier (see :meth:`send`).

    Because a socket is one ordered channel, reads are correlated by an id
    echoed on each ``decision`` frame rather than by arrival order.

    Attributes:
        limits: The server's ``session.created`` ack — its declared ceilings.
        stats: The ``session.stats`` frame, once the session has been closed.
    """

    def __init__(self, resource: "SystemOne", questions: Any, **kwargs: Any) -> None:
        """Open a websocket-backed decision stream.

        Args:
            resource: The :class:`SystemOne` resource supplying the URL and key.
            questions: Questions in either dialect; normalized once here.
            **kwargs: As :class:`DecisionStream`.

        Raises:
            InputError: If the questions are malformed or concurrency is < 1.
        """
        super().__init__(resource, questions, **kwargs)
        self._ws: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._reader: Any = None
        self._pending: dict[str, Any] = {}
        self._ids = itertools.count()
        self._inflight: Any = None
        self._limits: dict[str, Any] = {}
        self._stats: dict[str, Any] = {}
        self._sent_state: Any = None
        self._window = self._concurrency

    @property
    def url(self) -> str:
        """The ``wss://`` URL this stream connects to.

        Derived from the resource's ``/typesafe`` root, so a resource pointed at
        another deployment takes its socket along.
        """
        return typesafe_websocket_url(self._resource.base_url)

    @property
    def limits(self) -> dict[str, Any]:
        """The server's declared ceilings, from ``session.created``."""
        return dict(self._limits)

    @property
    def stats(self) -> dict[str, Any]:
        """The session's ``session.stats`` frame, available after closing."""
        return dict(self._stats)

    @property
    def is_open(self) -> bool:
        """Whether the session is connected and able to take reads."""
        return self._ws is not None

    def __enter__(self) -> "WebSocketStream":
        """Connect, create the session, and start reading frames.

        Returns:
            This stream.

        Raises:
            DependencyError: If ``websockets`` is not installed.
            InputError: If the server refuses the session.
        """
        websockets = require_websockets()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="vlmrun-systemone-ws",
            daemon=True,
        )
        self._thread.start()
        try:
            self._await(self._open(websockets), timeout=WEBSOCKET_OPEN_TIMEOUT)
        except BaseException:
            self._teardown()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the session and stop the event loop."""
        self.close(cancel=exc_type is not None)

    def _await(self, coro: Any, *, timeout: float | None = None) -> Any:
        """Run a coroutine on the session's loop and wait for it here."""
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _open(self, websockets: Any) -> None:
        """Upgrade, send ``session.create``, and accept the ack."""
        self._ws = await websockets.connect(
            self.url,
            additional_headers={"authorization": f"Bearer {self._resource._api_key()}"},
            open_timeout=WEBSOCKET_OPEN_TIMEOUT,
            # Frames carry base64 images; the server states its own cap.
            max_size=None,
        )
        await self._ws.send(
            json.dumps(
                {
                    "type": "session.create",
                    "model": self._options["model"] or self._resource.model,
                    "questions": self._questions,
                    # Ask for the concurrency the caller wanted. The route's own
                    # default is 2, so not asking silently throttles the session.
                    "max_inflight": min(self._concurrency, WEBSOCKET_MAX_INFLIGHT),
                }
            )
        )
        raw = await asyncio.wait_for(self._ws.recv(), timeout=WEBSOCKET_OPEN_TIMEOUT)
        ack = json.loads(raw)
        if ack.get("type") != "session.created":
            raise InputError(
                message=f"the gateway refused the session: {ack.get('message', ack)}",
                suggestion="Check the model and question spec with `vlmrun gw s1 models`.",
            )
        self._limits = ack
        # The ack is authoritative: the server may grant less than was asked.
        ceiling = ack.get("max_inflight") or self._concurrency
        self._window = max(1, min(self._concurrency, int(ceiling)))
        self._inflight = asyncio.Semaphore(self._window)
        self._reader = asyncio.create_task(self._pump())
        if self._state not in (None, ""):
            await self._send_state(self._state)

    async def _send_state(self, state: Any) -> None:
        """Set the session's state."""
        await self._ws.send(json.dumps({"type": "state", "state": state}))
        self._sent_state = state

    async def _pump(self) -> None:
        """Dispatch server frames to whoever is waiting for them."""
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "decision":
                    self._settle(message.get("id"), message)
                elif kind == "session.stats":
                    self._stats = message
                elif kind == "error":
                    # The server closes the socket after an error, so it ends
                    # every read in flight, not only an attributable one.
                    self._abort(
                        APIError(
                            message=str(message.get("message", message)),
                            suggestion="Check the request against `vlmrun gw s1 --help`.",
                        ),
                        only=message.get("id"),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - surfaced to every waiting read
            self._abort(e)

    def _settle(self, request_id: Any, message: dict[str, Any]) -> None:
        """Hand one decision to the read that asked for it."""
        future = self._pending.pop(str(request_id), None)
        if future is not None and not future.done():
            future.set_result(message)

    def _abort(self, error: BaseException, *, only: Any = None) -> None:
        """Fail one waiting read, or all of them."""
        if only is not None and str(only) in self._pending:
            future = self._pending.pop(str(only))
            if not future.done():
                future.set_exception(error)
            return
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    def _submit(self, image: str | Path) -> Future:
        """Pipeline one read over the socket."""
        self._check_open()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(self._decide(image), self._loop)

    def _read(self, image: str | Path, state: Any = None) -> "SystemOneResponse":
        """One read, blocking until its decision comes back."""
        self._check_open()
        return self._await(self._decide(image, state))

    async def _decide(
        self, image: str | Path, state: Any = None
    ) -> "SystemOneResponse":
        """Send one ``decide`` and await the matching ``decision``."""
        if state is not None and state != self._sent_state:
            await self._barrier(state)
        async with self._inflight:
            request_id = str(next(self._ids))
            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
            message: dict[str, Any] = {"type": "decide", "id": request_id}
            content = build_content(
                images=[image],
                detail=self._options["detail"],
                timeout=self._options["timeout"] or 30.0,
            )
            if content:
                message["content"] = content
            try:
                await self._ws.send(json.dumps(message))
                payload = await future
            finally:
                self._pending.pop(request_id, None)
            self._count += 1
            return self._as_response(payload)

    async def _barrier(self, state: Any) -> None:
        """Change the session state with no read in flight.

        The state belongs to the session, not to a request, so a read that is
        already out would otherwise pick up the new one. Draining first makes the
        change apply from here forward and no earlier.
        """
        for _ in range(self._window):
            await self._inflight.acquire()
        try:
            await self._send_state(state)
        finally:
            for _ in range(self._window):
                self._inflight.release()

    def _as_response(self, payload: dict[str, Any]) -> "SystemOneResponse":
        """Decode a ``decision`` frame into the same model the HTTP route returns.

        Callers should not have to know which transport answered, so the ws
        frame is validated into ``SystemOneResponse`` and carries the server's
        own ``latency_ms`` as the read's api time.
        """
        typesafe = require_typesafe()
        response = typesafe.SystemOneResponse.model_validate(
            {
                "model": self._limits.get("model") or self._resource.model,
                "answers": {
                    name: self._coerce_answer(answer)
                    for name, answer in (payload.get("answers") or {}).items()
                },
                "usage": payload.get("usage")
                or {"input_tokens": 0, "output_tokens": 0},
            }
        )
        latency = payload.get("latency_ms")
        response.__dict__["_vlmrun_timings"] = RequestTimings(
            prep_ms=0.0, ttfb_ms=None, api_ms=latency, total_ms=latency
        )
        # The typed Usage drops the gateway's cost and cached-token split, and
        # there is no raw body here to read them back off.
        response.__dict__["_vlmrun_usage"] = payload.get("usage") or {}
        return response

    @staticmethod
    def _coerce_answer(answer: Any) -> Any:
        """Make one wire answer match the shape the response model declares.

        A score's ``legend`` and ``probabilities`` are keyed by level, which JSON
        can only carry as strings. The HTTP path is decoded by the TypeSafe SDK,
        which converts them; validating a socket frame directly does not, and
        ``ScoreAnswer`` declares ``dict[int, ...]``. Without this, a score
        question works over HTTP and fails over the socket.
        """
        if not isinstance(answer, dict) or answer.get("type") != "score":
            return answer
        coerced = dict(answer)
        for field in ("legend", "probabilities"):
            value = coerced.get(field)
            if isinstance(value, dict):
                coerced[field] = {
                    (int(key) if str(key).lstrip("-").isdigit() else key): item
                    for key, item in value.items()
                }
        return coerced

    def close(self, *, cancel: bool = False) -> None:
        """Close the session, collecting its stats, then stop the loop.

        Args:
            cancel: Skip the polite ``session.close`` and drop in-flight reads.
                Used when the stream is abandoned rather than finished.
        """
        if self._ws is not None and not cancel:
            try:
                self._await(self._close_session(), timeout=WEBSOCKET_CLOSE_TIMEOUT)
            except Exception:
                # A session that will not close politely is still closed below.
                pass
        self._teardown()

    async def _close_session(self) -> None:
        """Ask the server to close, and take its stats frame if it sends one."""
        await self._ws.send(json.dumps({"type": "session.close"}))
        deadline = time.monotonic() + WEBSOCKET_CLOSE_TIMEOUT
        while not self._stats and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

    def _teardown(self) -> None:
        """Drop the socket, the reader task and the loop, in that order."""
        if self._loop is not None and self._ws is not None:
            try:
                self._await(self._shutdown(), timeout=WEBSOCKET_CLOSE_TIMEOUT)
            except Exception:
                pass
        self._ws = None
        loop, self._loop = self._loop, None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=WEBSOCKET_CLOSE_TIMEOUT)
        if loop is not None:
            loop.close()
        self._abort(
            InputError(
                message="the websocket session closed before this read finished",
                suggestion="Consume the iterator inside the `with` block.",
            )
        )

    async def _shutdown(self) -> None:
        """Cancel the reader and close the socket."""
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None
        if self._ws is not None:
            await self._ws.close()


class SystemOne:
    """Typed decisions on ``POST {gateway}/typesafe/v1/systemone``.

    Wraps the official ``typesafe-sdk`` client so responses are the SDK's own
    strictly-decoded models and its exceptions propagate unchanged.

    Attributes:
        base_url: The ``/typesafe`` root this resource talks to.
        model: Default model for reads.
    """

    def __init__(
        self,
        client: VLMRunProtocol,
        *,
        base_url: str | None = None,
        gateway_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the System One resource.

        Args:
            client: VLM Run API client instance (provides the API key).
            base_url: Explicit ``/typesafe`` root. Falls back to the
                ``TYPESAFE_BASE_URL`` environment variable, then to the URL
                derived from ``gateway_url``.
            gateway_url: Gateway base URL the route is mounted beside.
            model: Default model; falls back to the served model.
            timeout: Request timeout in seconds.
        """
        self._client = client
        self._base_url = (
            base_url
            or os.getenv(TYPESAFE_BASE_URL_ENV)
            or typesafe_base_url(gateway_url)
        ).rstrip("/")
        self._model = model or os.getenv("TYPESAFE_DEFAULT_MODEL") or SYSTEMONE_MODEL
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        """TypeSafe base URL (without trailing slash)."""
        return self._base_url

    @property
    def model(self) -> str:
        """Default model used for reads."""
        return self._model

    def _api_key(self) -> str:
        # The TypeSafe SDK refuses an empty key, but `vlmrun gw` is usable
        # without one against a deployment that does not authenticate.
        return self._client.api_key or os.getenv("TYPESAFE_API_KEY") or "EMPTY"

    @cached_property
    def client(self):
        """The underlying ``typesafe_sdk.TypeSafeClient``.

        Raises:
            DependencyError: If ``typesafe-sdk`` is not installed.

        Returns:
            A TypeSafeClient pointed at this gateway's ``/typesafe`` prefix.
        """
        typesafe = require_typesafe()
        import httpx2

        return typesafe.TypeSafeClient(
            api_key=self._api_key(),
            base_url=self.base_url,
            model=self._model,
            http_client=httpx2.Client(
                timeout=self._timeout or DEFAULT_READ_TIMEOUT,
                event_hooks=_timing_hooks(),
            ),
        )

    def _request_parts(
        self,
        state: Any,
        questions: Any,
        *,
        model: str | None,
        images: Sequence[str | Path] | None,
        document: str | Path | None,
        text: str | Sequence[str] | None,
        detail: ImageDetail,
        steps: int | None,
        samples: int | None,
        reasoning_effort: str | None,
        timeout: float | None,
        extra_body: Mapping[str, Any] | None,
    ) -> tuple[Any, dict[str, Any], str, dict[str, Any]]:
        """``(state, questions, model, extra)`` — the pieces of one request."""
        extra: dict[str, Any] = {}
        content = build_content(
            images=images,
            document=document,
            text=text,
            detail=detail,
            timeout=timeout or self._timeout or 30.0,
        )
        if content is not None:
            extra["content"] = content
        if steps is not None:
            extra["steps"] = steps
        if samples is not None:
            extra["samples"] = samples
        if reasoning_effort is not None:
            if reasoning_effort not in REASONING_EFFORTS:
                raise InputError(
                    message=f"reasoning_effort {reasoning_effort!r} is not one of "
                    f"{', '.join(REASONING_EFFORTS)}",
                    suggestion="Only a generative engine reasons; see SystemOne.models().",
                )
            extra["reasoning_effort"] = reasoning_effort
        if extra_body:
            extra.update(extra_body)
        return state, normalize_questions(questions), model or self._model, extra

    def build_request(
        self,
        state: Any,
        questions: Any,
        *,
        model: str | None = None,
        images: Sequence[str | Path] | None = None,
        document: str | Path | None = None,
        text: str | Sequence[str] | None = None,
        detail: ImageDetail = "auto",
        steps: int | None = None,
        samples: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The request body :meth:`decide` would send, without sending it.

        Fields are ordered and merged exactly as the TypeSafe SDK does — state,
        model and questions first, then everything else last-write-wins — so the
        returned dict is what actually goes over the wire. Requires no network
        and no ``typesafe-sdk``.

        Args:
            state: Text, a JSON object or an array the questions are about.
            questions: Questions in either dialect.
            model: Model override.
            images: Up to 8 images.
            document: One PDF path or http(s) URL.
            text: Extra text parts.
            detail: Vision budget for this request.
            steps: Denoise steps per read.
            samples: Noise draws to average.
            reasoning_effort: Thinking budget before the answer (generative engines).
            timeout: Seconds allowed for fetching a remote image.
            extra_body: Extra top-level body fields, merged last.

        Returns:
            The JSON-serializable request body.

        Raises:
            InputError: If the questions or media are malformed.
        """
        state, questions, model_id, extra = self._request_parts(
            state,
            questions,
            model=model,
            images=images,
            document=document,
            text=text,
            detail=detail,
            steps=steps,
            samples=samples,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            extra_body=extra_body,
        )
        return {"state": state, "model": model_id, "questions": questions, **extra}

    def decide(
        self,
        state: Any,
        questions: Any,
        *,
        model: str | None = None,
        images: Sequence[str | Path] | None = None,
        document: str | Path | None = None,
        text: str | Sequence[str] | None = None,
        detail: ImageDetail = "auto",
        steps: int | None = None,
        samples: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> "SystemOneResponse":
        """Answer named questions about a state, optionally with media.

        Args:
            state: Text, a JSON object or an array the questions are about.
            questions: Questions in either dialect (see :func:`normalize_questions`).
            model: Model override for this call.
            images: Up to 8 images (paths, http(s) URLs or data URLs).
            document: One PDF path or http(s) URL; its first pages are read.
            text: Extra text parts, read after the state.
            detail: Vision budget for this request: auto, low or high.
            steps: Denoise steps per read (1-8); the contract's default is 1.
            samples: Noise draws to average (1-32). Every draw is billed.
            reasoning_effort: Thinking budget before the answer — none, minimal,
                low, medium or high. Generative engines only; a diffusion engine
                rejects the field with a 422.
            timeout: Per-call timeout in seconds.
            extra_body: Extra top-level body fields, merged last (wins).
            extra_headers: Extra request headers.

        Returns:
            The TypeSafe SDK's ``SystemOneResponse``: answers keyed by question
            id, plus the model and token usage.

        Raises:
            DependencyError: If ``typesafe-sdk`` is not installed.
            InputError: If the questions or media are malformed.
            TypeSafeAPIError: If the gateway returns an unsuccessful response.
        """
        started = time.perf_counter()
        state, questions, model_id, extra = self._request_parts(
            state,
            questions,
            model=model,
            images=images,
            document=document,
            text=text,
            detail=detail,
            steps=steps,
            samples=samples,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            extra_body=extra_body,
        )
        prep_ms = (time.perf_counter() - started) * 1000

        # Build the HTTP client outside the timed window: it is constructed
        # once per resource, and folding that into the first read's latency
        # would misreport it by ~100ms.
        client = self.client
        api_started = time.perf_counter()
        response = client.system_one(
            state,
            questions,
            model=model_id,
            timeout=timeout,
            extra_body=extra or None,
            extra_headers=extra_headers,
        )
        api_ms = (time.perf_counter() - api_started) * 1000
        raw = getattr(response, "__dict__", {}).get("_raw")

        # Runtime state, kept out of the response schema the same way the
        # TypeSafe SDK stashes its own raw response and request id.
        response.__dict__["_vlmrun_timings"] = RequestTimings(
            prep_ms=prep_ms,
            ttfb_ms=(raw.extensions.get("vlmrun_ttfb") if raw is not None else None),
            api_ms=api_ms,
            total_ms=(time.perf_counter() - started) * 1000,
        )
        return response

    def stream(
        self,
        questions: Any,
        *,
        state: Any = "",
        model: str | None = None,
        detail: ImageDetail = "auto",
        steps: int | None = None,
        samples: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        concurrency: int = DEFAULT_STREAM_CONCURRENCY,
        quality: int = 85,
        transport: Literal["http", "ws"] = "http",
    ) -> DecisionStream:
        """Open a stream that reads many decisions against one question spec.

        For a series of inputs — a video's frames, a camera, a queue, a directory
        of stills — where the questions stay the same and the input changes. Use
        it as a context manager; see :class:`DecisionStream`.

        Args:
            questions: Questions in either dialect, normalized once up front.
            state: State every read is about.
            model: Model override for every read in the stream.
            detail: Vision budget per read.
            steps: Denoise steps per read (1-8).
            samples: Noise draws to average per read (1-32). Every draw is billed.
            reasoning_effort: Thinking budget before the answer; generative
                engines only.
            timeout: Per-read timeout in seconds.
            extra_body: Extra top-level body fields, merged last.
            concurrency: Reads in flight at once. On ``ws`` the server's own
                ``max_inflight`` caps this.
            quality: JPEG quality for frames encoded from arrays.
            transport: ``http`` (default) sends an independent request per read.
                ``ws`` opens one session on ``/typesafe/ws``: the questions go
                once and are cached server-side, and reads are pipelined over a
                single socket. Needs ``pip install vlmrun[ws]``.

        Returns:
            A stream, not yet open — enter it to start reading.

        Raises:
            InputError: If the questions are malformed, concurrency is < 1, or
                the transport is not one of ``http`` or ``ws``.

        Example:
            ```python
            from vlmrun.common.video import VideoReader

            with client.gateway.systemone.stream(
                questions=[{"id": "is_open", "type": "noul"}],
                state="Is the door open?",
                concurrency=8,
            ) as stream, VideoReader("door.mp4") as video:
                for decision in stream.map(video.frames(fps=2)):
                    print(decision.timestamp_s, decision.response.nouls["is_open"].noul)
            ```
        """
        if transport not in ("http", "ws"):
            raise InputError(
                message=f"transport must be 'http' or 'ws'; got {transport!r}",
                suggestion="Use 'ws' for one pipelined session, 'http' for independent reads.",
            )
        cls = WebSocketStream if transport == "ws" else DecisionStream
        return cls(
            self,
            questions,
            state=state,
            model=model,
            detail=detail,
            steps=steps,
            samples=samples,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            extra_body=extra_body,
            concurrency=concurrency,
            quality=quality,
        )

    def models(self) -> "ListModelsResponse":
        """List the models this route serves.

        Raises:
            DependencyError: If ``typesafe-sdk`` is not installed.

        Returns:
            The TypeSafe SDK's ``ListModelsResponse``.
        """
        return self.client.models.list()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if "client" in self.__dict__:
            self.client.close()
            del self.__dict__["client"]

    def __enter__(self) -> "SystemOne":
        """Enter the resource context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the underlying HTTP client on exit."""
        self.close()
