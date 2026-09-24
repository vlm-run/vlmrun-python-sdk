"""MIME sniffing and data-URL encoding shared by the client and the CLI.

Extensions lie — a ``.jpg`` that is really WebP is common — and every gateway
route trusts the media type we declare in the ``data:`` URL, so a wrong one
makes the server misroute the file. Magic bytes are checked first, the
filename extension second.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

# Magic-byte signatures, checked before the filename extension.
MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"%PDF", "application/pdf"),
)

# Image types every gateway route accepts.
IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif"}
)

# ``image/jpg`` is not a registered type, but object stores serve it.
MIME_ALIASES: dict[str, str] = {"image/jpg": "image/jpeg"}


def sniff_mime(data: bytes) -> str | None:
    """MIME type from a file's magic bytes, or None if unrecognized."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "video/x-msvideo"
    if data[4:8] == b"ftyp":
        return "video/mp4"
    for signature, mime in MAGIC_SIGNATURES:
        if data.startswith(signature):
            return mime
    return None


def guess_mime(path: Path, data: bytes | None = None) -> str:
    """Best-effort MIME type for a local file, preferring its actual content."""
    if data is None:
        try:
            with path.open("rb") as fh:
                data = fh.read(16)
        except OSError:
            data = b""
    sniffed = sniff_mime(data)
    if sniffed:
        return sniffed
    mime, _ = mimetypes.guess_type(str(path))
    return normalize_mime(mime or "application/octet-stream")


def normalize_mime(mime: str) -> str:
    """Canonical spelling of a MIME type (``image/jpg`` -> ``image/jpeg``)."""
    return MIME_ALIASES.get(mime.lower(), mime.lower())


def is_http_url(value: str) -> bool:
    """Return True if ``value`` looks like an http(s) URL."""
    return value.startswith(("http://", "https://"))


def suffix_from_url(url: str) -> str:
    """File extension from a URL path, ignoring query strings."""
    return Path(urlparse(url).path).suffix.lower()


def mime_from_url(url: str) -> str:
    """Best-effort MIME type for a remote URL, from its path extension."""
    mime, _ = mimetypes.guess_type(urlparse(url).path)
    return normalize_mime(mime or "application/octet-stream")


def data_url(data: bytes, mime: str) -> str:
    """Encode bytes as a base64 ``data:`` URL."""
    return (
        f"data:{normalize_mime(mime)};base64,{base64.b64encode(data).decode('ascii')}"
    )


def file_data_url(path: Path) -> tuple[str, str, int]:
    """Encode a local file as a ``data:`` URL.

    Args:
        path: Local file to read.

    Returns:
        Tuple of (data URL, resolved MIME type, raw byte count).

    Raises:
        OSError: If the file cannot be read.
    """
    raw = path.read_bytes()
    mime = guess_mime(path, raw)
    return data_url(raw, mime), mime, len(raw)
