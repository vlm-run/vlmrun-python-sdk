from __future__ import annotations

from pathlib import Path
import os

DEFAULT_BASE_URL = "https://api.vlm.run/v1"

# OpenAI-compatible model gateway (third-party OCR / VLM models).
DEFAULT_GATEWAY_URL = "https://gateway.vlm.run/v1"

# The environment variables that override the URLs above, named so downstream
# code can read or set them without hardcoding the strings.
VLMRUN_BASE_URL_ENV = "VLMRUN_BASE_URL"
VLMRUN_GATEWAY_BASE_URL_ENV = "VLMRUN_GATEWAY_BASE_URL"
#: The gateway's original variable name. Still honoured, after
#: :data:`VLMRUN_GATEWAY_BASE_URL_ENV`, so existing setups keep working.
VLMRUN_GATEWAY_URL_ENV = "VLMRUN_GATEWAY_URL"
TYPESAFE_BASE_URL_ENV = "TYPESAFE_BASE_URL"


def gateway_base_url(base_url: str | None = None) -> str:
    """Resolve the gateway's base URL.

    The one place the precedence lives, so every resource that talks to the
    gateway agrees on it: an explicit argument, then
    ``VLMRUN_GATEWAY_BASE_URL``, then the older ``VLMRUN_GATEWAY_URL``, then
    :data:`DEFAULT_GATEWAY_URL`.

    Args:
        base_url (str | None, optional): An explicit override, which wins.

    Returns:
        str: The gateway base URL, without a trailing slash.
    """
    return (
        base_url
        or os.getenv(VLMRUN_GATEWAY_BASE_URL_ENV)
        or os.getenv(VLMRUN_GATEWAY_URL_ENV)
        or DEFAULT_GATEWAY_URL
    ).rstrip("/")


# Cache directories - use VLMRUN_CACHE_DIR env var if set, otherwise default to ~/.vlmrun/cache
VLMRUN_HOME = Path.home() / ".vlmrun"
VLMRUN_HOME.mkdir(parents=True, exist_ok=True)
_cache_base = os.getenv("VLMRUN_CACHE_DIR", str(VLMRUN_HOME / "cache"))
VLMRUN_CACHE_DIR = Path(_cache_base)
VLMRUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Artifact cache directory for CLI downloads
VLMRUN_ARTIFACTS_CACHE_DIR = VLMRUN_CACHE_DIR / "artifacts"
VLMRUN_ARTIFACTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VLMRUN_TMP_DIR = VLMRUN_HOME / "tmp"
VLMRUN_TMP_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_VIDEO_FILETYPES = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
SUPPORTED_IMAGE_FILETYPES = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
SUPPORTED_DOCUMENT_FILETYPES = [".pdf", ".doc", ".docx"]
SUPPORTED_AUDIO_FILETYPES = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

# All supported file types for the chat CLI
SUPPORTED_INPUT_FILETYPES = (
    SUPPORTED_IMAGE_FILETYPES
    + SUPPORTED_VIDEO_FILETYPES
    + SUPPORTED_DOCUMENT_FILETYPES
    + SUPPORTED_AUDIO_FILETYPES
)
