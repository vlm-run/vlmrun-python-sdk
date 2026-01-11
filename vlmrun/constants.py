from pathlib import Path
import os

DEFAULT_BASE_URL = "https://api.vlm.run/v1"

# Cache directories - use VLMRUN_CACHE_DIR env var if set, otherwise default to ~/.vlm/cache
_cache_base = os.getenv("VLMRUN_CACHE_DIR", str(Path.home() / ".vlm" / "cache"))
VLMRUN_CACHE_DIR = Path(_cache_base)
VLMRUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Artifact cache directory for CLI downloads
VLMRUN_ARTIFACT_CACHE_DIR = VLMRUN_CACHE_DIR / "artifacts"
VLMRUN_ARTIFACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VLMRUN_TMP_DIR = Path.home() / ".vlm" / "tmp"
VLMRUN_TMP_DIR.mkdir(parents=True, exist_ok=True)

VLMRUN_ARTIFACTS_DIR = Path.home() / ".vlmrun" / "artifacts"
VLMRUN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
