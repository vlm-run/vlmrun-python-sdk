from pathlib import Path

DEFAULT_BASE_URL = "https://api.vlm.run/v1"

VLMRUN_CACHE_DIR = Path.home() / ".vlmrun" / "cache"
VLMRUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VLMRUN_TMP_DIR = Path.home() / ".vlmrun" / "tmp"
VLMRUN_TMP_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_VIDEO_FILETYPES = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
SUPPORTED_IMAGE_FILETYPES = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
