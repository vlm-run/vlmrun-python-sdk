from pathlib import Path

VLMRUN_CACHE_DIR = Path.home() / ".vlmrun" / "cache"
VLMRUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VLMRUN_TMP_DIR = Path.home() / ".vlmrun" / "tmp"
VLMRUN_TMP_DIR.mkdir(parents=True, exist_ok=True)
