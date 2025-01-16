from pathlib import Path

VLMRUN_CACHE_DIR = Path.home() / ".vlmrun" / "cache"
VLMRUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
