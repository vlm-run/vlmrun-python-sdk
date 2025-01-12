"""General utilities for VLMRun."""

from pathlib import Path


# VLMRun directory constants
VLMRUN_CACHE_DIR = Path.home() / ".vlmrun" / "cache"
VLMRUN_HUB_DIR = Path.home() / ".vlmrun" / "hub"

# Create necessary directories
VLMRUN_HUB_DIR.mkdir(parents=True, exist_ok=True)
VLMRUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
