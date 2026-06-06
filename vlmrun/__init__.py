"""Public package interface for the VLM Run Python SDK."""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from vlmrun.client import VLMRun
from vlmrun.version import __version__

__all__ = ["VLMRun", "__version__"]
