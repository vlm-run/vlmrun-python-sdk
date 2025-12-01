"""VLM Run Python SDK.

This module provides the official Python SDK for VLM Run API platform.
"""

# Extend the package path to support namespace packages (e.g., vlmrun.hub from vlmrun-hub)
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from vlmrun.client import VLMRun  # noqa: F401

__all__ = ["VLMRun"]
