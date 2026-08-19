"""Helpers for optional third-party dependencies."""

from __future__ import annotations

from vlmrun.client.exceptions import DependencyError


def _dependency_error(package: str, *, extra: str | None = None) -> DependencyError:
    """Build a :class:`DependencyError` with pip install guidance."""
    if extra:
        suggestion = (
            f"Install it with `pip install vlmrun[{extra}]` or `pip install {package}`"
        )
    else:
        suggestion = f"Install it with `pip install vlmrun` or `pip install {package}`"

    return DependencyError(
        message=f"{package} is not installed",
        suggestion=suggestion,
        error_type="missing_dependency",
    )


def require_openai():
    """Import the OpenAI SDK or raise :class:`DependencyError`."""
    try:
        import openai
    except ImportError as e:
        raise _dependency_error("openai") from e
    return openai


def require_pandas():
    """Import pandas or raise :class:`DependencyError`."""
    try:
        import pandas as pd
    except ImportError as e:
        raise _dependency_error("pandas", extra="all") from e
    return pd


def require_numpy():
    """Import numpy or raise :class:`DependencyError`."""
    try:
        import numpy as np
    except ImportError as e:
        raise _dependency_error("numpy", extra="video") from e
    return np


def require_cv2():
    """Import OpenCV or raise :class:`DependencyError`."""
    try:
        import cv2
    except ImportError as e:
        raise _dependency_error("opencv-python", extra="video") from e
    return cv2


def require_ipython_html():
    """Import IPython's HTML display helper or raise :class:`DependencyError`."""
    try:
        from IPython.display import HTML
    except ImportError as e:
        raise _dependency_error("ipython", extra="all") from e
    return HTML


def require_pypdfium2():
    """Import pypdfium2 or raise :class:`DependencyError`."""
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise _dependency_error("pypdfium2", extra="doc") from e
    return pdfium
