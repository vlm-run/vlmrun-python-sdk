"""Helpers for optional third-party dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vlmrun.client.exceptions import DependencyError


def _dependency_error(package: str, *, extra: str | None = None) -> "DependencyError":
    """Build a :class:`DependencyError` with pip install guidance.

    The exception is imported here rather than at module scope to keep this
    module importable on its own. ``vlmrun.client.exceptions`` cannot be
    imported without running ``vlmrun.client.__init__``, which reaches back into
    this module for its own guards — so a module-scope import makes the first
    ``vlmrun`` import in a process fail if it happens to be this one.
    """
    from vlmrun.client.exceptions import DependencyError

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


def require_websockets():
    """Import the ``websockets`` client or raise :class:`DependencyError`.

    Only the gateway's websocket transport needs it. Nothing imports it at
    module scope, so an install without the extra is unaffected until a caller
    actually asks for a websocket session.
    """
    try:
        import websockets
    except ImportError as e:
        raise _dependency_error("websockets", extra="ws") from e
    return websockets


def require_typesafe():
    """Import the TypeSafe SDK or raise :class:`DependencyError`.

    The System One route speaks TypeSafe's Jev contract, so the official
    ``typesafe-sdk`` client is what talks to it. It is an optional extra
    because nothing else in this SDK needs it.
    """
    try:
        import typesafe_sdk
    except ImportError as e:
        raise _dependency_error("typesafe-sdk", extra="typesafe") from e
    return typesafe_sdk
