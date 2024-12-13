# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from datetime import datetime
from typing_extensions import Literal, Required, Annotated, TypedDict

from .._utils import PropertyInfo

__all__ = ["ImageGenerateParams", "Metadata"]


class ImageGenerateParams(TypedDict, total=False):
    image: Required[str]
    """Base64 encoded image."""

    id: str
    """Unique identifier of the request."""

    callback_url: Optional[str]
    """The URL to call when the request is completed."""

    created_at: Annotated[Union[str, datetime], PropertyInfo(format="iso8601")]
    """Date and time when the request was created (in UTC timezone)"""

    detail: Literal["auto", "hi", "lo"]
    """The detail level to use for the model."""

    domain: Optional[
        Literal[
            "document.generative",
            "document.presentation",
            "document.invoice",
            "document.receipt",
            "document.markdown",
            "video.tv-news",
            "video.tv-intelligence",
        ]
    ]
    """The domain identifier."""

    json_schema: Optional[object]
    """The JSON schema to use for the model."""

    metadata: Metadata
    """Optional metadata to pass to the model."""

    model: Literal["vlm-1"]
    """The model to use for generating the response."""


class Metadata(TypedDict, total=False):
    allow_training: bool
    """Whether the file can be used for training"""

    environment: Literal["dev", "staging", "prod"]
    """The environment where the request was made."""

    session_id: Optional[str]
    """The session ID of the request"""
