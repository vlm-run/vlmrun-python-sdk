# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from datetime import datetime
from typing_extensions import Literal, Annotated, TypedDict

from ...._utils import PropertyInfo

__all__ = ["EmbeddingCreateParams", "Metadata"]


class EmbeddingCreateParams(TypedDict, total=False):
    id: str
    """Unique identifier of the request."""

    batch: bool
    """Whether to process the image/text in batch mode (async)."""

    callback_url: Optional[str]
    """The URL to call when the request is completed."""

    created_at: Annotated[Union[str, datetime], PropertyInfo(format="iso8601")]
    """Date and time when the request was created (in UTC timezone)"""

    image: Optional[str]
    """Base64 encoded image to embed (at least one of `image` or `text` is required)."""

    metadata: Optional[Metadata]
    """Metadata for the request."""

    model: Literal["vlm-1-embeddings"]
    """The model to use for generating the response."""

    text: Optional[str]
    """Text to embed (at least one of `image` or `text` is required)."""


class Metadata(TypedDict, total=False):
    allow_training: bool
    """Whether the file can be used for training"""

    environment: Literal["dev", "staging", "prod"]
    """The environment where the request was made."""

    session_id: Optional[str]
    """The session ID of the request"""
