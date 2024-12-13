# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from datetime import datetime
from typing_extensions import Literal, Required, Annotated, TypedDict

from .._utils import PropertyInfo

__all__ = ["WebGenerateParams", "Metadata"]


class WebGenerateParams(TypedDict, total=False):
    url: Required[str]
    """The URL of the web page."""

    id: str
    """Unique identifier of the request."""

    callback_url: Optional[str]
    """The URL to call when the request is completed."""

    created_at: Annotated[Union[str, datetime], PropertyInfo(format="iso8601")]
    """Date and time when the request was created (in UTC timezone)"""

    domain: Optional[Literal["web.ecommerce-product-catalog", "web.github-developer-stats", "web.market-research"]]
    """The domain identifier."""

    metadata: Metadata
    """Optional metadata to pass to the model."""

    mode: Literal["fast", "accurate"]
    """The mode to use for the model."""

    model: Literal["vlm-1"]
    """The model to use for generating the response."""


class Metadata(TypedDict, total=False):
    allow_training: bool
    """Whether the file can be used for training"""

    environment: Literal["dev", "staging", "prod"]
    """The environment where the request was made."""

    session_id: Optional[str]
    """The session ID of the request"""
