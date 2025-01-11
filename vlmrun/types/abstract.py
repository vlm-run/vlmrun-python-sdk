"""Abstract type definitions for VLM Run API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Client(Protocol):
    """VLM Run API client interface."""

    api_key: str | None
    base_url: str
    timeout: float

    def __post_init__(self) -> None:
        """Initialize the client after dataclass initialization."""
        ...
