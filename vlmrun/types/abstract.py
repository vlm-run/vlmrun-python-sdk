"""Abstract type definitions for VLM Run API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any, TypeVar, Optional

T = TypeVar("T")


@dataclass
class VLMRunProtocol(Protocol):
    """VLM Run API interface."""

    api_key: Optional[str]
    base_url: str
    timeout: float
    files: Any
    datasets: Any
    models: Any
    hub: Any
    image: Any
    video: Any
    document: Any
    fine_tuning: Any
    feedback: Any
    requestor: Any

    def __post_init__(self) -> None:
        """Initialize after dataclass initialization."""
        ...
