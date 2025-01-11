"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class FileResponse:
    """Response from file operations."""

    id: str
    filename: str
    bytes: int
    purpose: str
    created_at: datetime
    object: str = "file"


@dataclass
class FileList:
    """Response from listing files."""

    data: List[FileResponse]
    object: str = "list"


@dataclass
class ModelResponse:
    """Response from model operations."""

    id: str
    name: str
    domain: str
    created_at: datetime | None = None


@dataclass
class APIError(Exception):
    """API error response."""

    message: str
    http_status: int | None = None
    headers: Dict[str, str] | None = None
