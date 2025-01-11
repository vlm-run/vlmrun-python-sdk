"""Type definitions for VLM Run API responses."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


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
    created_at: Optional[datetime] = None


@dataclass
class APIError(Exception):
    """API error response."""

    message: str
    http_status: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
