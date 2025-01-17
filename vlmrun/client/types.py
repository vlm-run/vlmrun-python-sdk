"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class DatasetResponse:
    """Response from dataset operations."""

    dataset_id: str
    dataset_uri: str
    dataset_type: str
    domain: str
    message: str
    created_at: datetime


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

    model: str
    domain: str


@dataclass
class APIError(Exception):
    """API error response."""

    message: str
    http_status: int | None = None
    headers: Dict[str, str] | None = None
