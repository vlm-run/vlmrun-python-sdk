"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class HubHealthResponse(BaseModel):
    """Response from hub health check."""
    status: str = Field(..., description="Health check status")
    hub_version: str = Field(..., description="Hub version string")


class HubDomainsResponse(BaseModel):
    """Response from listing hub domains."""
    domains: List[str] = Field(..., description="List of supported domains")


@dataclass
class HubSchemaQueryRequest:
    """Request model for hub schema queries."""
    domain: str


@dataclass
class HubSchemaQueryResponse:
    """Response model for hub schema queries.
    
    Attributes:
        schema_json: The JSON schema for the domain
        schema_version: Schema version string
        schema_hash: First 8 characters of schema hash
    """
    schema_json: Dict[str, Any]
    schema_version: str
    schema_hash: str


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
