"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from typing import Dict, Any, Literal
from typing import List


@dataclass
class HubInfoResponse:
    """Response from hub info."""

    version: str


@dataclass
class HubSchemaQueryRequest:
    """Request model for hub schema queries."""

    domain: str


@dataclass
class HubDomainsResponse:
    """Response from listing hub domains."""

    domains: List[str]


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
class PredictionResponse:
    """Response from prediction operations."""

    id: str
    created_at: datetime
    completed_at: datetime | None
    response: Any | None
    status: str
    usage: CreditUsage


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


@dataclass
class CreditUsage:
    """Credit usage for fine-tuning operations."""

    elements_processed: int
    element_type: Literal["image", "page", "video", "audio"] | None
    credits_used: int


@dataclass
class FinetuningJobRequest:
    """Request for fine-tuning operations."""

    dataset_uri: str
    dataset_format: str
    model: str
    task_prompt: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    use_lora: bool
    track: bool
    wandb_project: str


@dataclass
class FinetuningJobResponse:
    """Response from fine-tuning operations."""

    id: str
    created_at: datetime
    completed_at: datetime | None
    status: Literal["pending", "running", "completed", "failed"]
    request: FinetuningJobRequest
    model: str
    usage: CreditUsage
