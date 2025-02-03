"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from pydantic.dataclasses import dataclass
from datetime import datetime

from typing import Dict, Any, Literal
from typing import List

JobStatus = Literal["enqueued", "pending", "running", "completed", "failed", "paused"]


@dataclass
class APIError(Exception):
    message: str
    http_status: int | None = None
    headers: Dict[str, str] | None = None


@dataclass
class FileResponse:
    id: str
    filename: str
    bytes: int
    purpose: Literal[
        "fine-tune",
        "assistants",
        "assistants_output",
        "batch",
        "batch_output",
        "vision",
        "datasets",
    ]
    created_at: datetime
    object: str = "file"


@dataclass
class CreditUsage:
    elements_processed: int | None = None
    element_type: Literal["image", "page", "video", "audio"] | None = None
    credits_used: int | None = None


@dataclass
class PredictionResponse:
    id: str
    created_at: datetime
    completed_at: datetime | None
    response: Any | None
    status: JobStatus
    usage: CreditUsage


@dataclass
class ModelInfoResponse:
    model: str
    domain: str


@dataclass
class HubInfoResponse:
    version: str


@dataclass
class HubDomainsResponse:
    domains: List[str]


@dataclass
class HubSchemaQueryResponse:
    schema_json: Dict[str, Any]
    schema_version: str
    schema_hash: str


@dataclass
class DatasetCreateResponse:
    id: str
    dataset_type: str
    dataset_name: str
    domain: str
    message: str
    created_at: datetime
    status: JobStatus
    usage: CreditUsage


@dataclass
class FinetuningResponse:
    id: str
    created_at: datetime
    completed_at: datetime | None
    status: JobStatus
    message: str
    model: str
    suffix: str | None
    usage: CreditUsage


@dataclass
class FinetuningProvisionResponse:
    id: str
    created_at: datetime
    completed_at: datetime | None
    model: str
    message: str


@dataclass
class FeedbackSubmitResponse:
    id: str
    created_at: datetime
    request_id: str
    response: Any
