"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from datetime import datetime

from typing import Dict, Any, Literal

JobStatus = Literal["enqueued", "pending", "running", "completed", "failed", "paused"]


@dataclass
class APIError(Exception):
    message: str
    http_status: int | None = None
    headers: Dict[str, str] | None = None


class FileResponse(BaseModel):
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


class CreditUsage(BaseModel):
    elements_processed: int | None = None
    element_type: Literal["image", "page", "video", "audio"] | None = None
    credits_used: int | None = None


class PredictionResponse(BaseModel):
    id: str
    created_at: datetime
    completed_at: datetime | None
    response: Any | None
    status: JobStatus
    usage: CreditUsage


class ModelInfoResponse(BaseModel):
    model: str
    domain: str


class HubInfoResponse(BaseModel):
    version: str


class HubDomainInfo(BaseModel):
    domain: str


class HubSchemaResponse(BaseModel):
    json_schema: Dict[str, Any]
    schema_version: str
    schema_hash: str


class DatasetCreateResponse(BaseModel):
    id: str
    dataset_type: str
    dataset_name: str
    domain: str
    message: str
    created_at: datetime
    status: JobStatus
    usage: CreditUsage


class FinetuningResponse(BaseModel):
    id: str
    created_at: datetime
    completed_at: datetime | None
    status: JobStatus
    message: str
    model: str
    suffix: str | None
    usage: CreditUsage


class FinetuningProvisionResponse(BaseModel):
    id: str
    created_at: datetime
    completed_at: datetime | None
    model: str
    message: str


class FeedbackSubmitResponse(BaseModel):
    id: str
    created_at: datetime
    request_id: str
    response: Any
