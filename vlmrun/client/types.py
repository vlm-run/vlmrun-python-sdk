"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from typing import Dict, Any, Literal
from typing import List


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
    purpose: str
    created_at: datetime
    object: str = "file"


@dataclass
class CreditUsage:
    elements_processed: int
    element_type: Literal["image", "page", "video", "audio"] | None
    credits_used: int


@dataclass
class PredictionResponse:
    id: str
    created_at: datetime
    completed_at: datetime | None
    response: Any | None
    status: str
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
class DatasetResponse:
    dataset_id: str
    dataset_uri: str
    dataset_type: str
    domain: str
    message: str
    created_at: datetime


@dataclass
class FinetuningJobRequest:
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
    id: str
    created_at: datetime
    completed_at: datetime | None
    status: Literal["pending", "running", "completed", "failed"]
    request: FinetuningJobRequest
    model: str
    usage: CreditUsage


@dataclass
class FeedbackSubmitResponse:
    id: str
    created_at: datetime
    request_id: str
    response: Any
