"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Literal, Optional, Type
from vlmrun.hub.utils import jsonschema_to_model

JobStatus = Literal["enqueued", "pending", "running", "completed", "failed", "paused"]


@dataclass
class APIError(Exception):
    message: str
    http_status: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


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
    elements_processed: Optional[int] = None
    element_type: Optional[Literal["image", "page", "video", "audio"]] = None
    credits_used: Optional[int] = None


class PredictionResponse(BaseModel):
    id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    response: Optional[Any] = None
    status: JobStatus
    usage: CreditUsage


class ModelInfo(BaseModel):
    model: str
    domain: str


class DomainInfo(BaseModel):
    domain: str


class SchemaResponse(BaseModel):
    domain: str
    description: Optional[str] = None
    schema_version: str
    schema_hash: str
    gql_stmt: str
    json_schema: Dict[str, Any]

    @property
    def response_model(self) -> Type[BaseModel]:
        return jsonschema_to_model(self.json_schema)


class HubInfoResponse(BaseModel):
    version: str


class HubDomainInfo(BaseModel):
    domain: str


class HubSchemaResponse(SchemaResponse): ...


class DatasetResponse(BaseModel):
    id: str
    dataset_type: str
    dataset_name: str
    domain: str
    file_id: Optional[str] = None
    message: Optional[str] = None

    wandb_base_url: Optional[str] = None
    wandb_project_name: Optional[str] = None
    wandb_url: Optional[str] = None

    created_at: datetime
    completed_at: Optional[datetime] = None
    status: JobStatus
    usage: CreditUsage


class FinetuningResponse(BaseModel):
    id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: JobStatus
    message: str
    model: str
    suffix: Optional[str] = None
    wandb_url: Optional[str] = None
    wandb_base_url: Optional[str] = None
    wandb_project_name: Optional[str] = None
    usage: CreditUsage


class FinetuningProvisionResponse(BaseModel):
    id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    model: str
    message: str


class FeedbackSubmitResponse(BaseModel):
    id: str
    created_at: datetime
    request_id: str
    response: Any


class GenerationConfig(BaseModel):
    prompt: Optional[str] = Field(default=None)
    response_model: Optional[Type[BaseModel]] = Field(default=None)
    json_schema: Optional[Dict[str, Any]] = Field(default=None)
    gql_stmt: Optional[str] = Field(default=None)
    max_retries: int = Field(default=3)
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.0)

    detail: Literal["auto", "lo", "hi"] = Field(default="auto")
    confidence: bool = Field(default=False)
    grounding: bool = Field(default=False)


class RequestMetadata(BaseModel):
    environment: Literal["dev", "staging", "prod"] = Field(default="dev")
    session_id: Optional[str] = Field(default=None)
    allow_training: bool = Field(default=True)
