"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


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
class ModelFinetuningRequest:
    """Request for fine-tuning a model."""

    model: str
    dataset_uri: str
    dataset_format: Optional[str] = None
    task_prompt: Optional[str] = None
    num_epochs: int = 1
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    use_lora: Optional[bool] = False
    track: Optional[bool] = False


@dataclass
class ModelFinetuningResponse:
    """Response from fine-tuning operations."""

    id: str
    model: str
    status: str
    request: ModelFinetuningRequest


@dataclass
class FinetunedImagePredictionRequest:
    """Request for generating predictions with a fine-tuned model."""

    model: str
    image_data: str  # base64-encoded image or URL
    prompt: Optional[str] = None
    task_prompt: Optional[str] = None
    schema: Optional[str] = None
    temperature: float = 1.0
    max_new_tokens: int = 128


@dataclass
class APIError(Exception):
    """API error response."""

    message: str
    http_status: int | None = None
    headers: Dict[str, str] | None = None
