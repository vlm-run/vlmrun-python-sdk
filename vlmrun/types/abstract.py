"""Abstract type definitions for VLM Run API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Any

from vlmrun.client.types import ModelFinetuningRequest, ModelFinetuningResponse, FinetunedImagePredictionRequest


@dataclass
class Client(Protocol):
    """VLM Run API client interface."""

    api_key: str | None
    base_url: str | None
    timeout: float

    def __post_init__(self) -> None:
        """Initialize the client after dataclass initialization."""
        ...

    def healthcheck_finetuning(self) -> bool:
        """Check the health of the fine-tuning service."""
        ...

    def list_finetuned_models(self) -> List[ModelFinetuningResponse]:
        """List all fine-tuned models."""
        ...

    def create_finetuning(self, request: ModelFinetuningRequest) -> ModelFinetuningResponse:
        """Create a fine-tuning job."""
        ...

    def retrieve_finetuning(self, training_id: str) -> ModelFinetuningResponse:
        """Get fine-tuning job status."""
        ...

    def generate_prediction(self, request: FinetunedImagePredictionRequest) -> Dict[str, Any]:
        """Generate predictions using a fine-tuned model."""
        ...
