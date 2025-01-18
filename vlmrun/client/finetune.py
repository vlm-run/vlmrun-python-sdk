"""VLM Run API Fine-tuning resource."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List

from vlmrun.client.types import (
    ModelFinetuningRequest,
    ModelFinetuningResponse,
    FinetunedImagePredictionRequest,
)

if TYPE_CHECKING:
    from vlmrun.client.client import Client


class FineTuning:
    """Fine-tuning resource for VLM Run API."""

    def __init__(self, client: "Client") -> None:
        """Initialize FineTuning resource with client.

        Args:
            client: VLM Run API client instance
        """
        self._client = client

    def health(self) -> bool:
        """Check the health of the fine-tuning service.

        Returns:
            bool: True if service is healthy
        """
        _, status_code, _ = self._client.requestor.request(
            method="GET", url="/health", raw_response=True
        )
        return status_code == 200

    def list_models(self) -> List[ModelFinetuningResponse]:
        """List all fine-tuned models.

        Returns:
            List[ModelFinetuningResponse]: List of fine-tuned models

        Raises:
            TypeError: If response is not in expected format
        """
        result, _, _ = self._client.requestor.request(
            method="GET", url="/models/"
        )
        if not isinstance(result, list):
            raise TypeError("Expected list response from /models/ endpoint")
        
        responses = []
        for item in result:
            if not isinstance(item, dict):
                raise TypeError("Expected dict items in /models/ response")
            if not all(isinstance(k, str) for k in item.keys()):
                raise TypeError("Expected string keys in /models/ response items")
            
            # Extract and validate request data
            request_data = item.get("request", {})
            if not isinstance(request_data, dict):
                raise TypeError("Expected dict for request data in /models/ response")
            
            # Create request object first
            request = ModelFinetuningRequest(**request_data)
            
            # Create clean response dictionary with required fields
            response_data = {
                "id": str(item.get("id", "")),
                "model": str(item.get("model", "")),
                "status": str(item.get("status", "")),
                "request": request
            }
            
            # Create response object
            responses.append(ModelFinetuningResponse(**response_data))
        return responses

    def create_fit(self, request: ModelFinetuningRequest) -> ModelFinetuningResponse:
        """Create a fine-tuning job.

        Args:
            request: Fine-tuning request parameters

        Returns:
            ModelFinetuningResponse: Created fine-tuning job details

        Raises:
            TypeError: If response is not in expected format
        """
        payload = dataclasses.asdict(request)
        result, _, _ = self._client.requestor.request(
            method="POST", url="/fit", data=payload
        )
        if not isinstance(result, dict):
            raise TypeError("Expected dict response from /fit endpoint")
        # Ensure request is included in response for proper typing
        result["request"] = request
        return ModelFinetuningResponse(**result)

    def retrieve_fit(self, training_id: str) -> ModelFinetuningResponse:
        """Get fine-tuning job status.

        Args:
            training_id: ID of job to retrieve

        Returns:
            ModelFinetuningResponse: Fine-tuning job status and details

        Raises:
            TypeError: If response is not in expected format
        """
        endpoint = f"/{training_id}"
        result, _, _ = self._client.requestor.request(
            method="GET", url=endpoint
        )
        if not isinstance(result, dict):
            raise TypeError("Expected dict response from /{training_id} endpoint")
        return ModelFinetuningResponse(**result)

    def generate(self, request: FinetunedImagePredictionRequest) -> Dict[str, Any]:
        """Generate predictions using a fine-tuned model.

        Args:
            request: Prediction request parameters

        Returns:
            Dict[str, Any]: Model predictions

        Raises:
            TypeError: If response is not in expected format
        """
        payload = dataclasses.asdict(request)
        result, _, _ = self._client.requestor.request(
            method="POST", url="/generate", data=payload
        )
        if not isinstance(result, dict):
            raise TypeError("Expected dict response from /generate endpoint")
        return result
