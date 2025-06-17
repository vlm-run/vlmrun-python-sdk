"""VLM Run API Feedback resource."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import (
    FeedbackSubmitResponse,
    FeedbackCreateParams,
    FeedbackResponse,
    FeedbackListResponse,
    FeedbackListParams,
)


class Feedback:
    """Feedback resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Feedback resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def list(
        self,
        request_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> FeedbackListResponse:
        """List feedback for a prediction request.

        Args:
            request_id: ID of the prediction request
            limit: Maximum number of feedback items to return
            offset: Number of feedback items to skip

        Returns:
            FeedbackListResponse: List of feedback items with pagination info
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"feedback/{request_id}",
            params={"limit": limit, "offset": offset},
        )
        return FeedbackListResponse(**response)

    def submit(
        self,
        request_id: str,
        response: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> FeedbackResponse:
        """Submit feedback for a prediction.

        Args:
            request_id: ID of the prediction request
            response: Feedback response data
            notes: Optional notes about the feedback

        Returns:
            FeedbackResponse: Created feedback object
        """
        feedback_data = FeedbackCreateParams(response=response, notes=notes)
        
        response_data, status_code, headers = self._requestor.request(
            method="POST",
            url=f"feedback/submit/{request_id}",
            data=feedback_data.model_dump(exclude_none=True),
        )
        return FeedbackResponse(**response_data)
