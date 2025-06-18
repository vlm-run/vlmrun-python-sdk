"""VLM Run API Feedback resource."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import (
    FeedbackSubmitRequest,
    FeedbackItem,
    FeedbackSubmitResponse,
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
    ) -> FeedbackSubmitResponse:
        """List feedback for a prediction request.

        Args:
            request_id: ID of the prediction request
            limit: Maximum number of feedback items to return
            offset: Number of feedback items to skip

        Returns:
            FeedbackSubmitResponse: Response with list of feedback items
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"v1/feedback/{request_id}",
            params={"limit": limit, "offset": offset},
        )
        return FeedbackSubmitResponse(**response)

    def submit(
        self,
        request_id: str,
        response: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> FeedbackSubmitResponse:
        """Submit feedback for a prediction.

        Args:
            request_id: ID of the prediction request
            response: Feedback response data
            notes: Optional notes about the feedback

        Returns:
            FeedbackSubmitResponse: Response with submitted feedback
        """
        feedback_data = FeedbackSubmitRequest(
            request_id=request_id,
            response=response, 
            notes=notes
        )
        
        response_data, status_code, headers = self._requestor.request(
            method="POST",
            url="v1/feedback/submit",
            data=feedback_data.model_dump(exclude_none=True),
        )
        return FeedbackSubmitResponse(**response_data)
