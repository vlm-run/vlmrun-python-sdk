"""VLM Run API Feedback resource."""

from typing import Optional, Dict, Any

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import (
    FeedbackSubmitRequest,
    FeedbackListResponse,
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

    def get(
        self,
        request_id: str,
        offset: int = 0,
        limit: int = 10,
    ) -> FeedbackListResponse:
        """Get feedback for a prediction request.

        Args:
            request_id: ID of the prediction request
            offset: Number of feedback items to skip
            limit: Maximum number of feedback items to return

        Returns:
            FeedbackListResponse: Response with list of feedback items
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"feedback/{request_id}",
            params={"offset": offset, "limit": limit},
        )
        return FeedbackListResponse(**response)

    def submit(
        self,
        request_id: str,
        response: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> FeedbackSubmitResponse:
        """Submit feedback for a prediction.

        Args:
            request_id: ID of the prediction request
            response: Optional feedback response data
            notes: Optional notes about the feedback

        Returns:
            FeedbackSubmitResponse: Response with submitted feedback
        """
        if response is None and notes is None:
            raise ValueError(
                "`response` or `notes` parameter is required and cannot be None"
            )

        feedback_data = FeedbackSubmitRequest(
            request_id=request_id, response=response, notes=notes
        )

        response_data, status_code, headers = self._requestor.request(
            method="POST",
            url="feedback/submit",
            data=feedback_data.model_dump(exclude_none=True),
        )
        return FeedbackSubmitResponse(**response_data)
