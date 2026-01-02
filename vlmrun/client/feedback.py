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
        request_id: Optional[str] = None,
        agent_execution_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        response: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> FeedbackSubmitResponse:
        """Submit feedback for a prediction, agent execution, or chat.

        Args:
            request_id: ID of the prediction request (mutually exclusive with agent_execution_id and chat_id)
            agent_execution_id: ID of the agent execution (mutually exclusive with request_id and chat_id)
            chat_id: ID of the chat (mutually exclusive with request_id and agent_execution_id)
            response: Optional feedback response data
            notes: Optional notes about the feedback

        Returns:
            FeedbackSubmitResponse: Response with submitted feedback

        Raises:
            ValueError: If none or more than one of request_id, agent_execution_id, chat_id is provided
            ValueError: If neither response nor notes is provided
        """
        provided_ids = sum(
            bool(id_value) for id_value in [request_id, agent_execution_id, chat_id]
        )

        if provided_ids == 0:
            raise ValueError(
                "Must provide one of: request_id, agent_execution_id, or chat_id"
            )
        elif provided_ids > 1:
            raise ValueError(
                "Must provide only one ID. Cannot specify multiple entity IDs simultaneously."
            )

        if response is None and notes is None:
            raise ValueError(
                "`response` or `notes` parameter is required and cannot be None"
            )

        feedback_data = FeedbackSubmitRequest(
            request_id=request_id,
            agent_execution_id=agent_execution_id,
            chat_id=chat_id,
            response=response,
            notes=notes,
        )

        response_data, status_code, headers = self._requestor.request(
            method="POST",
            url="feedback/submit",
            data=feedback_data.model_dump(exclude_none=True),
        )
        return FeedbackSubmitResponse(**response_data)
