"""VLM Run API Feedback resource."""

from typing import Optional
from pydantic import BaseModel

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import FeedbackSubmitResponse


class Feedback:
    """Feedback resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Feedback resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(
            client, base_url=f"{client.base_url}/experimental"
        )

    def submit(
        self,
        id: str,
        label: Optional[BaseModel] = None,
        notes: Optional[str] = None,
        flag: Optional[bool] = None,
    ) -> FeedbackSubmitResponse:
        """Create feedback for a prediction.

        Returns:
            List[ModelResponse]: List of model objects with their capabilities
        """
        if label is not None:
            if not isinstance(label, BaseModel):
                raise ValueError("label must be a Pydantic model")
            label = label.model_dump()

        response, status_code, headers = self._requestor.request(
            method="POST",
            url="feedback/submit",
            data={
                "request_id": id,
                "response": label,
                "notes": notes,
                "flag": flag,
            },
        )

        return FeedbackSubmitResponse(**response)

    def get(self, id: str) -> FeedbackSubmitResponse:
        """Get feedback by request ID.

        Returns:
            FeedbackSubmitResponse: Feedback for a prediction
        """
        response, status_code, headers = self._requestor.request(
            method="GET", url=f"feedback/{id}"
        )
        return FeedbackSubmitResponse(**response)
