"""VLM Run API Feedback resource."""

from pydantic import BaseModel

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import Client
from vlmrun.client.types import FeedbackSubmitResponse


class Feedback:
    """Feedback resource for VLM Run API."""

    def __init__(self, client: "Client") -> None:
        """Initialize Feedback resource with client.

        Args:
            client: VLM Run API client instance
        """
        self._client = client
        self._requestor = APIRequestor(
            client, base_url=f"{client.base_url}/experimental"
        )

    def submit(
        self,
        id: str,
        label: BaseModel | None = None,
        notes: str | None = None,
        flag: bool | None = None,
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
