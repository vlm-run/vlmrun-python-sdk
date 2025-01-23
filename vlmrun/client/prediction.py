"""VLM Run API Prediction resource."""

from __future__ import annotations


from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import Client
from vlmrun.client.types import PredictionResponse


class Prediction:
    """Prediction resource for VLM Run API."""

    def __init__(self, client: "Client") -> None:
        """Initialize Prediction resource with client.

        Args:
            client: VLM Run API client instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def get(self, id: str) -> PredictionResponse:
        """Get prediction by ID.

        Args:
            id: ID of prediction to retrieve

        Returns:
            PredictionResponse: Prediction metadata
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"response/{id}",
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return PredictionResponse(**response)
