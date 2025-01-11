"""VLM Run API Models resource."""

from typing import List

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.client.types import ModelResponse


class Models:
    """Models resource for VLM Run API."""

    def __init__(self, client) -> None:
        """Initialize Models resource with client."""
        self._client = client
        self._requestor = APIRequestor(client)

    def list(self) -> List[ModelResponse]:
        """List available models.

        Returns:
            List[ModelResponse]: List of model objects with their capabilities
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="v1/models",
        )

        return [ModelResponse(**model) for model in response]
