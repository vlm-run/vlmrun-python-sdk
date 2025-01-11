"""VLM Run API Models resource."""

from __future__ import annotations

from typing import List

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import Client
from vlmrun.client.types import ModelResponse


class Models:
    """Models resource for VLM Run API."""

    def __init__(self, client: "Client") -> None:
        """Initialize Models resource with client.

        Args:
            client: VLM Run API client instance
        """
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

        if not isinstance(response, list):
            raise TypeError("Expected list response")
        return [ModelResponse(**model) for model in response]
