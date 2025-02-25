"""VLM Run API Models resource."""

from __future__ import annotations

from typing import List

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import ModelInfo


class Models:
    """Models resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Models resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def list(self) -> List[ModelInfo]:
        """List available models.

        Returns:
            List[ModelResponse]: List of model objects with their capabilities
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="models",
        )

        if not isinstance(response, list):
            raise TypeError("Expected list response")
        return [ModelInfo(**model) for model in response]
