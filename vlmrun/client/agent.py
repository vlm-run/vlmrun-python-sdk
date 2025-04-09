"""VLM Run API Agent resource."""

from __future__ import annotations
from typing import List, Optional

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import (
    PredictionResponse,
    GenerationConfig,
    RequestMetadata,
)


class Agent:
    """Agent resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Agent resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def get(
        self,
        name: str,
        version: str = "latest",
    ) -> PredictionResponse:
        """Get an agent by name.

        Args:
            name: Name of the agent
            version: Version of the agent

        Returns:
            PredictionResponse: Agent response
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"agent/{name}/{version}",
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")

        return PredictionResponse(**response)

    def execute(
        self,
        name: str,
        version: str,
        file_ids: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        batch: bool = True,
        config: Optional[GenerationConfig] = None,
        metadata: Optional[RequestMetadata] = None,
        callback_url: Optional[str] = None,
    ) -> PredictionResponse:
        """Execute an agent with the given arguments.

        Args:
            name: Name of the agent to execute
            version: Version of the agent to execute
            file_ids: Optional list of file IDs to process
            urls: Optional list of URLs to process
            batch: Whether to process in batch mode (async)
            config: Optional generation configuration
            metadata: Optional request metadata
            callback_url: Optional URL to call when execution is complete

        Returns:
            PredictionResponse: Agent execution response

        Raises:
            ValueError: If neither file_ids nor urls are provided, or if both are provided
        """
        if not file_ids and not urls:
            raise ValueError("Either `file_ids` or `urls` must be provided")

        if file_ids and urls:
            raise ValueError("Only one of `file_ids` or `urls` can be provided")

        data = {
            "name": name,
            "version": version,
            "batch": batch,
        }

        if file_ids:
            data["file_ids"] = file_ids

        if urls:
            data["urls"] = urls

        if config:
            data["config"] = config.model_dump()

        if metadata:
            data["metadata"] = metadata.model_dump()

        if callback_url:
            data["callback_url"] = callback_url

        response, status_code, headers = self._requestor.request(
            method="POST",
            url="agent/execute",
            data=data,
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")

        return PredictionResponse(**response)
