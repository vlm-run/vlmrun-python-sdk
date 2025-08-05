"""VLM Run API Agent resource."""

from __future__ import annotations
from typing import Any, Optional

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import (
    RequestMetadata,
    AgentInfo,
    AgentExecutionResponse,
    AgentExecutionConfig,
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
    ) -> AgentInfo:
        """Get an agent by name.

        Args:
            name: Name of the agent
            version: Version of the agent

        Returns:
            AgentInfo: Agent information response
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"agent/{name}/{version}",
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")

        return AgentInfo(**response)

    def execute(
        self,
        name: str,
        version: str = "latest",
        inputs: Optional[dict[str, Any]] = None,
        batch: bool = True,
        config: Optional[AgentExecutionConfig] = None,
        metadata: Optional[RequestMetadata] = None,
        callback_url: Optional[str] = None,
    ) -> AgentExecutionResponse:
        """Execute an agent with the given arguments.

        Args:
            name: Name of the agent to execute
            version: Version of the agent to execute
            inputs: Optional inputs to the agent
            batch: Whether to process in batch mode (async)
            config: Optional agent execution configuration
            metadata: Optional request metadata
            callback_url: Optional URL to call when execution is complete

        Returns:
            AgentExecutionResponse: Agent execution response
        """
        if not batch:
            raise NotImplementedError("Batch mode is required for agent execution")

        data = {
            "name": name,
            "version": version,
            "batch": batch,
            "inputs": inputs,
        }

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

        return AgentExecutionResponse(**response)
