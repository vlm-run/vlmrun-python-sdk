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
    AgentCreationConfig,
    AgentCreationResponse,
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
        name: str | None = None,
        version: str | None = None,
        id: str | None = None,
    ) -> AgentInfo:
        """Get an agent by name and version.

        Args:
            name: Name of the agent (lookup either by name + version or by id alone)
            version: Version of the agent
            id: ID of the agent

        Raises:
            APIError: If the agent is not found (404) or the agent name is invalid (400)

        Returns:
            AgentInfo: Agent information response
        """
        if id and name:
            raise ValueError("Only one of `id` or `name` can be provided.")
        elif id is not None:
            data = {"id": id}
        elif name is not None:
            data = {"name": name, "version": version}
        else:
            raise ValueError("Either `id` or `name` must be provided.")

        response, status_code, headers = self._requestor.request(
            method="GET",
            url="agent/lookup",
            data=data,
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")

        return AgentInfo(**response)

    def list(self) -> list[AgentInfo]:
        """List all agents."""
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="agent",
        )

        if not isinstance(response, list):
            raise TypeError("Expected list response")

        return [AgentInfo(**agent) for agent in response]

    def create(
        self,
        config: AgentCreationConfig,
        name: str | None = None,
        inputs: Optional[dict[str, Any]] = None,
        callback_url: Optional[str] = None,
    ) -> AgentCreationResponse:
        """Create an agent.

        Args:
            config: Agent creation configuration
            name: Optional name of the agent to create
            inputs: Optional inputs to the agent (e.g. {"image": "https://..."})
            callback_url: Optional URL to call when creation is complete

        Returns:
            AgentCreationResponse: Agent creation response
        """
        if config.prompt is None:
            raise ValueError(
                "Prompt is not provided as a request parameter, please provide a prompt."
            )

        data = {
            "name": name,
            "inputs": inputs,
            "config": config.model_dump(),
        }

        if callback_url:
            data["callback_url"] = callback_url

        response, status_code, headers = self._requestor.request(
            method="POST",
            url="agent/create",
            data=data,
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")

        return AgentCreationResponse(**response)

    def execute(
        self,
        name: str,
        version: str | None = None,
        inputs: Optional[dict[str, Any]] = None,
        batch: bool = True,
        config: Optional[AgentExecutionConfig] = None,
        metadata: Optional[RequestMetadata] = None,
        callback_url: Optional[str] = None,
    ) -> AgentExecutionResponse:
        """Execute an agent with the given arguments.

        Args:
            name: Name of the agent to execute
            version: Optional version of the agent to execute
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

    def get_by_id(self, agent_id: str) -> AgentInfo:
        """Get agent information by ID.

        Args:
            agent_id: The ID of the agent to retrieve

        Returns:
            AgentInfo: Information about the agent
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"agents/{agent_id}",
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")

        return AgentInfo(**response)
