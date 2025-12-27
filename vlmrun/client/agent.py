"""VLM Run API Agent resource."""

from __future__ import annotations
from functools import cached_property
from typing import Any, Optional, Union

from pydantic import BaseModel

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
from vlmrun.client.exceptions import DependencyError


class Agent:
    """Agent resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Agent resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def _process_inputs(
        self, inputs: Union[dict[str, Any], BaseModel, None]
    ) -> dict[str, Any] | None:
        """Process inputs, converting BaseModel to dict if needed.

        Args:
            inputs: Input data as dict, BaseModel, or None

        Returns:
            Processed inputs as dict or None
        """
        if isinstance(inputs, BaseModel):
            return inputs.model_dump()
        return inputs

    def get(
        self,
        name: str | None = None,
        id: str | None = None,
        prompt: str | None = None,
    ) -> AgentInfo:
        """Get an agent by name, id, or prompt. Only one of `name`, `id`, or `prompt` can be provided.

        Args:
            name: Name of the agent
            id: ID of the agent
            prompt: Prompt of the agent

        Raises:
            APIError: If the agent is not found (404) or the agent name is invalid (400)

        Returns:
            AgentInfo: Agent information response
        """
        if id:
            if name or prompt:
                raise ValueError(
                    "Only one of `id` or `name` or `prompt` can be provided."
                )
            data = {"id": id}
        elif name:
            if id or prompt:
                raise ValueError(
                    "Only one of `id` or `name` or `prompt` can be provided."
                )
            data = {"name": name}
        elif prompt:
            if id or name:
                raise ValueError(
                    "Only one of `id` or `name` or `prompt` can be provided."
                )
            data = {"prompt": prompt}
        else:
            raise ValueError("Either `id` or `name` or `prompt` must be provided.")

        response, status_code, headers = self._requestor.request(
            method="POST",
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
        inputs: Optional[Union[dict[str, Any], BaseModel]] = None,
        callback_url: Optional[str] = None,
    ) -> AgentCreationResponse:
        """Create an agent.

        Args:
            config: Agent creation configuration
            name: Optional name of the agent to create
            inputs: Optional inputs to the agent (e.g. {"image": "https://..."}) or a BaseModel instance
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
            "inputs": self._process_inputs(inputs),
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
        name: str | None = None,
        inputs: Optional[Union[dict[str, Any], BaseModel]] = None,
        batch: bool = True,
        config: Optional[AgentExecutionConfig] = None,
        metadata: Optional[RequestMetadata] = None,
        callback_url: Optional[str] = None,
    ) -> AgentExecutionResponse:
        """Execute an agent with the given arguments.

        Args:
            name: Name of the agent to execute. If not provided, we use the prompt to identify the unique agent.
            inputs: Optional inputs to the agent or a BaseModel instance
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
            "batch": batch,
            "inputs": self._process_inputs(inputs),
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

    @cached_property
    def completions(self):
        """OpenAI-compatible chat completions interface (synchronous).

        Returns an OpenAI Completions object configured to use the VLMRun
        agent endpoint. This allows you to use the familiar OpenAI API
        for chat completions.

        Example:
            ```python
            from vlmrun import VLMRun

            client = VLMRun(api_key="your-key", base_url="https://agent.vlm.run/v1")

            response = client.agent.completions.create(
                model="vlmrun-orion-1",
                messages=[
                    {"role": "user", "content": "Hello!"}
                ]
            )
            ```

        Raises:
            DependencyError: If openai package is not installed

        Returns:
            OpenAI Completions object configured for VLMRun agent endpoint
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise DependencyError(
                message="OpenAI SDK is not installed",
                suggestion="Install it with `pip install vlmrun[openai]` or `pip install openai`",
                error_type="missing_dependency",
            )

        base_url = f"{self._client.base_url}/openai"
        openai_client = OpenAI(
            api_key=self._client.api_key,
            base_url=base_url,
            timeout=self._client.timeout,
            max_retries=self._client.max_retries,
        )

        return openai_client.chat.completions

    @cached_property
    def async_completions(self):
        """OpenAI-compatible chat completions interface (asynchronous).

        Returns an OpenAI AsyncCompletions object configured to use the VLMRun
        agent endpoint. This allows you to use the familiar OpenAI async API
        for chat completions.

        Example:
            ```python
            from vlmrun import VLMRun
            import asyncio

            client = VLMRun(api_key="your-key", base_url="https://agent.vlm.run/v1")

            async def main():
                response = await client.agent.async_completions.create(
                    model="vlmrun-orion-1",
                    messages=[
                        {"role": "user", "content": "Hello!"}
                    ]
                )
                return response

            asyncio.run(main())
            ```

        Raises:
            DependencyError: If openai package is not installed

        Returns:
            OpenAI AsyncCompletions object configured for VLMRun agent endpoint
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise DependencyError(
                message="OpenAI SDK is not installed",
                suggestion="Install it with `pip install vlmrun[openai]` or `pip install openai`",
                error_type="missing_dependency",
            )

        base_url = f"{self._client.base_url}/openai"
        async_openai_client = AsyncOpenAI(
            api_key=self._client.api_key,
            base_url=base_url,
            timeout=self._client.timeout,
            max_retries=self._client.max_retries,
        )

        return async_openai_client.chat.completions
