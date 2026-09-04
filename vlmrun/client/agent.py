"""VLM Run API Agent resource."""

from __future__ import annotations
import functools
import warnings
from functools import cached_property
from typing import Any, List, Optional, Union

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
    AgentToolset,
)
from vlmrun.common.dependencies import require_openai

# VLM Run-specific kwargs accepted by the agent API that are not part of the
# standard OpenAI chat completions signature. They are forwarded to the server
# via `extra_body`.
_VLM_EXTRA_KEYS: frozenset[str] = frozenset({"skills", "toolsets", "models"})

# Default wall-clock budget for following a Modal 303 long-request poll URL.
_DEFAULT_LONG_REQUEST_TIMEOUT = 900.0


def _pop_vlm_extra_kwargs(kwargs: dict[str, Any]) -> None:
    """Move VLM-specific kwargs into ``extra_body`` in-place."""
    vlm_kwargs = {k: kwargs.pop(k) for k in _VLM_EXTRA_KEYS if k in kwargs}
    if vlm_kwargs:
        kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), **vlm_kwargs}


def _status_code_from_exc(exc: BaseException) -> int | None:
    """Best-effort extract of an HTTP status code from an OpenAI/httpx error."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def _headers_from_exc(exc: BaseException) -> dict[str, Any]:
    """Best-effort extract of response headers from an OpenAI/httpx error."""
    response = getattr(exc, "response", None)
    if response is None:
        return {}
    headers = getattr(response, "headers", None) or {}
    try:
        return dict(headers)
    except Exception:
        return {}


def _parse_chat_completion(body: bytes) -> Any:
    """Parse poll-response bytes into an OpenAI ChatCompletion object."""
    import json

    from openai.types.chat import ChatCompletion

    payload = json.loads(body)
    return ChatCompletion.model_validate(payload)


def _follow_long_request_303(
    exc: BaseException,
    *,
    api_key: str | None,
    timeout: float,
) -> Any:
    """Poll the Modal ``Location`` URL from a 303 and return the ChatCompletion."""
    from vlmrun.client.long_request import (
        extract_location,
        poll_location,
    )

    location = extract_location(_headers_from_exc(exc))
    if not location:
        raise exc

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body, _status, _resp_headers = poll_location(
        location,
        headers=headers,
        timeout=timeout,
    )
    return _parse_chat_completion(body)


def _patch_create(create_fn: Any, *, api_key: str | None, timeout: float) -> Any:
    """Wrap an OpenAI ``create`` callable for VLM kwargs + Modal 303 polling.

    ``skills``, ``toolsets``, and ``models`` are popped from ``kwargs`` and
    merged into ``extra_body`` before the underlying call is made. If the
    gateway returns ``303 See Other`` (long request), the Location URL is
    polled until the chat completion is ready.
    """

    @functools.wraps(create_fn)
    def _create(*args: Any, **kwargs: Any) -> Any:
        _pop_vlm_extra_kwargs(kwargs)
        try:
            return create_fn(*args, **kwargs)
        except Exception as exc:
            if _status_code_from_exc(exc) != 303:
                raise
            return _follow_long_request_303(
                exc,
                api_key=api_key,
                timeout=max(float(timeout or 0), _DEFAULT_LONG_REQUEST_TIMEOUT),
            )

    return _create


def _patch_async_create(create_fn: Any, *, api_key: str | None, timeout: float) -> Any:
    """Async variant of :func:`_patch_create`."""
    import asyncio

    @functools.wraps(create_fn)
    async def _create(*args: Any, **kwargs: Any) -> Any:
        _pop_vlm_extra_kwargs(kwargs)
        try:
            return await create_fn(*args, **kwargs)
        except Exception as exc:
            if _status_code_from_exc(exc) != 303:
                raise
            return await asyncio.to_thread(
                _follow_long_request_303,
                exc,
                api_key=api_key,
                timeout=max(float(timeout or 0), _DEFAULT_LONG_REQUEST_TIMEOUT),
            )

    return _create


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
    ) -> Optional[dict[str, Any]]:
        """Process inputs, converting BaseModel to dict if needed.

        Args:
            inputs: Input data as dict, BaseModel, or None

        Returns:
            Processed inputs as dict or None
        """
        if isinstance(inputs, BaseModel):
            return inputs.model_dump(exclude_none=True)
        elif isinstance(inputs, dict):
            warnings.warn(
                "Passing inputs as a dictionary will be deprecated in the future. "
                "Please use a Pydantic BaseModel instead for better type safety and validation.",
                DeprecationWarning,
                stacklevel=3,
            )
        return inputs

    def get(
        self,
        name: Optional[str] = None,
        id: Optional[str] = None,
        prompt: Optional[str] = None,
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
        name: Optional[str] = None,
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
            "config": config.model_dump(exclude_none=True),
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
        name: Optional[str] = None,
        inputs: Optional[Union[dict[str, Any], BaseModel]] = None,
        batch: bool = True,
        config: Optional[AgentExecutionConfig] = None,
        metadata: Optional[RequestMetadata] = None,
        callback_url: Optional[str] = None,
        model: str = "vlmrun-orion-1:auto",
        toolsets: Optional[List[AgentToolset]] = None,
    ) -> AgentExecutionResponse:
        """Execute an agent with the given arguments.

        Args:
            name: Name of the agent to execute. If not provided, we use the prompt to identify the unique agent.
            inputs: Optional inputs to the agent or a BaseModel instance
            batch: Whether to process in batch mode (async)
            config: Optional agent execution configuration
            metadata: Optional request metadata
            callback_url: Optional URL to call when execution is complete
            model: VLM Run Agent model to use for execution (default: "vlmrun-orion-1:auto")
            toolsets: Optional list of tool categories to enable for this execution.
                Available categories: core, image, image-gen, world-gen,
                viz, document, video, web.
                When specified, only tools from these categories will be available.
                If None, defaults to 'core' tools only.

        Returns:
            AgentExecutionResponse: Agent execution response
        """
        if not batch:
            raise NotImplementedError("Batch mode is required for agent execution")

        data = {
            "model": model,
            "name": name,
            "batch": batch,
            "inputs": self._process_inputs(inputs),
        }

        if config:
            data["config"] = config.model_dump(exclude_none=True)

        if metadata:
            data["metadata"] = metadata.model_dump(exclude_none=True)

        if callback_url:
            data["callback_url"] = callback_url

        if toolsets is not None:
            data["toolsets"] = toolsets

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

            client = VLMRun(api_key="your-key", base_url="https://api.vlm.run/v1")

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
        openai = require_openai()
        base_url = f"{self._client.base_url}/openai"
        openai_client = openai.OpenAI(
            api_key=self._client.api_key,
            base_url=base_url,
            timeout=self._client.timeout if self._client.timeout is None else max(self._client.timeout, 600),
            max_retries=self._client.max_retries,
        )

        completions = openai_client.chat.completions
        completions.create = _patch_create(
            completions.create,
            api_key=self._client.api_key,
            timeout=float(self._client.timeout or _DEFAULT_LONG_REQUEST_TIMEOUT),
        )
        return completions

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

            client = VLMRun(api_key="your-key", base_url="https://api.vlm.run/v1")

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
        openai = require_openai()
        base_url = f"{self._client.base_url}/openai"
        async_openai_client = openai.AsyncOpenAI(
            api_key=self._client.api_key,
            base_url=base_url,
            timeout=self._client.timeout if self._client.timeout is None else max(self._client.timeout, 600),
            max_retries=self._client.max_retries,
        )

        completions = async_openai_client.chat.completions
        completions.create = _patch_async_create(
            completions.create,
            api_key=self._client.api_key,
            timeout=float(self._client.timeout or _DEFAULT_LONG_REQUEST_TIMEOUT),
        )
        return completions
