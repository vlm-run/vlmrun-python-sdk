"""VLM Run API Executions resource."""

from __future__ import annotations

import time
from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import (
    AgentExecutionResponse,
)


class Executions:
    """Executions resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Executions resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client, timeout=120)

    def list(self, skip: int = 0, limit: int = 10) -> list[AgentExecutionResponse]:
        """List all executions.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            List[AgentExecutionResponse]: List of execution objects
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="agent/executions",
            params={"skip": skip, "limit": limit},
        )
        return [AgentExecutionResponse(**execution) for execution in response]

    def get(self, id: str) -> AgentExecutionResponse:
        """Get execution by ID.

        Args:
            id: ID of execution to retrieve

        Returns:
            AgentExecutionResponse: Execution metadata
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"agent/executions/{id}",
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return AgentExecutionResponse(**response)

    def wait(
        self, id: str, timeout: int = 300, sleep: int = 5
    ) -> AgentExecutionResponse:
        """Wait for execution to complete.

        Args:
            id: ID of execution to wait for
            timeout: Maximum number of seconds to wait
            sleep: Time to wait between checks in seconds (default: 5)

        Returns:
            AgentExecutionResponse: Completed execution

        Raises:
            TimeoutError: If execution does not complete within timeout
        """
        start_time = time.time()
        while True:
            response: AgentExecutionResponse = self.get(id)
            if response.status == "completed":
                return response

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Execution {id} did not complete within {timeout} seconds. Last status: {response.status}"
                )
            time.sleep(min(sleep, timeout - elapsed))
