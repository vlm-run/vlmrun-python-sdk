"""VLM Run API Agent resource."""

from __future__ import annotations
from typing import List, Optional, Any, Dict, Union
from pathlib import Path

from loguru import logger

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import (
    PredictionResponse,
    GenerationConfig,
    RequestMetadata,
)


class Agents:
    """Agents resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Agents resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def create(
        self,
        name: str,
        domain: str = "agent",
        requirements: Optional[List[str]] = None,
        code: Optional[str] = None,
    ) -> PredictionResponse:
        """Create an agent with the given name and code.

        Args:
            name: Name of the agent
            domain: Domain identifier
            requirements: Optional requirements for the agent
            code: Optional custom code for the agent

        Returns:
            PredictionResponse: Agent creation response
        """
        data = {
            "name": name,
            "domain": domain,
        }
        
        if requirements is not None:
            data["requirements"] = requirements if isinstance(requirements, str) else ",".join(requirements)
            
        if code is not None:
            data["code"] = code
            
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="agents/create",
            data=data,
        )
        
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
            
        return PredictionResponse(**response)

    def get(
        self,
        name: str,
        domain: str = "agent",
        version: str = "latest",
    ) -> PredictionResponse:
        """Get an agent by name.

        Args:
            name: Name of the agent
            domain: Domain identifier
            version: Version of the agent

        Returns:
            PredictionResponse: Agent response
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"agents/{domain}/{name}/{version}",
        )
        
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
            
        return PredictionResponse(**response)

    def execute(
        self,
        name: str,
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
            url="agents/execute",
            data=data,
        )
        
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
            
        return PredictionResponse(**response)

    def save(
        self,
        name: str,
        domain: str = "agent",
        version: Optional[str] = None,
    ) -> PredictionResponse:
        """Save agent state to disk.

        Args:
            name: Name of the agent
            domain: Domain identifier
            version: Optional version string

        Returns:
            PredictionResponse: Agent save response
        """
        data = {
            "name": name,
            "domain": domain,
        }
        
        if version:
            data["version"] = version
            
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="agents/save",
            data=data,
        )
        
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
            
        return PredictionResponse(**response)

    def list(
        self,
        domain: str = "agent",
        skip: int = 0,
        limit: int = 10,
    ) -> List[PredictionResponse]:
        """List all agents.

        Args:
            domain: Domain identifier
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            List[PredictionResponse]: List of agent responses
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="agents",
            params={"domain": domain, "skip": skip, "limit": limit},
        )
        
        if not isinstance(response, list):
            raise TypeError("Expected list response")
            
        return [PredictionResponse(**agent) for agent in response]
