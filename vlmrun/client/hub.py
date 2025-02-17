"""VLM Run Hub API implementation."""

from typing import TYPE_CHECKING, List, Type
from pydantic import BaseModel

from vlmrun.client.base_requestor import APIError
from vlmrun.client.types import (
    HubSchemaResponse,
    HubInfoResponse,
    HubDomainInfo,
)
from vlmrun.hub.registry import registry

if TYPE_CHECKING:
    from vlmrun.types.abstract import VLMRunProtocol


class Hub:
    """Hub API for VLM Run.

    This module provides access to the hub routes for managing schemas and domains.
    """

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Hub resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client

    def info(self) -> HubInfoResponse:
        """Get the hub info.

        Returns:
            HubInfoResponse containing hub version

        Example:
            >>> client.hub.info()
            {
                "version": "1.0.0"
            }

        Raises:
            APIError: If the request fails
        """
        try:
            response, _, _ = self._client.requestor.request(
                method="GET", url="/hub/info", raw_response=False
            )
            if not isinstance(response, dict):
                raise APIError("Expected dict response from health check")
            return HubInfoResponse(**response)
        except Exception as e:
            raise APIError(f"Failed to check hub health: {str(e)}")

    def list_domains(self) -> List[HubDomainInfo]:
        """Get the list of supported domains.

        Returns:
            List of domain strings

        Example:
            >>> client.hub.list_domains()
            [
                "document.invoice",
                "document.receipt",
                "document.utility_bill"
            ]

        Raises:
            APIError: If the request fails
        """
        try:
            response, _, _ = self._client.requestor.request(
                method="GET", url="/hub/domains", raw_response=False
            )
            if not isinstance(response, list):
                raise TypeError("Expected list response")
            return [HubDomainInfo(**domain) for domain in response]
        except Exception as e:
            raise APIError(f"Failed to list domains: {str(e)}")

    def get_schema(self, domain: str) -> HubSchemaResponse:
        """Get the JSON schema for a given domain.

        Args:
            domain: Domain identifier (e.g. "document.invoice")

        Returns:
            HubSchemaQueryResponse containing:
            - schema_json: The JSON schema for the domain
            - schema_version: Schema version string
            - schema_hash: First 8 characters of schema hash

        Example:
            >>> response = client.hub.get_schema("document.invoice")
            >>> print(response.schema_version)
            "1.0.0"
            >>> print(response.schema_json)
            {
                "type": "object",
                "properties": {...}
            }

        Raises:
            APIError: If the request fails or domain is not found
        """
        try:
            response, _, _ = self._client.requestor.request(
                method="POST",
                url="/hub/schema",
                data={"domain": domain},
                raw_response=False,
            )
            if not isinstance(response, dict):
                raise APIError("Expected dict response from schema query")
            return HubSchemaResponse(**response)
        except Exception as e:
            raise APIError(f"Failed to get schema for domain {domain}: {str(e)}")

    def get_pydantic_model(self, domain: str) -> Type[BaseModel]:
        """Get the Pydantic model for a given domain.

        Args:
            domain: Domain identifier (e.g. "document.invoice")

        Returns:
            Type[BaseModel]: The Pydantic model class for the domain

        Raises:
            APIError: If the domain is not found
        """
        try:
            return registry[domain]
        except KeyError:
            raise APIError(f"Domain not found: {domain}")
