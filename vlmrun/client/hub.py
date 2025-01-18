"""VLM Run Hub API client implementation."""

from typing import Any, Dict, List
from dataclasses import asdict

from vlmrun.client.base_requestor import APIRequestor, APIError
from vlmrun.client.types import HubSchemaQueryRequest, HubSchemaQueryResponse


class Hub:
    """Hub API client for VLM Run.
    
    This client provides access to the hub routes for managing schemas and domains.
    """

    def __init__(self, client) -> None:
        """Initialize Hub client.
        
        Args:
            client: API client instance
        """
        self._client = client

    def get_health(self) -> Dict[str, Any]:
        """Check the health of the hub API.
        
        Returns:
            Dict containing status and hub version
            
        Example:
            >>> client.hub.get_health()
            {
                "status": "ok",
                "hub_version": "1.0.0"
            }
            
        Raises:
            APIError: If the request fails
        """
        try:
            response, _, _ = self._client.requestor.request(
                method="GET",
                url="/hub/health",
                raw_response=False
            )
            return response
        except Exception as e:
            raise APIError(f"Failed to check hub health: {str(e)}")

    def list_domains(self) -> List[str]:
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
                method="GET",
                url="/hub/domains",
                raw_response=False
            )
            return response
        except Exception as e:
            raise APIError(f"Failed to list domains: {str(e)}")

    def get_schema(self, domain: str) -> HubSchemaQueryResponse:
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
            request = HubSchemaQueryRequest(domain=domain)
            response, _, _ = self._client.requestor.request(
                method="POST",
                url="/hub/schema",
                data=asdict(request),
                raw_response=False
            )
            return HubSchemaQueryResponse(**response)
        except Exception as e:
            raise APIError(f"Failed to get schema for domain {domain}: {str(e)}")
