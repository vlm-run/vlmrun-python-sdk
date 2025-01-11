"""VLM Run API Models resource."""
from typing import Dict, List

from vlmrun.client.base_requestor import APIRequestor


class Models:
    """Models resource for VLM Run API."""

    def __init__(self, client) -> None:
        """Initialize Models resource with client."""
        self._client = client

    def list(self) -> List[Dict]:
        """List available models.
        
        Returns:
            List[Dict]: List of model objects
        """
        # TODO: Implement with APIRequestor
        return []
