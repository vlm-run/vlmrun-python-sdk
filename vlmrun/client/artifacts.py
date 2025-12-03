"""VLM Run API Artifacts resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vlmrun.client.base_requestor import APIRequestor

if TYPE_CHECKING:
    from vlmrun.types.abstract import VLMRunProtocol


class Artifacts:
    """Artifacts resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Artifacts resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def get(self, session_id: str, object_id: str) -> bytes:
        """Get an artifact by session ID and object ID.

        Args:
            session_id: Session ID for the artifact
            object_id: Object ID for the artifact

        Returns:
            bytes: The artifact content
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"artifacts/{session_id}/{object_id}",
            raw_response=True,
        )

        if not isinstance(response, bytes):
            raise TypeError("Expected bytes response")
        return response

    def list(self, session_id: str) -> None:
        """List artifacts for a session.

        Args:
            session_id: Session ID to list artifacts for

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Artifacts.list() is not yet implemented")
