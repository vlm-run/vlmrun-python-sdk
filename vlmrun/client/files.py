"""VLM Run API Files resource."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import Client
from vlmrun.client.types import FileList, FileResponse


class Files:
    """Files resource for VLM Run API."""

    def __init__(self, client: "Client") -> None:
        """Initialize Files resource with client.

        Args:
            client: VLM Run API client instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def list(self) -> FileList:
        """List all files.

        Returns:
            FileList: List of file objects
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="files",
        )

        return FileList(
            data=[FileResponse(**file) for file in response.get("data", [])]
        )

    def upload(
        self, file: Union[Path, str], purpose: str = "fine-tune"
    ) -> FileResponse:
        """Upload a file.

        Args:
            file: Path to file to upload
            purpose: Purpose of file (default: fine-tune)

        Returns:
            FileResponse: Uploaded file object
        """
        if isinstance(file, str):
            file = Path(file)

        with open(file, "rb") as f:
            files = {"file": (file.name, f)}
            response, status_code, headers = self._requestor.request(
                method="POST",
                url="files",
                params={"purpose": purpose},
                files=files,
            )

            if not isinstance(response, dict):
                raise TypeError("Expected dict response")
            return FileResponse(**response)

    def retrieve(self, file_id: str) -> FileResponse:
        """Get file metadata.

        Args:
            file_id: ID of file to retrieve

        Returns:
            FileResponse: File metadata
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"files/{file_id}",
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return FileResponse(**response)

    def retrieve_content(self, file_id: str) -> bytes:
        """Get file content.

        Args:
            file_id: ID of file to retrieve content for

        Returns:
            bytes: File content
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"files/{file_id}/content",
            raw_response=True,
        )

        return response

    def delete(self, file_id: str) -> FileResponse:
        """Delete a file.

        Args:
            file_id: ID of file to delete

        Returns:
            FileResponse: Deletion confirmation
        """
        response, status_code, headers = self._requestor.request(
            method="DELETE",
            url=f"files/{file_id}",
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return FileResponse(**response)
