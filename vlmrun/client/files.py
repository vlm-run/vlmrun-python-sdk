"""VLM Run API Files resource."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union, Literal

from loguru import logger
from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import Client
from vlmrun.client.types import FileResponse


class Files:
    """Files resource for VLM Run API."""

    def __init__(self, client: "Client") -> None:
        """Initialize Files resource with client.

        Args:
            client: VLM Run API client instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def list(self, skip: int = 0, limit: int = 10) -> list[FileResponse]:
        """List all files.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            List[FileResponse]: List of file objects
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="files",
            params={"skip": skip, "limit": limit},
        )
        return [FileResponse(**file) for file in response]

    def _check_file_exists(self, file: Union[Path, str]) -> FileResponse | None:
        if isinstance(file, str):
            file = Path(file)

        # Compute the md5 hash of the file
        logger.debug(f"Computing md5 hash for file [file={file}]")
        file_hash = hashlib.md5()
        with file.open("rb") as f:
            while chunk := f.read(4 * 1024 * 1024):
                file_hash.update(chunk)
            file_hash = file_hash.hexdigest()
        logger.debug(f"Computed md5 hash for file [file={file}, hash={file_hash}]")

        # Check if the file exists in the database
        logger.debug(
            f"Checking if file exists in the database [file={file}, hash={file_hash}]"
        )
        try:
            response, status_code, headers = self._requestor.request(
                method="GET",
                url=f"files/hash/{file_hash}",
            )
            return FileResponse(**response)
        except Exception as exc:
            logger.error(
                f"Failed to check if file exists in the database [file={file}, hash={file_hash}, exc={exc}]"
            )
            return None

    def upload(
        self,
        file: Union[Path, str],
        purpose: Literal[
            "datasets",
            "fine-tune",
            "assistants",
            "assistants_output",
            "batch",
            "batch_output",
            "vision",
        ] = "assistants",
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

        # Check if the file already exists in the database
        cached_response = self._check_file_exists(file)
        if cached_response:
            return cached_response

        # Upload the file
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

    def get(self, file_id: str) -> FileResponse:
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

    def get_content(self, file_id: str) -> bytes:
        """Get file content.

        Args:
            file_id: ID of file to retrieve content for

        Returns:
            bytes: File content
        """
        raise NotImplementedError("Not implemented")

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
