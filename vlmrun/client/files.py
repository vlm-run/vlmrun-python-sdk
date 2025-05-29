"""VLM Run API Files resource."""

from __future__ import annotations

import requests
import time
import hashlib
from pathlib import Path
from typing import Optional, Union, List, Literal

from loguru import logger
from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import FileResponse, PresignedUrlResponse


class Files:
    """Files resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Files resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def list(self, skip: int = 0, limit: int = 10) -> List[FileResponse]:
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

    def get_cached_file(self, file: Union[Path, str]) -> Optional[FileResponse]:
        """Get a cached file from the database, if it exists.

        Args:
            file: Path to file to check

        Returns:
            FileResponse: File object if it exists, None otherwise
        """
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
            if status_code == 200:
                file_response = FileResponse(**response)
                # Check if the file exists by checking if id is empty
                if file_response.id:
                    return file_response
                else:
                    return None
            else:
                return None
        except Exception as exc:
            logger.debug(
                f"File hash does not exist in the database [file={file}, hash={file_hash}, exc={exc}]"
            )
            return None

    def upload(
        self,
        file: Union[Path, str],
        purpose: str = "assistants",
        method: Literal["auto", "direct", "presigned-url"] = "auto",
        timeout: int = 3 * 60,
        expiration: int = 24 * 60 * 60,
        force: bool = False,
    ) -> FileResponse:
        """Upload a file.

        Args:
            file: Path to file to upload
            purpose: Purpose of file (default: fine-tune)
            timeout: Timeout for upload (default: 3 minutes)
            method: Method to use for upload (default: "auto")
            expiration: Expiration time for presigned URL (default: 24 hours)
            force: Force upload even if file already exists (default: False)

        Returns:
            FileResponse: Uploaded file object
        """
        if isinstance(file, str):
            try:
                file = Path(file)
            except Exception as exc:
                raise ValueError(f"Invalid file path: {file}") from exc
        elif not isinstance(file, Path):
            raise ValueError(f"Invalid file path: {file}")

        # Check if the file exists
        if not file.exists():
            raise FileNotFoundError(f"File does not exist: {file}")

        # Check if the file already exists in the database
        cached_response: Optional[FileResponse] = self.get_cached_file(file)
        if cached_response and not force:
            return cached_response

        # If method is "auto", check if the file is too large to upload directly
        if method == "auto":
            if file.stat().st_size > 32 * 1024 * 1024:  # 32MB
                method = "presigned-url"
            else:
                method = "direct"
            logger.debug(f"Upload method [file={file}, method={method}]")

        # If method is "presigned-url", generate a presigned URL for the file
        if method == "presigned-url":
            # Generate a presigned URL for the file
            response, status_code, headers = self._requestor.request(
                method="POST",
                url="files/presigned-url",
                data={
                    "filename": file.name,
                    "purpose": purpose,
                    "expiration": expiration,
                },
            )
            if not isinstance(response, dict):
                raise TypeError("Expected dict response")
            response = PresignedUrlResponse(**response)

            # PUT the file to the presigned URL
            start_t = time.time()
            logger.debug(
                f"Uploading file to presigned URL [file={file}, id={response.id}, url={response.url}]"
            )
            with file.open("rb") as f:
                put_response = requests.put(
                    response.url,
                    headers={"Content-Type": response.content_type},
                    data=f,
                )
                status_code = put_response.status_code
            end_t = time.time()
            logger.debug(
                f"Uploaded file to presigned URL [file={file}, url={response.url}, time={end_t - start_t:.1f}s]"
            )
            if status_code == 200:
                # Verify the file upload
                verify_response, status_code, headers = self._requestor.request(
                    method="GET",
                    url=f"files/verify-upload/{response.id}",
                )
                if status_code == 200:
                    return FileResponse(**verify_response)
                else:
                    raise Exception(f"Failed to verify file upload: {verify_response}")
            else:
                raise Exception(f"Failed to upload file to presigned URL: {response}")

        # If method is "direct", upload the file directly
        elif method == "direct":
            logger.debug(f"Uploading file directly [file={file}]")
            # Upload the file
            with open(file, "rb") as f:
                files = {"file": (file.name, f)}
                response, status_code, headers = self._requestor.request(
                    method="POST",
                    url="files",
                    params={"purpose": purpose},
                    files=files,
                    timeout=timeout,
                )
            if status_code == 201:
                if not isinstance(response, dict):
                    raise TypeError("Expected dict response")
                return FileResponse(**response)
            else:
                raise Exception(f"Failed to upload file directly: {response}")

        else:
            raise ValueError(f"Invalid upload method: {method}")

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
