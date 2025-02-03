"""VLM Run API Dataset resource."""

from __future__ import annotations

from loguru import logger
from pathlib import Path
from typing import List, Literal
from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import DatasetCreateResponse, FileResponse
from vlmrun.common.utils import create_archive


class Datasets:
    """Datasets resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Datasets resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client, base_url=f"{client.base_url}")

    def create(
        self,
        domain: str,
        dataset_directory: Path,
        dataset_name: str,
        dataset_type: Literal["images", "videos", "documents"],
    ) -> DatasetCreateResponse:
        """Create a dataset from an uploaded file.

        Args:
            dataset_directory: Directory of files to create dataset from
            domain: Domain for the dataset
            dataset_name: Name of the dataset
            dataset_type: Type of dataset (images, videos, or documents)

        Returns:
            DatasetResponse: Dataset creation response containing dataset_id and file_id
        """
        if dataset_type not in ["images", "videos", "documents"]:
            raise ValueError("dataset_type must be one of: images, videos, documents")

        # Create tar.gz file
        tar_path: Path = create_archive(dataset_directory, dataset_name)
        logger.debug(
            f"Created tar.gz file [path={tar_path}, size={tar_path.stat().st_size / 1024 / 1024:.2f} MB]"
        )

        # Upload tar.gz file
        upload_response, _, _ = self._requestor.request(
            method="POST",
            url="files/upload",
            files={"file": tar_path},
            data={"purpose": "datasets"},
        )
        if not isinstance(upload_response, dict):
            raise TypeError("Expected dict response")
        file_response = FileResponse(**upload_response)
        logger.debug(
            f"Uploaded tar.gz file [path={tar_path}, file_id={file_response.id}, size={file_response.bytes / 1024 / 1024:.2f} MB]"
        )

        # Create dataset
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="datasets/create",
            data={
                "file_id": file_response.id,
                "domain": domain,
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
            },
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return DatasetCreateResponse(**response)

    def get(self, dataset_id: str) -> DatasetCreateResponse:
        """Get dataset information by ID.

        Args:
            dataset_id: ID of the dataset to retrieve

        Returns:
            DatasetResponse: Dataset information including dataset_id and file_id
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"datasets/{dataset_id}",
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return DatasetCreateResponse(**response)

    def list(self, skip: int = 0, limit: int = 10) -> List[DatasetCreateResponse]:
        """List all datasets."""
        items, status_code, headers = self._requestor.request(
            method="GET",
            url="datasets",
            params={"skip": skip, "limit": limit},
        )
        if not isinstance(items, list):
            raise TypeError("Expected list response")
        return [DatasetCreateResponse(**response) for response in items]
