"""VLM Run API Dataset resource."""

from __future__ import annotations

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import Client
from vlmrun.client.types import DatasetResponse


class Dataset:
    """Dataset resource for VLM Run API."""

    def __init__(self, client: "Client") -> None:
        """Initialize Dataset resource with client.

        Args:
            client: VLM Run API client instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def create(
        self, file_id: str, domain: str, dataset_name: str, dataset_type: str = "images"
    ) -> DatasetResponse:
        """Create a dataset from an uploaded file.

        Args:
            file_id: ID of the uploaded file to create dataset from
            domain: Domain for the dataset
            dataset_name: Name of the dataset
            dataset_type: Type of dataset (images, videos, or documents)

        Returns:
            DatasetResponse: Dataset creation response containing dataset_id and file_id
        """
        if dataset_type not in ["images", "videos", "documents"]:
            raise ValueError("dataset_type must be one of: images, videos, documents")

        response, status_code, headers = self._requestor.request(
            method="POST",
            url="dataset/create",
            data={
                "file_id": file_id,
                "domain": domain,
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
            },
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return DatasetResponse(**response)

    def get(self, dataset_id: str) -> DatasetResponse:
        """Get dataset information by ID.

        Args:
            dataset_id: ID of the dataset to retrieve

        Returns:
            DatasetResponse: Dataset information including dataset_id and file_id
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"dataset/{dataset_id}",
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return DatasetResponse(**response)
