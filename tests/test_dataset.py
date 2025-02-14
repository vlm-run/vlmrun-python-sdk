"""Tests for dataset operations."""

import pytest
from datetime import datetime
from vlmrun.client.types import DatasetResponse


def test_dataset_create(mock_client):
    """Test dataset creation."""
    response = mock_client.dataset.create(
        file_id="file1",
        domain="test-domain",
        dataset_name="test-dataset",
        dataset_type="images",
    )
    assert isinstance(response, DatasetResponse)
    assert response.id == "dataset1"
    assert response.domain == "test-domain"
    assert response.dataset_type == "images"
    assert isinstance(response.created_at, datetime)


def test_dataset_get(mock_client):
    """Test dataset retrieval."""
    response = mock_client.dataset.get("dataset1")
    assert isinstance(response, DatasetResponse)
    assert response.id == "dataset1"
    assert response.domain == "test-domain"
    assert response.dataset_type == "images"
    assert isinstance(response.created_at, datetime)


def test_dataset_invalid_type(mock_client):
    """Test dataset creation with invalid type."""
    with pytest.raises(ValueError) as exc_info:
        mock_client.dataset.create(
            file_id="file1",
            domain="test-domain",
            dataset_name="test-dataset",
            dataset_type="invalid",
        )
    assert "dataset_type must be one of: images, videos, documents" in str(
        exc_info.value
    )
