"""Tests for artifacts operations."""

import pytest

from vlmrun.client.types import ArtifactListResponse


def test_get_artifact(mock_client):
    """Test getting an artifact by session_id and object_id."""
    response = mock_client.artifacts.get(
        object_id="test-object-456",
        session_id="550e8400-e29b-41d4-a716-446655440000",
    )
    assert isinstance(response, bytes)
    assert response == b"mock artifact content"


def test_get_artifact_by_filename(mock_client):
    """Test getting an artifact by filename."""
    response = mock_client.artifacts.get(
        filename="screenshot.png",
        session_id="550e8400-e29b-41d4-a716-446655440000",
    )
    assert isinstance(response, bytes)
    assert response == b"mock artifact content"


def test_get_artifact_requires_object_id_or_filename(mock_client):
    """Test that get() raises ValueError when neither object_id nor filename is provided."""
    with pytest.raises(
        ValueError, match="Either `object_id` or `filename` is required"
    ):
        mock_client.artifacts.get(
            session_id="550e8400-e29b-41d4-a716-446655440000",
        )


def test_get_artifact_rejects_both_object_id_and_filename(mock_client):
    """Test that get() raises ValueError when both object_id and filename are provided."""
    with pytest.raises(ValueError, match="Only one of `object_id` or `filename`"):
        mock_client.artifacts.get(
            object_id="img_a1b2c3",
            filename="screenshot.png",
            session_id="550e8400-e29b-41d4-a716-446655440000",
        )


def test_list_artifacts_by_session_id(mock_client):
    """Test listing artifacts by session_id."""
    session_id = "550e8400-e29b-41d4-a716-446655440000"
    result = mock_client.artifacts.list(session_id=session_id)
    assert isinstance(result, ArtifactListResponse)
    assert result.namespace_id == session_id
    assert len(result.items) == 2
    assert result.items[0].object_id == "img_a1b2c3"
    assert result.items[0].filename == "screenshot.png"
    assert result.items[0].source == "store"
    assert result.items[1].object_id == "doc_d4e5f6"
    assert result.items[1].filename is None
    assert result.items[1].source == "manifest"


def test_list_artifacts_by_execution_id(mock_client):
    """Test listing artifacts by execution_id."""
    execution_id = "exec-12345"
    result = mock_client.artifacts.list(execution_id=execution_id)
    assert isinstance(result, ArtifactListResponse)
    assert result.namespace_id == execution_id


def test_list_artifacts_requires_namespace(mock_client):
    """Test that list() raises ValueError when no namespace is provided."""
    with pytest.raises(
        ValueError, match="Either `session_id` or `execution_id` is required"
    ):
        mock_client.artifacts.list()


def test_list_artifacts_rejects_both_ids(mock_client):
    """Test that list() raises ValueError when both session_id and execution_id are provided."""
    with pytest.raises(ValueError, match="Only one of `session_id` or `execution_id`"):
        mock_client.artifacts.list(
            session_id="sess-123",
            execution_id="exec-456",
        )
