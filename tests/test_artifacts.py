"""Tests for artifacts operations."""

import pytest


def test_get_artifact(mock_client):
    """Test getting an artifact by session_id and object_id."""
    response = mock_client.artifacts.get(
        session_id="test-session-123",
        object_id="test-object-456"
    )
    assert isinstance(response, bytes)
    assert response == b"mock artifact content"


def test_get_artifact_with_path_like_ids(mock_client):
    """Test getting an artifact with path-like session_id and object_id."""
    response = mock_client.artifacts.get(
        session_id="session/with/slashes",
        object_id="object/with/slashes"
    )
    assert isinstance(response, bytes)
    assert response == b"mock artifact content"


def test_list_artifacts_not_implemented(mock_client):
    """Test that list() raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        mock_client.artifacts.list(session_id="test-session-123")
    assert "not yet implemented" in str(exc_info.value)
