"""Tests for artifacts operations."""

import pytest


def test_get_artifact(mock_client):
    """Test getting an artifact by session_id and object_id."""
    response = mock_client.artifacts.get(
        object_id="test-object-456",
        session_id="550e8400-e29b-41d4-a716-446655440000",
    )
    assert isinstance(response, bytes)
    assert response == b"mock artifact content"


def test_list_artifacts_not_implemented(mock_client):
    """Test that list() raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        mock_client.artifacts.list(session_id="550e8400-e29b-41d4-a716-446655440000")
    assert "not yet implemented" in str(exc_info.value)
