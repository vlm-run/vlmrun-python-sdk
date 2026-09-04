"""Tests for artifacts operations."""

import io
import warnings

import pytest
from PIL import Image

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


@pytest.mark.parametrize(
    "raw,expected_type,expected_id",
    [
        ("img_a1b2c3", "img", "img_a1b2c3"),
        ("img_4c129a.jpg", "img", "img_4c129a"),
        ("vid_4d0e56.mp4", "vid", "vid_4d0e56"),
        ("doc_abcdef.PDF", "doc", "doc_abcdef"),
        ("IMG_ABCDEF.PNG", "img", "img_abcdef"),
    ],
)
def test_normalize_object_id(raw, expected_type, expected_id):
    from vlmrun.client.artifacts import normalize_object_id

    obj_type, normalized = normalize_object_id(raw)
    assert obj_type == expected_type
    assert normalized == expected_id


def test_normalize_object_id_invalid():
    from vlmrun.client.artifacts import normalize_object_id

    with pytest.raises(ValueError):
        normalize_object_id("not-an-id")


def test_artifacts_get_strips_extension_and_accepts_octet_stream(monkeypatch, tmp_path):
    """artifacts.get should strip .jpg and not crash on application/octet-stream."""
    from vlmrun.client import artifacts as artifacts_mod
    from vlmrun.client.artifacts import Artifacts

    class FakeClient:
        api_key = "test"
        base_url = "https://agent.vlm.run/v1"
        max_retries = 1

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color=(0, 0, 255)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    def fake_request(self, method, url, params=None, raw_response=False, **kwargs):
        assert params["object_id"] == "img_4c129a"
        return png_bytes, 200, {"Content-Type": "application/octet-stream"}

    monkeypatch.setattr(artifacts_mod.APIRequestor, "request", fake_request)
    monkeypatch.setattr(artifacts_mod, "VLMRUN_ARTIFACTS_CACHE_DIR", tmp_path)

    art = Artifacts(FakeClient())
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        image = art.get(
            object_id="img_4c129a.jpg",
            session_id="550e8400-e29b-41d4-a716-446655440000",
        )
    assert isinstance(image, Image.Image)
    assert image.size == (2, 2)


def test_long_request_pending_and_result_helpers():
    from vlmrun.client.long_request import is_pending_status, is_result_status

    assert is_pending_status(204)
    assert is_pending_status(303)
    assert is_pending_status(500)
    assert is_pending_status(0)
    assert not is_pending_status(400)

    assert is_result_status(201, b'{"id":"x"}')
    assert is_result_status(200, b'{"id":"x"}')
    assert not is_result_status(201, b"")
    assert not is_result_status(204, b"")
