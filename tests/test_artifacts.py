"""Tests for artifacts operations."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from vlmrun.client.artifacts import Artifacts, _filename_from_url


# ---------------------------------------------------------------------------
# Mock-client tests (use the lightweight mock_client fixture from conftest)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _filename_from_url unit tests
# ---------------------------------------------------------------------------


class TestFilenameFromUrl:
    """Tests for the _filename_from_url helper."""

    def test_simple_url(self):
        assert _filename_from_url("https://example.com/path/to/file.jpg") == "file.jpg"

    def test_presigned_url_strips_query_params(self):
        url = (
            "https://storage.googleapis.com/bucket/img_a1b2c3.jpg"
            "?X-Goog-Algorithm=GOOG4&X-Goog-Credential=abc"
        )
        assert _filename_from_url(url) == "img_a1b2c3.jpg"

    def test_s3_presigned_url(self):
        url = (
            "https://s3.amazonaws.com/my-bucket/session/vid_f0e1d2.mp4"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=3600"
        )
        assert _filename_from_url(url) == "vid_f0e1d2.mp4"

    def test_url_with_fragment(self):
        url = "https://cdn.example.com/doc_abc123.pdf#page=1"
        assert _filename_from_url(url) == "doc_abc123.pdf"

    def test_url_without_extension(self):
        url = "https://example.com/artifacts/img_a1b2c3"
        assert _filename_from_url(url) == "img_a1b2c3"


# ---------------------------------------------------------------------------
# Redirect-handling tests (patch APIRequestor.request + requests.get)
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(
        self,
        content: bytes,
        headers: dict[str, str] | None = None,
        status_code: int = 200,
    ):
        self.content = content
        self.headers = headers or {}
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _StubClient:
    """Minimal protocol-compatible client for ``Artifacts.__init__``."""

    api_key = "test-key"
    base_url = "https://api.vlm.run/v1"
    timeout = 30.0
    max_retries = 1


def _make_1x1_jpeg() -> bytes:
    """Return a minimal valid JPEG in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color="red").save(buf, format="JPEG")
    return buf.getvalue()


class TestArtifactsRedirectHandling:
    """Verify that ``Artifacts.get`` follows redirects without auth headers."""

    def test_redirect_to_presigned_url_raw(self):
        """A 302 redirect should be followed and raw bytes returned."""
        redirect_headers = {
            "Location": "https://storage.googleapis.com/bucket/img_a1b2c3.jpg?sig=abc",
        }
        artifact_bytes = b"image-bytes-from-gcs"
        fake_redirect_resp = _FakeResponse(
            content=artifact_bytes,
            headers={"Content-Type": "image/jpeg"},
        )

        with (
            patch.object(
                Artifacts,
                "__init__",
                lambda self, client: setattr(self, "_client", client)
                or setattr(self, "_requestor", None),
            ),
            patch(
                "vlmrun.client.artifacts.APIRequestor",
            ),
            patch(
                "vlmrun.client.artifacts._follow_redirect",
                return_value=fake_redirect_resp,
            ) as mock_follow,
        ):
            arts = Artifacts.__new__(Artifacts)
            arts._client = _StubClient()

            # Simulate requestor returning a 302 with Location header
            class _FakeRequestor:
                def request(self, **kwargs):
                    return (b"", 302, redirect_headers)

            arts._requestor = _FakeRequestor()

            result = arts.get(
                object_id="img_a1b2c3",
                session_id="sess-001",
                raw_response=True,
            )

            assert result == artifact_bytes
            mock_follow.assert_called_once_with(
                "https://storage.googleapis.com/bucket/img_a1b2c3.jpg?sig=abc"
            )

    def test_redirect_to_presigned_url_image(self, tmp_path, monkeypatch):
        """A 302 redirect for an img artifact returns a PIL Image."""
        jpeg_bytes = _make_1x1_jpeg()
        redirect_headers = {
            "Location": "https://s3.amazonaws.com/bucket/img_a1b2c3.jpg?X-Amz-Sig=xyz",
        }
        fake_redirect_resp = _FakeResponse(
            content=jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
        )

        monkeypatch.setattr(
            "vlmrun.client.artifacts.VLMRUN_ARTIFACTS_CACHE_DIR", tmp_path
        )

        with patch(
            "vlmrun.client.artifacts._follow_redirect",
            return_value=fake_redirect_resp,
        ):
            arts = Artifacts.__new__(Artifacts)
            arts._client = _StubClient()

            class _FakeRequestor:
                def request(self, **kwargs):
                    return (b"", 302, redirect_headers)

            arts._requestor = _FakeRequestor()

            result = arts.get(
                object_id="img_a1b2c3",
                session_id="sess-002",
            )

            assert isinstance(result, Image.Image)
            assert result.mode == "RGB"

    def test_redirect_to_presigned_url_video(self, tmp_path, monkeypatch):
        """A 302 redirect for a vid artifact writes to disk and returns Path."""
        monkeypatch.setattr(
            "vlmrun.client.artifacts.VLMRUN_ARTIFACTS_CACHE_DIR", tmp_path
        )
        video_bytes = b"\x00\x00\x00\x1c" + b"\x00" * 100  # fake mp4 header
        redirect_headers = {
            "Location": "https://cdn.example.com/vid_f0e1d2.mp4?token=abc",
        }
        fake_redirect_resp = _FakeResponse(
            content=video_bytes,
            headers={"Content-Type": "video/mp4"},
        )

        with patch(
            "vlmrun.client.artifacts._follow_redirect",
            return_value=fake_redirect_resp,
        ):
            arts = Artifacts.__new__(Artifacts)
            arts._client = _StubClient()

            class _FakeRequestor:
                def request(self, **kwargs):
                    return (b"", 302, redirect_headers)

            arts._requestor = _FakeRequestor()

            result = arts.get(
                object_id="vid_f0e1d2",
                session_id="sess-003",
            )

            assert isinstance(result, Path)
            assert result.name == "vid_f0e1d2.mp4"
            assert result.read_bytes() == video_bytes

    def test_no_redirect_direct_response(self, tmp_path, monkeypatch):
        """A 200 response delivers bytes directly (no redirect)."""
        monkeypatch.setattr(
            "vlmrun.client.artifacts.VLMRUN_ARTIFACTS_CACHE_DIR", tmp_path
        )
        jpeg_bytes = _make_1x1_jpeg()

        arts = Artifacts.__new__(Artifacts)
        arts._client = _StubClient()

        class _FakeRequestor:
            def request(self, **kwargs):
                return (jpeg_bytes, 200, {"Content-Type": "image/jpeg"})

        arts._requestor = _FakeRequestor()

        result = arts.get(
            object_id="img_a1b2c3",
            session_id="sess-004",
        )
        assert isinstance(result, Image.Image)

    def test_redirect_without_location_raises(self):
        """A redirect without a Location header should raise ValueError."""
        arts = Artifacts.__new__(Artifacts)
        arts._client = _StubClient()

        class _FakeRequestor:
            def request(self, **kwargs):
                return (b"", 302, {})

        arts._requestor = _FakeRequestor()

        with pytest.raises(ValueError, match="without a Location header"):
            arts.get(
                object_id="img_a1b2c3",
                session_id="sess-005",
                raw_response=True,
            )

    def test_redirect_octet_stream_content_type_accepted(self, tmp_path, monkeypatch):
        """Cloud storage may return application/octet-stream for any file type."""
        monkeypatch.setattr(
            "vlmrun.client.artifacts.VLMRUN_ARTIFACTS_CACHE_DIR", tmp_path
        )
        pdf_bytes = b"%PDF-1.4 fake content"
        redirect_headers = {
            "Location": "https://s3.amazonaws.com/bucket/doc_abc123.pdf?sig=xyz",
        }
        fake_redirect_resp = _FakeResponse(
            content=pdf_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )

        with patch(
            "vlmrun.client.artifacts._follow_redirect",
            return_value=fake_redirect_resp,
        ):
            arts = Artifacts.__new__(Artifacts)
            arts._client = _StubClient()

            class _FakeRequestor:
                def request(self, **kwargs):
                    return (b"", 307, redirect_headers)

            arts._requestor = _FakeRequestor()

            result = arts.get(
                object_id="doc_abc123",
                session_id="sess-006",
            )

            assert isinstance(result, Path)
            assert result.name == "doc_abc123.pdf"
            assert result.read_bytes() == pdf_bytes
