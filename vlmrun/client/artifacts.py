"""VLM Run API Artifacts resource."""

from __future__ import annotations

import io
import logging
import requests
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from PIL import Image
from pydantic import AnyHttpUrl

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.common.utils import _HEADERS
from vlmrun.constants import VLMRUN_ARTIFACTS_CACHE_DIR


if TYPE_CHECKING:
    from vlmrun.types.abstract import VLMRunProtocol

logger = logging.getLogger(__name__)

_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}

_CONTENT_TYPE_EXT_MAP: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "video/mp4": "mp4",
    "audio/mpeg": "mp3",
    "application/pdf": "pdf",
    "application/octet-stream": "bin",
}


def _filename_from_url(url: str) -> str:
    """Extract a clean filename from a URL, stripping query parameters.

    Args:
        url: The URL to extract a filename from.

    Returns:
        The filename portion of the URL path with query params removed.
    """
    parsed = urlparse(url)
    return Path(parsed.path).name


def _follow_redirect(url: str, *, stream: bool = False) -> requests.Response:
    """Follow a redirect URL without authorization headers.

    Presigned cloud-storage URLs (S3, GCS, etc.) carry their own auth via
    query parameters and will reject requests that also include an
    ``Authorization`` header.  This helper fetches the URL using only the
    standard browser-like headers defined in ``_HEADERS``.

    Args:
        url: The redirect target URL.
        stream: Whether to stream the response body.

    Returns:
        The HTTP response from the redirect target.

    Raises:
        requests.HTTPError: If the downstream request fails.
    """
    response = requests.get(url, headers=_HEADERS, stream=stream, allow_redirects=True)
    response.raise_for_status()
    return response


class Artifacts:
    """Artifacts resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Artifacts resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def get(
        self,
        object_id: str,
        session_id: str | None = None,
        execution_id: str | None = None,
        raw_response: bool = False,
    ) -> bytes | Image.Image | AnyHttpUrl | Path:
        """Get an artifact by session ID or execution ID and object ID.

        The API may respond with the artifact bytes directly **or** with an
        HTTP redirect (301–308) to a presigned cloud-storage URL.  When a
        redirect is received the SDK follows it transparently, stripping the
        ``Authorization`` header so that presigned-URL authentication is not
        disrupted.

        Supported artifact types:
            - img: Returns PIL.Image.Image (JPEG/PNG)
            - url: Returns Path to the downloaded file
            - vid: Returns Path to MP4 file
            - aud: Returns Path to MP3 file
            - doc: Returns Path to PDF file
            - recon: Returns Path to SPZ file

        Args:
            object_id: Object ID for the artifact (format: <type>_<6-hex-chars>)
            session_id: Session ID for the artifact (mutually exclusive with execution_id)
            execution_id: Execution ID for the artifact (mutually exclusive with session_id)
            raw_response: Whether to return the raw response bytes

        Returns:
            The artifact content — type depends on object_id prefix and raw_response flag.

        Raises:
            ValueError: If neither session_id nor execution_id is provided, or if both are provided.
            TypeError: If the response body is not bytes.
        """
        if session_id is None and execution_id is None:
            raise ValueError("Either `session_id` or `execution_id` is required")
        if session_id is not None and execution_id is not None:
            raise ValueError(
                "Only one of `session_id` or `execution_id` is allowed, not both"
            )

        # Build query parameters, filtering out None values
        query_params: dict[str, str] = {"object_id": object_id}
        if session_id is not None:
            query_params["session_id"] = session_id
        if execution_id is not None:
            query_params["execution_id"] = execution_id

        response, status_code, headers = self._requestor.request(
            method="GET",
            url="artifacts",
            params=query_params,
            raw_response=True,
            allow_redirects=False,
        )

        # --- Handle redirect responses (e.g. presigned cloud-storage URLs) ---
        if status_code in _REDIRECT_STATUS_CODES:
            redirect_url = headers.get("Location") or headers.get("location")
            if not redirect_url:
                raise ValueError(
                    f"Received redirect ({status_code}) without a Location header"
                )
            logger.debug(f"Following artifact redirect to {redirect_url}")
            redirect_resp = _follow_redirect(redirect_url)
            response = redirect_resp.content
            headers = dict(redirect_resp.headers)
            status_code = redirect_resp.status_code

        if not isinstance(response, bytes):
            raise TypeError("Expected bytes response")

        # If raw response is requested, return the raw response as bytes
        if raw_response:
            return response

        # Otherwise, return the appropriate type based on the content type
        obj_type, _obj_id = object_id.split("_")
        if len(_obj_id) != 6:
            raise ValueError(
                f"Invalid object ID: {object_id}, expected format: <obj_type>_<6-digit-hex-string>"
            )

        # Create artifacts directory with session_id subdirectory
        sess_id: str = session_id or execution_id  # type: ignore[assignment]
        artifacts_dir: Path = VLMRUN_ARTIFACTS_CACHE_DIR / sess_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Extension and content-type mappings for file-based artifacts
        ext_mapping: dict[str, str] = {
            "vid": "mp4",
            "aud": "mp3",
            "doc": "pdf",
            "recon": "spz",
        }
        content_type_mapping: dict[str, str] = {
            "vid": "video/mp4",
            "aud": "audio/mpeg",
            "doc": "application/pdf",
            "recon": "application/octet-stream",
        }

        if obj_type == "img":
            content_type = headers.get("Content-Type", "")
            allowed_img_types = ("image/jpeg", "image/png", "application/octet-stream")
            assert (
                content_type in allowed_img_types
            ), f"Expected one of {allowed_img_types}, got {content_type}"
            return Image.open(io.BytesIO(response)).convert("RGB")
        elif obj_type == "url":
            # Decode the URL from the response body
            url: AnyHttpUrl = AnyHttpUrl(response.decode("utf-8"))
            filename: str = _filename_from_url(str(url))
            tmp_path: Path = artifacts_dir / filename
            if tmp_path.exists():
                return tmp_path

            # Download the file via the URL (no auth headers for presigned URLs)
            with _follow_redirect(str(url), stream=True) as r:
                with tmp_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return tmp_path
        elif obj_type in ("vid", "aud", "doc", "recon"):
            # Validate content type (allow application/octet-stream for
            # cloud-storage responses that don't set a specific MIME type)
            expected_content_type = content_type_mapping[obj_type]
            actual_content_type = headers.get("Content-Type", "")
            if actual_content_type not in (
                expected_content_type,
                "application/octet-stream",
            ):
                logger.warning(
                    f"Unexpected Content-Type for {obj_type} artifact: "
                    f"expected {expected_content_type}, got {actual_content_type}"
                )

            # Build file path with appropriate extension
            ext = ext_mapping[obj_type]
            tmp_path = artifacts_dir / f"{object_id}.{ext}"

            # Return cached version if it exists
            if tmp_path.exists():
                return tmp_path

            # Write the binary response to file
            with tmp_path.open("wb") as f:
                f.write(response)
            return tmp_path
        else:
            return response

    def list(self, session_id: str) -> None:
        """List artifacts for a session.

        Args:
            session_id: Session ID to list artifacts for

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Artifacts.list() is not yet implemented")
