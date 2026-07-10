"""VLM Run API Artifacts resource."""

from __future__ import annotations

import io
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Union

import requests
from PIL import Image
from pydantic import AnyHttpUrl

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.client.types import ArtifactListResponse
from vlmrun.common.utils import _HEADERS
from vlmrun.constants import VLMRUN_ARTIFACTS_CACHE_DIR

if TYPE_CHECKING:
    from vlmrun.types.abstract import VLMRunProtocol

# Object refs are `<type>_<6hex>` and may optionally include a file extension
# (e.g. `img_4c129a.jpg`, `vid_4d0e56.mp4`) in preview tags / markdown.
_OBJECT_ID_RE = re.compile(
    r"^(?P<prefix>[a-z]+)_(?P<hex>[0-9a-f]{6})(?:\.[a-z0-9]+)?$",
    re.IGNORECASE,
)

_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/gif",
        "application/octet-stream",
    }
)

_CONTENT_TYPE_MAPPING = {
    "vid": frozenset({"video/mp4", "application/octet-stream", "binary/octet-stream"}),
    "aud": frozenset(
        {"audio/mpeg", "audio/mp3", "application/octet-stream", "binary/octet-stream"}
    ),
    "doc": frozenset(
        {"application/pdf", "application/octet-stream", "binary/octet-stream"}
    ),
    "recon": frozenset({"application/octet-stream", "binary/octet-stream"}),
}

_EXT_MAPPING = {"vid": "mp4", "aud": "mp3", "doc": "pdf", "recon": "spz"}


def normalize_object_id(object_id: str) -> tuple[str, str]:
    """Normalize an artifact object ID, stripping any file extension.

    Args:
        object_id: Raw object ID, optionally with an extension
            (e.g. ``img_a1b2c3`` or ``img_a1b2c3.jpg``).

    Returns:
        Tuple of ``(obj_type, normalized_object_id)`` where the ID has no extension.

    Raises:
        ValueError: If the object ID does not match ``<type>_<6hex>[.ext]``.
    """
    match = _OBJECT_ID_RE.match(object_id.strip())
    if not match:
        raise ValueError(
            f"Invalid object ID: {object_id}, expected format: "
            "<obj_type>_<6-digit-hex-string> with optional file extension"
        )
    obj_type = match.group("prefix").lower()
    hex_id = match.group("hex").lower()
    return obj_type, f"{obj_type}_{hex_id}"


def _content_type_base(headers: dict[str, str]) -> str:
    """Return the Content-Type header without parameters (e.g. charset)."""
    raw = headers.get("Content-Type") or headers.get("content-type") or ""
    return raw.split(";", 1)[0].strip().lower()


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
        object_id: str | None = None,
        session_id: str | None = None,
        execution_id: str | None = None,
        filename: str | None = None,
        raw_response: bool = False,
    ) -> Union[bytes, Image.Image, AnyHttpUrl, Path]:
        """Get an artifact by session ID or execution ID and object ID or filename.

        Supported artifact types:
            - img: Returns PIL.Image.Image
            - url: Returns AnyHttpUrl
            - vid: Returns Path to MP4 file
            - aud: Returns Path to MP3 file
            - doc: Returns Path to PDF file
            - recon: Returns Path to SPZ file

        Args:
            object_id: Object ID for the artifact (format: ``<type>_<6-hex-chars>``,
                optionally with a file extension such as ``.jpg`` / ``.mp4``).
                Mutually exclusive with filename.
            session_id: Session ID for the artifact (mutually exclusive with execution_id)
            execution_id: Execution ID for the artifact (mutually exclusive with session_id)
            filename: Workspace or manifest filename to retrieve. Mutually exclusive
                with object_id.
            raw_response: Whether to return the raw response bytes

        Returns:
            The artifact content - type depends on object_id prefix and raw_response flag

        Raises:
            ValueError: If neither session_id nor execution_id is provided, or if both are provided,
                or if neither object_id nor filename is provided, or if both are provided.
        """
        if session_id is None and execution_id is None:
            raise ValueError("Either `session_id` or `execution_id` is required")
        if session_id is not None and execution_id is not None:
            raise ValueError(
                "Only one of `session_id` or `execution_id` is allowed, not both"
            )
        if object_id is None and filename is None:
            raise ValueError("Either `object_id` or `filename` is required")
        if object_id is not None and filename is not None:
            raise ValueError(
                "Only one of `object_id` or `filename` is allowed, not both"
            )

        query_params: dict[str, str] = {}
        normalized_id: str | None = None
        obj_type: str | None = None
        if object_id is not None:
            obj_type, normalized_id = normalize_object_id(object_id)
            query_params["object_id"] = normalized_id
        if filename is not None:
            query_params["filename"] = filename
        if session_id is not None:
            query_params["session_id"] = session_id
        if execution_id is not None:
            query_params["execution_id"] = execution_id

        response, status_code, headers = self._requestor.request(
            method="GET",
            url="artifacts",
            params=query_params,
            raw_response=True,
        )

        if not isinstance(response, bytes):
            raise TypeError("Expected bytes response")

        if raw_response:
            return response

        # If filename was used instead of object_id, return raw bytes
        if object_id is None:
            return response

        assert obj_type is not None and normalized_id is not None

        sess_id: str = session_id or execution_id  # type: ignore[assignment]
        artifacts_dir: Path = VLMRUN_ARTIFACTS_CACHE_DIR / sess_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        content_type = _content_type_base(headers)

        if obj_type == "img":
            if content_type and content_type not in _IMAGE_CONTENT_TYPES:
                warnings.warn(
                    f"Unexpected Content-Type for image artifact {normalized_id}: "
                    f"{content_type!r}; attempting to decode as an image anyway",
                    UserWarning,
                    stacklevel=2,
                )
            return Image.open(io.BytesIO(response)).convert("RGB")
        elif obj_type == "url":
            url: AnyHttpUrl = AnyHttpUrl(response.decode("utf-8"))
            path: Path = Path(str(url))
            url_filename: str = path.name.split("?")[0]
            ext: str = url_filename.split(".")[-1].lower()
            tmp_path: Path = artifacts_dir / f"{url_filename}.{ext}"
            if tmp_path.exists():
                return tmp_path

            with requests.get(str(url), headers=_HEADERS, stream=True) as r:
                r.raise_for_status()
                with tmp_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return tmp_path
        elif obj_type in ("vid", "aud", "doc", "recon"):
            expected = _CONTENT_TYPE_MAPPING[obj_type]
            if content_type and content_type not in expected:
                warnings.warn(
                    f"Unexpected Content-Type for {obj_type} artifact {normalized_id}: "
                    f"expected one of {sorted(expected)}, got {content_type!r}; "
                    "saving bytes anyway",
                    UserWarning,
                    stacklevel=2,
                )

            ext = _EXT_MAPPING.get(obj_type)
            if ext is None:
                raise IOError(f"Unsupported file type [object_id={normalized_id}]")
            tmp_path = artifacts_dir / f"{normalized_id}.{ext}"

            if tmp_path.exists():
                return tmp_path

            with tmp_path.open("wb") as f:
                f.write(response)
            return tmp_path
        else:
            return response

    def list(
        self,
        session_id: str | None = None,
        execution_id: str | None = None,
    ) -> ArtifactListResponse:
        """List artifacts for a session or execution.

        Args:
            session_id: Session ID to list artifacts for (mutually exclusive with execution_id)
            execution_id: Execution ID to list artifacts for (mutually exclusive with session_id)

        Returns:
            ArtifactListResponse containing the namespace ID and list of artifact items.

        Raises:
            ValueError: If neither session_id nor execution_id is provided, or if both are provided.
        """
        if session_id is None and execution_id is None:
            raise ValueError("Either `session_id` or `execution_id` is required")
        if session_id is not None and execution_id is not None:
            raise ValueError(
                "Only one of `session_id` or `execution_id` is allowed, not both"
            )

        query_params: dict[str, str] = {}
        if session_id is not None:
            query_params["session_id"] = session_id
        if execution_id is not None:
            query_params["execution_id"] = execution_id

        response, _, _ = self._requestor.request(
            method="GET",
            url="artifacts/list",
            params=query_params,
        )
        return ArtifactListResponse.model_validate(response)
