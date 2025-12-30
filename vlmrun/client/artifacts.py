"""VLM Run API Artifacts resource."""

from __future__ import annotations

import io
import requests
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from PIL import Image
from pydantic import AnyHttpUrl

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.common.utils import _HEADERS
from vlmrun.constants import VLMRUN_ARTIFACTS_DIR


if TYPE_CHECKING:
    from vlmrun.types.abstract import VLMRunProtocol


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
        session_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        raw_response: bool = False,
    ) -> Union[bytes, Image.Image, AnyHttpUrl, Path]:
        """Get an artifact by session ID or execution ID and object ID.

        Supported artifact types:
            - img: Returns PIL.Image.Image (JPEG)
            - url: Returns AnyHttpUrl
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
            The artifact content - type depends on object_id prefix and raw_response flag

        Raises:
            ValueError: If neither session_id nor execution_id is provided, or if both are provided
        """
        if session_id is None and execution_id is None:
            raise ValueError("Either `session_id` or `execution_id` is required")
        if session_id is not None and execution_id is not None:
            raise ValueError(
                "Only one of `session_id` or `execution_id` is allowed, not both"
            )

        # Build query parameters, filtering out None values
        query_params = {"object_id": object_id}
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
        sess_id: str = session_id or execution_id
        artifacts_dir: Path = VLMRUN_ARTIFACTS_DIR / sess_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Extension and content-type mappings for file-based artifacts
        ext_mapping = {"vid": "mp4", "aud": "mp3", "doc": "pdf", "recon": "spz"}
        content_type_mapping = {
            "vid": "video/mp4",
            "aud": "audio/mpeg",
            "doc": "application/pdf",
            "recon": "application/octet-stream",
        }

        if obj_type == "img":
            assert (
                headers["Content-Type"] == "image/jpeg"
            ), f"Expected image/jpeg, got {headers['Content-Type']}"
            return Image.open(io.BytesIO(response)).convert("RGB")
        elif obj_type == "url":
            # Get the filename including extension frm the URL by stripping any query parameters
            url: AnyHttpUrl = AnyHttpUrl(response.decode("utf-8"))
            path: Path = Path(str(url))
            filename: str = path.name.split("?")[0]
            ext: str = filename.split(".")[-1].lower()
            tmp_path: Path = artifacts_dir / f"{filename}.{ext}"
            if tmp_path.exists():
                return tmp_path

            # Download the file, and move it to the appropriate path
            with requests.get(url, headers=_HEADERS, stream=True) as r:
                r.raise_for_status()
                with tmp_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return tmp_path
        elif obj_type in ("vid", "aud", "doc", "recon"):
            # Validate content type
            expected_content_type = content_type_mapping[obj_type]
            actual_content_type = headers.get("Content-Type")
            assert (
                actual_content_type == expected_content_type
            ), f"Expected {expected_content_type}, got {actual_content_type}"

            # Build file path with appropriate extension
            ext = ext_mapping.get(obj_type, None)
            if ext is None:
                raise IOError(
                    f"Unsupported file type [file_type={filename}, object_id={object_id}]"
                )
            tmp_path: Path = artifacts_dir / f"{object_id}.{ext}"

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
