"""VLM Run API Artifacts resource."""

from __future__ import annotations

import io
from email.parser import HeaderParser
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

from PIL import Image
from pydantic import AnyHttpUrl

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.constants import VLMRUN_CACHE_DIR


def _get_disposition_params(disposition: str) -> Dict[str, str]:
    """Parse Content-Disposition header and return parameters.

    Args:
        disposition: Content-Disposition header value

    Returns:
        Dictionary of parameters from the header
    """
    parser = HeaderParser()
    msg = parser.parsestr(f"Content-Disposition: {disposition}")
    params = msg.get_params(header="Content-Disposition", unquote=True)
    if not params:
        return {}
    # First element is the main value (e.g., "attachment"), rest are params
    return dict(params[1:])


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
        self, session_id: str, object_id: str, raw_response: bool = False
    ) -> Union[bytes, Image.Image, AnyHttpUrl, Path]:
        """Get an artifact by session ID and object ID.

        Args:
            session_id: Session ID for the artifact
            object_id: Object ID for the artifact
            raw_response: Whether to return the raw response or not

        Returns:
            bytes: The artifact content if raw_response is True, otherwise
            converted to the appropriate type based on the content type
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"artifacts/{session_id}/{object_id}",
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

        if obj_type == "img":
            assert (
                headers["Content-Type"] == "image/jpeg"
            ), f"Expected image/jpeg, got {headers['Content-Type']}"
            return Image.open(io.BytesIO(response)).convert("RGB")
        elif obj_type == "url":
            return AnyHttpUrl(response.decode("utf-8"))
        elif obj_type == "vid":
            # Read the binary response as a video file and write it to a temporary file
            assert (
                headers["Content-Type"] == "video/mp4"
            ), f"Expected video/mp4, got {headers['Content-Type']}"

            # Parse Content-Disposition header to get file extension
            ext = "mp4"  # default extension
            disposition = headers.get("Content-Disposition") or headers.get(
                "content-disposition"
            )
            if disposition:
                params = _get_disposition_params(disposition)
                filename = params.get("filename")
                if filename:
                    suffix = Path(filename).suffix
                    if suffix:
                        ext = suffix.lstrip(".")

            # Build cache path with session_id and object_id
            safe_session_id = session_id.replace("-", "")
            tmp_path: Path = VLMRUN_CACHE_DIR / f"{safe_session_id}_{object_id}.{ext}"

            # Return cached version if it exists
            if tmp_path.exists():
                return tmp_path

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
