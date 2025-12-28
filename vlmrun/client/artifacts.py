"""VLM Run API Artifacts resource."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Union

from PIL import Image
from pydantic import AnyHttpUrl

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.constants import VLMRUN_ARTIFACTS_DIR
from vlmrun.common.utils import download_artifact


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
        session_id: str | None = None,
        execution_id: str | None = None,
        raw_response: bool = False,
    ) -> Union[bytes, Image.Image, AnyHttpUrl, Path]:
        """Get an artifact by session ID and object ID.

        Args:
            object_id: Object ID for the artifact
            session_id: Session ID for the artifact
            execution_id: Execution ID for the artifact
            object_id: Object ID for the artifact
            raw_response: Whether to return the raw response or not

        Returns:
            bytes: The artifact content if raw_response is True, otherwise
            converted to the appropriate type based on the content type
        """
        if session_id is None and execution_id is None:
            raise ValueError("Either `session_id` or `execution_id` is required")
        if session_id is not None and execution_id is not None:
            raise ValueError(
                "Only one of `session_id` or `execution_id` is allowed, not both"
            )

        response, status_code, headers = self._requestor.request(
            method="GET",
            url="artifacts",
            data={
                "session_id": session_id,
                "execution_id": execution_id,
                "object_id": object_id,
            },
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

        # Create temporary path if not already created
        sess_id: str = session_id or execution_id
        tmp_path: Path = VLMRUN_ARTIFACTS_DIR / sess_id
        tmp_path.mkdir(parents=True, exist_ok=True)
        match obj_type:
            # In-memory image object
            case "img":
                assert (
                    headers["Content-Type"] == "image/jpeg"
                ), f"Expected image/jpeg, got {headers['Content-Type']}"
                return Image.open(io.BytesIO(response)).convert("RGB")
            # Download the URL and return the local path
            case "url":
                url: AnyHttpUrl = AnyHttpUrl(response.decode("utf-8"))
                path: Path = download_artifact(str(url), format="file")
                return path
            case "vid" | "aud" | "doc" | "recon":
                # Read the binary response as a video file and write it to a temporary file
                ext = {"vid": "mp4", "aud": "mp3", "doc": "pdf", "recon": "spz"}[
                    obj_type
                ]
                tmp_path = tmp_path / f"{object_id}.{ext}"
                with tmp_path.open("wb") as f:
                    f.write(response)
                return tmp_path
            case _:
                return response

    def list(self, session_id: str) -> None:
        """List artifacts for a session.

        Args:
            session_id: Session ID to list artifacts for

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Artifacts.list() is not yet implemented")
