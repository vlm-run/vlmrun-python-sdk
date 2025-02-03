"""VLM Run API Prediction resource."""

from __future__ import annotations
from pathlib import Path
from PIL import Image
from loguru import logger

import time
from tqdm import tqdm
from vlmrun.common.image import encode_image
from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import PredictionResponse, FileResponse


class Predictions:
    """Predictions resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Predictions resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client, timeout=120)

    def list(self, skip: int = 0, limit: int = 10) -> list[PredictionResponse]:
        """List all predictions.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            List[FileResponse]: List of file objects
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="predictions",
            params={"skip": skip, "limit": limit},
        )
        return [PredictionResponse(**prediction) for prediction in response]

    def get(self, id: str) -> PredictionResponse:
        """Get prediction by ID.

        Args:
            id: ID of prediction to retrieve

        Returns:
            PredictionResponse: Prediction metadata
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"predictions/{id}",
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return PredictionResponse(**response)

    def wait(self, id: str, timeout: int = 60, sleep: int = 1) -> PredictionResponse:
        """Wait for prediction to complete.

        Args:
            id: ID of prediction to wait for
            timeout: Timeout in seconds
            sleep: Sleep time in seconds
        """
        for _ in tqdm(range(timeout), desc="Waiting for prediction to complete"):
            response: PredictionResponse = self.get(id)
            if response.status == "completed":
                return response
            time.sleep(sleep)
        raise TimeoutError(f"Prediction {id} did not complete within {timeout} seconds")


class ImagePredictions(Predictions):
    """Image prediction resource for VLM Run API."""

    def generate(
        self,
        images: list[Path | Image.Image],
        model: str,
        domain: str,
        json_schema: dict | None = None,
        detail: str = "auto",
        batch: bool = False,
        metadata: dict = {},
        callback_url: str | None = None,
    ) -> PredictionResponse:
        """Generate a document prediction.

        Args:
            images: List of images to generate predictions from
            model: Model to use for prediction
            domain: Domain to use for prediction
            json_schema: JSON schema to use for prediction
            detail: Detail level for prediction
            batch: Whether to run prediction in batch mode
            metadata: Metadata to include in prediction
            callback_url: URL to call when prediction is complete

        Returns:
            PredictionResponse: Prediction response
        """

        # Check if all images are of the same type
        image_type = type(images[0])
        if not all(isinstance(image, image_type) for image in images):
            raise ValueError("All images must be of the same type")
        if isinstance(images[0], Path):
            images = [Image.open(str(image)) for image in images]
        elif isinstance(images[0], Image.Image):
            pass
        else:
            raise ValueError("Image must be a path or a PIL Image")

        response, status_code, headers = self._requestor.request(
            method="POST",
            url="image/generate",
            data={
                "image": encode_image(images[0], format="JPEG"),
                "model": model,
                "domain": domain,
                "json_schema": json_schema,
                "detail": detail,
                "batch": batch,
                "metadata": metadata,
                "callback_url": callback_url,
            },
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return PredictionResponse(**response)


def FilePredictions(route: str):
    """File prediction resource for VLM Run API."""

    class _FilePredictions(Predictions):
        """File prediction resource for VLM Run API."""

        def generate(
            self,
            file_or_url: str | Path,
            model: str,
            domain: str,
            json_schema: dict | None = None,
            detail: str = "auto",
            batch: bool = False,
            metadata: dict = {},
            callback_url: str | None = None,
        ) -> PredictionResponse:
            """Generate a document prediction.

            Args:
                file_or_url: File (pathlib.Path) or file_id or URL to generate prediction from
                model: Model to use for prediction
                domain: Domain to use for prediction
                json_schema: JSON schema to use for prediction
                detail: Detail level for prediction
                batch: Whether to run prediction in batch mode
                metadata: Metadata to include in prediction
                callback_url: URL to call when prediction is complete

            Returns:
                PredictionResponse: Prediction response
            """
            is_url = False
            if isinstance(file_or_url, Path):
                logger.debug(
                    f"Uploading file [path={file_or_url}, size={file_or_url.stat().st_size / 1024 / 1024:.2f} MB] to VLM Run"
                )
                upload_response, _, _ = self._client.files.upload(
                    file=file_or_url, purpose="assistants"
                )
                if not isinstance(upload_response, dict):
                    raise TypeError("Expected dict response")
                response = FileResponse(**upload_response)
                logger.debug(
                    f"Uploaded file [file_id={response.id}, name={response.filename}]"
                )
                file_or_url = response.id
            elif isinstance(file_or_url, str):
                is_url = str(file_or_url).startswith(("http://", "https://"))
            else:
                raise ValueError(
                    "File or URL must be a pathlib.Path, str, or AnyHttpUrl"
                )

            response, status_code, headers = self._requestor.request(
                method="POST",
                url=f"{route}/generate",
                data={
                    "url" if is_url else "file_id": file_or_url,
                    "model": model,
                    "domain": domain,
                    "json_schema": json_schema,
                    "detail": detail,
                    "batch": batch,
                    "metadata": metadata,
                    "callback_url": callback_url,
                },
            )
            if not isinstance(response, dict):
                raise TypeError("Expected dict response")
            return PredictionResponse(**response)

    return _FilePredictions


DocumentPredictions = FilePredictions("document")
AudioPredictions = FilePredictions("audio")
VideoPredictions = FilePredictions("video")
