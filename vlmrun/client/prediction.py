"""VLM Run API Prediction resource."""

from __future__ import annotations
from pathlib import Path
from PIL import Image

import time
from tqdm import tqdm
from typing import Literal

from vlmrun.common.image import encode_image
from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import Client
from vlmrun.client.types import PredictionResponse


class Prediction:
    """Prediction resource for VLM Run API."""

    def __init__(self, client: "Client") -> None:
        """Initialize Prediction resource with client.

        Args:
            client: VLM Run API client instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def get(self, id: str) -> PredictionResponse:
        """Get prediction by ID.

        Args:
            id: ID of prediction to retrieve

        Returns:
            PredictionResponse: Prediction metadata
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"response/{id}",
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


class ImagePrediction(Prediction):
    """Image prediction resource for VLM Run API."""

    def generate(
        self,
        image: str | Path | Image.Image,
        model: str,
        domain: str,
        json_schema: dict | None = None,
        detail: Literal["auto", "lo", "hi"] = "auto",
        batch: bool = False,
        metadata: dict = {},
        callback_url: str = None,
    ) -> PredictionResponse:
        """Generate a document prediction.

        Args:
            image: Image to generate prediction from
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

        if isinstance(image, Path):
            image = Image.open(image)
        elif isinstance(image, str):
            raise ValueError("Image must be a path or a PIL Image")

        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image")

        response, status_code, headers = self._requestor.request(
            method="POST",
            url="image/generate",
            data={
                "image": encode_image(image, format="jpeg"),
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


class DocumentPrediction(Prediction):
    """Document prediction resource for VLM Run API."""

    def generate(
        self,
        file_id: str,
        model: str,
        domain: str,
        json_schema: dict | None = None,
        detail: Literal["auto", "lo", "hi"] = "auto",
        batch: bool = False,
        metadata: dict = {},
        callback_url: str = None,
    ) -> PredictionResponse:
        """Generate a document prediction.

        Args:
            file_id: ID of the file to generate prediction from
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
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="document/generate",
            data={
                "file_id": file_id,
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


class VideoPrediction(Prediction):
    """Video prediction resource for VLM Run API."""

    def generate(
        self,
        file_id: str,
        model: str,
        domain: str,
        json_schema: dict | None = None,
        detail: Literal["auto", "lo", "hi"] = "auto",
        batch: bool = False,
        metadata: dict = {},
        callback_url: str = None,
    ) -> PredictionResponse:
        """Generate a document prediction.

        Args:
            file_id: ID of the file to generate prediction from
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
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="video/generate",
            data={
                "file_id": file_id,
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
