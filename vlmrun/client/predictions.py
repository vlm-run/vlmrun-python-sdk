"""VLM Run API Prediction resource."""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
from loguru import logger

import time
from tqdm import tqdm
from vlmrun.common.image import encode_image
from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import (
    PredictionResponse,
    FileResponse,
    GenerationConfig,
    RequestMetadata,
)
from vlmrun.common.pydantic import schema_to_pydantic_model


class SchemaCastMixin:
    """Mixin class to handle schema casting for predictions."""

    def _cast_response_to_schema(
        self,
        prediction: PredictionResponse,
        domain: str,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        """Cast prediction response to appropriate schema.

        Args:
            prediction: PredictionResponse to cast
            domain: Domain identifier
            config: Optional GenerationConfig with custom schema
        """
        if prediction.status == "completed" and prediction.response:
            try:
                if config and hasattr(config, "json_schema"):
                    schema = schema_to_pydantic_model(config.json_schema)
                else:
                    schema = self._client.hub.get_pydantic_model(domain)

                if schema:
                    prediction.response = schema(**prediction.response)
            except Exception as e:
                logger.debug(f"Failed to cast response to schema: {e}")


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


class ImagePredictions(SchemaCastMixin, Predictions):
    """Image prediction resource for VLM Run API."""

    def generate(
        self,
        domain: str,
        images: Optional[List[Union[Path, Image.Image]]] = None,
        urls: Optional[List[str]] = None,
        batch: bool = False,
        metadata: Optional[RequestMetadata] = None,
        config: Optional[GenerationConfig] = None,
        callback_url: Optional[str] = None,
    ) -> PredictionResponse:
        """Generate a document prediction.

        Args:
            images: List of images to generate predictions from. Either images or urls must be provided.
            urls: List of image URLs to generate predictions from. Either images or urls must be provided.
            domain: Domain to use for prediction
            batch: Whether to run prediction in batch mode
            config: GenerateConfig to use for prediction
            metadata: Metadata to include in prediction
            callback_url: URL to call when prediction is complete

        Returns:
            PredictionResponse: Prediction response

        Raises:
            ValueError: If neither images nor urls are provided, or if both are provided
        """
        # Input validation
        if not images and not urls:
            raise ValueError("Either `images` or `urls` must be provided")
        if images and urls:
            raise ValueError("Only one of `images` or `urls` can be provided")

        if images:
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
            images_data = [encode_image(image, format="JPEG") for image in images]
        else:
            # URL handling
            if not urls:
                raise ValueError("URLs list cannot be empty")
            if not isinstance(urls[0], str):
                raise ValueError("URLs must be strings")
            if not all(isinstance(url, str) for url in urls):
                raise ValueError("All URLs must be strings")
            images_data = urls

        additional_kwargs = {}
        if config:
            additional_kwargs["config"] = config.model_dump()
        if metadata:
            additional_kwargs["metadata"] = metadata.model_dump()
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="image/generate",
            data={
                "images": images_data,
                "domain": domain,
                "batch": batch,
                "callback_url": callback_url,
                **additional_kwargs,
            },
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        prediction = PredictionResponse(**response)

        self._cast_response_to_schema(prediction, domain, config)
        return prediction


def FilePredictions(route: str):
    """File prediction resource for VLM Run API."""

    class _FilePredictions(SchemaCastMixin, Predictions):
        """File prediction resource for VLM Run API."""

        def generate(
            self,
            file: Optional[Union[Path, str]] = None,
            url: Optional[str] = None,
            domain: Optional[str] = None,
            batch: bool = False,
            config: Optional[GenerationConfig] = GenerationConfig(),
            metadata: Optional[RequestMetadata] = RequestMetadata(),
            callback_url: Optional[str] = None,
        ) -> PredictionResponse:
            """Generate a document prediction.

            Args:
                file: File (pathlib.Path) or file_id to generate prediction from
                url: URL to generate prediction from
                domain: Domain to use for prediction
                batch: Whether to run prediction in batch mode
                config: GenerateConfig to use for prediction
                metadata: Metadata to include in prediction
                callback_url: URL to call when prediction is complete

            Returns:
                PredictionResponse: Prediction response
            """
            is_url = False
            if not file and not url:
                raise ValueError("Either `file` or `url` must be provided")
            if file and url:
                raise ValueError("Only one of `file` or `url` can be provided")
            if file:
                if isinstance(file, Path) or (
                    isinstance(file, str) and Path(file).suffix
                ):
                    logger.debug(
                        f"Uploading file [path={file}, size={file.stat().st_size / 1024 / 1024:.2f} MB] to VLM Run"
                    )
                    response: FileResponse = self._client.files.upload(
                        file=Path(file), purpose="assistants"
                    )
                    logger.debug(
                        f"Uploaded file [file_id={response.id}, name={response.filename}]"
                    )
                    file_or_url = response.id
                elif isinstance(file, str):
                    logger.debug(f"Using file_id [file_id={file}]")
                    assert not Path(file).suffix, "File must not have an extension"
                    file_or_url = file
                else:
                    raise ValueError("File must be a pathlib.Path or a string")
            elif url:
                is_url = True
                file_or_url = url
            else:
                raise ValueError(
                    "File or URL must be a pathlib.Path, str, or AnyHttpUrl"
                )

            additional_kwargs = {}
            if config:
                additional_kwargs["config"] = config.model_dump()
            if metadata:
                additional_kwargs["metadata"] = metadata.model_dump()
            response, status_code, headers = self._requestor.request(
                method="POST",
                url=f"{route}/generate",
                data={
                    "url" if is_url else "file_id": file_or_url,
                    "domain": domain,
                    "batch": batch,
                    "callback_url": callback_url,
                    **additional_kwargs,
                },
            )
            if not isinstance(response, dict):
                raise TypeError("Expected dict response")
            prediction = PredictionResponse(**response)

            self._cast_response_to_schema(prediction, domain, config)
            return prediction

    return _FilePredictions


DocumentPredictions = FilePredictions("document")
AudioPredictions = FilePredictions("audio")
VideoPredictions = FilePredictions("video")
