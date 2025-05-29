"""VLM Run API Prediction resource."""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
from loguru import logger

import time
from vlmrun.common.image import encode_image, _open_image_with_exif
from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import (
    PredictionResponse,
    FileResponse,
    GenerationConfig,
    RequestMetadata,
    SchemaResponse,
    MarkdownDocument,
)
from typing import Type
from pydantic import BaseModel
import cachetools
from cachetools.keys import hashkey


@cachetools.cached(
    cache=cachetools.TTLCache(maxsize=100, ttl=3600),
    key=lambda _client, domain, config: hashkey(
        domain, json.dumps(config.model_dump())
    ),  # noqa: B007
)
def get_response_model(
    client, domain: str, config: Optional[GenerationConfig] = None
) -> Type[BaseModel]:
    """Get the schema type for a domain and a generation config.

    Note: This function is cached to avoid re-fetching the schema from the API.
    """
    schema_response: SchemaResponse = client.get_schema(domain, config=config)
    return schema_response.response_model


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
                response_model: Type[BaseModel] = get_response_model(
                    self._client, domain, config
                )

                if response_model:
                    prediction.response = response_model(**prediction.response)
            except Exception as e:
                logger.warning(f"Failed to cast response to schema: {e}")


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
        prediction = PredictionResponse(**response)

        # Always cast document.markdown responses to MarkdownDocument
        if (
            prediction.domain == "document.markdown"
            and prediction.status == "completed"
            and prediction.response
        ):
            try:
                prediction.response = MarkdownDocument(**prediction.response)
            except Exception as e:
                logger.warning(f"Failed to cast response to MarkdownDocument: {e}")

        return prediction

    def wait(self, id: str, timeout: int = 300, sleep: int = 5) -> PredictionResponse:
        """Wait for prediction to complete.

        Args:
            id: ID of prediction to wait for
            timeout: Maximum number of seconds to wait
            sleep: Time to wait between checks in seconds (default: 5)

        Returns:
            PredictionResponse: Completed prediction

        Raises:
            TimeoutError: If prediction does not complete within timeout
        """
        start_time = time.time()
        while True:
            response: PredictionResponse = self.get(id)
            if response.status == "completed":
                return response

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Prediction {id} did not complete within {timeout} seconds. Last status: {response.status}"
                )

            time.sleep(min(sleep, timeout - elapsed))


class ImagePredictions(SchemaCastMixin, Predictions):
    """Image prediction resource for VLM Run API."""

    @staticmethod
    def _handle_images_or_urls(
        images: Optional[List[Union[Path, Image.Image]]] = None,
        urls: Optional[List[str]] = None,
    ) -> List[str]:
        """Handle images and URLs.

        Args:
            images: List of images to handle
            urls: List of URLs to handle
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
                images = [_open_image_with_exif(str(image)) for image in images]
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
        return images_data

    def execute(
        self,
        name: str,
        version: str = "latest",
        images: Optional[List[Union[Path, Image.Image]]] = None,
        urls: Optional[List[str]] = None,
        batch: bool = False,
        metadata: Optional[RequestMetadata] = None,
        config: Optional[GenerationConfig] = None,
        callback_url: Optional[str] = None,
        autocast: bool = False,
    ) -> PredictionResponse:
        """Generate a document prediction using a named model.

        Args:
            name: Name of the model to use
            version: Version of the model to use
            images: List of file paths (Path) or PIL Image objects to process. Either images or urls must be provided.
            urls: List of HTTP URLs pointing to images. Either images or urls must be provided.
            batch: Whether to run prediction in batch mode
            metadata: Metadata to include in prediction
            config: GenerateConfig to use for prediction
            callback_url: URL to call when prediction is complete

        Returns:
            PredictionResponse: Prediction response

        Raises:
            ValueError: If neither images nor urls are provided, or if both are provided
        """
        images_data = self._handle_images_or_urls(images, urls)
        additional_kwargs = {}
        if config:
            additional_kwargs["config"] = config.model_dump()
        if metadata:
            additional_kwargs["metadata"] = metadata.model_dump()
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="image/execute",
            data={
                "name": name,
                "version": version,
                "images": images_data,
                "batch": batch,
                "callback_url": callback_url,
                **additional_kwargs,
            },
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        prediction = PredictionResponse(**response)

        if autocast:
            self._cast_response_to_schema(prediction, name, config)
        return prediction

    def generate(
        self,
        domain: str,
        images: Optional[List[Union[Path, Image.Image]]] = None,
        urls: Optional[List[str]] = None,
        model: str = "vlm-1",
        batch: bool = False,
        metadata: Optional[RequestMetadata] = None,
        config: Optional[GenerationConfig] = None,
        callback_url: Optional[str] = None,
        autocast: bool = False,
    ) -> PredictionResponse:
        """Generate a document prediction.

        Args:
            domain: Domain to use for prediction
            images: List of file paths (Path) or PIL Image objects to process. Either images or urls must be provided.
            urls: List of HTTP URLs pointing to images. Either images or urls must be provided.
            model: Model to use for prediction
            batch: Whether to run prediction in batch mode
            metadata: Metadata to include in prediction
            config: GenerateConfig to use for prediction
            callback_url: URL to call when prediction is complete
            autocast: Whether to autocast the response to the schema

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
                images = [_open_image_with_exif(str(image)) for image in images]
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
            if not all(url.startswith("http") for url in urls):
                raise ValueError("URLs must start with 'http'")
            images_data = urls

        images_data = self._handle_images_or_urls(images, urls)
        additional_kwargs = {}
        if config:
            additional_kwargs["config"] = config.model_dump()
        if metadata:
            additional_kwargs["metadata"] = metadata.model_dump()
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="image/generate",
            data={
                "model": model,
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

        if autocast:
            self._cast_response_to_schema(prediction, domain, config)
        return prediction

    def schema(
        self,
        images: Optional[List[Union[Path, Image.Image]]] = None,
        urls: Optional[List[str]] = None,
    ) -> PredictionResponse:
        """Auto-generate a schema for a given image or document.

        Args:
            images: List of images to generate the schema from
            urls: List of URLs to generate the schema from

        Returns:
            PredictionResponse: Prediction response
        """
        images_data = self._handle_images_or_urls(images, urls)
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="image/schema",
            data={"images": images_data},
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        prediction = PredictionResponse(**response)
        prediction.response = SchemaResponse(**prediction.response)
        return prediction


def FilePredictions(route: str):
    """File prediction resource for VLM Run API."""

    class _FilePredictions(SchemaCastMixin, Predictions):
        """File prediction resource for VLM Run API."""

        def _handle_file_or_url(
            self,
            file: Optional[Union[Path, str]] = None,
            url: Optional[str] = None,
        ) -> tuple[bool, str]:
            """Handle file or URL."""
            is_url = False
            file_or_url = None
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
            return is_url, file_or_url

        def generate(
            self,
            file: Optional[Union[Path, str]] = None,
            url: Optional[str] = None,
            model: str = "vlm-1",
            domain: Optional[str] = None,
            batch: bool = False,
            config: Optional[GenerationConfig] = GenerationConfig(),
            metadata: Optional[RequestMetadata] = RequestMetadata(),
            callback_url: Optional[str] = None,
            autocast: bool = False,
        ) -> PredictionResponse:
            """Generate a document prediction.

            Args:
                model: Model to use for prediction
                file: File (pathlib.Path) or file_id to generate prediction from
                url: URL to generate prediction from
                domain: Domain to use for prediction
                batch: Whether to run prediction in batch mode
                config: GenerateConfig to use for prediction
                metadata: Metadata to include in prediction
                callback_url: URL to call when prediction is complete
                autocast: Whether to autocast the response to the schema

            Returns:
                PredictionResponse: Prediction response
            """
            is_url, file_or_url = self._handle_file_or_url(file, url)

            additional_kwargs = {}
            if config:
                additional_kwargs["config"] = config.model_dump()
            if metadata:
                additional_kwargs["metadata"] = metadata.model_dump()
            response, status_code, headers = self._requestor.request(
                method="POST",
                url=f"{route}/generate",
                data={
                    "model": model,
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

            # Always cast document.markdown responses to MarkdownDocument
            if (
                domain == "document.markdown"
                and prediction.status == "completed"
                and prediction.response
            ):
                try:
                    prediction.response = MarkdownDocument(**prediction.response)
                except Exception as e:
                    logger.warning(f"Failed to cast response to MarkdownDocument: {e}")
            # Handle other domains with autocast
            elif autocast:
                self._cast_response_to_schema(prediction, domain, config)
            return prediction

        def execute(
            self,
            name: str,
            version: str = "latest",
            file: Optional[Union[Path, str]] = None,
            url: Optional[str] = None,
            batch: bool = False,
            config: Optional[GenerationConfig] = GenerationConfig(),
            metadata: Optional[RequestMetadata] = RequestMetadata(),
            callback_url: Optional[str] = None,
            autocast: bool = False,
        ) -> PredictionResponse:
            """Generate a document prediction using a named model.

            Args:
                name: Name of the model to execute
                version: Version of the model to execute
                file: File (pathlib.Path) or file_id to generate prediction from
                url: URL to generate prediction from
                batch: Whether to run prediction in batch mode
                config: GenerateConfig to use for prediction
                metadata: Metadata to include in prediction
                callback_url: URL to call when prediction is complete
                autocast: Whether to autocast the response to the schema

            Returns:
                PredictionResponse: Prediction response
            """
            is_url, file_or_url = self._handle_file_or_url(file, url)

            additional_kwargs = {}
            if config:
                additional_kwargs["config"] = config.model_dump()
            if metadata:
                additional_kwargs["metadata"] = metadata.model_dump()
            response, status_code, headers = self._requestor.request(
                method="POST",
                url=f"{route}/execute",
                data={
                    "name": name,
                    "version": version,
                    "url" if is_url else "file_id": file_or_url,
                    "batch": batch,
                    "callback_url": callback_url,
                    **additional_kwargs,
                },
            )
            if not isinstance(response, dict):
                raise TypeError("Expected dict response")
            prediction = PredictionResponse(**response)

            if autocast:
                self._cast_response_to_schema(prediction, name, config)
            return prediction

        def schema(
            self,
            file: Optional[Union[Path, str]] = None,
            url: Optional[str] = None,
        ) -> PredictionResponse:
            """Auto-generate a schema for a given document.

            Args:
                file: File (pathlib.Path) or file_id to generate the schema from
                url: URL to generate the schema from

            Returns:
                PredictionResponse: Prediction response
            """
            is_url, file_or_url = self._handle_file_or_url(file, url)
            response, status_code, headers = self._requestor.request(
                method="POST",
                url=f"{route}/schema",
                data={"url" if is_url else "file_id": file_or_url},
            )
            if not isinstance(response, dict):
                raise TypeError("Expected dict response")
            prediction = PredictionResponse(**response)
            prediction.response = SchemaResponse(**prediction.response)
            return prediction

    return _FilePredictions


DocumentPredictions = FilePredictions("document")
AudioPredictions = FilePredictions("audio")
VideoPredictions = FilePredictions("video")
