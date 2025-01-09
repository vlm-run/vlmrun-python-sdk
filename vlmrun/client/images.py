"""Image-related functionality for VLMRun."""

from typing import Any, Dict, Literal, Optional, Union
from pathlib import Path

from PIL.Image import Image
import requests

from ..common.image import encode_image


class Images:
    """Class for handling image-related operations."""

    def __init__(self, client: Any) -> None:
        """Initialize Images instance.

        Args:
            client: The VLMRun client instance
        """
        self._client = client

    def generate(
        self,
        image: Union[str, Path, Image],
        *,
        domain: Optional[Literal[
            "document.generative",
            "document.presentation",
            "document.invoice",
            "document.receipt",
            "document.markdown",
            "video.tv-news",
            "video.tv-intelligence"
        ]] = None,
        model: str = "vlm-1",
        detail: Optional[Literal["auto", "hi", "lo"]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        callback_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate structured prediction for the given image.

        Args:
            image: Image to analyze (PIL Image, path to image, or Path object)
            domain: Domain identifier for the prediction
            model: Model to use for generating the response (default: vlm-1)
            detail: Detail level to use for the model
            json_schema: JSON schema to use for the model
            callback_url: URL to call when the request is completed
            metadata: Optional metadata to pass to the model

        Returns:
            Dict containing the prediction response

        Raises:
            FileNotFoundError: If image path doesn't exist
            ValueError: If image type is invalid
            requests.RequestException: If the API request fails
        """
        # Convert image to base64
        image_data = encode_image(image)

        # Build request payload
        payload = {
            "image": image_data,
            "model": model,
        }

        if domain is not None:
            payload["domain"] = domain
        if detail is not None:
            payload["detail"] = detail
        if json_schema is not None:
            payload["json_schema"] = json_schema
        if callback_url is not None:
            payload["callback_url"] = callback_url
        if metadata is not None:
            payload["metadata"] = metadata

        # Make API request
        headers = {
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            "https://api.vlm.run/v1/image/generate",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        return response.json()
