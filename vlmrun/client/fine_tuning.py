"""VLM Run API Fine-tuning resource."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List

from PIL import Image

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.client.types import (
    FinetuningResponse,
    PredictionResponse,
    FinetuningProvisionResponse,
)
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.common.image import encode_image


class Finetuning:
    """Fine-tuning resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize FineTuning resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(
            client, base_url=f"{client.base_url}/fine_tuning", timeout=300
        )

    def create(
        self,
        model: str,
        training_file_id: str,
        validation_file_id: str | None = None,
        num_epochs: int = 1,
        batch_size: int | str = "auto",
        learning_rate: float = 2e-4,
        suffix: str | None = None,
        wandb_api_key: str | None = None,
        wandb_base_url: str | None = None,
        wandb_project_name: str | None = None,
        **kwargs,
    ) -> FinetuningResponse:
        """Create a fine-tuning job.

        Args:
            model: Base model to fine-tune
            training_file_id: File ID for training data
            validation_file_id: File ID for validation data (default: None)
            validation_file_id: File ID for validation data (default: None)
            num_epochs: Number of epochs (default: 1)
            batch_size: Batch size for training (default: "auto")
            learning_rate: Learning rate for training (default: 2e-4)
            suffix: Suffix for the fine-tuned model (default: None)
            wandb_api_key: Weights & Biases API key (default: None)
            wandb_base_url: Weights & Biases base URL (default: None)
            wandb_project_name: Weights & Biases project name (default: None)
            **kwargs: Additional fine-tuning parameters

        Returns:
            FinetuningJobResponse: Created fine-tuning job
        """
        if suffix:
            # ensure suffix contains only alphanumeric, hyphens or underscores.
            # no special characters like spaces, periods, etc.
            if not re.match(r"^[a-zA-Z0-9_-]+$", suffix):
                raise ValueError(
                    "Suffix must be alphanumeric, hyphens or underscores without spaces"
                )

        response, status_code, headers = self._requestor.request(
            method="POST",
            url="create",
            data={
                "model": model,
                "training_file_id": training_file_id,
                "validation_file_id": validation_file_id,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "suffix": suffix,
                "wandb_api_key": wandb_api_key,
                "wandb_base_url": wandb_base_url,
                "wandb_project_name": wandb_project_name,
            },
        )
        return FinetuningResponse(**response)

    def provision(
        self, model: str, duration: int = 10 * 60, concurrency: int = 1
    ) -> FinetuningProvisionResponse:
        """Provision a fine-tuning model.

        Args:
            model: Model to provision
            duration: Duration for the provisioned model (in seconds)
            concurrency: Concurrency for the provisioned model

        Returns:
            FinetuningProvisionResponse: Response containing provisioning details
        """
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="provision",
            data={"model": model, "duration": duration, "concurrency": concurrency},
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return FinetuningProvisionResponse(**response)

    def generate(
        self,
        images: list[Path | Image.Image],
        model: str,
        prompt: str | None = None,
        domain: str | None = None,
        json_schema: dict | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        detail: str = "auto",
        batch: bool = False,
        metadata: dict = {},
        callback_url: str | None = None,
    ) -> PredictionResponse:
        """Generate a document prediction.

        Args:
            images: List of images to generate predictions from
            model: Model to use for prediction
            prompt: Prompt to use for prediction
            domain: Domain to use for prediction
            json_schema: JSON schema to use for prediction
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for prediction
            detail: Detail level for prediction
            batch: Whether to run prediction in batch mode
            metadata: Metadata to include in prediction
            callback_url: URL to call when prediction is complete

        Returns:
            PredictionResponse: Prediction response
        """
        # Check various parameters
        if not json_schema:
            raise ValueError("JSON schema is required for fine-tuned model predictions")
        if not prompt:
            raise ValueError("Prompt is required for fine-tuned model predictions")
        if domain:
            raise NotImplementedError(
                "Domain is not supported for fine-tuned model predictions"
            )
        if detail != "auto":
            raise NotImplementedError(
                "Detail level is not supported for fine-tuned model predictions"
            )
        if batch:
            raise NotImplementedError(
                "Batch mode is not supported for fine-tuned models"
            )
        if callback_url:
            raise NotImplementedError(
                "Callback URL is not supported for fine-tuned model predictions"
            )
        if len(images) > 1:
            raise ValueError(
                "Only one image is supported for fine-tuned model predictions for now"
            )

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
            url="generate",
            data={
                "image": encode_image(images[0], format="JPEG"),
                "model": model,
                "prompt": prompt,
                "json_schema": json_schema,
                "detail": detail,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "metadata": metadata,
                "batch": batch,
                "callback_url": callback_url,
            },
        )
        if not isinstance(response, dict):
            raise TypeError("Expected dict response")
        return PredictionResponse(**response)

    def list(self, skip: int = 0, limit: int = 10) -> list[FinetuningResponse]:
        """List all fine-tuning jobs.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            List[FinetuningJobResponse]: List of fine-tuning jobs
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="jobs",
            params={"skip": skip, "limit": limit},
        )
        return [FinetuningResponse(**job) for job in response]

    def list_models(self, skip: int = 0, limit: int = 10) -> List[str]:
        """List all fine-tuning models.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            List[str]: List of fine-tuning models
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="models",
            params={"skip": skip, "limit": limit},
        )
        if not isinstance(response, list):
            raise TypeError("Expected list response")
        return [str(model) for model in response]

    def get(self, job_id: str) -> FinetuningResponse:
        """Get fine-tuning job details.

        Args:
            job_id: ID of job to retrieve

        Returns:
            FinetuningJobResponse: Fine-tuning job details
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"jobs/{job_id}",
        )
        return FinetuningResponse(**response)

    def cancel(self, job_id: str) -> Dict:
        """Cancel a fine-tuning job.

        Args:
            job_id: ID of job to cancel

        Returns:
            Dict: Cancelled job details
        """
        raise NotImplementedError("Not implemented")
