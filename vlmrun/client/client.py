"""VLM Run API client implementation."""

from dataclasses import dataclass
import os
from functools import cached_property
from typing import Dict, List, Any

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.client.dataset import Dataset
from vlmrun.client.files import Files
from vlmrun.client.models import Models
from vlmrun.client.finetune import FineTuning
from vlmrun.client.types import ModelFinetuningRequest, ModelFinetuningResponse, FinetunedImagePredictionRequest
from vlmrun.types.abstract import Client as AbstractClient


@dataclass
class Client(AbstractClient):
    """VLM Run API client.

    Attributes:
        api_key: API key for authentication. Can be provided through constructor
            or VLMRUN_API_KEY environment variable.
        base_url: Base URL for API. Defaults to None, which falls back to
            VLMRUN_BASE_URL environment variable or https://api.vlm.run/v1.
        timeout: Request timeout in seconds. Defaults to 30.0.
        files: Files resource for managing files
        models: Models resource for accessing available models
        finetune: Fine-tuning resource for model fine-tuning
    """

    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0

    def __post_init__(self):
        """Initialize the client after dataclass initialization.

        This method handles environment variable fallbacks:
        - api_key: Falls back to VLMRUN_API_KEY environment variable
        - base_url: Can be overridden by constructor or VLMRUN_BASE_URL environment variable
        """
        # Handle API key first
        if not self.api_key:  # Handle both None and empty string
            self.api_key = os.getenv("VLMRUN_API_KEY", None)
            if not self.api_key:  # Still None or empty after env check
                raise ValueError(
                    "API key must be provided either through constructor "
                    "or VLMRUN_API_KEY environment variable"
                )

        # Handle base URL
        if self.base_url is None:
            self.base_url = os.getenv("VLMRUN_BASE_URL", "https://api.vlm.run/v1")

        # Initialize resources
        self.dataset = Dataset(self)
        self.files = Files(self)
        self.models = Models(self)
        self.finetune = FineTuning(self)

    @cached_property
    def requestor(self):
        """Requestor for the API."""
        return APIRequestor(self)

    @cached_property
    def openai(self):
        """OpenAI client."""
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI client is not installed. Please install it with "
                "`pip install openai`"
            )

        return _OpenAI(api_key=self.api_key, base_url=f"{self.base_url}/openai")

    def healthcheck(self) -> bool:
        """Check the health of the API."""
        _, status_code, _ = self.requestor.request(
            method="GET", url="/health", raw_response=True
        )
        return status_code == 200

    # Deprecated methods - use resource classes instead
    def list_files(self):
        """Use client.files.list() instead. This method is deprecated."""
        return self.files.list()

    def upload_file(self, file_path: str, purpose: str = "fine-tune"):
        """Use client.files.upload() instead. This method is deprecated."""
        return self.files.upload(file_path, purpose=purpose)

    def delete_file(self, file_id: str):
        """Use client.files.delete() instead. This method is deprecated."""
        return self.files.delete(file_id)

    def get_file(self, file_id: str):
        """Use client.files.retrieve_content() instead. This method is deprecated."""
        return self.files.retrieve_content(file_id)

    def create_fine_tuning_job(self, training_file: str, model: str, **kwargs):
        """Use client.finetune.create_fit() instead. This method is deprecated."""
        request = ModelFinetuningRequest(model=model, dataset_uri=training_file, **kwargs)
        return self.finetune.create_fit(request)

    def list_fine_tuning_jobs(self):
        """Use client.finetune.list_models() instead. This method is deprecated."""
        return self.finetune.list_models()

    def get_fine_tuning_job(self, job_id: str):
        """Use client.finetune.retrieve_fit() instead. This method is deprecated."""
        return self.finetune.retrieve_fit(job_id)

    def cancel_fine_tuning_job(self, job_id: str):
        """Use client.finetune.retrieve_fit() instead. This method is deprecated."""
        # Note: Cancel operation is not supported in the new API
        return self.finetune.retrieve_fit(job_id)

    def get_fine_tuning_job_status(self, job_id: str):
        """Use client.finetune.retrieve_fit() instead. This method is deprecated."""
        return self.finetune.retrieve_fit(job_id)

    def list_models(self):
        """Use client.models.list() instead. This method is deprecated."""
        return self.models.list()

    def healthcheck_finetuning(self) -> bool:
        """Check the health of the fine-tuning service.

        Returns:
            bool: True if service is healthy
        """
        return self.finetune.health()

    def list_finetuned_models(self) -> List[ModelFinetuningResponse]:
        """List all fine-tuned models.

        Returns:
            List[ModelFinetuningResponse]: List of fine-tuned models
        """
        return self.finetune.list_models()

    def create_finetuning(self, request: ModelFinetuningRequest) -> ModelFinetuningResponse:
        """Create a fine-tuning job.

        Args:
            request: Fine-tuning request parameters

        Returns:
            ModelFinetuningResponse: Created fine-tuning job details
        """
        return self.finetune.create_fit(request)

    def retrieve_finetuning(self, training_id: str) -> ModelFinetuningResponse:
        """Get fine-tuning job status.

        Args:
            training_id: ID of job to retrieve

        Returns:
            ModelFinetuningResponse: Fine-tuning job status and details
        """
        return self.finetune.retrieve_fit(training_id)

    def generate_prediction(self, request: FinetunedImagePredictionRequest) -> Dict[str, Any]:
        """Generate predictions using a fine-tuned model.

        Args:
            request: Prediction request parameters

        Returns:
            Dict[str, Any]: Model predictions
        """
        return self.finetune.generate(request)

    # Legacy methods marked for future removal
    def generate_image(self, prompt: str):
        """Use generate_prediction() instead. This method is deprecated."""
        raise NotImplementedError("Use generate_prediction() instead")

    def generate_video(self, prompt: str):
        """Use generate_prediction() instead. This method is deprecated."""
        raise NotImplementedError("Use generate_prediction() instead")

    def generate_document(self, prompt: str):
        """Use generate_prediction() instead. This method is deprecated."""
        raise NotImplementedError("Use generate_prediction() instead")

    def get_hub_version(self):
        raise NotImplementedError("Hub version not yet implemented")

    def list_hub_items(self):
        raise NotImplementedError("Hub items not yet implemented")

    def submit_hub_item(self, path: str, name: str, version: str):
        raise NotImplementedError("Hub submission not yet implemented")
