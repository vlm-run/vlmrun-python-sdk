"""VLM Run API client implementation."""

from dataclasses import dataclass
import os

from vlmrun.client.files import Files
from vlmrun.client.models import Models
from vlmrun.client.finetune import FineTuning


@dataclass
class Client:
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
        - base_url: Falls back to VLMRUN_BASE_URL environment variable,
          then to https://api.vlm.run/v1
        """
        # Handle API key
        if self.api_key is None:
            self.api_key = os.getenv("VLMRUN_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "API key must be provided either through constructor "
                    "or VLMRUN_API_KEY environment variable"
                )

        # Handle base URL
        if self.base_url is None:
            self.base_url = os.getenv("VLMRUN_BASE_URL", "https://api.vlm.run/v1")

        # Initialize resources
        self.files = Files(self)
        self.models = Models(self)
        self.finetune = FineTuning(self)

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
        """Use client.finetune.create() instead. This method is deprecated."""
        return self.finetune.create(training_file=training_file, model=model, **kwargs)

    def list_fine_tuning_jobs(self):
        """Use client.finetune.list() instead. This method is deprecated."""
        return self.finetune.list()

    def get_fine_tuning_job(self, job_id: str):
        """Use client.finetune.retrieve() instead. This method is deprecated."""
        return self.finetune.retrieve(job_id)

    def cancel_fine_tuning_job(self, job_id: str):
        """Use client.finetune.cancel() instead. This method is deprecated."""
        return self.finetune.cancel(job_id)

    def get_fine_tuning_job_status(self, job_id: str):
        """Use client.finetune.retrieve() instead. This method is deprecated."""
        return self.finetune.retrieve(job_id)

    def list_models(self):
        """Use client.models.list() instead. This method is deprecated."""
        return self.models.list()

    def generate_image(self, prompt: str):
        raise NotImplementedError("Image generation not yet implemented")

    def generate_video(self, prompt: str):
        raise NotImplementedError("Video generation not yet implemented")

    def generate_document(self, prompt: str):
        raise NotImplementedError("Document generation not yet implemented")

    def get_hub_version(self):
        raise NotImplementedError("Hub version not yet implemented")

    def list_hub_items(self):
        raise NotImplementedError("Hub items not yet implemented")

    def submit_hub_item(self, path: str, name: str, version: str):
        raise NotImplementedError("Hub submission not yet implemented")
