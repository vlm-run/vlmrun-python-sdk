"""VLM Run API client implementation."""

from dataclasses import dataclass
from typing import Optional
import os

from vlmrun.client.files import Files
from vlmrun.client.models import Models
from vlmrun.client.finetune import FineTuning


@dataclass
class Client:
    """VLM Run API client.

    Attributes:
        api_key: API key for authentication
        files: Files resource for managing files
        models: Models resource for accessing available models
        finetune: Fine-tuning resource for model fine-tuning
    """

    api_key: Optional[str] = None
    base_url: str = "https://api.vlm.run/v1"
    timeout: float = 30.0

    def __post_init__(self):
        """Initialize the client after dataclass initialization."""
        if self.api_key is None:
            self.api_key = os.getenv("VLMRUN_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "API key must be provided either through constructor "
                    "or VLMRUN_API_KEY environment variable"
                )

        # Initialize resources
        self.files = Files(self)
        self.models = Models(self)
        self.finetune = FineTuning(self)

    # Deprecated methods - use resource classes instead
    def list_files(self):
        """List all files (deprecated).

        Deprecated:
            Use client.files.list() instead.
        """
        return self.files.list()

    def upload_file(self, file_path: str, purpose: str = "fine-tune"):
        """Upload a file (deprecated).

        Deprecated:
            Use client.files.upload() instead.
        """
        return self.files.upload(file_path, purpose=purpose)

    def delete_file(self, file_id: str):
        """Delete a file (deprecated).

        Deprecated:
            Use client.files.delete() instead.
        """
        return self.files.delete(file_id)

    def get_file(self, file_id: str):
        """Get file content (deprecated).

        Deprecated:
            Use client.files.retrieve_content() instead.
        """
        return self.files.retrieve_content(file_id)

    def create_fine_tuning_job(self, training_file: str, model: str, **kwargs):
        """Create a fine-tuning job (deprecated).

        Deprecated:
            Use client.finetune.create() instead.
        """
        return self.finetune.create(training_file=training_file, model=model, **kwargs)

    def list_fine_tuning_jobs(self):
        """List all fine-tuning jobs (deprecated).

        Deprecated:
            Use client.finetune.list() instead.
        """
        return self.finetune.list()

    def get_fine_tuning_job(self, job_id: str):
        """Get fine-tuning job details (deprecated).

        Deprecated:
            Use client.finetune.retrieve() instead.
        """
        return self.finetune.retrieve(job_id)

    def cancel_fine_tuning_job(self, job_id: str):
        """Cancel a fine-tuning job (deprecated).

        Deprecated:
            Use client.finetune.cancel() instead.
        """
        return self.finetune.cancel(job_id)

    def get_fine_tuning_job_status(self, job_id: str):
        """Get fine-tuning job status (deprecated).

        Deprecated:
            Use client.finetune.retrieve() instead.
        """
        return self.finetune.retrieve(job_id)

    def list_models(self):
        """List available models (deprecated).

        Deprecated:
            Use client.models.list() instead.
        """
        return self.models.list()

    # TODO: Implement these in future updates
    def generate_image(self, prompt: str):
        """Generate an image."""
        raise NotImplementedError("Image generation not yet implemented")

    def generate_video(self, prompt: str):
        """Generate a video."""
        raise NotImplementedError("Video generation not yet implemented")

    def generate_document(self, prompt: str):
        """Generate a document."""
        raise NotImplementedError("Document generation not yet implemented")

    def get_hub_version(self):
        """Get hub version."""
        raise NotImplementedError("Hub version not yet implemented")

    def list_hub_items(self):
        """List hub items."""
        raise NotImplementedError("Hub items not yet implemented")

    def submit_hub_item(self, path: str, name: str, version: str):
        """Submit an item to the hub."""
        raise NotImplementedError("Hub submission not yet implemented")
