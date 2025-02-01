"""VLM Run API client implementation."""

from dataclasses import dataclass
import os
from functools import cached_property

from vlmrun.version import __version__
from vlmrun.client.base_requestor import APIRequestor
from vlmrun.client.datasets import Datasets
from vlmrun.client.files import Files
from vlmrun.client.hub import Hub
from vlmrun.client.models import Models
from vlmrun.client.fine_tuning import Finetuning
from vlmrun.client.predictions import (
    Predictions,
    ImagePredictions,
    VideoPredictions,
    DocumentPredictions,
    AudioPredictions,
)
from vlmrun.client.feedback import Feedback


@dataclass
class VLMRun:
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
    timeout: float = 120.0

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
        self.datasets = Datasets(self)
        self.files = Files(self)
        self.hub = Hub(self)
        self.models = Models(self)
        self.fine_tuning = Finetuning(self)
        self.predictions = Predictions(self)
        self.image = ImagePredictions(self)
        self.document = DocumentPredictions(self)
        self.document._requestor._timeout = 120.0
        self.audio = AudioPredictions(self)
        self.audio._requestor._timeout = 120.0
        self.video = VideoPredictions(self)
        self.video._requestor._timeout = 120.0
        self.feedback = Feedback(self)

    def __repr__(self):
        return f"VLMRun(base_url={self.base_url}, api_key={f'{self.api_key[:8]}...' if self.api_key else 'None'}, version={self.version})"

    @property
    def version(self):
        return __version__

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
