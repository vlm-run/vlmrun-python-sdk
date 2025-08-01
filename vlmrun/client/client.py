"""VLM Run API client implementation."""

from dataclasses import dataclass
import os
from functools import cached_property
from typing import Optional, List, Type
from pydantic import BaseModel

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
from vlmrun.client.agent import Agent
from vlmrun.client.executions import Executions
from vlmrun.constants import DEFAULT_BASE_URL
from vlmrun.client.types import SchemaResponse, DomainInfo, GenerationConfig
from vlmrun.client.exceptions import (
    DependencyError,
    ConfigurationError,
    AuthenticationError,
)


@dataclass
class VLMRun:
    """VLM Run API client.

    Attributes:
        api_key: API key for authentication. Can be provided through constructor
            or VLMRUN_API_KEY environment variable.
        base_url: Base URL for API. Defaults to None, which falls back to
            VLMRUN_BASE_URL environment variable or https://api.vlm.run/v1.
        timeout: Request timeout in seconds. Defaults to 120.0.
        max_retries: Maximum number of retry attempts for failed requests. Defaults to 5.
        files: Files resource for managing files
        models: Models resource for accessing available models
        finetune: Fine-tuning resource for model fine-tuning
    """

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 5

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
                raise ConfigurationError(
                    message="Missing API key",
                    error_type="missing_api_key",
                    suggestion="Please provide your VLM Run API key:\n\n"
                    "1. Set it in your code:\n"
                    "   client = VLMRun(api_key='your-api-key')\n\n"
                    "2. Or set the environment variable:\n"
                    "   export VLMRUN_API_KEY='your-api-key'\n\n"
                    "Get your API key at https://app.vlm.run/dashboard",
                )

        # Handle base URL
        if self.base_url is None:
            self.base_url = os.getenv("VLMRUN_BASE_URL", DEFAULT_BASE_URL)

        # Initialize requestor for API key validation
        requestor = APIRequestor(
            self, timeout=self.timeout, max_retries=self.max_retries
        )

        # Validate API key by making a health check request
        try:
            _, status_code, _ = requestor.request(
                method="GET", url="/health", raw_response=True
            )
            if status_code != 200:
                raise AuthenticationError(
                    message="Invalid API key",
                    error_type="invalid_api_key",
                    suggestion="Please check your API key and ensure it is valid. You can get your API key at https://app.vlm.run/dashboard",
                )
        except AuthenticationError as e:
            raise AuthenticationError(
                message="Invalid API key",
                error_type="invalid_api_key",
                suggestion="Please check your API key and ensure it is valid. You can get your API key at https://app.vlm.run/dashboard",
            ) from e

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
        self.agent = Agent(self)
        self.executions = Executions(self)

    def __repr__(self):
        return f"VLMRun(base_url={self.base_url}, api_key={f'{self.api_key[:8]}...' if self.api_key else 'None'}, version={self.version})"

    @property
    def version(self):
        return __version__

    @cached_property
    def requestor(self):
        """Requestor for the API."""
        return APIRequestor(self, timeout=self.timeout, max_retries=self.max_retries)

    @cached_property
    def openai(self):
        """OpenAI client."""
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise DependencyError(
                message="OpenAI client is not installed",
                suggestion="Install it with `pip install openai`",
                error_type="missing_dependency",
            )

        return _OpenAI(api_key=self.api_key, base_url=f"{self.base_url}/openai")

    def healthcheck(self) -> bool:
        """Check the health of the API."""
        _, status_code, _ = self.requestor.request(
            method="GET", url="/health", raw_response=True
        )
        return status_code == 200

    def get_type(
        self, domain: str, config: Optional[GenerationConfig] = None
    ) -> Type[BaseModel]:
        """Get the type for a domain."""
        return self.get_schema(domain, config=config).response_model

    def get_schema(
        self, domain: str, config: Optional[GenerationConfig] = None
    ) -> SchemaResponse:
        """Get the schema for a domain.

        Args:
            domain: Domain name (e.g. "document.invoice")
            config: Generation config

        Returns:
            Schema response containing GraphQL schema and metadata
        """
        if config is None:
            config = GenerationConfig()
        response, status_code, headers = self.requestor.request(
            method="POST",
            url="/schema",
            data={"domain": domain, "config": config.model_dump()},
        )
        return SchemaResponse(**response)

    def list_domains(self) -> List[DomainInfo]:
        """List all available domains.

        Returns:
            List of domain names
        """
        response, status_code, headers = self.requestor.request(
            method="GET",
            url="/domains",
        )
        return [DomainInfo(**domain) for domain in response]
