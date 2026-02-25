"""Test fixtures for vlmrun tests."""

import hashlib
import pytest
from functools import cached_property
from typer.testing import CliRunner
from pydantic import BaseModel

from datetime import datetime
from typing import List
from vlmrun.client.types import (
    ModelInfo,
    DatasetResponse,
    HubInfoResponse,
    HubSchemaResponse,
    HubDomainInfo,
    FileResponse,
    PredictionResponse,
    CreditUsage,
    PresignedUrlResponse,
)


class MockInvoiceSchema(BaseModel):
    invoice_number: str
    total_amount: float


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_client(monkeypatch):
    """Mock the VLMRun class."""

    class SchemaCastMixin:
        def _cast_response_to_schema(self, prediction, domain, config=None):
            """Updated implementation of _cast_response_to_schema."""
            try:
                if prediction and prediction.response:
                    # Get model class for the domain
                    if isinstance(prediction.response, dict):
                        prediction.response = MockInvoiceSchema(**prediction.response)

            except Exception as e:
                import logging

                logging.warning(f"Failed to cast response to schema: {e}")

    class MockVLMRun:
        class AudioPredictions(SchemaCastMixin):
            def __init__(self, client):
                self._client = client

            def generate(self, domain: str = None, **kwargs):
                prediction = PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"invoice_number": "INV-001", "total_amount": 100.0},
                    usage=CreditUsage(credits_used=100),
                )

                if kwargs.get("autocast", False):
                    self._cast_response_to_schema(
                        prediction, domain, kwargs.get("config")
                    )
                return prediction

            def execute(self, name: str, **kwargs):
                prediction = PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"invoice_number": "INV-001", "total_amount": 100.0},
                    usage=CreditUsage(credits_used=100),
                )

                if kwargs.get("autocast", False):
                    self._cast_response_to_schema(
                        prediction, name, kwargs.get("config")
                    )
                return prediction

        def __init__(self, api_key=None, base_url=None, agent_base_url=None):
            self.api_key = api_key or "test-key"
            self.base_url = base_url or "https://api.vlm.run"
            self.agent_base_url = agent_base_url or self.base_url
            self.timeout = 120.0
            self.max_retries = 1
            self.dataset = self.Dataset(self)
            self.fine_tuning = self.FineTuning(self)
            self.predictions = self.Prediction(self)
            self.files = self.Files(self)
            self.models = self.Models(self)
            self.hub = self.Hub(self)
            self.image = self.ImagePredictions(self)
            self.video = self.VideoPredictions(self)
            self.document = self.DocumentPredictions(self)
            self.audio = self.AudioPredictions(self)
            self.feedback = self.Feedback(self)
            self.agent = self.Agent(self)
            self.artifacts = self.Artifacts(self)

        class FineTuning:
            def __init__(self, client):
                self._client = client

            def create(self, training_file, validation_file, model, **kwargs):
                return {"id": "job1"}

            def list(self):
                return [
                    {
                        "id": "job1",
                        "model": "test-model",
                        "status": "running",
                        "created_at": "2024-01-01",
                    }
                ]

            def get(self, job_id):
                return {"id": job_id, "status": "running"}

            def cancel(self, job_id):
                return True

        class Prediction:
            def __init__(self, client):
                self._client = client

            def create(self, model, prompt, **kwargs):
                return PredictionResponse(
                    id="prediction1",
                    status="running",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at=None,
                    response=None,
                    usage=CreditUsage(credits_used=0),
                )

            def list(self, skip: int = 0, limit: int = 10):
                return [
                    PredictionResponse(
                        id="prediction1",
                        status="running",
                        created_at="2024-01-01T00:00:00+00:00",
                        completed_at=None,
                        response=None,
                        usage=CreditUsage(credits_used=0),
                    )
                ]

            def get(self, prediction_id):
                return PredictionResponse(
                    id=prediction_id,
                    status="running",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at=None,
                    response=None,
                    usage=CreditUsage(credits_used=0),
                )

            def wait(self, prediction_id, timeout=60, sleep=1):
                return PredictionResponse(
                    id=prediction_id,
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"result": "test"},
                    usage=CreditUsage(credits_used=100),
                )

        class Files:
            def __init__(self, client):
                self._client = client

            def list(self):
                return [
                    FileResponse(
                        id="file1",
                        filename="test.txt",
                        bytes=10,
                        purpose="assistants",
                        created_at="2024-01-01T00:00:00+00:00",
                    )
                ]

            def upload(self, file_path, purpose="fine-tune"):
                return FileResponse(
                    id="file1",
                    filename=str(file_path),
                    bytes=10,
                    purpose=purpose,
                    created_at="2024-01-01T00:00:00+00:00",
                )

            def get(self, file_id):
                return FileResponse(
                    id=file_id,
                    filename="test.txt",
                    bytes=10,
                    purpose="assistants",
                    created_at="2024-01-01T00:00:00+00:00",
                )

            def get_content(self, file_id):
                return b"test content"

            def delete(self, file_id):
                return FileResponse(
                    id=file_id,
                    filename="test.txt",
                    bytes=10,
                    purpose="assistants",
                    created_at="2024-01-01T00:00:00+00:00",
                )

            def generate_presigned_url(self, params):
                return PresignedUrlResponse(
                    id="presigned1",
                    url="https://storage.example.com/upload/presigned1",
                    filename=params.filename,
                    content_type="application/octet-stream",
                    upload_method="PUT",
                    public_url="https://storage.example.com/files/presigned1",
                    created_at="2024-01-01T00:00:00+00:00",
                )

        class Models:
            def __init__(self, client):
                self._client = client

            def list(self):
                return [ModelInfo(model="model1", domain="test-domain")]

        class Hub:
            def __init__(self, client):
                self.client = client
                self.version = "0.1.0"

            def info(self):
                return HubInfoResponse(version="0.1.0")

            def list_domains(self):
                return [
                    HubDomainInfo(domain="document.invoice"),
                    HubDomainInfo(domain="document.receipt"),
                    HubDomainInfo(domain="document.utility_bill"),
                ]

            def get_schema(self, domain, gql_stmt=None):
                """Mock implementation of Hub's get_schema."""
                json_schema = MockInvoiceSchema.model_json_schema()

                return HubSchemaResponse(
                    domain=domain,
                    schema_version="1.0.0",
                    schema_hash="1234567890",
                    gql_stmt=gql_stmt or "",
                    json_schema=json_schema,
                )

            def get_pydantic_model(self, domain: str):
                """Mock implementation for schema lookup."""
                schemas = {"document.invoice": MockInvoiceSchema, "general": None}
                return schemas.get(domain)

        class ImagePredictions(SchemaCastMixin):
            def __init__(self, client):
                self._client = client

            def generate(self, domain: str, images=None, urls=None, **kwargs):
                if not images and not urls:
                    raise ValueError("Either `images` or `urls` must be provided")
                if images and urls:
                    raise ValueError("Only one of `images` or `urls` can be provided")

                prediction = PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"invoice_number": "INV-001", "total_amount": 100.0},
                    usage=CreditUsage(credits_used=100),
                )

                if kwargs.get("autocast", False):
                    self._cast_response_to_schema(
                        prediction, domain, kwargs.get("config")
                    )
                return prediction

            def execute(self, name: str, images=None, urls=None, **kwargs):
                if not images and not urls:
                    raise ValueError("Either `images` or `urls` must be provided")
                if images and urls:
                    raise ValueError("Only one of `images` or `urls` can be provided")

                prediction = PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"invoice_number": "INV-001", "total_amount": 100.0},
                    usage=CreditUsage(credits_used=100),
                )

                if kwargs.get("autocast", False):
                    self._cast_response_to_schema(
                        prediction, name, kwargs.get("config")
                    )
                return prediction

        class VideoPredictions(SchemaCastMixin):
            def __init__(self, client):
                self._client = client

            def generate(self, domain: str = None, **kwargs):
                prediction = PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"invoice_number": "INV-001", "total_amount": 100.0},
                    usage=CreditUsage(credits_used=100),
                )

                if kwargs.get("autocast", False):
                    self._cast_response_to_schema(
                        prediction, domain, kwargs.get("config")
                    )
                return prediction

            def execute(self, name: str, **kwargs):
                prediction = PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"invoice_number": "INV-001", "total_amount": 100.0},
                    usage=CreditUsage(credits_used=100),
                )

                if kwargs.get("autocast", False):
                    self._cast_response_to_schema(
                        prediction, name, kwargs.get("config")
                    )
                return prediction

        class DocumentPredictions(SchemaCastMixin):
            def __init__(self, client):
                self._client = client

            def generate(self, domain: str = None, **kwargs):
                prediction = PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"invoice_number": "INV-001", "total_amount": 100.0},
                    usage=CreditUsage(credits_used=100),
                )

                if kwargs.get("autocast", False):
                    self._cast_response_to_schema(
                        prediction, domain, kwargs.get("config")
                    )
                return prediction

            def execute(self, name: str, **kwargs):
                prediction = PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"invoice_number": "INV-001", "total_amount": 100.0},
                    usage=CreditUsage(credits_used=100),
                )

                if kwargs.get("autocast", False):
                    self._cast_response_to_schema(
                        prediction, name, kwargs.get("config")
                    )
                return prediction

        class Dataset:
            def __init__(self, client):
                self._client = client

            def create(
                self,
                file_id: str,
                domain: str,
                dataset_name: str,
                dataset_type: str = "images",
            ) -> DatasetResponse:
                if dataset_type not in ["images", "videos", "documents"]:
                    raise ValueError(
                        "dataset_type must be one of: images, videos, documents"
                    )
                return DatasetResponse(
                    id="dataset1",
                    dataset_uri="gs://vlmrun-test-bucket/dataset1.tar.gz",
                    dataset_type=dataset_type,
                    dataset_name=dataset_name,
                    domain=domain,
                    message="Dataset created successfully",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                    status="pending",
                    usage=CreditUsage(
                        credits_used=10,
                        elements_processed=10,
                        element_type="image",
                    ),
                )

            def get(self, dataset_id: str) -> DatasetResponse:
                return DatasetResponse(
                    id="dataset1",
                    dataset_uri="gs://vlmrun-test-bucket/dataset1.tar.gz",
                    dataset_type="images",
                    dataset_name="test-dataset",
                    domain="test-domain",
                    message="Dataset created successfully",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                    status="completed",
                    usage=CreditUsage(
                        credits_used=10,
                        elements_processed=10,
                        element_type="image",
                    ),
                )

            def list(self) -> List[DatasetResponse]:
                return [
                    DatasetResponse(
                        id="dataset1",
                        dataset_uri="gs://vlmrun-test-bucket/dataset1.tar.gz",
                        dataset_type="images",
                        domain="test-domain",
                        dataset_name="test-dataset",
                        message="Dataset created successfully",
                        created_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                        status="completed",
                        usage=CreditUsage(
                            credits_used=10,
                            elements_processed=10,
                            element_type="image",
                        ),
                    )
                ]

        class Feedback:
            def __init__(self, client):
                self._client = client

            def submit(self, request_id, response, notes=None):
                from vlmrun.client.types import FeedbackSubmitResponse

                return FeedbackSubmitResponse(
                    request_id=request_id,
                    id="feedback1",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                    response=response,
                    notes=notes,
                )

            def get(self, request_id, limit=10, offset=0):
                from vlmrun.client.types import FeedbackListResponse, FeedbackItem

                return FeedbackListResponse(
                    request_id=request_id,
                    items=[
                        FeedbackItem(
                            id="feedback1",
                            created_at=datetime.fromisoformat(
                                "2024-01-01T00:00:00+00:00"
                            ),
                            response={"score": 5},
                            notes="Test feedback",
                        )
                    ],
                )

        class Agent:
            def __init__(self, client):
                self._client = client

            @cached_property
            def completions(self):
                from openai import OpenAI

                base_url = f"{self._client.base_url}/openai"
                openai_client = OpenAI(
                    api_key=self._client.api_key,
                    base_url=base_url,
                    timeout=self._client.timeout,
                    max_retries=self._client.max_retries,
                )
                return openai_client.chat.completions

            @cached_property
            def async_completions(self):
                from openai import AsyncOpenAI

                base_url = f"{self._client.base_url}/openai"
                async_openai_client = AsyncOpenAI(
                    api_key=self._client.api_key,
                    base_url=base_url,
                    timeout=self._client.timeout,
                    max_retries=self._client.max_retries,
                )
                return async_openai_client.chat.completions

            def get(self, name=None, id=None, prompt=None):
                from vlmrun.client.types import AgentInfo
                from datetime import datetime

                if id:
                    if name or prompt:
                        raise ValueError(
                            "Only one of `id` or `name` or `prompt` can be provided."
                        )
                elif name:
                    if id or prompt:
                        raise ValueError(
                            "Only one of `id` or `name` or `prompt` can be provided."
                        )
                elif prompt:
                    if id or name:
                        raise ValueError(
                            "Only one of `id` or `name` or `prompt` can be provided."
                        )
                else:
                    raise ValueError(
                        "Either `id` or `name` or `prompt` must be provided."
                    )

                if id:
                    agent_id = id
                    agent_name = f"agent-{id}"
                elif name:
                    agent_id = f"agent-{name}"
                    agent_name = name
                elif prompt:
                    hash_prompt = hashlib.sha256(prompt.encode()).hexdigest()
                    agent_id = f"agent-{hash_prompt}"
                    agent_name = f"agent-{hash_prompt}"

                return AgentInfo(
                    id=agent_id,
                    name=agent_name,
                    description="Test agent description",
                    prompt="Test agent prompt",
                    json_schema={
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                    },
                    json_sample={"result": "test result"},
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                    updated_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                    status="completed",
                )

            def list(self):
                from vlmrun.client.types import AgentInfo
                from datetime import datetime

                return [
                    AgentInfo(
                        id="agent-1",
                        name="test-agent-1:1.0.0",
                        description="First test agent",
                        prompt="Test prompt 1",
                        json_schema={
                            "type": "object",
                            "properties": {"result": {"type": "string"}},
                        },
                        json_sample={"result": "test result 1"},
                        created_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                        updated_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                        status="completed",
                    ),
                    AgentInfo(
                        id="agent-2",
                        name="test-agent-2:2.0.0",
                        description="Second test agent",
                        prompt="Test prompt 2",
                        json_schema={
                            "type": "object",
                            "properties": {"output": {"type": "string"}},
                        },
                        json_sample={"output": "test output 2"},
                        created_at=datetime.fromisoformat("2024-01-02T00:00:00+00:00"),
                        updated_at=datetime.fromisoformat("2024-01-02T00:00:00+00:00"),
                        status="completed",
                    ),
                ]

            def create(self, config, name=None, inputs=None, callback_url=None):
                from vlmrun.client.types import AgentCreationResponse
                from datetime import datetime

                if config.prompt is None:
                    raise ValueError(
                        "Prompt is not provided as a request parameter, please provide a prompt."
                    )

                return AgentCreationResponse(
                    id=f"agent-{name or 'created'}",
                    name=name or "created-agent:1.0.0",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                    updated_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                    status="pending",
                )

            def execute(
                self,
                name,
                inputs=None,
                batch=True,
                config=None,
                metadata=None,
                callback_url=None,
            ):
                from vlmrun.client.types import AgentExecutionResponse, CreditUsage
                from datetime import datetime

                if not batch:
                    raise NotImplementedError(
                        "Batch mode is required for agent execution"
                    )

                return AgentExecutionResponse(
                    id=f"execution-{name}",
                    name=name,
                    status="completed",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                    completed_at=datetime.fromisoformat("2024-01-01T00:00:01+00:00"),
                    response={"result": "execution result"},
                    usage=CreditUsage(credits_used=50),
                )

        class Artifacts:
            def __init__(self, client):
                self._client = client

            def get(
                self,
                object_id: str,
                session_id: str = None,
                execution_id: str = None,
                raw_response: bool = False,
            ) -> bytes:
                if session_id is None and execution_id is None:
                    raise ValueError(
                        "Either `session_id` or `execution_id` is required"
                    )
                if session_id is not None and execution_id is not None:
                    raise ValueError(
                        "Only one of `session_id` or `execution_id` is allowed, not both"
                    )
                return b"mock artifact content"

            def list(self, session_id: str):
                raise NotImplementedError("Artifacts.list() is not yet implemented")

    monkeypatch.setattr("vlmrun.cli.cli.VLMRun", MockVLMRun)
    return MockVLMRun()


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """Create a temporary config file."""
    config_dir = tmp_path / ".vlmrun"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"

    monkeypatch.setenv("VLMRUN_API_KEY", "test-key")
    monkeypatch.setenv("VLMRUN_BASE_URL", "https://test.vlm.run")

    monkeypatch.setattr("vlmrun.cli._cli.config.CONFIG_FILE", config_path)
    return config_path
