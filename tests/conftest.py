"""Test fixtures for vlmrun tests."""

import pytest
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

        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key or "test-key"
            self.base_url = base_url or "https://api.vlm.run"
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
