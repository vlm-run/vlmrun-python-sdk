"""Test fixtures for vlmrun tests."""

import pytest
from typer.testing import CliRunner

from datetime import datetime
from typing import List
from vlmrun.client.types import (
    ModelInfoResponse,
    DatasetResponse,
    HubInfoResponse,
    HubDomainInfo,
    HubSchemaResponse,
    FileResponse,
    PredictionResponse,
    FeedbackSubmitResponse,
    CreditUsage,
)


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_client(monkeypatch):
    """Mock the VLMRun class."""

    class MockVLMRun:
        class AudioPredictions:
            def __init__(self, client):
                self._client = client

            def generate(self, *args, **kwargs):
                return PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"result": "test"},
                    usage=CreditUsage(credits_used=100),
                )

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
                return [ModelInfoResponse(model="model1", domain="test-domain")]

        class Hub:
            def __init__(self, client):
                self._client = client
                self.version = "0.1.0"

            def info(self):
                return HubInfoResponse(version="0.1.0")

            def list_domains(self):
                return [
                    HubDomainInfo(domain="document.invoice"),
                    HubDomainInfo(domain="document.receipt"),
                    HubDomainInfo(domain="document.utility_bill"),
                ]

            def get_schema(self, domain):
                return HubSchemaResponse(
                    json_schema={
                        "type": "object",
                        "properties": {
                            "invoice_number": {"type": "string"},
                            "total_amount": {"type": "number"},
                        },
                    },
                    schema_version="1.0.0",
                    schema_hash="abcd1234",
                )

        class ImagePredictions:
            def __init__(self, client):
                self._client = client

            def generate(self, domain: str, images=None, urls=None, **kwargs):
                if not images and not urls:
                    raise ValueError("Either `images` or `urls` must be provided")
                if images and urls:
                    raise ValueError("Only one of `images` or `urls` can be provided")
                return PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"result": "test"},
                    usage=CreditUsage(credits_used=100),
                )

        class VideoPredictions:
            def __init__(self, client):
                self._client = client

            def generate(self, *args, **kwargs):
                return PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"result": "test"},
                    usage=CreditUsage(credits_used=100),
                )

        class DocumentPredictions:
            def __init__(self, client):
                self._client = client

            def generate(self, *args, **kwargs):
                return PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00+00:00",
                    completed_at="2024-01-01T00:00:01+00:00",
                    response={"result": "test"},
                    usage=CreditUsage(credits_used=100),
                )

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

            def submit(self, id, label=None, notes=None, flag=None):
                return FeedbackSubmitResponse(
                    id="feedback1",
                    created_at="2024-01-01T00:00:00Z",
                    request_id=id,
                    response=label,
                )

            def get(self, id):
                return FeedbackSubmitResponse(
                    id="feedback1",
                    created_at="2024-01-01T00:00:00Z",
                    request_id=id,
                    response=None,
                )

    monkeypatch.setattr("vlmrun.cli.cli.VLMRun", MockVLMRun)
    return MockVLMRun()
