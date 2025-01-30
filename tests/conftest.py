"""Test fixtures for vlmrun tests."""

import pytest
from typer.testing import CliRunner

from datetime import datetime
from typing import List
from vlmrun.client.types import (
    ModelInfoResponse,
    DatasetCreateResponse,
    HubInfoResponse,
    HubDomainsResponse,
    HubSchemaQueryResponse,
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
    """Mock the Client class."""

    class MockClient:
        class AudioPredictions:
            def __init__(self, client):
                self._client = client

            def generate(self, *args, **kwargs):
                return PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00Z",
                    completed_at="2024-01-01T00:00:01Z",
                    response={"result": "test"},
                    usage={"total_tokens": 100},
                )

        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key or "test-key"
            self.base_url = base_url or "https://api.vlm.run"
            self.dataset = self.Dataset(self)
            self.fine_tuning = self.FineTuning(self)
            self.prediction = self.Prediction(self)
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
                    created_at="2024-01-01T00:00:00Z",
                    completed_at=None,
                    response=None,
                    usage={"total_tokens": 0},
                )

            def list(self):
                return [
                    PredictionResponse(
                        id="prediction1",
                        status="running",
                        created_at="2024-01-01T00:00:00Z",
                        completed_at=None,
                        response=None,
                        usage={"total_tokens": 0},
                    )
                ]

            def get(self, prediction_id):
                return PredictionResponse(
                    id=prediction_id,
                    status="running",
                    created_at="2024-01-01T00:00:00Z",
                    completed_at=None,
                    response=None,
                    usage={"total_tokens": 0},
                )

            def wait(self, prediction_id, timeout=60, sleep=1):
                return PredictionResponse(
                    id=prediction_id,
                    status="completed",
                    created_at="2024-01-01T00:00:00Z",
                    completed_at="2024-01-01T00:00:01Z",
                    response={"result": "test"},
                    usage={"total_tokens": 100},
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
                        created_at="2024-01-01T00:00:00Z",
                    )
                ]

            def upload(self, file_path, purpose="fine-tune"):
                return FileResponse(
                    id="file1",
                    filename=str(file_path),
                    bytes=10,
                    purpose=purpose,
                    created_at="2024-01-01T00:00:00Z",
                )

            def get(self, file_id):
                return FileResponse(
                    id=file_id,
                    filename="test.txt",
                    bytes=10,
                    purpose="assistants",
                    created_at="2024-01-01T00:00:00Z",
                )

            def get_content(self, file_id):
                return b"test content"

            def delete(self, file_id):
                return FileResponse(
                    id=file_id,
                    filename="test.txt",
                    bytes=10,
                    purpose="assistants",
                    created_at="2024-01-01T00:00:00Z",
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
                return HubDomainsResponse(
                    domains=[
                        "document.invoice",
                        "document.receipt",
                        "document.utility_bill",
                    ]
                )

            def get_schema(self, domain):
                return HubSchemaQueryResponse(
                    schema_json={
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

            def generate(self, *args, **kwargs):
                return PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00Z",
                    completed_at="2024-01-01T00:00:01Z",
                    response={"result": "test"},
                    usage={"total_tokens": 100},
                )

        class VideoPredictions:
            def __init__(self, client):
                self._client = client

            def generate(self, *args, **kwargs):
                return PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00Z",
                    completed_at="2024-01-01T00:00:01Z",
                    response={"result": "test"},
                    usage={"total_tokens": 100},
                )

        class DocumentPredictions:
            def __init__(self, client):
                self._client = client

            def generate(self, *args, **kwargs):
                return PredictionResponse(
                    id="prediction1",
                    status="completed",
                    created_at="2024-01-01T00:00:00Z",
                    completed_at="2024-01-01T00:00:01Z",
                    response={"result": "test"},
                    usage={"total_tokens": 100},
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
            ) -> DatasetCreateResponse:
                if dataset_type not in ["images", "videos", "documents"]:
                    raise ValueError(
                        "dataset_type must be one of: images, videos, documents"
                    )
                return DatasetCreateResponse(
                    dataset_id="dataset1",
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

            def get(self, dataset_id: str) -> DatasetCreateResponse:
                return DatasetCreateResponse(
                    dataset_id="dataset1",
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

            def list(self) -> List[DatasetCreateResponse]:
                return [
                    DatasetCreateResponse(
                        dataset_id="dataset1",
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

    monkeypatch.setattr("vlmrun.cli.cli.Client", MockClient)
    return MockClient()
