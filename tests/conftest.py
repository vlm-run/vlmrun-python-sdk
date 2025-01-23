"""Test fixtures for vlmrun tests."""

import pytest
from typer.testing import CliRunner

from datetime import datetime
from vlmrun.client.types import (
    ModelInfoResponse,
    DatasetResponse,
    HubInfoResponse,
    HubDomainsResponse,
    HubSchemaQueryResponse,
)


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_client(monkeypatch):
    """Mock the Client class."""

    class MockClient:
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key or "test-key"
            self.base_url = base_url or "https://api.vlm.run"
            self.dataset = self.Dataset(self)
            self.fine_tuning = self.FineTuning(self)
            self.prediction = self.Prediction(self)
            self.files = self.Files(self)
            self.models = self.Models(self)
            self.hub = self.Hub(self)
            self.image = self.Image(self)
            self.video = self.Video(self)
            self.document = self.Document(self)

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
                    dataset_id="dataset1",
                    dataset_uri="gs://vlmrun-test-bucket/dataset1.tar.gz",
                    dataset_type=dataset_type,
                    domain=domain,
                    message="Dataset created successfully",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                )

            def get(self, dataset_id: str) -> DatasetResponse:
                return DatasetResponse(
                    dataset_id="dataset1",
                    dataset_uri="gs://vlmrun-test-bucket/dataset1.tar.gz",
                    dataset_type="images",
                    domain="test-domain",
                    message="Dataset created successfully",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
                )

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
                return {"id": "prediction1"}

            def list(self):
                return [{"id": "prediction1", "status": "running"}]

            def get(self, prediction_id):
                return {"id": prediction_id, "status": "running"}

        class Files:
            def __init__(self, client):
                self._client = client

            def list(self):
                return [
                    {
                        "id": "file1",
                        "filename": "test.txt",
                        "size": 100,
                        "created_at": "2024-01-01",
                    }
                ]

            def upload(self, file_path, purpose="fine-tune"):
                return {"id": "file1", "filename": file_path}

            def get(self, file_id):
                return {
                    "id": file_id,
                    "filename": "test.txt",
                    "size": 100,
                    "created_at": "2024-01-01",
                }

            def get_content(self, file_id):
                return b"test content"

            def delete(self, file_id):
                return True

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

        class Image:
            def __init__(self, client):
                self._client = client

            def generate(self, *args, **kwargs):
                return b"image data"

        class Video:
            def __init__(self, client):
                self._client = client

            def generate(self, *args, **kwargs):
                return b"video data"

        class Document:
            def __init__(self, client):
                self._client = client

            def generate(self, *args, **kwargs):
                return b"document data"

    monkeypatch.setattr("vlmrun.cli.cli.Client", MockClient)
    return MockClient()
