"""Test fixtures for vlmrun tests."""

import pytest
from unittest.mock import MagicMock
from typer.testing import CliRunner

from datetime import datetime
from vlmrun.client.types import (
    ModelResponse,
    DatasetResponse,
    ModelFinetuningRequest,
    ModelFinetuningResponse,
    FinetunedImagePredictionRequest,
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
            self.finetune = self.FineTuning(self)

        class FineTuning:
            def __init__(self, client):
                self._client = client
                self.health = MagicMock(return_value=True)
                self.list_models = MagicMock(side_effect=self._list_models)
                self.create_fit = MagicMock(side_effect=self._create_fit)
                self.retrieve_fit = MagicMock(side_effect=self._retrieve_fit)
                self.generate = MagicMock(side_effect=self._generate)

            def _list_models(self):
                if getattr(self.list_models, 'return_value', None) == [{"invalid": "response"}]:
                    raise TypeError("Expected dict for request data in /models/ response")
                return [ModelFinetuningResponse(
                    id="job1",
                    model="ft:test-model:org:123",
                    status="pending",
                    request=ModelFinetuningRequest(
                        model="test-model",
                        dataset_uri="test-uri",
                        num_epochs=1
                    )
                )]

            def _create_fit(self, request):
                if not isinstance(request, ModelFinetuningRequest):
                    raise TypeError("Invalid request type")
                return ModelFinetuningResponse(
                    id="job1",
                    model="test-model",
                    status="pending",
                    request=request
                )

            def _retrieve_fit(self, training_id):
                if getattr(self.retrieve_fit, 'side_effect', None):
                    if isinstance(self.retrieve_fit.side_effect, Exception):
                        raise self.retrieve_fit.side_effect
                return ModelFinetuningResponse(
                    id=training_id,
                    model="test-model",
                    status="pending",
                    request=ModelFinetuningRequest(
                        model="test-model",
                        dataset_uri="test-uri",
                        num_epochs=1
                    )
                )

            def _generate(self, request):
                if not isinstance(request, FinetunedImagePredictionRequest):
                    raise TypeError("Invalid request type")
                if getattr(self.generate, 'return_value', None) == "invalid response type":
                    raise TypeError("Expected dict response from /generate endpoint")
                return {"prediction": "test prediction"}

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

        def list_files(self):
            return [
                {
                    "id": "file1",
                    "filename": "test.txt",
                    "size": 100,
                    "created_at": "2024-01-01",
                }
            ]

        def upload_file(self, file_path, purpose="fine-tune"):
            return {"id": "file1", "filename": file_path}

        def delete_file(self, file_id):
            return True

        def get_file(self, file_id):
            return b"test content"

        def create_fine_tuning_job(self, training_file, model, **kwargs):
            return {"id": "job1"}

        def list_fine_tuning_jobs(self):
            return [
                {
                    "id": "job1",
                    "model": "test-model",
                    "status": "pending",
                    "created_at": "2024-01-01",
                }
            ]

        def get_fine_tuning_job(self, job_id):
            return {"id": job_id, "status": "pending"}

        def cancel_fine_tuning_job(self, job_id):
            return True

        def get_fine_tuning_job_status(self, job_id):
            return {"status": "pending"}

        def list_models(self):
            return [ModelResponse(model="model1", domain="test-domain")]

        def generate_image(self, prompt):
            return b"image data"

        def generate_video(self, prompt):
            return b"video data"

        def generate_document(self, prompt):
            return b"document data"

        def get_hub_version(self):
            return "0.1.0"

        def list_hub_items(self):
            return [
                {
                    "id": "item1",
                    "name": "Test Item",
                    "type": "model",
                    "version": "1.0.0",
                }
            ]

        def submit_hub_item(self, path, name, version):
            return {"id": "item1"}

        def healthcheck_finetuning(self):
            """Check the health of the fine-tuning service."""
            return self.finetune.health()

        def list_finetuned_models(self):
            """List all fine-tuned models."""
            return self.finetune.list_models()

        def create_finetuning(self, request):
            """Create a fine-tuning job."""
            return self.finetune.create_fit(request)

        def retrieve_finetuning(self, training_id):
            """Get fine-tuning job status."""
            return self.finetune.retrieve_fit(training_id)

        def generate_prediction(self, request):
            """Generate predictions using a fine-tuned model."""
            return self.finetune.generate(request)

    monkeypatch.setattr("vlmrun.cli.cli.Client", MockClient)
    return MockClient()
