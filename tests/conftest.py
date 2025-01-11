"""Test fixtures for vlmrun tests."""

import pytest
from typer.testing import CliRunner

from vlmrun.client.types import ModelResponse


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
                    "status": "running",
                    "created_at": "2024-01-01",
                }
            ]

        def get_fine_tuning_job(self, job_id):
            return {"id": job_id, "status": "running"}

        def cancel_fine_tuning_job(self, job_id):
            return True

        def get_fine_tuning_job_status(self, job_id):
            return {"status": "running"}

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

    monkeypatch.setattr("vlmrun.cli.cli.Client", MockClient)
    return MockClient()
