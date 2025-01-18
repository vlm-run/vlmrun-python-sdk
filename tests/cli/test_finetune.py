"""Test fine-tuning subcommand."""

from typing import Dict, List

import pytest
from vlmrun.cli.cli import app
from vlmrun.client.types import (
    ModelFinetuningRequest,
    ModelFinetuningResponse,
    FinetunedImagePredictionRequest,
)


@pytest.fixture
def mock_response() -> ModelFinetuningResponse:
    """Create a mock fine-tuning response."""
    request = ModelFinetuningRequest(
        model="test-model",
        dataset_uri="https://example.com/data.jsonl",
        num_epochs=1,
    )
    return ModelFinetuningResponse(
        id="job1",
        model="ft:test-model:org:123",
        status="pending",
        request=request,
    )


def test_healthcheck_finetuning(mock_client):
    """Test fine-tuning healthcheck."""
    mock_client.finetune.health.return_value = True
    result = mock_client.healthcheck_finetuning()
    assert result is True
    mock_client.finetune.health.assert_called_once()


def test_list_finetuned_models(mock_client, mock_response):
    """Test listing fine-tuned models."""
    mock_client.finetune.list_models.return_value = [mock_response]
    result = mock_client.list_finetuned_models()
    assert len(result) == 1
    assert isinstance(result[0], ModelFinetuningResponse)
    assert result[0].id == "job1"
    assert result[0].model == "ft:test-model:org:123"
    mock_client.finetune.list_models.assert_called_once()


def test_create_finetuning(mock_client, mock_response):
    """Test creating fine-tuning job."""
    mock_client.finetune.create_fit.return_value = mock_response
    request = ModelFinetuningRequest(
        model="test-model",
        dataset_uri="https://example.com/data.jsonl",
        num_epochs=1,
    )
    result = mock_client.create_finetuning(request)
    assert isinstance(result, ModelFinetuningResponse)
    assert result.id == "job1"
    assert result.status == "pending"
    mock_client.finetune.create_fit.assert_called_once_with(request)


def test_retrieve_finetuning(mock_client, mock_response):
    """Test retrieving fine-tuning job status."""
    mock_client.finetune.retrieve_fit.return_value = mock_response
    result = mock_client.retrieve_finetuning("job1")
    assert isinstance(result, ModelFinetuningResponse)
    assert result.id == "job1"
    assert result.status == "pending"
    mock_client.finetune.retrieve_fit.assert_called_once_with("job1")


def test_generate_prediction(mock_client):
    """Test generating predictions with fine-tuned model."""
    expected_response = {"prediction": "test prediction"}
    mock_client.finetune.generate.return_value = expected_response
    request = FinetunedImagePredictionRequest(
        model="ft:test-model:org:123",
        image_data="base64_encoded_image",
        prompt="test prompt",
    )
    result = mock_client.generate_prediction(request)
    assert result == expected_response
    mock_client.finetune.generate.assert_called_once_with(request)


def test_list_models_invalid_response(mock_client):
    """Test handling of invalid response from list_models."""
    mock_client.finetune.list_models.return_value = [{"invalid": "response"}]
    with pytest.raises(TypeError, match="Expected dict for request data in /models/ response"):
        mock_client.list_finetuned_models()


def test_create_fit_invalid_request(mock_client):
    """Test handling of invalid request data in create_fit."""
    with pytest.raises(TypeError):
        mock_client.create_finetuning("invalid request")


def test_retrieve_fit_not_found(mock_client):
    """Test handling of non-existent training job."""
    mock_client.finetune.retrieve_fit.side_effect = Exception("Training not found")
    with pytest.raises(Exception, match="Training not found"):
        mock_client.retrieve_finetuning("nonexistent_job")


def test_generate_prediction_invalid_response(mock_client):
    """Test handling of invalid response from generate."""
    mock_client.finetune.generate.return_value = "invalid response type"
    request = FinetunedImagePredictionRequest(
        model="ft:test-model:org:123",
        image_data="base64_encoded_image",
    )
    with pytest.raises(TypeError, match="Expected dict response from /generate endpoint"):
        mock_client.generate_prediction(request)


def test_healthcheck_connection_error(mock_client):
    """Test handling of connection error in healthcheck."""
    mock_client.finetune.health.side_effect = ConnectionError("Failed to connect")
    with pytest.raises(ConnectionError, match="Failed to connect"):
        mock_client.healthcheck_finetuning()


# CLI Tests
def test_create_finetune(runner, mock_client, mock_response):
    """Test create fine-tuning command."""
    mock_client.finetune.create_fit.return_value = mock_response
    result = runner.invoke(app, ["fine-tuning", "create", "file1", "test-model"])
    assert result.exit_code == 0
    assert "job1" in result.stdout


def test_list_finetune(runner, mock_client, mock_response):
    """Test list fine-tuning command."""
    mock_client.finetune.list_models.return_value = [mock_response]
    result = runner.invoke(app, ["fine-tuning", "list"])
    assert result.exit_code == 0
    assert "job1" in result.stdout
    assert "test-model" in result.stdout


def test_get_finetune(runner, mock_client, mock_response):
    """Test get fine-tuning command."""
    mock_client.finetune.retrieve_fit.return_value = mock_response
    result = runner.invoke(app, ["fine-tuning", "get", "job1"])
    assert result.exit_code == 0
    assert "pending" in result.stdout


def test_cancel_finetune(runner, mock_client):
    """Test cancel fine-tuning command."""
    result = runner.invoke(app, ["fine-tuning", "cancel", "job1"])
    assert result.exit_code == 0


def test_status_finetune(runner, mock_client, mock_response):
    """Test status fine-tuning command."""
    mock_client.finetune.retrieve_fit.return_value = mock_response
    result = runner.invoke(app, ["fine-tuning", "status", "job1"])
    assert result.exit_code == 0
    assert "pending" in result.stdout
