"""Tests for models operations."""

from vlmrun.client.types import ModelInfo


def test_list_models(mock_client):
    """Test listing available models."""
    response = mock_client.models.list()
    assert isinstance(response, list)
    assert all(isinstance(model, ModelInfo) for model in response)
    assert len(response) > 0
    model = response[0]
    assert isinstance(model.model, str)
    assert isinstance(model.domain, str)
    assert model.model == "model1"
    assert model.domain == "test-domain"
