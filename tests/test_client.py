import os
import requests
import pytest
from functools import lru_cache
from unittest.mock import patch
from vlmrun.client import VLMRun
from vlmrun.client.exceptions import ConfigurationError, AuthenticationError


def test_client_with_api_key(monkeypatch):
    """Test client initialization with API key provided in constructor."""
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment

    # Mock the health check request
    with patch("vlmrun.client.base_requestor.APIRequestor.request") as mock_request:
        mock_request.return_value = (None, 200, {})
        client = VLMRun(api_key="test-key")
        assert client.api_key == "test-key"
        assert (
            client.base_url == "https://api.vlm.run/v1"
        )  # Default URL with /v1 suffix


def test_client_with_env_api_key(monkeypatch):
    """Test client initialization with API key from environment variable."""
    monkeypatch.setenv("VLMRUN_API_KEY", "env-key")
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment

    # Mock the health check request
    with patch("vlmrun.client.base_requestor.APIRequestor.request") as mock_request:
        mock_request.return_value = (None, 200, {})
        client = VLMRun()
        assert client.api_key == "env-key"
        assert (
            client.base_url == "https://api.vlm.run/v1"
        )  # Default URL with /v1 suffix


def test_client_with_base_url():
    """Test client initialization with custom base URL."""
    # Mock the health check request
    with patch("vlmrun.client.base_requestor.APIRequestor.request") as mock_request:
        mock_request.return_value = (None, 200, {})
        client = VLMRun(api_key="test-key", base_url="https://custom.api")
        assert client.api_key == "test-key"
        assert client.base_url == "https://custom.api"


def test_client_with_env_base_url(monkeypatch):
    """Test client initialization with base URL from environment variable."""
    monkeypatch.setenv("VLMRUN_BASE_URL", "https://env.api")

    # Mock the health check request
    with patch("vlmrun.client.base_requestor.APIRequestor.request") as mock_request:
        mock_request.return_value = (None, 200, {})
        client = VLMRun(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://env.api"


def test_client_missing_api_key(monkeypatch):
    """Test client initialization fails when API key is missing."""
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)  # Ensure no API key in env
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment

    with pytest.raises(ConfigurationError) as exc_info:
        VLMRun()
    assert "Missing API key" in str(exc_info.value)


def test_client_env_precedence(monkeypatch):
    """Test that constructor values take precedence over environment variables."""
    monkeypatch.setenv("VLMRUN_API_KEY", "env-key")
    monkeypatch.setenv("VLMRUN_BASE_URL", "https://env.api")

    # Mock the health check request
    with patch("vlmrun.client.base_requestor.APIRequestor.request") as mock_request:
        mock_request.return_value = (None, 200, {})
        client = VLMRun(api_key="test-key", base_url="https://custom.api")
        assert client.api_key == "test-key"  # Constructor value
        assert client.base_url == "https://custom.api"  # Constructor value


@lru_cache  # needs to be checked just once
def _healthcheck():
    return (
        requests.get(
            os.getenv("VLMRUN_BASE_URL", "https://api.vlm.run/v1") + "/health"
        ).status_code
        == 200
    )


@pytest.mark.skipif(
    os.getenv("VLMRUN_API_KEY", None) is None
    and os.getenv("VLMRUN_BASE_URL", None) is None,
    reason="No VLMRUN_API_KEY and VLMRUN_BASE_URL in environment",
)
@pytest.mark.skipif(not _healthcheck(), reason="API is not healthy")
def test_client_health():
    """Test client health check."""
    client = VLMRun()
    assert client.healthcheck()
    assert len(client.models.list()) > 0, "No models found"


@pytest.mark.skipif(
    os.getenv("VLMRUN_API_KEY", None) is None
    and os.getenv("VLMRUN_BASE_URL", None) is None,
    reason="No VLMRUN_API_KEY and VLMRUN_BASE_URL in environment",
)
@pytest.mark.skipif(not _healthcheck(), reason="API is not healthy")
def test_client_openai():
    """Test client OpenAI integration."""
    client = VLMRun()
    assert client.openai is not None
    assert client.openai.models.list() is not None


def test_client_invalid_api_key(monkeypatch):
    """Test client initialization fails when API key is invalid."""
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment

    # Mock the health check request to simulate an invalid API key
    with patch("vlmrun.client.base_requestor.APIRequestor.request") as mock_request:
        mock_request.side_effect = AuthenticationError(
            message="Invalid API key",
            error_type="invalid_api_key",
            suggestion="Please check your API key and ensure it is valid. You can get your API key at https://app.vlm.run/dashboard",
        )

        with pytest.raises(AuthenticationError) as exc_info:
            VLMRun(api_key="invalid-key")
        assert "Invalid API key" in str(exc_info.value)
        assert "Please check your API key" in exc_info.value.suggestion


def test_client_valid_api_key(monkeypatch):
    """Test client initialization succeeds with valid API key."""
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment

    # Mock the health check request to simulate a valid API key
    with patch("vlmrun.client.base_requestor.APIRequestor.request") as mock_request:
        mock_request.return_value = (None, 200, {})

        client = VLMRun(api_key="valid-key")
        assert client.api_key == "valid-key"
        assert (
            client.base_url == "https://api.vlm.run/v1"
        )  # Default URL with /v1 suffix


def test_client_health_check_failure(monkeypatch):
    """Test client initialization fails when health check returns non-200 status."""
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment

    # Mock the health check request to return a non-200 status
    with patch("vlmrun.client.base_requestor.APIRequestor.request") as mock_request:
        mock_request.return_value = (None, 401, {})

        with pytest.raises(AuthenticationError) as exc_info:
            VLMRun(api_key="test-key")
        assert "Invalid API key" in str(exc_info.value)
        assert "Please check your API key" in exc_info.value.suggestion
