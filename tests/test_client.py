import pytest
from vlmrun.client import Client


def test_client_with_api_key(monkeypatch):
    """Test client initialization with API key provided in constructor."""
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment
    client = Client(api_key="test-key")
    assert client.api_key == "test-key"
    assert client.base_url == "https://api.vlm.run/v1"  # Default URL with /v1 suffix


def test_client_with_env_api_key(monkeypatch):
    """Test client initialization with API key from environment variable."""
    monkeypatch.setenv("VLMRUN_API_KEY", "env-key")
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment
    client = Client()
    assert client.api_key == "env-key"
    assert client.base_url == "https://api.vlm.run/v1"  # Default URL with /v1 suffix


def test_client_with_base_url():
    """Test client initialization with custom base URL."""
    client = Client(api_key="test-key", base_url="https://custom.api")
    assert client.api_key == "test-key"
    assert client.base_url == "https://custom.api"


def test_client_with_env_base_url(monkeypatch):
    """Test client initialization with base URL from environment variable."""
    monkeypatch.setenv("VLMRUN_BASE_URL", "https://env.api")
    client = Client(api_key="test-key")
    assert client.api_key == "test-key"
    assert client.base_url == "https://env.api"


def test_client_missing_api_key(monkeypatch):
    """Test client initialization fails when API key is missing."""
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)  # Ensure no API key in env
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment
    with pytest.raises(ValueError) as exc_info:
        Client()
    assert "API key must be provided" in str(exc_info.value)


def test_client_env_precedence(monkeypatch):
    """Test that constructor values take precedence over environment variables."""
    monkeypatch.setenv("VLMRUN_API_KEY", "env-key")
    monkeypatch.setenv("VLMRUN_BASE_URL", "https://env.api")

    client = Client(api_key="test-key", base_url="https://custom.api")
    assert client.api_key == "test-key"  # Constructor value
    assert client.base_url == "https://custom.api"  # Constructor value
