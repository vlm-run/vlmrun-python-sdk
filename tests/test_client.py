import os
import requests
import pytest
from functools import lru_cache
from vlmrun.client import VLMRun


def test_vlmrun_with_api_key(monkeypatch):
    """Test VLMRun initialization with API key provided in constructor."""
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment
    vlm = VLMRun(api_key="test-key")
    assert vlm.api_key == "test-key"
    assert vlm.base_url == "https://api.vlm.run/v1"  # Default URL with /v1 suffix


def test_vlmrun_with_env_api_key(monkeypatch):
    """Test VLMRun initialization with API key from environment variable."""
    monkeypatch.setenv("VLMRUN_API_KEY", "env-key")
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment
    vlm = VLMRun()
    assert vlm.api_key == "env-key"
    assert vlm.base_url == "https://api.vlm.run/v1"  # Default URL with /v1 suffix


def test_vlmrun_with_base_url():
    """Test VLMRun initialization with custom base URL."""
    vlm = VLMRun(api_key="test-key", base_url="https://custom.api")
    assert vlm.api_key == "test-key"
    assert vlm.base_url == "https://custom.api"


def test_vlmrun_with_env_base_url(monkeypatch):
    """Test VLMRun initialization with base URL from environment variable."""
    monkeypatch.setenv("VLMRUN_BASE_URL", "https://env.api")
    vlm = VLMRun(api_key="test-key")
    assert vlm.api_key == "test-key"
    assert vlm.base_url == "https://env.api"


def test_vlmrun_missing_api_key(monkeypatch):
    """Test VLMRun initialization fails when API key is missing."""
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)  # Ensure no API key in env
    monkeypatch.delenv("VLMRUN_BASE_URL", raising=False)  # Ensure clean environment
    with pytest.raises(ValueError) as exc_info:
        VLMRun()
    assert "API key must be provided" in str(exc_info.value)


def test_vlmrun_env_precedence(monkeypatch):
    """Test that VLMRun constructor values take precedence over environment variables."""
    monkeypatch.setenv("VLMRUN_API_KEY", "env-key")
    monkeypatch.setenv("VLMRUN_BASE_URL", "https://env.api")

    vlm = VLMRun(api_key="test-key", base_url="https://custom.api")
    assert vlm.api_key == "test-key"  # Constructor value
    assert vlm.base_url == "https://custom.api"  # Constructor value


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
def test_vlmrun_health():
    """Test VLMRun health check."""
    vlm = VLMRun()
    assert vlm.healthcheck()
    assert len(vlm.models.list()) > 0, "No models found"


@pytest.mark.skipif(
    os.getenv("VLMRUN_API_KEY", None) is None
    and os.getenv("VLMRUN_BASE_URL", None) is None,
    reason="No VLMRUN_API_KEY and VLMRUN_BASE_URL in environment",
)
@pytest.mark.skipif(not _healthcheck(), reason="API is not healthy")
def test_vlmrun_openai():
    """Test VLMRun OpenAI integration."""
    vlm = VLMRun()
    assert vlm.openai is not None
    assert vlm.openai.models.list() is not None
