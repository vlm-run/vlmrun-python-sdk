"""Tests for the exceptions module."""

import pytest
import requests
from unittest.mock import MagicMock, patch

from vlmrun.client.exceptions import (
    VLMRunError,
    APIError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
    ServerError,
    ResourceNotFoundError,
    ClientError,
    ConfigurationError,
    DependencyError,
    InputError,
    TimeoutError,
)
from vlmrun.client.base_requestor import APIRequestor


def test_vlmrun_error():
    """Test VLMRunError."""
    error = VLMRunError("Test error")
    assert str(error) == "Test error"
    assert error.message == "Test error"


def test_api_error():
    """Test APIError."""
    error = APIError(
        message="API error",
        http_status=400,
        headers={"X-Request-ID": "req-123"},
        request_id="req-123",
        error_type="test_error",
        suggestion="Try again",
    )
    assert error.message == "API error"
    assert error.http_status == 400
    assert error.headers == {"X-Request-ID": "req-123"}
    assert error.request_id == "req-123"
    assert error.error_type == "test_error"
    assert error.suggestion == "Try again"
    assert "API error" in str(error)
    assert "status=400" in str(error)
    assert "type=test_error" in str(error)
    assert "id=req-123" in str(error)
    assert "Try again" in str(error)


def test_authentication_error():
    """Test AuthenticationError."""
    error = AuthenticationError()
    assert error.message == "Authentication failed"
    assert error.http_status == 401
    assert error.error_type == "authentication_error"
    assert "Check your API key" in error.suggestion

    error = AuthenticationError(message="Invalid API key")
    assert error.message == "Invalid API key"
    assert error.http_status == 401


def test_validation_error():
    """Test ValidationError."""
    error = ValidationError()
    assert error.message == "Validation failed"
    assert error.http_status == 400
    assert error.error_type == "validation_error"
    assert "Check your request parameters" in error.suggestion

    error = ValidationError(message="Invalid parameter")
    assert error.message == "Invalid parameter"
    assert error.http_status == 400


def test_rate_limit_error():
    """Test RateLimitError."""
    error = RateLimitError()
    assert error.message == "Rate limit exceeded"
    assert error.http_status == 429
    assert error.error_type == "rate_limit_error"
    assert "Reduce request frequency" in error.suggestion

    error = RateLimitError(message="Too many requests")
    assert error.message == "Too many requests"
    assert error.http_status == 429


def test_server_error():
    """Test ServerError."""
    error = ServerError()
    assert error.message == "Server error"
    assert error.http_status == 500
    assert error.error_type == "server_error"
    assert "try again later" in error.suggestion.lower()

    error = ServerError(message="Internal server error")
    assert error.message == "Internal server error"
    assert error.http_status == 500


def test_resource_not_found_error():
    """Test ResourceNotFoundError."""
    error = ResourceNotFoundError()
    assert error.message == "Resource not found"
    assert error.http_status == 404
    assert error.error_type == "not_found_error"
    assert "Check the resource ID" in error.suggestion

    error = ResourceNotFoundError(message="Domain not found")
    assert error.message == "Domain not found"
    assert error.http_status == 404


def test_client_error():
    """Test ClientError."""
    error = ClientError(
        message="Client error",
        error_type="test_error",
        suggestion="Fix your client",
    )
    assert error.message == "Client error"
    assert error.error_type == "test_error"
    assert error.suggestion == "Fix your client"
    assert "Client error" in str(error)
    assert "type=test_error" in str(error)
    assert "Fix your client" in str(error)


def test_configuration_error():
    """Test ConfigurationError."""
    error = ConfigurationError()
    assert error.message == "Invalid configuration"
    assert error.error_type == "configuration_error"
    assert "Check your client configuration" in error.suggestion

    error = ConfigurationError(message="Missing configuration")
    assert error.message == "Missing configuration"


def test_dependency_error():
    """Test DependencyError."""
    error = DependencyError()
    assert error.message == "Missing dependency"
    assert error.error_type == "dependency_error"
    assert "Install the required dependency" in error.suggestion

    error = DependencyError(message="Missing OpenAI package")
    assert error.message == "Missing OpenAI package"


def test_input_error():
    """Test InputError."""
    error = InputError()
    assert error.message == "Invalid input"
    assert error.error_type == "input_error"
    assert "Check your input parameters" in error.suggestion

    error = InputError(message="Invalid image format")
    assert error.message == "Invalid image format"


def test_timeout_error():
    """Test TimeoutError."""
    error = TimeoutError()
    assert error.message == "Request timed out"
    assert error.error_type == "timeout_error"
    assert "Try again later" in error.suggestion

    error = TimeoutError(message="Operation timed out")
    assert error.message == "Operation timed out"


@patch("requests.Session.request")
def test_api_requestor_authentication_error(mock_request):
    """Test APIRequestor raises AuthenticationError for 401 responses."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
    mock_response.headers = {"X-Request-ID": "req-123"}
    
    mock_request.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = "https://api.test.com"
    
    requestor = APIRequestor(client)
    
    with pytest.raises(AuthenticationError) as exc_info:
        requestor.request(method="GET", url="/test")
    
    assert exc_info.value.http_status == 401
    assert "Invalid API key" in str(exc_info.value)


@patch("requests.Session.request")
def test_api_requestor_validation_error(mock_request):
    """Test APIRequestor raises ValidationError for 400 responses."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"error": {"message": "Invalid parameter"}}
    mock_response.headers = {"X-Request-ID": "req-123"}
    
    mock_request.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = "https://api.test.com"
    
    requestor = APIRequestor(client)
    
    with pytest.raises(ValidationError) as exc_info:
        requestor.request(method="GET", url="/test")
    
    assert exc_info.value.http_status == 400
    assert "Invalid parameter" in str(exc_info.value)


@patch("requests.Session.request")
def test_api_requestor_not_found_error(mock_request):
    """Test APIRequestor raises ResourceNotFoundError for 404 responses."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = {"error": {"message": "Resource not found"}}
    mock_response.headers = {"X-Request-ID": "req-123"}
    
    mock_request.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = "https://api.test.com"
    
    requestor = APIRequestor(client)
    
    with pytest.raises(ResourceNotFoundError) as exc_info:
        requestor.request(method="GET", url="/test")
    
    assert exc_info.value.http_status == 404
    assert "Resource not found" in str(exc_info.value)


@patch("requests.Session.request")
def test_api_requestor_rate_limit_error(mock_request):
    """Test APIRequestor raises RateLimitError for 429 responses."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    mock_response.headers = {"X-Request-ID": "req-123"}
    
    mock_request.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = "https://api.test.com"
    
    requestor = APIRequestor(client)
    
    with pytest.raises(RateLimitError) as exc_info:
        requestor.request(method="GET", url="/test")
    
    assert exc_info.value.http_status == 429
    assert "Rate limit exceeded" in str(exc_info.value)


@patch("requests.Session.request")
def test_api_requestor_server_error(mock_request):
    """Test APIRequestor raises ServerError for 5xx responses."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"error": {"message": "Internal server error"}}
    mock_response.headers = {"X-Request-ID": "req-123"}
    
    mock_request.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = "https://api.test.com"
    
    requestor = APIRequestor(client)
    
    with pytest.raises(ServerError) as exc_info:
        requestor.request(method="GET", url="/test")
    
    assert exc_info.value.http_status == 500
    assert "Internal server error" in str(exc_info.value)


@patch("requests.Session.request")
def test_api_requestor_timeout_error(mock_request):
    """Test APIRequestor raises TimeoutError for timeout exceptions."""
    mock_request.side_effect = requests.exceptions.Timeout("Request timed out")
    
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = "https://api.test.com"
    
    requestor = APIRequestor(client)
    
    with pytest.raises(TimeoutError) as exc_info:
        requestor.request(method="GET", url="/test")
    
    assert "Request timed out" in str(exc_info.value)
