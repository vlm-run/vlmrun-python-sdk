"""Tests for the exceptions module."""

import pytest
from tenacity import RetryError

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
    RequestTimeoutError,
    NetworkError,
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
        request_id="req-123",
        error_type="test_error",
        suggestion="Try again",
    )
    assert error.message == "API error"
    assert error.http_status == 400
    assert error.request_id == "req-123"
    assert error.headers == {}
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
    """Test RequestTimeoutError."""
    error = RequestTimeoutError()
    assert error.message == "Request timed out"
    assert error.http_status == 408
    assert error.error_type == "timeout_error"
    assert "Try again later" in error.suggestion

    error = RequestTimeoutError(message="Operation timed out")
    assert error.message == "Operation timed out"
    assert error.http_status == 408

    assert isinstance(error, APIError)


def test_network_error():
    """Test NetworkError."""
    error = NetworkError()
    assert error.message == "Network error"
    assert error.error_type == "network_error"
    assert "Check your internet connection" in error.suggestion

    error = NetworkError(message="Connection failed")
    assert error.message == "Connection failed"

    assert isinstance(error, APIError)


def test_retry_error_handling():
    """Test that RetryError is properly handled and converts to appropriate exceptions."""

    # Create a mock client
    class MockClient:
        def __init__(self):
            self.api_key = "test-key"
            self.base_url = "https://api.test"
            self.max_retries = 3

    client = MockClient()
    requestor = APIRequestor(client)

    # Test all our custom error types are preserved
    error_types = [
        (AuthenticationError("Invalid API key"), "Invalid API key"),
        (ValidationError("Invalid parameter"), "Invalid parameter"),
        (RateLimitError("Too many requests"), "Too many requests"),
        (ResourceNotFoundError("Resource not found"), "Resource not found"),
        (ServerError("Server is down"), "Server is down"),
        (RequestTimeoutError("Request timed out"), "Request timed out"),
        (NetworkError("Connection failed"), "Connection failed"),
    ]

    for error, expected_message in error_types:
        retry_error = RetryError(
            last_attempt=type("obj", (object,), {"exception": lambda: error})
        )

        with pytest.raises(type(error)) as exc_info:
            requestor._handle_retry_error(retry_error)
        assert exc_info.value.message == expected_message

    # Test unknown error wrapping
    unknown_error = Exception("Unknown error")
    retry_error = RetryError(
        last_attempt=type("obj", (object,), {"exception": lambda: unknown_error})
    )

    with pytest.raises(APIError) as exc_info:
        requestor._handle_retry_error(retry_error)
    assert "Request failed after 3 retries" in exc_info.value.message
    assert "Unknown error" in exc_info.value.message
