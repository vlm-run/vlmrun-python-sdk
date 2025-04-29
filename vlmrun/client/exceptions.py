"""VLM Run API exceptions."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class VLMRunError(Exception):
    """Base exception for all VLM Run errors."""

    message: str

    def __post_init__(self):
        super().__init__(self.message)


@dataclass
class APIError(VLMRunError):
    """Base exception for API errors."""

    message: str
    http_status: Optional[int] = None
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    error_type: Optional[str] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        """Return string representation of error."""
        parts = []
        if self.http_status:
            parts.append(f"status={self.http_status}")
        if self.error_type:
            parts.append(f"type={self.error_type}")
        if self.request_id:
            parts.append(f"id={self.request_id}")

        formatted = f"[{', '.join(parts)}] {self.message}"
        if self.suggestion:
            formatted += f" (Suggestion: {self.suggestion})"
        return formatted


@dataclass
class AuthenticationError(APIError):
    """Exception raised when authentication fails."""

    message: str = "Authentication failed"
    http_status: int = 401
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    error_type: str = "authentication_error"
    suggestion: str = "Check your API key and ensure it is valid"


@dataclass
class ValidationError(APIError):
    """Exception raised when request validation fails."""

    message: str = "Validation failed"
    http_status: int = 400
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    error_type: str = "validation_error"
    suggestion: str = "Check your request parameters"


@dataclass
class RateLimitError(APIError):
    """Exception raised when rate limit is exceeded."""

    message: str = "Rate limit exceeded"
    http_status: int = 429
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    error_type: str = "rate_limit_error"
    suggestion: str = (
        "Reduce request frequency or contact support to increase your rate limit"
    )


@dataclass
class ServerError(APIError):
    """Exception raised when server returns 5xx error."""

    message: str = "Server error"
    http_status: int = 500
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    error_type: str = "server_error"
    suggestion: str = "Please try again later or contact support if the issue persists"


@dataclass
class ResourceNotFoundError(APIError):
    """Exception raised when resource is not found."""

    message: str = "Resource not found"
    http_status: int = 404
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    error_type: str = "not_found_error"
    suggestion: str = "Check the resource ID or path"


@dataclass
class ClientError(VLMRunError):
    """Base exception for client-side errors."""

    message: str
    error_type: Optional[str] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        """Return string representation of error."""
        parts = []
        if self.error_type:
            parts.append(f"type={self.error_type}")

        formatted = f"[{', '.join(parts)}] {self.message}"
        if self.suggestion:
            formatted += f" (Suggestion: {self.suggestion})"
        return formatted


@dataclass
class ConfigurationError(ClientError):
    """Exception raised when client configuration is invalid."""

    message: str = "Invalid configuration"
    error_type: str = "configuration_error"
    suggestion: str = "Check your client configuration"


@dataclass
class DependencyError(ClientError):
    """Exception raised when a required dependency is missing."""

    message: str = "Missing dependency"
    error_type: str = "dependency_error"
    suggestion: str = "Install the required dependency"


@dataclass
class InputError(ClientError):
    """Exception raised when input is invalid."""

    message: str = "Invalid input"
    error_type: str = "input_error"
    suggestion: str = "Check your input parameters"


@dataclass
class RequestTimeoutError(APIError):
    """Exception raised when a request times out."""

    message: str = "Request timed out"
    http_status: int = 408
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    error_type: str = "timeout_error"
    suggestion: str = "Try again later or increase the timeout"


@dataclass
class NetworkError(APIError):
    """Exception raised when a network error occurs."""

    message: str = "Network error"
    http_status: Optional[int] = None
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    error_type: str = "network_error"
    suggestion: str = "Check your internet connection and try again"
