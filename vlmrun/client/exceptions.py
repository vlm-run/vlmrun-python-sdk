"""VLM Run API exceptions."""

from typing import Dict, Optional


class VLMRunError(Exception):
    """Base exception for all VLM Run errors."""

    def __init__(self, message: str):
        """Initialize VLMRunError.

        Args:
            message: Error message
        """
        super().__init__(message)
        self.message = message


class APIError(VLMRunError):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        http_status: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
        error_type: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize APIError.

        Args:
            message: Error message
            http_status: HTTP status code
            headers: Response headers
            request_id: Request ID
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(message)
        self.http_status = http_status
        self.headers = headers or {}
        self.request_id = request_id
        self.error_type = error_type
        self.suggestion = suggestion

    def __str__(self) -> str:
        """Return string representation of error."""
        parts = [self.message]
        if self.http_status:
            parts.append(f"HTTP Status: {self.http_status}")
        if self.error_type:
            parts.append(f"Error Type: {self.error_type}")
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class AuthenticationError(APIError):
    """Exception raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        http_status: Optional[int] = 401,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
        error_type: Optional[str] = "authentication_error",
        suggestion: Optional[str] = "Check your API key and ensure it is valid",
    ):
        """Initialize AuthenticationError.

        Args:
            message: Error message
            http_status: HTTP status code
            headers: Response headers
            request_id: Request ID
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(
            message=message,
            http_status=http_status,
            headers=headers,
            request_id=request_id,
            error_type=error_type,
            suggestion=suggestion,
        )


class ValidationError(APIError):
    """Exception raised when request validation fails."""

    def __init__(
        self,
        message: str = "Validation failed",
        http_status: Optional[int] = 400,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
        error_type: Optional[str] = "validation_error",
        suggestion: Optional[str] = "Check your request parameters",
    ):
        """Initialize ValidationError.

        Args:
            message: Error message
            http_status: HTTP status code
            headers: Response headers
            request_id: Request ID
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(
            message=message,
            http_status=http_status,
            headers=headers,
            request_id=request_id,
            error_type=error_type,
            suggestion=suggestion,
        )


class RateLimitError(APIError):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        http_status: Optional[int] = 429,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
        error_type: Optional[str] = "rate_limit_error",
        suggestion: Optional[str] = "Reduce request frequency or contact support to increase your rate limit",
    ):
        """Initialize RateLimitError.

        Args:
            message: Error message
            http_status: HTTP status code
            headers: Response headers
            request_id: Request ID
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(
            message=message,
            http_status=http_status,
            headers=headers,
            request_id=request_id,
            error_type=error_type,
            suggestion=suggestion,
        )


class ServerError(APIError):
    """Exception raised when server returns 5xx error."""

    def __init__(
        self,
        message: str = "Server error",
        http_status: Optional[int] = 500,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
        error_type: Optional[str] = "server_error",
        suggestion: Optional[str] = "Please try again later or contact support if the issue persists",
    ):
        """Initialize ServerError.

        Args:
            message: Error message
            http_status: HTTP status code
            headers: Response headers
            request_id: Request ID
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(
            message=message,
            http_status=http_status,
            headers=headers,
            request_id=request_id,
            error_type=error_type,
            suggestion=suggestion,
        )


class ResourceNotFoundError(APIError):
    """Exception raised when resource is not found."""

    def __init__(
        self,
        message: str = "Resource not found",
        http_status: Optional[int] = 404,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
        error_type: Optional[str] = "not_found_error",
        suggestion: Optional[str] = "Check the resource ID or path",
    ):
        """Initialize ResourceNotFoundError.

        Args:
            message: Error message
            http_status: HTTP status code
            headers: Response headers
            request_id: Request ID
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(
            message=message,
            http_status=http_status,
            headers=headers,
            request_id=request_id,
            error_type=error_type,
            suggestion=suggestion,
        )


class ClientError(VLMRunError):
    """Base exception for client-side errors."""

    def __init__(
        self,
        message: str,
        error_type: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize ClientError.

        Args:
            message: Error message
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(message)
        self.error_type = error_type
        self.suggestion = suggestion

    def __str__(self) -> str:
        """Return string representation of error."""
        parts = [self.message]
        if self.error_type:
            parts.append(f"Error Type: {self.error_type}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class ConfigurationError(ClientError):
    """Exception raised when client configuration is invalid."""

    def __init__(
        self,
        message: str = "Invalid configuration",
        error_type: Optional[str] = "configuration_error",
        suggestion: Optional[str] = "Check your client configuration",
    ):
        """Initialize ConfigurationError.

        Args:
            message: Error message
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(
            message=message,
            error_type=error_type,
            suggestion=suggestion,
        )


class DependencyError(ClientError):
    """Exception raised when a required dependency is missing."""

    def __init__(
        self,
        message: str = "Missing dependency",
        error_type: Optional[str] = "dependency_error",
        suggestion: Optional[str] = "Install the required dependency",
    ):
        """Initialize DependencyError.

        Args:
            message: Error message
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(
            message=message,
            error_type=error_type,
            suggestion=suggestion,
        )


class InputError(ClientError):
    """Exception raised when input is invalid."""

    def __init__(
        self,
        message: str = "Invalid input",
        error_type: Optional[str] = "input_error",
        suggestion: Optional[str] = "Check your input parameters",
    ):
        """Initialize InputError.

        Args:
            message: Error message
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(
            message=message,
            error_type=error_type,
            suggestion=suggestion,
        )


class TimeoutError(VLMRunError):
    """Exception raised when a request times out."""

    def __init__(
        self,
        message: str = "Request timed out",
        error_type: Optional[str] = "timeout_error",
        suggestion: Optional[str] = "Try again later or increase the timeout",
    ):
        """Initialize TimeoutError.

        Args:
            message: Error message
            error_type: Error type identifier
            suggestion: Suggestion for resolution
        """
        super().__init__(message)
        self.error_type = error_type
        self.suggestion = suggestion

    def __str__(self) -> str:
        """Return string representation of error."""
        parts = [self.message]
        if self.error_type:
            parts.append(f"Error Type: {self.error_type}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)
