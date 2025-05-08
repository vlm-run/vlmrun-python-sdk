"""VLM Run API requestor implementation."""

from typing import Any, Dict, Tuple, TYPE_CHECKING, Union, Optional
from urllib.parse import urljoin

if TYPE_CHECKING:
    from vlmrun.types.abstract import VLMRunProtocol

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)

from vlmrun.client.exceptions import (
    APIError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
    ServerError,
    ResourceNotFoundError,
    RequestTimeoutError,
    NetworkError,
)

# Constants
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 10  # seconds


class APIRequestor:
    """Handles API requests with retry logic."""

    def __init__(
        self,
        client: "VLMRunProtocol",
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: Optional[int] = None,
    ) -> None:
        """Initialize API requestor.

        Args:
            client: VLMRun API instance
            base_url: Base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self._client = client
        self._base_url = base_url or client.base_url
        self._timeout = timeout
        self._max_retries = (
            max_retries
            if max_retries is not None
            else getattr(client, "max_retries", DEFAULT_MAX_RETRIES)
        )
        self._session = requests.Session()

    def request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        raw_response: bool = False,
        timeout: Optional[float] = None,
    ) -> Union[
        Tuple[Dict[str, Any], int, Dict[str, str]], Tuple[bytes, int, Dict[str, str]]
    ]:
        """Make an API request with retry logic.

        Args:
            method: HTTP method
            url: API endpoint
            params: Query parameters
            data: Request body
            files: Files to upload
            headers: Request headers
            raw_response: Whether to return raw response content
            timeout: Request timeout in seconds

        Returns:
            Tuple of (response_data, status_code, response_headers)

        Raises:
            AuthenticationError: If authentication fails
            ValidationError: If request validation fails
            RateLimitError: If rate limit is exceeded
            ResourceNotFoundError: If resource is not found
            ServerError: If server returns 5xx error
            APIError: For other API errors
            RequestTimeoutError: If request times out
            NetworkError: If a network error occurs
        """
        retry_decorator = retry(
            retry=retry_if_exception_type(
                (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    ServerError,
                    RequestTimeoutError,
                    NetworkError,
                    RateLimitError,
                )
            ),
            wait=wait_exponential(
                multiplier=INITIAL_RETRY_DELAY,
                min=INITIAL_RETRY_DELAY,
                max=MAX_RETRY_DELAY,
            ),
            stop=stop_after_attempt(self._max_retries),
        )

        _headers = {} if headers is None else headers.copy()

        @retry_decorator
        def _request_with_retry():
            # Add authorization
            if self._client.api_key:
                _headers["Authorization"] = f"Bearer {self._client.api_key}"

            # Build full URL
            full_url = urljoin(self._base_url.rstrip("/") + "/", url.lstrip("/"))

            try:
                response = self._session.request(
                    method=method,
                    url=full_url,
                    params=params,
                    json=data,
                    files=files,
                    headers=_headers,
                    timeout=timeout or self._timeout,
                )

                response.raise_for_status()

                if raw_response:
                    return (
                        response.content,
                        response.status_code,
                        dict(response.headers),
                    )
                return response.json(), response.status_code, dict(response.headers)

            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.HTTPError):
                    # Extract error details from response
                    try:
                        error_data = e.response.json()
                        # First try to get error from error object
                        error_obj = error_data.get("error", {})
                        message = error_obj.get("message")
                        # If not found, try to get detail directly
                        if message is None:
                            message = error_data.get("detail", str(e))
                        error_type = error_obj.get("type")
                        request_id = error_obj.get("id")
                    except Exception:
                        message = str(e)
                        error_type = None
                        request_id = None

                    status_code = e.response.status_code
                    headers = dict(e.response.headers)

                    if status_code == 401:
                        raise AuthenticationError(
                            message=message,
                            http_status=status_code,
                            headers=headers,
                            request_id=request_id,
                            error_type=error_type,
                        ) from e
                    elif status_code == 400:
                        raise ValidationError(
                            message=message,
                            http_status=status_code,
                            headers=headers,
                            request_id=request_id,
                            error_type=error_type,
                        ) from e
                    elif status_code == 404:
                        raise ResourceNotFoundError(
                            message=message,
                            http_status=status_code,
                            headers=headers,
                            request_id=request_id,
                            error_type=error_type,
                        ) from e
                    elif status_code == 429:
                        raise RateLimitError(
                            message=message,
                            http_status=status_code,
                            headers=headers,
                            request_id=request_id,
                            error_type=error_type,
                        ) from e
                    elif 500 <= status_code < 600:
                        raise ServerError(
                            message=message,
                            http_status=status_code,
                            headers=headers,
                            request_id=request_id,
                            error_type=error_type,
                        ) from e
                    else:
                        raise APIError(
                            message=message,
                            http_status=status_code,
                            headers=headers,
                            request_id=request_id,
                            error_type=error_type,
                        ) from e
                elif isinstance(e, requests.exceptions.Timeout):
                    raise RequestTimeoutError(f"Request timed out: {str(e)}") from e
                elif isinstance(e, requests.exceptions.ConnectionError):
                    raise NetworkError(f"Connection error: {str(e)}") from e
                else:
                    raise APIError(str(e)) from e

        try:
            return _request_with_retry()
        except RetryError as e:
            return self._handle_retry_error(e)

    def _handle_retry_error(self, e: RetryError) -> None:
        """Handle RetryError by extracting and raising the appropriate exception.

        Args:
            e: The RetryError to handle

        Raises:
            AuthenticationError: If authentication fails
            ValidationError: If request validation fails
            RateLimitError: If rate limit is exceeded
            ResourceNotFoundError: If resource is not found
            ServerError: If server returns 5xx error
            RequestTimeoutError: If request times out
            NetworkError: If a network error occurs
            APIError: For other API errors
        """
        # Extract the last exception from the retry error
        last_exception = e.last_attempt.exception()

        # Preserve all our custom error types
        if isinstance(
            last_exception,
            (
                AuthenticationError,
                ValidationError,
                RateLimitError,
                ResourceNotFoundError,
                ServerError,
                RequestTimeoutError,
                NetworkError,
            ),
        ):
            raise last_exception
        else:
            raise APIError(
                f"Request failed after {self._max_retries} retries: {str(last_exception)}"
            ) from last_exception
