"""VLM Run API requestor implementation."""

from typing import Any, Dict, Tuple, TYPE_CHECKING, Union, Optional
from urllib.parse import urljoin
from dataclasses import dataclass

if TYPE_CHECKING:
    from vlmrun.types.abstract import VLMRunProtocol

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Constants
DEFAULT_TIMEOUT = 30.0  # seconds
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 10  # seconds


@dataclass
class APIError(Exception):
    """Base exception for API errors."""

    message: str
    http_status: Optional[int] = None
    headers: Optional[Dict[str, str]] = None

    def __post_init__(self):
        super().__init__(self.message)
        if self.headers is None:
            self.headers = {}


class APIRequestor:
    """Handles API requests with retry logic."""

    def __init__(
        self,
        client: "VLMRunProtocol",
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize API requestor.

        Args:
            client: VLMRun API instance
            base_url: Base URL for API
            timeout: Request timeout in seconds
        """
        self._client = client
        self._base_url = base_url or client.base_url
        self._timeout = timeout
        self._session = requests.Session()

    @retry(
        retry=retry_if_exception_type(
            (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
        ),
        wait=wait_exponential(
            multiplier=INITIAL_RETRY_DELAY, min=INITIAL_RETRY_DELAY, max=MAX_RETRY_DELAY
        ),
        stop=stop_after_attempt(MAX_RETRIES),
    )
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
            APIError: If request fails
        """
        if not headers:
            headers = {}

        # Add authorization
        if self._client.api_key:
            headers["Authorization"] = f"Bearer {self._client.api_key}"

        # Build full URL
        full_url = urljoin(self._base_url.rstrip("/") + "/", url.lstrip("/"))

        try:
            response = self._session.request(
                method=method,
                url=full_url,
                params=params,
                json=data,
                files=files,
                headers=headers,
                timeout=timeout or self._timeout,
            )

            response.raise_for_status()

            if raw_response:
                return response.content, response.status_code, dict(response.headers)
            return response.json(), response.status_code, dict(response.headers)

        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError):
                # Try to get error details from response
                try:
                    error_data = e.response.json()
                    message = error_data.get("error", {}).get("message", str(e))
                except Exception:
                    message = str(e)

                raise APIError(
                    message=message,
                    http_status=e.response.status_code,
                    headers=dict(e.response.headers),
                ) from e

            raise APIError(str(e)) from e
