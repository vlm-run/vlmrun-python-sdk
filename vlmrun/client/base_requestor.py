"""VLM Run API requestor implementation."""

from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urljoin

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


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        http_status: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.http_status = http_status
        self.headers = headers or {}


class APIRequestor:
    """Handles API requests with retry logic."""

    def __init__(
        self,
        client,
        base_url: str = "https://api.vlm.run/v1",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize API requestor.

        Args:
            client: API client instance
            base_url: Base URL for API
            timeout: Request timeout in seconds
        """
        self._client = client
        self._base_url = base_url
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
    ) -> Tuple[Union[Dict[str, Any], bytes], int, Dict[str, str]]:
        """Make an API request with retry logic.

        Args:
            method: HTTP method
            url: API endpoint
            params: Query parameters
            data: Request body
            files: Files to upload
            headers: Request headers

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
        full_url = urljoin(self._base_url, url)

        try:
            response = self._session.request(
                method=method,
                url=full_url,
                params=params,
                json=data,
                files=files,
                headers=headers,
                timeout=self._timeout,
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
