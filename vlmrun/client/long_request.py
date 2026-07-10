"""Helpers for Modal's long-request 303 See Other poll flow.

When an Orion chat-completions request exceeds the Modal gateway idle timeout
(~150s), the gateway returns ``303 See Other`` with a ``Location`` URL. The
job keeps running server-side; clients must poll ``Location`` until the result
is ready.

Poll semantics observed on the live API:
  - connection hold-then-timeout, ``202``, ``204``, ``303``, ``5xx``
    → keep waiting (job still running)
  - ``200`` / ``201`` with a non-empty body → result
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional
from urllib.parse import urljoin

import requests

# Status codes that mean "job still running — keep polling".
_PENDING_STATUS_CODES = frozenset({0, 202, 204, 303})


class LongRequestTimeoutError(TimeoutError):
    """Raised when polling a long-request Location URL times out."""


def is_pending_status(status_code: int) -> bool:
    """Return True if ``status_code`` means the long job is still running."""
    if status_code in _PENDING_STATUS_CODES:
        return True
    # Transient 5xx mid-job does not mean failure on Modal poll URLs.
    if 500 <= status_code < 600:
        return True
    return False


def is_result_status(status_code: int, body: bytes | str | None) -> bool:
    """Return True if the poll response is the finished result."""
    if status_code not in (200, 201):
        return False
    if body is None:
        return False
    if isinstance(body, bytes):
        return len(body) > 0
    return len(body) > 0


def poll_location(
    location: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 900.0,
    poll_interval: float = 2.0,
    request_timeout: float = 130.0,
    session: Optional[requests.Session] = None,
) -> tuple[bytes, int, dict[str, str]]:
    """Poll a Modal long-request ``Location`` URL until the result is ready.

    Args:
        location: Absolute or relative URL from the ``303`` ``Location`` header.
        headers: Optional request headers (Authorization, etc.).
        timeout: Maximum wall-clock seconds to wait for the result.
        poll_interval: Seconds to sleep between polls after a pending response.
        request_timeout: Per-request timeout. Modal may hold the connection open
            for up to ~2 minutes before returning; keep this above that.
        session: Optional ``requests.Session`` to reuse.

    Returns:
        Tuple of ``(body_bytes, status_code, response_headers)``.

    Raises:
        LongRequestTimeoutError: If ``timeout`` elapses before a result arrives.
        requests.HTTPError: For non-pending client errors.
    """
    http = session or requests.Session()
    own_session = session is None
    started = time.monotonic()
    _headers = dict(headers or {})

    try:
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                raise LongRequestTimeoutError(
                    f"Long request did not complete within {timeout:.0f}s "
                    f"(last Location={location!r})"
                )

            try:
                response = http.get(
                    location,
                    headers=_headers,
                    timeout=min(request_timeout, max(1.0, timeout - elapsed)),
                    allow_redirects=False,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                # Connection hold / network blip while the job is still running.
                time.sleep(poll_interval)
                continue

            status = response.status_code
            body = response.content
            resp_headers = dict(response.headers)

            if is_result_status(status, body):
                return body, status, resp_headers

            if status in (301, 302, 303, 307, 308):
                next_location = resp_headers.get("Location") or resp_headers.get(
                    "location"
                )
                if next_location:
                    location = urljoin(location, next_location)
                time.sleep(poll_interval)
                continue

            if is_pending_status(status):
                time.sleep(poll_interval)
                continue

            response.raise_for_status()
            time.sleep(poll_interval)
    finally:
        if own_session:
            http.close()


def extract_location(headers: Mapping[str, Any] | None) -> Optional[str]:
    """Extract a ``Location`` header value (case-insensitive)."""
    if not headers:
        return None
    for key, value in headers.items():
        if str(key).lower() == "location" and value:
            return str(value)
    return None
