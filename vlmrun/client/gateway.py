"""VLM Run OpenAI-compatible model gateway resource.

The gateway (``https://gateway.vlm.run/v1``) exposes an OpenAI-compatible
surface for third-party OCR / vision-language models (e.g. ``glm-ocr``,
``paddle-ocrv6``, ``qwen3.6-0.8b``). It authenticates with the same
``VLMRUN_API_KEY`` used everywhere else in the SDK.

This mirrors the :class:`~vlmrun.client.agent.Agent` completions pattern:
we point the OpenAI SDK at ``{gateway_url}/openai`` and reuse the familiar
chat-completions / models interface.
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Any, List, Optional

from vlmrun.constants import DEFAULT_GATEWAY_URL
from vlmrun.client.exceptions import DependencyError
from vlmrun.types.abstract import VLMRunProtocol


def _require_openai():
    """Import the OpenAI SDK or raise a helpful :class:`DependencyError`."""
    try:
        import openai  # noqa: F401
    except ImportError as e:
        raise DependencyError(
            message="OpenAI SDK is not installed",
            suggestion="Install it with `pip install vlmrun[openai]` or `pip install openai`",
            error_type="missing_dependency",
        ) from e
    return openai


class Gateway:
    """OpenAI-compatible model gateway resource for VLM Run.

    Provides access to third-party OCR / VLM models hosted behind the VLM Run
    gateway using the standard OpenAI chat-completions and models interfaces.

    Attributes:
        base_url: Gateway base URL (defaults to ``VLMRUN_GATEWAY_URL`` env var or
            ``https://gateway.vlm.run/v1``).
    """

    def __init__(
        self, client: "VLMRunProtocol", base_url: Optional[str] = None
    ) -> None:
        """Initialize the Gateway resource.

        Args:
            client: VLM Run API client instance (provides the API key).
            base_url: Optional gateway base URL override. Falls back to the
                ``VLMRUN_GATEWAY_URL`` environment variable, then the default.
        """
        self._client = client
        self._base_url = (
            base_url or os.getenv("VLMRUN_GATEWAY_URL") or DEFAULT_GATEWAY_URL
        )

    @property
    def base_url(self) -> str:
        """Gateway base URL (without trailing slash)."""
        return self._base_url.rstrip("/")

    @property
    def openai_base_url(self) -> str:
        """OpenAI-compatible base URL used by the OpenAI SDK."""
        return f"{self.base_url}/openai"

    def _timeout(self) -> Optional[float]:
        timeout = self._client.timeout
        return timeout if timeout is None else max(timeout, 600)

    @cached_property
    def _openai(self):
        """Synchronous OpenAI client pointed at the gateway."""
        openai = _require_openai()
        return openai.OpenAI(
            api_key=self._client.api_key,
            base_url=self.openai_base_url,
            timeout=self._timeout(),
            max_retries=self._client.max_retries,
        )

    @cached_property
    def _async_openai(self):
        """Asynchronous OpenAI client pointed at the gateway."""
        openai = _require_openai()
        return openai.AsyncOpenAI(
            api_key=self._client.api_key,
            base_url=self.openai_base_url,
            timeout=self._timeout(),
            max_retries=self._client.max_retries,
        )

    @cached_property
    def completions(self):
        """OpenAI-compatible chat completions interface (synchronous).

        Example:
            ```python
            from vlmrun import VLMRun

            client = VLMRun()
            response = client.gateway.completions.create(
                model="glm-ocr",
                messages=[{"role": "user", "content": [
                    {"type": "document_url", "document_url": {"url": "data:application/pdf;base64,..."}},
                ]}],
            )
            ```

        Raises:
            DependencyError: If the ``openai`` package is not installed.

        Returns:
            OpenAI Completions object configured for the VLM Run gateway.
        """
        return self._openai.chat.completions

    @cached_property
    def async_completions(self):
        """OpenAI-compatible chat completions interface (asynchronous).

        Raises:
            DependencyError: If the ``openai`` package is not installed.

        Returns:
            OpenAI AsyncCompletions object configured for the VLM Run gateway.
        """
        return self._async_openai.chat.completions

    def models(self) -> List[Any]:
        """List models available on the gateway.

        Returns the raw OpenAI ``Model`` objects. Gateway models carry extra
        metadata (input/output pricing, modality support, etc.) beyond the
        standard OpenAI fields; those are preserved on each object's
        ``model_extra``.

        Raises:
            DependencyError: If the ``openai`` package is not installed.

        Returns:
            List of OpenAI ``Model`` objects.
        """
        return list(self._openai.models.list())

    def health(self) -> bool:
        """Check gateway liveness.

        Attempts a ``GET {gateway}/health`` request and falls back to listing
        models as a liveness probe if no dedicated health endpoint responds.

        Returns:
            True if the gateway is reachable and authenticated, else False.
        """
        # httpx is a hard dependency of the openai SDK, so it is always
        # available whenever the gateway is usable.
        import httpx

        headers = {"Authorization": f"Bearer {self._client.api_key}"}
        try:
            resp = httpx.get(f"{self.base_url}/health", headers=headers, timeout=30.0)
        except Exception:
            # No dedicated health route reachable — fall back to a real call.
            try:
                self.models()
                return True
            except Exception:
                return False

        if resp.status_code == 404:
            try:
                self.models()
                return True
            except Exception:
                return False
        return resp.is_success
