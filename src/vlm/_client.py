# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, Union, Mapping
from typing_extensions import Self, override

import httpx

from . import resources, _exceptions
from ._qs import Querystring
from ._types import (
    NOT_GIVEN,
    Body,
    Omit,
    Query,
    Headers,
    Timeout,
    NotGiven,
    Transport,
    ProxiesTypes,
    RequestOptions,
)
from ._utils import (
    is_given,
    get_async_library,
)
from ._version import __version__
from ._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ._streaming import Stream as Stream, AsyncStream as AsyncStream
from ._exceptions import VlmError, APIStatusError
from ._base_client import (
    DEFAULT_MAX_RETRIES,
    SyncAPIClient,
    AsyncAPIClient,
    make_request_options,
)

__all__ = [
    "Timeout",
    "Transport",
    "ProxiesTypes",
    "RequestOptions",
    "resources",
    "Vlm",
    "AsyncVlm",
    "Client",
    "AsyncClient",
]


class Vlm(SyncAPIClient):
    openai: resources.OpenAIResource
    experimental: resources.ExperimentalResource
    models: resources.ModelsResource
    files: resources.FilesResource
    response: resources.ResponseResource
    document: resources.DocumentResource
    audio: resources.AudioResource
    image: resources.ImageResource
    web: resources.WebResource
    schema: resources.SchemaResource
    with_raw_response: VlmWithRawResponse
    with_streaming_response: VlmWithStreamedResponse

    # client options
    bearer_token: str

    def __init__(
        self,
        *,
        bearer_token: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        # Configure a custom httpx client.
        # We provide a `DefaultHttpxClient` class that you can pass to retain the default values we use for `limits`, `timeout` & `follow_redirects`.
        # See the [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
        http_client: httpx.Client | None = None,
        # Enable or disable schema validation for data returned by the API.
        # When enabled an error APIResponseValidationError is raised
        # if the API responds with invalid data for the expected schema.
        #
        # This parameter may be removed or changed in the future.
        # If you rely on this feature, please open a GitHub issue
        # outlining your use-case to help us decide if it should be
        # part of our public interface in the future.
        _strict_response_validation: bool = False,
    ) -> None:
        """Construct a new synchronous vlm client instance.

        This automatically infers the `bearer_token` argument from the `BEARER_TOKEN` environment variable if it is not provided.
        """
        if bearer_token is None:
            bearer_token = os.environ.get("BEARER_TOKEN")
        if bearer_token is None:
            raise VlmError(
                "The bearer_token client option must be set either by passing bearer_token to the client or by setting the BEARER_TOKEN environment variable"
            )
        self.bearer_token = bearer_token

        if base_url is None:
            base_url = os.environ.get("VLM_BASE_URL")
        if base_url is None:
            base_url = f"https://api.vlm.run"

        super().__init__(
            version=__version__,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            custom_headers=default_headers,
            custom_query=default_query,
            _strict_response_validation=_strict_response_validation,
        )

        self.openai = resources.OpenAIResource(self)
        self.experimental = resources.ExperimentalResource(self)
        self.models = resources.ModelsResource(self)
        self.files = resources.FilesResource(self)
        self.response = resources.ResponseResource(self)
        self.document = resources.DocumentResource(self)
        self.audio = resources.AudioResource(self)
        self.image = resources.ImageResource(self)
        self.web = resources.WebResource(self)
        self.schema = resources.SchemaResource(self)
        self.with_raw_response = VlmWithRawResponse(self)
        self.with_streaming_response = VlmWithStreamedResponse(self)

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="comma")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        bearer_token = self.bearer_token
        return {"Authorization": f"Bearer {bearer_token}"}

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            "X-Stainless-Async": "false",
            **self._custom_headers,
        }

    def copy(
        self,
        *,
        bearer_token: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
        http_client: httpx.Client | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._client
        return self.__class__(
            bearer_token=bearer_token or self.bearer_token,
            base_url=base_url or self.base_url,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            http_client=http_client,
            max_retries=max_retries if is_given(max_retries) else self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    def health(
        self,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> object:
        """Health check endpoint."""
        return self.get(
            "/v1/health",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=object,
        )

    @override
    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=body)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(err_msg, response=response, body=body)

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(err_msg, response=response, body=body)

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=body)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=body)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(err_msg, response=response, body=body)

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=body)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(err_msg, response=response, body=body)
        return APIStatusError(err_msg, response=response, body=body)


class AsyncVlm(AsyncAPIClient):
    openai: resources.AsyncOpenAIResource
    experimental: resources.AsyncExperimentalResource
    models: resources.AsyncModelsResource
    files: resources.AsyncFilesResource
    response: resources.AsyncResponseResource
    document: resources.AsyncDocumentResource
    audio: resources.AsyncAudioResource
    image: resources.AsyncImageResource
    web: resources.AsyncWebResource
    schema: resources.AsyncSchemaResource
    with_raw_response: AsyncVlmWithRawResponse
    with_streaming_response: AsyncVlmWithStreamedResponse

    # client options
    bearer_token: str

    def __init__(
        self,
        *,
        bearer_token: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        # Configure a custom httpx client.
        # We provide a `DefaultAsyncHttpxClient` class that you can pass to retain the default values we use for `limits`, `timeout` & `follow_redirects`.
        # See the [httpx documentation](https://www.python-httpx.org/api/#asyncclient) for more details.
        http_client: httpx.AsyncClient | None = None,
        # Enable or disable schema validation for data returned by the API.
        # When enabled an error APIResponseValidationError is raised
        # if the API responds with invalid data for the expected schema.
        #
        # This parameter may be removed or changed in the future.
        # If you rely on this feature, please open a GitHub issue
        # outlining your use-case to help us decide if it should be
        # part of our public interface in the future.
        _strict_response_validation: bool = False,
    ) -> None:
        """Construct a new async vlm client instance.

        This automatically infers the `bearer_token` argument from the `BEARER_TOKEN` environment variable if it is not provided.
        """
        if bearer_token is None:
            bearer_token = os.environ.get("BEARER_TOKEN")
        if bearer_token is None:
            raise VlmError(
                "The bearer_token client option must be set either by passing bearer_token to the client or by setting the BEARER_TOKEN environment variable"
            )
        self.bearer_token = bearer_token

        if base_url is None:
            base_url = os.environ.get("VLM_BASE_URL")
        if base_url is None:
            base_url = f"https://api.vlm.run"

        super().__init__(
            version=__version__,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            custom_headers=default_headers,
            custom_query=default_query,
            _strict_response_validation=_strict_response_validation,
        )

        self.openai = resources.AsyncOpenAIResource(self)
        self.experimental = resources.AsyncExperimentalResource(self)
        self.models = resources.AsyncModelsResource(self)
        self.files = resources.AsyncFilesResource(self)
        self.response = resources.AsyncResponseResource(self)
        self.document = resources.AsyncDocumentResource(self)
        self.audio = resources.AsyncAudioResource(self)
        self.image = resources.AsyncImageResource(self)
        self.web = resources.AsyncWebResource(self)
        self.schema = resources.AsyncSchemaResource(self)
        self.with_raw_response = AsyncVlmWithRawResponse(self)
        self.with_streaming_response = AsyncVlmWithStreamedResponse(self)

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="comma")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        bearer_token = self.bearer_token
        return {"Authorization": f"Bearer {bearer_token}"}

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            "X-Stainless-Async": f"async:{get_async_library()}",
            **self._custom_headers,
        }

    def copy(
        self,
        *,
        bearer_token: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._client
        return self.__class__(
            bearer_token=bearer_token or self.bearer_token,
            base_url=base_url or self.base_url,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            http_client=http_client,
            max_retries=max_retries if is_given(max_retries) else self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    async def health(
        self,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> object:
        """Health check endpoint."""
        return await self.get(
            "/v1/health",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=object,
        )

    @override
    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=body)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(err_msg, response=response, body=body)

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(err_msg, response=response, body=body)

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=body)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=body)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(err_msg, response=response, body=body)

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=body)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(err_msg, response=response, body=body)
        return APIStatusError(err_msg, response=response, body=body)


class VlmWithRawResponse:
    def __init__(self, client: Vlm) -> None:
        self.openai = resources.OpenAIResourceWithRawResponse(client.openai)
        self.experimental = resources.ExperimentalResourceWithRawResponse(client.experimental)
        self.models = resources.ModelsResourceWithRawResponse(client.models)
        self.files = resources.FilesResourceWithRawResponse(client.files)
        self.response = resources.ResponseResourceWithRawResponse(client.response)
        self.document = resources.DocumentResourceWithRawResponse(client.document)
        self.audio = resources.AudioResourceWithRawResponse(client.audio)
        self.image = resources.ImageResourceWithRawResponse(client.image)
        self.web = resources.WebResourceWithRawResponse(client.web)
        self.schema = resources.SchemaResourceWithRawResponse(client.schema)

        self.health = to_raw_response_wrapper(
            client.health,
        )


class AsyncVlmWithRawResponse:
    def __init__(self, client: AsyncVlm) -> None:
        self.openai = resources.AsyncOpenAIResourceWithRawResponse(client.openai)
        self.experimental = resources.AsyncExperimentalResourceWithRawResponse(client.experimental)
        self.models = resources.AsyncModelsResourceWithRawResponse(client.models)
        self.files = resources.AsyncFilesResourceWithRawResponse(client.files)
        self.response = resources.AsyncResponseResourceWithRawResponse(client.response)
        self.document = resources.AsyncDocumentResourceWithRawResponse(client.document)
        self.audio = resources.AsyncAudioResourceWithRawResponse(client.audio)
        self.image = resources.AsyncImageResourceWithRawResponse(client.image)
        self.web = resources.AsyncWebResourceWithRawResponse(client.web)
        self.schema = resources.AsyncSchemaResourceWithRawResponse(client.schema)

        self.health = async_to_raw_response_wrapper(
            client.health,
        )


class VlmWithStreamedResponse:
    def __init__(self, client: Vlm) -> None:
        self.openai = resources.OpenAIResourceWithStreamingResponse(client.openai)
        self.experimental = resources.ExperimentalResourceWithStreamingResponse(client.experimental)
        self.models = resources.ModelsResourceWithStreamingResponse(client.models)
        self.files = resources.FilesResourceWithStreamingResponse(client.files)
        self.response = resources.ResponseResourceWithStreamingResponse(client.response)
        self.document = resources.DocumentResourceWithStreamingResponse(client.document)
        self.audio = resources.AudioResourceWithStreamingResponse(client.audio)
        self.image = resources.ImageResourceWithStreamingResponse(client.image)
        self.web = resources.WebResourceWithStreamingResponse(client.web)
        self.schema = resources.SchemaResourceWithStreamingResponse(client.schema)

        self.health = to_streamed_response_wrapper(
            client.health,
        )


class AsyncVlmWithStreamedResponse:
    def __init__(self, client: AsyncVlm) -> None:
        self.openai = resources.AsyncOpenAIResourceWithStreamingResponse(client.openai)
        self.experimental = resources.AsyncExperimentalResourceWithStreamingResponse(client.experimental)
        self.models = resources.AsyncModelsResourceWithStreamingResponse(client.models)
        self.files = resources.AsyncFilesResourceWithStreamingResponse(client.files)
        self.response = resources.AsyncResponseResourceWithStreamingResponse(client.response)
        self.document = resources.AsyncDocumentResourceWithStreamingResponse(client.document)
        self.audio = resources.AsyncAudioResourceWithStreamingResponse(client.audio)
        self.image = resources.AsyncImageResourceWithStreamingResponse(client.image)
        self.web = resources.AsyncWebResourceWithStreamingResponse(client.web)
        self.schema = resources.AsyncSchemaResourceWithStreamingResponse(client.schema)

        self.health = async_to_streamed_response_wrapper(
            client.health,
        )


Client = Vlm

AsyncClient = AsyncVlm
