# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import httpx

from .image import (
    ImageResource,
    AsyncImageResource,
    ImageResourceWithRawResponse,
    AsyncImageResourceWithRawResponse,
    ImageResourceWithStreamingResponse,
    AsyncImageResourceWithStreamingResponse,
)
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .document import (
    DocumentResource,
    AsyncDocumentResource,
    DocumentResourceWithRawResponse,
    AsyncDocumentResourceWithRawResponse,
    DocumentResourceWithStreamingResponse,
    AsyncDocumentResourceWithStreamingResponse,
)
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from .image.image import ImageResource, AsyncImageResource
from ..._base_client import make_request_options
from .document.document import DocumentResource, AsyncDocumentResource

__all__ = ["ExperimentalResource", "AsyncExperimentalResource"]


class ExperimentalResource(SyncAPIResource):
    @cached_property
    def image(self) -> ImageResource:
        return ImageResource(self._client)

    @cached_property
    def document(self) -> DocumentResource:
        return DocumentResource(self._client)

    @cached_property
    def with_raw_response(self) -> ExperimentalResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return the
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/vlm-python#accessing-raw-response-data-eg-headers
        """
        return ExperimentalResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ExperimentalResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/vlm-python#with_streaming_response
        """
        return ExperimentalResourceWithStreamingResponse(self)

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
        """Health"""
        return self._get(
            "/v1/experimental/health",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=object,
        )


class AsyncExperimentalResource(AsyncAPIResource):
    @cached_property
    def image(self) -> AsyncImageResource:
        return AsyncImageResource(self._client)

    @cached_property
    def document(self) -> AsyncDocumentResource:
        return AsyncDocumentResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncExperimentalResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return the
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/vlm-python#accessing-raw-response-data-eg-headers
        """
        return AsyncExperimentalResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncExperimentalResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/vlm-python#with_streaming_response
        """
        return AsyncExperimentalResourceWithStreamingResponse(self)

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
        """Health"""
        return await self._get(
            "/v1/experimental/health",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=object,
        )


class ExperimentalResourceWithRawResponse:
    def __init__(self, experimental: ExperimentalResource) -> None:
        self._experimental = experimental

        self.health = to_raw_response_wrapper(
            experimental.health,
        )

    @cached_property
    def image(self) -> ImageResourceWithRawResponse:
        return ImageResourceWithRawResponse(self._experimental.image)

    @cached_property
    def document(self) -> DocumentResourceWithRawResponse:
        return DocumentResourceWithRawResponse(self._experimental.document)


class AsyncExperimentalResourceWithRawResponse:
    def __init__(self, experimental: AsyncExperimentalResource) -> None:
        self._experimental = experimental

        self.health = async_to_raw_response_wrapper(
            experimental.health,
        )

    @cached_property
    def image(self) -> AsyncImageResourceWithRawResponse:
        return AsyncImageResourceWithRawResponse(self._experimental.image)

    @cached_property
    def document(self) -> AsyncDocumentResourceWithRawResponse:
        return AsyncDocumentResourceWithRawResponse(self._experimental.document)


class ExperimentalResourceWithStreamingResponse:
    def __init__(self, experimental: ExperimentalResource) -> None:
        self._experimental = experimental

        self.health = to_streamed_response_wrapper(
            experimental.health,
        )

    @cached_property
    def image(self) -> ImageResourceWithStreamingResponse:
        return ImageResourceWithStreamingResponse(self._experimental.image)

    @cached_property
    def document(self) -> DocumentResourceWithStreamingResponse:
        return DocumentResourceWithStreamingResponse(self._experimental.document)


class AsyncExperimentalResourceWithStreamingResponse:
    def __init__(self, experimental: AsyncExperimentalResource) -> None:
        self._experimental = experimental

        self.health = async_to_streamed_response_wrapper(
            experimental.health,
        )

    @cached_property
    def image(self) -> AsyncImageResourceWithStreamingResponse:
        return AsyncImageResourceWithStreamingResponse(self._experimental.image)

    @cached_property
    def document(self) -> AsyncDocumentResourceWithStreamingResponse:
        return AsyncDocumentResourceWithStreamingResponse(self._experimental.document)
