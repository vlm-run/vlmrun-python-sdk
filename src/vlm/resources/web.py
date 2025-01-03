# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from datetime import datetime
from typing_extensions import Literal

import httpx

from ..types import web_generate_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import (
    maybe_transform,
    async_maybe_transform,
)
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from .._base_client import make_request_options
from ..types.shared.prediction_response import PredictionResponse

__all__ = ["WebResource", "AsyncWebResource"]


class WebResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> WebResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return the
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#accessing-raw-response-data-eg-headers
        """
        return WebResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> WebResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#with_streaming_response
        """
        return WebResourceWithStreamingResponse(self)

    def generate(
        self,
        *,
        url: str,
        id: str | NotGiven = NOT_GIVEN,
        callback_url: Optional[str] | NotGiven = NOT_GIVEN,
        created_at: Union[str, datetime] | NotGiven = NOT_GIVEN,
        domain: Optional[Literal["web.ecommerce-product-catalog", "web.github-developer-stats", "web.market-research"]]
        | NotGiven = NOT_GIVEN,
        metadata: web_generate_params.Metadata | NotGiven = NOT_GIVEN,
        mode: Literal["fast", "accurate"] | NotGiven = NOT_GIVEN,
        model: Literal["vlm-1"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> PredictionResponse:
        """
        Generate structured prediction for the given url.

        Args:
          url: The URL of the web page.

          id: Unique identifier of the request.

          callback_url: The URL to call when the request is completed.

          created_at: Date and time when the request was created (in UTC timezone)

          domain: The domain identifier.

          metadata: Optional metadata to pass to the model.

          mode: The mode to use for the model.

          model: The model to use for generating the response.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._post(
            "/v1/web/generate",
            body=maybe_transform(
                {
                    "url": url,
                    "id": id,
                    "callback_url": callback_url,
                    "created_at": created_at,
                    "domain": domain,
                    "metadata": metadata,
                    "mode": mode,
                    "model": model,
                },
                web_generate_params.WebGenerateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=PredictionResponse,
        )


class AsyncWebResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncWebResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return the
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#accessing-raw-response-data-eg-headers
        """
        return AsyncWebResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncWebResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#with_streaming_response
        """
        return AsyncWebResourceWithStreamingResponse(self)

    async def generate(
        self,
        *,
        url: str,
        id: str | NotGiven = NOT_GIVEN,
        callback_url: Optional[str] | NotGiven = NOT_GIVEN,
        created_at: Union[str, datetime] | NotGiven = NOT_GIVEN,
        domain: Optional[Literal["web.ecommerce-product-catalog", "web.github-developer-stats", "web.market-research"]]
        | NotGiven = NOT_GIVEN,
        metadata: web_generate_params.Metadata | NotGiven = NOT_GIVEN,
        mode: Literal["fast", "accurate"] | NotGiven = NOT_GIVEN,
        model: Literal["vlm-1"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> PredictionResponse:
        """
        Generate structured prediction for the given url.

        Args:
          url: The URL of the web page.

          id: Unique identifier of the request.

          callback_url: The URL to call when the request is completed.

          created_at: Date and time when the request was created (in UTC timezone)

          domain: The domain identifier.

          metadata: Optional metadata to pass to the model.

          mode: The mode to use for the model.

          model: The model to use for generating the response.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post(
            "/v1/web/generate",
            body=await async_maybe_transform(
                {
                    "url": url,
                    "id": id,
                    "callback_url": callback_url,
                    "created_at": created_at,
                    "domain": domain,
                    "metadata": metadata,
                    "mode": mode,
                    "model": model,
                },
                web_generate_params.WebGenerateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=PredictionResponse,
        )


class WebResourceWithRawResponse:
    def __init__(self, web: WebResource) -> None:
        self._web = web

        self.generate = to_raw_response_wrapper(
            web.generate,
        )


class AsyncWebResourceWithRawResponse:
    def __init__(self, web: AsyncWebResource) -> None:
        self._web = web

        self.generate = async_to_raw_response_wrapper(
            web.generate,
        )


class WebResourceWithStreamingResponse:
    def __init__(self, web: WebResource) -> None:
        self._web = web

        self.generate = to_streamed_response_wrapper(
            web.generate,
        )


class AsyncWebResourceWithStreamingResponse:
    def __init__(self, web: AsyncWebResource) -> None:
        self._web = web

        self.generate = async_to_streamed_response_wrapper(
            web.generate,
        )
