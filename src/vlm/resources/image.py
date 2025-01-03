# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from datetime import datetime
from typing_extensions import Literal

import httpx

from ..types import image_generate_params
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

__all__ = ["ImageResource", "AsyncImageResource"]


class ImageResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> ImageResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return the
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#accessing-raw-response-data-eg-headers
        """
        return ImageResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ImageResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#with_streaming_response
        """
        return ImageResourceWithStreamingResponse(self)

    def generate(
        self,
        *,
        image: str,
        id: str | NotGiven = NOT_GIVEN,
        callback_url: Optional[str] | NotGiven = NOT_GIVEN,
        created_at: Union[str, datetime] | NotGiven = NOT_GIVEN,
        detail: Literal["auto", "hi", "lo"] | NotGiven = NOT_GIVEN,
        domain: Optional[
            Literal[
                "document.generative",
                "document.presentation",
                "document.invoice",
                "document.receipt",
                "document.markdown",
                "video.tv-news",
                "video.tv-intelligence",
            ]
        ]
        | NotGiven = NOT_GIVEN,
        json_schema: Optional[object] | NotGiven = NOT_GIVEN,
        metadata: image_generate_params.Metadata | NotGiven = NOT_GIVEN,
        model: Literal["vlm-1"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> PredictionResponse:
        """
        Generate structured prediction for the given image.

        Args:
          image: Base64 encoded image.

          id: Unique identifier of the request.

          callback_url: The URL to call when the request is completed.

          created_at: Date and time when the request was created (in UTC timezone)

          detail: The detail level to use for the model.

          domain: The domain identifier.

          json_schema: The JSON schema to use for the model.

          metadata: Optional metadata to pass to the model.

          model: The model to use for generating the response.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._post(
            "/v1/image/generate",
            body=maybe_transform(
                {
                    "image": image,
                    "id": id,
                    "callback_url": callback_url,
                    "created_at": created_at,
                    "detail": detail,
                    "domain": domain,
                    "json_schema": json_schema,
                    "metadata": metadata,
                    "model": model,
                },
                image_generate_params.ImageGenerateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=PredictionResponse,
        )


class AsyncImageResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncImageResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return the
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#accessing-raw-response-data-eg-headers
        """
        return AsyncImageResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncImageResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#with_streaming_response
        """
        return AsyncImageResourceWithStreamingResponse(self)

    async def generate(
        self,
        *,
        image: str,
        id: str | NotGiven = NOT_GIVEN,
        callback_url: Optional[str] | NotGiven = NOT_GIVEN,
        created_at: Union[str, datetime] | NotGiven = NOT_GIVEN,
        detail: Literal["auto", "hi", "lo"] | NotGiven = NOT_GIVEN,
        domain: Optional[
            Literal[
                "document.generative",
                "document.presentation",
                "document.invoice",
                "document.receipt",
                "document.markdown",
                "video.tv-news",
                "video.tv-intelligence",
            ]
        ]
        | NotGiven = NOT_GIVEN,
        json_schema: Optional[object] | NotGiven = NOT_GIVEN,
        metadata: image_generate_params.Metadata | NotGiven = NOT_GIVEN,
        model: Literal["vlm-1"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> PredictionResponse:
        """
        Generate structured prediction for the given image.

        Args:
          image: Base64 encoded image.

          id: Unique identifier of the request.

          callback_url: The URL to call when the request is completed.

          created_at: Date and time when the request was created (in UTC timezone)

          detail: The detail level to use for the model.

          domain: The domain identifier.

          json_schema: The JSON schema to use for the model.

          metadata: Optional metadata to pass to the model.

          model: The model to use for generating the response.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post(
            "/v1/image/generate",
            body=await async_maybe_transform(
                {
                    "image": image,
                    "id": id,
                    "callback_url": callback_url,
                    "created_at": created_at,
                    "detail": detail,
                    "domain": domain,
                    "json_schema": json_schema,
                    "metadata": metadata,
                    "model": model,
                },
                image_generate_params.ImageGenerateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=PredictionResponse,
        )


class ImageResourceWithRawResponse:
    def __init__(self, image: ImageResource) -> None:
        self._image = image

        self.generate = to_raw_response_wrapper(
            image.generate,
        )


class AsyncImageResourceWithRawResponse:
    def __init__(self, image: AsyncImageResource) -> None:
        self._image = image

        self.generate = async_to_raw_response_wrapper(
            image.generate,
        )


class ImageResourceWithStreamingResponse:
    def __init__(self, image: ImageResource) -> None:
        self._image = image

        self.generate = to_streamed_response_wrapper(
            image.generate,
        )


class AsyncImageResourceWithStreamingResponse:
    def __init__(self, image: AsyncImageResource) -> None:
        self._image = image

        self.generate = async_to_streamed_response_wrapper(
            image.generate,
        )
