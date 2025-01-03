# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from datetime import datetime
from typing_extensions import Literal

import httpx

from ..types import document_generate_params
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

__all__ = ["DocumentResource", "AsyncDocumentResource"]


class DocumentResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> DocumentResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return the
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#accessing-raw-response-data-eg-headers
        """
        return DocumentResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> DocumentResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#with_streaming_response
        """
        return DocumentResourceWithStreamingResponse(self)

    def generate(
        self,
        *,
        id: str | NotGiven = NOT_GIVEN,
        batch: bool | NotGiven = NOT_GIVEN,
        callback_url: Optional[str] | NotGiven = NOT_GIVEN,
        created_at: Union[str, datetime] | NotGiven = NOT_GIVEN,
        detail: Literal["auto", "hi", "lo"] | NotGiven = NOT_GIVEN,
        domain: Union[
            Literal[
                "document.file",
                "document.pdf",
                "document.generative",
                "document.markdown",
                "document.presentation",
                "document.invoice",
                "document.receipt",
                "document.resume",
                "document.utility-bill",
            ],
            str,
        ]
        | NotGiven = NOT_GIVEN,
        file_id: Optional[str] | NotGiven = NOT_GIVEN,
        json_schema: Optional[object] | NotGiven = NOT_GIVEN,
        metadata: Optional[document_generate_params.Metadata] | NotGiven = NOT_GIVEN,
        model: Literal["vlm-1"] | NotGiven = NOT_GIVEN,
        url: Optional[str] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> PredictionResponse:
        """
        Generate structured prediction for the given document.

        Args:
          id: Unique identifier of the request.

          batch: Whether to process the document in batch mode (async).

          callback_url: The URL to call when the request is completed.

          created_at: Date and time when the request was created (in UTC timezone)

          detail: The detail level to use for the model.

          domain: The domain identifier for the model.

          file_id: The ID of the uploaded file (provide either `file_id` or `url`).

          json_schema: The schema to use for the model.

          metadata: Metadata for the request.

          model: The model to use for generating the response.

          url: The URL of the file (provide either `file_id` or `url`).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._post(
            "/v1/document/generate",
            body=maybe_transform(
                {
                    "id": id,
                    "batch": batch,
                    "callback_url": callback_url,
                    "created_at": created_at,
                    "detail": detail,
                    "domain": domain,
                    "file_id": file_id,
                    "json_schema": json_schema,
                    "metadata": metadata,
                    "model": model,
                    "url": url,
                },
                document_generate_params.DocumentGenerateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=PredictionResponse,
        )


class AsyncDocumentResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncDocumentResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return the
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#accessing-raw-response-data-eg-headers
        """
        return AsyncDocumentResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncDocumentResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/vlm-run/vlmrun-python-sdk#with_streaming_response
        """
        return AsyncDocumentResourceWithStreamingResponse(self)

    async def generate(
        self,
        *,
        id: str | NotGiven = NOT_GIVEN,
        batch: bool | NotGiven = NOT_GIVEN,
        callback_url: Optional[str] | NotGiven = NOT_GIVEN,
        created_at: Union[str, datetime] | NotGiven = NOT_GIVEN,
        detail: Literal["auto", "hi", "lo"] | NotGiven = NOT_GIVEN,
        domain: Union[
            Literal[
                "document.file",
                "document.pdf",
                "document.generative",
                "document.markdown",
                "document.presentation",
                "document.invoice",
                "document.receipt",
                "document.resume",
                "document.utility-bill",
            ],
            str,
        ]
        | NotGiven = NOT_GIVEN,
        file_id: Optional[str] | NotGiven = NOT_GIVEN,
        json_schema: Optional[object] | NotGiven = NOT_GIVEN,
        metadata: Optional[document_generate_params.Metadata] | NotGiven = NOT_GIVEN,
        model: Literal["vlm-1"] | NotGiven = NOT_GIVEN,
        url: Optional[str] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> PredictionResponse:
        """
        Generate structured prediction for the given document.

        Args:
          id: Unique identifier of the request.

          batch: Whether to process the document in batch mode (async).

          callback_url: The URL to call when the request is completed.

          created_at: Date and time when the request was created (in UTC timezone)

          detail: The detail level to use for the model.

          domain: The domain identifier for the model.

          file_id: The ID of the uploaded file (provide either `file_id` or `url`).

          json_schema: The schema to use for the model.

          metadata: Metadata for the request.

          model: The model to use for generating the response.

          url: The URL of the file (provide either `file_id` or `url`).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post(
            "/v1/document/generate",
            body=await async_maybe_transform(
                {
                    "id": id,
                    "batch": batch,
                    "callback_url": callback_url,
                    "created_at": created_at,
                    "detail": detail,
                    "domain": domain,
                    "file_id": file_id,
                    "json_schema": json_schema,
                    "metadata": metadata,
                    "model": model,
                    "url": url,
                },
                document_generate_params.DocumentGenerateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=PredictionResponse,
        )


class DocumentResourceWithRawResponse:
    def __init__(self, document: DocumentResource) -> None:
        self._document = document

        self.generate = to_raw_response_wrapper(
            document.generate,
        )


class AsyncDocumentResourceWithRawResponse:
    def __init__(self, document: AsyncDocumentResource) -> None:
        self._document = document

        self.generate = async_to_raw_response_wrapper(
            document.generate,
        )


class DocumentResourceWithStreamingResponse:
    def __init__(self, document: DocumentResource) -> None:
        self._document = document

        self.generate = to_streamed_response_wrapper(
            document.generate,
        )


class AsyncDocumentResourceWithStreamingResponse:
    def __init__(self, document: AsyncDocumentResource) -> None:
        self._document = document

        self.generate = async_to_streamed_response_wrapper(
            document.generate,
        )
