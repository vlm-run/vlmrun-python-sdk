# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import httpx

from .models import (
    ModelsResource,
    AsyncModelsResource,
    ModelsResourceWithRawResponse,
    AsyncModelsResourceWithRawResponse,
    ModelsResourceWithStreamingResponse,
    AsyncModelsResourceWithStreamingResponse,
)
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ..._base_client import make_request_options
from .chat_completions import (
    ChatCompletionsResource,
    AsyncChatCompletionsResource,
    ChatCompletionsResourceWithRawResponse,
    AsyncChatCompletionsResourceWithRawResponse,
    ChatCompletionsResourceWithStreamingResponse,
    AsyncChatCompletionsResourceWithStreamingResponse,
)

__all__ = ["OpenAIResource", "AsyncOpenAIResource"]


class OpenAIResource(SyncAPIResource):
    @cached_property
    def chat_completions(self) -> ChatCompletionsResource:
        return ChatCompletionsResource(self._client)

    @cached_property
    def models(self) -> ModelsResource:
        return ModelsResource(self._client)

    @cached_property
    def with_raw_response(self) -> OpenAIResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return the
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/vlm-python#accessing-raw-response-data-eg-headers
        """
        return OpenAIResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> OpenAIResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/vlm-python#with_streaming_response
        """
        return OpenAIResourceWithStreamingResponse(self)

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
            "/v1/openai/health",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=object,
        )


class AsyncOpenAIResource(AsyncAPIResource):
    @cached_property
    def chat_completions(self) -> AsyncChatCompletionsResource:
        return AsyncChatCompletionsResource(self._client)

    @cached_property
    def models(self) -> AsyncModelsResource:
        return AsyncModelsResource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncOpenAIResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return the
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/vlm-python#accessing-raw-response-data-eg-headers
        """
        return AsyncOpenAIResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncOpenAIResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/vlm-python#with_streaming_response
        """
        return AsyncOpenAIResourceWithStreamingResponse(self)

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
            "/v1/openai/health",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=object,
        )


class OpenAIResourceWithRawResponse:
    def __init__(self, openai: OpenAIResource) -> None:
        self._openai = openai

        self.health = to_raw_response_wrapper(
            openai.health,
        )

    @cached_property
    def chat_completions(self) -> ChatCompletionsResourceWithRawResponse:
        return ChatCompletionsResourceWithRawResponse(self._openai.chat_completions)

    @cached_property
    def models(self) -> ModelsResourceWithRawResponse:
        return ModelsResourceWithRawResponse(self._openai.models)


class AsyncOpenAIResourceWithRawResponse:
    def __init__(self, openai: AsyncOpenAIResource) -> None:
        self._openai = openai

        self.health = async_to_raw_response_wrapper(
            openai.health,
        )

    @cached_property
    def chat_completions(self) -> AsyncChatCompletionsResourceWithRawResponse:
        return AsyncChatCompletionsResourceWithRawResponse(self._openai.chat_completions)

    @cached_property
    def models(self) -> AsyncModelsResourceWithRawResponse:
        return AsyncModelsResourceWithRawResponse(self._openai.models)


class OpenAIResourceWithStreamingResponse:
    def __init__(self, openai: OpenAIResource) -> None:
        self._openai = openai

        self.health = to_streamed_response_wrapper(
            openai.health,
        )

    @cached_property
    def chat_completions(self) -> ChatCompletionsResourceWithStreamingResponse:
        return ChatCompletionsResourceWithStreamingResponse(self._openai.chat_completions)

    @cached_property
    def models(self) -> ModelsResourceWithStreamingResponse:
        return ModelsResourceWithStreamingResponse(self._openai.models)


class AsyncOpenAIResourceWithStreamingResponse:
    def __init__(self, openai: AsyncOpenAIResource) -> None:
        self._openai = openai

        self.health = async_to_streamed_response_wrapper(
            openai.health,
        )

    @cached_property
    def chat_completions(self) -> AsyncChatCompletionsResourceWithStreamingResponse:
        return AsyncChatCompletionsResourceWithStreamingResponse(self._openai.chat_completions)

    @cached_property
    def models(self) -> AsyncModelsResourceWithStreamingResponse:
        return AsyncModelsResourceWithStreamingResponse(self._openai.models)
