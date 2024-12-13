# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from vlm import Vlm, AsyncVlm
from tests.utils import assert_matches_type
from vlm.types.openai import Completion

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestChatCompletions:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Vlm) -> None:
        chat_completion = client.openai.chat_completions.create(
            messages=[{}],
        )
        assert_matches_type(Completion, chat_completion, path=["response"])

    @parametrize
    def test_method_create_with_all_params(self, client: Vlm) -> None:
        chat_completion = client.openai.chat_completions.create(
            messages=[
                {
                    "content": "string",
                    "role": "user",
                }
            ],
            id="id",
            domain="domain",
            logprobs=0,
            max_tokens=0,
            metadata={
                "allow_training": True,
                "environment": "dev",
                "session_id": "session_id",
            },
            model="model",
            n=0,
            response_format={},
            schema={},
            stream=True,
            temperature=0,
            top_k=0,
            top_p=0,
        )
        assert_matches_type(Completion, chat_completion, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Vlm) -> None:
        response = client.openai.chat_completions.with_raw_response.create(
            messages=[{}],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        chat_completion = response.parse()
        assert_matches_type(Completion, chat_completion, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Vlm) -> None:
        with client.openai.chat_completions.with_streaming_response.create(
            messages=[{}],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            chat_completion = response.parse()
            assert_matches_type(Completion, chat_completion, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncChatCompletions:
    parametrize = pytest.mark.parametrize("async_client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    async def test_method_create(self, async_client: AsyncVlm) -> None:
        chat_completion = await async_client.openai.chat_completions.create(
            messages=[{}],
        )
        assert_matches_type(Completion, chat_completion, path=["response"])

    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncVlm) -> None:
        chat_completion = await async_client.openai.chat_completions.create(
            messages=[
                {
                    "content": "string",
                    "role": "user",
                }
            ],
            id="id",
            domain="domain",
            logprobs=0,
            max_tokens=0,
            metadata={
                "allow_training": True,
                "environment": "dev",
                "session_id": "session_id",
            },
            model="model",
            n=0,
            response_format={},
            schema={},
            stream=True,
            temperature=0,
            top_k=0,
            top_p=0,
        )
        assert_matches_type(Completion, chat_completion, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncVlm) -> None:
        response = await async_client.openai.chat_completions.with_raw_response.create(
            messages=[{}],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        chat_completion = await response.parse()
        assert_matches_type(Completion, chat_completion, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncVlm) -> None:
        async with async_client.openai.chat_completions.with_streaming_response.create(
            messages=[{}],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            chat_completion = await response.parse()
            assert_matches_type(Completion, chat_completion, path=["response"])

        assert cast(Any, response.is_closed) is True
