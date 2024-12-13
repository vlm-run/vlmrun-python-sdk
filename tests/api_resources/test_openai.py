# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from vlm import Vlm, AsyncVlm
from tests.utils import assert_matches_type

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestOpenAI:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_health(self, client: Vlm) -> None:
        openai = client.openai.health()
        assert_matches_type(object, openai, path=["response"])

    @parametrize
    def test_raw_response_health(self, client: Vlm) -> None:
        response = client.openai.with_raw_response.health()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        openai = response.parse()
        assert_matches_type(object, openai, path=["response"])

    @parametrize
    def test_streaming_response_health(self, client: Vlm) -> None:
        with client.openai.with_streaming_response.health() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            openai = response.parse()
            assert_matches_type(object, openai, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncOpenAI:
    parametrize = pytest.mark.parametrize("async_client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    async def test_method_health(self, async_client: AsyncVlm) -> None:
        openai = await async_client.openai.health()
        assert_matches_type(object, openai, path=["response"])

    @parametrize
    async def test_raw_response_health(self, async_client: AsyncVlm) -> None:
        response = await async_client.openai.with_raw_response.health()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        openai = await response.parse()
        assert_matches_type(object, openai, path=["response"])

    @parametrize
    async def test_streaming_response_health(self, async_client: AsyncVlm) -> None:
        async with async_client.openai.with_streaming_response.health() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            openai = await response.parse()
            assert_matches_type(object, openai, path=["response"])

        assert cast(Any, response.is_closed) is True
