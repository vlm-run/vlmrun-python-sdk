# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from vlm import Vlm, AsyncVlm
from tests.utils import assert_matches_type
from vlm.types.shared import PredictionResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestResponse:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_retrieve(self, client: Vlm) -> None:
        response = client.response.retrieve(
            "id",
        )
        assert_matches_type(PredictionResponse, response, path=["response"])

    @parametrize
    def test_raw_response_retrieve(self, client: Vlm) -> None:
        http_response = client.response.with_raw_response.retrieve(
            "id",
        )

        assert http_response.is_closed is True
        assert http_response.http_request.headers.get("X-Stainless-Lang") == "python"
        response = http_response.parse()
        assert_matches_type(PredictionResponse, response, path=["response"])

    @parametrize
    def test_streaming_response_retrieve(self, client: Vlm) -> None:
        with client.response.with_streaming_response.retrieve(
            "id",
        ) as http_response:
            assert not http_response.is_closed
            assert http_response.http_request.headers.get("X-Stainless-Lang") == "python"

            response = http_response.parse()
            assert_matches_type(PredictionResponse, response, path=["response"])

        assert cast(Any, http_response.is_closed) is True

    @parametrize
    def test_path_params_retrieve(self, client: Vlm) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `id` but received ''"):
            client.response.with_raw_response.retrieve(
                "",
            )


class TestAsyncResponse:
    parametrize = pytest.mark.parametrize("async_client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    async def test_method_retrieve(self, async_client: AsyncVlm) -> None:
        response = await async_client.response.retrieve(
            "id",
        )
        assert_matches_type(PredictionResponse, response, path=["response"])

    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncVlm) -> None:
        http_response = await async_client.response.with_raw_response.retrieve(
            "id",
        )

        assert http_response.is_closed is True
        assert http_response.http_request.headers.get("X-Stainless-Lang") == "python"
        response = await http_response.parse()
        assert_matches_type(PredictionResponse, response, path=["response"])

    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncVlm) -> None:
        async with async_client.response.with_streaming_response.retrieve(
            "id",
        ) as http_response:
            assert not http_response.is_closed
            assert http_response.http_request.headers.get("X-Stainless-Lang") == "python"

            response = await http_response.parse()
            assert_matches_type(PredictionResponse, response, path=["response"])

        assert cast(Any, http_response.is_closed) is True

    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncVlm) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `id` but received ''"):
            await async_client.response.with_raw_response.retrieve(
                "",
            )
