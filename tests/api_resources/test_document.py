# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from vlm import Vlm, AsyncVlm
from vlm._utils import parse_datetime
from tests.utils import assert_matches_type
from vlm.types.shared import PredictionResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestDocument:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_generate(self, client: Vlm) -> None:
        document = client.document.generate()
        assert_matches_type(PredictionResponse, document, path=["response"])

    @parametrize
    def test_method_generate_with_all_params(self, client: Vlm) -> None:
        document = client.document.generate(
            id="id",
            batch=True,
            callback_url="https://example.com",
            created_at=parse_datetime("2019-12-27T18:11:19.117Z"),
            detail="auto",
            domain="document.file",
            file_id="file_id",
            json_schema={},
            metadata={
                "allow_training": True,
                "environment": "dev",
                "session_id": "session_id",
            },
            model="vlm-1",
            url="url",
        )
        assert_matches_type(PredictionResponse, document, path=["response"])

    @parametrize
    def test_raw_response_generate(self, client: Vlm) -> None:
        response = client.document.with_raw_response.generate()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        document = response.parse()
        assert_matches_type(PredictionResponse, document, path=["response"])

    @parametrize
    def test_streaming_response_generate(self, client: Vlm) -> None:
        with client.document.with_streaming_response.generate() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            document = response.parse()
            assert_matches_type(PredictionResponse, document, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncDocument:
    parametrize = pytest.mark.parametrize("async_client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    async def test_method_generate(self, async_client: AsyncVlm) -> None:
        document = await async_client.document.generate()
        assert_matches_type(PredictionResponse, document, path=["response"])

    @parametrize
    async def test_method_generate_with_all_params(self, async_client: AsyncVlm) -> None:
        document = await async_client.document.generate(
            id="id",
            batch=True,
            callback_url="https://example.com",
            created_at=parse_datetime("2019-12-27T18:11:19.117Z"),
            detail="auto",
            domain="document.file",
            file_id="file_id",
            json_schema={},
            metadata={
                "allow_training": True,
                "environment": "dev",
                "session_id": "session_id",
            },
            model="vlm-1",
            url="url",
        )
        assert_matches_type(PredictionResponse, document, path=["response"])

    @parametrize
    async def test_raw_response_generate(self, async_client: AsyncVlm) -> None:
        response = await async_client.document.with_raw_response.generate()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        document = await response.parse()
        assert_matches_type(PredictionResponse, document, path=["response"])

    @parametrize
    async def test_streaming_response_generate(self, async_client: AsyncVlm) -> None:
        async with async_client.document.with_streaming_response.generate() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            document = await response.parse()
            assert_matches_type(PredictionResponse, document, path=["response"])

        assert cast(Any, response.is_closed) is True
