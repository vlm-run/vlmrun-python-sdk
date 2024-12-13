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


class TestEmbeddings:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Vlm) -> None:
        embedding = client.experimental.image.embeddings.create()
        assert_matches_type(PredictionResponse, embedding, path=["response"])

    @parametrize
    def test_method_create_with_all_params(self, client: Vlm) -> None:
        embedding = client.experimental.image.embeddings.create(
            id="id",
            batch=True,
            callback_url="https://example.com",
            created_at=parse_datetime("2019-12-27T18:11:19.117Z"),
            image="image",
            metadata={
                "allow_training": True,
                "environment": "dev",
                "session_id": "session_id",
            },
            model="vlm-1-embeddings",
            text="text",
        )
        assert_matches_type(PredictionResponse, embedding, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Vlm) -> None:
        response = client.experimental.image.embeddings.with_raw_response.create()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        embedding = response.parse()
        assert_matches_type(PredictionResponse, embedding, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Vlm) -> None:
        with client.experimental.image.embeddings.with_streaming_response.create() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            embedding = response.parse()
            assert_matches_type(PredictionResponse, embedding, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncEmbeddings:
    parametrize = pytest.mark.parametrize("async_client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    async def test_method_create(self, async_client: AsyncVlm) -> None:
        embedding = await async_client.experimental.image.embeddings.create()
        assert_matches_type(PredictionResponse, embedding, path=["response"])

    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncVlm) -> None:
        embedding = await async_client.experimental.image.embeddings.create(
            id="id",
            batch=True,
            callback_url="https://example.com",
            created_at=parse_datetime("2019-12-27T18:11:19.117Z"),
            image="image",
            metadata={
                "allow_training": True,
                "environment": "dev",
                "session_id": "session_id",
            },
            model="vlm-1-embeddings",
            text="text",
        )
        assert_matches_type(PredictionResponse, embedding, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncVlm) -> None:
        response = await async_client.experimental.image.embeddings.with_raw_response.create()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        embedding = await response.parse()
        assert_matches_type(PredictionResponse, embedding, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncVlm) -> None:
        async with async_client.experimental.image.embeddings.with_streaming_response.create() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            embedding = await response.parse()
            assert_matches_type(PredictionResponse, embedding, path=["response"])

        assert cast(Any, response.is_closed) is True
