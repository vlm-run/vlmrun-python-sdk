# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from vlm import Vlm, AsyncVlm
from vlm.types import FileListResponse, StoreFileResponse
from tests.utils import assert_matches_type

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestFiles:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Vlm) -> None:
        file = client.files.create(
            file=b"raw file contents",
        )
        assert_matches_type(StoreFileResponse, file, path=["response"])

    @parametrize
    def test_method_create_with_all_params(self, client: Vlm) -> None:
        file = client.files.create(
            file=b"raw file contents",
            purpose="assistants",
        )
        assert_matches_type(StoreFileResponse, file, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Vlm) -> None:
        response = client.files.with_raw_response.create(
            file=b"raw file contents",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = response.parse()
        assert_matches_type(StoreFileResponse, file, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Vlm) -> None:
        with client.files.with_streaming_response.create(
            file=b"raw file contents",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = response.parse()
            assert_matches_type(StoreFileResponse, file, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_method_retrieve(self, client: Vlm) -> None:
        file = client.files.retrieve(
            "file_id",
        )
        assert_matches_type(StoreFileResponse, file, path=["response"])

    @parametrize
    def test_raw_response_retrieve(self, client: Vlm) -> None:
        response = client.files.with_raw_response.retrieve(
            "file_id",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = response.parse()
        assert_matches_type(StoreFileResponse, file, path=["response"])

    @parametrize
    def test_streaming_response_retrieve(self, client: Vlm) -> None:
        with client.files.with_streaming_response.retrieve(
            "file_id",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = response.parse()
            assert_matches_type(StoreFileResponse, file, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_retrieve(self, client: Vlm) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `file_id` but received ''"):
            client.files.with_raw_response.retrieve(
                "",
            )

    @parametrize
    def test_method_list(self, client: Vlm) -> None:
        file = client.files.list()
        assert_matches_type(FileListResponse, file, path=["response"])

    @parametrize
    def test_method_list_with_all_params(self, client: Vlm) -> None:
        file = client.files.list(
            limit=1,
            skip=0,
        )
        assert_matches_type(FileListResponse, file, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Vlm) -> None:
        response = client.files.with_raw_response.list()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = response.parse()
        assert_matches_type(FileListResponse, file, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Vlm) -> None:
        with client.files.with_streaming_response.list() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = response.parse()
            assert_matches_type(FileListResponse, file, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncFiles:
    parametrize = pytest.mark.parametrize("async_client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    async def test_method_create(self, async_client: AsyncVlm) -> None:
        file = await async_client.files.create(
            file=b"raw file contents",
        )
        assert_matches_type(StoreFileResponse, file, path=["response"])

    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncVlm) -> None:
        file = await async_client.files.create(
            file=b"raw file contents",
            purpose="assistants",
        )
        assert_matches_type(StoreFileResponse, file, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncVlm) -> None:
        response = await async_client.files.with_raw_response.create(
            file=b"raw file contents",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = await response.parse()
        assert_matches_type(StoreFileResponse, file, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncVlm) -> None:
        async with async_client.files.with_streaming_response.create(
            file=b"raw file contents",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = await response.parse()
            assert_matches_type(StoreFileResponse, file, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_method_retrieve(self, async_client: AsyncVlm) -> None:
        file = await async_client.files.retrieve(
            "file_id",
        )
        assert_matches_type(StoreFileResponse, file, path=["response"])

    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncVlm) -> None:
        response = await async_client.files.with_raw_response.retrieve(
            "file_id",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = await response.parse()
        assert_matches_type(StoreFileResponse, file, path=["response"])

    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncVlm) -> None:
        async with async_client.files.with_streaming_response.retrieve(
            "file_id",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = await response.parse()
            assert_matches_type(StoreFileResponse, file, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncVlm) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `file_id` but received ''"):
            await async_client.files.with_raw_response.retrieve(
                "",
            )

    @parametrize
    async def test_method_list(self, async_client: AsyncVlm) -> None:
        file = await async_client.files.list()
        assert_matches_type(FileListResponse, file, path=["response"])

    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncVlm) -> None:
        file = await async_client.files.list(
            limit=1,
            skip=0,
        )
        assert_matches_type(FileListResponse, file, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncVlm) -> None:
        response = await async_client.files.with_raw_response.list()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = await response.parse()
        assert_matches_type(FileListResponse, file, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncVlm) -> None:
        async with async_client.files.with_streaming_response.list() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = await response.parse()
            assert_matches_type(FileListResponse, file, path=["response"])

        assert cast(Any, response.is_closed) is True
