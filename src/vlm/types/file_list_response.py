# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List
from typing_extensions import TypeAlias

from .store_file_response import StoreFileResponse

__all__ = ["FileListResponse"]

FileListResponse: TypeAlias = List[StoreFileResponse]
