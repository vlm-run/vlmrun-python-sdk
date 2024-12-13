# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .._types import FileTypes

__all__ = ["FileCreateParams"]


class FileCreateParams(TypedDict, total=False):
    file: Required[FileTypes]

    purpose: Literal[
        "assistants", "assistants_output", "batch", "batch_output", "fine-tune", "fine-tune-results", "vision"
    ]
