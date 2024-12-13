# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict

__all__ = [
    "ChatCompletionCreateParams",
    "Message",
    "MessageContentUnionMember1",
    "MessageContentUnionMember1ImageURL",
    "Metadata",
]


class ChatCompletionCreateParams(TypedDict, total=False):
    messages: Required[Iterable[Message]]

    id: str

    domain: Optional[str]
    """Domain of the request"""

    logprobs: Optional[int]

    max_tokens: int

    metadata: Metadata
    """Metadata of the request"""

    model: str

    n: Optional[int]

    response_format: Optional[object]
    """Format of the response"""

    schema: Optional[object]
    """Schema of the request"""

    stream: bool

    temperature: float

    top_k: Optional[int]

    top_p: float


class MessageContentUnionMember1ImageURL(TypedDict, total=False):
    detail: Required[Literal["auto", "low", "high"]]

    url: Required[str]


class MessageContentUnionMember1(TypedDict, total=False):
    type: Required[Literal["text", "image_url"]]

    image_url: Optional[MessageContentUnionMember1ImageURL]

    text: Optional[str]


class Message(TypedDict, total=False):
    content: Union[str, Iterable[MessageContentUnionMember1], None]

    role: Literal["user", "assistant", "system"]


class Metadata(TypedDict, total=False):
    allow_training: bool
    """Whether the file can be used for training"""

    environment: Literal["dev", "staging", "prod"]
    """The environment where the request was made."""

    session_id: Optional[str]
    """The session ID of the request"""
