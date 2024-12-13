# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["Completion", "Usage"]


class Usage(BaseModel):
    completion_tokens: int

    prompt_tokens: int

    total_tokens: int


class Completion(BaseModel):
    choices: List[object]

    model: str

    id: Optional[str] = None

    created: Optional[int] = None

    object: Optional[Literal["chat.completion", "chat.completion.chunk"]] = None

    system_fingerprint: Optional[str] = None

    usage: Optional[Usage] = None
