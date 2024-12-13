# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ChatModel"]


class ChatModel(BaseModel):
    id: str

    created: Optional[int] = None

    object: Optional[Literal["model"]] = None

    owned_by: Optional[str] = None
