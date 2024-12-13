# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from ..._models import BaseModel
from .chat_model import ChatModel

__all__ = ["Model"]


class Model(BaseModel):
    data: List[ChatModel]

    object: Optional[str] = None
