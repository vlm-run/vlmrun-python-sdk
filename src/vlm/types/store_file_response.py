# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from datetime import datetime
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["StoreFileResponse"]


class StoreFileResponse(BaseModel):
    bytes: int
    """Size of the file in bytes"""

    filename: str
    """Name of the file"""

    purpose: Literal[
        "assistants", "assistants_output", "batch", "batch_output", "fine-tune", "fine-tune-results", "vision"
    ]
    """Purpose of the file"""

    id: Optional[str] = None
    """Unique identifier of the file"""

    created_at: Optional[datetime] = None
    """Date and time when the file was created (in UTC timezone)"""

    object: Optional[Literal["file"]] = None
    """Type of the file"""
