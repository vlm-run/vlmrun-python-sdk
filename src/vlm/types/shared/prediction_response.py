# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from datetime import datetime

from ..._models import BaseModel

__all__ = ["PredictionResponse"]


class PredictionResponse(BaseModel):
    id: Optional[str] = None
    """Unique identifier of the response."""

    completed_at: Optional[datetime] = None
    """Date and time when the response was completed (in UTC timezone)"""

    created_at: Optional[datetime] = None
    """Date and time when the request was created (in UTC timezone)"""

    response: Optional[object] = None
    """The response from the model."""

    status: Optional[str] = None
    """The status of the job."""
