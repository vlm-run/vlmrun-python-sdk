# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["ModelInfoResponse"]


class ModelInfoResponse(BaseModel):
    domain: Literal[
        "document.generative",
        "document.presentation",
        "document.invoice",
        "document.receipt",
        "document.markdown",
        "video.tv-news",
        "video.tv-intelligence",
        "audio.transcription",
        "video.transcription",
        "document.file",
        "document.pdf",
        "document.resume",
        "document.utility-bill",
        "web.ecommerce-product-catalog",
        "web.github-developer-stats",
        "web.market-research",
        "vlm-1-embeddings",
    ]
    """The domain identifier for the model."""

    model: Optional[Literal["vlm-1"]] = None
    """The model identifier."""
