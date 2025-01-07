"""Common utilities for VLMRun."""

from .image import encode_image, download_image

__all__ = ["encode_image", "download_image"]

try:
    from .video import VideoItertools
    __all__.append("VideoItertools")
except ImportError:
    pass  # VideoItertools requires optional torch dependencies
