from dataclasses import dataclass
from typing import Any

from .images import Images


@dataclass
class Client:
    """VLMRun API client."""

    api_key: str

    def __post_init__(self) -> None:
        """Initialize client resources after instantiation."""
        self._image = Images(self)

    @property
    def image(self) -> Images:
        """Get the image resource interface.

        Returns:
            Images: The image resource interface
        """
        return self._image
