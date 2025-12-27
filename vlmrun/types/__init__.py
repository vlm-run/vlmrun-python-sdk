"""Type definitions for VLM Run API."""

import uuid
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field, model_validator

from vlmrun.types.refs import (
    ArrayRef as ArrayRef,
    AudioRef as AudioRef,
    DocumentRef as DocumentRef,
    ImageRef as ImageRef,
    ReconRef as ReconRef,
    UrlRef as UrlRef,
    VideoRef as VideoRef,
)


class ImageUrl(BaseModel):
    """Image URL with optional detail level."""

    url: AnyHttpUrl = Field(..., description="The URL of the image")
    detail: Literal["auto", "low", "high"] = Field(
        default="auto", description="The detail level to use for the image"
    )


class FileUrl(BaseModel):
    """Base class for file URLs."""

    url: AnyHttpUrl = Field(..., description="The URL of the file")


class VideoUrl(FileUrl):
    """Video URL."""

    url: AnyHttpUrl = Field(..., description="The URL of the video")


class AudioUrl(FileUrl):
    """Audio URL."""

    url: AnyHttpUrl = Field(..., description="The URL of the audio")


class DocumentUrl(FileUrl):
    """Document URL."""

    url: AnyHttpUrl = Field(..., description="The URL of the document")


class MessageContent(BaseModel):
    """Message content with various input types."""

    type: Literal[
        "text", "image_url", "video_url", "audio_url", "file_url", "input_file"
    ]
    """The type of the message content"""
    text: str | None = None
    """The text content of the message"""
    image_url: ImageUrl | None = None
    """The image URL content of the message"""
    video_url: VideoUrl | None = None
    """The video URL content of the message"""
    audio_url: AudioUrl | None = None
    """The audio URL content of the message"""
    file_url: FileUrl | None = None
    """The file URL content of the message"""
    file_id: str | uuid.UUID | None = None
    """The file ID content of the message (either a string or a UUID)"""

    @model_validator(mode="after")
    def check_type_and_content(self) -> "MessageContent":
        """Validate that the content matches the type."""
        if self.type == "input_file":
            if self.file_id is None and self.file_url is None:
                raise ValueError("Must have either file_id or file_url")
            return self

        # For other types, the type name matches the field name
        if self.type in ("text", "image_url", "video_url", "audio_url", "file_url"):
            if getattr(self, self.type) is None:
                raise ValueError(f"Must have {self.type}")
        else:
            raise ValueError(f"Invalid type: {self.type}")
        return self
