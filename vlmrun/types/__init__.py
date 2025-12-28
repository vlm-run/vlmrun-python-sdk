"""Type definitions for VLM Run API."""

import uuid
from typing import Literal
from pydantic import BaseModel, Field, model_validator
from vlmrun.types.refs import (
    ImageRef as ImageRef,
    AudioRef as AudioRef,
    VideoRef as VideoRef,
    DocumentRef as DocumentRef,
    ReconRef as ReconRef,
    ArrayRef as ArrayRef,
    UrlRef as UrlRef,
)  # noqa: F401


class ImageUrl(BaseModel):
    url: str = Field(..., description="The URL of the image", pattern=r"^https?://.*$")
    detail: Literal["auto", "low", "high"] = Field(
        default="auto", description="The detail level to use for the image"
    )


class FileUrl(BaseModel):
    url: str = Field(..., description="The URL of the file", pattern=r"^https?://.*$")


class VideoUrl(FileUrl):
    url: str = Field(..., description="The URL of the video", pattern=r"^https?://.*$")


class AudioUrl(FileUrl):
    url: str = Field(..., description="The URL of the audio", pattern=r"^https?://.*$")


class DocumentUrl(FileUrl):
    url: str = Field(
        ..., description="The URL of the document", pattern=r"^https?://.*$"
    )


class MessageContent(BaseModel):
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
        if self.type in ("text", "image_url", "video_url", "audio_url", "file_url", ):
            if getattr(self, self.type) is None:
                raise ValueError(f"Must have {self.type} content [{self}]")
        elif self.type == "input_file":
            if self.file_id is None:
                raise ValueError(f"Must have file_id content [{self}]")
        else:
            raise ValueError(f"Invalid type: {self.type}")
        return self
