"""Type definitions for VLM Run API."""

import uuid
from typing import Literal
from pydantic import BaseModel, Field, AnyHttpUrl, model_validator
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
    url: AnyHttpUrl = Field(..., description="The URL of the image")
    detail: Literal["auto", "low", "high"] = Field(
        default="auto", description="The detail level to use for the image"
    )


class FileUrl(BaseModel):
    url: AnyHttpUrl = Field(..., description="The URL of the file")


class VideoUrl(FileUrl):
    url: AnyHttpUrl = Field(..., description="The URL of the video")


class AudioUrl(FileUrl):
    url: AnyHttpUrl = Field(..., description="The URL of the audio")


class DocumentUrl(FileUrl):
    url: AnyHttpUrl = Field(..., description="The URL of the document")


class MessageContent(BaseModel):
    type: Literal["text", "image_url", "video_url", "audio_url", "input_file"]
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
        if self.type == "text":
            if self.text is None:
                raise ValueError("Must have text")
        elif self.type == "image_url":
            if self.image_url is None:
                raise ValueError("Must have image_url")
        elif self.type == "video_url":
            if self.video_url is None:
                raise ValueError("Must have video_url")
        elif self.type == "audio_url":
            if self.audio_url is None:
                raise ValueError("Must have audio_url")
        elif self.type == "file_url":
            if self.file_url is None:
                raise ValueError("Must have file_url")
        elif self.type == "input_file":
            if self.file_id is None and self.file_url is None:
                raise ValueError("Must have either file_id or file_url")
        else:
            raise ValueError(f"Invalid type: {self.type}")
        return self
