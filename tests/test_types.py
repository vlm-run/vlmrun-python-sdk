"""Tests for vlmrun.types module."""

import uuid
import pytest
from pydantic import ValidationError

from vlmrun.types import (
    ImageUrl,
    FileUrl,
    VideoUrl,
    AudioUrl,
    DocumentUrl,
    MessageContent,
)


class TestImageUrl:
    """Tests for ImageUrl model."""

    def test_image_url_with_default_detail(self):
        """Test ImageUrl with default detail level."""
        image_url = ImageUrl(url="https://example.com/image.jpg")
        assert str(image_url.url) == "https://example.com/image.jpg"
        assert image_url.detail == "auto"

    def test_image_url_with_custom_detail(self):
        """Test ImageUrl with custom detail level."""
        image_url = ImageUrl(url="https://example.com/image.jpg", detail="high")
        assert image_url.detail == "high"

    def test_image_url_invalid_detail(self):
        """Test ImageUrl with invalid detail level."""
        with pytest.raises(ValidationError):
            ImageUrl(url="https://example.com/image.jpg", detail="invalid")


class TestFileUrl:
    """Tests for FileUrl and subclasses."""

    def test_file_url(self):
        """Test FileUrl model."""
        file_url = FileUrl(url="https://example.com/file.pdf")
        assert str(file_url.url) == "https://example.com/file.pdf"

    def test_video_url(self):
        """Test VideoUrl model."""
        video_url = VideoUrl(url="https://example.com/video.mp4")
        assert str(video_url.url) == "https://example.com/video.mp4"

    def test_audio_url(self):
        """Test AudioUrl model."""
        audio_url = AudioUrl(url="https://example.com/audio.mp3")
        assert str(audio_url.url) == "https://example.com/audio.mp3"

    def test_document_url(self):
        """Test DocumentUrl model."""
        doc_url = DocumentUrl(url="https://example.com/document.pdf")
        assert str(doc_url.url) == "https://example.com/document.pdf"


class TestMessageContent:
    """Tests for MessageContent model."""

    def test_text_content(self):
        """Test MessageContent with text type."""
        content = MessageContent(type="text", text="Hello, world!")
        assert content.type == "text"
        assert content.text == "Hello, world!"

    def test_text_content_missing_text(self):
        """Test MessageContent with text type but missing text field."""
        with pytest.raises(ValidationError) as exc_info:
            MessageContent(type="text")
        assert "Must have text" in str(exc_info.value)

    def test_image_url_content(self):
        """Test MessageContent with image_url type."""
        image_url = ImageUrl(url="https://example.com/image.jpg")
        content = MessageContent(type="image_url", image_url=image_url)
        assert content.type == "image_url"
        assert content.image_url is not None

    def test_image_url_content_missing_image_url(self):
        """Test MessageContent with image_url type but missing image_url field."""
        with pytest.raises(ValidationError) as exc_info:
            MessageContent(type="image_url")
        assert "Must have image_url" in str(exc_info.value)

    def test_video_url_content(self):
        """Test MessageContent with video_url type."""
        video_url = VideoUrl(url="https://example.com/video.mp4")
        content = MessageContent(type="video_url", video_url=video_url)
        assert content.type == "video_url"
        assert content.video_url is not None

    def test_video_url_content_missing_video_url(self):
        """Test MessageContent with video_url type but missing video_url field."""
        with pytest.raises(ValidationError) as exc_info:
            MessageContent(type="video_url")
        assert "Must have video_url" in str(exc_info.value)

    def test_audio_url_content(self):
        """Test MessageContent with audio_url type."""
        audio_url = AudioUrl(url="https://example.com/audio.mp3")
        content = MessageContent(type="audio_url", audio_url=audio_url)
        assert content.type == "audio_url"
        assert content.audio_url is not None

    def test_audio_url_content_missing_audio_url(self):
        """Test MessageContent with audio_url type but missing audio_url field."""
        with pytest.raises(ValidationError) as exc_info:
            MessageContent(type="audio_url")
        assert "Must have audio_url" in str(exc_info.value)

    def test_file_url_content(self):
        """Test MessageContent with file_url type."""
        file_url = FileUrl(url="https://example.com/file.pdf")
        content = MessageContent(type="file_url", file_url=file_url)
        assert content.type == "file_url"
        assert content.file_url is not None

    def test_file_url_content_missing_file_url(self):
        """Test MessageContent with file_url type but missing file_url field."""
        with pytest.raises(ValidationError) as exc_info:
            MessageContent(type="file_url")
        assert "Must have file_url" in str(exc_info.value)

    def test_input_file_with_file_id(self):
        """Test MessageContent with input_file type and file_id."""
        content = MessageContent(type="input_file", file_id="file-123")
        assert content.type == "input_file"
        assert content.file_id == "file-123"

    def test_input_file_with_uuid(self):
        """Test MessageContent with input_file type and UUID file_id."""
        test_uuid = uuid.uuid4()
        content = MessageContent(type="input_file", file_id=test_uuid)
        assert content.type == "input_file"
        assert content.file_id == test_uuid

    def test_input_file_with_file_url(self):
        """Test MessageContent with input_file type and file_url."""
        file_url = FileUrl(url="https://example.com/file.pdf")
        content = MessageContent(type="input_file", file_url=file_url)
        assert content.type == "input_file"
        assert content.file_url is not None

    def test_input_file_missing_both(self):
        """Test MessageContent with input_file type but missing both file_id and file_url."""
        with pytest.raises(ValidationError) as exc_info:
            MessageContent(type="input_file")
        assert "Must have either file_id or file_url" in str(exc_info.value)

    def test_invalid_type(self):
        """Test MessageContent with invalid type."""
        with pytest.raises(ValidationError):
            MessageContent(type="invalid_type", text="test")
