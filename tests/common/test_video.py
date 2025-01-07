"""Tests for video utilities."""
import os
from pathlib import Path

import numpy as np
import pytest

from vlmrun.common.video import VideoReader, VideoWriter


@pytest.fixture
def sample_video(tmp_path) -> Path:
    """Create a sample video file for testing.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        Path: Path to the sample video file.
    """
    # Create a sample video using VideoWriter
    video_path = tmp_path / "sample.mp4"
    frames = 30
    width, height = 640, 480

    with VideoWriter(video_path, fps=30.0) as writer:
        # Create a sequence of frames with different colors
        for i in range(frames):
            # Create a frame with a moving gradient
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 255 // frames)  # Red channel
            frame[:, :, 1] = ((frames - i) * 255 // frames)  # Green channel
            frame[:, :, 2] = (i * 255 // frames)  # Blue channel
            writer.write(frame)

    return video_path


def test_video_reader_basic(sample_video):
    """Test basic VideoReader functionality."""
    with VideoReader(sample_video) as reader:
        # Test length
        assert len(reader) == 30

        # Test iteration
        frame_count = 0
        for frame in reader:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)
            assert frame.dtype == np.uint8
            frame_count += 1
        assert frame_count == 30

        # Test position
        assert reader.pos() == 30


def test_video_reader_seeking(sample_video):
    """Test VideoReader seeking functionality."""
    with VideoReader(sample_video) as reader:
        # Test seeking to specific frame
        reader.seek(15)
        assert reader.pos() == 15

        # Test reading frame after seeking
        frame = next(reader)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

        # Test seeking to start
        reader.reset()
        assert reader.pos() == 0

        # Test seeking to last frame
        reader.seek(len(reader) - 1)
        frame = next(reader)
        assert isinstance(frame, np.ndarray)
        with pytest.raises(StopIteration):
            next(reader)


def test_video_reader_getitem(sample_video):
    """Test VideoReader indexing functionality."""
    with VideoReader(sample_video) as reader:
        # Test single frame access
        frame = reader[0]
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

        # Test multiple frame access
        frames = reader[[0, 15, 29]]
        assert isinstance(frames, list)
        assert len(frames) == 3
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)

        # Test invalid index
        with pytest.raises(IndexError):
            reader[len(reader)]

        # Test invalid index type
        with pytest.raises(TypeError):
            reader["invalid"]


def test_video_reader_errors():
    """Test VideoReader error handling."""
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        VideoReader("nonexistent.mp4")


def test_video_writer_basic(tmp_path):
    """Test basic VideoWriter functionality."""
    output_path = tmp_path / "output.mp4"
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test writing frames
    with VideoWriter(output_path, fps=30.0) as writer:
        for _ in range(10):
            writer.write(frame)

    # Verify output file exists
    assert output_path.exists()

    # Verify video can be read back
    with VideoReader(output_path) as reader:
        assert len(reader) == 10
        for frame in reader:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)
            assert frame.dtype == np.uint8


def test_video_writer_errors(tmp_path):
    """Test VideoWriter error handling."""
    output_path = tmp_path / "output.mp4"

    # Create a video file
    with VideoWriter(output_path) as writer:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(frame)

    # Test file exists error
    with pytest.raises(FileExistsError):
        VideoWriter(output_path)


def test_video_context_managers(sample_video, tmp_path):
    """Test context manager functionality for both VideoReader and VideoWriter."""
    # Test VideoReader context manager
    with VideoReader(sample_video) as reader:
        assert len(reader) > 0
        frame = next(reader)
        assert isinstance(frame, np.ndarray)

    # Verify reader is closed
    assert reader._video is None

    # Test VideoWriter context manager
    output_path = tmp_path / "output.mp4"
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with VideoWriter(output_path) as writer:
        writer.write(frame)

    # Verify writer is closed
    assert writer.writer is None
    assert output_path.exists()
