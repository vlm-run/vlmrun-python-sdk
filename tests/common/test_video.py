"""Tests for video utilities."""

from pathlib import Path

import numpy as np
import pytest

from vlmrun.common.video import VideoReader, VideoWriter

# Constants for real test video properties
REAL_VIDEO_WIDTH = 294
REAL_VIDEO_HEIGHT = 240
REAL_VIDEO_FRAMES = 168
REAL_VIDEO_FPS = 25


@pytest.fixture
def sample_video() -> Path:
    """Return path to the test video file.

    Returns:
        Path: Path to the test video file.
    """
    return Path(__file__).parent.parent / "test_data" / "test.mp4"


def test_video_reader_basic(sample_video):
    """Test basic VideoReader functionality."""
    with VideoReader(sample_video) as reader:
        # Test length
        assert len(reader) == REAL_VIDEO_FRAMES

        # Test iteration
        frame_count = 0
        for frame in reader:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (REAL_VIDEO_HEIGHT, REAL_VIDEO_WIDTH, 3)
            assert frame.dtype == np.uint8
            frame_count += 1
        assert frame_count == REAL_VIDEO_FRAMES

        # Test position
        assert reader.pos() == REAL_VIDEO_FRAMES


def test_video_reader_seeking(sample_video):
    """Test VideoReader seeking functionality."""
    with VideoReader(sample_video) as reader:
        # Test seeking to specific frame
        mid_frame = REAL_VIDEO_FRAMES // 2
        reader.seek(mid_frame)
        assert reader.pos() == mid_frame

        # Test reading frame after seeking
        frame = next(reader)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (REAL_VIDEO_HEIGHT, REAL_VIDEO_WIDTH, 3)

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
        assert frame.shape == (REAL_VIDEO_HEIGHT, REAL_VIDEO_WIDTH, 3)

        # Test multiple frame access
        frames = reader[[0, REAL_VIDEO_FRAMES // 2, REAL_VIDEO_FRAMES - 1]]
        assert isinstance(frames, list)
        assert len(frames) == 3
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (REAL_VIDEO_HEIGHT, REAL_VIDEO_WIDTH, 3)

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


@pytest.fixture
def real_video_path() -> Path:
    """Return path to the real test video file.

    Returns:
        Path: Path to the test video file.
    """
    return Path(__file__).parent.parent / "test_data" / "test.mp4"


def test_video_reader_real_video(real_video_path):
    """Test VideoReader with a real video file."""
    with VideoReader(real_video_path) as reader:
        # Test basic properties
        assert len(reader) == REAL_VIDEO_FRAMES

        # Test frame dimensions
        frame = next(reader)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (REAL_VIDEO_HEIGHT, REAL_VIDEO_WIDTH, 3)
        assert frame.dtype == np.uint8

        # Test seeking
        reader.seek(REAL_VIDEO_FRAMES // 2)
        mid_frame = next(reader)
        assert isinstance(mid_frame, np.ndarray)
        assert mid_frame.shape == (REAL_VIDEO_HEIGHT, REAL_VIDEO_WIDTH, 3)

        # Test reading all frames
        reader.reset()
        frame_count = sum(1 for _ in reader)
        assert frame_count == REAL_VIDEO_FRAMES
