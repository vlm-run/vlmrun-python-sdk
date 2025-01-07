"""Tests for video utilities."""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

try:
    from vlmrun.common import VideoItertools
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

# Skip all tests if CLIP is not available
pytestmark = pytest.mark.skipif(
    not HAS_CLIP,
    reason="CLIP model not available. Install torch extras with: pip install vlmrun[torch]"
)


# Use a small test video URL that's reliable
VIDEO_URL = "https://zackakil.github.io/video-intelligence-api-visualiser/assets/test_video.mp4"


@pytest.fixture
def sample_video(tmp_path) -> Path:
    """Download and cache a sample video for testing."""
    video_path = tmp_path / "test_video.mp4"
    if not video_path.exists():
        import requests
        response = requests.get(VIDEO_URL)
        response.raise_for_status()
        video_path.write_bytes(response.content)
    return video_path


@pytest.fixture
def frame_generator(sample_video):
    """Create a frame generator from the sample video."""
    frames = []
    cap = cv2.VideoCapture(str(sample_video))
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return iter(frames)


def test_video_itertools_initialization():
    """Test VideoItertools initialization."""
    try:
        video = VideoItertools()
        assert video.model is not None
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependencies: {e}")


def test_frame_sampling(frame_generator):
    """Test frame sampling with VideoItertools."""
    try:
        video = VideoItertools()
        
        # Sample frames
        frames = list(video.islice(
            frame_generator,
            step=5,
            similarity_threshold=0.9
        ))
        
        # Verify we got some frames
        assert len(frames) > 0
        
        # Verify frame format
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3  # Height, width, channels
            assert frame.shape[2] == 3  # RGB channels
            
            # Convert to PIL and verify
            pil_img = Image.fromarray(frame).convert("RGB")
            assert isinstance(pil_img, Image.Image)
            assert pil_img.mode == "RGB"
            
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependencies: {e}")


def test_frame_sampling_parameters(frame_generator):
    """Test different parameters for frame sampling."""
    try:
        video = VideoItertools()
        
        # Test with different thresholds
        high_threshold = list(video.islice(frame_generator, step=5, similarity_threshold=0.95))
        low_threshold = list(video.islice(frame_generator, step=5, similarity_threshold=0.5))
        
        # Lower threshold should result in fewer frames (more strict)
        assert len(low_threshold) <= len(high_threshold)
        
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependencies: {e}")


def test_invalid_input():
    """Test handling of invalid inputs."""
    try:
        video = VideoItertools()
        
        # Create invalid frame (wrong shape)
        invalid_frame = np.zeros((10, 10))  # 2D array instead of 3D
        with pytest.raises((AssertionError, ValueError)):
            list(video.islice([invalid_frame], step=1))
            
        # Create invalid frame type
        invalid_type = [np.array([1, 2, 3])]  # Wrong dimensions
        with pytest.raises((AssertionError, ValueError)):
            list(video.islice(invalid_type, step=1))
            
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependencies: {e}")
