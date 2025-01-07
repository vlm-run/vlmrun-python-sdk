"""Video utilities for VLMRun.

This module provides utilities for video frame processing and sampling,
with a focus on extracting unique frames using CLIP embeddings.
"""
from dataclasses import dataclass
from itertools import islice
from typing import Iterable, Optional, Union

import cv2
import numpy as np
from tqdm import tqdm

HAS_CLIP = False
try:
    from nos.models import CLIP  # type: ignore
    HAS_CLIP = True
except ImportError:
    pass


@dataclass
class VideoItertools:
    """Utility class for video frame sampling using CLIP embeddings.
    
    This class provides methods for sampling unique frames from a video stream
    based on CLIP embeddings similarity. It uses the CLIP model to compute
    embeddings for each frame and selects frames that are sufficiently different
    from previously selected frames.
    
    The similarity between frames is computed using cosine similarity between
    their CLIP embeddings. Frames are only selected if their similarity to the
    previously selected frame is below a specified threshold.
    
    Note:
        Requires the CLIP model from nos.models. Install with torch extras:
        ```bash
        pip install vlmrun[torch]
        ```
    
    Example:
        ```python
        import cv2
        from vlmrun.common import VideoItertools
        
        # Initialize video capture and frame sampler
        cap = cv2.VideoCapture("video.mp4")
        video = VideoItertools()
        
        # Create frame generator
        frames = (cap.read()[1] for _ in iter(lambda: cap.isOpened(), False))
        
        # Sample unique frames
        unique_frames = video.islice(frames, step=10, similarity_threshold=0.9)
        for frame in unique_frames:
            # Process unique frames
            pass
        ```
    """
    model = None

    def __post_init__(self):
        """Initialize the CLIP model if available."""
        if not HAS_CLIP:
            raise ImportError(
                "CLIP model not available. Please install the torch extras: "
                "pip install vlmrun[torch]"
            )
        self.model = CLIP()

    def islice(
        self,
        stream: Iterable[np.ndarray],
        start: int = 0,
        step: int = 10,
        end: Optional[int] = None,
        similarity_threshold: float = 0.9,
    ) -> Iterable[np.ndarray]:
        """Sample unique frames from a video stream based on CLIP embeddings.
        
        Args:
            stream: Iterator of video frames as numpy arrays
            start: Start frame index
            step: Number of frames to skip between samples
            end: End frame index (None for entire stream)
            similarity_threshold: Threshold for frame similarity (0-1)
        
        Returns:
            Iterator of unique video frames
        
        Raises:
            AssertionError: If input frames are not numpy arrays
        """
        last_emb = None

        for img in tqdm(islice(stream, start, end, step)):
            assert isinstance(img, np.ndarray), f"Expected np.ndarray, got {type(img)}"
            _img = cv2.resize(img, (224, 224))
            emb = self.model.encode_image(_img)
            emb /= np.linalg.norm(emb)
            if last_emb is None:
                last_emb = emb
                yield img
            else:
                sim = (emb @ last_emb.T).item()
                if sim < similarity_threshold:
                    last_emb = emb
                    yield img

    def __del__(self):
        """Clean up CLIP model."""
        del self.model
