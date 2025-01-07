"""Video utilities for reading and writing video files using OpenCV."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

import cv2
import numpy as np


T = np.ndarray


class BaseVideoReader(ABC):
    """Abstract base class for video readers."""

    def __init__(self, filename: Union[str, Path]):
        """Initialize the base video reader.

        Args:
            filename (Union[str, Path]): Path to the video file.
        """
        self.filename = Path(str(filename))
        self._video = None

    def __repr__(self) -> str:
        """Return a string representation of the video reader.

        Returns:
            str: A string representation of the video reader.
        """
        return f"{self.__class__.__name__}(filename={self.filename})"

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of frames in the video.

        Returns:
            int: The number of frames in the video.
        """
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the video frames.

        Returns:
            Iterator[T]: An iterator over the video frames.
        """
        raise NotImplementedError()

    @abstractmethod
    def __next__(self) -> T:
        """Return the next frame in the video.

        Returns:
            T: The next frame in the video.

        Raises:
            StopIteration: If there are no more frames in the video.
        """
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx: Union[int, List[int]]) -> Union[T, List[T]]:
        """Return the frame(s) at the given index/indices.

        Args:
            idx (Union[int, List[int]]): The index or list of indices to retrieve.

        Returns:
            Union[T, List[T]]: The frame or list of frames at the given index/indices.

        Raises:
            IndexError: If any index is out of bounds.
            TypeError: If the index type is not supported.
        """
        raise NotImplementedError()

    @abstractmethod
    def open(self) -> None:
        """Open the video file.

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If the video file cannot be opened.
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Close the video file."""
        raise NotImplementedError()

    @abstractmethod
    def pos(self) -> Optional[int]:
        """Return the current position in the video.

        Returns:
            Optional[int]: The current frame position, or None if position cannot be determined.
        """
        raise NotImplementedError()

    @abstractmethod
    def seek(self, idx: int) -> None:
        """Seek to the given frame index in the video.

        Args:
            idx (int): The frame index to seek to.

        Raises:
            IndexError: If the index is out of bounds.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Reset the video reader to the beginning."""
        self.seek(0)

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        self.close()


class VideoReader(BaseVideoReader):
    """Video reader implementation using OpenCV."""

    def __init__(
        self, filename: Union[str, Path], transform: Optional[Callable] = None
    ):
        """Initialize the video reader.

        Args:
            filename (Union[str, Path]): Path to the video file.
            transform (Optional[Callable], optional): Optional transform to apply to each frame.
                Defaults to None.

        Raises:
            FileNotFoundError: If the video file does not exist.
        """
        super().__init__(filename)
        if not self.filename.exists():
            raise FileNotFoundError(f"{self.filename} does not exist")
        self.transform = transform
        self._video = self.open()

    def __len__(self) -> int:
        """Return the number of frames in the video.

        Returns:
            int: The number of frames in the video.
        """
        if self._video is None:
            return 0
        return int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the video frames.

        Returns:
            Iterator[T]: An iterator over the video frames.
        """
        return self

    def __next__(self) -> T:
        """Return the next frame in the video.

        Returns:
            T: The next frame in the video.

        Raises:
            StopIteration: If there are no more frames in the video.
            RuntimeError: If the video is not opened.
        """
        if self._video is None:
            raise RuntimeError("Video is not opened")
        ret, frame = self._video.read()
        if not ret:
            raise StopIteration()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.transform:
            frame = self.transform(frame)
        return frame

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[T, List[T]]:
        """Return the frame(s) at the given index/indices.

        Args:
            idx (Union[int, List[int]]): The index or list of indices to retrieve.

        Returns:
            Union[T, List[T]]: The frame or list of frames at the given index/indices.

        Raises:
            IndexError: If any index is out of bounds.
            TypeError: If the index type is not supported.
        """
        if isinstance(idx, int):
            self.seek(idx)
            return next(self)
        elif isinstance(idx, list):
            frames = []
            for i in idx:
                frames.append(self.__getitem__(i))
            return frames
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def open(self) -> cv2.VideoCapture:
        """Open the video file.

        Returns:
            cv2.VideoCapture: The opened video capture object.

        Raises:
            RuntimeError: If the video file cannot be opened.
        """
        video = cv2.VideoCapture(str(self.filename))
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.filename}")
        return video

    def close(self) -> None:
        """Close the video file."""
        if self._video is not None:
            self._video.release()
            self._video = None

    def pos(self) -> Optional[int]:
        """Return the current position in the video.

        Returns:
            Optional[int]: The current frame position, or None if position cannot be determined.
        """
        if self._video is None:
            return None
        try:
            return int(self._video.get(cv2.CAP_PROP_POS_FRAMES))
        except Exception:
            return None

    def seek(self, idx: int) -> None:
        """Seek to the given frame index in the video.

        Args:
            idx (int): The frame index to seek to.

        Raises:
            IndexError: If the index is out of bounds.
            RuntimeError: If the video is not opened.
        """
        if self._video is None:
            raise RuntimeError("Video is not opened")
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Frame index out of bounds: {idx}")
        self._video.set(cv2.CAP_PROP_POS_FRAMES, idx)


class VideoWriter:
    """Video writer implementation using OpenCV."""

    def __init__(self, filename: Union[str, Path], fps: float = 30.0):
        """Initialize the video writer.

        Args:
            filename (Union[str, Path]): Path to the output video file.
            fps (float, optional): Frames per second. Defaults to 30.0.

        Raises:
            FileExistsError: If the output file already exists.
        """
        self.filename = Path(str(filename))
        if self.filename.exists():
            raise FileExistsError(f"Output file already exists: {self.filename}")
        self.fps = fps
        self.writer = None

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video.

        Args:
            frame (np.ndarray): The frame to write. Should be an RGB image.
        """
        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                str(self.filename), fourcc, self.fps, (width, height), frame.ndim == 3
            )

        # Convert RGB to BGR for OpenCV
        if frame.ndim == 3:
            frame = frame[..., ::-1]
        self.writer.write(frame)

    def close(self) -> None:
        """Close the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        self.close()
