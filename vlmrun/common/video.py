"""Video utilities for reading and writing video files using OpenCV."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Union,
)

from vlmrun.client.exceptions import InputError
from vlmrun.common.dependencies import require_cv2, require_numpy

if TYPE_CHECKING:
    import numpy as np

    Frame = np.ndarray
else:
    Frame = Any


class SampledFrame(NamedTuple):
    """One frame drawn out of a video, with where it came from.

    Attributes:
        index: The frame's position in the source video, counting from 0.
        timestamp_s: Its presentation time in seconds.
        frame: The frame itself, RGB.
    """

    index: int
    timestamp_s: float
    frame: Frame


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
    def __iter__(self) -> Iterator[Frame]:
        """Return an iterator over the video frames.

        Returns:
            Iterator[Frame]: An iterator over the video frames.
        """
        raise NotImplementedError()

    @abstractmethod
    def __next__(self) -> Frame:
        """Return the next frame in the video.

        Returns:
            Frame: The next frame in the video.

        Raises:
            StopIteration: If there are no more frames in the video.
        """
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Frame, List[Frame]]:
        """Return the frame(s) at the given index/indices.

        Args:
            idx (Union[int, List[int]]): The index or list of indices to retrieve.

        Returns:
            Union[Frame, List[Frame]]: The frame or list of frames at the given index/indices.

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

    def __del__(self):
        """Release the capture if the reader was never closed.

        ``VideoReader(path).frames(fps=2)`` is a reasonable thing to write, and
        it leaves no handle to close. A context manager is still the explicit
        way; this keeps the terse form from holding a decoder open until the
        interpreter happens to collect it.
        """
        try:
            self.close()
        except Exception:
            # Interpreter shutdown can pull the module out from under us.
            pass


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
        self._cv2 = require_cv2()
        require_numpy()
        self.transform = transform
        self._video = self.open()

    def __len__(self) -> int:
        """Return the number of frames in the video.

        Returns:
            int: The number of frames in the video.
        """
        if self._video is None:
            return 0
        return int(self._video.get(self._cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[Frame]:
        """Return an iterator over the video frames.

        Returns:
            Iterator[Frame]: An iterator over the video frames.
        """
        return self

    def __next__(self) -> Frame:
        """Return the next frame in the video.

        Returns:
            Frame: The next frame in the video.

        Raises:
            StopIteration: If there are no more frames in the video.
            RuntimeError: If the video is not opened.
        """
        if self._video is None:
            raise RuntimeError("Video is not opened")
        ret, frame = self._video.read()
        if not ret:
            raise StopIteration()
        frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        if self.transform:
            frame = self.transform(frame)
        return frame

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Frame, List[Frame]]:
        """Return the frame(s) at the given index/indices.

        Args:
            idx (Union[int, List[int]]): The index or list of indices to retrieve.

        Returns:
            Union[Frame, List[Frame]]: The frame or list of frames at the given index/indices.

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

    def open(self):
        """Open the video file.

        Returns:
            The opened video capture object.

        Raises:
            RuntimeError: If the video file cannot be opened.
        """
        video = self._cv2.VideoCapture(str(self.filename))
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
            return int(self._video.get(self._cv2.CAP_PROP_POS_FRAMES))
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
        self._video.set(self._cv2.CAP_PROP_POS_FRAMES, idx)

    @property
    def fps(self) -> float:
        """The video's native frame rate.

        Returns:
            float: Frames per second, or 0.0 when the container does not report
                a usable rate (some streams and remuxed files report 0 or NaN).
        """
        if self._video is None:
            return 0.0
        try:
            rate = float(self._video.get(self._cv2.CAP_PROP_FPS))
        except Exception:
            return 0.0
        # NaN fails its own equality test; a four-figure rate is a bad header.
        return rate if rate == rate and 0.0 < rate < 1000.0 else 0.0

    @property
    def duration_s(self) -> float:
        """The video's duration.

        Returns:
            float: Seconds, or 0.0 when the frame count or rate is unknown.
        """
        count, rate = len(self), self.fps
        return count / rate if count > 0 and rate > 0 else 0.0

    def _rewind(self) -> None:
        """Seek to the first frame without :meth:`seek`'s bounds check.

        ``seek`` validates against ``len(self)``, which is ``CAP_PROP_FRAME_COUNT``
        and is 0 for a container with no frame count — a file this class can
        still read straight through.
        """
        if self._video is None:
            raise RuntimeError("Video is not opened")
        self._video.set(self._cv2.CAP_PROP_POS_FRAMES, 0)

    def frames(
        self, fps: float, *, max_frames: Optional[int] = None
    ) -> Iterator[SampledFrame]:
        """Draw frames at a fixed rate, in presentation order.

        Frames are decoded sequentially and emitted whenever the presentation
        clock reaches the next ``1/fps`` mark, rather than by seeking to
        computed indices. Sequential decoding costs more CPU but is right in
        the two cases seeking is wrong: a container that misreports
        ``CAP_PROP_FRAME_COUNT``, and a variable-rate video, where frame index
        and time do not scale together. For the sampling rates this is built
        for — a few frames a second, each one sent to a model — decoding is not
        the bottleneck.

        A rate at or above the video's own returns every frame; nothing is
        interpolated. After a gap in the source, sampling resumes from the
        frame that ends it instead of emitting a burst to catch up.

        Args:
            fps (float): Frames to emit per second of video.
            max_frames (Optional[int], optional): Stop after this many frames.
                Defaults to None, which reads to the end.

        Yields:
            SampledFrame: The frame, its index and its timestamp.

        Raises:
            InputError: If ``fps`` is not positive.
            RuntimeError: If the video is not opened.

        Example:
            ```python
            from itertools import islice
            from vlmrun.common.video import VideoReader

            with VideoReader("door.mp4") as reader:
                for frame in islice(reader.frames(fps=2), 10):
                    print(frame.timestamp_s, frame.frame.shape)
            ```
        """
        if fps <= 0:
            raise InputError(
                message=f"sampling rate must be positive; got {fps}",
                suggestion="Pass a rate like 1 (one frame a second) or 0.5 (one every two).",
            )
        if self._video is None:
            raise RuntimeError("Video is not opened")

        interval_s = 1.0 / fps
        native = self.fps
        self._rewind()

        # Land on the frame nearest each mark rather than the first one past
        # it: a container whose clock is a rounding error short of 1.000s
        # should still answer a 1 fps request with that frame.
        tolerance_s = 0.5 / native if native > 0 else 1e-6

        emitted, index, next_at = 0, -1, 0.0
        while True:
            try:
                frame = next(self)
            except StopIteration:
                return
            index += 1
            # Read the clock after the decode: while positioned on a frame,
            # CAP_PROP_POS_MSEC is the time of the frame already returned.
            try:
                position_ms = float(self._video.get(self._cv2.CAP_PROP_POS_MSEC))
            except Exception:
                position_ms = 0.0

            timestamp_s = position_ms / 1000.0
            if timestamp_s <= 0.0 and index > 0:
                # No usable clock — fall back to counting frames.
                timestamp_s = index / native if native > 0 else 0.0

            if timestamp_s + tolerance_s < next_at:
                continue
            yield SampledFrame(index=index, timestamp_s=timestamp_s, frame=frame)
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                return
            next_at = max(next_at + interval_s, timestamp_s + interval_s)


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
        self._cv2 = require_cv2()
        require_numpy()
        self.fps = fps
        self.writer = None

    def write(self, frame: Frame) -> None:
        """Write a frame to the video.

        Args:
            frame: The frame to write. Should be an RGB image.
        """
        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = self._cv2.VideoWriter(
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
