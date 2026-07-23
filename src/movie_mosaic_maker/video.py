from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import av
import numpy as np

from .models import VideoInfo

logger = logging.getLogger(__name__)

SEEK_DRIFT_WARNING_THRESHOLD_S = 0.5

# Video decoding uses PyAV rather than cv2.VideoCapture. PyAV's wheels bundle
# their own FFmpeg build on every platform (Linux/macOS/Windows), so reading a
# given container/codec doesn't depend on how the locally installed OpenCV
# wheel happened to be built (e.g. some opencv-python(-headless) builds ship
# without FFMPEG support and fall back to a platform backend -- AVFoundation on
# macOS, MSMF on Windows -- with narrower format coverage). cv2 is still used
# elsewhere (color.py, compositor.py) for pure image-processing calls that
# don't touch video I/O at all, so this doesn't need the FFMPEG capability.


class VideoLoadError(Exception):
    """Raised when a video file can't be opened or has no usable metadata/frames."""


@contextmanager
def open_video(path: Path) -> Iterator[av.container.InputContainer]:
    try:
        container = av.open(str(path))
    except av.error.FFmpegError as exc:
        raise VideoLoadError(f"could not open video {path}: {exc}") from exc
    try:
        if not container.streams.video:
            raise VideoLoadError(f"no video stream found in {path}")
        yield container
    finally:
        container.close()


def probe_video(path: Path) -> VideoInfo:
    with open_video(path) as container:
        return _probe_open_container(container, path)


def _probe_open_container(container: av.container.InputContainer, path: Path) -> VideoInfo:
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    width, height = stream.width, stream.height

    if stream.duration is not None:
        duration_s = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        duration_s = float(container.duration / av.time_base)
    else:
        duration_s = 0.0

    if fps <= 0 or width <= 0 or height <= 0 or duration_s <= 0:
        raise VideoLoadError(f"could not determine usable video metadata for {path}")

    frame_count = stream.frames if stream.frames > 0 else round(duration_s * fps)

    return VideoInfo(
        duration_s=duration_s,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def extract_frame_at_time(container: av.container.InputContainer, t: float, info: VideoInfo) -> np.ndarray:
    """Seek to the frame nearest timestamp `t` and return it as an RGB uint8 array.

    Clamps against duration/fps (continuous time) rather than a literal frame
    index, since `frame_count` may only be an estimate for containers that
    don't store it. Seeking can still drift on variable-frame-rate video or
    unusual keyframe intervals -- flagged with a logged warning rather than
    failing, since a slightly-off frame is still a usable mosaic tile.
    """
    t_clamped = min(max(t, 0.0), max(0.0, info.duration_s - 1.0 / info.fps))

    stream = container.streams.video[0]
    offset = round(t_clamped / stream.time_base)
    container.seek(offset, stream=stream, any_frame=False, backward=True)

    frame = next(container.decode(stream), None)
    if frame is None:
        raise VideoLoadError(f"could not read a frame near t={t:.3f}s")

    actual_t = float(frame.time) if frame.time is not None else t_clamped
    if abs(actual_t - t) > SEEK_DRIFT_WARNING_THRESHOLD_S:
        logger.warning(
            "seek drift: requested t=%.3fs but landed on t=%.3fs; video may have "
            "variable frame rate or an unusual keyframe interval",
            t,
            actual_t,
        )
    return frame.to_ndarray(format="rgb24")


def iter_all_frames(path: Path) -> Iterator[np.ndarray]:
    """Sequentially decode every frame (no seeking) -- used by exhaustive-mode sampling."""
    with open_video(path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")
