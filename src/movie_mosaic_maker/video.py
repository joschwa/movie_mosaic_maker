from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from .models import VideoInfo

logger = logging.getLogger(__name__)

SEEK_DRIFT_WARNING_THRESHOLD_S = 0.5


class VideoLoadError(Exception):
    """Raised when a video file can't be opened or has no usable metadata/frames."""


@contextmanager
def open_video(path: Path) -> Iterator[cv2.VideoCapture]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoLoadError(f"could not open video {path}")
    try:
        yield cap
    finally:
        cap.release()


def probe_video(path: Path) -> VideoInfo:
    with open_video(path) as cap:
        return _probe_open_capture(cap, path)


def _probe_open_capture(cap: cv2.VideoCapture, path: Path) -> VideoInfo:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise VideoLoadError(f"could not determine usable video metadata for {path}")
    return VideoInfo(
        duration_s=frame_count / fps,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def extract_frame_at_time(cap: cv2.VideoCapture, t: float, info: VideoInfo) -> np.ndarray:
    """Seek to the frame nearest timestamp `t` and return it as an RGB uint8 array.

    Seeking by frame index (rather than CAP_PROP_POS_MSEC, which is unreliable
    across codecs) is the recommended approach, but can still drift on
    variable-frame-rate video or unusual keyframe intervals -- flagged with a
    logged warning rather than failing, since a slightly-off frame is still a
    usable mosaic tile.
    """
    frame_index = min(max(round(t * info.fps), 0), info.frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame_bgr = cap.read()
    if not ret:
        raise VideoLoadError(f"could not read frame {frame_index} (requested t={t:.3f}s)")

    actual_index = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
    actual_t = actual_index / info.fps
    if abs(actual_t - t) > SEEK_DRIFT_WARNING_THRESHOLD_S:
        logger.warning(
            "seek drift: requested t=%.3fs (frame %d) but landed on frame %d (t=%.3fs); "
            "video may have variable frame rate or an unusual keyframe interval",
            t,
            frame_index,
            actual_index,
            actual_t,
        )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def iter_all_frames(path: Path) -> Iterator[np.ndarray]:
    """Sequentially decode every frame (no seeking) -- used by exhaustive-mode sampling."""
    with open_video(path) as cap:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
