from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def make_video(tmp_path: Path):
    """Factory fixture building a tiny synthetic .avi (MJPG, broadly supported
    without extra codecs) from a list of RGB frame colors, for tests that need
    a real, readable, seekable video without shipping a binary fixture file."""

    def _make(
        name: str,
        colors: list[tuple[int, int, int]],
        fps: float = 10.0,
        size: tuple[int, int] = (64, 48),
    ) -> Path:
        path = tmp_path / name
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        assert writer.isOpened(), "test video writer failed to open"
        for color in colors:
            frame_bgr = np.full((size[1], size[0], 3), color[::-1], dtype=np.uint8)
            writer.write(frame_bgr)
        writer.release()
        return path

    return _make
