from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    fps: float
    frame_count: int
    width: int
    height: int
