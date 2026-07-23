from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class Candidate:
    source_id: str
    kind: Literal["image", "video"]
    path: Path
    frame_index: int | None
    timestamp: float | None
    aspect_ratio: float
    lab_color: tuple[float, float, float]
