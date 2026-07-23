from __future__ import annotations

import hashlib
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .color import mean_lab
from .models import VideoInfo

THUMBNAIL_SIZE = 32
DEFAULT_REFERENCE_SIZE = 512


@dataclass(frozen=True)
class CachedFrame:
    frame_index: int
    timestamp: float
    lab: tuple[float, float, float]
    aspect_ratio: float
    thumbnail: np.ndarray


@dataclass(frozen=True)
class CachedImage:
    lab: tuple[float, float, float]
    aspect_ratio: float
    width: int
    height: int


def _source_id(path: Path, size: int, mtime: float) -> str:
    """Stable id for a file's current content. A modified file naturally gets a
    different id (rather than needing explicit cache invalidation), leaving the
    old row/files as harmless orphans."""
    digest = hashlib.sha1()
    digest.update(str(path.resolve()).encode())
    digest.update(str(size).encode())
    digest.update(str(mtime).encode())
    return digest.hexdigest()


def _resize_max_dim(img_rgb: np.ndarray, max_dim: int) -> np.ndarray:
    height, width = img_rgb.shape[:2]
    if max(height, width) <= max_dim:
        return img_rgb
    scale = max_dim / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)


class FrameCache:
    """SQLite-backed cache of extracted video frame colors/thumbnails/reference
    images, plus folder-image colors. Disabling (`enabled=False`) turns every
    `put_*` into a no-op and every `get_*` into a guaranteed cache miss, without
    callers needing to branch on whether caching is on."""

    def __init__(self, cache_dir: Path, reference_size: int = DEFAULT_REFERENCE_SIZE, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.reference_size = reference_size
        self.enabled = enabled
        self._conn: sqlite3.Connection | None = None
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.cache_dir / "index.sqlite3")
            self._init_schema()

    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                source_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                duration REAL NOT NULL,
                fps REAL NOT NULL,
                frame_count INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS frames (
                source_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                lab_l REAL NOT NULL,
                lab_a REAL NOT NULL,
                lab_b REAL NOT NULL,
                aspect_ratio REAL NOT NULL,
                thumb BLOB NOT NULL,
                PRIMARY KEY (source_id, frame_index)
            );
            CREATE TABLE IF NOT EXISTS images (
                source_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                lab_l REAL NOT NULL,
                lab_a REAL NOT NULL,
                lab_b REAL NOT NULL,
                aspect_ratio REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            );
            """
        )
        self._conn.commit()

    def _refs_dir(self, source_id: str) -> Path:
        return self.cache_dir / "refs" / source_id

    @staticmethod
    def _stat(path: Path) -> tuple[int, float]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime

    def _source_id_for_file(self, path: Path) -> str:
        size, mtime = self._stat(path)
        return _source_id(path, size, mtime)

    def source_id_for(self, path: Path) -> str:
        """Public accessor for a file's current stable content id (see `_source_id`)."""
        return self._source_id_for_file(path)

    # -- videos --------------------------------------------------------

    def get_video_info(self, path: Path) -> VideoInfo | None:
        if not self.enabled:
            return None
        assert self._conn is not None
        source_id = self._source_id_for_file(path)
        row = self._conn.execute(
            "SELECT duration, fps, frame_count, width, height FROM videos WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            return None
        duration, fps, frame_count, width, height = row
        return VideoInfo(duration_s=duration, fps=fps, frame_count=frame_count, width=width, height=height)

    def put_video_info(self, path: Path, info: VideoInfo) -> None:
        if not self.enabled:
            return
        assert self._conn is not None
        source_id = self._source_id_for_file(path)
        self._conn.execute(
            "INSERT OR REPLACE INTO videos (source_id, path, duration, fps, frame_count, width, height) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source_id, str(path), info.duration_s, info.fps, info.frame_count, info.width, info.height),
        )
        self._conn.commit()

    # -- video frames ----------------------------------------------------

    def get_frame(self, path: Path, frame_index: int) -> CachedFrame | None:
        if not self.enabled:
            return None
        assert self._conn is not None
        source_id = self._source_id_for_file(path)
        row = self._conn.execute(
            "SELECT timestamp, lab_l, lab_a, lab_b, aspect_ratio, thumb FROM frames "
            "WHERE source_id = ? AND frame_index = ?",
            (source_id, frame_index),
        ).fetchone()
        if row is None:
            return None
        timestamp, l, a, b, ar, thumb_blob = row
        thumbnail = np.frombuffer(thumb_blob, dtype=np.uint8).reshape(THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)
        return CachedFrame(frame_index=frame_index, timestamp=timestamp, lab=(l, a, b), aspect_ratio=ar, thumbnail=thumbnail)

    def put_frame(self, path: Path, frame_index: int, timestamp: float, image_rgb: np.ndarray) -> CachedFrame:
        height, width = image_rgb.shape[:2]
        lab = mean_lab(image_rgb)
        aspect_ratio = width / height
        thumbnail = cv2.resize(image_rgb, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), interpolation=cv2.INTER_AREA)

        if self.enabled:
            assert self._conn is not None
            source_id = self._source_id_for_file(path)
            self._conn.execute(
                "INSERT OR REPLACE INTO frames "
                "(source_id, frame_index, timestamp, lab_l, lab_a, lab_b, aspect_ratio, thumb) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (source_id, frame_index, timestamp, *lab, aspect_ratio, thumbnail.tobytes()),
            )
            self._conn.commit()

            reference = _resize_max_dim(image_rgb, self.reference_size)
            refs_dir = self._refs_dir(source_id)
            refs_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(reference).save(refs_dir / f"{frame_index}.jpg", quality=90)

        return CachedFrame(
            frame_index=frame_index, timestamp=timestamp, lab=lab, aspect_ratio=aspect_ratio, thumbnail=thumbnail
        )

    def get_reference_image(self, path: Path, frame_index: int) -> np.ndarray | None:
        if not self.enabled:
            return None
        source_id = self._source_id_for_file(path)
        ref_path = self._refs_dir(source_id) / f"{frame_index}.jpg"
        if not ref_path.exists():
            return None
        with Image.open(ref_path) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)

    # -- folder images -----------------------------------------------------

    def get_image(self, path: Path) -> CachedImage | None:
        if not self.enabled:
            return None
        assert self._conn is not None
        source_id = self._source_id_for_file(path)
        row = self._conn.execute(
            "SELECT lab_l, lab_a, lab_b, aspect_ratio, width, height FROM images WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            return None
        l, a, b, ar, width, height = row
        return CachedImage(lab=(l, a, b), aspect_ratio=ar, width=width, height=height)

    def put_image(self, path: Path, image_rgb: np.ndarray) -> CachedImage:
        height, width = image_rgb.shape[:2]
        lab = mean_lab(image_rgb)
        aspect_ratio = width / height

        if self.enabled:
            assert self._conn is not None
            source_id = self._source_id_for_file(path)
            self._conn.execute(
                "INSERT OR REPLACE INTO images (source_id, path, lab_l, lab_a, lab_b, aspect_ratio, width, height) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (source_id, str(path), *lab, aspect_ratio, width, height),
            )
            self._conn.commit()

        return CachedImage(lab=lab, aspect_ratio=aspect_ratio, width=width, height=height)

    # -- lifecycle -----------------------------------------------------

    def clear(self) -> None:
        """Delete all cached data (sqlite index + reference images) and reinitialize an empty cache."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.cache_dir / "index.sqlite3")
            self._init_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> FrameCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
