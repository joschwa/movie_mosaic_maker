import os
import time
from pathlib import Path

import numpy as np
import pytest

from movie_mosaic_maker.cache import THUMBNAIL_SIZE, FrameCache, _source_id
from movie_mosaic_maker.models import VideoInfo


def _touch(path: Path, content: bytes = b"placeholder") -> Path:
    path.write_bytes(content)
    return path


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (20, 10)) -> np.ndarray:
    w, h = size
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_source_id_is_deterministic_and_changes_with_mtime(tmp_path: Path) -> None:
    path = _touch(tmp_path / "video.mp4")
    stat = path.stat()
    id_a = _source_id(path, stat.st_size, stat.st_mtime)
    id_b = _source_id(path, stat.st_size, stat.st_mtime)
    assert id_a == id_b

    new_mtime = stat.st_mtime + 100
    os.utime(path, (new_mtime, new_mtime))
    new_stat = path.stat()
    id_c = _source_id(path, new_stat.st_size, new_stat.st_mtime)
    assert id_c != id_a


def test_video_info_roundtrip(tmp_path: Path) -> None:
    video_path = _touch(tmp_path / "clip.mp4")
    cache = FrameCache(tmp_path / "cache")
    info = VideoInfo(duration_s=12.5, fps=24.0, frame_count=300, width=640, height=480)

    assert cache.get_video_info(video_path) is None
    cache.put_video_info(video_path, info)

    assert cache.get_video_info(video_path) == info


def test_frame_roundtrip(tmp_path: Path) -> None:
    video_path = _touch(tmp_path / "clip.mp4")
    cache = FrameCache(tmp_path / "cache")
    frame = _solid((200, 50, 10), size=(64, 48))

    assert cache.get_frame(video_path, 5) is None
    put_result = cache.put_frame(video_path, frame_index=5, timestamp=1.23, image_rgb=frame)

    assert put_result.thumbnail.shape == (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)
    assert put_result.aspect_ratio == pytest.approx(64 / 48)

    got = cache.get_frame(video_path, 5)
    assert got is not None
    assert got.frame_index == 5
    assert got.timestamp == pytest.approx(1.23)
    assert got.lab == pytest.approx(put_result.lab)
    assert got.aspect_ratio == pytest.approx(64 / 48)
    assert got.thumbnail.shape == (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)
    assert got.thumbnail.dtype == np.uint8


def test_get_frame_missing_returns_none(tmp_path: Path) -> None:
    video_path = _touch(tmp_path / "clip.mp4")
    cache = FrameCache(tmp_path / "cache")
    assert cache.get_frame(video_path, 0) is None


def test_reference_image_roundtrip(tmp_path: Path) -> None:
    video_path = _touch(tmp_path / "clip.mp4")
    cache = FrameCache(tmp_path / "cache", reference_size=256)
    frame = _solid((10, 220, 30), size=(64, 48))

    assert cache.get_reference_image(video_path, 2) is None
    cache.put_frame(video_path, frame_index=2, timestamp=0.5, image_rgb=frame)

    ref = cache.get_reference_image(video_path, 2)
    assert ref is not None
    assert ref.mean(axis=(0, 1)) == pytest.approx((10, 220, 30), abs=10)


def test_reference_image_is_downscaled_to_max_dimension(tmp_path: Path) -> None:
    video_path = _touch(tmp_path / "clip.mp4")
    cache = FrameCache(tmp_path / "cache", reference_size=32)
    frame = _solid((1, 2, 3), size=(128, 64))

    cache.put_frame(video_path, frame_index=0, timestamp=0.0, image_rgb=frame)
    ref = cache.get_reference_image(video_path, 0)

    assert ref is not None
    assert max(ref.shape[:2]) <= 32


def test_image_roundtrip(tmp_path: Path) -> None:
    image_path = _touch(tmp_path / "photo.jpg")
    cache = FrameCache(tmp_path / "cache")
    img = _solid((5, 6, 7), size=(30, 20))

    assert cache.get_image(image_path) is None
    put_result = cache.put_image(image_path, img)

    got = cache.get_image(image_path)
    assert got is not None
    assert got.lab == pytest.approx(put_result.lab)
    assert got.aspect_ratio == pytest.approx(30 / 20)
    assert got.width == 30
    assert got.height == 20


def test_disabled_cache_computes_but_does_not_persist(tmp_path: Path) -> None:
    video_path = _touch(tmp_path / "clip.mp4")
    cache_dir = tmp_path / "cache"
    cache = FrameCache(cache_dir, enabled=False)
    frame = _solid((9, 8, 7), size=(20, 10))

    result = cache.put_frame(video_path, frame_index=0, timestamp=0.0, image_rgb=frame)

    assert result.aspect_ratio == pytest.approx(2.0)
    assert cache.get_frame(video_path, 0) is None
    assert cache.get_reference_image(video_path, 0) is None
    assert not cache_dir.exists()


def test_clear_removes_all_cached_data(tmp_path: Path) -> None:
    video_path = _touch(tmp_path / "clip.mp4")
    cache_dir = tmp_path / "cache"
    cache = FrameCache(cache_dir)
    frame = _solid((1, 1, 1), size=(20, 10))
    cache.put_frame(video_path, frame_index=0, timestamp=0.0, image_rgb=frame)
    assert cache.get_frame(video_path, 0) is not None

    cache.clear()

    assert cache.get_frame(video_path, 0) is None
    assert cache.get_reference_image(video_path, 0) is None
    # cache remains usable after clearing
    cache.put_frame(video_path, frame_index=0, timestamp=0.0, image_rgb=frame)
    assert cache.get_frame(video_path, 0) is not None


def test_context_manager_closes_connection(tmp_path: Path) -> None:
    video_path = _touch(tmp_path / "clip.mp4")
    with FrameCache(tmp_path / "cache") as cache:
        cache.put_video_info(video_path, VideoInfo(1.0, 10.0, 10, 4, 4))
    assert cache._conn is None
