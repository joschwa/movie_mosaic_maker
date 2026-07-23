from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeVar

from .cache import FrameCache
from .images import ImageLoadError, list_top_level_image_paths, load_image_rgb
from .models import Candidate, VideoInfo
from .video import VideoLoadError, extract_frame_at_time, iter_all_frames, open_video, probe_video

logger = logging.getLogger(__name__)

T = TypeVar("T")


def cyclic_take(items: Sequence[T], count: int, shuffle: bool = True, seed: int | None = None) -> list[T]:
    """Return `count` items drawn from `items`, cycling through the full sequence
    again (optionally reshuffled each lap) once exhausted, as many times as needed."""
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []
    if not items:
        raise ValueError("items must not be empty when count > 0")

    result: list[T] = []
    lap = 0
    while len(result) < count:
        lap_items = list(items)
        if shuffle:
            rng = random.Random(f"{seed}:{lap}" if seed is not None else None)
            rng.shuffle(lap_items)
        needed = count - len(result)
        result.extend(lap_items[:needed])
        lap += 1
    return result


def even_time_offsets(duration_s: float, n: int) -> list[float]:
    """n timestamps at even gaps across [0, duration_s], offset by half a gap so
    neither the very first nor very last instant is sampled."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    gap = duration_s / n
    return [gap * (i + 0.5) for i in range(n)]


def even_index_offsets(n_available: int, k: int) -> list[int]:
    """k evenly-spaced indices into a sequence of n_available items -- the
    discrete analog of even_time_offsets, for sampling from a folder of images
    rather than a continuous-time video."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n_available < 1:
        raise ValueError(f"n_available must be >= 1, got {n_available}")
    gap = n_available / k
    return [min(n_available - 1, int(gap * (i + 0.5))) for i in range(k)]


def allocate_pool_across_videos(durations: Sequence[float], pool_size: int) -> list[int]:
    """Split pool_size slots across videos proportional to duration, summing to
    exactly pool_size via the largest-remainder (Hamilton) method."""
    if not durations:
        raise ValueError("durations must not be empty")
    if any(d <= 0 for d in durations):
        raise ValueError("all durations must be > 0")
    if pool_size < 0:
        raise ValueError(f"pool_size must be >= 0, got {pool_size}")

    total = sum(durations)
    raw_shares = [pool_size * d / total for d in durations]
    floors = [math.floor(s) for s in raw_shares]
    remainder = pool_size - sum(floors)

    fractional_parts = sorted(
        range(len(durations)), key=lambda i: raw_shares[i] - floors[i], reverse=True
    )
    result = list(floors)
    for i in fractional_parts[:remainder]:
        result[i] += 1
    return result


def _candidate_from_cached_image(path: Path, cache: FrameCache) -> Candidate | None:
    cached = cache.get_image(path)
    if cached is None:
        try:
            image_rgb = load_image_rgb(path)
        except ImageLoadError as exc:
            logger.warning("skipping unreadable image %s: %s", path, exc)
            return None
        cached = cache.put_image(path, image_rgb)
    return Candidate(
        source_id=cache.source_id_for(path),
        kind="image",
        path=path,
        frame_index=None,
        timestamp=None,
        aspect_ratio=cached.aspect_ratio,
        lab_color=cached.lab,
    )


def _even_directory_candidates(image_paths: list[Path], k: int, cache: FrameCache, seed: int | None) -> list[Candidate]:
    if k <= len(image_paths):
        selected = [image_paths[i] for i in even_index_offsets(len(image_paths), k)]
    else:
        selected = cyclic_take(image_paths, k, shuffle=True, seed=seed)

    candidates = []
    for path in selected:
        candidate = _candidate_from_cached_image(path, cache)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _exhaustive_directory_candidates(directory: Path, cache: FrameCache) -> list[Candidate]:
    candidates = []
    for path in list_top_level_image_paths(directory):
        candidate = _candidate_from_cached_image(path, cache)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _even_video_candidates(path: Path, info: VideoInfo, k: int, cache: FrameCache) -> list[Candidate]:
    source_id = cache.source_id_for(path)
    candidates = []
    with open_video(path) as container:
        for i, t in enumerate(even_time_offsets(info.duration_s, k)):
            cached = cache.get_frame(path, i)
            if cached is None:
                try:
                    frame_rgb = extract_frame_at_time(container, t, info)
                except VideoLoadError as exc:
                    logger.warning("skipping unreadable frame in %s at t=%.3fs: %s", path, t, exc)
                    continue
                cached = cache.put_frame(path, i, t, frame_rgb)
            candidates.append(
                Candidate(
                    source_id=source_id,
                    kind="video",
                    path=path,
                    frame_index=i,
                    timestamp=cached.timestamp,
                    aspect_ratio=cached.aspect_ratio,
                    lab_color=cached.lab,
                )
            )
    return candidates


def _exhaustive_video_candidates(
    path: Path, info: VideoInfo, cache: FrameCache, max_candidates: int | None
) -> list[Candidate]:
    source_id = cache.source_id_for(path)
    candidates = []
    for i, frame_rgb in enumerate(iter_all_frames(path)):
        if max_candidates is not None and i >= max_candidates:
            logger.warning(
                "stopping exhaustive sampling of %s after %d candidates (--max-candidates-per-source)",
                path,
                max_candidates,
            )
            break
        cached = cache.get_frame(path, i)
        if cached is None:
            cached = cache.put_frame(path, i, i / info.fps, frame_rgb)
        candidates.append(
            Candidate(
                source_id=source_id,
                kind="video",
                path=path,
                frame_index=i,
                timestamp=cached.timestamp,
                aspect_ratio=cached.aspect_ratio,
                lab_color=cached.lab,
            )
        )
    return candidates


def build_pool(
    sources: Sequence[Path],
    pool_size: int,
    mode: Literal["even", "exhaustive"] = "even",
    *,
    cache: FrameCache,
    seed: int | None = None,
    max_candidates_per_source: int | None = None,
) -> list[Candidate]:
    """Build the candidate pool from a mix of image directories and video files.

    "even" mode allocates pool_size slots proportionally across sources --
    reusing allocate_pool_across_videos with each source's "weight" being video
    duration or, for an image directory, its file count as a discrete analog --
    then samples each source's allocation at even gaps, resampling (cycling)
    if a directory has fewer images than its allocation. "exhaustive" mode
    ignores pool_size/allocation entirely and returns every available image and
    every decoded video frame as a candidate.

    Unreadable sources/files are skipped with a logged warning rather than
    aborting the whole run.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")

    directory_sources: list[Path] = []
    video_sources: list[tuple[Path, VideoInfo]] = []
    for source in sources:
        source = Path(source)
        if source.is_dir():
            directory_sources.append(source)
            continue
        try:
            info = cache.get_video_info(source)
            if info is None:
                info = probe_video(source)
                cache.put_video_info(source, info)
        except VideoLoadError as exc:
            logger.warning("skipping unreadable source %s: %s", source, exc)
            continue
        video_sources.append((source, info))

    if mode == "exhaustive":
        candidates: list[Candidate] = []
        for directory in directory_sources:
            candidates.extend(_exhaustive_directory_candidates(directory, cache))
        for path, info in video_sources:
            candidates.extend(_exhaustive_video_candidates(path, info, cache, max_candidates_per_source))
        return candidates

    weighted_sources: list[tuple[Literal["image", "video"], Path, float, object]] = []
    for directory in directory_sources:
        image_paths = list_top_level_image_paths(directory)
        if not image_paths:
            logger.warning("skipping empty image directory %s", directory)
            continue
        weighted_sources.append(("image", directory, float(len(image_paths)), image_paths))
    for path, info in video_sources:
        weighted_sources.append(("video", path, info.duration_s, info))

    if not weighted_sources:
        return []

    allocations = allocate_pool_across_videos([w for _, _, w, _ in weighted_sources], pool_size)

    candidates = []
    for (kind, path, _weight, extra), k in zip(weighted_sources, allocations):
        if k == 0:
            continue
        if kind == "image":
            candidates.extend(_even_directory_candidates(extra, k, cache, seed))  # type: ignore[arg-type]
        else:
            candidates.extend(_even_video_candidates(path, extra, k, cache))  # type: ignore[arg-type]
    return candidates
