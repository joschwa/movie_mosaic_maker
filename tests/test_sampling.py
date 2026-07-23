from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from movie_mosaic_maker.cache import FrameCache
from movie_mosaic_maker.color import to_lab
from movie_mosaic_maker.sampling import (
    allocate_pool_across_videos,
    build_pool,
    cyclic_take,
    even_index_offsets,
    even_time_offsets,
)


def test_cyclic_take_returns_empty_for_count_zero() -> None:
    assert cyclic_take([], 0) == []
    assert cyclic_take(["a", "b"], 0) == []


def test_cyclic_take_rejects_negative_count() -> None:
    with pytest.raises(ValueError):
        cyclic_take(["a"], -1)


def test_cyclic_take_rejects_empty_items_when_count_positive() -> None:
    with pytest.raises(ValueError):
        cyclic_take([], 3)


def test_cyclic_take_no_shuffle_preserves_order_within_count() -> None:
    assert cyclic_take(["a", "b", "c"], 2, shuffle=False) == ["a", "b"]
    assert cyclic_take(["a", "b", "c"], 3, shuffle=False) == ["a", "b", "c"]


def test_cyclic_take_no_shuffle_cycles_deterministically() -> None:
    result = cyclic_take(["a", "b", "c"], 7, shuffle=False)
    assert result == ["a", "b", "c", "a", "b", "c", "a"]


def test_cyclic_take_reuse_counts_are_balanced_across_laps() -> None:
    items = ["a", "b", "c"]
    result = cyclic_take(items, 7, shuffle=False)
    counts = Counter(result)
    assert counts["a"] == 3
    assert counts["b"] == 2
    assert counts["c"] == 2


def test_cyclic_take_with_seed_is_reproducible() -> None:
    items = list(range(10))
    first = cyclic_take(items, 25, shuffle=True, seed=42)
    second = cyclic_take(items, 25, shuffle=True, seed=42)
    assert first == second


@given(
    n_items=st.integers(min_value=1, max_value=20),
    count=st.integers(min_value=0, max_value=100),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_cyclic_take_invariants(n_items: int, count: int, seed: int) -> None:
    items = list(range(n_items))
    result = cyclic_take(items, count, shuffle=True, seed=seed)
    assert len(result) == count
    assert all(item in items for item in result)
    if count > 0:
        full_laps, remainder = divmod(count, n_items)
        counts = Counter(result)
        for item in items:
            expected = full_laps + (1 if counts.get(item, 0) > full_laps else 0)
            assert counts.get(item, 0) in (full_laps, full_laps + 1)
        assert sum(1 for item in items if counts.get(item, 0) == full_laps + 1) == remainder


def test_even_time_offsets_basic() -> None:
    offsets = even_time_offsets(8.0, 4)
    assert offsets == pytest.approx([1.0, 3.0, 5.0, 7.0])


def test_even_time_offsets_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        even_time_offsets(10.0, 0)
    with pytest.raises(ValueError):
        even_time_offsets(0.0, 5)
    with pytest.raises(ValueError):
        even_time_offsets(-1.0, 5)


@given(
    duration=st.floats(min_value=0.01, max_value=100_000, allow_nan=False, allow_infinity=False),
    n=st.integers(min_value=1, max_value=1000),
)
def test_even_time_offsets_invariants(duration: float, n: int) -> None:
    offsets = even_time_offsets(duration, n)
    assert len(offsets) == n
    assert all(0.0 < o < duration for o in offsets)
    assert offsets == sorted(offsets)
    if n > 1:
        gaps = [b - a for a, b in zip(offsets, offsets[1:])]
        assert gaps == pytest.approx([gaps[0]] * len(gaps))


def test_allocate_pool_across_videos_exact_proportions() -> None:
    assert allocate_pool_across_videos([1, 2, 1], 8) == [2, 4, 2]
    assert allocate_pool_across_videos([2, 4], 9) == [3, 6]


def test_allocate_pool_across_videos_equal_durations_split_as_evenly_as_possible() -> None:
    result = allocate_pool_across_videos([1, 1, 1], 10)
    assert sum(result) == 10
    assert sorted(result) == [3, 3, 4]


def test_allocate_pool_across_videos_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        allocate_pool_across_videos([], 10)
    with pytest.raises(ValueError):
        allocate_pool_across_videos([1, 0, 2], 10)
    with pytest.raises(ValueError):
        allocate_pool_across_videos([1, 2], -1)


@given(
    durations=st.lists(st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
    pool_size=st.integers(min_value=0, max_value=5000),
)
def test_allocate_pool_across_videos_invariants(durations: list[float], pool_size: int) -> None:
    result = allocate_pool_across_videos(durations, pool_size)
    assert len(result) == len(durations)
    assert sum(result) == pool_size
    assert all(share >= 0 for share in result)


def test_even_index_offsets_basic() -> None:
    assert even_index_offsets(10, 5) == [1, 3, 5, 7, 9]


def test_even_index_offsets_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        even_index_offsets(10, 0)
    with pytest.raises(ValueError):
        even_index_offsets(0, 5)


@given(n_available=st.integers(min_value=1, max_value=1000), k=st.integers(min_value=1, max_value=1000))
def test_even_index_offsets_invariants(n_available: int, k: int) -> None:
    indices = even_index_offsets(n_available, k)
    assert len(indices) == k
    assert all(0 <= i < n_available for i in indices)
    assert indices == sorted(indices)


# -- build_pool -----------------------------------------------------------

_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]


def _make_image_dir(tmp_path: Path, name: str, colors: list[tuple[int, int, int]]) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    for i, color in enumerate(colors):
        Image.new("RGB", (8, 6), color).save(directory / f"{i}.png")
    return directory


def _lab_of(color: tuple[int, int, int]) -> tuple[float, float, float]:
    lab = to_lab(np.array([[color]], dtype=np.uint8))[0, 0]
    return float(lab[0]), float(lab[1]), float(lab[2])


def test_build_pool_even_mode_directory_picks_evenly_spaced_images(tmp_path: Path) -> None:
    directory = _make_image_dir(tmp_path, "images", _COLORS)  # 6 images, sorted 0..5
    cache = FrameCache(tmp_path / "cache")

    candidates = build_pool([directory], pool_size=3, mode="even", cache=cache)

    assert len(candidates) == 3
    assert all(c.kind == "image" for c in candidates)
    # even_index_offsets(6, 3) == [1, 3, 5]
    expected = {_lab_of(_COLORS[1]), _lab_of(_COLORS[3]), _lab_of(_COLORS[5])}
    actual = {tuple(round(x, 6) for x in c.lab_color) for c in candidates}
    assert actual == {tuple(round(x, 6) for x in lab) for lab in expected}


def test_build_pool_even_mode_directory_resamples_when_pool_bigger_than_available(tmp_path: Path) -> None:
    directory = _make_image_dir(tmp_path, "images", _COLORS[:2])
    cache = FrameCache(tmp_path / "cache")

    candidates = build_pool([directory], pool_size=5, mode="even", cache=cache, seed=1)

    assert len(candidates) == 5
    observed_colors = {tuple(round(x, 6) for x in c.lab_color) for c in candidates}
    expected_colors = {tuple(round(x, 6) for x in _lab_of(c)) for c in _COLORS[:2]}
    assert observed_colors == expected_colors


def test_build_pool_even_mode_video_only(make_video) -> None:
    colors = [_COLORS[0]] * 10 + [_COLORS[1]] * 10 + [_COLORS[2]] * 10
    path = make_video("clip.avi", colors, fps=10.0, size=(32, 24))
    cache = FrameCache(path.parent / "cache")

    candidates = build_pool([path], pool_size=6, mode="even", cache=cache)

    assert len(candidates) == 6
    assert all(c.kind == "video" for c in candidates)
    assert [c.frame_index for c in candidates] == list(range(6))
    timestamps = [c.timestamp for c in candidates]
    assert timestamps == sorted(timestamps)


def test_build_pool_even_mode_mixed_sources_sum_to_pool_size(tmp_path: Path, make_video) -> None:
    directory = _make_image_dir(tmp_path, "images", _COLORS)
    video_path = make_video("clip.avi", [_COLORS[0]] * 30, fps=10.0, size=(32, 24))
    cache = FrameCache(tmp_path / "cache")

    candidates = build_pool([directory, video_path], pool_size=10, mode="even", cache=cache)

    assert len(candidates) == 10
    kinds = Counter(c.kind for c in candidates)
    assert kinds["image"] > 0
    assert kinds["video"] > 0


def test_build_pool_exhaustive_mode_directory_uses_every_image(tmp_path: Path) -> None:
    directory = _make_image_dir(tmp_path, "images", _COLORS)
    cache = FrameCache(tmp_path / "cache")

    candidates = build_pool([directory], pool_size=1, mode="exhaustive", cache=cache)

    assert len(candidates) == len(_COLORS)


def test_build_pool_exhaustive_mode_video_uses_every_frame(make_video) -> None:
    path = make_video("clip.avi", [_COLORS[0]] * 7, fps=10.0, size=(16, 16))
    cache = FrameCache(path.parent / "cache")

    candidates = build_pool([path], pool_size=1, mode="exhaustive", cache=cache)

    assert len(candidates) == 7
    assert [c.frame_index for c in candidates] == list(range(7))


def test_build_pool_exhaustive_mode_respects_max_candidates_per_source(make_video) -> None:
    path = make_video("clip.avi", [_COLORS[0]] * 20, fps=10.0, size=(16, 16))
    cache = FrameCache(path.parent / "cache")

    candidates = build_pool([path], pool_size=1, mode="exhaustive", cache=cache, max_candidates_per_source=3)

    assert len(candidates) == 3


def test_build_pool_skips_unreadable_source_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = _make_image_dir(tmp_path, "images", _COLORS)
    bad_video = tmp_path / "bad.avi"
    bad_video.write_bytes(b"not a real video" * 10)
    cache = FrameCache(tmp_path / "cache")

    with caplog.at_level("WARNING"):
        candidates = build_pool([directory, bad_video], pool_size=3, mode="even", cache=cache)

    assert len(candidates) == 3
    assert all(c.kind == "image" for c in candidates)
    assert any("bad.avi" in r.message for r in caplog.records)


def test_build_pool_skips_empty_directory_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    cache = FrameCache(tmp_path / "cache")

    with caplog.at_level("WARNING"):
        candidates = build_pool([empty_dir], pool_size=3, mode="even", cache=cache)

    assert candidates == []
    assert any("empty" in r.message for r in caplog.records)


def test_build_pool_rejects_invalid_pool_size(tmp_path: Path) -> None:
    cache = FrameCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        build_pool([tmp_path], pool_size=0, cache=cache)


def test_build_pool_reuses_cache_across_calls(make_video) -> None:
    path = make_video("clip.avi", [_COLORS[0]] * 10 + [_COLORS[1]] * 10, fps=10.0, size=(16, 16))
    cache = FrameCache(path.parent / "cache")

    first = build_pool([path], pool_size=4, mode="even", cache=cache)
    assert cache.get_frame(path, 0) is not None

    second = build_pool([path], pool_size=4, mode="even", cache=cache)

    assert [c.lab_color for c in first] == [c.lab_color for c in second]
    assert [c.frame_index for c in first] == [c.frame_index for c in second]
