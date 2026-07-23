from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from movie_mosaic_maker.sampling import allocate_pool_across_videos, cyclic_take, even_time_offsets


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
