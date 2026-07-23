from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import TypeVar

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
