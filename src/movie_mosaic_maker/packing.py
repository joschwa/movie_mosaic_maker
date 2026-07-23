from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int
    row_heights: list[int]
    col_widths: list[int]
    tile_aspect_ratio: float
    grid_count: int
    leftover_fraction: float

    @cached_property
    def row_offsets(self) -> list[int]:
        offsets = [0]
        for h in self.row_heights:
            offsets.append(offsets[-1] + h)
        return offsets

    @cached_property
    def col_offsets(self) -> list[int]:
        offsets = [0]
        for w in self.col_widths:
            offsets.append(offsets[-1] + w)
        return offsets


def distribute_sizes(total: int, n: int) -> list[int]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if total < n:
        raise ValueError(f"total ({total}) must be >= n ({n}) for every bin to get at least 1px")
    boundaries = [round(i * total / n) for i in range(n + 1)]
    return [boundaries[i + 1] - boundaries[i] for i in range(n)]


def _ideal_rc(n: float, ar: float, width: float, height: float) -> tuple[float, float]:
    rows = math.sqrt(n * ar * height / width)
    cols = math.sqrt(n * width / (ar * height))
    return rows, cols


def _tile_size(width: int, height: int, ar: float, rows: int, cols: int) -> tuple[float, float]:
    tile_h = min(width / (cols * ar), height / rows)
    tile_w = tile_h * ar
    return tile_w, tile_h


def compute_grid(
    width: int,
    height: int,
    tile_aspect_ratio: float,
    n_target: int,
    growth_slack: float = 0.15,
    search_pad: int = 3,
) -> GridSpec:
    if width < 1 or height < 1:
        raise ValueError(f"width and height must be >= 1, got {width}x{height}")
    if tile_aspect_ratio <= 0:
        raise ValueError(f"tile_aspect_ratio must be > 0, got {tile_aspect_ratio}")
    if n_target < 1:
        raise ValueError(f"n_target must be >= 1, got {n_target}")
    if growth_slack < 0:
        raise ValueError(f"growth_slack must be >= 0, got {growth_slack}")

    n_target = min(n_target, width * height)
    n_max = max(n_target, math.ceil(n_target * (1 + growth_slack)))

    r_lo, _ = _ideal_rc(n_target, tile_aspect_ratio, width, height)
    r_hi, _ = _ideal_rc(n_max, tile_aspect_ratio, width, height)
    r_min = min(height, max(1, math.floor(min(r_lo, r_hi)) - search_pad))
    r_max = min(height, max(r_min, math.ceil(max(r_lo, r_hi)) + search_pad))

    # (rows, cols) pairs to score. The minimal-cols arrangement for a given
    # `rows` (satisfying rows*cols >= n_target) is always included even if it
    # overshoots n_max, so a feasible candidate always exists within the
    # window; n_max only bounds how far we opportunistically grow cols beyond
    # that minimum in search of less wasted space.
    candidates: set[tuple[int, int]] = set()
    for rows in range(r_min, r_max + 1):
        cols = max(1, math.ceil(n_target / rows))
        if cols > width:
            continue
        candidates.add((rows, cols))
        while rows * (cols + 1) <= n_max and cols + 1 <= width:
            cols += 1
            candidates.add((rows, cols))

    if not candidates:
        # The window above found nothing usable (e.g. tile_aspect_ratio
        # wildly mismatched with the target's own aspect ratio). Fall back to
        # these two arrangements, both always satisfiable given
        # n_target <= width * height, ignoring growth_slack as a last resort.
        candidates.add((height, min(width, max(1, math.ceil(n_target / height)))))
        candidates.add((min(height, max(1, math.ceil(n_target / width))), width))

    best: tuple[tuple[float, int], int, int, float, float] | None = None
    for rows, cols in candidates:
        if rows * cols < n_target:
            continue
        tile_w, tile_h = _tile_size(width, height, tile_aspect_ratio, rows, cols)
        leftover = width * height - rows * cols * tile_w * tile_h
        key = (round(leftover, 3), rows * cols)
        if best is None or key < best[0]:
            best = (key, rows, cols, tile_w, tile_h)

    assert best is not None, "fallback candidates guarantee a feasible arrangement always exists"
    (leftover, grid_count), rows, cols, _tile_w, _tile_h = best

    row_heights = distribute_sizes(height, rows)
    col_widths = distribute_sizes(width, cols)

    return GridSpec(
        rows=rows,
        cols=cols,
        row_heights=row_heights,
        col_widths=col_widths,
        tile_aspect_ratio=tile_aspect_ratio,
        grid_count=grid_count,
        leftover_fraction=leftover / (width * height),
    )
