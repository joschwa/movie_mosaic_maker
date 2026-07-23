import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from movie_mosaic_maker.packing import compute_grid, distribute_sizes


def test_distribute_sizes_sums_exactly() -> None:
    assert sum(distribute_sizes(1000, 7)) == 1000
    assert sum(distribute_sizes(1, 1)) == 1
    assert sum(distribute_sizes(1000, 1000)) == 1000


def test_distribute_sizes_bin_count_and_evenness() -> None:
    sizes = distribute_sizes(1000, 7)
    assert len(sizes) == 7
    assert max(sizes) - min(sizes) <= 1


def test_distribute_sizes_rejects_more_bins_than_pixels() -> None:
    with pytest.raises(ValueError):
        distribute_sizes(5, 10)


@given(total=st.integers(min_value=1, max_value=100_000), n=st.integers(min_value=1, max_value=1000))
def test_distribute_sizes_invariants(total: int, n: int) -> None:
    if n > total:
        with pytest.raises(ValueError):
            distribute_sizes(total, n)
        return
    sizes = distribute_sizes(total, n)
    assert len(sizes) == n
    assert sum(sizes) == total
    assert max(sizes) - min(sizes) <= 1
    assert all(s >= 1 for s in sizes)


def test_compute_grid_perfect_square() -> None:
    grid = compute_grid(width=1000, height=1000, tile_aspect_ratio=1.0, n_target=100, growth_slack=0.0)
    assert grid.rows == grid.cols == 10
    assert grid.grid_count == 100


def test_compute_grid_n1() -> None:
    grid = compute_grid(width=640, height=480, tile_aspect_ratio=1.5, n_target=1)
    assert grid.rows == 1
    assert grid.cols == 1
    assert grid.grid_count == 1


def test_compute_grid_portrait_tiles_favor_more_columns_on_square_target() -> None:
    grid = compute_grid(width=1000, height=1000, tile_aspect_ratio=0.5, n_target=200)
    assert grid.cols > grid.rows


def test_compute_grid_landscape_tiles_favor_more_rows_on_square_target() -> None:
    grid = compute_grid(width=1000, height=1000, tile_aspect_ratio=2.0, n_target=200)
    assert grid.rows > grid.cols


def test_compute_grid_canvas_matches_target_exactly() -> None:
    grid = compute_grid(width=1920, height=1080, tile_aspect_ratio=1.777, n_target=500)
    assert sum(grid.row_heights) == 1080
    assert sum(grid.col_widths) == 1920
    assert grid.row_offsets[-1] == 1080
    assert grid.col_offsets[-1] == 1920


def test_compute_grid_growth_slack_can_reduce_waste_for_prime_n() -> None:
    # 11 has no nearby factor pairs that pack tightly on their own, so allowing
    # the grid to grow finds a noticeably better-fitting arrangement (16=4x4
    # vs. 12, both already >11 due to unavoidable rounding even at zero slack).
    no_growth = compute_grid(width=1000, height=600, tile_aspect_ratio=1.5, n_target=11, growth_slack=0.0)
    with_growth = compute_grid(width=1000, height=600, tile_aspect_ratio=1.5, n_target=11, growth_slack=0.5)
    assert with_growth.leftover_fraction < no_growth.leftover_fraction
    assert with_growth.grid_count >= 11
    assert no_growth.grid_count >= 11


def test_compute_grid_never_undershoots_n_target() -> None:
    for n_target in (1, 7, 50, 999, 1000):
        grid = compute_grid(width=800, height=600, tile_aspect_ratio=1.33, n_target=n_target)
        assert grid.grid_count >= n_target


def test_compute_grid_clamps_n_target_to_pixel_count() -> None:
    grid = compute_grid(width=5, height=5, tile_aspect_ratio=1.0, n_target=1000)
    assert grid.rows <= 5
    assert grid.cols <= 5
    assert grid.grid_count <= 25


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(width=0, height=10, tile_aspect_ratio=1.0, n_target=5),
        dict(width=10, height=0, tile_aspect_ratio=1.0, n_target=5),
        dict(width=10, height=10, tile_aspect_ratio=0.0, n_target=5),
        dict(width=10, height=10, tile_aspect_ratio=-1.0, n_target=5),
        dict(width=10, height=10, tile_aspect_ratio=1.0, n_target=0),
        dict(width=10, height=10, tile_aspect_ratio=1.0, n_target=5, growth_slack=-0.1),
    ],
)
def test_compute_grid_rejects_invalid_inputs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        compute_grid(**kwargs)


@given(
    width=st.integers(min_value=10, max_value=4000),
    height=st.integers(min_value=10, max_value=4000),
    ar=st.floats(min_value=0.2, max_value=5.0, allow_nan=False, allow_infinity=False),
    n_target=st.integers(min_value=1, max_value=2000),
)
def test_compute_grid_invariants(width: int, height: int, ar: float, n_target: int) -> None:
    grid = compute_grid(width=width, height=height, tile_aspect_ratio=ar, n_target=n_target)
    assert grid.rows >= 1
    assert grid.cols >= 1
    assert grid.rows <= height
    assert grid.cols <= width
    assert grid.grid_count == grid.rows * grid.cols
    assert grid.grid_count >= min(n_target, width * height)
    assert len(grid.row_heights) == grid.rows
    assert len(grid.col_widths) == grid.cols
    assert sum(grid.row_heights) == height
    assert sum(grid.col_widths) == width
    assert 0.0 <= grid.leftover_fraction < 1.0 or math.isclose(grid.leftover_fraction, 0.0, abs_tol=1e-9)
