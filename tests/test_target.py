from itertools import accumulate
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from movie_mosaic_maker.color import to_lab
from movie_mosaic_maker.packing import GridSpec, compute_grid
from movie_mosaic_maker.target import ImageLoadError, compute_cell_lab_colors, load_target_image


def _quadrant_image() -> np.ndarray:
    # 20x20 image split into four 10x10 solid-color quadrants (RGB).
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[0:10, 0:10] = (255, 0, 0)  # top-left: red
    img[0:10, 10:20] = (0, 255, 0)  # top-right: green
    img[10:20, 0:10] = (0, 0, 255)  # bottom-left: blue
    img[10:20, 10:20] = (255, 255, 255)  # bottom-right: white
    return img


def _manual_grid_2x2() -> GridSpec:
    return GridSpec(
        rows=2,
        cols=2,
        row_heights=[10, 10],
        col_widths=[10, 10],
        tile_aspect_ratio=1.0,
        grid_count=4,
        leftover_fraction=0.0,
    )


def test_compute_cell_lab_colors_matches_known_quadrants() -> None:
    img = _quadrant_image()
    grid = _manual_grid_2x2()

    colors = compute_cell_lab_colors(img, grid)

    assert colors.shape == (2, 2, 3)
    assert colors[0, 0] == pytest.approx(to_lab(np.array([[[255, 0, 0]]], dtype=np.uint8))[0, 0], abs=1e-6)
    assert colors[0, 1] == pytest.approx(to_lab(np.array([[[0, 255, 0]]], dtype=np.uint8))[0, 0], abs=1e-6)
    assert colors[1, 0] == pytest.approx(to_lab(np.array([[[0, 0, 255]]], dtype=np.uint8))[0, 0], abs=1e-6)
    assert colors[1, 1] == pytest.approx(to_lab(np.array([[[255, 255, 255]]], dtype=np.uint8))[0, 0], abs=1e-6)


def test_compute_cell_lab_colors_rejects_size_mismatch() -> None:
    img = np.zeros((19, 20, 3), dtype=np.uint8)  # off by one row vs the grid
    grid = _manual_grid_2x2()
    with pytest.raises(ValueError):
        compute_cell_lab_colors(img, grid)


def test_compute_cell_lab_colors_independent_reference_check() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    grid = compute_grid(width=53, height=37, tile_aspect_ratio=1.3, n_target=20)

    colors = compute_cell_lab_colors(img, grid)

    lab = to_lab(img)
    row_bounds = list(accumulate([0, *grid.row_heights]))
    col_bounds = list(accumulate([0, *grid.col_widths]))
    for r in range(grid.rows):
        for c in range(grid.cols):
            expected = lab[row_bounds[r] : row_bounds[r + 1], col_bounds[c] : col_bounds[c + 1]].mean(axis=(0, 1))
            assert colors[r, c] == pytest.approx(expected, abs=1e-9)


def test_compute_cell_lab_colors_integrates_with_compute_grid() -> None:
    rng = np.random.default_rng(1)
    width, height = 200, 120
    img = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    grid = compute_grid(width=width, height=height, tile_aspect_ratio=1.6, n_target=50)

    colors = compute_cell_lab_colors(img, grid)

    assert colors.shape == (grid.rows, grid.cols, 3)
    assert np.isfinite(colors).all()


def test_load_target_image_returns_rgb_array(tmp_path: Path) -> None:
    path = tmp_path / "target.png"
    Image.new("RGB", (12, 8), (1, 2, 3)).save(path)

    arr = load_target_image(path)

    assert arr.shape == (8, 12, 3)
    assert tuple(arr[0, 0]) == (1, 2, 3)


def test_load_target_image_raises_on_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        load_target_image(path)
