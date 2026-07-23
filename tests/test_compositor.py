from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from movie_mosaic_maker.compositor import compose_mosaic, fit_tile, save_mosaic
from movie_mosaic_maker.packing import GridSpec


def _solid(color: tuple[int, int, int], size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_fit_tile_stretch_matches_exact_box_shape() -> None:
    img = _solid((10, 20, 30), size=(40, 10))
    tile = fit_tile(img, box_width=15, box_height=25, mode="stretch")
    assert tile.shape == (25, 15, 3)


def test_fit_tile_stretch_preserves_corner_correspondence() -> None:
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[0, 0] = (255, 0, 0)  # top-left corner marker
    img[0, -1] = (0, 255, 0)  # top-right corner marker
    img[-1, 0] = (0, 0, 255)  # bottom-left corner marker
    img[-1, -1] = (255, 255, 255)  # bottom-right corner marker

    tile = fit_tile(img, box_width=40, box_height=20, mode="stretch")

    assert tuple(tile[0, 0]) == (255, 0, 0)
    assert tuple(tile[0, -1]) == (0, 255, 0)
    assert tuple(tile[-1, 0]) == (0, 0, 255)
    assert tuple(tile[-1, -1]) == (255, 255, 255)


@pytest.mark.parametrize(
    "src_size,box_size",
    [
        ((40, 10), (15, 25)),
        ((10, 40), (25, 15)),
        ((30, 30), (10, 40)),
        ((7, 13), (13, 7)),
        ((100, 20), (5, 5)),
    ],
)
def test_fit_tile_cover_always_matches_exact_box_shape(src_size, box_size) -> None:
    img = _solid((1, 2, 3), size=src_size)
    box_w, box_h = box_size
    tile = fit_tile(img, box_width=box_w, box_height=box_h, mode="cover")
    assert tile.shape == (box_h, box_w, 3)


def test_fit_tile_cover_no_crop_needed_matches_plain_resize() -> None:
    img = _solid((5, 6, 7), size=(20, 10))
    tile = fit_tile(img, box_width=40, box_height=20, mode="cover")
    assert tile.shape == (20, 40, 3)
    assert tuple(tile[0, 0]) == (5, 6, 7)


def test_fit_tile_cover_crops_from_center_of_taller_axis() -> None:
    # Horizontal stripes: top=red, middle=green, bottom=blue. A box much wider
    # than the source forces scaling by width, cropping the vertical excess --
    # the surviving center should be (mostly) the green middle stripe.
    img = np.zeros((30, 10, 3), dtype=np.uint8)
    img[0:10] = (255, 0, 0)
    img[10:20] = (0, 255, 0)
    img[20:30] = (0, 0, 255)

    tile = fit_tile(img, box_width=100, box_height=10, mode="cover")

    assert tile.shape == (10, 100, 3)
    mean_color = tile.mean(axis=(0, 1))
    assert mean_color == pytest.approx((0, 255, 0), abs=15)


def test_fit_tile_rejects_invalid_box_dimensions() -> None:
    img = _solid((1, 1, 1), size=(10, 10))
    with pytest.raises(ValueError):
        fit_tile(img, box_width=0, box_height=10, mode="cover")
    with pytest.raises(ValueError):
        fit_tile(img, box_width=10, box_height=-1, mode="cover")


def test_fit_tile_rejects_unknown_mode() -> None:
    img = _solid((1, 1, 1), size=(10, 10))
    with pytest.raises(ValueError):
        fit_tile(img, box_width=10, box_height=10, mode="pad")  # type: ignore[arg-type]


def _grid_2x2() -> GridSpec:
    return GridSpec(
        rows=2,
        cols=2,
        row_heights=[10, 10],
        col_widths=[10, 10],
        tile_aspect_ratio=1.0,
        grid_count=4,
        leftover_fraction=0.0,
    )


def test_compose_mosaic_assembles_expected_canvas() -> None:
    grid = _grid_2x2()
    colors = {
        0: (255, 0, 0),  # cell (0,0) top-left
        1: (0, 255, 0),  # cell (0,1) top-right
        2: (0, 0, 255),  # cell (1,0) bottom-left
        3: (255, 255, 0),  # cell (1,1) bottom-right
    }

    def load(idx: int) -> np.ndarray:
        return _solid(colors[idx], size=(6, 6))

    canvas = compose_mosaic(grid, candidate_index_per_cell=[0, 1, 2, 3], load_candidate_image=load, tile_fit="cover")

    assert canvas.shape == (20, 20, 3)
    assert tuple(canvas[0, 0]) == (255, 0, 0)
    assert tuple(canvas[0, 19]) == (0, 255, 0)
    assert tuple(canvas[19, 0]) == (0, 0, 255)
    assert tuple(canvas[19, 19]) == (255, 255, 0)


def test_compose_mosaic_respects_non_square_row_col_sizes() -> None:
    grid = GridSpec(
        rows=1,
        cols=2,
        row_heights=[5],
        col_widths=[7, 3],
        tile_aspect_ratio=1.4,
        grid_count=2,
        leftover_fraction=0.0,
    )

    def load(idx: int) -> np.ndarray:
        return _solid((idx * 10, idx * 10, idx * 10), size=(4, 4))

    canvas = compose_mosaic(grid, candidate_index_per_cell=[0, 1], load_candidate_image=load, tile_fit="stretch")

    assert canvas.shape == (5, 10, 3)


def test_save_mosaic_roundtrip(tmp_path: Path) -> None:
    img = _solid((12, 34, 56), size=(8, 6))
    path = tmp_path / "mosaic.png"

    save_mosaic(img, path)
    reloaded = np.array(Image.open(path).convert("RGB"))

    assert reloaded.shape == img.shape
    assert tuple(reloaded[0, 0]) == (12, 34, 56)
