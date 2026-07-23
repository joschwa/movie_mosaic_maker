from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from .packing import GridSpec

TileFit = Literal["cover", "stretch"]


def _resize(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    src_h, src_w = image_rgb.shape[:2]
    new_w, new_h = size
    interpolation = cv2.INTER_AREA if new_w * new_h < src_w * src_h else cv2.INTER_LINEAR
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=interpolation)


def fit_tile(image_rgb: np.ndarray, box_width: int, box_height: int, mode: TileFit = "cover") -> np.ndarray:
    """Fit `image_rgb` into an exact (box_height, box_width, 3) box.

    "stretch" resizes width/height independently, allowing distortion.
    "cover" resizes to fully cover the box preserving aspect ratio, then
    center-crops the overflow -- no distortion, but some content is cropped.
    """
    if box_width < 1 or box_height < 1:
        raise ValueError(f"box dimensions must be >= 1, got {box_width}x{box_height}")
    src_h, src_w = image_rgb.shape[:2]
    if src_w < 1 or src_h < 1:
        raise ValueError("image_rgb must not be empty")

    if mode == "stretch":
        return _resize(image_rgb, (box_width, box_height))

    if mode != "cover":
        raise ValueError(f"unknown tile fit mode: {mode!r}")

    scale = max(box_width / src_w, box_height / src_h)
    new_w = max(box_width, math.ceil(src_w * scale))
    new_h = max(box_height, math.ceil(src_h * scale))
    resized = _resize(image_rgb, (new_w, new_h))

    x0 = (new_w - box_width) // 2
    y0 = (new_h - box_height) // 2
    return resized[y0 : y0 + box_height, x0 : x0 + box_width]


def compose_mosaic(
    grid: GridSpec,
    candidate_index_per_cell: list[int],
    load_candidate_image: Callable[[int], np.ndarray],
    tile_fit: TileFit = "cover",
) -> np.ndarray:
    """Assemble the final mosaic canvas.

    `candidate_index_per_cell` must be in row-major cell order (cell index
    r*grid.cols + c), matching numpy's default (C-order) flattening of the
    (rows, cols, 3) array produced by target.compute_cell_lab_colors.
    """
    canvas = np.empty((grid.row_offsets[-1], grid.col_offsets[-1], 3), dtype=np.uint8)
    for r in range(grid.rows):
        r0, r1 = grid.row_offsets[r], grid.row_offsets[r + 1]
        for c in range(grid.cols):
            c0, c1 = grid.col_offsets[c], grid.col_offsets[c + 1]
            cell_index = r * grid.cols + c
            candidate_image = load_candidate_image(candidate_index_per_cell[cell_index])
            canvas[r0:r1, c0:c1] = fit_tile(candidate_image, box_width=c1 - c0, box_height=r1 - r0, mode=tile_fit)
    return canvas


def save_mosaic(image_rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(image_rgb).save(path)
