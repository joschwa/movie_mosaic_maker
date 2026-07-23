from __future__ import annotations

from pathlib import Path

import numpy as np

from .color import to_lab
from .images import ImageLoadError, load_image_rgb
from .packing import GridSpec

__all__ = ["ImageLoadError", "load_image_rgb", "load_target_image", "compute_cell_lab_colors"]


def load_target_image(path: Path) -> np.ndarray:
    """Load the target image as an HxWx3 uint8 RGB array."""
    return load_image_rgb(path)


def compute_cell_lab_colors(target_rgb: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Mean Lab color of each grid cell's region of the target image.

    Returns an array of shape (grid.rows, grid.cols, 3). `grid` must have been
    computed against target_rgb's own dimensions (as compute_grid always is),
    so its row/col boundaries exactly tile the image with no gaps.
    """
    height, width = target_rgb.shape[:2]
    if height != grid.row_offsets[-1] or width != grid.col_offsets[-1]:
        raise ValueError(
            f"target image size {width}x{height} does not match grid's tiled "
            f"size {grid.col_offsets[-1]}x{grid.row_offsets[-1]}"
        )

    lab = to_lab(target_rgb)
    colors = np.empty((grid.rows, grid.cols, 3), dtype=np.float64)
    for r in range(grid.rows):
        r0, r1 = grid.row_offsets[r], grid.row_offsets[r + 1]
        for c in range(grid.cols):
            c0, c1 = grid.col_offsets[c], grid.col_offsets[c + 1]
            colors[r, c] = lab[r0:r1, c0:c1].mean(axis=(0, 1))
    return colors
