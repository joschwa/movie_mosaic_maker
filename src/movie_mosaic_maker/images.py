from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)


class ImageLoadError(Exception):
    """Raised when a file can't be loaded as an image."""


def list_top_level_image_paths(directory: Path) -> list[Path]:
    """Sorted, non-recursive listing of a directory's files, skipping dotfiles.

    No format allowlist is applied here — every file is a candidate; whether it's
    actually a readable image is determined by attempting to load it."""
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    return sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image file as an HxWx3 uint8 RGB array, EXIF orientation corrected."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            return np.array(img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"could not load image {path}: {exc}") from exc


def load_directory_images(directory: Path) -> list[tuple[Path, np.ndarray]]:
    """Load every readable top-level image in `directory`; unreadable files are
    skipped with a logged warning rather than aborting the whole scan."""
    loaded: list[tuple[Path, np.ndarray]] = []
    for path in list_top_level_image_paths(directory):
        try:
            loaded.append((path, load_image_rgb(path)))
        except ImageLoadError as exc:
            logger.warning(str(exc))
    return loaded
