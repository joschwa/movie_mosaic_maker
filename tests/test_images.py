from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from movie_mosaic_maker.images import (
    ImageLoadError,
    list_top_level_image_paths,
    load_directory_images,
    load_image_rgb,
)


def _save_solid(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGB", size, color).save(path)


def test_list_top_level_image_paths_sorted_and_excludes_dotfiles_and_dirs(tmp_path: Path) -> None:
    _save_solid(tmp_path / "b.png", (4, 4), (1, 2, 3))
    _save_solid(tmp_path / "a.png", (4, 4), (1, 2, 3))
    (tmp_path / ".hidden.png").write_bytes(b"not a real image")
    (tmp_path / "subdir").mkdir()

    paths = list_top_level_image_paths(tmp_path)

    assert [p.name for p in paths] == ["a.png", "b.png"]


def test_list_top_level_image_paths_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.png"
    _save_solid(file_path, (4, 4), (1, 2, 3))
    with pytest.raises(NotADirectoryError):
        list_top_level_image_paths(file_path)


def test_load_image_rgb_returns_uint8_rgb_array(tmp_path: Path) -> None:
    path = tmp_path / "solid.png"
    _save_solid(path, (6, 4), (10, 20, 30))

    arr = load_image_rgb(path)

    assert arr.shape == (4, 6, 3)
    assert arr.dtype == np.uint8
    assert tuple(arr[0, 0]) == (10, 20, 30)


def test_load_image_rgb_converts_grayscale_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.new("L", (4, 4), 128).save(path)

    arr = load_image_rgb(path)

    assert arr.shape == (4, 4, 3)
    assert tuple(arr[0, 0]) == (128, 128, 128)


def test_load_image_rgb_applies_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    img = Image.new("RGB", (6, 4), (10, 20, 30))
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: 90 deg rotation needed to display upright
    img.save(path, exif=exif)

    arr = load_image_rgb(path)

    # A correctly-applied 90 degree correction swaps width and height.
    assert arr.shape == (6, 4, 3)


def test_load_image_rgb_raises_on_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is definitely not a valid image file")

    with pytest.raises(ImageLoadError):
        load_image_rgb(path)


def test_load_directory_images_skips_corrupt_files_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _save_solid(tmp_path / "good1.png", (4, 4), (1, 2, 3))
    _save_solid(tmp_path / "good2.png", (4, 4), (4, 5, 6))
    (tmp_path / "bad.png").write_bytes(b"garbage")

    with caplog.at_level("WARNING"):
        loaded = load_directory_images(tmp_path)

    assert [p.name for p, _ in loaded] == ["good1.png", "good2.png"]
    assert any("bad.png" in record.message for record in caplog.records)


def test_load_directory_images_empty_directory(tmp_path: Path) -> None:
    assert load_directory_images(tmp_path) == []
