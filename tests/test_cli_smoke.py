from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image

from movie_mosaic_maker.cli import main


def _make_source_images(directory: Path, colors: list[tuple[int, int, int]]) -> None:
    directory.mkdir()
    for i, color in enumerate(colors):
        Image.new("RGB", (10, 10), color).save(directory / f"{i}.png")


def test_cli_end_to_end_directory_sources(tmp_path: Path) -> None:
    target_path = tmp_path / "target.png"
    target_img = np.zeros((20, 20, 3), dtype=np.uint8)
    target_img[:, :10] = (200, 20, 20)  # left half reddish
    target_img[:, 10:] = (20, 20, 200)  # right half bluish
    Image.fromarray(target_img).save(target_path)

    sources_dir = tmp_path / "sources"
    colors = [(255, 0, 0), (200, 30, 30), (0, 0, 255), (30, 30, 200)] * 3
    _make_source_images(sources_dir, colors)

    output_path = tmp_path / "mosaic.png"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--target", str(target_path),
            "--sources", str(sources_dir),
            "--output", str(output_path),
            "--grid-size", "4",
            "--cache-dir", str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()

    mosaic = np.array(Image.open(output_path).convert("RGB"))
    assert mosaic.shape[:2] == target_img.shape[:2]

    left_mean = mosaic[:, : mosaic.shape[1] // 2].mean(axis=(0, 1))
    right_mean = mosaic[:, mosaic.shape[1] // 2 :].mean(axis=(0, 1))
    assert left_mean[0] > left_mean[2]  # left half more red than blue
    assert right_mean[2] > right_mean[0]  # right half more blue than red


def test_cli_fails_cleanly_with_no_usable_sources(tmp_path: Path) -> None:
    target_path = tmp_path / "target.png"
    Image.new("RGB", (10, 10), (1, 2, 3)).save(target_path)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--target", str(target_path),
            "--sources", str(empty_dir),
            "--output", str(tmp_path / "out.png"),
            "--cache-dir", str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code != 0
    assert "no usable candidate" in result.output


def test_cli_fails_cleanly_on_corrupt_target(tmp_path: Path) -> None:
    target_path = tmp_path / "target.png"
    target_path.write_bytes(b"not an image")
    sources_dir = tmp_path / "sources"
    _make_source_images(sources_dir, [(1, 2, 3)])

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--target", str(target_path),
            "--sources", str(sources_dir),
            "--output", str(tmp_path / "out.png"),
            "--cache-dir", str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code != 0


def test_cli_with_video_source(tmp_path: Path, make_video) -> None:
    target_path = tmp_path / "target.png"
    Image.new("RGB", (12, 8), (10, 200, 10)).save(target_path)

    video_path = make_video("clip.avi", [(10, 200, 10)] * 20, fps=10.0, size=(16, 16))

    output_path = tmp_path / "mosaic.png"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--target", str(target_path),
            "--sources", str(video_path),
            "--output", str(output_path),
            "--grid-size", "4",
            "--cache-dir", str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_cli_tile_fit_stretch_option(tmp_path: Path) -> None:
    target_path = tmp_path / "target.png"
    Image.new("RGB", (10, 10), (100, 150, 200)).save(target_path)
    sources_dir = tmp_path / "sources"
    _make_source_images(sources_dir, [(100, 150, 200)] * 4)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--target", str(target_path),
            "--sources", str(sources_dir),
            "--output", str(tmp_path / "out.png"),
            "--grid-size", "4",
            "--tile-fit", "stretch",
            "--cache-dir", str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0, result.output


def test_cli_rerun_reuses_cache(tmp_path: Path, make_video) -> None:
    target_path = tmp_path / "target.png"
    Image.new("RGB", (12, 8), (10, 200, 10)).save(target_path)
    video_path = make_video("clip.avi", [(10, 200, 10)] * 20, fps=10.0, size=(16, 16))
    cache_dir = tmp_path / "cache"

    runner = CliRunner()
    for output_name in ("mosaic1.png", "mosaic2.png"):
        result = runner.invoke(
            main,
            [
                "--target", str(target_path),
                "--sources", str(video_path),
                "--output", str(tmp_path / output_name),
                "--grid-size", "4",
                "--cache-dir", str(cache_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / output_name).exists()
