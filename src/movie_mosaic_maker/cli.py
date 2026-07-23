from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import platformdirs

from .assignment import assign_candidates
from .cache import FrameCache
from .compositor import compose_mosaic, save_mosaic
from .diagnostics import Diagnostics
from .images import ImageLoadError, load_image_rgb
from .packing import compute_grid
from .sampling import build_pool
from .target import compute_cell_lab_colors, load_target_image
from .video import VideoLoadError, extract_frame_at_time, open_video, probe_video

DEFAULT_GRID_SIZE = 1000


def _default_cache_dir() -> Path:
    return Path(platformdirs.user_cache_dir("movie_mosaic_maker"))


def _load_candidate_image_factory(candidates, cache: FrameCache):
    def load(idx: int):
        candidate = candidates[idx]
        if candidate.kind == "image":
            return load_image_rgb(candidate.path)

        reference = cache.get_reference_image(candidate.path, candidate.frame_index)
        if reference is not None:
            return reference

        info = cache.get_video_info(candidate.path) or probe_video(candidate.path)
        with open_video(candidate.path) as container:
            frame = extract_frame_at_time(container, candidate.timestamp, info)
        cache.put_frame(candidate.path, candidate.frame_index, candidate.timestamp, frame)
        return frame

    return load


EPILOG = """\
Examples:

\b
  movie-mosaic-maker --target photo.jpg --sources ./my_photos --output mosaic.png
  movie-mosaic-maker --target photo.jpg --sources movie.mp4 --output mosaic.png --grid-size 2000
  movie-mosaic-maker --target photo.jpg --sources ./my_photos --sources movie1.mp4 --sources movie2.mp4 --output mosaic.png

--sources may be repeated and mixes freely: each value is either a directory of
images (top-level files only, not recursive) or a video file.
"""


@click.command(epilog=EPILOG)
@click.option(
    "--target",
    "target_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Target image to recreate as a photomosaic.",
)
@click.option(
    "--sources",
    "source_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Image directories and/or video files to sample tiles from. May be repeated.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to save the resulting mosaic image.",
)
@click.option("--grid-size", type=int, default=None, help="Target tile count (default: min(1000, target width*height)).")
@click.option(
    "--sample-mode",
    type=click.Choice(["even", "exhaustive"]),
    default="even",
    show_default=True,
    help=(
        "'even': sample tiles/frames at even gaps to best represent the whole source, "
        "resampling if more are needed than exist. 'exhaustive': use every available "
        "image/frame as a candidate (slower, no pre-thinning)."
    ),
)
@click.option(
    "--tile-fit",
    type=click.Choice(["cover", "stretch"]),
    default="cover",
    show_default=True,
    help=(
        "How a tile image fills its cell. 'cover': center-crop to fill, no distortion. "
        "'stretch': resize width/height independently, may distort."
    ),
)
@click.option("--max-reuse", type=int, default=1, show_default=True, help="Max times a single candidate can be reused in the grid.")
@click.option(
    "--growth-slack",
    type=float,
    default=0.15,
    show_default=True,
    help="How far the grid may grow beyond --grid-size to reduce wasted canvas space.",
)
@click.option(
    "--max-candidates-per-source",
    type=int,
    default=None,
    help="Safety cap on candidates pulled from one source in --sample-mode=exhaustive.",
)
@click.option("--seed", type=int, default=None, help="Random seed, for reproducible resampling.")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None, help="Cache directory (default: platform user cache dir).")
@click.option("--no-cache", is_flag=True, default=False, help="Disable the frame/color cache entirely.")
@click.option("--cache-clear", is_flag=True, default=False, help="Clear the cache before running.")
def main(
    target_path: Path,
    source_paths: tuple[Path, ...],
    output_path: Path,
    grid_size: int | None,
    sample_mode: str,
    tile_fit: str,
    max_reuse: int,
    growth_slack: float,
    max_candidates_per_source: int | None,
    seed: int | None,
    cache_dir: Path | None,
    no_cache: bool,
    cache_clear: bool,
) -> None:
    """Recreate --target as a photomosaic built from tiles sampled from --sources."""
    resolved_cache_dir = cache_dir or _default_cache_dir()

    with FrameCache(resolved_cache_dir, enabled=not no_cache) as cache:
        if cache_clear:
            cache.clear()

        with Diagnostics() as diagnostics:
            try:
                target_rgb = load_target_image(target_path)
            except ImageLoadError as exc:
                raise click.ClickException(str(exc)) from exc
            height, width = target_rgb.shape[:2]

            effective_grid_size = grid_size if grid_size is not None else min(DEFAULT_GRID_SIZE, width * height)

            candidates = build_pool(
                list(source_paths),
                pool_size=effective_grid_size,
                mode=sample_mode,
                cache=cache,
                seed=seed,
                max_candidates_per_source=max_candidates_per_source,
            )
            if not candidates:
                raise click.ClickException("no usable candidate images/frames were found in --sources")

            tile_aspect_ratio = float(np.median([c.aspect_ratio for c in candidates]))
            grid = compute_grid(width, height, tile_aspect_ratio, effective_grid_size, growth_slack=growth_slack)
            diagnostics.note_leftover_fraction(grid.leftover_fraction)

            cell_colors = compute_cell_lab_colors(target_rgb, grid).reshape(-1, 3)
            candidate_colors = np.array([c.lab_color for c in candidates])

            try:
                result = assign_candidates(cell_colors, candidate_colors, max_reuse=max_reuse)
            except ValueError as exc:
                raise click.ClickException(str(exc)) from exc
            diagnostics.note_passes(result.passes)

            load_candidate_image = _load_candidate_image_factory(candidates, cache)

            try:
                canvas = compose_mosaic(grid, result.candidate_index_per_cell, load_candidate_image, tile_fit=tile_fit)
            except (ImageLoadError, VideoLoadError) as exc:
                raise click.ClickException(f"failed to load a candidate tile image: {exc}") from exc
            save_mosaic(canvas, output_path)

        click.echo(
            f"Wrote {output_path} ({grid.rows}x{grid.cols} = {grid.grid_count} tiles, "
            f"{len(candidates)} candidates, {result.passes} pass(es))"
        )
        click.echo(diagnostics.report())


if __name__ == "__main__":
    main()
