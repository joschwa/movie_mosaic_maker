# Movie Mosaic Maker

Command line program to turn movies (and photo collections) into photomosaics — things that make you go mmm.

Give it a **target image** and a **source** of tiles — a folder of photos, a video file, several video files, or a mix — and it rebuilds the target out of a grid of small tiles sampled from that source, each tile chosen by color match.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. Video decoding uses [PyAV](https://pyav.org/), which bundles its own FFmpeg build on every platform, so reading a given video file doesn't depend on how your system's OpenCV happens to be built.

## Usage

```bash
movie-mosaic-maker --target photo.jpg --sources ./my_photos --output mosaic.png
movie-mosaic-maker --target photo.jpg --sources movie.mp4 --output mosaic.png --grid-size 2000
movie-mosaic-maker --target photo.jpg --sources ./my_photos --sources movie1.mp4 --sources movie2.mp4 --output mosaic.png
```

`--sources` may be repeated and mixes freely: each value is either a directory of images (top-level files only, not recursive) or a video file.

### Options

| Option | Default | Description |
|---|---|---|
| `--target` | *required* | Image to recreate as a mosaic. |
| `--sources` | *required* | Image directory or video file to sample tiles from. Repeatable. |
| `--output` | *required* | Where to save the resulting mosaic. |
| `--grid-size` | `min(1000, width*height)` | Target number of tiles. |
| `--sample-mode` | `even` | `even`: sample tiles/frames at even gaps to best represent the whole source, resampling if more are needed than exist. `exhaustive`: use every available image/frame as a candidate. |
| `--tile-fit` | `cover` | How a tile image fills its cell: `cover` (center-crop, no distortion) or `stretch` (resize independently, may distort). |
| `--max-reuse` | `1` | Max times a single candidate can be reused in the grid; cycles through the pool again (in extra passes) if the grid needs more tiles than that allows. |
| `--growth-slack` | `0.15` | How far the grid may grow beyond `--grid-size` if it reduces wasted canvas space. |
| `--max-candidates-per-source` | unlimited | Safety cap on candidates pulled from one source in `--sample-mode=exhaustive`. |
| `--seed` | none | Random seed, for reproducible resampling. |
| `--cache-dir` | platform user cache dir | Where extracted video frame colors/thumbnails/reference images are cached. |
| `--no-cache` | off | Disable the cache entirely. |
| `--cache-clear` | off | Clear the cache before running. |

## How it works

1. **Sample pool**: candidate tiles are gathered from `--sources` — either at even gaps (default) or exhaustively — and each candidate's mean color (in CIE Lab) and aspect ratio are computed (and cached, for video frames).
2. **Grid packing**: rather than a naive fixed rows×columns grid, the tile grid shape is chosen to preserve the sample pool's own aspect ratio/orientation while maximizing tile size against the target canvas — growing the tile count slightly (bounded by `--growth-slack`) if that reduces wasted space.
3. **Color assignment**: each grid cell is matched to its nearest-color candidate (via a KD-tree over Lab colors), processing rare/distinctive colors first so they get first pick of the pool, capped by `--max-reuse` and cycling through extra passes if the pool runs out before the grid is full.
4. **Composite**: each assigned candidate image is fit into its cell (`--tile-fit`) and the final image is assembled and saved.

Known limitation: the grid uses one uniform tile shape for the whole mosaic (based on the pool's *median* aspect ratio), so a source with widely mixed portrait/landscape images will have some tiles cropped/stretched more than others to fit.

## Development

```bash
pip install -e ".[dev]"
pytest
```
