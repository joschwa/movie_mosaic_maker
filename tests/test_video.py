from pathlib import Path

import numpy as np
import pytest

from movie_mosaic_maker.video import (
    VideoLoadError,
    extract_frame_at_time,
    iter_all_frames,
    open_video,
    probe_video,
)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
COLOR_TOL = 20  # MJPG is lossy; allow some per-channel drift


def test_probe_video_reports_expected_metadata(make_video) -> None:
    path = make_video("clip.avi", [RED] * 30, fps=10.0, size=(64, 48))

    info = probe_video(path)

    assert info.frame_count == 30
    assert info.fps == pytest.approx(10.0, abs=0.5)
    assert info.width == 64
    assert info.height == 48
    assert info.duration_s == pytest.approx(3.0, abs=0.5)


def test_probe_video_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VideoLoadError):
        probe_video(tmp_path / "does_not_exist.avi")


def test_probe_video_raises_on_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.avi"
    path.write_bytes(b"this is not a real video file" * 10)
    with pytest.raises(VideoLoadError):
        probe_video(path)


def test_open_video_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VideoLoadError):
        with open_video(tmp_path / "nope.avi"):
            pass


def test_extract_frame_at_time_returns_frame_with_expected_color(make_video) -> None:
    colors = [RED] * 10 + [GREEN] * 10 + [BLUE] * 10
    path = make_video("clip.avi", colors, fps=10.0, size=(64, 48))
    info = probe_video(path)

    with open_video(path) as cap:
        frame_red = extract_frame_at_time(cap, t=0.5, info=info)  # ~frame 5 -> red
        frame_green = extract_frame_at_time(cap, t=1.5, info=info)  # ~frame 15 -> green
        frame_blue = extract_frame_at_time(cap, t=2.5, info=info)  # ~frame 25 -> blue

    assert frame_red.mean(axis=(0, 1)) == pytest.approx(RED, abs=COLOR_TOL)
    assert frame_green.mean(axis=(0, 1)) == pytest.approx(GREEN, abs=COLOR_TOL)
    assert frame_blue.mean(axis=(0, 1)) == pytest.approx(BLUE, abs=COLOR_TOL)


def test_extract_frame_at_time_clamps_out_of_range_timestamps(make_video) -> None:
    colors = [RED] * 5 + [BLUE] * 5
    path = make_video("clip.avi", colors, fps=10.0, size=(32, 32))
    info = probe_video(path)

    with open_video(path) as cap:
        frame_before_start = extract_frame_at_time(cap, t=-5.0, info=info)
        frame_after_end = extract_frame_at_time(cap, t=1000.0, info=info)

    assert frame_before_start.mean(axis=(0, 1)) == pytest.approx(RED, abs=COLOR_TOL)
    assert frame_after_end.mean(axis=(0, 1)) == pytest.approx(BLUE, abs=COLOR_TOL)


def test_iter_all_frames_yields_all_frames_in_order(make_video) -> None:
    colors = [RED, GREEN, BLUE, RED, GREEN]
    path = make_video("clip.avi", colors, fps=10.0, size=(16, 16))

    frames = list(iter_all_frames(path))

    assert len(frames) == len(colors)
    for frame, expected in zip(frames, colors):
        assert frame.mean(axis=(0, 1)) == pytest.approx(expected, abs=COLOR_TOL)


def test_iter_all_frames_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VideoLoadError):
        list(iter_all_frames(tmp_path / "nope.avi"))


def test_frames_are_rgb_not_bgr(make_video) -> None:
    path = make_video("clip.avi", [RED] * 5, fps=10.0, size=(16, 16))
    frame = next(iter_all_frames(path))
    r, g, b = frame.mean(axis=(0, 1))
    assert r > g and r > b
