import numpy as np
import pytest

from movie_mosaic_maker.color import lab_distance, mean_lab, to_lab


def _solid(rgb: tuple[int, int, int], size: int = 4) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


@pytest.mark.parametrize(
    "rgb,expected_lab,tol",
    [
        ((255, 255, 255), (100.0, 0.0, 0.0), 0.5),
        ((0, 0, 0), (0.0, 0.0, 0.0), 0.5),
        ((128, 128, 128), (53.6, 0.0, 0.0), 1.0),
        ((255, 0, 0), (53.24, 80.09, 67.20), 1.5),
        ((0, 255, 0), (87.74, -86.18, 83.18), 1.5),
        ((0, 0, 255), (32.30, 79.19, -107.86), 1.5),
    ],
)
def test_to_lab_matches_known_reference_values(
    rgb: tuple[int, int, int], expected_lab: tuple[float, float, float], tol: float
) -> None:
    lab = to_lab(_solid(rgb, size=1))
    assert lab.shape == (1, 1, 3)
    for actual, expected in zip(lab[0, 0], expected_lab):
        assert actual == pytest.approx(expected, abs=tol)


def test_to_lab_preserves_shape() -> None:
    img = np.zeros((5, 7, 3), dtype=np.uint8)
    lab = to_lab(img)
    assert lab.shape == (5, 7, 3)
    assert lab.dtype == np.float64


def test_mean_lab_of_solid_color_matches_single_pixel_conversion() -> None:
    img = _solid((10, 200, 30), size=6)
    single_pixel_lab = to_lab(_solid((10, 200, 30), size=1))[0, 0]
    l, a, b = mean_lab(img)
    assert (l, a, b) == pytest.approx(tuple(single_pixel_lab), abs=1e-9)


def test_mean_lab_averages_in_lab_space_not_rgb_space() -> None:
    half_red = _solid((255, 0, 0), size=2)
    half_blue = _solid((0, 0, 255), size=2)
    combined = np.concatenate([half_red, half_blue], axis=0)

    lab_red = np.array(to_lab(half_red[:1, :1]))[0, 0]
    lab_blue = np.array(to_lab(half_blue[:1, :1]))[0, 0]
    expected = tuple((lab_red + lab_blue) / 2)

    assert mean_lab(combined) == pytest.approx(expected, abs=1e-9)


def test_lab_distance_zero_for_identical_colors() -> None:
    assert lab_distance((50.0, 10.0, -5.0), (50.0, 10.0, -5.0)) == 0.0


def test_lab_distance_matches_euclidean_norm() -> None:
    assert lab_distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)
