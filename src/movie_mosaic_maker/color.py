from __future__ import annotations

import cv2
import numpy as np

# All functions in this module assume RGB channel order (not BGR). cv2.imread
# / cv2.VideoCapture.read() return BGR; callers must convert at the IO
# boundary (images.py / video.py) before anything reaches this module.


def to_lab(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 uint8 RGB image to float64 CIE Lab (L in [0,100], a/b roughly in [-128,127])."""
    lab_uint8 = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab_uint8[..., 0] *= 100.0 / 255.0
    lab_uint8[..., 1] -= 128.0
    lab_uint8[..., 2] -= 128.0
    return lab_uint8


def mean_lab(img_rgb_uint8: np.ndarray) -> tuple[float, float, float]:
    """Mean Lab color of an HxWx3 uint8 RGB image (averaged in Lab space, not RGB)."""
    lab = to_lab(img_rgb_uint8)
    l, a, b = lab.reshape(-1, 3).mean(axis=0)
    return float(l), float(a), float(b)


def lab_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """CIE76 Euclidean distance between two Lab colors."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
