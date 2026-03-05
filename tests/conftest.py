"""Shared pytest fixtures for imtile tests.

Every fixture builds a **deterministic** image from a formula so that
pixel values at any position are predictable and tests can verify
exact (perfect-match) round-trip reconstruction.

Image formula
-------------
    grayscale:  pixel[y, x]    = (y * W + x) % 256
    RGB:        pixel[y, x, c] = (y * W + x + c * 80) % 256

This guarantees every pixel has a unique-ish but reproducible value.
"""

import numpy as np
import pytest


# ------------------------------------------------------------------
# Helper: build a deterministic image from the formula above
# ------------------------------------------------------------------

def _make_gray(h: int, w: int) -> np.ndarray:
    """Return a (H, W) uint8 image with pixel[y,x] = (y*W + x) % 256."""
    ys = np.arange(h, dtype=np.int64).reshape(-1, 1)
    xs = np.arange(w, dtype=np.int64).reshape(1, -1)
    return ((ys * w + xs) % 256).astype(np.uint8)


def _make_rgb(h: int, w: int) -> np.ndarray:
    """Return a (H, W, 3) uint8 image with pixel[y,x,c] = (y*W + x + c*80) % 256."""
    ys = np.arange(h, dtype=np.int64).reshape(-1, 1, 1)
    xs = np.arange(w, dtype=np.int64).reshape(1, -1, 1)
    cs = np.arange(3, dtype=np.int64).reshape(1, 1, -1)
    return ((ys * w + xs + cs * 80) % 256).astype(np.uint8)


# ------------------------------------------------------------------
# Grayscale fixtures
# ------------------------------------------------------------------

@pytest.fixture
def gray_64():
    """64×64 grayscale uint8 — exact multiple of common tile sizes."""
    return _make_gray(64, 64)


@pytest.fixture
def gray_1243x530():
    """1243×530 grayscale uint8 — H=530, W=1243 (non-square, large)."""
    return _make_gray(530, 1243)


# ------------------------------------------------------------------
# RGB fixtures
# ------------------------------------------------------------------

@pytest.fixture
def rgb_100():
    """100×100 RGB uint8 — exact multiple of tile_size=50."""
    return _make_rgb(100, 100)


@pytest.fixture
def rgb_97x103():
    """97×103 RGB uint8 — non-divisible dimensions."""
    return _make_rgb(97, 103)


@pytest.fixture
def rgb_1243x530():
    """1243×530 RGB uint8 — H=530, W=1243 (non-square, large)."""
    return _make_rgb(530, 1243)


@pytest.fixture
def small_rgb():
    """10×10 RGB uint8 — smaller than typical tile sizes."""
    return _make_rgb(10, 10)
