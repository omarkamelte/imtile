"""Comprehensive deterministic test suite for imtile.

Every round-trip test uses ``np.testing.assert_array_equal`` (exact match).
After reconstruction, specific pixel positions are spot-checked against
pre-computed expected values derived from the deterministic image formula:

    grayscale:  pixel[y, x]    = (y * W + x) % 256
    RGB:        pixel[y, x, c] = (y * W + x + c * 80) % 256
"""

import math

import numpy as np
import pytest

from imtile import ImageTiler, tile_image, reconstruct_image, gpu_available


# ------------------------------------------------------------------
# Helpers – expected pixel values from the deterministic formula
# ------------------------------------------------------------------

def _expected_gray(y: int, x: int, w: int) -> int:
    """Expected grayscale value at position (y, x) for image width *w*."""
    return (y * w + x) % 256


def _expected_rgb(y: int, x: int, w: int) -> np.ndarray:
    """Expected [R, G, B] at position (y, x) for image width *w*."""
    return np.array(
        [(y * w + x + c * 80) % 256 for c in range(3)], dtype=np.uint8,
    )


def _spot_check_gray(recon: np.ndarray, h: int, w: int):
    """Verify exact pixel values at the four corners and center."""
    positions = {
        "top-left":      (0, 0),
        "top-right":     (0, w - 1),
        "bottom-left":   (h - 1, 0),
        "bottom-right":  (h - 1, w - 1),
        "center":        (h // 2, w // 2),
    }
    for label, (y, x) in positions.items():
        expected = _expected_gray(y, x, w)
        actual = recon[y, x]
        assert actual == expected, (
            f"Gray spot-check failed at {label} ({y},{x}): "
            f"expected {expected}, got {actual}"
        )


def _spot_check_rgb(recon: np.ndarray, h: int, w: int):
    """Verify exact pixel values at the four corners and center (all 3 channels)."""
    positions = {
        "top-left":      (0, 0),
        "top-right":     (0, w - 1),
        "bottom-left":   (h - 1, 0),
        "bottom-right":  (h - 1, w - 1),
        "center":        (h // 2, w // 2),
    }
    for label, (y, x) in positions.items():
        expected = _expected_rgb(y, x, w)
        actual = recon[y, x, :]
        np.testing.assert_array_equal(
            actual, expected,
            err_msg=(
                f"RGB spot-check failed at {label} ({y},{x}): "
                f"expected {expected}, got {actual}"
            ),
        )


# ===================================================================
# 1. Deterministic grayscale round-trip tests
# ===================================================================
class TestDeterministicGrayscale:
    """Tile → reconstruct with exact pixel verification for grayscale images."""

    # ---- 64×64 ----
    def test_64x64_no_overlap(self, gray_64):
        tiler = ImageTiler(tile_size=32, overlap=0)
        tiles = tiler.tile(gray_64)
        recon = tiler.reconstruct(tiles, gray_64.shape)
        assert recon.shape == gray_64.shape
        assert recon.dtype == gray_64.dtype
        np.testing.assert_array_equal(recon, gray_64)
        _spot_check_gray(recon, 64, 64)

    def test_64x64_with_overlap(self, gray_64):
        tiler = ImageTiler(tile_size=32, overlap=8)
        tiles = tiler.tile(gray_64)
        recon = tiler.reconstruct(tiles, gray_64.shape)
        assert recon.shape == gray_64.shape
        assert recon.dtype == gray_64.dtype
        np.testing.assert_array_equal(recon, gray_64)
        _spot_check_gray(recon, 64, 64)

    # ---- 1243×530 (W=1243, H=530) ----
    def test_1243x530_no_overlap(self, gray_1243x530):
        tiler = ImageTiler(tile_size=64, overlap=0)
        tiles = tiler.tile(gray_1243x530)
        recon = tiler.reconstruct(tiles, gray_1243x530.shape)
        assert recon.shape == gray_1243x530.shape
        assert recon.dtype == gray_1243x530.dtype
        np.testing.assert_array_equal(recon, gray_1243x530)
        _spot_check_gray(recon, 530, 1243)

    def test_1243x530_with_overlap(self, gray_1243x530):
        tiler = ImageTiler(tile_size=64, overlap=16)
        tiles = tiler.tile(gray_1243x530)
        recon = tiler.reconstruct(tiles, gray_1243x530.shape)
        assert recon.shape == gray_1243x530.shape
        assert recon.dtype == gray_1243x530.dtype
        np.testing.assert_array_equal(recon, gray_1243x530)
        _spot_check_gray(recon, 530, 1243)


# ===================================================================
# 2. Deterministic RGB round-trip tests
# ===================================================================
class TestDeterministicRGB:
    """Tile → reconstruct with exact pixel verification for RGB images."""

    # ---- 100×100 (exact-divisible by tile_size=50) ----
    def test_100x100_no_overlap(self, rgb_100):
        tiler = ImageTiler(tile_size=50, overlap=0)
        tiles = tiler.tile(rgb_100)
        recon = tiler.reconstruct(tiles, rgb_100.shape)
        assert recon.shape == rgb_100.shape
        assert recon.dtype == rgb_100.dtype
        np.testing.assert_array_equal(recon, rgb_100)
        _spot_check_rgb(recon, 100, 100)

    def test_100x100_with_overlap(self, rgb_100):
        tiler = ImageTiler(tile_size=50, overlap=10)
        tiles = tiler.tile(rgb_100)
        recon = tiler.reconstruct(tiles, rgb_100.shape)
        assert recon.shape == rgb_100.shape
        assert recon.dtype == rgb_100.dtype
        np.testing.assert_array_equal(recon, rgb_100)
        _spot_check_rgb(recon, 100, 100)

    # ---- 97×103 (non-divisible) ----
    def test_97x103_no_overlap(self, rgb_97x103):
        tiler = ImageTiler(tile_size=32, overlap=0)
        tiles = tiler.tile(rgb_97x103)
        recon = tiler.reconstruct(tiles, rgb_97x103.shape)
        assert recon.shape == rgb_97x103.shape
        assert recon.dtype == rgb_97x103.dtype
        np.testing.assert_array_equal(recon, rgb_97x103)
        _spot_check_rgb(recon, 97, 103)

    def test_97x103_with_overlap(self, rgb_97x103):
        tiler = ImageTiler(tile_size=32, overlap=8)
        tiles = tiler.tile(rgb_97x103)
        recon = tiler.reconstruct(tiles, rgb_97x103.shape)
        assert recon.shape == rgb_97x103.shape
        assert recon.dtype == rgb_97x103.dtype
        np.testing.assert_array_equal(recon, rgb_97x103)
        _spot_check_rgb(recon, 97, 103)

    # ---- 1243×530 (W=1243, H=530) ----
    def test_1243x530_no_overlap(self, rgb_1243x530):
        tiler = ImageTiler(tile_size=64, overlap=0)
        tiles = tiler.tile(rgb_1243x530)
        recon = tiler.reconstruct(tiles, rgb_1243x530.shape)
        assert recon.shape == rgb_1243x530.shape
        assert recon.dtype == rgb_1243x530.dtype
        np.testing.assert_array_equal(recon, rgb_1243x530)
        _spot_check_rgb(recon, 530, 1243)

    def test_1243x530_with_overlap(self, rgb_1243x530):
        tiler = ImageTiler(tile_size=64, overlap=16)
        tiles = tiler.tile(rgb_1243x530)
        recon = tiler.reconstruct(tiles, rgb_1243x530.shape)
        assert recon.shape == rgb_1243x530.shape
        assert recon.dtype == rgb_1243x530.dtype
        np.testing.assert_array_equal(recon, rgb_1243x530)
        _spot_check_rgb(recon, 530, 1243)


# ===================================================================
# 3. Tile shapes are correct
# ===================================================================
class TestTileShapes:
    """Every tile must be exactly (tile_size, tile_size[, C])."""

    def test_gray_tiles_2d(self, gray_64):
        tiler = ImageTiler(tile_size=32, overlap=0)
        tiles = tiler.tile(gray_64)
        assert all(t.ndim == 2 for t in tiles)
        assert all(t.shape == (32, 32) for t in tiles)

    def test_rgb_tiles_3d(self, rgb_97x103):
        tiler = ImageTiler(tile_size=32, overlap=8)
        tiles = tiler.tile(rgb_97x103)
        assert all(t.ndim == 3 for t in tiles)
        assert all(t.shape == (32, 32, 3) for t in tiles)


# ===================================================================
# 4. Single tile (image smaller than tile_size)
# ===================================================================
class TestSingleTile:
    def test_image_smaller_than_tile(self, small_rgb):
        """10×10 image with tile_size=64 → 1 tile, zero-padded."""
        tiler = ImageTiler(tile_size=64, overlap=0)
        tiles = tiler.tile(small_rgb)
        assert len(tiles) == 1
        assert tiles[0].shape == (64, 64, 3)
        # Top-left corner should match original
        np.testing.assert_array_equal(tiles[0][:10, :10, :], small_rgb)
        # Padded area should be zeros
        np.testing.assert_array_equal(tiles[0][10:, :, :], 0)
        np.testing.assert_array_equal(tiles[0][:10, 10:, :], 0)
        # Spot-check a known pixel inside the original region
        expected = _expected_rgb(5, 5, 10)
        np.testing.assert_array_equal(tiles[0][5, 5, :], expected)


# ===================================================================
# 5. Dtype preservation
# ===================================================================
class TestDtypePreservation:
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
    def test_dtype_roundtrip(self, dtype):
        h, w = 64, 64
        if np.issubdtype(dtype, np.integer):
            ys = np.arange(h, dtype=np.int64).reshape(-1, 1)
            xs = np.arange(w, dtype=np.int64).reshape(1, -1)
            img = ((ys * w + xs) % 256).astype(dtype)
        else:
            ys = np.arange(h, dtype=np.float64).reshape(-1, 1)
            xs = np.arange(w, dtype=np.float64).reshape(1, -1)
            img = ((ys * w + xs) % 256).astype(dtype) / 256.0
        tiler = ImageTiler(tile_size=32, overlap=0)
        tiles = tiler.tile(img)
        recon = tiler.reconstruct(tiles, img.shape)
        assert recon.dtype == dtype


# ===================================================================
# 6. Tile count
# ===================================================================
class TestTileCount:
    def test_count_no_overlap(self, rgb_100):
        tiler = ImageTiler(tile_size=32, overlap=0)
        tiles = tiler.tile(rgb_100)
        step = 32
        expected_h = math.ceil(100 / step)
        expected_w = math.ceil(100 / step)
        assert len(tiles) == expected_h * expected_w

    def test_count_with_overlap(self, rgb_100):
        tiler = ImageTiler(tile_size=32, overlap=8)
        tiles = tiler.tile(rgb_100)
        step = 24  # 32 - 8
        expected_h = math.ceil(100 / step)
        expected_w = math.ceil(100 / step)
        assert len(tiles) == expected_h * expected_w

    def test_count_non_square(self, gray_1243x530):
        tiler = ImageTiler(tile_size=64, overlap=0)
        tiles = tiler.tile(gray_1243x530)
        expected_h = math.ceil(530 / 64)
        expected_w = math.ceil(1243 / 64)
        assert len(tiles) == expected_h * expected_w


# ===================================================================
# 7. Tile positions
# ===================================================================
class TestTilePositions:
    def test_positions_match_tiles(self, rgb_97x103):
        tiler = ImageTiler(tile_size=32, overlap=8)
        positions = tiler._get_tile_positions(rgb_97x103.shape)
        tiles = tiler.tile(rgb_97x103)
        assert len(positions) == len(tiles)
        for (sy, sx, ey, ex), tile in zip(positions, tiles):
            assert ey - sy <= tiler.tile_size
            assert ex - sx <= tiler.tile_size
            assert sy >= 0 and sx >= 0

    def test_positions_cover_image(self, rgb_100):
        """Every pixel must be covered by at least one tile."""
        tiler = ImageTiler(tile_size=32, overlap=8)
        positions = tiler._get_tile_positions(rgb_100.shape)
        coverage = np.zeros(rgb_100.shape[:2], dtype=int)
        for sy, sx, ey, ex in positions:
            coverage[sy:ey, sx:ex] += 1
        assert coverage.min() >= 1, "Some pixels are not covered by any tile"

    def test_positions_cover_non_square(self, gray_1243x530):
        """Every pixel of the 1243×530 image must be covered."""
        tiler = ImageTiler(tile_size=64, overlap=16)
        positions = tiler._get_tile_positions(gray_1243x530.shape)
        coverage = np.zeros(gray_1243x530.shape[:2], dtype=int)
        for sy, sx, ey, ex in positions:
            coverage[sy:ey, sx:ex] += 1
        assert coverage.min() >= 1, "Some pixels are not covered by any tile"


# ===================================================================
# 8. Invalid inputs
# ===================================================================
class TestInvalidInputs:
    def test_overlap_ge_tile_size(self):
        with pytest.raises(ValueError, match="overlap"):
            ImageTiler(tile_size=32, overlap=32)

    def test_overlap_negative(self):
        with pytest.raises(ValueError, match="overlap"):
            ImageTiler(tile_size=32, overlap=-1)

    def test_tile_size_zero(self):
        with pytest.raises(ValueError, match="tile_size"):
            ImageTiler(tile_size=0)

    def test_1d_image(self):
        tiler = ImageTiler(tile_size=32)
        with pytest.raises(ValueError, match="2-D"):
            tiler.tile(np.zeros((100,)))

    def test_1d_shape_reconstruct(self):
        tiler = ImageTiler(tile_size=32)
        with pytest.raises(ValueError, match="2 dimensions"):
            tiler.reconstruct([np.zeros((32, 32))], (100,))


# ===================================================================
# 9. GPU fallback
# ===================================================================
class TestGPUFallback:
    def test_gpu_available_returns_bool(self):
        """gpu_available() should return a bool without raising."""
        result = gpu_available()
        assert isinstance(result, bool)


# ===================================================================
# 10. Convenience functions
# ===================================================================
class TestConvenienceFunctions:
    def test_tile_image_function(self, rgb_100):
        tiles = tile_image(rgb_100, tile_size=50, overlap=0)
        assert len(tiles) == 4

    def test_reconstruct_image_function(self, rgb_100):
        tiles = tile_image(rgb_100, tile_size=50, overlap=0)
        recon = reconstruct_image(tiles, rgb_100.shape, tile_size=50, overlap=0)
        np.testing.assert_array_equal(recon, rgb_100)
        _spot_check_rgb(recon, 100, 100)

    def test_convenience_roundtrip_gray(self, gray_1243x530):
        tiles = tile_image(gray_1243x530, tile_size=64, overlap=0)
        recon = reconstruct_image(
            tiles, gray_1243x530.shape, tile_size=64, overlap=0,
        )
        np.testing.assert_array_equal(recon, gray_1243x530)
        _spot_check_gray(recon, 530, 1243)


# ===================================================================
# 11. Repr
# ===================================================================
class TestRepr:
    def test_repr(self):
        tiler = ImageTiler(tile_size=256, overlap=32)
        assert "256" in repr(tiler)
        assert "32" in repr(tiler)
