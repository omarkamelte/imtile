"""Core image tiling and reconstruction logic.

Provides the :class:`ImageTiler` class and convenience functions
:func:`tile_image` and :func:`reconstruct_image` for splitting large
images into fixed-size overlapping tiles and reassembling them with
weighted-average blending.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .backend import get_array_module, to_cpu

__all__ = [
    "ImageTiler",
    "tile_image",
    "reconstruct_image",
]


class ImageTiler:
    """Split images into tiles and reconstruct them.

    Args:
        tile_size: Side length of each square tile (pixels).
        overlap: Number of overlapping pixels between adjacent tiles.
            Must satisfy ``0 <= overlap < tile_size``.

    Raises:
        ValueError: If ``tile_size < 1`` or ``overlap`` is invalid.

    Example::

        tiler = ImageTiler(tile_size=256, overlap=32)
        tiles = tiler.tile(large_image)          # list of (256, 256, C)
        reconstructed = tiler.reconstruct(tiles, large_image.shape)
        assert np.allclose(large_image, reconstructed)
    """

    def __init__(self, tile_size: int, overlap: int = 0) -> None:
        if tile_size < 1:
            raise ValueError(f"tile_size must be >= 1, got {tile_size}")
        if not (0 <= overlap < tile_size):
            raise ValueError(
                f"overlap must satisfy 0 <= overlap < tile_size, "
                f"got overlap={overlap}, tile_size={tile_size}"
            )
        self.tile_size = tile_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Effective step between tile origins."""
        return self.tile_size - self.overlap

    def _grid_end(self, length: int) -> int:
        """Compute the iteration limit for one dimension.

        If *length* is not an exact multiple of *step*, round up to the
        next multiple so that the entire image is covered.
        """
        s = self.step
        if length % s == 0:
            return length
        return ((length // s) + 1) * s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tile_positions(
        self, image_shape: Tuple[int, ...]
    ) -> List[Tuple[int, int, int, int]]:
        """Compute ``(y_start, x_start, y_end, x_end)`` for every tile.

        The positions account for boundary snapping: when a tile would
        extend past the image edge, its origin is shifted so that the
        tile ends exactly at the edge.

        Args:
            image_shape: ``(H, W)`` or ``(H, W, C)``.

        Returns:
            List of ``(y_start, x_start, y_end, x_end)`` tuples.

        Raises:
            ValueError: If *image_shape* has fewer than 2 dimensions.
        """
        if len(image_shape) < 2:
            raise ValueError(
                f"image_shape must have at least 2 dimensions, got {len(image_shape)}"
            )

        height, width = image_shape[0], image_shape[1]
        ts = self.tile_size
        step = self.step

        eof_h = self._grid_end(height)
        eof_w = self._grid_end(width)

        positions: List[Tuple[int, int, int, int]] = []
        for y in range(0, eof_h, step):
            for x in range(0, eof_w, step):
                # Boundary snapping
                if y + ts > height and x + ts > width:
                    sy, sx = height - ts, width - ts
                elif y + ts > height:
                    sy, sx = height - ts, x
                elif x + ts > width:
                    sy, sx = y, width - ts
                else:
                    sy, sx = y, x

                sy = max(0, sy)
                sx = max(0, sx)
                ey = min(sy + ts, height)
                ex = min(sx + ts, width)
                positions.append((sy, sx, ey, ex))
        return positions

    def tile(self, image: np.ndarray) -> List[np.ndarray]:
        """Split *image* into tiles.

        Handles images whose dimensions are not exact multiples of the
        tile size by snapping boundary tiles to the image edge (so that
        every tile has shape ``(tile_size, tile_size[, C])``).

        Args:
            image: Array of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            List of tile arrays, each with the same ndim as the input.

        Raises:
            ValueError: If *image* has fewer than 2 dimensions.
        """
        if image.ndim < 2:
            raise ValueError(
                f"image must be at least 2-D, got {image.ndim}-D"
            )

        xp = get_array_module(image)
        is_2d = image.ndim == 2
        positions = self.get_tile_positions(image.shape)

        tiles: List[np.ndarray] = []
        for sy, sx, ey, ex in positions:
            if is_2d:
                t = image[sy:ey, sx:ex]
                if t.shape[0] < self.tile_size or t.shape[1] < self.tile_size:
                    padded = xp.zeros(
                        (self.tile_size, self.tile_size), dtype=image.dtype,
                    )
                    padded[: t.shape[0], : t.shape[1]] = t
                    t = padded
            else:
                t = image[sy:ey, sx:ex, :]
                if t.shape[0] < self.tile_size or t.shape[1] < self.tile_size:
                    padded = xp.zeros(
                        (self.tile_size, self.tile_size, image.shape[2]),
                        dtype=image.dtype,
                    )
                    padded[: t.shape[0], : t.shape[1], :] = t
                    t = padded
            tiles.append(t)

        return tiles

    def reconstruct(
        self,
        tiles: Union[List[np.ndarray], np.ndarray],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct an image from its tiles.

        Overlapping regions are blended using weighted averaging so that
        the reconstruction is lossless when tiles were produced by
        :meth:`tile` with the same ``tile_size`` and ``overlap``.

        Args:
            tiles: List (or array) of tile arrays.  Each tile may have
                shape ``(tile_size, tile_size[, C])`` or
                ``(1, tile_size, tile_size[, C])`` (batch dim stripped
                automatically).
            original_shape: ``(H, W)`` or ``(H, W, C)`` of the original
                image.

        Returns:
            Reconstructed image with the original shape and dtype.

        Raises:
            ValueError: If *original_shape* has fewer than 2 dimensions.
        """
        if len(original_shape) < 2:
            raise ValueError(
                f"original_shape must have at least 2 dimensions, "
                f"got {len(original_shape)}"
            )

        # Normalise tiles list
        if isinstance(tiles, np.ndarray) and tiles.ndim >= 3:
            tiles = [tiles[i] for i in range(tiles.shape[0])]
        tiles_clean: List[np.ndarray] = []
        for t in tiles:
            arr = np.asarray(t)
            # Strip leading batch dim of size 1
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr[0]
            tiles_clean.append(arr)

        is_2d = len(original_shape) == 2
        height, width = original_shape[0], original_shape[1]
        positions = self.get_tile_positions(original_shape)

        weights = np.zeros((height, width), dtype=np.float64)
        out_dtype = tiles_clean[0].dtype if tiles_clean else np.uint8

        if is_2d:
            canvas = np.zeros((height, width), dtype=np.float64)
            for idx, (sy, sx, ey, ex) in enumerate(positions):
                if idx >= len(tiles_clean):
                    break
                tile = tiles_clean[idx].astype(np.float64)
                th, tw = ey - sy, ex - sx
                canvas[sy:ey, sx:ex] += tile[:th, :tw]
                weights[sy:ey, sx:ex] += 1.0
            mask = weights > 0
            canvas[mask] /= weights[mask]
            return canvas.astype(out_dtype)
        else:
            channels = original_shape[2] if len(original_shape) > 2 else 1
            canvas = np.zeros((height, width, channels), dtype=np.float64)
            for idx, (sy, sx, ey, ex) in enumerate(positions):
                if idx >= len(tiles_clean):
                    break
                tile = tiles_clean[idx].astype(np.float64)
                th, tw = ey - sy, ex - sx
                canvas[sy:ey, sx:ex, :] += tile[:th, :tw, :]
                weights[sy:ey, sx:ex] += 1.0
            mask = weights > 0
            for c in range(channels):
                canvas[:, :, c][mask] /= weights[mask]
            return canvas.astype(out_dtype)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tile_size={self.tile_size}, overlap={self.overlap})"
        )


# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------


def tile_image(
    image: np.ndarray,
    tile_size: int,
    overlap: int = 0,
) -> List[np.ndarray]:
    """Split *image* into square tiles.

    Thin wrapper around :meth:`ImageTiler.tile`.

    Args:
        image: Array of shape ``(H, W)`` or ``(H, W, C)``.
        tile_size: Side length of each tile.
        overlap: Overlap in pixels between adjacent tiles.

    Returns:
        List of tile arrays.
    """
    return ImageTiler(tile_size, overlap).tile(image)


def reconstruct_image(
    tiles: Union[List[np.ndarray], np.ndarray],
    original_shape: Tuple[int, ...],
    tile_size: int,
    overlap: int = 0,
) -> np.ndarray:
    """Reconstruct an image from tiles.

    Thin wrapper around :meth:`ImageTiler.reconstruct`.

    Args:
        tiles: List of tile arrays.
        original_shape: Shape of the original image.
        tile_size: Side length used during tiling.
        overlap: Overlap in pixels used during tiling.

    Returns:
        Reconstructed image array.
    """
    return ImageTiler(tile_size, overlap).reconstruct(tiles, original_shape)
