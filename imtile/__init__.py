"""imtile — Lightweight image tiling for computer vision & deep learning.

Split large images into overlapping tiles and reconstruct them with
weighted-average blending.  Works with NumPy arrays out of the box and
auto-detects CuPy for transparent GPU acceleration.

Quick start::

    from imtile import ImageTiler

    tiler = ImageTiler(tile_size=256, overlap=32)
    tiles = tiler.tile(large_image)
    reconstructed = tiler.reconstruct(tiles, large_image.shape)
"""

__version__ = "0.1.3"

from .tiler import ImageTiler, reconstruct_image, tile_image  # noqa: F401
from .backend import gpu_available  # noqa: F401

__all__ = [
    "ImageTiler",
    "tile_image",
    "reconstruct_image",
    "gpu_available",
    "__version__",
]
