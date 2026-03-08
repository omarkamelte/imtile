# imtile

**Lightweight image tiling and reconstruction for computer vision & deep learning.**

[![PyPI version](https://img.shields.io/pypi/v/imtile)](https://pypi.org/project/imtile/)
[![Downloads](https://static.pepy.tech/badge/imtile/month)](https://pepy.tech/projects/imtile)
[![Python](https://img.shields.io/pypi/pyversions/imtile)](https://pypi.org/project/imtile/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/omarkamelte/imtile/actions/workflows/ci.yml/badge.svg)](https://github.com/omarkamelte/imtile/actions)

Split large images into fixed-size overlapping tiles for neural network inference, then reassemble them with **weighted-average blending** for seamless, lossless reconstruction.

## Features

- ✅ **Configurable overlap** — eliminate boundary artifacts in segmentation / detection
- ✅ **Lossless round-trip** — `tile → predict → reconstruct` produces the exact original dimensions
- ✅ **Boundary snapping** — handles images whose dimensions aren't multiples of tile size
- ✅ **GPU acceleration** — auto-detects CuPy for transparent GPU processing
- ✅ **Framework agnostic** — works with plain NumPy arrays (no PyTorch/TF dependency)
- ✅ **Grayscale & multi-channel** — supports 2-D and 3-D arrays

## Installation

```bash
pip install imtile
```

With GPU support (requires CUDA):

```bash
pip install imtile[gpu]
```

## Quick Start

```python
import numpy as np
from imtile import ImageTiler

# Load your large image (H, W, C)
image = np.random.randint(0, 256, (2048, 2048, 3), dtype=np.uint8)

# Create tiler with 256×256 tiles and 32px overlap
tiler = ImageTiler(tile_size=256, overlap=32)

# Split into tiles
tiles = tiler.tile(image)
print(f"Generated {len(tiles)} tiles")

# Process each tile (e.g., run through a neural network)
predictions = [my_model(tile) for tile in tiles]

# Reconstruct the full-size output
result = tiler.reconstruct(predictions, image.shape)
assert result.shape == image.shape
```

### Convenience Functions

```python
from imtile import tile_image, reconstruct_image

tiles = tile_image(image, tile_size=256, overlap=32)
result = reconstruct_image(tiles, image.shape, tile_size=256, overlap=32)
```

## Algorithm

```
┌──────────────────────────────────────────┐
│            Original Image (H×W)          │
│                                          │
│  ┌─────────┐                             │
│  │ Tile 0   │                            │
│  │          │─overlap─┐                  │
│  └─────────┘         │                  │
│        ┌─────────────┤                  │
│        │  Tile 1      │                  │
│        │              │                  │
│        └──────────────┘                  │
│              ...                         │
│                        ┌────────────┐    │
│  Boundary tiles snap → │  Tile N     │   │
│  to image edge         │  (snapped)  │   │
│                        └────────────┘    │
└──────────────────────────────────────────┘

Reconstruction: canvas += tile; weights += 1
                result = canvas / weights  (weighted average)
```

**Complexity:** O(H × W) — linear in image area, optimal.

## API Reference

### `ImageTiler(tile_size, overlap=0)`

| Method | Description |
|---|---|
| `tile(image)` | Split image into tiles. Returns `List[ndarray]`. |
| `reconstruct(tiles, original_shape)` | Reassemble tiles with weighted averaging. |

### Module Functions

| Function | Description |
|---|---|
| `tile_image(image, tile_size, overlap)` | Convenience wrapper for `ImageTiler.tile`. |
| `reconstruct_image(tiles, shape, tile_size, overlap)` | Convenience wrapper for `ImageTiler.reconstruct`. |
| `gpu_available()` | Returns `True` if CuPy/CUDA is detected. |

## Use Cases

- **Semantic segmentation** of satellite / aerial / medical imagery
- **Object detection** on high-resolution images (complementary to SAHI)
- **Super-resolution** inference on large inputs
- **Any pipeline** that needs to process images larger than GPU memory

## License

[MIT](LICENSE) — free for personal, academic, and commercial use.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/omarkamelte/imtile). See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
