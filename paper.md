# imtile: Lightweight Image Tiling and Reconstruction for Deep Learning Inference

**Omar Kamel**

---

## Abstract

Processing high-resolution images with deep learning models is fundamentally constrained by GPU memory. This paper presents **imtile**, a lightweight, framework-agnostic Python library for splitting large images into fixed-size overlapping tiles and reconstructing the full output via weighted-average blending. We describe the algorithm, analyse its complexity, and compare it with existing approaches including patchify, SAHI, and manual PyTorch-based implementations. imtile achieves optimal O(H×W) complexity, supports automatic GPU acceleration through CuPy, and guarantees lossless round-trip reconstruction — making it suitable for semantic segmentation, object detection, and super-resolution pipelines.

---

## 1. Introduction

Deep learning models for computer vision — particularly for semantic segmentation and object detection — typically accept fixed-size inputs (e.g., 256×256 or 512×512 pixels). However, real-world imagery in domains such as remote sensing, medical imaging, and autonomous driving often exceeds these dimensions by orders of magnitude (e.g., satellite images at 10,000×10,000 pixels or whole-slide histopathology images at 100,000×100,000 pixels).

Naive downsampling to fit model input dimensions causes **loss of fine-grained detail**, degrading detection of small objects and thin structures. The standard solution is **image tiling**: decomposing the large image into manageable patches, processing each independently, and stitching the results back together.

However, tiling introduces **boundary artifacts** — objects split across tile edges may be missed or produce inconsistent predictions. **Overlapping tiles** with blended reconstruction mitigate this problem but require careful handling of:

1. **Grid computation** — ensuring full image coverage
2. **Boundary conditions** — tiles near edges that would extend past the image
3. **Reconstruction blending** — averaging overlapping regions to avoid discontinuities
4. **Memory efficiency** — minimising copies and supporting GPU arrays

imtile addresses all four concerns in a single, dependency-light package.

---

## 2. Algorithm Description

### 2.1 Tiling

Given an input image of shape (H, W, C) and parameters `tile_size` and `overlap`:

1. **Compute step size:** `step = tile_size − overlap`
2. **Compute grid extents:**
   - `eof_H = H` if `H mod step = 0`, else `(⌊H/step⌋ + 1) × step`
   - `eof_W = W` if `W mod step = 0`, else `(⌊W/step⌋ + 1) × step`
3. **Iterate** over grid positions `(y, x)` for `y ∈ {0, step, 2·step, …, eof_H − step}` and similarly for x.
4. **Boundary snapping:** if `y + tile_size > H`, set `y_start = H − tile_size` (and analogously for x). This ensures every tile is exactly `tile_size × tile_size` without zero-padding at boundaries.
5. **Extract** `image[y_start : y_start + tile_size, x_start : x_start + tile_size, :]`.

The result is a list of `n_tiles = ⌈H/step⌉ × ⌈W/step⌉` arrays, each of shape `(tile_size, tile_size, C)`.

### 2.2 Reconstruction

Given tiles and the original image shape:

1. **Allocate** a float64 canvas of shape (H, W, C) and a weights matrix of shape (H, W), both initialised to zero.
2. **Iterate** over tiles in the same grid order as tiling.
3. For each tile at position `(y_start, x_start)`:
   - `canvas[y_start:y_end, x_start:x_end, :] += tile`
   - `weights[y_start:y_end, x_start:x_end] += 1`
4. **Blend:** `canvas /= weights` (element-wise, per channel).
5. **Cast** back to the original dtype and return.

The weighted averaging guarantees that overlapping regions are seamlessly blended. When `overlap = 0`, each pixel has weight 1 and reconstruction is exact.

### 2.3 Boundary Snapping Strategy

The key design choice is **edge snapping** rather than zero-padding for boundary tiles. When a tile would extend past the image edge:

```
Standard padding:              Edge snapping (imtile):
┌────────┬───┐                 ┌────────┐
│ image  │ 0 │  ← pad zeros   │ image  │  ← shift tile left/up
│        │ 0 │                 │ ←←←←←  │
└────────┴───┘                 └────────┘
```

Edge snapping preserves real image content in every tile, avoiding the need for models to handle artificial zero regions. The trade-off is that boundary tiles partially overlap with their neighbours even when `overlap = 0`, but the weighted-average reconstruction handles this correctly.

---

## 3. Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|---|---|---|
| Grid computation | O(1) | O(1) |
| Tile extraction | O(n × tile_size²) | O(n × tile_size² × C) |
| Reconstruction | O(n × tile_size²) | O(H × W × C) |
| Weighted averaging | O(H × W × C) | In-place |

Where `n = ⌈H/step⌉ × ⌈W/step⌉`.

Since `n × tile_size² ≈ H × W × (tile_size / step)²`, the total complexity is **O(H × W)** when the overlap ratio is fixed — **linear in image area**, which is optimal since every pixel must be read at least once.

**GPU acceleration:** When CuPy is available, array operations execute on GPU with the same algorithmic complexity. The only additional cost is the initial host-to-device transfer of O(H × W × C).

---

## 4. Comparison with Existing Approaches

### 4.1 Overview

| Feature | imtile | patchify | SAHI | torchvision (manual) |
|---|---|---|---|---|
| **Primary use** | General tiling | Patch extraction | Object detection | Transforms |
| **Overlap** | ✅ Configurable (pixels) | ✅ Fixed step | ✅ Ratio-based | ❌ Manual |
| **Reconstruction** | ✅ Weighted average | ❌ Not provided | N/A (NMS merging) | ❌ Manual |
| **Lossless round-trip** | ✅ | ❌ | N/A | ❌ |
| **GPU support** | ✅ CuPy auto-detect | ❌ | ❌ | ✅ PyTorch |
| **Framework dependency** | NumPy only | NumPy only | PyTorch + detection models | PyTorch |
| **Boundary handling** | Edge-snap | Zero-pad | Crop | Manual |
| **Tile positions API** | ✅ | ❌ | ❌ | ❌ |
| **Install size** | ~0 KB (numpy dep) | ~10 KB | ~50 MB+ | ~800 MB+ |
| **Task flexibility** | Segmentation, detection, SR | Patch extraction | Detection only | General |

### 4.2 patchify

[patchify](https://github.com/dovahcrow/patchify.py) extracts patches using NumPy's `as_strided` for memory-efficient views. However, it:
- Does not provide a reconstruction function
- Uses zero-padding for boundary conditions, introducing artificial content
- Does not support GPU arrays

imtile provides both tiling and reconstruction, uses edge-snapping instead of padding, and auto-detects CuPy.

### 4.3 SAHI (Slicing Aided Hyper Inference)

[SAHI](https://github.com/obss/sahi) (Akyon et al., 2022) is a framework for small object detection that slices images into overlapping patches, runs inference, and merges detections using Non-Maximum Suppression (NMS). While effective for object detection, SAHI:
- Is tightly coupled to detection pipelines (bounding boxes, not pixel-level outputs)
- Requires PyTorch and a detection model framework
- Does not reconstruct dense pixel-level outputs (segmentation masks)
- Has a large dependency footprint

imtile is complementary to SAHI: it handles the generic tiling/reconstruction that SAHI doesn't cover (segmentation, super-resolution), with minimal dependencies.

### 4.4 Manual PyTorch Implementations

Many practitioners implement tiling manually using `torch.Tensor.unfold()` or custom loops. These approaches:
- Are not reusable across projects
- Require PyTorch as a dependency
- Rarely handle edge cases (non-divisible dimensions, overlap blending) correctly
- Are not tested or documented

imtile provides a tested, documented, reusable solution.

---

## 5. Design Decisions

### 5.1 Edge Snapping vs. Zero Padding

Zero-padding boundary tiles introduces artificial content that can confuse models, particularly for segmentation where every pixel matters. Edge snapping shifts the tile origin so that the tile remains within the image, at the cost of increased overlap near boundaries. The weighted-average reconstruction naturally handles this variable overlap.

### 5.2 Weighted Average vs. Cropping

Some approaches reconstruct by cropping the centre of each tile and discarding the overlap region. While this avoids blending artefacts, it wastes compute (the model processes pixels that are discarded). Weighted averaging uses all predictions and tends to produce smoother outputs in practice, as predictions near tile centres are typically more reliable than those near edges.

### 5.3 Optional GPU Support

Rather than requiring PyTorch or TensorFlow for GPU support, imtile uses CuPy — a NumPy-compatible GPU array library. This keeps the core dependency to NumPy alone, while users who need GPU acceleration can install CuPy separately via `pip install imtile[gpu]`.

---

## 6. Conclusion

imtile provides a minimal, well-tested solution for image tiling and reconstruction that addresses common pain points in deep learning inference pipelines:

1. **Lossless round-trip** through weighted-average blending
2. **Edge snapping** to avoid artificial padding
3. **O(H×W) complexity** — optimal for the task
4. **Zero framework dependency** beyond NumPy
5. **Automatic GPU acceleration** via CuPy

The library is available on PyPI (`pip install imtile`) and GitHub under the MIT license.

---

## References

1. Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection. *IEEE International Conference on Image Processing (ICIP)*. arXiv:2202.06934.
2. patchify — https://github.com/dovahcrow/patchify.py
3. CuPy — https://cupy.dev/
