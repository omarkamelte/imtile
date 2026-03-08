# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-03-08

### Added
- `ImageTiler` class with `tile()` and `reconstruct()` methods.
- Convenience functions `tile_image()` and `reconstruct_image()`.
- Configurable overlap with weighted-average blending for seamless reconstruction.
- Edge-snapping boundary handling (no zero-padding).
- Automatic GPU acceleration via CuPy (optional `[gpu]` extra).
- Support for 2-D (grayscale) and 3-D (multi-channel) NumPy arrays.
- Comprehensive test suite (32 tests).
- CI pipeline (GitHub Actions) across Python 3.8, 3.10, 3.12.
- Automated PyPI publishing via Trusted Publishing (OIDC).
