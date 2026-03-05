"""Backend utilities for automatic GPU/CPU array module detection."""

import importlib
from typing import Any

import numpy as np

_cupy = None
_cupy_checked = False


def _get_cupy():
    """Lazy-load CuPy. Returns the module or None if unavailable."""
    global _cupy, _cupy_checked
    if not _cupy_checked:
        try:
            _cupy = importlib.import_module("cupy")
        except (ImportError, Exception):
            _cupy = None
        _cupy_checked = True
    return _cupy


def gpu_available() -> bool:
    """Return True if CuPy is installed and a CUDA GPU is accessible."""
    cp = _get_cupy()
    if cp is None:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def get_array_module(arr: Any):
    """Return the array module (numpy or cupy) for the given array.

    Args:
        arr: An ndarray (numpy or cupy).

    Returns:
        The module that owns ``arr`` (``numpy`` or ``cupy``).
    """
    cp = _get_cupy()
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def to_gpu(arr: np.ndarray) -> Any:
    """Transfer a NumPy array to GPU memory.

    If CuPy is unavailable, returns the input unchanged.

    Args:
        arr: A NumPy ndarray.

    Returns:
        A CuPy ndarray on GPU, or the original NumPy array.
    """
    cp = _get_cupy()
    if cp is not None:
        return cp.asarray(arr)
    return arr


def to_cpu(arr: Any) -> np.ndarray:
    """Transfer an array to CPU memory.

    If the input is already a NumPy array, returns it unchanged.

    Args:
        arr: A NumPy or CuPy ndarray.

    Returns:
        A NumPy ndarray on CPU.
    """
    cp = _get_cupy()
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)
