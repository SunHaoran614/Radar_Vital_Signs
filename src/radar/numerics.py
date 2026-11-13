"""Stable powers for complex64 IQ without changing its physical scale."""
from __future__ import annotations

import numpy as np


def magnitude_squared(values: np.ndarray) -> np.ndarray:
    """Accumulate IQ power in float64, including tiny/large finite float32 IQ.

    Casting *before* multiplication prevents overflow and underflow that would
    otherwise silently change energy-based target selection across gain units.
    """
    z = np.asarray(values, dtype=np.complex128)
    return z.real * z.real + z.imag * z.imag
